"""
Extract feature data from a step file using Open Cascade
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)

from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid

# BRepNet
from pipeline.entity_mapper import EntityMapper
from pipeline.face_index_validator import FaceIndexValidator
from pipeline.segmentation_file_crosschecker import SegmentationFileCrosschecker

import utils.scale_utils as scale_utils 
from utils.create_occwl_from_occ import create_occwl

class BRepNetExtractor:
    def __init__(self, step_file, output_dir, scale_body=False):
        self.step_file = step_file
        self.output_dir = output_dir
        self.scale_body = scale_body


    def process(self):
        """
        Process the file and extract the derivative data
        """
        # Load the body from the STEP file
        body = self.load_body_from_step()

        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            body = scale_utils.scale_solid_to_unit_box(body)

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)

        if not self.check_manifold(top_exp):
            print(f"{self.step_file.name} Non-manifold bodies are not supported")
            return

        if not self.check_closed(body):
            print(f"{self.step_file.name} Bodies which are not closed are not supported")
            return
                
        if not self.check_unique_coedges(top_exp):
            print(f"{self.step_file.name} Bodies where the same coedge is uses in multiple loops are not supported")
            return
        
        entity_mapper = EntityMapper(body)

        num_edges = entity_mapper.get_nr_of_edges()

        face_point_grids = self.extract_face_point_grids(body, entity_mapper)
        assert face_point_grids.shape[1] == 9
        coedge_point_grids = self.extract_coedge_point_grids(body, entity_mapper)
        assert coedge_point_grids.shape[1] == 13

        coedge_lcs = self.extract_coedge_local_coordinate_systems(body, entity_mapper)
        coedge_reverse_flags = self.extract_coedge_reverse_flags(body, entity_mapper)

        next, mate, face, edge  = self.build_incidence_arrays(body, entity_mapper)

        # 检查并替换coedge_lcs中的不可逆矩阵
        coedge_lcs = self.check_singular_matrix(coedge_lcs)

        # 将coedge点网格转换到局部坐标系
        coedge_point_grids_local = self.transform_coedge_point_grids_to_local(coedge_point_grids, coedge_lcs)

        # 将face点网格转换到局部坐标系
        face_point_grids_local = self.transform_face_point_grids_to_local(face_point_grids, coedge_lcs, mate, face)

        output_pathname = self.output_dir / f"{self.step_file.stem}.npz"
        np.savez(
            output_pathname, 
            face_point_grids=face_point_grids,
            coedge_point_grids=coedge_point_grids,
            coedge_lcs=coedge_lcs,
            coedge_reverse_flags=coedge_reverse_flags,
            next=next, 
            mate=mate, 
            face=face, 
            edge=edge,
            face_point_grids_local=face_point_grids_local,
            coedge_point_grids_local=coedge_point_grids_local,
            num_edges=num_edges,
            savez_compressed = True
        )


    def load_body_from_step(self):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape


    def extract_face_point_grids(self, body, entity_mapper):
        """
        Extract a UV-Net point grid for each face.

        Returns a tensor [ num_faces x 9 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast
            - u, v (uv_values)

        """
        face_grids = []
        solid = create_occwl(body)
        for face in solid.faces():
            assert len(face_grids) == entity_mapper.face_index(face.topods_shape())
            face_grids.append(self.extract_face_point_grid(face))
        return np.stack(face_grids)

    def extract_face_point_grid(self, face):
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 9 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast
            - u, v (uv_values)

        """
        num_u=10
        num_v=10

        points, uv_values = uvgrid(face, num_u, num_v, uvs=True, method="point")
        normals = uvgrid(face, num_u, num_v, method="normal")
        mask = uvgrid(face, num_u, num_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 9 ]
        single_grid = np.concatenate([points, normals, mask, uv_values], axis=2)
       
        return np.transpose(single_grid, (2, 0, 1))


    def extract_coedge_point_grids(self, body, entity_mapper):
        """
        Extract coedge grids (aligned with the coedge direction).

        The coedge grids will be of size

            [ num_coedges x 13 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
            - u (u_params)
        """
        coedge_grids = []
        solid = create_occwl(body)
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_grids) == entity_mapper.halfedge_index(coedge)
                occwl_oriented_edge = Edge(coedge)
                faces = [ f for f in solid.faces_from_edge(occwl_oriented_edge) ]
                coedge_grids.append(self.extract_coedge_point_grid(occwl_oriented_edge, faces))
        return np.stack(coedge_grids)


    def extract_coedge_point_grid(self, coedge, faces):
        """
        Extract a coedge grid (aligned with the coedge direction).

        The coedge grids will be of size

            [ num_coedges x 13 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
            - u (u_params)
        """
        num_u = 10
        coedge_data = EdgeDataExtractor(coedge, faces, num_samples=num_u, use_arclength_params=True)
        if not coedge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((13, num_u)) 

        u_params_array = np.array(coedge_data.u_params).reshape(-1, 1)
        single_grid = np.concatenate(
            [
                coedge_data.points, 
                coedge_data.tangents, 
                coedge_data.left_normals,
                coedge_data.right_normals,
                u_params_array
            ],
            axis = 1
        )
        return np.transpose(single_grid, (1,0))


    
    def extract_coedge_local_coordinate_systems(self, body, entity_mapper):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.  
        
            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face. 
            - The v_ref will be the coedge tangent at the midpoint.  We get the v_vec by projecting this normal
              to the w_vec
            - The u_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor
 
        Returns a tensor of size [ num_coedges x 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
        """
        solid = create_occwl(body)
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        coedge_lcs = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_lcs) == entity_mapper.halfedge_index(coedge)
                occwl_oriented_edge = Edge(coedge)
                faces = [ f for f in solid.faces_from_edge(occwl_oriented_edge) ]
                coedge_lcs.append(self.extract_coedge_local_coordinate_system(occwl_oriented_edge, faces))

        return np.stack(coedge_lcs)


    def extract_coedge_local_coordinate_system(self, oriented_edge, faces):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.  
        
            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face. 
            - The v_ref will be the coedge tangent at the midpoint.  We get the v_vec by projecting this normal
              to the w_vec
            - The u_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor
 
        Returns a tensor of size [ 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
            [[ u_vec.x  v_vec.x  w_vec.x  orig.x]
             [ u_vec.y  v_vec.y  w_vec.y  orig.y]
             [ u_vec.z  v_vec.z  w_vec.z  orig.z]
             [ 0        0        0        1     ]]
        """
        num_u = 3
        edge_data = EdgeDataExtractor(oriented_edge, faces, num_samples=num_u, use_arclength_params=True)
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # We want to return zeros in this case
            return np.zeros((4,4))
        origin = edge_data.points[1]
        w_vec = edge_data.left_normals[1]

        # Make sure w_vec is a unit vector
        w_vec = w_vec/np.linalg.norm(w_vec)

        # We need to project v_ref normal to w_vec
        v_ref =  edge_data.tangents[1]
        v_vec = self.try_to_project_normal(w_vec, v_ref)
        if v_vec is None:
            # This happens when v_ref is parallel to w_vec.
            # In this case we just pick any v_vec at random
            v_vec = self.any_orthogonal(v_vec)

        u_vec = np.cross(v_vec, w_vec)
        
        # The upper part of the matric should look like this
        # [[ u_vec.x  v_vec.x  w_vec.x  orig.x]
        #  [ u_vec.y  v_vec.y  w_vec.y  orig.y]
        #  [ u_vec.z  v_vec.z  w_vec.z  orig.z]]
        mat_upper = np.transpose(np.stack([u_vec, v_vec, w_vec, origin]))

        mat_lower = np.expand_dims(np.array([0, 0, 0, 1]), axis=0)
        mat = np.concatenate([mat_upper, mat_lower], axis=0)

        return mat


    def check_singular_matrix(self, coedge_lcs):
        for i in range(len(coedge_lcs)):
            lcs_matrix = coedge_lcs[i]
            # 检查是否为全0矩阵或奇异矩阵
            if np.allclose(lcs_matrix, 0) or np.linalg.det(lcs_matrix) < 1e-10:
                print(f"Warning: Invalid LCS matrix at coedge index {i}")
                # 使用单位矩阵作为备用
                coedge_lcs[i] = np.eye(4)
        
        return coedge_lcs


    def transform_coedge_point_grids_to_local(self, coedge_point_grids, coedge_lcs):
        num_coedges = coedge_point_grids.shape[0]
        num_u = coedge_point_grids.shape[2]

        coedge_point_grids_local = np.zeros_like(coedge_point_grids)

        for i in range(num_coedges):
            grid = coedge_point_grids[i]  # [13, num_u]
            lcs_matrix = coedge_lcs[i]  # [4, 4]

            points_global = grid[:3, :]  # [3, num_u]
            vectors_global = grid[3:12, :]  # [9, num_u]
            us = grid[12:, :]

            points_homogeneous = np.vstack([points_global, np.ones((1, num_u))])
            lcs_inv = np.linalg.inv(lcs_matrix)
            points_local_homogeneous = lcs_inv @ points_homogeneous
            points_local = points_local_homogeneous[:3, :]

            rotation_matrix = lcs_inv[:3, :3]
            vectors_local = np.zeros_like(vectors_global)  # [9, num_u]
            for j in range(3):
                vectors_local[j*3:(j+1)*3, :] = rotation_matrix @ vectors_global[j*3:(j+1)*3, :]

            coedge_point_grids_local[i, :3, :] = points_local
            coedge_point_grids_local[i, 3:12, :] = vectors_local
            coedge_point_grids_local[i, 12:, :] = us

        return coedge_point_grids_local


    def transform_face_point_grids_to_local(self, face_point_grids, coedge_lcs, mate, face):
        """
        将全局坐标系下的face点网格转换到局部坐标系下
        
        参数:
            face_point_grids: [num_faces, 9, num_u, num_v] 全局坐标系下的face网格
            coedge_lcs: [num_coedges, 4, 4] 每个coedge的局部坐标系变换矩阵
            mate: [num_coedges] mate关系数组，mate[i]是coedge i的mate coedge索引
            face: [num_coedges] coedge到face的映射关系，face[i]是coedge i所属的face索引
        
        返回:
            [num_coedges, 2, 9, num_u, num_v] 局部坐标系下的face网格
            第0个维度：coedge本身对应的face网格（转换到coedge的LCS）
            第1个维度：mate coedge对应的face网格（转换到mate coedge的LCS）
        """
        num_coedges = len(coedge_lcs)
        num_u = face_point_grids.shape[2]
        num_v = face_point_grids.shape[3]
        
        face_grids_local = np.zeros((num_coedges, 2, 9, num_u, num_v))
        
        for i in range(num_coedges):
            f_index = face[i]
            
            mate_i = mate[i]
            
            mf_index = face[mate_i]
            
            lcs_i = coedge_lcs[i]
            
            lcs_mate = coedge_lcs[mate_i]
            
            # 转换f到当前coedge的局部坐标系
            if f_index < len(face_point_grids):
                face_grid_global_f = face_point_grids[f_index]  # [9, num_u, num_v]
                face_grid_local_f = self.transform_face_point_grid_to_local(face_grid_global_f, lcs_i, num_u, num_v)
                face_grids_local[i, 0] = face_grid_local_f
            
            # 转换mf到mate coedge的局部坐标系
            if mf_index < len(face_point_grids):
                face_grid_global_mf = face_point_grids[mf_index]  # [9, num_u, num_v]
                face_grid_local_mf = self.transform_face_point_grid_to_local(face_grid_global_mf, lcs_mate, num_u, num_v)
                face_grids_local[i, 1] = face_grid_local_mf
        
        return face_grids_local


    def transform_face_point_grid_to_local(self, face_grid_global, lcs_matrix, num_u, num_v):
        """
        将face点网格变换到指定局部坐标系
        """
        points_global = face_grid_global[:3, :, :]  # [3, num_u, num_v]

        normals_global = face_grid_global[3:6, :, :]  # [3, num_u, num_v]

        mask = face_grid_global[6:7, :, :]  # [1, num_u, num_v]

        uvs = face_grid_global[7:, :, :]  # [2, num_u, num_v]

        lcs_inv = np.linalg.inv(lcs_matrix)
        rotation_matrix = lcs_inv[:3, :3]

        points_reshaped = points_global.reshape(3, -1)  # [3, num_u*num_v]
        points_homogeneous = np.vstack([points_reshaped, np.ones((1, points_reshaped.shape[1]))])
        points_local_homogeneous = lcs_inv @ points_homogeneous
        points_local = points_local_homogeneous[:3, :]
        points_local = points_local.reshape(3, num_u, num_v)

        normals_reshaped = normals_global.reshape(3, -1)
        normals_local_reshaped = rotation_matrix @ normals_reshaped
        normals_local = normals_local_reshaped.reshape(3, num_u, num_v)

        face_grid_local = np.concatenate([points_local, normals_local, mask, uvs], axis=0)
        return face_grid_local


    def try_to_project_normal(self, vec, ref):
        """
        Try to project the vector `ref` normal to vec
        """
        dp = np.dot(vec, ref)
        delta = dp*vec
        normal_dir = ref - delta
        length = np.linalg.norm(normal_dir)
        eps = 1e-7
        if length < eps:
            # Failed to project this vector normal
            return None

        # Return a unit vector
        return normal_dir/length


    def any_orthogonal(self, vec):
        """
        Find any random vector orthogonal to the given vector
        """
        nx = self.try_to_project_normal(vec, np.array([1, 0, 0]))
        if nx is not None:
            return nx
                
        ny = self.try_to_project_normal(vec, np.array([0, 1, 0]))
        if ny is not None:
            return ny

        nz = self.try_to_project_normal(vec, np.array([0, 0, 1]))
        assert nz is not None, f"Something is wrong with vec {vec}.  No orthogonal vector found"
        return nz


    def bounding_box_point_cloud(self, pts):
        assert pts.shape[1] == 3
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return np.array(box)

    
    def extract_coedge_reverse_flags(self, body, entity_mapper):
        """
        The flags for each coedge telling us if it is reversed wrt
        its parent edge.   Notice that when coedge features are 
        created, we need to reverse point ordering, flip tangent directions
        and swap left and right faces based on this flag.
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        reverse_flags = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(reverse_flags) == entity_mapper.halfedge_index(coedge)
                reverse_flags.append(self.reversed_edge_feature(coedge))
        return np.stack(reverse_flags)
    

    def reversed_edge_feature(self, edge):
        if edge.Orientation() == TopAbs_REVERSED:
            return 1.0
        return 0.0


    def build_incidence_arrays(self, body, entity_mapper):
        oriented_top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        num_coedges = len(entity_mapper.halfedge_map)

        next = np.zeros(num_coedges, dtype=np.uint32)
        mate = np.zeros(num_coedges, dtype=np.uint32)

        # Create the next, pervious and mate permutations
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            first_coedge_index = None
            previous_coedge_index = None
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)

                # Set up the mating coedge
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself.  This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                mate[coedge_index] = mating_coedge_index

                # Set up the next coedge
                if first_coedge_index == None:
                    first_coedge_index = coedge_index
                else:
                    next[previous_coedge_index] = coedge_index
                previous_coedge_index = coedge_index

            # Close the loop
            next[previous_coedge_index] = first_coedge_index

        # Create the arrays from coedge to face
        coedge_to_edge = np.zeros(num_coedges, dtype=np.uint32)
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)
                edge_index = entity_mapper.edge_index(coedge)
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself.  This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                coedge_to_edge[coedge_index] = edge_index
                coedge_to_edge[mating_coedge_index] = edge_index

        # Loop over the faces and make the back 
        # pointers back to the edges
        coedge_to_face = np.zeros(num_coedges, dtype=np.uint32)
        unoriented_top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        for face in unoriented_top_exp.faces():
            face_index = entity_mapper.face_index(face)
            for loop in unoriented_top_exp.wires_from_face(face):
                wire_exp =  TopologyUtils.WireExplorer(loop)
                for coedge in wire_exp.ordered_edges():
                    coedge_index = entity_mapper.halfedge_index(coedge)
                    coedge_to_face[coedge_index] = face_index

        return next, mate, coedge_to_face, coedge_to_edge


    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False

                coedge_set.add(tup)

        return True
        
    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0


    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set


    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set


    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True
            

def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)

def check_face_indices(step_file, mesh_dir):
    if mesh_dir is None:
        # Nothing to check
        return True
    # Check against the given meshes and Fusion labels    
    validator = FaceIndexValidator(step_file, mesh_dir)
    return validator.validate()

def crosscheck_faces_and_seg_file(file, seg_dir):
    seg_pathname = None
    if seg_dir is None:
        # Look to see if the seg file is in the step dir
        step_dir = file.parent
        trial_seg_pathname = step_dir / (file.stem + ".seg")
        if trial_seg_pathname.exists():
            seg_pathname = trial_seg_pathname
    else:
        # We expect to find the segmentation file in the 
        # seg dir
        seg_pathname = seg_dir / (file.stem + ".seg")
        if not seg_pathname.exists():
            print(f"Warning!! Segmentation file {seg_pathname} is missing")
            return False
    
    if seg_pathname is not None:
        checker = SegmentationFileCrosschecker(file, seg_pathname)
        data_ok = checker.check_data()
        if not data_ok:
            print(f"Warning!! Segmentation file {seg_pathname} and step file {file} have different numbers of faces")
        return data_ok
    
    # In the case where we don't know the seg pathname we don't do 
    # any extra checking
    return True

def extract_brepnet_features(file, output_path, mesh_dir, seg_dir):
    if not check_face_indices(file, mesh_dir):
        return
    if not crosscheck_faces_and_seg_file(file, seg_dir):
        return
    extractor = BRepNetExtractor(file, output_path)
    extractor.process()

def run_worker(worker_args):
    file = worker_args[0]
    output_path = worker_args[1]
    mesh_dir = worker_args[2]
    seg_dir = worker_args[3]
    extract_brepnet_features(file, output_path, mesh_dir, seg_dir)

def filter_out_files_which_are_already_converted(files, output_path):
    files_to_convert = []
    for file in files:
        output_file = output_path / (file.stem + ".npz")
        if not output_file.exists():
            files_to_convert.append(file)
    return files_to_convert


def extract_brepnet_data_from_step(
        step_path, 
        output_path, 
        mesh_dir=None,
        seg_dir=None,
        force_regeneration=True,
        num_workers=1
    ):
    files = [ f for f in step_path.glob("**/*.stp")]
    step_files = [ f for f in step_path.glob("**/*.step")]
    files.extend(step_files)

    if not force_regeneration:
        files = filter_out_files_which_are_already_converted(files, output_path)

    use_many_threads = num_workers > 1
    if use_many_threads:
        worker_args = [(f, output_path, mesh_dir, seg_dir) for f in files]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(run_worker, worker_args), total=len(worker_args)))
    else:
        for file in tqdm(files):
            extract_brepnet_features(file, output_path, mesh_dir, seg_dir)

    gc.collect()
    print("Completed pipeline/extract_feature_data_from_step.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument(
        "--mesh_dir", 
        type=str,  
        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels"
    )
    parser.add_argument(
        "--seg_dir", 
        type=str,  
        help="Optionally provide a directory containing segmentation labels seg files."
    )
    args = parser.parse_args()

    step_path = Path(args.step_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    mesh_dir = None
    if args.mesh_dir is not None:
        mesh_dir = Path(args.mesh_dir)

    seg_dir = None
    if args.seg_dir is not None:
        seg_dir = Path(args.seg_dir)

    extract_brepnet_data_from_step(step_path, output_path, mesh_dir, seg_dir, args.num_workers)