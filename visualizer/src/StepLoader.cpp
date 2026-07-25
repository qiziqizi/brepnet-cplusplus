#include "StepLoader.h"
#include <STEPControl_Reader.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

StepLoader::StepLoader() {
}

StepLoader::~StepLoader() {
}

bool StepLoader::loadFile(const std::string& filePath) {
    clear();

    // 检查文件是否存在
    if (!fs::exists(filePath)) {
        std::cerr << "[StepLoader] 文件不存在: " << filePath << std::endl;
        return false;
    }

    // 使用OCCT的STEPControl_Reader加载文件
    STEPControl_Reader reader;
    IFSelect_ReturnStatus status = reader.ReadFile(filePath.c_str());

    if (status != IFSelect_RetDone) {
        std::cerr << "[StepLoader] 无法读取STEP文件: " << filePath << std::endl;
        return false;
    }

    // 转换为OpenCASCADE形状
    Standard_Integer nbRoots = reader.TransferRoots();
    if (nbRoots == 0) {
        std::cerr << "[StepLoader] STEP文件没有有效的根形状" << std::endl;
        return false;
    }

    shape_ = reader.OneShape();
    if (shape_.IsNull()) {
        std::cerr << "[StepLoader] 形状为空" << std::endl;
        return false;
    }

    // 提取所有面
    extractFaces();

    // 保存文件名
    fileName_ = fs::path(filePath).filename().string();

    std::cout << "[StepLoader] 成功加载: " << fileName_
              << " (" << faces_.size() << " 个面)" << std::endl;

    return true;
}

void StepLoader::extractFaces() {
    faces_.clear();

    // 使用 TopTools_IndexedMapOfShape 去重，与 BRepPipeline 保持一致
    // BRepPipeline 用 unique_faces.Add(f) 去重，顺序为首次出现顺序
    TopTools_IndexedMapOfShape faceMap;
    for (TopExp_Explorer exp(shape_, TopAbs_FACE); exp.More(); exp.Next()) {
        faceMap.Add(TopoDS::Face(exp.Current()));
    }

    // 按去重后的顺序填充 faces_（与 BRepPipeline 的 unique_faces 遍历顺序一致）
    for (int i = 1; i <= faceMap.Extent(); ++i) {
        faces_.push_back(TopoDS::Face(faceMap.FindKey(i)));
    }
}

void StepLoader::clear() {
    shape_.Nullify();
    faces_.clear();
    fileName_.clear();
}
