#include "StepLoader.h"
#include <STEPControl_Reader.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <IFSelect_ReturnStatus.hxx>
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

    // 使用TopExp_Explorer遍历所有面
    // 注意：遍历顺序必须与预测模块保持一致！
    for (TopExp_Explorer exp(shape_, TopAbs_FACE); exp.More(); exp.Next()) {
        TopoDS_Face face = TopoDS::Face(exp.Current());
        faces_.push_back(face);
    }
}

void StepLoader::clear() {
    shape_.Nullify();
    faces_.clear();
    fileName_.clear();
}
