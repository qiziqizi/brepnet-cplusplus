#pragma once

// ============================================================================
// 调试输出配置（全局统一控制）
// ============================================================================
//
// 用途：控制所有头文件中的调试输出
//
// 设置为 0：禁用调试输出（适用于批量处理10000个文件）
// 设置为 1：启用调试输出（适用于调试单个文件）
//
// ============================================================================

#ifndef ENABLE_DEBUG_OUTPUT
#define ENABLE_DEBUG_OUTPUT 0  // 默认禁用
#endif

#include <iostream>
#include <iomanip>
#include <sstream>

#if ENABLE_DEBUG_OUTPUT
    #define DEBUG_COUT std::cout
#else
    // 禁用时：使用临时的 ostringstream 对象吸收所有输出但不产生任何副作用
    // 编译器会优化掉所有这些临时对象的创建
    #define DEBUG_COUT if (false) std::cout
#endif




