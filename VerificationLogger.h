#pragma once
#include <iostream>
#include <string>
#include <set>
#include "BRepTorch.h"

#ifndef ENABLE_VERIFICATION
#define ENABLE_VERIFICATION 0
#endif

namespace Verification {
    /**
     * Log a message with a tag (always executed when ENABLE_VERIFICATION is on)
     */
    template<typename T>
    inline void Log(const std::string& tag, const T& value) {
        #if ENABLE_VERIFICATION
        std::cout << "[Verify:" << tag << "] " << value << std::endl;
        #endif
    }

    /**
     * Log a message only once per tag (useful for avoiding repeated output in loops)
     */
    template<typename T>
    inline void LogOnce(const std::string& tag, const T& value) {
        #if ENABLE_VERIFICATION
        static std::set<std::string> printed;
        if (printed.find(tag) == printed.end()) {
            std::cout << "[Verify:" << tag << "] " << value << std::endl;
            printed.insert(tag);
        }
        #endif
    }

    /**
     * Log tensor statistics (shape, mean, max)
     */
    inline void LogTensor(const std::string& tag, const breptorch::Tensor& t) {
        #if ENABLE_VERIFICATION
        std::cout << "[Verify:" << tag << "] Shape=[";
        for (int i = 0; i < t.dim(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << t.size(i);
        }
        std::cout << "] Mean=" << t.mean().item<float>()
                  << " Max=" << t.max().item<float>() << std::endl;
        #endif
    }

    /**
     * Log tensor slice (for detailed inspection)
     */
    inline void LogTensorSlice(const std::string& tag, const breptorch::Tensor& t,
                                int dim0_start, int dim0_end, int dim1_start, int dim1_end) {
        #if ENABLE_VERIFICATION
        std::cout << "[Verify:" << tag << "] Slice[" << dim0_start << ":" << dim0_end
                  << ", " << dim1_start << ":" << dim1_end << "]:\n"
                  << t.slice(0, dim0_start, dim0_end).slice(1, dim1_start, dim1_end) << std::endl;
        #endif
    }
}
