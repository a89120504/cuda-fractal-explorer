// helper_cuda.h
#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> // 必須有

// 使用 do-while(0) 結構使宏在任何地方都能安全地作為單條語句使用
#define CUDA_CHECK(ans) \
    do { \
        gpuAssert((ans), __FILE__, __LINE__); \
    } while (0)

// 確保這個 inline 函數對所有包含它的檔案都可見
static inline void gpuAssert(cudaError_t code, const char *file, int line) { // <<--- 加入 static
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#endif // HELPER_CUDA_H