// mandelbrot.cu - Mandelbrot 碎形模組實作
#include "mandelbrot.h"
#include "../core/color.h"
#include "../utils/helper_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// 核心函數：計算 Mandelbrot 集合
__global__ void mandelbrotKernel(unsigned char* Pout, int width, int height,
                                 float centerX, float centerY, float scale,
                                 int maxIterations) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= width || pixel_y >= height) {
        return;
    }

    float x0 = centerX + (pixel_x - width / 2.0f) * scale / width;
    float y0 = centerY + (pixel_y - height / 2.0f) * scale / width;

    float z_real = 0.0f;
    float z_imag = 0.0f;
    float c_real = x0;
    float c_imag = y0;

    int iter = 0;
    for (iter = 0; iter < maxIterations; ++iter) {
        float z_real_sq = z_real * z_real;
        float z_imag_sq = z_imag * z_imag;

        if (z_real_sq + z_imag_sq > 4.0f) {
            break;
        }

        float z_real_temp = z_real_sq - z_imag_sq + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = z_real_temp;
    }

    int pixelIdx = (pixel_y * width + pixel_x) * 3;

    if (iter == maxIterations) {
        // 點在集合內，設為黑色
        Pout[pixelIdx + 0] = 0;   // R
        Pout[pixelIdx + 1] = 0;   // G
        Pout[pixelIdx + 2] = 0;   // B
    } else {
        // 點在集合外，進行平滑著色
        float mod_z_sq = z_real * z_real + z_imag * z_imag;

        // 計算平滑迭代值
        float smooth_iter = (float)iter + 1.0f - logf(logf(sqrtf(mod_z_sq))) / logf(2.0f);

        if (smooth_iter < 0.0f) {
            smooth_iter = 0.0f;
        }

        HsvColor hsv;
        hsv.h = fmodf(smooth_iter * 12.0f, 360.0f);
        if (hsv.h < 0.0f) hsv.h += 360.0f;
        hsv.s = 0.85f;
        hsv.v = 0.90f;

        RgbColor rgb = hsvToRgb(hsv);

        Pout[pixelIdx + 0] = rgb.r;
        Pout[pixelIdx + 1] = rgb.g;
        Pout[pixelIdx + 2] = rgb.b;
    }
}

// Host 端函數，用於呼叫 CUDA 核心
extern "C" void launchMandelbrotKernel(unsigned char* d_image, int width, int height,
                                       float centerX, float centerY, float scale,
                                       int maxIterations) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, width, height, 
                                               centerX, centerY, scale, maxIterations);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
