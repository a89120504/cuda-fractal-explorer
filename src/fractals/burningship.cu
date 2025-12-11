// burningship.cu - Burning Ship 碎形模組實作
#include "burningship.h"
#include "../core/color.h"
#include "../utils/helper_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// 核心函數：計算 Burning Ship 碎形
__global__ void burningShipKernel(unsigned char* Pout, int width, int height,
                                  float centerX, float centerY, float scale,
                                  int maxIterations) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= width || pixel_y >= height) {
        return;
    }

    // 將像素座標映射到複數平面
    float x0 = centerX + (pixel_x - width / 2.0f) * scale / width;
    float y0 = centerY + (pixel_y - height / 2.0f) * scale / width;

    // Burning Ship: c 是像素座標, z_initial = 0
    float c_real = x0;
    float c_imag = y0;
    float z_real = 0.0f;
    float z_imag = 0.0f;

    int iter = 0;
    for (iter = 0; iter < maxIterations; ++iter) {
        float z_real_sq_for_escape = z_real * z_real;
        float z_imag_sq_for_escape = z_imag * z_imag;

        if (z_real_sq_for_escape + z_imag_sq_for_escape > 4.0f) {
            break;
        }

        // Burning Ship 迭代公式: z_new = (|Re(z_old)| + i*|Im(z_old)|)^2 + c
        float abs_z_real = fabsf(z_real);
        float abs_z_imag = fabsf(z_imag);

        float z_real_next = (abs_z_real * abs_z_real) - (abs_z_imag * abs_z_imag) + c_real;
        float z_imag_next = 2.0f * abs_z_real * abs_z_imag + c_imag;

        z_real = z_real_next;
        z_imag = z_imag_next;
    }

    int pixelIdx = (pixel_y * width + pixel_x) * 3;

    if (iter == maxIterations) {
        // 點在集合內，設為黑色
        Pout[pixelIdx + 0] = 0;   // R
        Pout[pixelIdx + 1] = 0;   // G
        Pout[pixelIdx + 2] = 0;   // B
    } else {
        // 點在集合外，進行平滑 HSV 著色
        float mod_z_sq = z_real * z_real + z_imag * z_imag;

        float smooth_iter_val = (float)iter + 1.0f - logf(logf(sqrtf(mod_z_sq))) / logf(2.0f);

        if (smooth_iter_val < 0.0f) {
            smooth_iter_val = 0.0f;
        }

        HsvColor hsv;
        hsv.h = fmodf(smooth_iter_val * 20.0f, 360.0f);
        if (hsv.h < 0.0f) hsv.h += 360.0f;
        hsv.s = 0.9f;
        hsv.v = 0.85f;

        RgbColor rgb = hsvToRgb(hsv);

        Pout[pixelIdx + 0] = rgb.r;
        Pout[pixelIdx + 1] = rgb.g;
        Pout[pixelIdx + 2] = rgb.b;
    }
}

// Host 端啟動函數
extern "C" void launchBurningShipKernel(unsigned char* d_image, int width, int height,
                                        float centerX, float centerY, float scale,
                                        int maxIterations) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    burningShipKernel<<<gridSize, blockSize>>>(d_image, width, height,
                                                centerX, centerY, scale, maxIterations);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
