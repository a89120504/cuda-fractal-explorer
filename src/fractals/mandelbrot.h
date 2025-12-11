// mandelbrot.h - Mandelbrot 碎形模組標頭檔
#ifndef MANDELBROT_H
#define MANDELBROT_H

#ifdef __cplusplus
extern "C" {
#endif

// 啟動 Mandelbrot CUDA 核心
void launchMandelbrotKernel(unsigned char* d_image, int width, int height,
                           float centerX, float centerY, float scale,
                           int maxIterations);

#ifdef __cplusplus
}
#endif

#endif // MANDELBROT_H
