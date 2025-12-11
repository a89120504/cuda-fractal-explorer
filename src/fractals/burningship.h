// burningship.h - Burning Ship 碎形模組標頭檔
#ifndef BURNINGSHIP_H
#define BURNINGSHIP_H

#ifdef __cplusplus
extern "C" {
#endif

// 啟動 Burning Ship CUDA 核心
void launchBurningShipKernel(unsigned char* d_image, int width, int height,
                             float centerX, float centerY, float scale,
                             int maxIterations);

#ifdef __cplusplus
}
#endif

#endif // BURNINGSHIP_H
