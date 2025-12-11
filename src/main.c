#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "utils/helper_cuda.h"
#include "core/image_io.h"
#include "fractals/mandelbrot.h"
#include "fractals/burningship.h"

// 碎形類型列舉
typedef enum {
    FRACTAL_MANDELBROT,
    FRACTAL_BURNINGSHIP
} FractalType;

int main(void) {
    int width = 1280;
    int height = 960;
    int maxIterations = 500;

    // 讓使用者選擇碎形類型
    FractalType fractalType;
    printf("=== CUDA 碎形圖案生成與探索 ===\n");
    printf("請選擇碎形類型:\n");
    printf("  1: Mandelbrot Set\n");
    printf("  2: Burning Ship\n");
    printf("選擇 (1 或 2): ");
    
    int choice;
    if (scanf("%d", &choice) != 1 || (choice != 1 && choice != 2)) {
        fprintf(stderr, "無效的選擇，預設使用 Mandelbrot Set\n");
        fractalType = FRACTAL_MANDELBROT;
    } else {
        fractalType = (choice == 1) ? FRACTAL_MANDELBROT : FRACTAL_BURNINGSHIP;
    }
    
    // 清除輸入緩衝區
    int c;
    while ((c = getchar()) != '\n' && c != EOF);

    // 根據碎形類型設定初始參數
    float centerX, centerY, scale;
    const char* fractalName;
    
    if (fractalType == FRACTAL_MANDELBROT) {
        centerX = -0.75f;
        centerY = 0.0f;
        scale = 2.5f;
        fractalName = "mandelbrot";
    } else {
        centerX = -0.4f;
        centerY = -1.0f;
        scale = 2.0f;
        fractalName = "burningship";
    }

    unsigned char* h_image = NULL;
    unsigned char* d_image = NULL;

    size_t imageBytes = (size_t)width * height * 3 * sizeof(unsigned char);

    // 分配 Host 記憶體
    h_image = (unsigned char*)malloc(imageBytes);
    if (!h_image) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return -1;
    }

    // 分配 Device 記憶體
    CUDA_CHECK(cudaMalloc((void**)&d_image, imageBytes));

    char command;
    int frameCount = 0;
    char filename[256];

    printf("\n%s 碎形探索器 (CUDA-C 版本)\n", 
           fractalType == FRACTAL_MANDELBROT ? "Mandelbrot" : "Burning Ship");
    printf("初始視圖: CenterX=%.10f, CenterY=%.10f, Scale=%.10f\n", centerX, centerY, scale);
    printf("控制說明:\n");
    printf("  z: 放大 (縮小 scale)\n");
    printf("  x: 縮小 (放大 scale)\n");
    printf("  w: 向上移動\n");
    printf("  s: 向下移動\n");
    printf("  a: 向左移動\n");
    printf("  d: 向右移動\n");
    printf("  i: 增加迭代次數\n");
    printf("  k: 減少迭代次數\n");
    printf("  q: 結束\n");

    do {
        printf("\n正在生成第 %d 幀...\n", frameCount);
        printf("當前: CX=%.10f, CY=%.10f, Scale=%.10f, Iter=%d\n", 
               centerX, centerY, scale, maxIterations);

        // 根據碎形類型啟動對應的 CUDA 核心
        if (fractalType == FRACTAL_MANDELBROT) {
            launchMandelbrotKernel(d_image, width, height, centerX, centerY, scale, maxIterations);
        } else {
            launchBurningShipKernel(d_image, width, height, centerX, centerY, scale, maxIterations);
        }

        // 將結果從 Device 複製回 Host
        CUDA_CHECK(cudaMemcpy(h_image, d_image, imageBytes, cudaMemcpyDeviceToHost));

        // 儲存影像到 output 目錄
        sprintf(filename, "output/%s_%03d.ppm", fractalName, frameCount);
        savePPM(filename, h_image, width, height);

        frameCount++;

        printf("輸入指令 (z,x,w,s,a,d,i,k,q): ");
        if (scanf(" %c", &command) != 1) {
            fprintf(stderr, "讀取指令錯誤，結束程式。\n");
            command = 'q';
        }
        
        // 清除輸入緩衝區
        while ((c = getchar()) != '\n' && c != EOF);

        float moveStep = scale * 0.1f;
        float scaleFactor = 0.5f;

        switch (command) {
            case 'z': scale *= scaleFactor; break;
            case 'x': scale /= scaleFactor; break;
            case 'w': centerY += moveStep; break;
            case 's': centerY -= moveStep; break;
            case 'a': centerX -= moveStep; break;
            case 'd': centerX += moveStep; break;
            case 'i': maxIterations = (int)(maxIterations * 1.5); break;
            case 'k': 
                maxIterations = (int)(maxIterations / 1.5); 
                if (maxIterations < 50) maxIterations = 50; 
                break;
            case 'q': printf("結束程式...\n"); break;
            default: printf("未知指令。\n"); break;
        }

    } while (command != 'q');

    // 釋放記憶體
    free(h_image);
    CUDA_CHECK(cudaFree(d_image));

    // 重置 CUDA 設備
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
