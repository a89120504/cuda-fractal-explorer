// color.h - 色彩處理模組標頭檔
#ifndef COLOR_H
#define COLOR_H

// RGB 色彩結構
typedef struct {
    unsigned char r, g, b;
} RgbColor;

// HSV 色彩結構 (Hue: 0-360度, Saturation: 0-1, Value: 0-1)
typedef struct {
    float h, s, v;
} HsvColor;

// HSV 到 RGB 轉換函數 (在 GPU 上執行)
#ifdef __CUDACC__
__device__ RgbColor hsvToRgb(HsvColor hsv);
#endif

#endif // COLOR_H
