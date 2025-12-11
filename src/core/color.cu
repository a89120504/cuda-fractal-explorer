// color.cu - 色彩處理模組實作
#include "color.h"
#include <math.h>

// HSV 到 RGB 轉換函數 (在 GPU 上執行)
__device__ RgbColor hsvToRgb(HsvColor hsv) {
    RgbColor rgb_color;
    float r_f, g_f, b_f; // 使用浮點數進行中間計算
    int i;
    float f, p, q, t;

    if (hsv.s < 0.00001f) { // 飽和度極低 (接近0)，視為灰色
        r_f = g_f = b_f = hsv.v;
    } else {
        // hsv.h 假定在 [0, 360] 範圍內
        float h_scaled = hsv.h / 60.0f; // 將 Hue 縮放到 [0, 6)
        if (h_scaled >= 6.0f) h_scaled = 0.0f; // 處理 h=360 的情況
        if (h_scaled < 0.0f) h_scaled += 6.0f; // 處理 h<0 的情況

        i = (int)floorf(h_scaled);
        f = h_scaled - i; // Hue 的小數部分

        p = hsv.v * (1.0f - hsv.s);
        q = hsv.v * (1.0f - hsv.s * f);
        t = hsv.v * (1.0f - hsv.s * (1.0f - f));

        switch (i) { // i 只可能是 0 到 5
            case 0: r_f = hsv.v; g_f = t;     b_f = p;     break;
            case 1: r_f = q;     g_f = hsv.v; b_f = p;     break;
            case 2: r_f = p;     g_f = hsv.v; b_f = t;     break;
            case 3: r_f = p;     g_f = q;     b_f = hsv.v; break;
            case 4: r_f = t;     g_f = p;     b_f = hsv.v; break;
            default:r_f = hsv.v; g_f = p;     b_f = q;     break; // case 5
        }
    }
    rgb_color.r = (unsigned char)(r_f * 255.0f);
    rgb_color.g = (unsigned char)(g_f * 255.0f);
    rgb_color.b = (unsigned char)(b_f * 255.0f);
    return rgb_color;
}
