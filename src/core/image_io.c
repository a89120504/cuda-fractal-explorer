// image_io.c - 影像輸入/輸出模組實作
#include "image_io.h"
#include <stdio.h>

// 儲存 PPM 格式影像 (P6 格式 - 二進制 RGB)
void savePPM(const char* filename, const unsigned char* data, int width, int height) {
    FILE* outfile = fopen(filename, "wb"); // "wb" for binary write
    if (!outfile) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fprintf(outfile, "P6\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), (size_t)width * height * 3, outfile);
    fclose(outfile);
    printf("Image saved as %s\n", filename);
}
