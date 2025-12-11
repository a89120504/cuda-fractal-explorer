// image_io.h - 影像輸入/輸出模組標頭檔
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

// 儲存 PPM 格式影像 (P6 格式 - 二進制 RGB)
void savePPM(const char* filename, const unsigned char* data, int width, int height);

#endif // IMAGE_IO_H
