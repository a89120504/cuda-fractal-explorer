# Makefile for CUDA Fractal Explorer
# ====================================

# 編譯器設定
NVCC = nvcc
CC = gcc

# 編譯旗標
NVCC_FLAGS = -O2 -arch=sm_75
CC_FLAGS = -O2

# 目錄設定
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output

# 目標執行檔
TARGET = fractal_explorer.exe

# 原始碼檔案
CU_SOURCES = $(SRC_DIR)/core/color.cu \
             $(SRC_DIR)/fractals/mandelbrot.cu \
             $(SRC_DIR)/fractals/burningship.cu

C_SOURCES = $(SRC_DIR)/core/image_io.c \
            $(SRC_DIR)/main.c

# 目標檔案
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))

OBJECTS = $(CU_OBJECTS) $(C_OBJECTS)

# 預設目標
all: $(TARGET)

# 建立目標執行檔
$(TARGET): $(OBJECTS) | $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJECTS)
	@echo "建置完成: $(TARGET)"

# 編譯 CUDA 原始碼
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@if not exist "$(dir $@)" mkdir "$(subst /,\,$(dir $@))"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 編譯 C 原始碼
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@if not exist "$(dir $@)" mkdir "$(subst /,\,$(dir $@))"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 建立建置目錄
$(BUILD_DIR):
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
	@if not exist "$(BUILD_DIR)\core" mkdir "$(BUILD_DIR)\core"
	@if not exist "$(BUILD_DIR)\fractals" mkdir "$(BUILD_DIR)\fractals"

# 建立輸出目錄
$(OUTPUT_DIR):
	@if not exist "$(OUTPUT_DIR)" mkdir "$(OUTPUT_DIR)"

# 清理建置產物
clean:
	@if exist "$(BUILD_DIR)" rmdir /s /q "$(BUILD_DIR)"
	@if exist "$(TARGET)" del /q "$(TARGET)"
	@if exist "$(OUTPUT_DIR)\*.ppm" del /q "$(OUTPUT_DIR)\*.ppm"
	@echo "清理完成"

# 執行程式
run: $(TARGET)
	.\$(TARGET)

# 宣告偽目標
.PHONY: all clean run
