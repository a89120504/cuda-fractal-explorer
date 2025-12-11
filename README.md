# CUDA 碎形圖案生成與互動式探索

使用 CUDA 實現的高效能碎形圖案生成器，支援 Mandelbrot Set 與 Burning Ship 兩種經典碎形，提供互動式探索功能。

## 特性

- **高效能 CUDA 加速**：利用 GPU 並行運算快速生成高解析度碎形圖案
- **多種碎形支援**：Mandelbrot Set 和 Burning Ship
- **互動式探索**：即時縮放、移動和調整渲染參數
- **平滑著色**：採用 HSV 色彩空間實現平滑漸變效果
- **模組化架構**：清晰的程式碼組織，易於維護和擴展

## 系統需求

- **GPU**：支援 CUDA 的 NVIDIA 顯示卡（建議 Compute Capability 7.5 以上）
- **CUDA Toolkit**：CUDA 11.0 或更高版本
- **編譯器**：
  - **Windows**：Visual Studio 2017/2019/2022（需包含 C++ 開發工具）
  - **Linux**：GCC 或 Clang

## 專案結構

```
使用 CUDA 實現高效碎形圖案生成與互動式探索/
├── src/                      # 原始碼目錄
│   ├── core/                # 核心功能模組
│   │   ├── color.cu        # HSV/RGB 色彩轉換
│   │   ├── color.h
│   │   ├── image_io.c      # PPM 影像檔案 I/O
│   │   └── image_io.h
│   ├── fractals/            # 碎形演算法模組
│   │   ├── mandelbrot.cu   # Mandelbrot Set
│   │   ├── mandelbrot.h
│   │   ├── burningship.cu  # Burning Ship
│   │   └── burningship.h
│   ├── utils/               # 工具模組
│   │   └── helper_cuda.h   # CUDA 錯誤檢查
│   └── main.c               # 主程式
├── output/                   # 輸出影像目錄
├── build/                    # 建置產物目錄
├── Makefile                  # 建置系統
└── README.md                 # 本文件
```

## 建置說明

### Windows 環境

#### 方法 1：使用批次檔（推薦）

**重要**：需在 Visual Studio Developer Command Prompt 中執行

1. 開啟 "Developer Command Prompt for VS 2022"（或其他版本）
2. 切換到專案目錄
3. 執行建置指令：

```batch
# 編譯專案
build.bat

# 清理建置產物
build.bat clean

# 重新建置
build.bat rebuild

# 編譯並執行
build.bat run
```

如果沒有 Developer Command Prompt，可以在普通 PowerShell 中先執行：
```powershell
# 設定 Visual Studio 環境變數（路徑可能不同，請根據實際安裝位置調整）
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

#### 方法 2：使用 PowerShell 腳本

```powershell
# 編譯專案
.\build.ps1 all

# 清理建置產物
.\build.ps1 clean

# 重新建置
.\build.ps1 rebuild

# 編譯並執行
.\build.ps1 run
```

**注意**：PowerShell 腳本同樣需要 Visual Studio 環境。

### Linux 環境

#### 使用 Makefile

```bash
# 編譯專案
make

# 清理建置產物
make clean

# 編譯並執行
make run
```

### 手動編譯

```bash
# 建立必要目錄
mkdir build build\core build\fractals output

# 編譯 CUDA 和 C 原始碼
nvcc -O2 -arch=sm_75 -c src/core/color.cu -o build/core/color.o
nvcc -O2 -arch=sm_75 -c src/fractals/mandelbrot.cu -o build/fractals/mandelbrot.o
nvcc -O2 -arch=sm_75 -c src/fractals/burningship.cu -o build/fractals/burningship.o
nvcc -O2 -arch=sm_75 -c src/core/image_io.c -o build/core/image_io.o
nvcc -O2 -arch=sm_75 -c src/main.c -o build/main.o

# 連結
nvcc -O2 -arch=sm_75 -o fractal_explorer.exe build/core/color.o build/fractals/mandelbrot.o build/fractals/burningship.o build/core/image_io.o build/main.o
```

**注意**：`-arch=sm_75` 適用於 RTX 20/30 系列顯卡，請根據您的 GPU 調整：
- RTX 40 系列：`sm_89`
- RTX 30 系列：`sm_86`
- RTX 20 系列：`sm_75`
- GTX 16 系列：`sm_75`

## 使用說明

### 啟動程式

```bash
.\fractal_explorer.exe
```

### 選擇碎形類型

程式啟動時會提示選擇碎形類型：
- **1**: Mandelbrot Set（經典 Mandelbrot 集合）
- **2**: Burning Ship（燃燒船碎形）

### 互動控制

程式會依據您的指令生成碎形影像並儲存至 `output/` 目錄。

| 按鍵 | 功能 |
|------|------|
| `z` | 放大視圖（縮小 scale 值） |
| `x` | 縮小視圖（放大 scale 值） |
| `w` | 向上移動 |
| `s` | 向下移動 |
| `a` | 向左移動 |
| `d` | 向右移動 |
| `i` | 增加迭代次數（提高細節） |
| `k` | 減少迭代次數（加快渲染） |
| `q` | 結束程式 |

### 輸出檔案

影像會以 PPM 格式儲存於 `output/` 目錄：
- Mandelbrot：`output/mandelbrot_000.ppm`, `output/mandelbrot_001.ppm`, ...
- Burning Ship：`output/burningship_000.ppm`, `output/burningship_001.ppm`, ...

可使用 GIMP、IrfanView 或線上 PPM 檢視器開啟。

## 技術細節

### 碎形演算法

**Mandelbrot Set**：
- 迭代公式：`z_(n+1) = z_n^2 + c`
- 初始值：`z_0 = 0`
- 逃逸半徑：`|z| > 2`

**Burning Ship**：
- 迭代公式：`z_(n+1) = (|Re(z_n)| + i|Im(z_n)|)^2 + c`
- 特色：對實部和虛部取絕對值後再平方

### 平滑著色

採用平滑迭代值公式避免色帶：
```
smooth_iter = iter + 1 - log(log(|z|)) / log(2)
```
將結果映射至 HSV 色彩空間，再轉換為 RGB 輸出。

### 效能優化

- GPU 並行運算：每個像素由獨立的 CUDA thread 計算
- Block 大小：16×16（可根據 GPU 調整）
- 記憶體管理：Host 和 Device 記憶體分離，最小化傳輸

## 範例探索路徑

### Mandelbrot Set 有趣區域

1. **經典全景**
   - Center: (-0.75, 0.0)
   - Scale: 2.5

2. **海馬谷**
   - Center: (-0.77568377, 0.13646737)
   - Scale: 0.0001

### Burning Ship 有趣區域

1. **整體結構**
   - Center: (-0.4, -1.0)
   - Scale: 2.0

2. **細節探索**
   - Center: (-1.75, -0.05)
   - Scale: 0.15

## 故障排除

### 編譯錯誤

- **找不到 nvcc**：確認 CUDA Toolkit 已安裝且加入 PATH
- **架構不匹配**：調整 Makefile 中的 `-arch=sm_XX` 參數

### 執行錯誤

- **CUDA 記憶體不足**：降低 `width` 和 `height` 值（在 `main.c` 中）
- **無影像輸出**：確認 `output/` 目錄存在

## 授權

本專案僅供學習和研究使用。

## 貢獻

歡迎提交 Issue 和 Pull Request！

---

**享受探索無限複雜的碎形世界！** 🌀
