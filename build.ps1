# build.ps1 - CUDA 碎形專案建置腳本
# =========================================

param(
    [Parameter(Position=0)]
    [string]$Target = "all"
)

# 設定
$SRC_DIR = "src"
$BUILD_DIR = "build"
$OUTPUT_DIR = "output"
$TARGET_EXE = "fractal_explorer.exe"
$NVCC = "nvcc"
$ARCH = "-arch=sm_75"  # RTX 3070 使用 sm_86，請根據您的 GPU 調整

# 顏色輸出函數
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# 建立目錄的函數
function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

# 清理函數
function Clean-Build {
    Write-ColorOutput Yellow "正在清理建置產物..."
    
    if (Test-Path $BUILD_DIR) {
        Remove-Item -Recurse -Force $BUILD_DIR
    }
    
    if (Test-Path $TARGET_EXE) {
        Remove-Item -Force $TARGET_EXE
    }
    
    if (Test-Path "$OUTPUT_DIR\*.ppm") {
        Remove-Item -Force "$OUTPUT_DIR\*.ppm"
    }
    
    Write-ColorOutput Green "清理完成"
}

# 建置函數
function Build-Project {
    Write-ColorOutput Yellow "正在建置專案..."
    
    # 建立必要目錄
    Ensure-Directory $BUILD_DIR
    Ensure-Directory "$BUILD_DIR\core"
    Ensure-Directory "$BUILD_DIR\fractals"
    Ensure-Directory $OUTPUT_DIR
    
    # 編譯旗標
    $COMPILE_FLAGS = "-O2", $ARCH
    
    # 編譯各個模組
    $sources = @(
        @{src="$SRC_DIR\core\color.cu"; obj="$BUILD_DIR\core\color.o"},
        @{src="$SRC_DIR\fractals\mandelbrot.cu"; obj="$BUILD_DIR\fractals\mandelbrot.o"},
        @{src="$SRC_DIR\fractals\burningship.cu"; obj="$BUILD_DIR\fractals\burningship.o"},
        @{src="$SRC_DIR\core\image_io.c"; obj="$BUILD_DIR\core\image_io.o"},
        @{src="$SRC_DIR\main.c"; obj="$BUILD_DIR\main.o"}
    )
    
    $allSuccess = $true
    foreach ($source in $sources) {
        Write-Host "編譯: $($source.src)"
        $compileCmd = "& $NVCC $COMPILE_FLAGS -c `"$($source.src)`" -o `"$($source.obj)`""
        Invoke-Expression $compileCmd
        
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput Red "編譯失敗: $($source.src)"
            $allSuccess = $false
            break
        }
    }
    
    if (-not $allSuccess) {
        return $false
    }
    
    # 連結
    Write-Host "連結執行檔: $TARGET_EXE"
    $objects = ($sources | ForEach-Object { "`"$($_.obj)`"" }) -join " "
    $linkCmd = "& $NVCC $COMPILE_FLAGS -o `"$TARGET_EXE`" $objects"
    Invoke-Expression $linkCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "連結失敗"
        return $false
    }
    
    Write-ColorOutput Green "建置成功: $TARGET_EXE"
    return $true
}

# 執行函數
function Run-Program {
    if (-not (Test-Path $TARGET_EXE)) {
        Write-ColorOutput Red "執行檔不存在，請先建置專案"
        return
    }
    
    Write-ColorOutput Yellow "執行程式..."
    & ".\$TARGET_EXE"
}

# 主邏輯
switch ($Target.ToLower()) {
    "clean" {
        Clean-Build
    }
    "all" {
        Build-Project
    }
    "rebuild" {
        Clean-Build
        Build-Project
    }
    "run" {
        if (Build-Project) {
            Run-Program
        }
    }
    default {
        Write-ColorOutput Red "無效的目標: $Target"
        Write-Host "支援的目標: all (預設), clean, rebuild, run"
        exit 1
    }
}
