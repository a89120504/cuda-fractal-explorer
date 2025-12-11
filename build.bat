@echo off
REM build.bat - CUDA 碎形專案建置腳本（需要 Visual Studio 環境）
REM =====================================================================

setlocal

REM 設定路徑
set SRC_DIR=src
set BUILD_DIR=build
set OUTPUT_DIR=output
set TARGET_EXE=fractal_explorer.exe

REM CUDA 編譯器設定（根據您的 GPU 調整 arch）
REM RTX 3070: sm_86, RTX 2070: sm_75
set NVCC=nvcc
set ARCH=-arch=sm_86
set COMPILE_FLAGS=-O2 %ARCH%

REM 檢查參數
if "%1"=="clean" goto clean
if "%1"=="rebuild" goto rebuild
if "%1"=="run" goto run
if "%1"=="" goto build
if "%1"=="all" goto build

echo 無效的目標: %1
echo 支援的目標: all, clean, rebuild, run
exit /b 1

:clean
echo 正在清理建置產物...
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
if exist %TARGET_EXE% del /q %TARGET_EXE%
if exist %OUTPUT_DIR%\*.ppm del /q %OUTPUT_DIR%\*.ppm
echo 清理完成
exit /b 0

:rebuild
call :clean
call :build
exit /b %ERRORLEVEL%

:build
echo 正在建置專案...

REM 建立必要目錄
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
if not exist %BUILD_DIR%\core mkdir %BUILD_DIR%\core
if not exist %BUILD_DIR%\fractals mkdir %BUILD_DIR%\fractals
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 編譯各個模組
echo 編譯: %SRC_DIR%\core\color.cu
%NVCC% %COMPILE_FLAGS% -c %SRC_DIR%\core\color.cu -o %BUILD_DIR%\core\color.o
if errorlevel 1 goto error

echo 編譯: %SRC_DIR%\fractals\mandelbrot.cu
%NVCC% %COMPILE_FLAGS% -c %SRC_DIR%\fractals\mandelbrot.cu -o %BUILD_DIR%\fractals\mandelbrot.o
if errorlevel 1 goto error

echo 編譯: %SRC_DIR%\fractals\burningship.cu
%NVCC% %COMPILE_FLAGS% -c %SRC_DIR%\fractals\burningship.cu -o %BUILD_DIR%\fractals\burningship.o
if errorlevel 1 goto error

echo 編譯: %SRC_DIR%\core\image_io.c
%NVCC% %COMPILE_FLAGS% -c %SRC_DIR%\core\image_io.c -o %BUILD_DIR%\core\image_io.o
if errorlevel 1 goto error

echo 編譯: %SRC_DIR%\main.c
%NVCC% %COMPILE_FLAGS% -c %SRC_DIR%\main.c -o %BUILD_DIR%\main.o
if errorlevel 1 goto error

REM 連結
echo 連結執行檔: %TARGET_EXE%
%NVCC% %COMPILE_FLAGS% -o %TARGET_EXE% ^
    %BUILD_DIR%\core\color.o ^
    %BUILD_DIR%\fractals\mandelbrot.o ^
    %BUILD_DIR%\fractals\burningship.o ^
    %BUILD_DIR%\core\image_io.o ^
    %BUILD_DIR%\main.o

if errorlevel 1 goto error

echo.
echo ====================================
echo 建置成功: %TARGET_EXE%
echo ====================================
exit /b 0

:run
if not exist %TARGET_EXE% (
    echo 執行檔不存在，正在建置...
    call :build
    if errorlevel 1 exit /b 1
)
echo 執行程式...
%TARGET_EXE%
exit /b %ERRORLEVEL%

:error
echo.
echo ====================================
echo 建置失敗！
echo ====================================
echo.
echo 常見問題排除:
echo 1. 確認已安裝 Visual Studio 並包含 C++ 開發工具
echo 2. 請從 "Developer Command Prompt for VS" 執行此腳本
echo 3. 或者執行以下命令後再執行此腳本:
echo    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo 4. 如果路徑不同，請尋找您的 Visual Studio 安裝路徑
exit /b 1

endlocal
