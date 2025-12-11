@echo off
REM setup_and_build.bat - 自動設定 VS 環境並建置專案
REM ========================================================

setlocal

echo ====================================
echo CUDA 碎形專案 - 自動建置腳本
echo ====================================
echo.

REM 尋找 Visual Studio 環境設定腳本
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

REM 嘗試使用 vswhere 找到 VS 安裝路徑
if exist %VSWHERE% (
    for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set VS_PATH=%%i
    )
)

REM 如果找到 VS，設定環境變數
if defined VS_PATH (
    set VCVARS="%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
    if exist %VCVARS% (
        echo 找到 Visual Studio: %VS_PATH%
        echo 正在設定編譯環境...
        call %VCVARS% > nul
        if errorlevel 1 (
            echo 設定 Visual Studio 環境失敗！
            goto error
        )
        echo 環境設定完成
        echo.
    ) else (
        echo 找不到 vcvars64.bat
        goto error
    )
) else (
    echo 找不到 Visual Studio 安裝
    goto manual_check
)

REM 執行建置
echo 開始建置專案...
call build.bat %1
exit /b %ERRORLEVEL%

:manual_check
echo.
echo 嘗試手動尋找 Visual Studio...
set VS2022_PATH=C:\Program Files\Microsoft Visual Studio\2022
set VS2019_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019
set VS2017_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2017

for %%V in (Community Professional Enterprise) do (
    if exist "%VS2022_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" (
        echo 找到 Visual Studio 2022 %%V
        call "%VS2022_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" > nul
        goto build_now
    )
    if exist "%VS2019_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" (
        echo 找到 Visual Studio 2019 %%V
        call "%VS2019_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" > nul
        goto build_now
    )
    if exist "%VS2017_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" (
        echo 找到 Visual Studio 2017 %%V
        call "%VS2017_PATH%\%%V\VC\Auxiliary\Build\vcvars64.bat" > nul
        goto build_now
    )
)

echo 找不到 Visual Studio！
goto error

:build_now
echo 環境設定完成
echo.
echo 開始建置專案...
call build.bat %1
exit /b %ERRORLEVEL%

:error
echo.
echo ====================================
echo 錯誤：無法設定建置環境
echo ====================================
echo.
echo 請確認：
echo 1. 已安裝 Visual Studio 2017/2019/2022
echo 2. 安裝時選擇了 "使用 C++ 的桌面開發" 工作負載
echo 3. 或手動從 "Developer Command Prompt for VS" 執行 build.bat
echo.
exit /b 1

endlocal
