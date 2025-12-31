@echo off
chcp 65001 >nul
echo ============================================
echo 天空龍 三視窗同時啟動
echo ============================================
echo.
echo 即將開啟 3 個命令提示字元視窗...
echo.

:: 取得目前腳本所在目錄
cd /d "%~dp0"

:: 啟動三個視窗（各自獨立的 cmd 視窗）
start "天空龍 視窗 1" cmd /k "run_skydragon_window1.bat"
timeout /t 5 /nobreak >nul

start "天空龍 視窗 2" cmd /k "run_skydragon_window2.bat"
timeout /t 5 /nobreak >nul

start "天空龍 視窗 3" cmd /k "run_skydragon_window3.bat"

echo.
echo ✅ 三個視窗已啟動！
echo.
echo 提示：
echo   - 每個視窗獨立運行
echo   - Checkpoint 會同步到 Google Drive
echo   - 可隨時關閉任一視窗，下次從 checkpoint 繼續
echo.
pause
