@echo off
chcp 65001 >nul
echo ============================================
echo 天空龍 視窗 1 啟動中...
echo ============================================

:: 設定環境變數
set WINDOW_ID=1
set FINLAB_API_KEY=R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m
set DISCORD_WEBHOOK=https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk

:: 如果自動偵測失敗，手動設定 Google Drive 路徑（取消註解並修改）
:: set GOOGLE_DRIVE_PATH=G:\My Drive

:: 執行程式
python 十二組合快快龍_優化版_高胃納量.py --window 1

pause
