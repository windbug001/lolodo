#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 九組合媽媽龍 GA v4.0 - Google Colab 快速啟動腳本
================================================================================

【使用方式】

方式一：直接在 Colab 執行
--------------------------
# 步驟 1: 下載程式
!curl -sL "https://raw.githubusercontent.com/windbug001/lolodo/claude/ga-stock-optimizer-1HrX1/nine_combo_ga_v4.py" -o nine_combo_ga_v4.py

# 步驟 2: 執行優化
import os
os.environ['WINDOW_ID'] = '1'                    # 視窗 ID (多視窗執行時修改)
os.environ['CONTINUE_EVOLUTION'] = 'false'       # 是否繼續上次進度
os.environ['EVOLUTION_GENERATIONS'] = '100'      # 演化世代數

%run nine_combo_ga_v4.py


方式二：使用此啟動腳本
--------------------------
# 直接執行此檔案
%run colab_ga_runner.py


【參數說明】

環境變數:
- WINDOW_ID: 視窗編號 (1-4)，多視窗平行優化時使用
- CONTINUE_EVOLUTION: 'true' 繼續上次進度 / 'false' 重新開始
- EVOLUTION_GENERATIONS: 總演化世代數 (建議 100-300)
- FINLAB_API_KEY: FinLab API 金鑰 (可選，已內建預設)

【系統需求】
- Google Colab Pro+ (建議使用 High-RAM 模式)
- Python 3.8+
- 預計執行時間: 每 10 代約 30-60 分鐘

================================================================================
"""

import os
import sys

# =============================================================================
# 配置區 - 請根據需求修改
# =============================================================================

# 視窗 ID (多視窗平行執行時修改此值)
WINDOW_ID = os.environ.get('WINDOW_ID', '1')

# 是否繼續上次的演化進度
CONTINUE = os.environ.get('CONTINUE_EVOLUTION', 'false')

# 總演化世代數
GENERATIONS = os.environ.get('EVOLUTION_GENERATIONS', '100')

# FinLab API Key (如需更換請修改)
API_KEY = os.environ.get('FINLAB_API_KEY', '')

# =============================================================================
# 自動配置
# =============================================================================

# 設定環境變數
os.environ['WINDOW_ID'] = WINDOW_ID
os.environ['CONTINUE_EVOLUTION'] = CONTINUE
os.environ['EVOLUTION_GENERATIONS'] = GENERATIONS

if API_KEY:
    os.environ['FINLAB_API_KEY'] = API_KEY

# 顯示配置
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           🚀 九組合媽媽龍 GA v4.0 - Colab 快速啟動                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  視窗 ID:     {WINDOW_ID:<5}                                                   ║
║  繼續進度:    {CONTINUE:<5}                                                   ║
║  演化世代:    {GENERATIONS:<5}                                                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 執行主程式
# =============================================================================

def main():
    """啟動 GA 優化系統"""
    try:
        # 檢查是否在 Colab 環境
        try:
            from google.colab import drive
            print("✅ 偵測到 Google Colab 環境")

            # 檢查 Drive 是否已掛載
            if not os.path.exists('/content/drive/MyDrive'):
                print("📁 掛載 Google Drive...")
                drive.mount('/content/drive')

        except ImportError:
            print("⚠️  非 Colab 環境，使用本地模式")

        # 嘗試載入主程式
        script_path = 'nine_combo_ga_v4.py'

        if not os.path.exists(script_path):
            print(f"⚠️  找不到 {script_path}，嘗試下載...")

            # 下載主程式
            import urllib.request
            url = "https://raw.githubusercontent.com/windbug001/lolodo/claude/ga-stock-optimizer-1HrX1/nine_combo_ga_v4.py"

            try:
                urllib.request.urlretrieve(url, script_path)
                print(f"✅ 已下載 {script_path}")
            except Exception as e:
                print(f"❌ 下載失敗: {e}")
                print("請手動下載程式檔案")
                return

        # 執行主程式
        print("\n🚀 啟動 GA 優化系統...\n")
        exec(open(script_path, encoding='utf-8').read(), globals())

    except KeyboardInterrupt:
        print("\n⚠️  使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
