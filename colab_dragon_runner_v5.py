#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 九組合天空龍 GA v5.0 - Google Colab 快速啟動腳本
================================================================================

【使用方式】

方式一：直接在 Colab 執行（推薦）
--------------------------------
# 步驟 1: 下載程式
!curl -sL "https://raw.githubusercontent.com/windbug001/lolodo/claude/ga-stock-optimizer-1HrX1/nine_combo_dragon_ga_v5.py" -o nine_combo_dragon_ga_v5.py

# 步驟 2: 設定參數並執行
import os
os.environ['WINDOW_ID'] = '1'                    # 視窗 ID (1-4)
os.environ['CONTINUE_EVOLUTION'] = 'false'       # 是否繼續上次進度
os.environ['EVOLUTION_GENERATIONS'] = '100'      # 演化世代數
os.environ['POPULATION_SIZE'] = '30'             # 族群大小

%run nine_combo_dragon_ga_v5.py


方式二：使用此啟動腳本
--------------------------------
# 直接執行此檔案
%run colab_dragon_runner_v5.py


【參數說明】

環境變數:
- WINDOW_ID            : 視窗編號 (1-4)，多視窗平行優化時使用
- CONTINUE_EVOLUTION   : 'true' 繼續上次 / 'false' 重新開始
- EVOLUTION_GENERATIONS: 總演化世代數 (建議 100-300)
- POPULATION_SIZE      : 族群大小 (建議 20-50)
- USE_PARALLEL         : 'true' 使用並行 / 'false' 序列
- BASE_DIR             : 輸出目錄 (預設 Google Drive)
- FINLAB_API_KEY       : FinLab API 金鑰

【系統需求】
- Google Colab Pro+ (建議使用 High-RAM 模式)
- Python 3.8+
- 預計執行時間: 每 10 代約 20-40 分鐘

【九大策略說明】
S1: 低波動價值股    - PE合理、低波動、營收穩定
S2: 小型成長股      - 中小市值、正現金流、RSV動能
S3: 營收雙渦輪      - 營收創高、股價創高、強動能
S4: 高殖利率價值股  - 高殖利率、董監持股、多頭排列
S5: 低波動穩健股    - 最低波動、均線多頭、低融資
S6: 創高突破股      - 260日新高、量增、營收正成長
S7: 財務品質股      - 綜合財務指標排名
S8: 技術動能股      - SMA動能、斜率、量能趨勢
S9: 綜合品質動能股  - 品質+動能+安全綜合

================================================================================
"""

import os
import sys

# =============================================================================
# 配置區 - 請根據需求修改
# =============================================================================

# 視窗 ID (多視窗平行執行時修改此值: 1, 2, 3, 4)
WINDOW_ID = os.environ.get('WINDOW_ID', '1')

# 是否繼續上次的演化進度
CONTINUE = os.environ.get('CONTINUE_EVOLUTION', 'false')

# 總演化世代數
GENERATIONS = os.environ.get('EVOLUTION_GENERATIONS', '100')

# 族群大小
POP_SIZE = os.environ.get('POPULATION_SIZE', '30')

# 是否使用並行運算
USE_PARALLEL = os.environ.get('USE_PARALLEL', 'true')

# FinLab API Key (如需更換請修改)
API_KEY = os.environ.get('FINLAB_API_KEY', '')

# 輸出目錄
BASE_DIR = os.environ.get('BASE_DIR', '')

# =============================================================================
# 自動配置
# =============================================================================

# 設定環境變數
os.environ['WINDOW_ID'] = WINDOW_ID
os.environ['CONTINUE_EVOLUTION'] = CONTINUE
os.environ['EVOLUTION_GENERATIONS'] = GENERATIONS
os.environ['POPULATION_SIZE'] = POP_SIZE
os.environ['USE_PARALLEL'] = USE_PARALLEL

if API_KEY:
    os.environ['FINLAB_API_KEY'] = API_KEY
if BASE_DIR:
    os.environ['BASE_DIR'] = BASE_DIR

# 顯示配置
print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🚀 九組合天空龍 GA v5.0 - Colab 快速啟動                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 視窗 ID:     {WINDOW_ID:<5}                                                        ║
║  🔄 繼續進度:    {CONTINUE:<5}                                                        ║
║  🧬 演化世代:    {GENERATIONS:<5}                                                        ║
║  👥 族群大小:    {POP_SIZE:<5}                                                        ║
║  ⚡ 並行運算:    {USE_PARALLEL:<5}                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 執行主程式
# =============================================================================

def main():
    """啟動 GA 優化系統"""
    try:
        # 檢查環境
        try:
            from google.colab import drive
            print("✅ 偵測到 Google Colab 環境")

            if not os.path.exists('/content/drive/MyDrive'):
                print("📁 掛載 Google Drive...")
                drive.mount('/content/drive')

        except ImportError:
            print("⚠️  非 Colab 環境，使用本地模式")

        # 尋找主程式
        script_path = 'nine_combo_dragon_ga_v5.py'

        if not os.path.exists(script_path):
            print(f"⚠️  找不到 {script_path}，嘗試下載...")

            import urllib.request
            url = "https://raw.githubusercontent.com/windbug001/lolodo/claude/ga-stock-optimizer-1HrX1/nine_combo_dragon_ga_v5.py"

            try:
                urllib.request.urlretrieve(url, script_path)
                print(f"✅ 已下載 {script_path}")
            except Exception as e:
                print(f"❌ 下載失敗: {e}")
                print("請手動下載程式檔案")
                return

        # 執行
        print("\n🚀 啟動九組合天空龍 GA 優化系統...\n")
        exec(open(script_path, encoding='utf-8').read(), globals())

    except KeyboardInterrupt:
        print("\n⚠️  使用者中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
