#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 三策略小小龍 v2.0 - Google Colab 快速啟動腳本
================================================================================

【使用方法】
1. 開啟 Google Colab
2. 複製此腳本到 Colab 儲存格
3. 修改 WINDOW_ID (1-4) 用於多視窗執行
4. 執行！

【多視窗執行】
- 視窗 1: WINDOW_ID = 1
- 視窗 2: WINDOW_ID = 2
- 視窗 3: WINDOW_ID = 3
- 視窗 4: WINDOW_ID = 4

================================================================================
"""

# ===== 🔥 設定視窗 ID（多視窗執行時請修改）=====
WINDOW_ID = 1  # 修改此值 (1-4)

# ===== 設定環境變數 =====
import os
os.environ['WINDOW_ID'] = str(WINDOW_ID)
os.environ['FINLAB_DISABLE_CACHE'] = '1'

print(f"🐉 三策略小小龍 v2.0 - 視窗 {WINDOW_ID}")
print("=" * 60)

# ===== 掛載 Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== 安裝必要套件 =====
print("\n📦 安裝必要套件...")
!pip install -q finlab deap joblib tqdm

# ===== 建立目錄並下載最新版本 =====
print("\n📥 下載最新版本...")
TARGET_DIR = "/content/drive/MyDrive/FinLab_GA_小小龍_v2"

# 🔥 先建立目錄
import os
os.makedirs(TARGET_DIR, exist_ok=True)

# 下載腳本
REPO_BRANCH = "claude/genetic-algorithm-stock-optimizer-hoaRM"
SCRIPT_URL = f"https://raw.githubusercontent.com/windbug001/lolodo/{REPO_BRANCH}/finlab_genetic_optimizer_v2.py"

!curl -sL "{SCRIPT_URL}" -o "{TARGET_DIR}/finlab_genetic_optimizer_v2.py"

# 驗證下載
if os.path.exists(f"{TARGET_DIR}/finlab_genetic_optimizer_v2.py"):
    print(f"✅ 下載完成: {TARGET_DIR}/finlab_genetic_optimizer_v2.py")
else:
    print("❌ 下載失敗，請檢查網路連線")
    raise FileNotFoundError("無法下載腳本")

# ===== 執行優化 =====
print("\n🚀 開始執行遺傳演算法優化...")
print("=" * 60)

%cd {TARGET_DIR}
%run finlab_genetic_optimizer_v2.py
