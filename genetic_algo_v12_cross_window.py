#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 六組合快快龍 基因演算法優化系統 v12.3 (完整版)
   統一功能：樣本外測試 + 斷點續傳 Checkpoint
================================================================================

【v12.0 新增功能】
✅ 3視窗交互取優秀基因機制
✅ shared_best 共享資料夾自動同步
✅ 每N代自動交換各視窗精英
✅ 跨視窗基因多樣性維護

【v12.1 新增功能 - 安全防護】
✅ 安全寫入機制（先暫存再改名）
✅ 自動備份舊檔案
✅ 檔案完整性驗證
✅ 防止中斷導致檔案損壞

【v12.2 新增功能 - 樣本外測試】
✅ 訓練期/測試期分離 (2014~2022 訓練, 2023~2025 測試)
✅ 適應度評估僅使用訓練期數據
✅ 每5代顯示訓練期 vs 測試期夏普比較
✅ 自動過擬合警告（測試/訓練比 < 60%）
✅ 最終報告包含完整樣本外測試結果

【v12.3 新增功能 - 斷點續傳】
✅ CheckpointManager 完整管理
✅ 每5代自動保存 checkpoint
✅ 重啟時自動從上次世代繼續
✅ 保存/恢復隨機狀態確保可重現性

【v12.4 新增功能 - 指定起點演化】
✅ CHECKPOINT_PATH 環境變數支援
✅ 從任意 checkpoint 檔案開始演化
✅ 強制重新評估（使用新適應度函數）
✅ 最大回檔限制調整為 17%

版本：v12.4 Full-Featured (2026-01-02)
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 🔥 FinLab 快取清理
# =============================================================================
import os
import shutil
import gc

def clear_finlab_cache():
    cache_paths = [
        os.path.expanduser('~/.finlab'),
        os.path.expanduser('~/.cache/finlab'),
        '/tmp/finlab_cache',
        '/root/.finlab',
    ]
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
            except:
                pass
    gc.collect()

clear_finlab_cache()

# =============================================================================
# 導入套件
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import sys
import json
import pickle
import time
import glob
import hashlib
import requests
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import reduce

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    os.system('pip install tqdm -q')
    from tqdm.auto import tqdm

# =============================================================================
# 核心設定
# =============================================================================
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))
FINLAB_API_KEY = os.environ.get('FINLAB_API_KEY', 'YOUR_API_KEY_HERE')

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"

# 🔥 新增：從指定 checkpoint 開始演化
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)  # 例如: checkpoint_window_1_latest.pkl

# 🎯 2026 統一優化目標
TARGET_SHARPE = 4.2          # 🔥 目標夏普值 4.2
MIN_CAPACITY = 10_000_000    # 🔥 胃納量 1000萬
MAX_DRAWDOWN = 0.17          # 🔥 最大回檔 17%
TARGET_ANNUAL_RETURN = 0.4   # 年化報酬 40%

# 🔥 持股配比約束
MIN_POSITION_WEIGHT = 0.03   # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股必須是 3% 倍數

# 🔄 自適應持股設定
ADAPTIVE_POSITION = True     # 啟用自適應持股
BULL_MARKET_STOCKS = 10      # 多頭市場持股數
BEAR_MARKET_STOCKS = 5       # 空頭市場持股數
MARKET_THRESHOLD = 0.5       # 市場判斷閾值（>50% 股票在均線上=多頭）

# GA 演化參數
POPULATION_SIZE = 100        # 🔥 增加族群到 100
N_GENERATIONS = 300          # 🔥 增加至 300 代
MUTATION_RATE = 0.15         # 降低突變率以穩定
CROSSOVER_RATE = 0.8
ELITE_RATIO = 0.1

# 🔥 3視窗交互設定
CROSS_WINDOW_INTERVAL = 10   # 每10代交換一次
CROSS_WINDOW_TOP_N = 5       # 每次交換最佳5個基因
ALL_WINDOW_IDS = [1, 2, 3]   # 所有視窗ID

BACKTEST_START = '2014-01-01'
FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000
FULL_BACKTEST_INTERVAL = 5

# =============================================================================
# 🔬 樣本外測試設定 (Out-of-Sample Testing)
# =============================================================================
TRAIN_START = '2014-01-01'   # 訓練期開始
TRAIN_END = '2022-12-31'     # 訓練期結束
TEST_START = '2023-01-01'    # 測試期開始
TEST_END = '2025-12-31'      # 測試期結束（或使用最新日期）

# 過擬合警告閾值
OVERFIT_SHARPE_RATIO = 0.6   # 測試期夏普 / 訓練期夏普 < 0.6 則警告
OVERFIT_RETURN_RATIO = 0.5   # 測試期報酬 / 訓練期報酬 < 0.5 則警告

print(f"{'='*80}")
print(f"🚀 六組合快快龍 v12.4 (完整版) - Window {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"   📉 最大回檔限制：{MAX_DRAWDOWN*100:.0f}%")
print(f"   🔄 視窗交互：每 {CROSS_WINDOW_INTERVAL} 代交換 Top {CROSS_WINDOW_TOP_N} 基因")
print(f"   📊 訓練期：{TRAIN_START} ~ {TRAIN_END}")
print(f"   🔬 測試期：{TEST_START} ~ {TEST_END}")
if CHECKPOINT_PATH:
    print(f"   🔥 指定起點：{CHECKPOINT_PATH}")
print(f"{'='*80}")

# =============================================================================
# 套件載入
# =============================================================================
def install_packages():
    required = {'finlab': 'finlab', 'deap': 'deap', 'joblib': 'joblib'}
    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            os.system(f'pip install {pkg_name} -q')

install_packages()

import finlab
from finlab import data
from finlab.backtest import sim
from deap import base, creator, tools, algorithms

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 環境設定
# =============================================================================
def setup_environment():
    global BASE_DIR, IN_COLAB, SHARED_DIR
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        BASE_DIR = '/content/drive/MyDrive/投資策略優化_v12_六組合快快龍'
        IN_COLAB = True
        print("✅ Google Drive 已掛載")
    except:
        BASE_DIR = './ga_v12_output'
        IN_COLAB = False

    # 🔥 共享資料夾（所有視窗都會讀寫）
    SHARED_DIR = f"{BASE_DIR}/shared_best"
    Path(SHARED_DIR).mkdir(parents=True, exist_ok=True)

    if FINLAB_API_KEY != "YOUR_API_KEY_HERE":
        finlab.login(FINLAB_API_KEY)
        print("✅ FinLab VIP 登入成功")
    else:
        print("⚠️ 請設定 FINLAB_API_KEY")

    return BASE_DIR, IN_COLAB, SHARED_DIR

BASE_DIR, IN_COLAB, SHARED_DIR = setup_environment()
WINDOW_DIR = f"{BASE_DIR}/window_{WINDOW_ID}"
Path(WINDOW_DIR).mkdir(parents=True, exist_ok=True)
print(f"📁 工作目錄: {WINDOW_DIR}")
print(f"📁 共享目錄: {SHARED_DIR}")

# =============================================================================
# 🔒 安全檔案管理器 (v12.1 新增)
# =============================================================================
class SafeFileManager:
    """
    安全的檔案寫入管理器
    - 使用暫存檔 + 改名的原子性寫入
    - 自動備份舊檔案
    - 寫入後驗證完整性
    """

    @staticmethod
    def safe_pickle_save(data, filepath, min_size=100):
        """
        安全的 pickle 寫入（防止中斷導致損壞）

        Args:
            data: 要保存的資料
            filepath: 目標檔案路徑
            min_size: 最小有效檔案大小（bytes）

        Returns:
            bool: 是否成功
        """
        temp_path = filepath + '.tmp'
        backup_path = filepath + '.backup'

        try:
            # 1. 先寫入暫存檔
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)

            # 2. 驗證暫存檔
            if not SafeFileManager.verify_pickle_file(temp_path, min_size):
                print(f"   ⚠️ 暫存檔驗證失敗: {temp_path}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

            # 3. 備份舊檔案（如果存在且有效）
            if os.path.exists(filepath):
                if SafeFileManager.verify_pickle_file(filepath, min_size):
                    shutil.copy2(filepath, backup_path)

            # 4. 原子性替換（move 比 copy + delete 更安全）
            shutil.move(temp_path, filepath)

            # 5. 最終驗證
            if SafeFileManager.verify_pickle_file(filepath, min_size):
                return True
            else:
                # 嘗試從備份恢復
                if os.path.exists(backup_path):
                    print(f"   ⚠️ 最終驗證失敗，從備份恢復...")
                    shutil.copy2(backup_path, filepath)
                return False

        except Exception as e:
            print(f"   ⚠️ 安全寫入失敗: {e}")
            # 清理暫存檔
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            # 嘗試從備份恢復
            if os.path.exists(backup_path) and not os.path.exists(filepath):
                try:
                    shutil.copy2(backup_path, filepath)
                    print(f"   ✅ 已從備份恢復: {filepath}")
                except:
                    pass
            return False

    @staticmethod
    def verify_pickle_file(filepath, min_size=100):
        """
        驗證 pkl 檔案是否正常

        Args:
            filepath: 檔案路徑
            min_size: 最小有效大小（bytes）

        Returns:
            bool: 檔案是否有效
        """
        try:
            # 檢查檔案存在
            if not os.path.exists(filepath):
                return False

            # 檢查檔案大小
            size = os.path.getsize(filepath)
            if size < min_size:
                return False

            # 嘗試讀取並解析
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            return True

        except Exception as e:
            return False

    @staticmethod
    def safe_json_save(data, filepath):
        """安全的 JSON 寫入"""
        temp_path = filepath + '.tmp'

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            shutil.move(temp_path, filepath)
            return True

        except Exception as e:
            print(f"   ⚠️ JSON 寫入失敗: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

# 創建全域安全檔案管理器
safe_file_mgr = SafeFileManager()
print("✅ 安全檔案管理器已初始化")

# =============================================================================
# 持股配比管理器（3% 倍數約束）
# =============================================================================
class PositionWeightManager:
    """
    持股配比管理器

    功能：
    1. 確保每隻股票至少 3%
    2. 確保持股是 3% 倍數（3%, 6%, 9%, 12%...）
    3. 低於 3% 的直接設為 0%
    """

    @staticmethod
    def normalize_position(position_df) -> 'pd.DataFrame':
        """
        正規化持股比例

        規則：
        1. 每行獨立處理
        2. 低於 3% 的設為 0
        3. 剩餘的正規化並量化為 3% 倍數
        """
        if position_df is None or position_df.empty:
            return position_df

        try:
            # 確保是數值型
            result = position_df.astype(float).copy()

            for idx in result.index:
                row = result.loc[idx].copy()
                row_sum = row.sum()

                if row_sum == 0 or pd.isna(row_sum):
                    continue

                # 正規化
                normalized = row / row_sum

                # 過濾低於 3% 的
                filtered = normalized.where(normalized >= MIN_POSITION_WEIGHT, 0.0)

                # 重新正規化剩餘的
                filtered_sum = filtered.sum()
                if filtered_sum > 0:
                    filtered = filtered / filtered_sum

                    # 量化為 3% 倍數
                    quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

                    # 調整總和為 1
                    total = quantized.sum()
                    if total > 0 and abs(total - 1.0) > 0.01:
                        max_col = quantized.idxmax()
                        quantized.loc[max_col] += (1.0 - total)

                    result.loc[idx] = quantized.values
                else:
                    result.loc[idx] = 0.0

            return result.astype(float)

        except Exception as e:
            # 如果處理失敗，返回原始 DataFrame
            print(f"   ⚠️ 持股正規化失敗: {e}")
            return position_df

position_weight_mgr = PositionWeightManager()

# =============================================================================
# Discord 通知
# =============================================================================
class DiscordNotifier:
    def __init__(self, webhook_url=None):
        self.enabled = webhook_url and webhook_url != ""
        self.webhook_url = webhook_url
        self.start_time = None

        if self.enabled:
            try:
                payload = {"content": f"✅ 六組合快快龍 v12.3 (完整版) - Window {WINDOW_ID} 啟動\n   訓練期: {TRAIN_START}~{TRAIN_END}\n   測試期: {TEST_START}~{TEST_END}", "username": "快快龍"}
                requests.post(self.webhook_url, json=payload, timeout=5)
                print("✅ Discord 通知系統已連接")
            except:
                self.enabled = False

    def send(self, message):
        if not self.enabled:
            return
        try:
            payload = {"content": message, "username": "快快龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            time.sleep(1)
        except:
            pass

    def send_embed(self, title, description, color='info', fields=None):
        if not self.enabled:
            return
        try:
            color_map = {'success': 3066993, 'error': 15158332, 'warning': 16776960, 'info': 3447003}
            embed = {"title": title, "description": description, "color": color_map.get(color, 3447003)}
            if fields:
                embed["fields"] = fields
            payload = {"embeds": [embed], "username": "快快龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            time.sleep(1)
        except:
            pass

notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# =============================================================================
# 數據載入 (完整版 - 包含所有策略需要的數據)
# =============================================================================
print("\n📊 載入市場數據 (完整版)...")
data_start = time.time()

# === 價格數據 ===
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

# === 估值數據 ===
pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')

# === 月營收數據 ===
rev = data.get('monthly_revenue:當月營收')
rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
rev_mom = data.get('monthly_revenue:上月比較增減(%)')
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)

# === 基本面數據 ===
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
稅前淨利率 = data.get("fundamental_features:稅前淨利率")
營業利益率 = data.get('fundamental_features:營業利益率')

# === 籌碼數據 ===
融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")
inventory = data.get("inventory")

# === 財務報表數據 ===
股本 = data.get('financial_statement:股本')
投資活動淨現金 = data.get('financial_statement:投資活動之淨現金流入_流出')
營業活動淨現金 = data.get('financial_statement:營業活動之淨現金流入_流出')
自由現金流 = (投資活動淨現金 + 營業活動淨現金).rolling(4).mean()
稅後淨利 = data.get('fundamental_features:經常稅後淨利')
權益總計 = data.get('financial_statement:股東權益總額')
股東權益報酬率 = 稅後淨利 / 權益總計

# === 市值與成交金額 ===
市值 = data.get('etl:market_value')
成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

# === 當月營收計算市值營收比 ===
當月營收 = data.get('monthly_revenue:當月營收') * 1000
當季營收 = 當月營收.rolling(4).sum()
市值營收比 = 市值 / 當季營收

# === 技術指標 ===
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)
entry_volatility = atr / adj_close

# === 漲停計算 ===
limit_up = (close > close.shift(1) * 1.095)
entry_close = (close * close).replace(0.0, np.nan)
entry_high = (close * high).replace(0.0, np.nan)
entry_low = (close * low).replace(0.0, np.nan)
entry_open = (close * open_).replace(0.0, np.nan)

close_vs_high = (entry_close == entry_high).replace(False, np.nan)
close_vs_low = (entry_close == entry_low).replace(False, np.nan)
close_vs_open = (entry_close == entry_open).replace(False, np.nan)
close_high_low = (close_vs_high == close_vs_low).replace(False, np.nan)
close_high_open = (close_vs_high == close_vs_open).replace(False, np.nan)
close_high_low_open = (close_high_low == close_high_open).replace(False, np.nan)
limit_up_all_day = (close_high_low_open == limit_up)
limit_up_all_day = limit_up_all_day.fillna(False)

print(f"✅ 數據載入完成 ({time.time()-data_start:.1f}秒)")
print(f"   股票數: {len(close.columns)}, 期間: {close.index[0].date()} ~ {close.index[-1].date()}")

# =============================================================================
# 🔥 Checkpoint 管理器（支援斷點續傳）
# =============================================================================
class CheckpointManager:
    """Checkpoint 管理器 - 支援斷點續傳"""

    def __init__(self, window_id: int, base_dir: str):
        self.window_id = window_id
        self.checkpoint_dir = f"{base_dir}/window_{window_id}"
        self.checkpoint_file = f"{self.checkpoint_dir}/checkpoint.pkl"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(self, population, generation, halloffame=None, history=None):
        """保存 checkpoint"""
        try:
            checkpoint_data = {
                'population': [{'genes': list(ind), 'fitness': ind.fitness.values[0] if ind.fitness.valid else None}
                              for ind in population],
                'generation': generation,
                'halloffame': [{'genes': list(ind), 'fitness': ind.fitness.values[0]}
                              for ind in halloffame] if halloffame else [],
                'history': history or [],
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state(),
                'timestamp': datetime.now().isoformat(),
                'window_id': self.window_id
            }

            success = SafeFileManager.safe_pickle_save(checkpoint_data, self.checkpoint_file)
            if success:
                print(f"   💾 Checkpoint 已保存 (第 {generation} 代)")
            return success

        except Exception as e:
            print(f"   ⚠️ Checkpoint 保存失敗: {e}")
            return False

    def load(self):
        """載入 checkpoint"""
        if not os.path.exists(self.checkpoint_file):
            print("ℹ️ 無 checkpoint，從頭開始")
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            # 驗證 checkpoint 有效性
            if not data.get('population') or not data.get('generation'):
                print("⚠️ Checkpoint 無效，從頭開始")
                return None

            # 恢復隨機狀態
            if data.get('random_state'):
                random.setstate(data['random_state'])
            if data.get('numpy_state'):
                np.random.set_state(data['numpy_state'])

            print(f"✅ 從 checkpoint 恢復: 第 {data['generation']} 代, 族群 {len(data['population'])}")
            print(f"   📅 保存時間: {data.get('timestamp', 'N/A')}")

            return data

        except Exception as e:
            print(f"⚠️ Checkpoint 載入失敗: {e}")
            return None

    def exists(self) -> bool:
        """檢查是否有 checkpoint"""
        return os.path.exists(self.checkpoint_file)

    def get_info(self):
        """取得 checkpoint 資訊"""
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            return {
                'generation': data.get('generation', 0),
                'timestamp': data.get('timestamp', 'N/A'),
                'population_size': len(data.get('population', []))
            }
        except:
            return None

    def load_from_file(self, filepath: str, force_reevaluate: bool = True):
        """
        從指定的 checkpoint 檔案載入族群

        Args:
            filepath: checkpoint 檔案路徑
            force_reevaluate: 是否強制重新評估（用於換到新適應度函數）

        Returns:
            dict: checkpoint 資料，或 None 如果失敗
        """
        if not os.path.exists(filepath):
            print(f"⚠️ 指定的 checkpoint 不存在: {filepath}")
            return None

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # 驗證 checkpoint 有效性
            if not data.get('population'):
                print(f"⚠️ Checkpoint 無效（無族群資料）: {filepath}")
                return None

            pop_size = len(data['population'])
            gen = data.get('generation', 0)

            print(f"\n{'='*60}")
            print(f"🔥 從指定 Checkpoint 載入族群")
            print(f"   📂 檔案: {filepath}")
            print(f"   📊 族群大小: {pop_size}")
            print(f"   🔢 原始世代: {gen}")
            print(f"   📅 保存時間: {data.get('timestamp', 'N/A')}")

            if force_reevaluate:
                # 🔥 強制清除所有 fitness，用新適應度函數重新評估
                print(f"   ⚠️ 將強制重新評估所有個體（使用新適應度函數）")
                for ind_data in data['population']:
                    ind_data['fitness'] = None  # 清除舊 fitness
                # 從第 0 代開始（新的演化）
                data['generation'] = 0
                data['history'] = []

            print(f"{'='*60}\n")

            return data

        except Exception as e:
            print(f"⚠️ Checkpoint 載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

# 初始化 checkpoint 管理器（稍後初始化，需要 WINDOW_DIR）
checkpoint_mgr = None

# =============================================================================
# 🔥 3視窗交互機制 - 核心功能
# =============================================================================
class CrossWindowManager:
    """3視窗交互管理器"""

    def __init__(self, base_dir, window_id, shared_dir):
        self.base_dir = base_dir
        self.window_id = window_id
        self.shared_dir = shared_dir
        self.other_windows = [w for w in ALL_WINDOW_IDS if w != window_id]

        # 確保所有目錄存在
        Path(shared_dir).mkdir(parents=True, exist_ok=True)
        for w in ALL_WINDOW_IDS:
            Path(f"{base_dir}/window_{w}").mkdir(parents=True, exist_ok=True)

    def save_elites_to_shared(self, elites: List, generation: int):
        """
        將本視窗的精英基因保存到共享資料夾

        Args:
            elites: 精英基因列表 (DEAP Individual)
            generation: 當前代數
        """
        try:
            elite_data = {
                'window_id': self.window_id,
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'elites': []
            }

            for ind in elites:
                elite_data['elites'].append({
                    'gene': list(ind),
                    'fitness': ind.fitness.values[0] if ind.fitness.valid else 0.0
                })

            # 🔒 使用安全寫入保存到共享資料夾
            filename = f"{self.shared_dir}/window_{self.window_id}_elites.pkl"
            success = SafeFileManager.safe_pickle_save(elite_data, filename)

            if success:
                print(f"   💾 Window {self.window_id} 精英已安全保存到共享區 ({len(elites)} 個)")
                return True
            else:
                print(f"   ⚠️ 安全保存失敗，嘗試傳統寫入...")
                # 降級到傳統寫入
                with open(filename, 'wb') as f:
                    pickle.dump(elite_data, f)
                print(f"   💾 Window {self.window_id} 精英已保存到共享區 ({len(elites)} 個)")
                return True

        except Exception as e:
            print(f"   ⚠️ 保存精英失敗: {e}")
            return False

    def load_elites_from_other_windows(self, max_per_window: int = CROSS_WINDOW_TOP_N) -> List[Dict]:
        """
        從其他視窗載入精英基因

        Args:
            max_per_window: 每個視窗最多載入幾個

        Returns:
            來自其他視窗的精英基因列表
        """
        all_external_elites = []

        for other_window in self.other_windows:
            try:
                filename = f"{self.shared_dir}/window_{other_window}_elites.pkl"

                if not os.path.exists(filename):
                    continue

                # 檢查檔案是否太舊（超過1天就跳過）
                file_age = time.time() - os.path.getmtime(filename)
                if file_age > 86400:  # 24小時
                    continue

                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                elites = data.get('elites', [])[:max_per_window]

                for elite in elites:
                    all_external_elites.append({
                        'gene': elite['gene'],
                        'fitness': elite['fitness'],
                        'source_window': other_window,
                        'generation': data.get('generation', 0)
                    })

                print(f"   📥 從 Window {other_window} 載入 {len(elites)} 個精英")

            except Exception as e:
                print(f"   ⚠️ 載入 Window {other_window} 失敗: {e}")
                continue

        # 按適應度排序
        all_external_elites.sort(key=lambda x: x['fitness'], reverse=True)

        return all_external_elites

    def perform_cross_window_exchange(self, population: List, generation: int) -> List:
        """
        執行跨視窗基因交換

        Args:
            population: 當前族群
            generation: 當前代數

        Returns:
            交換後的族群
        """
        print(f"\n🔄 第 {generation} 代 - 執行3視窗基因交換")

        # 1. 保存本視窗精英到共享區
        local_elites = tools.selBest(population, CROSS_WINDOW_TOP_N)
        self.save_elites_to_shared(local_elites, generation)

        # 2. 從其他視窗載入精英
        external_elites = self.load_elites_from_other_windows()

        if not external_elites:
            print("   ℹ️ 暫無其他視窗精英可用")
            return population

        # 3. 將外部精英注入族群（替換最差的個體）
        n_inject = min(len(external_elites), len(population) // 4)  # 最多替換 1/4 族群

        # 按適應度排序族群（升序，最差的在前面）
        population.sort(key=lambda x: x.fitness.values[0] if x.fitness.valid else 0)

        injected_count = 0
        for i, ext_elite in enumerate(external_elites[:n_inject]):
            # 🔧 清理基因值（防止複數/NaN導致錯誤）
            clean_gene = sanitize_gene(ext_elite['gene'])
            new_ind = creator.Individual(clean_gene)
            new_ind.fitness.values = (ext_elite['fitness'],)

            # 替換最差的個體
            population[i] = new_ind
            injected_count += 1

        print(f"   ✅ 注入 {injected_count} 個外部精英 (來自 Window {[e['source_window'] for e in external_elites[:n_inject]]})")

        notifier.send(
            f"🔄 **Window {self.window_id} 基因交換**\n"
            f"• 第 {generation} 代\n"
            f"• 注入 {injected_count} 個外部精英\n"
            f"• 最佳外部適應度: {external_elites[0]['fitness']:.4f}"
        )

        return population

    def get_all_windows_best(self) -> Dict:
        """獲取所有視窗的最佳成績"""
        results = {}

        for window_id in ALL_WINDOW_IDS:
            try:
                filename = f"{self.shared_dir}/window_{window_id}_elites.pkl"
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                    if data.get('elites'):
                        results[window_id] = {
                            'best_fitness': data['elites'][0]['fitness'],
                            'generation': data.get('generation', 0),
                            'timestamp': data.get('timestamp', 'N/A')
                        }
            except:
                continue

        return results

# 初始化跨視窗管理器
cross_window_mgr = CrossWindowManager(BASE_DIR, WINDOW_ID, SHARED_DIR)

# 初始化 checkpoint 管理器
checkpoint_mgr = CheckpointManager(WINDOW_ID, BASE_DIR)
print("✅ Checkpoint 管理器已初始化")

# =============================================================================
# 安全條件計算函數
# =============================================================================
def safe_sustain(condition, periods, min_periods=None):
    try:
        if min_periods is None:
            result = condition.sustain(periods)
        else:
            result = condition.sustain(periods, min_periods)
        return result.fillna(False).astype(float)
    except Exception as e:
        return condition.fillna(False).astype(float) * 0

def safe_condition(cond):
    try:
        return cond.fillna(False).astype(float)
    except:
        return cond * 0

# =============================================================================
# gene_to_params (完整版 - 160 參數全用)
# =============================================================================
def gene_to_params(gene):
    """將基因轉換為策略參數 (完整版)"""
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    current_length = len(gene)

    if current_length < required_length:
        extension = [0.0] * (required_length - current_length)
        gene = gene + extension
    elif current_length > required_length:
        gene = gene[:required_length]

    # === 策略配置權重 (gene[0:6]) ===
    alloc_sum = sum(gene[0:6])
    if alloc_sum == 0:
        allocation = [1/6] * 6
    else:
        allocation = [gene[i]/alloc_sum for i in range(6)]

    # === 策略1: 低波動本益比策略參數 ===
    low_vol_pe_params = {
        'rev_ma3_ma12_ratio': gene[6],
        'rev_consistency': gene[7],
        'volatility_threshold': gene[8]/1000,
        'margin_usage_limit': gene[9],
        'non_op_income_limit': gene[10],
        'min_volume': abs(gene[11] * 1000),
        'pe_min': gene[12],
        'pe_max': gene[13],
        'top_n': max(1, int(gene[14])),
        'min_rev_yoy_threshold': -gene[120],
        'rev_decline_period': max(1, int(gene[121])),
        'max_rev_yoy_threshold': gene[122],
        'old_trend_period': max(1, int(gene[123])),
        'old_trend_match': max(1, int(gene[124])),
        'rev_bottom_window': max(1, int(gene[125])),
        'rev_bottom_ratio': gene[126],
        'rev_bottom_sustain': max(1, int(gene[127])),
        'min_rev_mom_growth': -gene[128],
        'rev_mom_sustain': max(1, int(gene[129])),
        'quarter_ma': max(1, int(gene[130])),
        'half_year_ma': max(1, int(gene[131])),
        'long_ma': max(1, int(gene[132])),
        'recent_rev_period': max(1, int(gene[133])),
        'annual_rev_period': max(1, int(gene[134])),
        'pb_min': gene[135],
        'pb_max': gene[136],
        'min_gpm': gene[137],
        'gpm_sustain_period': max(1, int(gene[138])),
        'min_roe': gene[139],
        'roe_sustain_period': max(1, int(gene[140]))
    }

    # === 策略2: 小資族策略參數 ===
    small_inv_params = {
        'market_value_limit': gene[15] * 1e9,
        'market_rev_ratio_limit': gene[16],
        'rev_yoy_growth_limit': -gene[17],
        'rev_mom_growth_limit': -gene[18],
        'rsv_period': max(1, int(gene[19])),
        'ma_period': max(1, int(gene[20])),
        'volume_threshold': abs(gene[21] * 1000),
        'top_n': max(1, int(gene[22])),
        'min_free_cash_flow': gene[150],
        'min_roe': gene[151],
        'min_op_profit_growth': gene[152]
    }

    # === 策略3: 營收股價雙渦輪策略參數 ===
    turbo_params = {
        'rev_ma_period': max(1, int(gene[23])),
        'rev_ma_lookback': max(1, int(gene[24])),
        'price_high_window': max(1, int(gene[25])),
        'min_volume': abs(gene[26] * 1000),
        'min_price': gene[27],
        'rsi_threshold': gene[28],
        'pe_limit': gene[29],
        'top_n': max(1, int(gene[30])),
        'performance_ma_period': max(1, int(gene[100])),
        'performance_threshold': gene[101],
        'rsi_trend_period': max(1, int(gene[102])),
        'min_gpm': gene[103],
        'gpm_sustain_period': max(1, int(gene[104])),
        'min_btpm': gene[105],
        'btpm_sustain_period': max(1, int(gene[106])),
        'min_atpm': gene[107],
        'atpm_sustain_period': max(1, int(gene[108])),
        'rev_growth_percentile': gene[109],
        'boss_min_level': max(1, int(gene[110])),
        'boss_max_level': max(1, int(gene[111])),
        'min_boss_ratio': gene[112]
    }

    # === 策略4: 高殖利率烏龜策略參數 ===
    high_yield_turtle_params = {
        'min_yield_ratio': gene[31],
        'min_op_earn_ratio': gene[32],
        'min_boss_hold': gene[33],
        'min_volume': gene[34] * 1000,
        'max_volume': gene[35] * 1000,
        'top_n': int(gene[36])
    }

    # === 策略5: 低波動性指標策略參數 ===
    low_vol_index_params = {
        'min_volume': gene[37] * 1000,
        'std_window': int(gene[38]),
        'std_threshold': gene[39],
        'top_n': int(gene[40])
    }

    # === 策略6: 藏獒外掛大盤指針策略參數 ===
    market_indicator_params = {
        'new_high_window': int(gene[43]),
        'min_year_growth': -gene[44],
        'max_year_growth': gene[45],
        'rev_bottom_ratio': gene[46],
        'min_month_growth': -gene[47],
        'min_volume': gene[48] * 1000,
        'top_n': int(gene[49])
    }

    # === 總體參數 ===
    overall_params = {
        'stop_loss': gene[51]/100,
        'trail_stop': gene[52]/100,
        'take_profit': gene[53]/100,
        'position_limit': gene[54]/100,
        'trade_at_price': ["open", "close", "high_low_avg", "open_close_avg"][int(gene[55]) % 4],
        'liquidity_threshold': gene[56] * 1e6,
    }

    return allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_turtle_params, low_vol_index_params, market_indicator_params, overall_params


# =============================================================================
# 🔄 自適應持股機制 - 根據市場狀況動態調整持股數量
# =============================================================================
def get_market_sentiment(close_df, ma_period=60):
    """
    計算市場情緒指標
    返回：每日市場情緒分數 (0~1)，越高代表越多股票在均線之上
    """
    ma = close_df.rolling(ma_period).mean()
    above_ma = (close_df > ma).astype(float)
    sentiment = above_ma.mean(axis=1)
    return sentiment


def get_adaptive_top_n(close_df, base_top_n, sentiment=None):
    """
    根據市場情緒動態調整持股數量
    - 多頭市場（>50% 股票在均線上）：增加持股
    - 空頭市場（<50% 股票在均線上）：減少持股
    """
    if not ADAPTIVE_POSITION:
        return base_top_n

    if sentiment is None:
        sentiment = get_market_sentiment(close_df)

    # 根據情緒調整持股數量
    bull_ratio = BULL_MARKET_STOCKS / base_top_n if base_top_n > 0 else 1
    bear_ratio = BEAR_MARKET_STOCKS / base_top_n if base_top_n > 0 else 0.5

    # 線性插值：空頭(0) -> 多頭(1)
    adaptive_ratio = bear_ratio + (bull_ratio - bear_ratio) * sentiment.clip(0, 1)
    adaptive_top_n = (base_top_n * adaptive_ratio).round().astype(int).clip(lower=3)

    return adaptive_top_n


# =============================================================================
# 策略函數 (完整版 - 包含所有條件)
# =============================================================================

def strategy_low_vol_pe(params):
    """策略1: 低波動本益比策略 (完整版)"""
    try:
        # PEG 計算
        peg = pe / 營業利益成長率

        # 營收條件
        cond1 = rev_ma3 / rev_ma12 > params['rev_ma3_ma12_ratio']
        cond2 = rev / rev.shift(1) > params['rev_consistency']

        # 篩選條件
        tree_select_factor = (
            (融資使用率 <= params['margin_usage_limit']) &
            (entry_volatility <= params['volatility_threshold']) &
            (業外收支營收率 < params['non_op_income_limit'])
        )

        # 成交量條件
        condition_近1日成交均量大於閾值 = vol.average(1) > params['min_volume']

        # 排除月營收連續衰退
        cond排除月營收連3月衰退 = ~(rev_yoy < params['min_rev_yoy_threshold']).sustain(params['rev_decline_period'])

        # 排除營收成長趨勢過老
        cond排除月營收成長趨勢過老 = ~(rev_yoy > params['max_rev_yoy_threshold']).sustain(
            params['old_trend_period'], params['old_trend_match'])

        # 確認營收底部
        cond確認營收底部 = ((rev.rolling(params['rev_bottom_window']).min()) / rev < params['rev_bottom_ratio']).sustain(
            params['rev_bottom_sustain'])

        # 單月營收月增率
        cond單月營收月增率 = (rev_mom > params['min_rev_mom_growth']).sustain(params['rev_mom_sustain'])

        # 收盤價大於均線
        cond收盤價大於季線及半年線 = (
            (close > close.average(params['quarter_ma'])) &
            (close > close.average(params['half_year_ma'])) &
            (close > close.average(params['long_ma']))
        )

        # 近期營收大於年營收
        cond近三個月營收大於年營收 = rev.average(params['recent_rev_period']) > rev.average(params['annual_rev_period'])

        # PE/PB 範圍
        pe_range_1 = (params['pe_min'] <= pe) & (pe <= params['pe_max'])
        pb_range_1 = (params['pb_min'] <= pb) & (pb <= params['pb_max'])

        # 毛利率趨勢
        gpm_trend_1 = (營業毛利率 > params['min_gpm']).sustain(params['gpm_sustain_period'])

        # ROE趨勢
        roe_trend_1 = (ROE綜合損益 > params['min_roe']).sustain(params['roe_sustain_period'])

        # 集保小戶佔比
        small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                             .reset_index()
                             .groupby(["date", "stock_id"])
                             .agg({"占集保庫存數比例": "sum"})
                             .reset_index()
                             .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46

        # 合併所有條件
        cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老 &
                    cond排除月營收連3月衰退 & cond收盤價大於季線及半年線 & cond近三個月營收大於年營收 &
                    cond單月營收月增率 & ~(limit_up_all_day) & condition_近1日成交均量大於閾值 &
                    gpm_trend_1 & roe_trend_1 & small_inv_under50 & pb_range_1 & pe_range_1)

        # 選擇最小 PEG
        position = peg[cond_all & (peg > 0)].is_smallest(params['top_n'])
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_low_vol_pe 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def strategy_small_cap(params):
    """策略2: 小資族策略 (完整版)"""
    try:
        # 市值條件
        cond1 = 市值 < params['market_value_limit']

        # 自由現金流
        cond2 = 自由現金流 > params['min_free_cash_flow']

        # 股東權益報酬率
        cond3 = 股東權益報酬率 > params['min_roe']

        # 營業利益成長率
        cond4 = 營業利益成長率 > params['min_op_profit_growth']

        # 市值營收比
        cond5 = 市值營收比 < params['market_rev_ratio_limit']

        # 成交量
        cond6 = vol > params['volume_threshold']

        # 排除月營收連3月衰退
        cond排除月營收連3月衰退 = ~(rev_yoy < params['rev_yoy_growth_limit']).sustain(3)

        # 排除營收成長趨勢過老
        cond排除月營收成長趨勢過老 = ~(rev_yoy > 60).sustain(12, 8)

        # 確認營收底部
        cond確認營收底部 = ((rev.rolling(12).min()) / rev < 1.2).sustain(3)

        # 單月營收月增率連續3月大於閾值
        cond單月營收月增率連續3月大於閾值 = (rev_mom > params['rev_mom_growth_limit']).sustain(3)

        # 收盤價大於均線
        ma_period = params['ma_period']
        cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(ma_period * 2))

        # 近三個月營收大於年營收
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)

        # 業外收支營收率佔比低
        業外收支營收率占比低 = 業外收支營收率 < 7.3

        # RSV 計算
        rsv_period = params['rsv_period']
        rsv = (close - close.rolling(rsv_period).min()) / (
            close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

        # 合併條件並依 RSV 排序
        position = ((cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond排除月營收成長趨勢過老 &
                     cond單月營收月增率連續3月大於閾值 & cond排除月營收連3月衰退 & cond近三個月營收大於年營收 &
                     cond收盤價大於均線 & 業外收支營收率占比低 & cond確認營收底部) * rsv).is_largest(params['top_n'])

        position = position.reindex(當月營收.index_str_to_date().index, method='ffill')
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_small_cap 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def strategy_turbo(params):
    """策略3: 營收股價雙渦輪策略 (完整版)"""
    try:
        rev_ma_period = max(1, int(params['rev_ma_period']))
        rev_ma = rev.average(rev_ma_period)
        rev_ma_lookback = max(rev_ma_period + 1, int(params['rev_ma_lookback']))

        # 近N月平均營收創M個月來新高
        condition_近N月平均營收創M個月來新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=1).max()

        # 近N日內有1日股價創新高
        price_high_window = max(1, int(params['price_high_window']))
        condition_近N日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)

        # 成交量條件
        condition_成交均量大於閾值 = vol.average(1) > params['min_volume']

        # 多頭均線排列
        long_ma_pattern = (
            (close > close.average(5)) &
            (close > close.average(10)) &
            (close > close.average(20)) &
            (close > close.average(60)) &
            (close > close.average(120))
        )

        # 收盤價超級績效
        收盤價_超級績效 = close > (close.average(params['performance_ma_period']) * params['performance_threshold'])

        # RSI 高趨勢
        rsi_higt_trend = (rsi > params['rsi_threshold']).sustain(params['rsi_trend_period'])

        # 毛利率趨勢
        gpm_trend_1 = (營業毛利率 > params['min_gpm']).sustain(params['gpm_sustain_period'])

        # 稅前淨利率趨勢
        btpm_trend_1 = (稅前淨利率 > params['min_btpm']).sustain(params['btpm_sustain_period'])

        # 稅後淨利率趨勢
        atpm_trend_1 = (稅後淨利率 > params['min_atpm']).sustain(params['atpm_sustain_period'])

        # 營收年增百分位排名
        rev_rise_nsatisfy_2 = rev_yoy.rank(pct=True, axis=1) > params['rev_growth_percentile']

        # 集保大戶佔比
        boss_inventory_over400 = (inventory[
            (inventory.持股分級.astype(int) >= params['boss_min_level']) &
            (inventory.持股分級.astype(int) <= params['boss_max_level'])]
            .reset_index()
            .groupby(["date", "stock_id"])
            .agg({"占集保庫存數比例": "sum"})
            .reset_index()
            .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= params['min_boss_ratio']

        # PE 上限
        pe_limit = params['pe_limit']
        pe_range_1 = pe_limit <= pe

        # 最低價格
        min_price = params['min_price']

        # 合併所有條件
        conditions = (
            condition_近N月平均營收創M個月來新高 & condition_近N日內有1日股價創新高 &
            condition_成交均量大於閾值 & long_ma_pattern & gpm_trend_1 & btpm_trend_1 &
            atpm_trend_1 & rev_rise_nsatisfy_2 & (close > min_price) & (業外收支營收率 < 7.3) &
            ~pe_range_1 & 收盤價_超級績效 & (vol >= vol.rolling(20).mean() * 0.8) &
            rsi_higt_trend & boss_inventory_over400 & ~limit_up_all_day
        )

        # 依營收年增排序
        position = rev_yoy * conditions
        position = position[position > 0].is_largest(params['top_n'])
        position = position.reindex(rev.index_str_to_date().index, method="ffill")
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_turbo 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def strategy_high_yield(params):
    """策略4: 高殖利率烏龜策略 (完整版)"""
    try:
        sma20 = close.average(20)
        sma60 = close.average(60)

        # 殖利率條件
        cond1 = dividend_yield >= params['min_yield_ratio']

        # 趨勢條件
        cond2 = (close > sma20) & (close > sma60)

        # 營收條件
        cond3 = rev.average(3) > rev.average(12)

        # 營業利益率條件
        cond4 = 營業利益率 >= params['min_op_earn_ratio']

        # 董監持股條件
        cond5 = 董監持有股數占比 >= params['min_boss_hold']

        # 成交量範圍
        cond6 = (vol.average(5) >= params['min_volume']) & (vol.average(5) <= params['max_volume'])

        # 合併條件
        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        cond_all = cond_all * rev_yoy

        # 依營收年增排序
        position = cond_all[cond_all > 0].is_largest(params['top_n'])
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_high_yield 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def strategy_low_vol(params):
    """策略5: 低波動性指標策略 (完整版)"""
    try:
        cap = 市值

        # 波動率排名
        std = close.pct_change().rolling(params['std_window']).std().rank(axis=1, pct=True)

        # 條件組合
        position = cap[
            (vol.average(20) > params['min_volume']) &
            (close > close.average(60)) &
            (close > close.average(120)) &
            (close > close.average(250)) &
            (std < params['std_threshold'])
        ].is_smallest(params['top_n'])

        position = position.reindex(close.index_str_to_date().index, method='ffill')
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_low_vol 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)


def strategy_market(params):
    """策略6: 藏獒外掛大盤指針策略 (完整版)"""
    try:
        vol_ma = vol.average(10)

        # 股價創新高
        cond1 = close == close.rolling(params['new_high_window']).max()

        # 排除月營收連3月衰退
        cond2 = ~(rev_yoy < params['min_year_growth']).sustain(3)

        # 排除營收成長趨勢過老
        cond3 = ~(rev_yoy > params['max_year_growth']).sustain(12, 8)

        # 確認營收底部
        cond4 = ((rev.rolling(12).min()) / rev < params['rev_bottom_ratio']).sustain(3)

        # 月營收月增率連續3月
        cond5 = (rev_mom > params['min_month_growth']).sustain(3)

        # 成交量條件
        cond6 = vol_ma > params['min_volume']

        # 合併條件
        buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        buy = vol_ma * buy
        buy = buy[buy > 0]
        buy = buy.is_smallest(params['top_n'])

        position = buy.reindex(rev.index_str_to_date().index, method='ffill')
        return position.fillna(0).astype(float)

    except Exception as e:
        print(f"   ⚠️ strategy_market 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

# =============================================================================
# 持股配比正規化
# =============================================================================
def normalize_weights(position_df):
    result = pd.DataFrame(0.0, index=position_df.index, columns=position_df.columns)

    for date in position_df.index:
        row = position_df.loc[date]
        total = row.sum()
        if total == 0:
            continue

        normalized = row / total
        filtered = normalized[normalized >= MIN_POSITION_WEIGHT]

        if filtered.empty:
            max_stocks = int(1.0 / MIN_POSITION_WEIGHT)
            filtered = normalized.nlargest(min(len(normalized), max_stocks))
            if filtered.sum() > 0:
                filtered = filtered / filtered.sum()

        if filtered.sum() == 0:
            continue

        filtered = filtered / filtered.sum()
        quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

        diff = 1.0 - quantized.sum()
        if abs(diff) > 0.001 and len(quantized) > 0:
            quantized[quantized.idxmax()] += diff

        quantized = quantized[quantized >= MIN_POSITION_WEIGHT - 0.001]
        result.loc[date, quantized.index] = quantized

    return result

# =============================================================================
# 合併策略
# =============================================================================
def combined_strategy(gene, apply_normalization=True, start_date=None, end_date=None):
    """
    合併六個策略並生成持倉

    Args:
        gene: 基因向量
        apply_normalization: 是否正規化權重
        start_date: 開始日期 (用於樣本外測試)
        end_date: 結束日期 (用於樣本外測試)
    """
    try:
        allocation, p1, p2, p3, p4, p5, p6, overall = gene_to_params(gene)

        pos1 = strategy_low_vol_pe(p1) * allocation[0]
        pos2 = strategy_small_cap(p2) * allocation[1]
        pos3 = strategy_turbo(p3) * allocation[2]
        pos4 = strategy_high_yield(p4) * allocation[3]
        pos5 = strategy_low_vol(p5) * allocation[4]
        pos6 = strategy_market(p6) * allocation[5]

        combined = pos1.add(pos2, fill_value=0).add(pos3, fill_value=0).add(pos4, fill_value=0).add(pos5, fill_value=0).add(pos6, fill_value=0)

        liquid = 平均成交金額 > overall['liquidity_threshold']
        combined = combined * liquid.astype(float)

        # 🔥 關鍵修正：reindex 到月營收截止日（稀疏 index）
        # 這樣 stop_loss 才能在非換股日正確生效
        monthly_rev_index = rev.index_str_to_date().index
        combined = combined.reindex(monthly_rev_index, method='ffill')

        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index, errors='coerce')

        # 使用指定日期或預設值
        filter_start = start_date if start_date else BACKTEST_START
        combined = combined[combined.index >= pd.Timestamp(filter_start)]

        # 如果有結束日期，過濾結束日期
        if end_date:
            combined = combined[combined.index <= pd.Timestamp(end_date)]

        if apply_normalization and not combined.empty:
            combined = normalize_weights(combined)

        return combined, overall

    except Exception as e:
        print(f"⚠️ 合併策略錯誤: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

# =============================================================================
# 回測
# =============================================================================
def run_backtest(gene, upload=False, name="Strategy"):
    """完整期間回測（用於最終結果）"""
    try:
        position, params = combined_strategy(gene)
        if position.empty or position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        position = position_weight_mgr.normalize_position(position)

        # 🔥 修正：不使用 resample，讓 position 的月營收稀疏 index 自然換股
        # 使用 params 中的 stop_loss/trail_stop/take_profit
        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
            name=name,
            upload=upload
        )

        if report is None:
            return None

        metrics = report.get_metrics()
        return {
            'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
            'sortino': metrics['ratio'].get('sortinoRatio', 0) or 0,
            'calmar': metrics['ratio'].get('calmarRatio', 0) or 0,
            'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
            'capacity': metrics['liquidity'].get('capacity', 0) or 0,
            'report': report,
            'position': position,
            'gene': gene
        }
    except Exception as e:
        print(f"   ⚠️ 回測錯誤: {e}")
        return None


def run_backtest_period(gene, start_date, end_date, name="Strategy"):
    """
    🔬 指定期間回測（用於樣本外測試）

    Args:
        gene: 基因向量
        start_date: 開始日期
        end_date: 結束日期
        name: 策略名稱
    """
    try:
        position, params = combined_strategy(gene, start_date=start_date, end_date=end_date)
        if position.empty or position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        position = position_weight_mgr.normalize_position(position)

        # 🔥 修正：不使用 resample，讓 position 的月營收稀疏 index 自然換股
        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
            name=name,
            upload=False
        )

        if report is None:
            return None

        metrics = report.get_metrics()
        return {
            'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
            'sortino': metrics['ratio'].get('sortinoRatio', 0) or 0,
            'calmar': metrics['ratio'].get('calmarRatio', 0) or 0,
            'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
            'capacity': metrics['liquidity'].get('capacity', 0) or 0,
            'period': f"{start_date} ~ {end_date}",
            'position': position,
            'gene': gene
        }
    except Exception as e:
        return None


def run_oos_test(gene, name="OOS_Test"):
    """
    🔬 執行樣本外測試（同時回測訓練期和測試期）

    Returns:
        dict: 包含 train_result, test_result, overfit_warning
    """
    # 訓練期回測
    train_result = run_backtest_period(gene, TRAIN_START, TRAIN_END, f"{name}_Train")

    # 測試期回測
    test_result = run_backtest_period(gene, TEST_START, TEST_END, f"{name}_Test")

    if train_result is None or test_result is None:
        return None

    # 計算過擬合指標
    sharpe_ratio = test_result['sharpe'] / train_result['sharpe'] if train_result['sharpe'] > 0 else 0
    return_ratio = test_result['annual_return'] / train_result['annual_return'] if train_result['annual_return'] > 0 else 0

    # 過擬合警告
    overfit_warnings = []
    if sharpe_ratio < OVERFIT_SHARPE_RATIO and train_result['sharpe'] > 1.0:
        overfit_warnings.append(f"⚠️ 夏普衰減嚴重: {sharpe_ratio:.1%} (測試/訓練)")
    if return_ratio < OVERFIT_RETURN_RATIO and train_result['annual_return'] > 0.2:
        overfit_warnings.append(f"⚠️ 報酬衰減嚴重: {return_ratio:.1%} (測試/訓練)")

    return {
        'train': train_result,
        'test': test_result,
        'sharpe_ratio': sharpe_ratio,
        'return_ratio': return_ratio,
        'overfit_warnings': overfit_warnings,
        'is_overfit': len(overfit_warnings) > 0
    }


def run_detailed_oos_test(gene, gen):
    """
    🔬 執行詳細樣本外測試（顯示完整回測報告）

    每 FULL_BACKTEST_INTERVAL 代執行一次，顯示測試期的詳細回測結果
    """
    try:
        # === 訓練期回測（簡要） ===
        train_position, params = combined_strategy(gene, start_date=TRAIN_START, end_date=TRAIN_END)
        if train_position.empty or train_position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        train_position = position_weight_mgr.normalize_position(train_position)

        # 🔥 修正：不使用 resample，讓 position 的月營收稀疏 index 自然換股
        train_report = sim(
            position=train_position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
            upload=False,
            name=f"快快龍_W{WINDOW_ID}_Gen{gen}_訓練期"
        )

        if train_report is None:
            return None
        train_metrics = train_report.get_metrics()
        train_sharpe = train_metrics['ratio'].get('sharpeRatio', 0) or 0

        # === 測試期回測（詳細顯示） ===
        test_position, params = combined_strategy(gene, start_date=TEST_START, end_date=TEST_END)
        if test_position.empty or test_position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        test_position = position_weight_mgr.normalize_position(test_position)

        print(f"\n{'='*60}")
        print(f"📊 第 {gen} 代 - 測試期詳細回測 ({TEST_START[:4]}~{TEST_END[:4]})")
        print(f"{'='*60}")

        # 🔥 修正：不使用 resample，讓 position 的月營收稀疏 index 自然換股
        test_report = sim(
            position=test_position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
            upload=False,
            name=f"快快龍_W{WINDOW_ID}_Gen{gen}_測試期"
        )

        if test_report is None:
            return None

        # 顯示詳細報告
        test_report.display()

        test_metrics = test_report.get_metrics()
        test_sharpe = test_metrics['ratio'].get('sharpeRatio', 0) or 0

        # 計算過擬合指標
        sharpe_ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0
        is_overfit = sharpe_ratio < OVERFIT_SHARPE_RATIO and train_sharpe > 1.0

        print(f"\n📈 訓練期夏普: {train_sharpe:.3f}")
        print(f"📈 測試期夏普: {test_sharpe:.3f}")
        print(f"📊 測試/訓練比: {sharpe_ratio:.1%} {'🔴 過擬合警告' if is_overfit else '🟢 通過'}")

        return {
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'sharpe_ratio': sharpe_ratio,
            'is_overfit': is_overfit
        }

    except Exception as e:
        print(f"   ⚠️ 詳細OOS測試失敗: {e}")
        return None


def print_oos_comparison(oos_result):
    """打印樣本外測試比較結果"""
    if oos_result is None:
        print("   ⚠️ 樣本外測試失敗")
        return

    train = oos_result['train']
    test = oos_result['test']

    print(f"\n   {'─'*50}")
    print(f"   📊 樣本外測試結果比較")
    print(f"   {'─'*50}")
    print(f"   {'指標':<12} {'訓練期':>12} {'測試期':>12} {'比率':>10}")
    print(f"   {'':<12} {'('+TRAIN_START[:4]+'~'+TRAIN_END[:4]+')':>12} {'('+TEST_START[:4]+'~'+TEST_END[:4]+')':>12}")
    print(f"   {'─'*50}")
    print(f"   {'夏普值':<10} {train['sharpe']:>12.3f} {test['sharpe']:>12.3f} {oos_result['sharpe_ratio']:>9.1%}")
    print(f"   {'年化報酬':<10} {train['annual_return']*100:>11.1f}% {test['annual_return']*100:>11.1f}% {oos_result['return_ratio']:>9.1%}")
    print(f"   {'最大回檔':<10} {train['max_drawdown']*100:>11.1f}% {test['max_drawdown']*100:>11.1f}%")
    print(f"   {'胃納量(萬)':<10} {train['capacity']/1e4:>12.0f} {test['capacity']/1e4:>12.0f}")
    print(f"   {'─'*50}")

    if oos_result['is_overfit']:
        print(f"   🔴 過擬合警告:")
        for warning in oos_result['overfit_warnings']:
            print(f"      {warning}")
    else:
        print(f"   🟢 通過樣本外測試！測試期表現穩定")


def evaluate_fitness(gene):
    """
    適應度評估（僅使用訓練期數據，防止過擬合）
    """
    try:
        # 🔬 只使用訓練期數據進行評估
        position, params = combined_strategy(gene, start_date=TRAIN_START, end_date=TRAIN_END)
        if position.empty:
            return (0.0,)

        # 🔥 應用 3% 持股約束
        position = position_weight_mgr.normalize_position(position)

        # 🔥 修正：不使用 resample，讓 position 的月營收稀疏 index 自然換股
        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
            upload=False,
            name="Eval"
        )

        if report is None:
            return (0.0,)

        metrics = report.get_metrics()
        sharpe = metrics['ratio'].get('sharpeRatio', 0) or 0
        capacity = metrics['liquidity'].get('capacity', 0) or 0
        max_dd = abs(metrics['risk'].get('maxDrawdown', 1))
        annual_return = metrics['profitability'].get('annualReturn', 0) or 0

        sharpe_score = min(1.0, sharpe / TARGET_SHARPE) * 4.0
        capacity_score = min(1.0, capacity / MIN_CAPACITY) * 2.5
        drawdown_score = 2.0 if max_dd <= MAX_DRAWDOWN else max(0, 2.0 - (max_dd - MAX_DRAWDOWN) * 10)
        return_score = min(1.0, annual_return / TARGET_ANNUAL_RETURN) * 1.5

        fitness = sharpe_score + capacity_score + drawdown_score + return_score
        return (fitness,)

    except Exception as e:
        return (0.0,)

# =============================================================================
# DEAP 設定
# =============================================================================
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

GENE_LENGTH = 160
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

print("✅ DEAP GA 配置完成")

# =============================================================================
# 歷史精英載入
# =============================================================================
def load_historical_elites(top_n=20):
    print("\n🔍 搜尋歷史精英...")

    BASE_PATH = '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014'
    SEARCH_PATHS = [
        f'{BASE_PATH}/window_1',
        f'{BASE_PATH}/window_2',
        f'{BASE_PATH}/window_3',
        f'{BASE_PATH}/shared_best',
        '/content/drive/MyDrive/投資策略優化_六策略純夏普值適應度_2014',
        '/content/drive/MyDrive/投資策略優化_六策略純夏普值適應度_2014_分散式',
        '/content/drive/MyDrive/十二組合_Calmar_Sortino_優化',
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_1/working',
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_2/working',
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_3/working',
        # 新版本資料夾
        f'{BASE_DIR}/window_1',
        f'{BASE_DIR}/window_2',
        f'{BASE_DIR}/window_3',
        SHARED_DIR,
    ]

    SEARCH_PATHS = [path for path in SEARCH_PATHS if os.path.exists(path)]
    print(f"   搜尋範圍: {len(SEARCH_PATHS)} 個資料夾")

    all_individuals = []
    total_files_found = 0

    for search_idx, search_path in enumerate(SEARCH_PATHS, 1):
        checkpoint_pattern = os.path.join(search_path, "**", "*.pkl")
        matching_files = glob.glob(checkpoint_pattern, recursive=True)
        total_files_found += len(matching_files)

        files_loaded = 0
        for file in matching_files:
            try:
                with open(file, 'rb') as f:
                    cp = pickle.load(f)

                if "halloffame" in cp and cp["halloffame"]:
                    for ind in cp["halloffame"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            fitness = ind.fitness.values[0]
                            all_individuals.append({
                                'gene': list(ind),
                                'fitness': fitness,
                                'source': file
                            })
                            files_loaded += 1

                if "population" in cp and cp["population"]:
                    for ind in cp["population"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            fitness = ind.fitness.values[0]
                            if fitness > 2.0:
                                all_individuals.append({
                                    'gene': list(ind),
                                    'fitness': fitness,
                                    'source': file
                                })
                                files_loaded += 1

                # 支援新格式
                if "elites" in cp:
                    for elite in cp["elites"]:
                        all_individuals.append({
                            'gene': elite['gene'],
                            'fitness': elite['fitness'],
                            'source': file
                        })
                        files_loaded += 1
            except:
                continue

    print(f"   找到 {total_files_found} 個檔案, {len(all_individuals)} 個有效個體")

    seen_hashes = set()
    unique_individuals = []

    for ind in sorted(all_individuals, key=lambda x: x['fitness'], reverse=True):
        gene_hash = hashlib.md5(str(ind['gene'][:30]).encode()).hexdigest()
        if gene_hash not in seen_hashes:
            seen_hashes.add(gene_hash)
            unique_individuals.append(ind)

    elites = unique_individuals[:top_n]

    if elites:
        print(f"✅ 找到 {len(elites)} 個歷史精英，最佳適應度: {elites[0]['fitness']:.4f}")
    else:
        print("⚠️ 未找到歷史精英，從頭開始")

    return elites

# =============================================================================
# 🔧 基因清理函數（防止複數/NaN值導致錯誤）
# =============================================================================
def sanitize_gene(gene):
    """
    清理基因值，確保都是有效的浮點數 [0, 1]
    修復 TypeError: '>' not supported between 'float' and 'complex'
    """
    sanitized = []
    for val in gene:
        try:
            # 處理複數
            if isinstance(val, complex):
                val = abs(val)  # 取絕對值
            # 轉換為浮點數
            val = float(val)
            # 處理 NaN 和 Inf
            if np.isnan(val) or np.isinf(val):
                val = random.random()
            # 限制在 [0, 1] 範圍
            val = max(0.0, min(1.0, val))
        except:
            val = random.random()
        sanitized.append(val)
    return sanitized

# =============================================================================
# 進度追蹤器
# =============================================================================
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.start_time = None
        self.pbar = None

    def start(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total, desc="🧬 演化", unit="代", ncols=100)

    def update(self, gen, stats):
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (gen + 1)
        eta = avg_time * (self.total - gen - 1)

        if eta < 60:
            eta_str = f"{eta:.0f}秒"
        elif eta < 3600:
            eta_str = f"{eta/60:.1f}分"
        else:
            eta_str = f"{eta/3600:.1f}時"

        self.pbar.set_postfix({
            '最佳': f"{stats.get('best_fitness', 0):.3f}",
            'ETA': eta_str
        })
        self.pbar.update(1)
        return eta_str

    def close(self):
        if self.pbar:
            self.pbar.close()
        return time.time() - self.start_time

# =============================================================================
# 🔥 演化引擎 (含3視窗交互)
# =============================================================================
class EvolutionEngine:
    def __init__(self):
        self.best_ever = None
        self.best_metrics = None
        self.history = []

    def run(self, n_generations=N_GENERATIONS, inject_elites=True):
        print(f"\n{'='*80}")
        print(f"🚀 開始演化 - 共 {n_generations} 代, 族群 {POPULATION_SIZE}")
        print(f"   🔄 3視窗交互: 每 {CROSS_WINDOW_INTERVAL} 代交換 Top {CROSS_WINDOW_TOP_N}")
        print(f"{'='*80}\n")

        progress = ProgressTracker(n_generations)
        progress.start()

        # 顯示 checkpoint 來源
        if CHECKPOINT_PATH:
            notifier.send(f"🚀 **Window {WINDOW_ID} 從指定 Checkpoint 開始演化**\n• 📂 {CHECKPOINT_PATH}\n• {n_generations}代, 族群{POPULATION_SIZE}\n• 將使用新適應度函數重新評估")
        else:
            notifier.send(f"🚀 **Window {WINDOW_ID} 開始演化**\n• {n_generations}代, 族群{POPULATION_SIZE}\n• 每{CROSS_WINDOW_INTERVAL}代交換基因")

        # 🔥 嘗試從 checkpoint 恢復
        start_gen = 0
        checkpoint_data = None

        # 🔥 優先從指定的 CHECKPOINT_PATH 載入
        if CHECKPOINT_PATH:
            print(f"🔥 偵測到 CHECKPOINT_PATH 環境變數: {CHECKPOINT_PATH}")
            checkpoint_data = checkpoint_mgr.load_from_file(CHECKPOINT_PATH, force_reevaluate=True)
            if checkpoint_data:
                print(f"✅ 成功從指定 checkpoint 載入，將使用新適應度函數重新評估所有個體")
            else:
                print(f"⚠️ 無法從指定 checkpoint 載入，嘗試預設 checkpoint...")
                checkpoint_data = checkpoint_mgr.load()
        else:
            checkpoint_data = checkpoint_mgr.load()

        if checkpoint_data:
            # 從 checkpoint 恢復族群
            population = []
            for ind_data in checkpoint_data['population']:
                clean_gene = sanitize_gene(ind_data['genes'])
                new_ind = creator.Individual(clean_gene)
                if ind_data['fitness'] is not None:
                    new_ind.fitness.values = (ind_data['fitness'],)
                population.append(new_ind)

            start_gen = checkpoint_data['generation']
            self.history = checkpoint_data.get('history', [])
            print(f"🔄 從第 {start_gen} 代繼續演化...")
            print(f"   剩餘世代: {n_generations - start_gen}")

            # 重新評估沒有 fitness 的個體
            invalid = [ind for ind in population if not ind.fitness.valid]
            if invalid:
                print(f"📊 重新評估 {len(invalid)} 個個體...")
                for ind in tqdm(invalid, desc="重新評估", ncols=80):
                    ind.fitness.values = toolbox.evaluate(ind)
        else:
            # 初始化族群
            population = toolbox.population(n=POPULATION_SIZE)

            # 注入歷史精英
            if inject_elites:
                historical_elites = load_historical_elites(top_n=30)

                if historical_elites:
                    # 🔥 注入 80% 歷史精英（提高比例，加速收斂）
                    n_inject = min(len(historical_elites), int(POPULATION_SIZE * 0.8))
                    print(f"💉 注入 {n_inject} 個歷史精英到初始族群（80%）")
                    print(f"   🔄 將使用新功能（3%選股、OOS測試）重新評估所有個體")

                    for i, elite_data in enumerate(historical_elites[:n_inject]):
                        # 🔧 清理基因值（防止複數/NaN導致錯誤）
                        clean_gene = sanitize_gene(elite_data['gene'])
                        new_ind = creator.Individual(clean_gene)
                        # 🔥 不設定 fitness，強制重新評估！
                        # new_ind.fitness.values = (elite_data['fitness'],)  # 移除這行
                        population[i] = new_ind

                    print(f"   📊 歷史最佳（舊評估）: {historical_elites[0]['fitness']:.4f}")
                    print(f"   ⚠️ 注意：新評估結果可能不同（因為新功能約束）")

                    if historical_elites[0]['fitness'] > 0:
                        self.best_ever = sanitize_gene(historical_elites[0]['gene'])

            # 🔥 強制重新評估所有個體（使用新功能：3%選股、OOS測試等）
            print("📊 使用新功能重新評估所有個體...")
            for i, ind in enumerate(tqdm(population, desc="新功能評估", ncols=80)):
                ind.fitness.values = toolbox.evaluate(ind)  # 強制評估所有個體

            # 顯示重新評估後的最佳結果
            best_after_eval = tools.selBest(population, 1)[0]
            print(f"   ✅ 重新評估後最佳適應度: {best_after_eval.fitness.values[0]:.4f}")

        # 演化（從 start_gen 繼續）
        for gen in range(start_gen, n_generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            n_elite = max(2, int(POPULATION_SIZE * ELITE_RATIO))
            elites = tools.selBest(population, n_elite)
            population = elites + tools.selBest(offspring, POPULATION_SIZE - n_elite)

            fitnesses = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]

            stats = {'best_fitness': best_fitness, 'avg_fitness': np.mean(fitnesses)}
            self.history.append(stats)

            eta_str = progress.update(gen, stats)

            # 🔥 3視窗交互 - 每 CROSS_WINDOW_INTERVAL 代執行
            if (gen + 1) % CROSS_WINDOW_INTERVAL == 0:
                population = cross_window_mgr.perform_cross_window_exchange(population, gen + 1)

                # 重新評估新注入的個體
                for ind in population:
                    if not ind.fitness.valid:
                        ind.fitness.values = toolbox.evaluate(ind)

            # 🔥 每 5 代保存 checkpoint
            if (gen + 1) % 5 == 0:
                halloffame = tools.selBest(population, 10)
                checkpoint_mgr.save(
                    population=population,
                    generation=gen + 1,
                    halloffame=halloffame,
                    history=self.history
                )

            # 每 5 代完整回測 + 詳細樣本外測試
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0:
                print(f"\n📊 第 {gen+1} 代 - 詳細樣本外測試...")

                # 🔬 執行詳細樣本外測試（顯示完整回測報告）
                oos_result = run_detailed_oos_test(best_ind, gen + 1)

                if oos_result:
                    train_sharpe = oos_result['train_sharpe']
                    test_sharpe = oos_result['test_sharpe']

                    is_best = self.best_metrics is None or train_sharpe > self.best_metrics.get('sharpe', 0)

                    if is_best:
                        self.best_ever = list(best_ind)
                        self.best_metrics = {'sharpe': train_sharpe}
                        self.best_metrics['test_sharpe'] = test_sharpe
                        self.best_metrics['sharpe_ratio'] = oos_result['sharpe_ratio']

                        # Discord 通知新最佳
                        overfit_flag = "🔴" if oos_result['is_overfit'] else "🟢"
                        print(f"\n   🏆 新最佳!")

                        notifier.send_embed(
                            f"🏆 快快龍 W{WINDOW_ID} 第{gen+1}代 新最佳",
                            f"訓練期夏普: {train_sharpe:.3f}\n"
                            f"測試期夏普: {test_sharpe:.3f}\n"
                            f"比率: {oos_result['sharpe_ratio']:.1%} {overfit_flag}",
                            'success' if not oos_result['is_overfit'] else 'warning'
                        )
                    else:
                        # 非新最佳也發送通知
                        overfit_flag = "🔴" if oos_result['is_overfit'] else "🟢"
                        notifier.send(
                            f"✅ 快快龍 W{WINDOW_ID} 第{gen+1}代 OOS測試\n"
                            f"訓練期夏普: {train_sharpe:.3f}\n"
                            f"測試期夏普: {test_sharpe:.3f}\n"
                            f"比率: {oos_result['sharpe_ratio']:.1%} {overfit_flag}"
                        )

            # 每 20 代發送進度 + 顯示各視窗狀態
            if (gen + 1) % 20 == 0:
                all_best = cross_window_mgr.get_all_windows_best()
                status_msg = f"📈 Window {WINDOW_ID} 進度: {gen+1}/{n_generations}代\n最佳: {best_fitness:.3f}\n"
                for w, info in all_best.items():
                    status_msg += f"  W{w}: {info['best_fitness']:.3f} (G{info['generation']})\n"
                notifier.send(status_msg)

        total_time = progress.close()

        # 最終回測 + 樣本外測試
        print(f"\n{'='*80}")
        print("📊 最終回測 + 樣本外測試...")

        final_best = tools.selBest(population, 1)[0]

        # 🔬 執行最終樣本外測試
        final_oos = run_oos_test(final_best, name=f"快快龍_視窗{WINDOW_ID}_Final")

        # 同時上傳完整期間的回測到 FinLab（覆蓋舊版本）
        final_result = run_backtest(final_best, upload=True, name=f"快快龍_視窗{WINDOW_ID}_最佳策略")
        print(f"\n✅ 最佳策略已上傳到 FinLab: 快快龍_視窗{WINDOW_ID}_最佳策略")

        if final_oos:
            print(f"\n{'='*60}")
            print(f"🏆 最終結果 - 樣本外測試")
            print(f"{'='*60}")

            # 打印樣本外比較表
            print_oos_comparison(final_oos)

            print(f"\n📈 完整期間回測:")
            if final_result:
                print(f"   夏普值: {final_result['sharpe']:.4f}")
                print(f"   年化報酬: {final_result['annual_return']*100:.1f}%")
                print(f"   最大回檔: {final_result['max_drawdown']*100:.1f}%")
                print(f"   胃納量: {final_result['capacity']/1e4:.1f}萬")

            print(f"\n🔍 持股配比:")
            if final_result:
                last_pos = final_result['position'].iloc[-1]
                non_zero = last_pos[last_pos > 0].sort_values(ascending=False)
                print(f"   持股數: {len(non_zero)}")
                for stock, weight in non_zero.head(10).items():
                    print(f"   {stock}: {weight*100:.1f}%")

            # Discord 通知包含過擬合資訊
            overfit_status = "🔴 過擬合警告" if final_oos['is_overfit'] else "🟢 測試通過"
            notifier.send_embed(
                f"🎉 Window {WINDOW_ID} 演化完成",
                f"總耗時: {total_time/60:.1f}分鐘\n"
                f"{'─'*20}\n"
                f"📊 訓練期({TRAIN_START[:4]}~{TRAIN_END[:4]}):\n"
                f"  夏普: {final_oos['train']['sharpe']:.3f}\n"
                f"  年化: {final_oos['train']['annual_return']*100:.1f}%\n"
                f"{'─'*20}\n"
                f"🔬 測試期({TEST_START[:4]}~{TEST_END[:4]}):\n"
                f"  夏普: {final_oos['test']['sharpe']:.3f}\n"
                f"  年化: {final_oos['test']['annual_return']*100:.1f}%\n"
                f"{'─'*20}\n"
                f"測試/訓練比: {final_oos['sharpe_ratio']:.1%}\n"
                f"{overfit_status}",
                'success' if not final_oos['is_overfit'] else 'warning'
            )
        elif final_result:
            # 如果樣本外測試失敗，至少顯示完整期間結果
            print(f"\n🏆 最終結果 (完整期間):")
            print(f"   夏普值: {final_result['sharpe']:.4f}")
            print(f"   年化報酬: {final_result['annual_return']*100:.1f}%")
            print(f"   最大回檔: {final_result['max_drawdown']*100:.1f}%")
            print(f"   胃納量: {final_result['capacity']/1e4:.1f}萬")

            notifier.send_embed(
                f"🎉 Window {WINDOW_ID} 演化完成",
                f"總耗時: {total_time/60:.1f}分鐘\n"
                f"夏普: {final_result['sharpe']:.4f}\n"
                f"年化: {final_result['annual_return']*100:.1f}%\n"
                f"胃納: {final_result['capacity']/1e4:.1f}萬",
                'success'
            )

        # 保存（包括共享區）
        self._save_results(population, final_result)

        return population, final_result

    def _save_results(self, population, result):
        try:
            print("\n🔒 使用安全寫入保存結果...")

            # 🔒 安全保存最佳基因
            best_gene_data = {
                'gene': self.best_ever or list(tools.selBest(population, 1)[0]),
                'metrics': self.best_metrics or result,
                'timestamp': datetime.now().isoformat()
            }
            best_gene_path = f"{WINDOW_DIR}/best_gene.pkl"
            if SafeFileManager.safe_pickle_save(best_gene_data, best_gene_path):
                print(f"   ✅ best_gene.pkl 安全保存成功")
            else:
                print(f"   ⚠️ best_gene.pkl 安全保存失敗，使用傳統方式")
                with open(best_gene_path, 'wb') as f:
                    pickle.dump(best_gene_data, f)

            # 🔒 安全保存 checkpoint
            checkpoint_data = {
                'halloffame': tools.selBest(population, 10),
                'population': population,
                'generation': N_GENERATIONS,
                'timestamp': datetime.now().isoformat()
            }
            checkpoint_path = f"{WINDOW_DIR}/checkpoint_latest.pkl"
            if SafeFileManager.safe_pickle_save(checkpoint_data, checkpoint_path):
                print(f"   ✅ checkpoint_latest.pkl 安全保存成功")
            else:
                print(f"   ⚠️ checkpoint_latest.pkl 安全保存失敗，使用傳統方式")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

            # 🔒 安全保存歷史記錄
            history_path = f"{WINDOW_DIR}/history.json"
            if SafeFileManager.safe_json_save(self.history, history_path):
                print(f"   ✅ history.json 安全保存成功")
            else:
                print(f"   ⚠️ history.json 安全保存失敗，使用傳統方式")
                with open(history_path, 'w') as f:
                    json.dump(self.history, f)

            # 🔥 最終結果也保存到共享區 (已使用安全寫入)
            cross_window_mgr.save_elites_to_shared(tools.selBest(population, CROSS_WINDOW_TOP_N), N_GENERATIONS)

            print(f"\n💾 已安全保存至: {WINDOW_DIR}")
            print(f"💾 精英已同步至: {SHARED_DIR}")

        except Exception as e:
            print(f"⚠️ 保存失敗: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# 主程式
# =============================================================================
def main():
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║      六組合快快龍 基因演算法 v12.2 (樣本外測試版)                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e4:.0f}萬+, 回檔 {MAX_DRAWDOWN*100:.0f}%以內           ║
║  📊 訓練期：{TRAIN_START} ~ {TRAIN_END}  (用於演化優化)                  ║
║  🔬 測試期：{TEST_START} ~ {TEST_END}  (用於過擬合檢測)                  ║
║  🔄 3視窗交互：每 {CROSS_WINDOW_INTERVAL} 代交換 Top {CROSS_WINDOW_TOP_N} 精英                              ║
║  📁 共享區：{SHARED_DIR[-40:]:40s} ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    engine = EvolutionEngine()
    population, result = engine.run(N_GENERATIONS)

    print(f"\n✅ 完成！結果保存於: {WINDOW_DIR}")
    return population, result

if __name__ == "__main__":
    population, result = main()
