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

版本：v12.3 Full-Featured (2025-12-28)
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

TARGET_SHARPE = 4.2
MIN_CAPACITY = 7_000_000
MAX_DRAWDOWN = 0.17
TARGET_ANNUAL_RETURN = 0.6

POPULATION_SIZE = 80
N_GENERATIONS = 200
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.8
ELITE_RATIO = 0.1

# 🔥 3視窗交互設定
CROSS_WINDOW_INTERVAL = 10  # 每10代交換一次
CROSS_WINDOW_TOP_N = 5      # 每次交換最佳5個基因
ALL_WINDOW_IDS = [1, 2, 3]  # 所有視窗ID

MIN_POSITION_WEIGHT = 0.03
POSITION_WEIGHT_STEP = 0.03

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
print(f"🚀 六組合快快龍 v12.3 (完整版) - Window {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"   🔄 視窗交互：每 {CROSS_WINDOW_INTERVAL} 代交換 Top {CROSS_WINDOW_TOP_N} 基因")
print(f"   📊 訓練期：{TRAIN_START} ~ {TRAIN_END}")
print(f"   🔬 測試期：{TEST_START} ~ {TEST_END}")
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
# 數據載入
# =============================================================================
print("\n📊 載入市場數據...")
data_start = time.time()

close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')

pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')

rev = data.get('monthly_revenue:當月營收')
rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
rev_mom = data.get('monthly_revenue:上月比較增減(%)')

市值 = data.get('etl:market_value')
成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=14)

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
# gene_to_params
# =============================================================================
def gene_to_params(gene):
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    if len(gene) < required_length:
        gene = gene + [0.5] * (required_length - len(gene))
    gene = gene[:required_length]

    alloc_sum = sum(gene[0:6])
    allocation = [gene[i]/alloc_sum for i in range(6)] if alloc_sum > 0 else [1/6]*6

    low_vol_pe_params = {
        'pe_min': gene[12] * 20,
        'pe_max': gene[13] * 30 + 10,
        'pb_min': gene[135] * 2,
        'pb_max': gene[136] * 5 + 1,
        'min_volume': abs(gene[11] * 500) + 100,
        'top_n': max(3, int(gene[14] * 10) + 5),
        'ma_short': max(5, int(gene[130] * 20)),
        'ma_long': max(20, int(gene[132] * 100) + 60),
    }

    small_inv_params = {
        'market_value_limit': gene[15] * 50e9 + 10e9,
        'min_volume': abs(gene[21] * 500) + 100,
        'top_n': max(3, int(gene[22] * 10) + 5),
        'ma_period': max(5, int(gene[20] * 30) + 10),
    }

    turbo_params = {
        'min_volume': abs(gene[26] * 500) + 100,
        'min_price': gene[27] * 50 + 10,
        'top_n': max(3, int(gene[30] * 10) + 5),
        'momentum_period': max(5, int(gene[23] * 20) + 5),
    }

    high_yield_params = {
        'min_yield': gene[31] * 5 + 2,
        'min_volume': gene[34] * 500 + 100,
        'max_volume': gene[35] * 5000 + 2000,
        'top_n': max(3, int(gene[36] * 10) + 5),
    }

    low_vol_params = {
        'min_volume': gene[37] * 500 + 100,
        'std_window': max(10, int(gene[38] * 50) + 20),
        'std_threshold': gene[39] * 0.5 + 0.1,
        'top_n': max(3, int(gene[40] * 10) + 5),
    }

    market_params = {
        'new_high_window': max(20, int(gene[43] * 200) + 60),
        'min_volume': gene[48] * 500 + 100,
        'top_n': max(3, int(gene[49] * 10) + 5),
    }

    overall_params = {
        'stop_loss': gene[51] * 0.15 + 0.05,
        'trail_stop': gene[52] * 0.1 + 0.03,
        'take_profit': gene[53] * 0.3 + 0.15,
        'position_limit': gene[54] * 0.2 + 0.2,
        'trade_at_price': 'close',
        'liquidity_threshold': gene[56] * 5e6 + 1e6,
    }

    return allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_params, low_vol_params, market_params, overall_params

# =============================================================================
# 策略函數
# =============================================================================
def strategy_low_vol_pe(params):
    try:
        pe_cond = (pe >= params['pe_min']) & (pe <= params['pe_max'])
        pb_cond = (pb >= params['pb_min']) & (pb <= params['pb_max'])
        vol_cond = vol.average(5) > params['min_volume']
        ma_cond = close > close.average(params['ma_long'])

        score = safe_condition(pe_cond & pb_cond & vol_cond & ma_cond)
        score = score / (pe.fillna(999) + 1)

        position = score[score > 0].is_smallest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_low_vol_pe 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_small_cap(params):
    try:
        cap_cond = 市值 < params['market_value_limit']
        vol_cond = vol.average(5) > params['min_volume']
        growth_cond = rev_yoy > 0
        ma_cond = close > close.average(params['ma_period'])

        score = safe_condition(cap_cond & vol_cond & growth_cond & ma_cond)
        score = score * rev_yoy.fillna(0).clip(lower=0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_small_cap 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_turbo(params):
    try:
        vol_cond = vol.average(5) > params['min_volume']
        price_cond = close > params['min_price']

        momentum = close / close.shift(params['momentum_period']) - 1
        momentum_cond = momentum > 0

        rev_ma = rev.average(3)
        rev_new_high = rev_ma >= rev_ma.rolling(12, min_periods=1).max()

        score = safe_condition(vol_cond & price_cond & momentum_cond & rev_new_high)
        score = score * momentum.fillna(0).clip(lower=0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_turbo 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_high_yield(params):
    try:
        yield_cond = dividend_yield >= params['min_yield']
        vol_cond = (vol.average(5) >= params['min_volume']) & (vol.average(5) <= params['max_volume'])
        trend_cond = (close > close.average(20)) & (close > close.average(60))
        rev_cond = rev.average(3) > rev.average(12)

        score = safe_condition(yield_cond & vol_cond & trend_cond & rev_cond)
        score = score * dividend_yield.fillna(0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_high_yield 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_low_vol(params):
    try:
        vol_cond = vol.average(20) > params['min_volume']
        trend_cond = (close > close.average(60)) & (close > close.average(120))

        returns = close.pct_change()
        volatility = returns.rolling(params['std_window']).std()
        vol_rank = volatility.rank(axis=1, pct=True)
        low_vol_cond = vol_rank < params['std_threshold']

        score = safe_condition(vol_cond & trend_cond & low_vol_cond)
        score = score * 市值.fillna(0)

        position = score[score > 0].is_smallest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_low_vol 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_market(params):
    try:
        new_high = close >= close.rolling(params['new_high_window'], min_periods=1).max()
        vol_cond = vol.average(10) > params['min_volume']

        score = safe_condition(new_high & vol_cond)
        score = score * vol.average(10).fillna(0)

        position = score[score > 0].is_smallest(params['top_n'])
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

        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
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

        notifier.send(f"🚀 **Window {WINDOW_ID} 開始演化**\n• {n_generations}代, 族群{POPULATION_SIZE}\n• 每{CROSS_WINDOW_INTERVAL}代交換基因")

        # 🔥 嘗試從 checkpoint 恢復
        start_gen = 0
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
                    n_inject = min(len(historical_elites), POPULATION_SIZE // 2)
                    print(f"💉 注入 {n_inject} 個歷史精英到初始族群")

                    for i, elite_data in enumerate(historical_elites[:n_inject]):
                        # 🔧 清理基因值（防止複數/NaN導致錯誤）
                        clean_gene = sanitize_gene(elite_data['gene'])
                        new_ind = creator.Individual(clean_gene)
                        new_ind.fitness.values = (elite_data['fitness'],)
                        population[i] = new_ind

                    if historical_elites[0]['fitness'] > 0:
                        self.best_ever = sanitize_gene(historical_elites[0]['gene'])
                        print(f"   🏆 最佳歷史基因適應度: {historical_elites[0]['fitness']:.4f}")

            print("📊 評估初始族群...")
            for i, ind in enumerate(tqdm(population, desc="初始評估", ncols=80)):
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

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

            # 每 5 代完整回測 + 樣本外測試
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0:
                print(f"\n📊 第 {gen+1} 代 - 樣本外測試...")

                # 🔬 執行樣本外測試
                oos_result = run_oos_test(best_ind, name=f"Gen{gen+1}")

                if oos_result:
                    train_sharpe = oos_result['train']['sharpe']
                    test_sharpe = oos_result['test']['sharpe']

                    is_best = self.best_metrics is None or train_sharpe > self.best_metrics.get('sharpe', 0)

                    if is_best:
                        self.best_ever = list(best_ind)
                        self.best_metrics = oos_result['train']
                        self.best_metrics['test_sharpe'] = test_sharpe
                        self.best_metrics['sharpe_ratio'] = oos_result['sharpe_ratio']

                        # 顯示訓練期 vs 測試期
                        overfit_flag = "🔴" if oos_result['is_overfit'] else "🟢"
                        print(f"   🏆 新最佳!")
                        print(f"      訓練期({TRAIN_START[:4]}~{TRAIN_END[:4]}): 夏普 {train_sharpe:.3f}")
                        print(f"      測試期({TEST_START[:4]}~{TEST_END[:4]}): 夏普 {test_sharpe:.3f} {overfit_flag}")
                        print(f"      測試/訓練比: {oos_result['sharpe_ratio']:.1%}")

                        notifier.send_embed(
                            f"🏆 Window {WINDOW_ID} 新最佳",
                            f"訓練期夏普: {train_sharpe:.3f}\n"
                            f"測試期夏普: {test_sharpe:.3f}\n"
                            f"比率: {oos_result['sharpe_ratio']:.1%} {overfit_flag}",
                            'success' if not oos_result['is_overfit'] else 'warning'
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
        final_oos = run_oos_test(final_best, name=f"GA_v12_W{WINDOW_ID}_Final")

        # 同時上傳完整期間的回測
        final_result = run_backtest(final_best, upload=True, name=f"GA_v12_W{WINDOW_ID}_Final")

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
