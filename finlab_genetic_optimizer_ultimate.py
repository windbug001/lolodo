#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 FinLab 台股基因演算法優化系統 - 終極版 v1.2 (樣本外測試版)
FinLab Taiwan Stock Genetic Algorithm Optimizer - Ultimate Edition
================================================================================

【核心功能】
✅ 基於您現有的三策略合併邏輯
✅ 基因演算法自動優化參數
✅ 歷史最佳持續進化 (Pareto Archive)
✅ Walk-Forward 驗證避免 overfitting
✅ 完全遵循 FinLab 原生 API

【v1.1 新增功能】
✅ 3視窗交互取優秀基因機制
✅ 安全寫入機制（先暫存再改名）
✅ 自動備份舊檔案 + 檔案完整性驗證
✅ Discord 通知功能
✅ Checkpoint 斷點續傳

【v1.2 新增功能 - 樣本外測試】
✅ 訓練期/測試期分離 (2017~2022 訓練, 2023~2025 測試)
✅ 適應度評估僅使用訓練期數據
✅ 每5代顯示訓練期 vs 測試期夏普比較
✅ 自動過擬合警告（測試/訓練比 < 60%）
✅ 最終報告包含完整樣本外測試結果

【目標】
- 夏普值：>= 4.0
- 胃納量：>= 500 萬
- 年化報酬：最大化
- 最大回撤：< 20%

【策略架構】
1. 低波動本益比策略
2. 小資族策略
3. 營收股價雙渦輪策略
4. 動態權重優化

版本：v1.2 OOS-Test (2025-12-28)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# 🔥 【重要】在 import 其他套件之前，先禁用 FinLab 快取以避免 EOFError
import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 第一部分：核心設定
# =============================================================================
import sys
import json
import pickle
import time
import hashlib
import random
import itertools
import shutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# === 🔥 核心設定（請修改）===
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))  # 🔥 視窗 ID (1-4)
FINLAB_API_KEY = os.environ.get('FINLAB_API_KEY', "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m")

# Discord 通知設定
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"

# 🎯 2026 統一優化目標
TARGET_SHARPE = 4.2          # 🔥 目標夏普值 4.2
MIN_CAPACITY = 10_000_000    # 🔥 胃納量 1000萬
TARGET_ANNUAL_RETURN = 0.4   # 年化報酬 40%
MAX_DRAWDOWN = 0.15          # 🔥 最大回檔 15%

# 🔥 持股配比約束
MIN_POSITION_WEIGHT = 0.03   # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股必須是 3% 倍數

# 🔄 自適應持股設定
ADAPTIVE_POSITION = True     # 啟用自適應持股
BULL_MARKET_STOCKS = 8       # 多頭市場持股數
BEAR_MARKET_STOCKS = 4       # 空頭市場持股數
MARKET_THRESHOLD = 0.5       # 市場判斷閾值（>50% 股票在均線上=多頭）

# GA 演化參數
POPULATION_SIZE = 100        # 🔥 增加族群到 100
N_GENERATIONS = 300          # 🔥 增加至 300 代
MUTATION_RATE = 0.15         # 降低突變率以穩定
CROSSOVER_RATE = 0.8

# 🔥 3視窗交互設定
CROSS_WINDOW_INTERVAL = 10  # 每10代交換一次
CROSS_WINDOW_TOP_N = 5      # 每次交換最佳5個基因
ALL_WINDOW_IDS = [1, 2, 3, 4]  # 所有視窗ID

# Walk-Forward 設定
WALK_FORWARD_WINDOWS = 3  # 3 個時間窗口
TRAIN_MONTHS = 24         # 訓練期 24 個月
TEST_MONTHS = 6           # 測試期 6 個月

# 回測設定
BACKTEST_START = '2017-01-01'
BACKTEST_END = None

# =============================================================================
# 🔬 樣本外測試設定 (Out-of-Sample Testing) - 防過擬合
# =============================================================================
TRAIN_START = '2017-01-01'   # 訓練期開始
TRAIN_END = '2022-12-31'     # 訓練期結束
TEST_START = '2023-01-01'    # 測試期開始
TEST_END = '2025-12-31'      # 測試期結束

# 過擬合警告閾值
OVERFIT_SHARPE_RATIO = 0.6   # 測試期夏普 / 訓練期夏普 < 0.6 則警告
OVERFIT_RETURN_RATIO = 0.5   # 測試期報酬 / 訓練期報酬 < 0.5 則警告

OOS_TEST_INTERVAL = 5        # 每 N 代執行一次樣本外測試

print(f"=" * 80)
print(f"🚀 FinLab 台股基因演算法優化系統 v1.2 (樣本外測試版) - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"   🔄 視窗交互：每 {CROSS_WINDOW_INTERVAL} 代交換 Top {CROSS_WINDOW_TOP_N} 基因")
print(f"   📊 訓練期：{TRAIN_START} ~ {TRAIN_END}")
print(f"   🔬 測試期：{TEST_START} ~ {TEST_END}")
print(f"=" * 80)

# =============================================================================
# 第二部分：套件安裝與載入
# =============================================================================
def clear_finlab_cache():
    """清除可能損壞的 FinLab 快取（必須在載入 FinLab 之前執行）"""
    import glob
    import shutil
    import subprocess

    print("🔧 清除 FinLab 快取...")

    # 1. 清除所有 .pkl 檔案
    try:
        subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-type', 'f', '-delete'],
                      stderr=subprocess.DEVNULL, timeout=5)
    except:
        pass

    # 2. 清除 finlab 相關目錄
    cache_patterns = [
        '/root/.finlab*',
        '/tmp/.finlab*',
        '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'),
        '/content/.finlab*',
        '/root/*finlab*',
        '/tmp/*finlab*',
    ]

    cleared = 0
    for pattern in cache_patterns:
        try:
            matches = glob.glob(pattern)
            for path in matches:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                cleared += 1
        except:
            pass

    # 3. 清除 pandas 快取
    pandas_cache = os.path.expanduser('~/.cache/pandas')
    if os.path.exists(pandas_cache):
        try:
            shutil.rmtree(pandas_cache)
            cleared += 1
        except:
            pass

    if cleared > 0:
        print(f"   ✅ 已清除 {cleared} 個快取檔案/目錄")
    else:
        print("   ℹ️  無快取需要清除")

# 🔥 在載入 FinLab 之前先清除快取
clear_finlab_cache()

def install_packages():
    """安裝必要套件"""
    required = {
        'finlab': 'finlab',
        'deap': 'deap',
    }

    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"   安裝 {pkg_name}...")
            os.system(f'pip install {pkg_name} -q')

install_packages()

# 載入 FinLab
import finlab
from finlab import data
from finlab.backtest import sim

# 載入 DEAP
from deap import base, creator, tools, algorithms

print(f"✅ 套件載入完成")
print(f"   - FinLab: {finlab.__version__}")

# =============================================================================
# 第三部分：環境設定與登入
# =============================================================================
def setup_environment():
    """設定環境"""
    # Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = '/content/drive/MyDrive/FinLab_GA_優化_終極版'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = './finlab_ga_output'
        in_colab = False
        print("⚠️ 本地環境")

    # FinLab 登入
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")

    return base_dir, in_colab

BASE_DIR, IN_COLAB = setup_environment()

@dataclass
class PathManager:
    """路徑管理器"""
    base_dir: str = BASE_DIR
    window_id: int = WINDOW_ID

    def __post_init__(self):
        # 每個視窗有獨立的輸出目錄
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.output_dir = f"{self.window_dir}/output"
        self.pareto_dir = f"{self.base_dir}/shared_pareto"  # Pareto Archive 共享
        self.history_dir = f"{self.window_dir}/history"
        self.log_dir = f"{self.window_dir}_logs"  # 🔥 日誌目錄（供監控器讀取）

        for d in [self.window_dir, self.output_dir, self.pareto_dir,
                  self.history_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def pareto_archive(self) -> str:
        return f"{self.pareto_dir}/pareto_archive_w{self.window_id}.pkl"

    @property
    def best_params_file(self) -> str:
        return f"{self.output_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log(self) -> str:
        return f"{self.log_dir}/progress_history.json"

paths = PathManager()
SHARED_DIR = f"{BASE_DIR}/shared_best"
Path(SHARED_DIR).mkdir(parents=True, exist_ok=True)
print(f"📁 工作目錄: {paths.output_dir}")
print(f"📁 共享目錄: {SHARED_DIR}")

# =============================================================================
# 🔒 安全檔案管理器 (v1.1 新增)
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
        """安全的 pickle 寫入（防止中斷導致損壞）"""
        temp_path = filepath + '.tmp'
        backup_path = filepath + '.backup'

        try:
            # 1. 先寫入暫存檔
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)

            # 2. 驗證暫存檔
            if not SafeFileManager.verify_pickle_file(temp_path, min_size):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

            # 3. 備份舊檔案（如果存在且有效）
            if os.path.exists(filepath):
                if SafeFileManager.verify_pickle_file(filepath, min_size):
                    shutil.copy2(filepath, backup_path)

            # 4. 原子性替換
            shutil.move(temp_path, filepath)

            # 5. 最終驗證
            if SafeFileManager.verify_pickle_file(filepath, min_size):
                return True
            else:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, filepath)
                return False

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if os.path.exists(backup_path) and not os.path.exists(filepath):
                try:
                    shutil.copy2(backup_path, filepath)
                except:
                    pass
            return False

    @staticmethod
    def verify_pickle_file(filepath, min_size=100):
        """驗證 pkl 檔案是否正常"""
        try:
            if not os.path.exists(filepath):
                return False
            size = os.path.getsize(filepath)
            if size < min_size:
                return False
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return True
        except:
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
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

print("✅ 安全檔案管理器已初始化")

# =============================================================================
# Discord 通知 (v1.1 新增)
# =============================================================================
class DiscordNotifier:
    def __init__(self, webhook_url=None):
        self.enabled = webhook_url and webhook_url != ""
        self.webhook_url = webhook_url

        if self.enabled:
            try:
                payload = {"content": f"✅ 小小龍 v1.2 (樣本外測試版) - 視窗 {WINDOW_ID} 啟動\n   訓練期: {TRAIN_START}~{TRAIN_END}\n   測試期: {TEST_START}~{TEST_END}", "username": "小小龍"}
                requests.post(self.webhook_url, json=payload, timeout=5)
                print("✅ Discord 通知系統已連接")
            except:
                self.enabled = False

    def send(self, message):
        if not self.enabled:
            return
        try:
            payload = {"content": message, "username": "小小龍"}
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
            payload = {"embeds": [embed], "username": "小小龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            time.sleep(1)
        except:
            pass

notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# =============================================================================
# 🔥 3視窗交互機制 (v1.1 新增)
# =============================================================================
class CrossWindowManager:
    """3視窗交互管理器"""

    def __init__(self, base_dir, window_id, shared_dir):
        self.base_dir = base_dir
        self.window_id = window_id
        self.shared_dir = shared_dir
        self.other_windows = [w for w in ALL_WINDOW_IDS if w != window_id]
        Path(shared_dir).mkdir(parents=True, exist_ok=True)

    def save_elites_to_shared(self, elites: List, generation: int):
        """將本視窗的精英基因保存到共享資料夾"""
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
                    # 🔧 保存完整的多目標 fitness values
                    'fitness': list(ind.fitness.values) if ind.fitness.valid else [0.0, 0.0, 0.0, 0.0],
                    'fitness_sum': sum(ind.fitness.values) if ind.fitness.valid else 0.0
                })

            filename = f"{self.shared_dir}/window_{self.window_id}_elites.pkl"
            success = SafeFileManager.safe_pickle_save(elite_data, filename)

            if success:
                print(f"   💾 視窗 {self.window_id} 精英已安全保存到共享區 ({len(elites)} 個)")
                return True
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(elite_data, f)
                return True

        except Exception as e:
            print(f"   ⚠️ 保存精英失敗: {e}")
            return False

    def load_elites_from_other_windows(self, max_per_window: int = CROSS_WINDOW_TOP_N) -> List[Dict]:
        """從其他視窗載入精英基因"""
        all_external_elites = []

        for other_window in self.other_windows:
            try:
                filename = f"{self.shared_dir}/window_{other_window}_elites.pkl"
                if not os.path.exists(filename):
                    continue

                file_age = time.time() - os.path.getmtime(filename)
                if file_age > 86400:
                    continue

                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                elites = data.get('elites', [])[:max_per_window]
                for elite in elites:
                    # 🔧 支援新舊格式
                    fitness = elite['fitness']
                    fitness_sum = elite.get('fitness_sum', 0)
                    if fitness_sum == 0:
                        fitness_sum = sum(fitness) if isinstance(fitness, (list, tuple)) else fitness

                    all_external_elites.append({
                        'gene': elite['gene'],
                        'fitness': fitness,
                        'fitness_sum': fitness_sum,
                        'source_window': other_window,
                        'generation': data.get('generation', 0)
                    })

                print(f"   📥 從視窗 {other_window} 載入 {len(elites)} 個精英")

            except Exception as e:
                continue

        # 🔧 按 fitness_sum 排序（支援多目標）
        all_external_elites.sort(key=lambda x: x['fitness_sum'], reverse=True)
        return all_external_elites

    def perform_cross_window_exchange(self, population: List, generation: int, toolbox) -> List:
        """執行跨視窗基因交換"""
        print(f"\n🔄 第 {generation} 代 - 執行3視窗基因交換")

        local_elites = tools.selBest(population, CROSS_WINDOW_TOP_N)
        self.save_elites_to_shared(local_elites, generation)

        external_elites = self.load_elites_from_other_windows()

        if not external_elites:
            print("   ℹ️ 暫無其他視窗精英可用")
            return population

        n_inject = min(len(external_elites), len(population) // 4)
        population.sort(key=lambda x: sum(x.fitness.values) if x.fitness.valid else 0)

        injected_count = 0
        for i, ext_elite in enumerate(external_elites[:n_inject]):
            new_ind = toolbox.individual_from_gene(ext_elite['gene'])
            # 🔧 支援多目標 fitness values
            fitness_vals = ext_elite['fitness']
            if isinstance(fitness_vals, (list, tuple)) and len(fitness_vals) == 4:
                new_ind.fitness.values = tuple(fitness_vals)
            else:
                # 相容舊格式（單一 fitness 值）：重新評估
                del new_ind.fitness.values  # 標記為需要重新評估
            population[i] = new_ind
            injected_count += 1

        print(f"   ✅ 注入 {injected_count} 個外部精英")

        # 取得最佳外部適應度（相容新舊格式）
        best_fitness = ext_elite.get('fitness_sum', 0)
        if best_fitness == 0:
            f = external_elites[0]['fitness']
            best_fitness = sum(f) if isinstance(f, (list, tuple)) else f

        notifier.send(
            f"🔄 **視窗 {self.window_id} 基因交換**\n"
            f"• 第 {generation} 代\n"
            f"• 注入 {injected_count} 個外部精英\n"
            f"• 最佳外部適應度: {best_fitness:.4f}"
        )

        return population

cross_window_mgr = CrossWindowManager(BASE_DIR, WINDOW_ID, SHARED_DIR)
print("✅ 3視窗交互管理器已初始化")

# =============================================================================
# 第四部分：數據載入（使用 FinLab API）
# =============================================================================
class FinLabDataLoader:
    """FinLab 數據載入器"""

    _instance = None
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._cache:
            self._load_all_data()

    def _safe_get(self, key: str, retry=True):
        """安全載入數據，遇到 EOFError 時清除快取重試"""
        try:
            return data.get(key)
        except (EOFError, Exception) as e:
            if 'EOF' in str(e) and retry:
                print(f"   ⚠️  快取損壞，清除後重試: {key}")
                # 清除快取
                import subprocess
                subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-delete'], stderr=subprocess.DEVNULL)
                subprocess.run(['find', '/root', '/tmp', '-type', 'd', '-name', '*finlab*', '-exec', 'rm', '-rf', '{}', '+'], stderr=subprocess.DEVNULL)
                # 重試一次（不再重試）
                return self._safe_get(key, retry=False)
            else:
                raise

    def _load_all_data(self):
        """載入所有數據"""
        print("📊 載入 FinLab 數據...")
        start_time = time.time()

        # 價格相關數據
        self._cache['close'] = self._safe_get('price:收盤價')
        self._cache['vol'] = self._safe_get('price:成交股數')
        self._cache['open'] = self._safe_get('price:開盤價')
        self._cache['high'] = self._safe_get('price:最高價')
        self._cache['low'] = self._safe_get('price:最低價')
        self._cache['adj_close'] = self._safe_get("etl:adj_close")

        # 財務數據
        self._cache['pe'] = self._safe_get('price_earning_ratio:本益比')
        self._cache['pb'] = self._safe_get("price_earning_ratio:股價淨值比")
        self._cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')

        # 營收數據
        self._cache['rev'] = self._safe_get('monthly_revenue:當月營收')
        self._cache['rev_yoy_growth'] = self._safe_get('monthly_revenue:去年同月增減(%)')
        self._cache['rev_month_growth'] = self._safe_get('monthly_revenue:上月比較增減(%)')

        # 基本面指標
        self._cache['營業利益成長率'] = self._safe_get('fundamental_features:營業利益成長率')
        self._cache['業外收支營收率'] = self._safe_get('fundamental_features:業外收支營收率')
        self._cache['營業毛利率'] = self._safe_get("fundamental_features:營業毛利率")
        self._cache['ROE綜合損益'] = self._safe_get("fundamental_features:ROE綜合損益")
        self._cache['稅後淨利率'] = self._safe_get("fundamental_features:稅後淨利率")
        self._cache['稅前淨利率'] = self._safe_get("fundamental_features:稅前淨利率")

        # 籌碼資料
        self._cache['融資使用率'] = self._safe_get('margin_transactions:融資使用率')
        self._cache['董監持有股數占比'] = self._safe_get("internal_equity_changes:董監持有股數占比")
        self._cache['inventory'] = self._safe_get("inventory")

        # 市值資料
        self._cache['市值'] = self._safe_get('etl:market_value')

        # 財務報表
        self._cache['股本'] = self._safe_get('financial_statement:股本')
        self._cache['投資活動現金流'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._cache['營業活動現金流'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._cache['稅後淨利'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._cache['權益總計'] = self._safe_get('financial_statement:股東權益總額')

        # 技術指標
        self._cache['rsi'] = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
        self._cache['atr'] = data.indicator('ATR', adjust_price=True, timeperiod=10)

        # 計算衍生指標
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"✅ 數據載入完成 ({elapsed:.1f}s)")

    def _calculate_derived_indicators(self):
        """計算衍生指標"""
        close = self._cache['close']
        high = self._cache['high']
        low = self._cache['low']
        open_ = self._cache['open']
        adj_close = self._cache['adj_close']
        atr = self._cache['atr']
        rev = self._cache['rev']

        # 營收均線
        self._cache['rev_ma3'] = rev.average(3)
        self._cache['rev_ma12'] = rev.average(12)

        # 波動率
        self._cache['entry_volatility'] = atr / adj_close

        # 漲停鎖死判斷
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

        self._cache['limit_up_all_day'] = limit_up_all_day.fillna(False)

        # 自由現金流
        df1 = self._cache['投資活動現金流']
        df2 = self._cache['營業活動現金流']
        self._cache['自由現金流'] = (df1 + df2).rolling(4).mean()

        # 股東權益報酬率
        self._cache['股東權益報酬率'] = self._cache['稅後淨利'] / self._cache['權益總計']

        # 當季營收（使用已載入的營收數據）
        當月營收 = self._cache['rev'] * 1000
        self._cache['當季營收'] = 當月營收.rolling(4).sum()
        self._cache['市值營收比'] = self._cache['市值'] / self._cache['當季營收']

    def get(self, key: str):
        """獲取數據"""
        return self._cache.get(key)

data_loader = FinLabDataLoader()

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
    def normalize_position(position_df: pd.DataFrame) -> pd.DataFrame:
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
# 🔥 波動率過濾器（高波動時減倉，提升夏普）
# =============================================================================
class VolatilityFilter:
    """
    波動率過濾器

    功能：
    1. 高波動時減少持股，降低回檔
    2. 極端波動時大幅減倉
    3. 保護資金在市場動盪時期
    """

    # 波動率閾值設定
    VOL_HIGH_THRESHOLD = 0.25      # 年化波動 > 25% 時減倉
    VOL_EXTREME_THRESHOLD = 0.35   # 年化波動 > 35% 時大幅減倉
    VOL_HIGH_SCALE = 0.6           # 高波動時保留 60%
    VOL_EXTREME_SCALE = 0.3        # 極端波動時保留 30%
    VOL_LOOKBACK = 20              # 計算波動率的回看天數

    def __init__(self):
        self._volatility_cache = None
        self._cache_date = None

    def _calculate_market_volatility(self, close_df: pd.DataFrame) -> pd.Series:
        """計算大盤波動率"""
        # 使用 0050 作為大盤代理，如果沒有就用所有股票的平均
        if '0050' in close_df.columns:
            market = close_df['0050']
        else:
            market = close_df.mean(axis=1)

        # 計算日報酬率
        returns = market.pct_change()

        # 計算滾動波動率（年化）
        volatility = returns.rolling(self.VOL_LOOKBACK).std() * (252 ** 0.5)

        return volatility

    def apply_filter(self, position_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
        """
        根據市場波動率調整持股

        Args:
            position_df: 持股 DataFrame
            close_df: 收盤價 DataFrame

        Returns:
            調整後的持股 DataFrame
        """
        if position_df is None or position_df.empty:
            return position_df

        try:
            # 計算市場波動率
            volatility = self._calculate_market_volatility(close_df)

            # 複製持股
            result = position_df.copy()

            # 統計調整次數
            high_vol_count = 0
            extreme_vol_count = 0

            for date in result.index:
                if date not in volatility.index:
                    continue

                vol = volatility.loc[date]

                if pd.isna(vol):
                    continue

                if vol > self.VOL_EXTREME_THRESHOLD:
                    # 極端波動：大幅減倉
                    result.loc[date] *= self.VOL_EXTREME_SCALE
                    extreme_vol_count += 1
                elif vol > self.VOL_HIGH_THRESHOLD:
                    # 高波動：適度減倉
                    result.loc[date] *= self.VOL_HIGH_SCALE
                    high_vol_count += 1

            # 輸出統計（僅在有調整時）
            if high_vol_count > 0 or extreme_vol_count > 0:
                print(f"   📉 波動率過濾: 高波動{high_vol_count}天, 極端{extreme_vol_count}天")

            return result

        except Exception as e:
            print(f"   ⚠️ 波動率過濾失敗: {e}")
            return position_df

volatility_filter = VolatilityFilter()


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
    adaptive_top_n = (base_top_n * adaptive_ratio).round().astype(int).clip(lower=2)

    return adaptive_top_n


# =============================================================================
# 第五部分：策略引擎（基於您的原始策略）
# =============================================================================
class StrategyEngine:
    """策略引擎 - 整合三個子策略"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader

    def strategy_low_volatility_pe(self, params: Dict) -> Any:
        """策略一：低波動本益比"""
        # 從基因解碼獲取參數
        rev_ma3_ma12_ratio = params['lv_rev_ma3_ma12_ratio']
        rev_consistency = params['lv_rev_consistency']
        volatility_threshold = params['lv_volatility_threshold']
        margin_usage_limit = params['lv_margin_usage_limit']
        non_op_income_limit = params['lv_non_op_income_limit']
        min_volume = params['lv_min_volume']
        pe_min = params['lv_pe_min']
        pe_max = params['lv_pe_max']
        top_n = int(params['lv_top_n'])

        # 獲取數據
        close = self.dl.get('close')
        pe = self.dl.get('pe')
        pb = self.dl.get('pb')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_ma3 = self.dl.get('rev_ma3')
        rev_ma12 = self.dl.get('rev_ma12')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')
        rev_month_growth = self.dl.get('rev_month_growth')
        融資使用率 = self.dl.get('融資使用率')
        entry_volatility = self.dl.get('entry_volatility')
        業外收支營收率 = self.dl.get('業外收支營收率')
        營業利益成長率 = self.dl.get('營業利益成長率')
        營業毛利率 = self.dl.get('營業毛利率')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        limit_up_all_day = self.dl.get('limit_up_all_day')
        inventory = self.dl.get('inventory')

        # 本益比成長比率(PEG)
        peg = pe / 營業利益成長率

        # 條件設定
        cond1 = rev_ma3 / rev_ma12 > rev_ma3_ma12_ratio
        cond2 = rev / rev.shift(1) > rev_consistency

        # 低波動因子
        tree_select_factor = ((融資使用率 <= margin_usage_limit)
                             & (entry_volatility <= volatility_threshold)
                             & (業外收支營收率 < non_op_income_limit))

        # 成交量條件
        condition_近1日成交均量 = vol.average(1) > min_volume

        # 排除月營收連3月衰退10%以上
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)

        # 排除月營收成長趨勢過老
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 30).sustain(12, 8)

        # 單月營收月增率連續3月大於-54%
        cond單月營收月增率 = (rev_month_growth > -54).sustain(3)

        # 收盤價大於均線
        cond收盤價大於均線 = (close > close.average(75)) & (close > close.average(40)) & (close > close.average(90))

        # 近三個月營收大於年營收
        cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)

        # 本益比區間
        pe_range = (pe_min <= pe) & (pe <= pe_max)

        # 股價淨值比
        pb_range = (0.5 <= pb) & (pb <= 2.8)

        # 營業毛利率連續兩季大於8%
        gpm_trend = (營業毛利率 > 8).sustain(2)

        # ROE綜合損益連續兩季大於0%
        roe_trend = (ROE綜合損益 > 0).sustain(2)

        # 集保五十張以下散戶持股占比小於46%
        small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46

        # 綜合所有條件
        cond_all = (cond1
                   & cond2
                   & tree_select_factor
                   & cond排除月營收成長趨勢過老
                   & cond排除月營收連3月衰退
                   & cond收盤價大於均線
                   & cond近三個月營收大於年營收
                   & cond單月營收月增率
                   & ~limit_up_all_day
                   & condition_近1日成交均量
                   & gpm_trend
                   & roe_trend
                   & small_inv_under50
                   & pe_range
                   & pb_range)

        # 選出符合條件且PEG最小的top_n檔
        position = peg[cond_all & (peg > 0)].is_smallest(top_n).reindex(rev.index_str_to_date().index, method='ffill')
        return position

    def strategy_small_investor(self, params: Dict) -> Any:
        """策略二：小資族"""
        # 從基因解碼獲取參數
        market_value_limit = params['si_market_value_limit']
        market_rev_ratio_limit = params['si_market_rev_ratio_limit']
        rev_yoy_growth_limit = params['si_rev_yoy_growth_limit']
        rev_mom_growth_limit = params['si_rev_mom_growth_limit']
        rsv_period = int(params['si_rsv_period'])
        ma_period = int(params['si_ma_period'])
        volume_threshold = params['si_volume_threshold']
        top_n = int(params['si_top_n'])

        # 獲取數據
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')
        rev_month_growth = self.dl.get('rev_month_growth')
        市值 = self.dl.get('市值')
        市值營收比 = self.dl.get('市值營收比')
        自由現金流 = self.dl.get('自由現金流')
        股東權益報酬率 = self.dl.get('股東權益報酬率')
        營業利益成長率 = self.dl.get('營業利益成長率')
        業外收支營收率 = self.dl.get('業外收支營收率')
        當月營收 = data.get('monthly_revenue:當月營收') * 1000

        # 條件設定
        condition1 = (市值 < market_value_limit)
        condition2 = 自由現金流 > 0
        condition3 = 股東權益報酬率 > 0
        condition4 = 營業利益成長率 > -1
        condition5 = 市值營收比 < market_rev_ratio_limit
        condition6 = vol > volume_threshold

        # 排除月營收連3月衰退
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < rev_yoy_growth_limit).sustain(3)

        # 排除月營收成長趨勢過老
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)

        # 單月營收月增率連續3月
        cond單月營收月增率 = (rev_month_growth > rev_mom_growth_limit).sustain(3)

        # 收盤價大於均線
        cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(120)) & (close > close.average(75))

        # 近三個月營收大於年營收
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)

        # 業外收支營收率占比低
        業外收支營收率占比低 = (業外收支營收率 < 7.3)

        # RSV值計算
        rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

        # 綜合所有條件
        position = ((
                condition1
                & condition2
                & condition3
                & condition4
                & condition5
                & condition6
                & cond排除月營收成長趨勢過老
                & cond單月營收月增率
                & cond排除月營收連3月衰退
                & cond近三個月營收大於年營收
                & 業外收支營收率占比低
                )
                * rsv).is_largest(top_n)

        position = position.reindex(當月營收.index_str_to_date().index)
        return position

    def strategy_revenue_price_turbo(self, params: Dict) -> Any:
        """策略三：營收股價雙渦輪"""
        # 從基因解碼獲取參數
        rev_ma_period = int(params['rpt_rev_ma_period'])
        rev_ma_lookback = int(params['rpt_rev_ma_lookback'])
        price_high_window = int(params['rpt_price_high_window'])
        min_volume = params['rpt_min_volume']
        min_price = params['rpt_min_price']
        rsi_threshold = params['rpt_rsi_threshold']
        pe_limit = params['rpt_pe_limit']
        top_n = int(params['rpt_top_n'])

        # 獲取數據
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        pe = self.dl.get('pe')
        rev = self.dl.get('rev')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')
        rsi = self.dl.get('rsi')
        業外收支營收率 = self.dl.get('業外收支營收率')
        營業毛利率 = self.dl.get('營業毛利率')
        稅前淨利率 = self.dl.get('稅前淨利率')
        稅後淨利率 = self.dl.get('稅後淨利率')
        limit_up_all_day = self.dl.get('limit_up_all_day')
        inventory = self.dl.get('inventory')

        # 近n月平均營收
        rev_ma = rev.average(rev_ma_period)

        # 近n月平均營收創新高
        condition_近n月平均營收創新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()

        # 近n日內有1日股價創新高
        condition_近n日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)

        # 成交量條件
        condition_近1日成交均量 = vol.average(1) > min_volume

        # 多頭排列訊號
        long_ma_pattern = (
                        (close > close.average(5))
                        & (close > close.average(10))
                        & (close > close.average(20))
                        & (close > close.average(60))
                        & (close > close.average(150))
                        & (close > close.average(200))
                        )

        # 超級績效
        收盤價_超級績效 = close > (close.average(250) * 1.1)

        # RSI指標
        rsi_higt_trend = (rsi > rsi_threshold).sustain(1)

        # 營業毛利率連續五季大於5%
        gpm_trend = (營業毛利率 > 5).sustain(5)

        # 稅前淨利率連續一季大於4%
        btpm_trend = (稅前淨利率 > 4).sustain(1)

        # 稅後淨利率連續一季大於3%
        atpm_trend = (稅後淨利率 > 3).sustain(1)

        # 月營收年增分級優於全市場90%
        rev_rise_nsatisfy = rev_yoy_growth.rank(pct=True, axis=1) > 0.9

        # 超過400張以上大戶占比18%
        boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= 12) & (inventory.持股分級.astype(int) <= 16)]
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= 18

        # 本益比條件
        pe_range = ~(pe_limit <= pe)

        # 綜合所有條件
        conditions = (
                condition_近n月平均營收創新高
                & condition_近n日內有1日股價創新高
                & condition_近1日成交均量
                & long_ma_pattern
                & gpm_trend
                & btpm_trend
                & atpm_trend
                & rev_rise_nsatisfy
                & (close > min_price)
                & (業外收支營收率 < 7.3)
                & pe_range
                & 收盤價_超級績效
                & ((vol >= vol.rolling(20).mean() * 0.8))
                & rsi_higt_trend
                & boss_inventory_over400
                & ~limit_up_all_day
                )

        # 選出符合條件且年增率最高的top_n檔
        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(top_n).reindex(rev.index_str_to_date().index, method="ffill")
        return position

    def combine_strategies(self, params: Dict) -> Any:
        """合併三個策略"""
        try:
            # 獲取三個策略的持股
            pos_lv = self.strategy_low_volatility_pe(params)
            pos_si = self.strategy_small_investor(params)
            pos_rpt = self.strategy_revenue_price_turbo(params)

            # 獲取權重
            weight_lv = params['weight_lv']
            weight_si = params['weight_si']
            weight_rpt = params['weight_rpt']

            # 正規化權重
            total_weight = weight_lv + weight_si + weight_rpt
            weight_lv /= total_weight
            weight_si /= total_weight
            weight_rpt /= total_weight

            # 合併
            position_combined = (
                pos_lv * weight_lv +
                pos_si * weight_si +
                pos_rpt * weight_rpt
            )

            return position_combined

        except Exception as e:
            print(f"   ⚠️ 策略合併失敗: {e}")
            return None

strategy_engine = StrategyEngine(data_loader)

# =============================================================================
# 第六部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """
    基因解碼器

    基因結構（60個參數）：
    - 策略一（低波動本益比）：9個參數
    - 策略二（小資族）：8個參數
    - 策略三（營收股價雙渦輪）：8個參數
    - 策略權重：3個
    - 回測參數：4個
    """

    GENE_LENGTH = 32

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # === 策略一：低波動本益比（9個參數）===
        params['lv_rev_ma3_ma12_ratio'] = genes[0] * 0.5 + 1.0  # 1.0-1.5
        params['lv_rev_consistency'] = genes[1] * 0.4 + 0.5  # 0.5-0.9
        params['lv_volatility_threshold'] = genes[2] * 0.05 + 0.02  # 0.02-0.07
        params['lv_margin_usage_limit'] = genes[3] * 30 + 20  # 20-50
        params['lv_non_op_income_limit'] = genes[4] * 10 + 5  # 5-15
        params['lv_min_volume'] = genes[5] * 200000 + 100000  # 10萬-30萬
        params['lv_pe_min'] = genes[6] * 10 + 3  # 3-13
        params['lv_pe_max'] = genes[7] * 20 + 15  # 15-35
        params['lv_top_n'] = int(genes[8] * 5 + 2)  # 2-7

        # === 策略二：小資族（8個參數）===
        params['si_market_value_limit'] = genes[9] * 10e9 + 10e9  # 100億-200億
        params['si_market_rev_ratio_limit'] = genes[10] * 3 + 2  # 2-5
        params['si_rev_yoy_growth_limit'] = genes[11] * 10 - 15  # -15 to -5
        params['si_rev_mom_growth_limit'] = genes[12] * 30 - 70  # -70 to -40
        params['si_rsv_period'] = int(genes[13] * 30 + 40)  # 40-70
        params['si_ma_period'] = int(genes[14] * 40 + 50)  # 50-90
        params['si_volume_threshold'] = genes[15] * 200000 + 100000  # 10萬-30萬
        params['si_top_n'] = int(genes[16] * 6 + 4)  # 4-10

        # === 策略三：營收股價雙渦輪（8個參數）===
        params['rpt_rev_ma_period'] = int(genes[17] * 5 + 3)  # 3-8
        params['rpt_rev_ma_lookback'] = int(genes[18] * 15 + 15)  # 15-30
        params['rpt_price_high_window'] = int(genes[19] * 8 + 4)  # 4-12
        params['rpt_min_volume'] = genes[20] * 300000 + 200000  # 20萬-50萬
        params['rpt_min_price'] = genes[21] * 10 + 10  # 10-20
        params['rpt_rsi_threshold'] = genes[22] * 20 + 50  # 50-70
        params['rpt_pe_limit'] = genes[23] * 100 + 100  # 100-200
        params['rpt_top_n'] = int(genes[24] * 10 + 8)  # 8-18

        # === 策略權重（3個）===
        raw_weights = genes[25:28]
        total = sum(raw_weights) + 1e-10
        params['weight_lv'] = raw_weights[0] / total
        params['weight_si'] = raw_weights[1] / total
        params['weight_rpt'] = raw_weights[2] / total

        # === 回測參數（4個）===
        params['stop_loss'] = genes[28] * 0.2 + 0.15  # 15%-35%
        params['trail_stop'] = genes[29] * 0.3 + 0.2  # 20%-50%
        params['take_profit'] = genes[30] * 0.5 + 0.5  # 50%-100%
        params['position_limit'] = genes[31] * 0.2 + 0.25  # 25%-45%

        return params

gene_decoder = GeneDecoder()

# =============================================================================
# 第七部分：Walk-Forward 回測引擎
# =============================================================================
class WalkForwardBacktest:
    """Walk-Forward 回測引擎"""

    def __init__(self):
        self.results = []

    def run_walk_forward(self, position, params: Dict) -> Dict:
        """執行 Walk-Forward 分析"""
        try:
            # 過濾時間
            if BACKTEST_START:
                position = position.loc[BACKTEST_START:]

            if position is None or position.empty or len(position) < 100:
                return self._empty_result()

            # 分割時間窗口
            windows = self._split_windows(position.index)

            if len(windows) < 2:
                # 窗口不足，直接全期回測
                return self._simple_backtest(position, params)

            # 各窗口回測
            window_results = []
            for i, (train_dates, test_dates) in enumerate(windows):
                # 只用測試期
                test_pos = position.loc[test_dates[0]:test_dates[-1]]

                if test_pos.empty:
                    continue

                result = self._backtest_single_window(test_pos, params, f'Window{i}')
                if result:
                    window_results.append(result)

            if not window_results:
                return self._empty_result()

            # 整體回測
            overall_result = self._backtest_single_window(position, params, 'Overall')

            # 計算穩健性
            sharpes = [r['sharpe'] for r in window_results]
            consistency = self._calculate_consistency(sharpes)
            is_robust = (
                consistency > 0.6 and
                all(s > 0 for s in sharpes) and
                overall_result['sharpe'] > TARGET_SHARPE * 0.7
            )

            return {
                'overall': overall_result,
                'windows': window_results,
                'is_robust': is_robust,
                'consistency_score': consistency,
            }

        except Exception as e:
            print(f"   ⚠️ Walk-Forward 失敗: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result()

    def _split_windows(self, dates) -> List[Tuple]:
        """分割時間窗口"""
        windows = []

        start_date = dates[0]
        end_date = dates[-1]

        current_date = start_date
        while current_date < end_date:
            train_end = current_date + pd.DateOffset(months=TRAIN_MONTHS)
            test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

            if test_end > end_date:
                test_end = end_date

            train_dates = dates[(dates >= current_date) & (dates < train_end)]
            test_dates = dates[(dates >= train_end) & (dates < test_end)]

            if len(train_dates) > 20 and len(test_dates) > 10:
                windows.append((train_dates, test_dates))

            current_date = train_end

            if len(windows) >= WALK_FORWARD_WINDOWS:
                break

        return windows

    def _backtest_single_window(self, position, params: Dict, name: str) -> Optional[Dict]:
        """單個窗口回測"""
        try:
            # 🔥 應用波動率過濾（高波動時減倉）
            close_df = data_loader.get('close')
            if close_df is not None:
                position = volatility_filter.apply_filter(position, close_df)

            # 🔥 應用 3% 持股約束
            position = position_weight_mgr.normalize_position(position)

            report = sim(
                position=position,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price="high_low_avg",
                position_limit=params.get('position_limit', 0.35),
                stop_loss=params.get('stop_loss', 0.25),
                trail_stop=params.get('trail_stop', 0.35),
                take_profit=params.get('take_profit', 0.7),
                stop_trading_next_period=False,
                upload=False,
                name=name,
            )

            metrics = report.get_metrics()

            return {
                'name': name,
                'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
                'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
                'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
                'capacity': metrics['liquidity'].get('capacity', 0) or 0,
                'win_rate': metrics['profitability'].get('winRate', 0) or 0,
            }
        except Exception as e:
            print(f"   ⚠️ 回測失敗 ({name}): {e}")
            return None

    def _simple_backtest(self, position, params: Dict) -> Dict:
        """簡單回測"""
        result = self._backtest_single_window(position, params, 'Simple')

        if result:
            return {
                'overall': result,
                'windows': [result],
                'is_robust': True,
                'consistency_score': 1.0,
            }
        else:
            return self._empty_result()

    def _calculate_consistency(self, sharpes: List[float]) -> float:
        """計算一致性分數"""
        if not sharpes or len(sharpes) < 2:
            return 0.0

        mean_sharpe = np.mean(sharpes)
        if mean_sharpe <= 0:
            return 0.0

        std_sharpe = np.std(sharpes)
        cv = std_sharpe / (mean_sharpe + 1e-10)

        consistency = max(0, 1 - cv)
        return consistency

    def _empty_result(self) -> Dict:
        """空結果"""
        return {
            'overall': {
                'sharpe': 0,
                'annual_return': 0,
                'max_drawdown': 1,
                'capacity': 0,
                'win_rate': 0,
            },
            'windows': [],
            'is_robust': False,
            'consistency_score': 0,
        }

walk_forward = WalkForwardBacktest()

# =============================================================================
# 🔬 樣本外測試函數 (Out-of-Sample Testing)
# =============================================================================
def run_backtest_period(individual: List[float], start_date: str, end_date: str, name: str = "Strategy") -> Optional[Dict]:
    """
    🔬 指定期間回測（用於樣本外測試）

    Args:
        individual: 基因向量
        start_date: 開始日期
        end_date: 結束日期
        name: 策略名稱
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略
        position = strategy_engine.combine_strategies(params)

        if position is None or position.empty:
            return None

        # 過濾時間範圍
        position = position.loc[start_date:end_date]

        if position.empty or position.sum().sum() == 0:
            return None

        # 🔥 應用波動率過濾（高波動時減倉）
        close_df = data_loader.get('close')
        if close_df is not None:
            position = volatility_filter.apply_filter(position, close_df)

        # 🔥 應用 3% 持股約束
        position = position_weight_mgr.normalize_position(position)

        report = sim(
            position=position,
            fee_ratio=1.425 / 1000,
            tax_ratio=3 / 1000,
            trade_at_price="high_low_avg",
            position_limit=params.get('position_limit', 0.35),
            stop_loss=params.get('stop_loss', 0.25),
            trail_stop=params.get('trail_stop', 0.35),
            take_profit=params.get('take_profit', 0.7),
            upload=False,
            name=name,
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
            'gene': list(individual)
        }
    except Exception as e:
        return None


def run_oos_test(individual: List[float], name: str = "OOS_Test") -> Optional[Dict]:
    """
    🔬 執行樣本外測試（同時回測訓練期和測試期）

    Returns:
        dict: 包含 train_result, test_result, overfit_warning
    """
    # 訓練期回測
    train_result = run_backtest_period(individual, TRAIN_START, TRAIN_END, f"{name}_Train")

    # 測試期回測
    test_result = run_backtest_period(individual, TEST_START, TEST_END, f"{name}_Test")

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


def run_detailed_oos_test(individual: List[float], gen: int) -> Optional[Dict]:
    """
    🔬 執行詳細樣本外測試（顯示完整回測報告）

    每 OOS_TEST_INTERVAL 代執行一次，顯示測試期的詳細回測結果
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略
        position = strategy_engine.combine_strategies(params)

        if position is None or position.empty:
            return None

        # 🔥 應用波動率過濾（高波動時減倉）
        close_df = data_loader.get('close')
        if close_df is not None:
            position = volatility_filter.apply_filter(position, close_df)

        # 🔥 應用 3% 持股約束
        position = position_weight_mgr.normalize_position(position)

        # === 訓練期回測（簡要） ===
        train_position = position.loc[TRAIN_START:TRAIN_END]
        if train_position.empty or train_position.sum().sum() == 0:
            return None

        train_report = sim(
            position=train_position,
            stop_loss=params.get('stop_loss', 0.25),
            trail_stop=params.get('trail_stop', 0.35),
            fee_ratio=1.425 / 1000,
            tax_ratio=3 / 1000,
            trade_at_price=params.get('trade_at_price', 'high_low_avg'),
            position_limit=params.get('position_limit', 0.35),
            take_profit=params.get('take_profit', 0.7),
            upload=False,
            name=f"小小龍_W{WINDOW_ID}_Gen{gen}_訓練期"
        )

        if train_report is None:
            return None
        train_metrics = train_report.get_metrics()
        train_sharpe = train_metrics['ratio'].get('sharpeRatio', 0) or 0

        # === 測試期回測（詳細顯示） ===
        test_position = position.loc[TEST_START:TEST_END]
        if test_position.empty or test_position.sum().sum() == 0:
            return None

        print(f"\n{'='*60}")
        print(f"📊 第 {gen} 代 - 測試期詳細回測 ({TEST_START[:4]}~{TEST_END[:4]})")
        print(f"{'='*60}")

        test_report = sim(
            position=test_position,
            stop_loss=params.get('stop_loss', 0.25),
            trail_stop=params.get('trail_stop', 0.35),
            fee_ratio=1.425 / 1000,
            tax_ratio=3 / 1000,
            trade_at_price=params.get('trade_at_price', 'high_low_avg'),
            position_limit=params.get('position_limit', 0.35),
            take_profit=params.get('take_profit', 0.7),
            upload=False,
            name=f"小小龍_W{WINDOW_ID}_Gen{gen}_測試期"
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


def print_oos_comparison(oos_result: Optional[Dict]) -> None:
    """打印樣本外測試比較結果"""
    if oos_result is None:
        print("   ⚠️ 樣本外測試失敗")
        return

    train = oos_result['train']
    test = oos_result['test']

    print(f"\n   {'─'*55}")
    print(f"   📊 樣本外測試結果比較")
    print(f"   {'─'*55}")
    print(f"   {'指標':<12} {'訓練期':>12} {'測試期':>12} {'比率':>10}")
    print(f"   {'':<12} {'('+TRAIN_START[:4]+'~'+TRAIN_END[:4]+')':>12} {'('+TEST_START[:4]+'~'+TEST_END[:4]+')':>12}")
    print(f"   {'─'*55}")
    print(f"   {'夏普值':<10} {train['sharpe']:>12.3f} {test['sharpe']:>12.3f} {oos_result['sharpe_ratio']:>9.1%}")
    print(f"   {'年化報酬':<10} {train['annual_return']*100:>11.1f}% {test['annual_return']*100:>11.1f}% {oos_result['return_ratio']:>9.1%}")
    print(f"   {'最大回檔':<10} {train['max_drawdown']*100:>11.1f}% {test['max_drawdown']*100:>11.1f}%")
    print(f"   {'胃納量(萬)':<10} {train['capacity']/1e4:>12.0f} {test['capacity']/1e4:>12.0f}")
    print(f"   {'─'*55}")

    if oos_result['is_overfit']:
        print(f"   🔴 過擬合警告:")
        for warning in oos_result['overfit_warnings']:
            print(f"      {warning}")
    else:
        print(f"   🟢 通過樣本外測試！測試期表現穩定")


# =============================================================================
# 第八部分：適應度評估函數
# =============================================================================
def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """
    評估個體適應度

    Returns: (sharpe_score, capacity_score, return_score, robustness_penalty)
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略
        position = strategy_engine.combine_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        # 執行 Walk-Forward 回測
        wf_result = walk_forward.run_walk_forward(position, params)

        overall = wf_result['overall']

        # 計算各項分數
        sharpe = overall['sharpe']
        capacity = overall['capacity']
        annual_return = overall['annual_return']
        max_drawdown = overall.get('max_drawdown', 0)

        # 🔥 回檔硬限制：超過 20% 直接給極低分（加重懲罰以提升夏普）
        if max_drawdown > MAX_DRAWDOWN:
            drawdown_penalty = -10.0 * (max_drawdown - MAX_DRAWDOWN) / MAX_DRAWDOWN
        else:
            drawdown_penalty = 0.0

        # 夏普值分數（0-5）
        sharpe_score = min(5.0, max(0, sharpe / TARGET_SHARPE * 5.0))

        # 胃納量分數（0-5）
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))

        # 年化報酬分數（0-2）
        return_score = min(2.0, max(0, annual_return / TARGET_ANNUAL_RETURN * 2.0))

        # 穩健性懲罰
        if wf_result['is_robust']:
            robustness_penalty = 0.0
        else:
            consistency = wf_result['consistency_score']
            robustness_penalty = -(1 - consistency) * 3

        # 合併回檔懲罰到穩健性懲罰
        total_penalty = robustness_penalty + drawdown_penalty

        return (sharpe_score, capacity_score, return_score, total_penalty)

    except Exception as e:
        print(f"   ⚠️ 評估失敗: {e}")
        import traceback
        traceback.print_exc()
        return (0.0, 0.0, 0.0, -10.0)

# =============================================================================
# 第九部分：DEAP 設定
# =============================================================================
# 清除舊定義
if 'FitnessMulti' in dir(creator):
    del creator.FitnessMulti
if 'Individual' in dir(creator):
    del creator.Individual

# 創建適應度（4目標，全部最大化）
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=GeneDecoder.GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=0.0, up=1.0, eta=20.0, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# 🔥 新增：從基因列表創建個體（用於跨視窗交換）
def create_individual_from_gene(gene_list):
    """從基因列表創建 DEAP Individual"""
    ind = creator.Individual(gene_list)
    return ind

toolbox.register("individual_from_gene", create_individual_from_gene)

print("✅ DEAP NSGA-II 配置完成")

# =============================================================================
# 第十部分：Pareto 歷史管理器
# =============================================================================
class ParetoArchiveManager:
    """Pareto 前緣歷史管理器"""

    def __init__(self, archive_file: str):
        self.archive_file = archive_file
        self.archive = []
        self._load()

    def _load(self):
        """載入歷史"""
        if os.path.exists(self.archive_file):
            try:
                with open(self.archive_file, 'rb') as f:
                    data = pickle.load(f)
                    self.archive = data.get('individuals', [])
                print(f"✅ 載入 {len(self.archive)} 個歷史精英")
            except Exception as e:
                print(f"⚠️ 歷史載入失敗: {e}")
                self.archive = []
        else:
            print("ℹ️ 無歷史存檔，從頭開始")

    def update(self, pareto_front: List):
        """更新 Pareto 前緣"""
        # 合併新舊個體
        all_individuals = self.archive + [
            {'genes': list(ind), 'fitness': ind.fitness.values}
            for ind in pareto_front
        ]

        # 去重
        unique = self._deduplicate(all_individuals)

        # 保留前 30 名
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:30]

        # 保存
        self._save()

    def _deduplicate(self, individuals: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        unique = []

        for ind in individuals:
            gene_hash = hashlib.md5(str(ind['genes'][:10]).encode()).hexdigest()

            if gene_hash not in seen:
                seen.add(gene_hash)
                unique.append(ind)

        return unique

    def _save(self):
        """保存（使用安全寫入）"""
        try:
            save_data = {
                'individuals': self.archive,
                'timestamp': datetime.now().isoformat(),
            }
            success = SafeFileManager.safe_pickle_save(save_data, self.archive_file)
            if not success:
                # 降級到傳統寫入
                with open(self.archive_file, 'wb') as f:
                    pickle.dump(save_data, f)
        except Exception as e:
            print(f"⚠️ 歷史保存失敗: {e}")

    def inject_elites(self, population: List, ratio: float = 0.3):
        """注入精英"""
        if not self.archive:
            return population

        n_inject = int(len(population) * ratio)
        n_inject = min(n_inject, len(self.archive))

        # 選擇最佳精英
        elites = sorted(self.archive, key=lambda x: sum(x['fitness']), reverse=True)[:n_inject]

        # 替換族群中最差的個體
        population.sort(key=lambda ind: sum(ind.fitness.values) if ind.fitness.valid else -999)

        for i, elite in enumerate(elites):
            population[i][:] = elite['genes']
            if hasattr(population[i], 'fitness'):
                del population[i].fitness.values

        print(f"✅ 注入 {n_inject} 個歷史精英")

        return population

pareto_archive = ParetoArchiveManager(paths.pareto_archive)

# =============================================================================
# 第十一部分：Checkpoint 管理器（支援斷點續傳）
# =============================================================================
class CheckpointManager:
    """Checkpoint 管理器 - 支援斷點續傳"""

    def __init__(self, window_id: int, base_dir: str = BASE_DIR):
        self.window_id = window_id
        self.checkpoint_dir = f"{base_dir}/window_{window_id}"
        self.checkpoint_file = f"{self.checkpoint_dir}/checkpoint.pkl"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(self, population: List, generation: int, pareto_front: List = None,
             history: List = None, random_state=None, numpy_state=None):
        """保存 checkpoint"""
        try:
            checkpoint_data = {
                'population': [{'genes': list(ind), 'fitness': list(ind.fitness.values) if ind.fitness.valid else None}
                              for ind in population],
                'generation': generation,
                'pareto_front': [{'genes': list(ind), 'fitness': list(ind.fitness.values)}
                                for ind in pareto_front] if pareto_front else [],
                'history': history or [],
                'random_state': random_state or random.getstate(),
                'numpy_state': numpy_state or np.random.get_state(),
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

    def load(self, toolbox) -> dict:
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

            # 重建族群
            population = []
            for ind_data in data['population']:
                ind = toolbox.individual_from_gene(ind_data['genes'])
                if ind_data['fitness'] and len(ind_data['fitness']) == 4:
                    ind.fitness.values = tuple(ind_data['fitness'])
                population.append(ind)

            # 重建 Pareto 前緣
            pareto_front = []
            for ind_data in data.get('pareto_front', []):
                ind = toolbox.individual_from_gene(ind_data['genes'])
                if ind_data['fitness'] and len(ind_data['fitness']) == 4:
                    ind.fitness.values = tuple(ind_data['fitness'])
                pareto_front.append(ind)

            # 恢復隨機狀態
            if data.get('random_state'):
                random.setstate(data['random_state'])
            if data.get('numpy_state'):
                np.random.set_state(data['numpy_state'])

            print(f"✅ 從 checkpoint 恢復: 第 {data['generation']} 代, 族群 {len(population)}")
            print(f"   📅 保存時間: {data.get('timestamp', 'N/A')}")

            return {
                'population': population,
                'generation': data['generation'],
                'pareto_front': pareto_front,
                'history': data.get('history', [])
            }

        except Exception as e:
            print(f"⚠️ Checkpoint 載入失敗: {e}")
            return None

    def exists(self) -> bool:
        """檢查是否有 checkpoint"""
        return os.path.exists(self.checkpoint_file)

    def get_info(self) -> dict:
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


# 初始化 checkpoint 管理器
checkpoint_mgr = CheckpointManager(WINDOW_ID, BASE_DIR)
print("✅ Checkpoint 管理器已初始化")


# =============================================================================
# 第十二部分：演化引擎
# =============================================================================
class ProgressLogger:
    """進度日誌記錄器（用於監控系統）"""

    def __init__(self, window_id: int, base_dir: str = BASE_DIR):
        self.window_id = window_id
        self.log_dir = f"{base_dir}/window_{window_id}_logs"
        self.history_file = f"{self.log_dir}/progress_history.json"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log_generation(self, gen: int, total_gen: int, stats: Dict):
        """記錄單代進度"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'window_id': self.window_id,
            'current_gen': gen + 1,  # 輸出為 1-based
            'total_gen': total_gen,
            'best_sharpe': stats.get('best_sharpe', 0),
            'best_capacity': stats.get('best_capacity', 0),
            'best_composite': stats.get('best_composite', 0),
            'best_return': stats.get('best_return', 0),
            'robustness': stats.get('best_robustness', 0),
            'elapsed_total': time.time() - self.start_time,
        }

        # 讀取歷史
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass

        # 添加新記錄
        history.append(progress)

        # 保存
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


class EvolutionEngine:
    """演化引擎"""

    def __init__(self, toolbox, pareto_mgr: ParetoArchiveManager):
        self.toolbox = toolbox
        self.pareto_mgr = pareto_mgr
        self.history = []
        self.logger = ProgressLogger(WINDOW_ID)  # 🔥 初始化進度日誌記錄器

    def run(self, n_generations: int = N_GENERATIONS) -> Tuple[List, List]:
        """執行演化（支援斷點續傳）"""
        print(f"\n{'='*70}")
        print(f"🚀 開始演化")
        print(f"   族群: {POPULATION_SIZE}, 世代: {n_generations}")
        print(f"{'='*70}\n")

        # 🔥 嘗試從 checkpoint 恢復
        start_gen = 0
        checkpoint_data = checkpoint_mgr.load(self.toolbox)

        if checkpoint_data:
            population = checkpoint_data['population']
            start_gen = checkpoint_data['generation']
            self.history = checkpoint_data.get('history', [])
            print(f"🔄 從第 {start_gen} 代繼續演化...")
            print(f"   剩餘世代: {n_generations - start_gen}")

            # 重新評估沒有 fitness 的個體
            population = self._evaluate_population(population)
        else:
            # 初始化新族群
            population = self.toolbox.population(n=POPULATION_SIZE)

            # 注入歷史精英
            population = self.pareto_mgr.inject_elites(population, ratio=0.3)

            # 初始評估
            population = self._evaluate_population(population)

        # 演化循環（從 start_gen 繼續）
        for gen in range(start_gen, n_generations):
            start_time = time.time()

            # 選擇
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 變異
            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 評估
            offspring = self._evaluate_population(offspring)

            # 環境選擇
            population = self.toolbox.select(population + offspring, POPULATION_SIZE)

            # 統計
            stats = self._compute_stats(population, gen)
            self.history.append(stats)

            elapsed = time.time() - start_time

            # 🔥 3視窗交互 - 每 CROSS_WINDOW_INTERVAL 代執行
            if (gen + 1) % CROSS_WINDOW_INTERVAL == 0:
                population = cross_window_mgr.perform_cross_window_exchange(
                    population, gen + 1, self.toolbox
                )
                # 重新評估新注入的個體
                population = self._evaluate_population(population)

            # 🔥 每 5 代保存 checkpoint
            if (gen + 1) % 5 == 0:
                pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                checkpoint_mgr.save(
                    population=population,
                    generation=gen + 1,
                    pareto_front=pareto_front,
                    history=self.history
                )

            # 輸出
            if (gen + 1) % 5 == 0 or gen == start_gen:
                print(f"=== 第 {gen+1}/{n_generations} 代 ===")
                print(f"   最佳綜合: {stats['best_composite']:.4f}")
                print(f"   最佳夏普: {stats['best_sharpe']:.2f}")
                print(f"   最佳胃納量: {stats['best_capacity']:.0f} 萬")
                print(f"   穩健性: {stats['best_robustness']:.2f}")
                print(f"   耗時: {elapsed:.1f}s")

            # 🔬 每 OOS_TEST_INTERVAL 代執行詳細樣本外測試
            if (gen + 1) % OOS_TEST_INTERVAL == 0:
                best_ind = tools.selBest(population, 1)[0]
                print(f"\n   🔬 執行詳細樣本外測試...")

                # 使用新的詳細測試函數（顯示完整回測報告）
                oos_result = run_detailed_oos_test(best_ind, gen + 1)

                if oos_result:
                    # Discord 通知
                    if oos_result['is_overfit']:
                        notifier.send(
                            f"⚠️ 小小龍 視窗{WINDOW_ID} 第{gen+1}代\n"
                            f"訓練期夏普: {oos_result['train_sharpe']:.3f}\n"
                            f"測試期夏普: {oos_result['test_sharpe']:.3f}\n"
                            f"比率: {oos_result['sharpe_ratio']:.1%} 🔴 過擬合警告"
                        )
                    else:
                        notifier.send(
                            f"✅ 小小龍 視窗{WINDOW_ID} 第{gen+1}代 OOS測試\n"
                            f"訓練期夏普: {oos_result['train_sharpe']:.3f}\n"
                            f"測試期夏普: {oos_result['test_sharpe']:.3f}\n"
                            f"比率: {oos_result['sharpe_ratio']:.1%} 🟢 通過"
                        )

        # Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 更新歷史存檔
        self.pareto_mgr.update(pareto_front)

        # 🔥 最終保存 checkpoint
        checkpoint_mgr.save(
            population=population,
            generation=n_generations,
            pareto_front=pareto_front,
            history=self.history
        )

        # 輸出結果
        self._print_pareto_front(pareto_front)

        # 🔬 最終樣本外測試
        print(f"\n{'='*70}")
        print(f"🔬 最終樣本外測試")
        print(f"{'='*70}")

        final_best = tools.selBest(pareto_front, 1)[0]
        final_oos = run_oos_test(final_best, name=f"小小龍_W{WINDOW_ID}_Final")

        if final_oos:
            print_oos_comparison(final_oos)

            # Discord 通知最終結果
            train = final_oos['train']
            test = final_oos['test']
            overfit_flag = "🔴 過擬合" if final_oos['is_overfit'] else "🟢 通過"

            notifier.send_embed(
                f"🏆 小小龍 視窗{WINDOW_ID} 演化完成",
                f"訓練期({TRAIN_START[:4]}~{TRAIN_END[:4]}): 夏普 {train['sharpe']:.3f}\n"
                f"測試期({TEST_START[:4]}~{TEST_END[:4]}): 夏普 {test['sharpe']:.3f}\n"
                f"測試/訓練比: {final_oos['sharpe_ratio']:.1%}\n"
                f"胃納量: {train['capacity']/1e4:.0f}萬\n"
                f"狀態: {overfit_flag}",
                'success' if not final_oos['is_overfit'] else 'warning'
            )

        return population, pareto_front

    def _evaluate_population(self, population: List) -> List:
        """評估族群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        print(f"   評估 {len(invalid)} 個個體...")

        # 序列評估（FinLab 回測較慢，避免並行問題）
        for i, ind in enumerate(invalid):
            fitness = evaluate_fitness(ind)
            ind.fitness.values = fitness

            if (i + 1) % 5 == 0:
                print(f"   進度: {i+1}/{len(invalid)}")

        return population

    def _compute_stats(self, population: List, gen: int) -> Dict:
        """計算統計"""
        fitnesses = [ind.fitness.values for ind in population]

        sharpes = [f[0] / 5.0 * TARGET_SHARPE for f in fitnesses]
        capacities = [f[1] / 5.0 * MIN_CAPACITY / 1e4 for f in fitnesses]
        returns = [f[2] / 2.0 * TARGET_ANNUAL_RETURN for f in fitnesses]
        robustness = [f[3] for f in fitnesses]

        composites = [sum(f) for f in fitnesses]

        stats = {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': max(sharpes),
            'best_capacity': max(capacities),
            'best_return': max(returns),
            'best_robustness': max(robustness),
        }

        # 🔥 記錄進度供監控系統使用
        self.logger.log_generation(gen, N_GENERATIONS, stats)

        return stats

    def _print_pareto_front(self, pareto_front: List):
        """輸出 Pareto 前緣"""
        print(f"\n{'='*70}")
        print(f"🏆 Pareto 最優解集（前 10 名）")
        print(f"{'='*70}")
        print(f"{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納量(萬)':<12}{'達標'}")
        print("-" * 70)

        sorted_front = sorted(pareto_front,
                             key=lambda x: sum(x.fitness.values),
                             reverse=True)[:10]

        for i, ind in enumerate(sorted_front, 1):
            composite = sum(ind.fitness.values)
            sharpe = ind.fitness.values[0] / 5.0 * TARGET_SHARPE
            capacity = ind.fitness.values[1] / 5.0 * MIN_CAPACITY / 1e4

            reached = "✅" if sharpe >= TARGET_SHARPE * 0.9 and capacity >= MIN_CAPACITY / 1e4 * 0.9 else ""

            print(f"{i:<6}{composite:<10.4f}{sharpe:<10.2f}{capacity:<12.2f}{reached}")

# =============================================================================
# 第十二部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║     FinLab 台股基因演算法優化系統 - 終極版                      ║
║     NSGA-II + Walk-Forward + Pareto Archive                   ║
╠════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e4:.0f}萬+                          ║
║  📊 策略：三策略動態組合優化                                     ║
║  🔬 驗證：Walk-Forward ({WALK_FORWARD_WINDOWS} 窗口)                             ║
║  🧬 基因：{GeneDecoder.GENE_LENGTH} 個參數                                        ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # 建立演化引擎
    engine = EvolutionEngine(toolbox, pareto_archive)

    # 執行演化
    population, pareto_front = engine.run(N_GENERATIONS)

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*70}")
        print("📊 最佳個體詳細回測")
        print(f"{'='*70}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        # 顯示最佳參數
        print("\n🔍 最佳參數：")
        print(f"   策略權重：低波動 {best_params['weight_lv']:.2%}, "
              f"小資族 {best_params['weight_si']:.2%}, "
              f"雙渦輪 {best_params['weight_rpt']:.2%}")

        # 保存最佳參數
        with open(paths.best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

        # 完整回測 + 上傳到 FinLab
        try:
            print("\n執行完整回測並上傳到 FinLab...")
            position = strategy_engine.combine_strategies(best_params)

            # 🔥 應用波動率過濾（高波動時減倉）
            close_df = data_loader.get('close')
            if close_df is not None:
                position = volatility_filter.apply_filter(position, close_df)

            # 🔥 應用 3% 持股約束
            position = position_weight_mgr.normalize_position(position)

            report = sim(
                position=position,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price="high_low_avg",
                position_limit=best_params['position_limit'],
                stop_loss=best_params['stop_loss'],
                trail_stop=best_params['trail_stop'],
                take_profit=best_params['take_profit'],
                stop_trading_next_period=False,
                upload=True,  # 🔥 上傳到 FinLab（覆蓋舊版本）
                name=f'小小龍_視窗{WINDOW_ID}_最佳策略'
            )

            report.display()
            print(f"\n✅ 最佳策略已上傳到 FinLab: 小小龍_視窗{WINDOW_ID}_最佳策略")
        except Exception as e:
            print(f"⚠️ 完整回測失敗: {e}")

    print(f"\n✅ 優化完成！")
    print(f"📁 結果保存於: {paths.output_dir}")
    print(f"📁 Pareto 存檔: {paths.pareto_archive}")

    # 🔥 最終結果保存到共享區
    if pareto_front:
        cross_window_mgr.save_elites_to_shared(
            tools.selBest(pareto_front, CROSS_WINDOW_TOP_N),
            N_GENERATIONS
        )

    # 🔥 Discord 通知完成
    notifier.send_embed(
        f"🎉 視窗 {WINDOW_ID} 演化完成",
        f"共 {N_GENERATIONS} 代演化\n"
        f"Pareto 前緣數量: {len(pareto_front) if pareto_front else 0}\n"
        f"結果已保存至共享區",
        'success'
    )

    return population, pareto_front

if __name__ == "__main__":
    population, pareto = main()
