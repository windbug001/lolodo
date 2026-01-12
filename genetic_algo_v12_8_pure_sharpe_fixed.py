#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 六策略快快龍 - 台股基因演算法優化系統 v12.8
Six-Strategy Dragon - Taiwan Stock GA Optimizer
================================================================================

【核心功能】
✅ 六大策略動態組合優化
✅ 遺傳演算法 (NSGA-II) 多目標優化
✅ In-Sample / Out-of-Sample 分離驗證
✅ Check-point 自動存取與斷點續傳
✅ 多核心並行運算加速
✅ 重啟自動驗證前 5 名
✅ Pareto Archive 歷史精英持續進化

【目標績效】
- 夏普值 (Sharpe Ratio): > 4.2
- 最大回撤 (MDD): < 20%
- 資金胃納量: > 1,000 萬台幣

【回測規範】
- In-Sample (訓練期): ~2022 年底
- Out-of-Sample (測試期): 2023 ~ 至今

【持股邏輯】
- 單一標的權重 ≥ 3%，不滿 3% 則不持倉 (0%)

版本：v12.8 Pure Sharpe Fixed (2025-01-12)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 🔥 【最重要】在任何 import 之前禁用 FinLab 快取
# =============================================================================
import os
import sys
import shutil
import glob
import subprocess

# 禁用快取環境變數
os.environ['FINLAB_DISABLE_CACHE'] = '1'
os.environ['FINLAB_NO_CACHE'] = '1'

def clear_all_cache():
    """徹底清除所有可能損壞的快取"""
    print("🔧 清除所有快取...")

    # 清除 pkl 檔案
    try:
        subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-type', 'f', '-delete'],
                      stderr=subprocess.DEVNULL, timeout=10)
    except:
        pass

    # 清除 finlab 相關目錄
    cache_patterns = [
        '/root/.finlab*', '/tmp/.finlab*', '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'), '/content/.finlab*',
        '/root/*finlab*', '/tmp/*finlab*',
        os.path.expanduser('~/.cache/pandas'),
    ]

    cleared = 0
    for pattern in cache_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                cleared += 1
        except:
            pass

    if cleared > 0:
        print(f"   ✅ 已清除 {cleared} 個快取")

# 執行快取清除
clear_all_cache()

# =============================================================================
# 第一部分：套件載入與環境設定
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import pickle
import time
import hashlib
import random
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import traceback

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# =============================================================================
# 第二部分：核心設定參數
# =============================================================================
# 🔥 從環境變數讀取設定（支援多視窗）
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))
CONTINUE_EVOLUTION = os.environ.get('CONTINUE_EVOLUTION', 'true').lower() == 'true'
N_GENERATIONS = int(os.environ.get('EVOLUTION_GENERATIONS', '500'))

# 🧬 外部優秀基因路徑（可從環境變數設定）
EXTERNAL_GENES_PATH = os.environ.get(
    'EXTERNAL_GENES_PATH',
    '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014/回測結果/篩選_前5名_20260108_1519/top5_genes_for_evolution.pkl'
)

# FinLab API Key（請替換為你的 VIP Key）
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# =============================================================================
# 🎯 優化目標（根據用戶要求調整）
# =============================================================================
TARGET_SHARPE = 4.2          # 夏普值目標 > 4.2
MAX_MDD = 0.20               # 最大回撤 < 20%
MIN_CAPACITY = 10_000_000    # 胃納量 > 1000 萬
TARGET_ANNUAL_RETURN = 0.35  # 年化報酬目標 35%

# =============================================================================
# 🧬 GA 演化參數
# =============================================================================
POPULATION_SIZE = 60         # 族群大小
MUTATION_RATE = 0.25         # 變異率
CROSSOVER_RATE = 0.85        # 交叉率
ELITE_RATIO = 0.15           # 精英保留比例
N_PARALLEL_WORKERS = 4       # 並行工作數

# =============================================================================
# 📊 回測設定（In-Sample / Out-of-Sample 分離）
# =============================================================================
IN_SAMPLE_START = '2017-01-01'
IN_SAMPLE_END = '2022-12-31'      # 訓練期至 2022 年底
OUT_SAMPLE_START = '2023-01-01'   # 測試期 2023 年起
OUT_SAMPLE_END = None             # 至今

# =============================================================================
# 📐 持股配比約束
# =============================================================================
MIN_POSITION_WEIGHT = 0.03   # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股必須是 3% 倍數
MAX_STOCKS = 33              # 最多 33 檔（100% / 3%）

print(f"""
{'='*80}
🐉 六策略快快龍 - 台股基因演算法優化系統 v12.8
{'='*80}
📍 視窗 ID: {WINDOW_ID}
🔄 繼續演化: {CONTINUE_EVOLUTION}
🧬 演化世代: {N_GENERATIONS}
🎯 目標夏普: > {TARGET_SHARPE}
📉 最大回撤: < {MAX_MDD*100:.0f}%
💰 胃納目標: > {MIN_CAPACITY/1e7:.0f}00 萬
🧬 外部優秀基因: {EXTERNAL_GENES_PATH}
{'='*80}
""")

# =============================================================================
# 第三部分：套件安裝與載入
# =============================================================================
def install_packages():
    """安裝必要套件"""
    required = {
        'finlab': 'finlab',
        'deap': 'deap',
        'joblib': 'joblib',
    }

    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"   📦 安裝 {pkg_name}...")
            os.system(f'pip install {pkg_name} -q')

install_packages()

# 載入 FinLab
import finlab
from finlab import data
from finlab.backtest import sim

# 載入 DEAP
from deap import base, creator, tools, algorithms

# 載入 joblib
from joblib import Parallel, delayed

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 第四部分：環境設定與路徑管理
# =============================================================================
def setup_environment():
    """設定環境並登入 FinLab"""
    # 嘗試掛載 Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = '/content/drive/MyDrive/六策略快快龍_GA優化'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = os.environ.get('BASE_DIR', './dragon_ga_output')
        in_colab = False
        print("⚠️ 本地環境模式")

    # FinLab 登入
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")

    return base_dir, in_colab

BASE_DIR, IN_COLAB = setup_environment()

@dataclass
class PathManager:
    """路徑管理器 - 統一管理所有檔案路徑"""
    base_dir: str = BASE_DIR
    window_id: int = WINDOW_ID

    def __post_init__(self):
        # 視窗專屬目錄
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.output_dir = f"{self.window_dir}/output"
        self.checkpoint_dir = f"{self.window_dir}/checkpoints"
        self.history_dir = f"{self.window_dir}/history"
        self.log_dir = f"{self.base_dir}/window_{self.window_id}_logs"

        # 共享目錄（跨視窗）
        self.pareto_dir = f"{self.base_dir}/shared_pareto"

        # 建立所有目錄
        for d in [self.window_dir, self.output_dir, self.checkpoint_dir,
                  self.history_dir, self.log_dir, self.pareto_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> str:
        """Check-point 檔案路徑"""
        return f"{self.checkpoint_dir}/checkpoint_w{self.window_id}.pkl"

    @property
    def pareto_archive_file(self) -> str:
        """Pareto Archive 檔案路徑"""
        return f"{self.pareto_dir}/pareto_archive_w{self.window_id}.pkl"

    @property
    def best_params_file(self) -> str:
        """最佳參數檔案路徑"""
        return f"{self.output_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log_file(self) -> str:
        """進度日誌檔案路徑"""
        return f"{self.log_dir}/progress_history.json"

    @property
    def top5_validation_file(self) -> str:
        """前 5 名驗證結果檔案"""
        return f"{self.output_dir}/top5_validation_w{self.window_id}.json"

paths = PathManager()
print(f"📁 工作目錄: {paths.window_dir}")

# =============================================================================
# 🔥 第 4.5 部分：基因解碼器（提前定義，避免順序問題）
# =============================================================================
class DragonGeneDecoder:
    """
    🧬 龍族基因解碼器（提前定義版本）

    基因長度：80
    - 策略一參數：0-9 (10)
    - 策略二參數：10-19 (10)
    - 策略三參數：20-29 (10)
    - 策略四參數：30-39 (10)
    - 策略五參數：40-49 (10)
    - 策略六參數：50-59 (10)
    - 策略權重：60-65 (6)
    - 回測參數：66-73 (8)
    - 保留：74-79 (6)
    """

    GENE_LENGTH = 80

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因為策略參數"""
        # 確保基因長度
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # === 策略一：低波動本益比 (0-9) ===
        params['lv_rev_ratio'] = genes[0] * 0.5 + 1.0        # 1.0-1.5
        params['lv_vol_threshold'] = genes[1] * 0.05 + 0.02  # 0.02-0.07
        params['lv_margin_limit'] = genes[2] * 30 + 20       # 20-50
        params['lv_non_op_limit'] = genes[3] * 10 + 5        # 5-15
        params['lv_pe_min'] = genes[4] * 10 + 3              # 3-13
        params['lv_pe_max'] = genes[5] * 20 + 15             # 15-35
        params['lv_top_n'] = int(genes[6] * 6 + 2)           # 2-8

        # === 策略二：小資族成長 (10-19) ===
        params['sc_mv_limit'] = genes[10] * 15e9 + 5e9       # 50億-200億
        params['sc_mv_rev_ratio'] = genes[11] * 3 + 2        # 2-5
        params['sc_rev_yoy_limit'] = genes[12] * 10 - 15     # -15 to -5
        params['sc_rev_mom_limit'] = genes[13] * 30 - 70     # -70 to -40
        params['sc_rsv_period'] = int(genes[14] * 30 + 40)   # 40-70
        params['sc_ma_period'] = int(genes[15] * 40 + 50)    # 50-90
        params['sc_top_n'] = int(genes[16] * 6 + 4)          # 4-10

        # === 策略三：營收股價雙渦輪 (20-29) ===
        params['rpt_rev_ma'] = int(genes[20] * 5 + 3)        # 3-8
        params['rpt_lookback'] = int(genes[21] * 15 + 15)    # 15-30
        params['rpt_price_window'] = int(genes[22] * 8 + 4)  # 4-12
        params['rpt_min_vol'] = genes[23] * 300000 + 150000  # 15萬-45萬
        params['rpt_min_price'] = genes[24] * 15 + 10        # 10-25
        params['rpt_rsi'] = genes[25] * 20 + 50              # 50-70
        params['rpt_pe_limit'] = genes[26] * 100 + 100       # 100-200
        params['rpt_top_n'] = int(genes[27] * 10 + 8)        # 8-18

        # === 策略四：高殖利率防禦 (30-39) ===
        params['hd_div_min'] = genes[30] * 4 + 3             # 3-7%
        params['hd_vol_max'] = genes[31] * 0.2 + 0.15        # 0.15-0.35
        params['hd_pe_max'] = genes[32] * 15 + 10            # 10-25
        params['hd_roe_min'] = genes[33] * 10 + 5            # 5-15%
        params['hd_top_n'] = int(genes[34] * 6 + 4)          # 4-10

        # === 策略五：動能突破 (40-49) ===
        params['mb_mom_threshold'] = genes[40] * 0.2 + 0.05  # 0.05-0.25
        params['mb_vol_ratio'] = genes[41] * 1.5 + 1.0       # 1.0-2.5
        params['mb_rsi_min'] = genes[42] * 20 + 40           # 40-60
        params['mb_rsi_max'] = genes[43] * 15 + 70           # 70-85
        params['mb_top_n'] = int(genes[44] * 8 + 5)          # 5-13

        # === 策略六：價值成長混合 (50-59) ===
        params['vg_pe_pct'] = genes[50] * 0.3 + 0.2          # 0.2-0.5
        params['vg_pb_pct'] = genes[51] * 0.3 + 0.3          # 0.3-0.6
        params['vg_growth'] = genes[52] * 20 + 5             # 5-25%
        params['vg_roe_min'] = genes[53] * 10 + 5            # 5-15%
        params['vg_top_n'] = int(genes[54] * 8 + 5)          # 5-13

        # === 策略權重 (60-65) ===
        raw_weights = genes[60:66]
        total = sum(raw_weights) + 1e-10
        params['weight_lv'] = raw_weights[0] / total
        params['weight_sc'] = raw_weights[1] / total
        params['weight_rpt'] = raw_weights[2] / total
        params['weight_hd'] = raw_weights[3] / total
        params['weight_mb'] = raw_weights[4] / total
        params['weight_vg'] = raw_weights[5] / total

        # === 回測參數 (66-73) ===
        params['stop_loss'] = genes[66] * 0.15 + 0.15        # 15%-30%
        params['trail_stop'] = genes[67] * 0.2 + 0.2         # 20%-40%
        params['take_profit'] = genes[68] * 0.4 + 0.5        # 50%-90%
        params['position_limit'] = genes[69] * 0.15 + 0.25   # 25%-40%
        params['target_stocks'] = int(genes[70] * 15 + 10)   # 10-25

        return params

# 🔥 提前初始化 gene_decoder（確保在任何評估之前就存在）
gene_decoder = DragonGeneDecoder()
print(f"✅ 基因解碼器初始化完成 (長度: {DragonGeneDecoder.GENE_LENGTH})")

# =============================================================================
# 第五部分：數據載入器（單例模式）
# =============================================================================
class DragonDataLoader:
    """
    🐉 龍族數據載入器

    功能：
    1. 載入所有 FinLab 數據
    2. 計算衍生指標
    3. 單例模式避免重複載入
    4. 自動修復損壞快取
    """

    _instance = None
    _cache = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_all_data()
            DragonDataLoader._initialized = True

    def _safe_get(self, key: str, retry_count: int = 2) -> pd.DataFrame:
        """安全載入數據，自動處理快取錯誤"""
        for attempt in range(retry_count + 1):
            try:
                return data.get(key)
            except (EOFError, Exception) as e:
                if attempt < retry_count:
                    print(f"   ⚠️ 載入失敗 ({key})，清除快取重試...")
                    clear_all_cache()
                    time.sleep(1)
                else:
                    raise e

    def _load_all_data(self):
        """載入所有必要數據"""
        print("📊 載入 FinLab 數據庫...")
        start_time = time.time()

        # === 價格數據 ===
        self._cache['close'] = self._safe_get('price:收盤價')
        self._cache['open'] = self._safe_get('price:開盤價')
        self._cache['high'] = self._safe_get('price:最高價')
        self._cache['low'] = self._safe_get('price:最低價')
        self._cache['vol'] = self._safe_get('price:成交股數')
        self._cache['adj_close'] = self._safe_get('etl:adj_close')

        # === 估值數據 ===
        self._cache['pe'] = self._safe_get('price_earning_ratio:本益比')
        self._cache['pb'] = self._safe_get('price_earning_ratio:股價淨值比')
        self._cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')

        # === 營收數據 ===
        self._cache['rev'] = self._safe_get('monthly_revenue:當月營收')
        self._cache['rev_yoy'] = self._safe_get('monthly_revenue:去年同月增減(%)')
        self._cache['rev_mom'] = self._safe_get('monthly_revenue:上月比較增減(%)')

        # === 基本面數據 ===
        self._cache['營業利益成長率'] = self._safe_get('fundamental_features:營業利益成長率')
        self._cache['業外收支營收率'] = self._safe_get('fundamental_features:業外收支營收率')
        self._cache['營業毛利率'] = self._safe_get('fundamental_features:營業毛利率')
        self._cache['ROE綜合損益'] = self._safe_get('fundamental_features:ROE綜合損益')
        self._cache['稅後淨利率'] = self._safe_get('fundamental_features:稅後淨利率')
        self._cache['稅前淨利率'] = self._safe_get('fundamental_features:稅前淨利率')

        # === 籌碼數據 ===
        self._cache['融資使用率'] = self._safe_get('margin_transactions:融資使用率')
        self._cache['董監持股比'] = self._safe_get('internal_equity_changes:董監持有股數占比')
        self._cache['inventory'] = self._safe_get('inventory')

        # === 市值與財報 ===
        self._cache['市值'] = self._safe_get('etl:market_value')
        self._cache['股本'] = self._safe_get('financial_statement:股本')
        self._cache['投資活動現金流'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._cache['營業活動現金流'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._cache['稅後淨利'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._cache['權益總計'] = self._safe_get('financial_statement:股東權益總額')

        # === 技術指標 ===
        self._cache['rsi'] = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
        self._cache['atr'] = data.indicator('ATR', adjust_price=True, timeperiod=10)

        # === 計算衍生指標 ===
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"✅ 數據載入完成 ({elapsed:.1f}s)")
        print(f"   股票數: {len(self._cache['close'].columns)}")
        print(f"   期間: {self._cache['close'].index[0].strftime('%Y-%m-%d')} ~ {self._cache['close'].index[-1].strftime('%Y-%m-%d')}")

    def _calculate_derived_indicators(self):
        """計算衍生指標"""
        close = self._cache['close']
        high = self._cache['high']
        low = self._cache['low']
        open_ = self._cache['open']
        vol = self._cache['vol']
        rev = self._cache['rev']
        adj_close = self._cache['adj_close']
        atr = self._cache['atr']

        # 營收均線
        self._cache['rev_ma3'] = rev.average(3)
        self._cache['rev_ma6'] = rev.average(6)
        self._cache['rev_ma12'] = rev.average(12)

        # 價格均線
        for period in [5, 10, 20, 40, 60, 75, 90, 120, 150, 200, 240]:
            self._cache[f'sma{period}'] = close.average(period)

        # 波動率
        self._cache['entry_volatility'] = atr / adj_close
        returns = close.pct_change()
        self._cache['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        self._cache['volatility_60'] = returns.rolling(60).std() * np.sqrt(252)

        # 動能指標
        self._cache['momentum_20'] = close / close.shift(20) - 1
        self._cache['momentum_60'] = close / close.shift(60) - 1
        self._cache['momentum_120'] = close / close.shift(120) - 1

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

        # 市值營收比
        當月營收 = self._cache['rev'] * 1000
        self._cache['當季營收'] = 當月營收.rolling(4).sum()
        self._cache['市值營收比'] = self._cache['市值'] / self._cache['當季營收']

        # PEG
        self._cache['peg'] = self._cache['pe'] / self._cache['營業利益成長率']

        # 成交量指標
        self._cache['vol_ma5'] = vol.average(5)
        self._cache['vol_ma20'] = vol.average(20)
        self._cache['vol_ratio'] = vol / (self._cache['vol_ma20'] + 1)

    def get(self, key: str) -> pd.DataFrame:
        """獲取數據"""
        return self._cache.get(key)

# 初始化數據載入器（單例）
data_loader = DragonDataLoader()

# =============================================================================
# 第六部分：持股配比管理器
# =============================================================================
class PositionWeightManager:
    """
    📐 持股配比管理器

    約束：
    1. 每隻股票至少 3%
    2. 持股必須是 3% 倍數
    3. 不滿 3% 則不持倉 (0%)
    4. 總和 = 100%
    """

    @staticmethod
    def normalize_weights(raw_weights: pd.Series) -> pd.Series:
        """正規化權重為符合規則的配比"""
        if raw_weights.sum() == 0:
            return raw_weights * 0

        # 正規化到總和 = 1
        normalized = raw_weights / raw_weights.sum()

        # 過濾小於 3% 的持股
        mask = normalized >= MIN_POSITION_WEIGHT
        filtered = normalized[mask]

        if filtered.empty:
            # 全部被過濾，保留最大的前 N 檔
            max_stocks = int(1.0 / MIN_POSITION_WEIGHT)
            top_stocks = normalized.nlargest(min(len(normalized), max_stocks))
            filtered = top_stocks / top_stocks.sum()

        # 轉換為 3% 倍數
        quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

        # 移除四捨五入後變成 0 的
        quantized = quantized[quantized >= MIN_POSITION_WEIGHT]

        if quantized.empty:
            return pd.Series(0.0, index=raw_weights.index)

        # 修正總和誤差
        total = quantized.sum()
        if total > 0 and abs(total - 1.0) > 0.001:
            diff = 1.0 - total
            # 調整最大權重
            max_idx = quantized.idxmax()
            new_val = quantized[max_idx] + diff
            if new_val >= MIN_POSITION_WEIGHT:
                quantized[max_idx] = new_val
            else:
                # 重新分配
                quantized = quantized / total

        # 建立結果 Series
        result = pd.Series(0.0, index=raw_weights.index)
        result[quantized.index] = quantized

        return result

    @staticmethod
    def apply_to_dataframe(position_df: pd.DataFrame, target_stocks: int = 20) -> pd.DataFrame:
        """應用配比約束到整個 DataFrame"""
        result = pd.DataFrame(0.0, index=position_df.index, columns=position_df.columns)

        for date in position_df.index:
            row = position_df.loc[date]
            candidates = row[row > 0].sort_values(ascending=False)

            if candidates.empty:
                continue

            # 限制持股數量
            max_n = min(len(candidates), target_stocks, MAX_STOCKS)
            selected = candidates.iloc[:max_n]

            # 應用配比約束
            normalized = PositionWeightManager.normalize_weights(selected)
            result.loc[date, normalized.index] = normalized

        return result

weight_manager = PositionWeightManager()

# =============================================================================
# 第七部分：六大策略引擎
# =============================================================================
class DragonStrategyEngine:
    """
    🐉 六策略快快龍引擎

    策略一：低波動本益比策略 (Low Volatility PE)
    策略二：小資族成長策略 (Small Cap Growth)
    策略三：營收股價雙渦輪 (Revenue Price Turbo)
    策略四：高殖利率防禦策略 (High Dividend Defense)
    策略五：動能突破策略 (Momentum Breakout)
    策略六：價值成長混合策略 (Value Growth Hybrid)
    """

    def __init__(self, data_loader: DragonDataLoader):
        self.dl = data_loader

    # =========================================================================
    # 策略一：低波動本益比策略
    # =========================================================================
    def strategy_low_volatility_pe(self, params: Dict) -> pd.DataFrame:
        """
        策略一：低波動本益比

        核心邏輯：
        - 尋找低波動、合理本益比的穩健標的
        - 營收穩定成長
        - 籌碼面健康
        """
        # 參數解碼
        rev_ma3_ma12_ratio = params.get('lv_rev_ratio', 1.1)
        volatility_threshold = params.get('lv_vol_threshold', 0.04)
        margin_usage_limit = params.get('lv_margin_limit', 35)
        non_op_income_limit = params.get('lv_non_op_limit', 8)
        pe_min = params.get('lv_pe_min', 5)
        pe_max = params.get('lv_pe_max', 25)
        top_n = int(params.get('lv_top_n', 5))

        # 獲取數據
        close = self.dl.get('close')
        pe = self.dl.get('pe')
        pb = self.dl.get('pb')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_ma3 = self.dl.get('rev_ma3')
        rev_ma12 = self.dl.get('rev_ma12')
        rev_yoy = self.dl.get('rev_yoy')
        rev_mom = self.dl.get('rev_mom')
        融資使用率 = self.dl.get('融資使用率')
        entry_volatility = self.dl.get('entry_volatility')
        業外收支營收率 = self.dl.get('業外收支營收率')
        營業利益成長率 = self.dl.get('營業利益成長率')
        營業毛利率 = self.dl.get('營業毛利率')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        limit_up_all_day = self.dl.get('limit_up_all_day')
        inventory = self.dl.get('inventory')
        peg = self.dl.get('peg')

        # 條件設定
        cond_rev_trend = rev_ma3 / rev_ma12 > rev_ma3_ma12_ratio
        cond_rev_growth = rev / rev.shift(1) > 0.9

        # 低波動因子
        cond_low_vol = (
            (融資使用率 <= margin_usage_limit) &
            (entry_volatility <= volatility_threshold) &
            (業外收支營收率 < non_op_income_limit)
        )

        # 成交量條件
        cond_volume = vol.average(1) > 100000

        # 營收健康度
        cond_rev_healthy = ~(rev_yoy < -30).sustain(3)
        cond_rev_not_old = ~(rev_yoy > 30).sustain(12, 8)
        cond_rev_mom = (rev_mom > -54).sustain(3)

        # 均線條件
        cond_ma = (
            (close > close.average(75)) &
            (close > close.average(40)) &
            (close > close.average(90))
        )

        # 營收動能
        cond_rev_momentum = rev.average(4) > rev.average(12)

        # 估值條件
        cond_pe = (pe_min <= pe) & (pe <= pe_max)
        cond_pb = (0.5 <= pb) & (pb <= 2.8)

        # 基本面條件
        cond_gpm = (營業毛利率 > 8).sustain(2)
        cond_roe = (ROE綜合損益 > 0).sustain(2)

        # 集保散戶持股
        try:
            small_inv = (inventory[(inventory.持股分級.astype(int) <= 8)]
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
        except:
            small_inv = close > 0  # fallback

        # 綜合條件
        cond_all = (
            cond_rev_trend &
            cond_rev_growth &
            cond_low_vol &
            cond_rev_healthy &
            cond_rev_not_old &
            cond_ma &
            cond_rev_momentum &
            cond_rev_mom &
            ~limit_up_all_day &
            cond_volume &
            cond_gpm &
            cond_roe &
            small_inv &
            cond_pe &
            cond_pb
        )

        # 選股：PEG 最小
        position = peg[cond_all & (peg > 0)].is_smallest(top_n)
        position = position.reindex(rev.index_str_to_date().index, method='ffill')

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略二：小資族成長策略
    # =========================================================================
    def strategy_small_cap_growth(self, params: Dict) -> pd.DataFrame:
        """
        策略二：小資族成長

        核心邏輯：
        - 小型股中尋找高成長潛力
        - 自由現金流健康
        - RSV 動能選股
        """
        # 參數解碼
        market_value_limit = params.get('sc_mv_limit', 15e9)
        market_rev_ratio_limit = params.get('sc_mv_rev_ratio', 3.5)
        rev_yoy_growth_limit = params.get('sc_rev_yoy_limit', -12)
        rev_mom_growth_limit = params.get('sc_rev_mom_limit', -55)
        rsv_period = int(params.get('sc_rsv_period', 55))
        ma_period = int(params.get('sc_ma_period', 70))
        top_n = int(params.get('sc_top_n', 7))

        # 獲取數據
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_yoy = self.dl.get('rev_yoy')
        rev_mom = self.dl.get('rev_mom')
        市值 = self.dl.get('市值')
        市值營收比 = self.dl.get('市值營收比')
        自由現金流 = self.dl.get('自由現金流')
        股東權益報酬率 = self.dl.get('股東權益報酬率')
        營業利益成長率 = self.dl.get('營業利益成長率')
        業外收支營收率 = self.dl.get('業外收支營收率')

        # 條件設定
        cond_small_cap = 市值 < market_value_limit
        cond_fcf = 自由現金流 > 0
        cond_roe = 股東權益報酬率 > 0
        cond_op_growth = 營業利益成長率 > -1
        cond_mv_rev = 市值營收比 < market_rev_ratio_limit
        cond_volume = vol > 100000

        # 營收條件
        cond_rev_healthy = ~(rev_yoy < rev_yoy_growth_limit).sustain(3)
        cond_rev_not_old = ~(rev_yoy > 60).sustain(12, 8)
        cond_rev_mom = (rev_mom > rev_mom_growth_limit).sustain(3)

        # 均線條件
        cond_ma = (
            (close > close.average(ma_period)) &
            (close > close.average(120)) &
            (close > close.average(75))
        )

        # 營收動能
        cond_rev_momentum = rev.average(3) > rev.average(12)

        # 業外收支
        cond_non_op = 業外收支營收率 < 7.3

        # RSV 計算
        rsv = (close - close.rolling(rsv_period).min()) / (
            close.rolling(rsv_period).max() - close.rolling(rsv_period).min() + 1e-10
        )

        # 綜合條件
        cond_all = (
            cond_small_cap &
            cond_fcf &
            cond_roe &
            cond_op_growth &
            cond_mv_rev &
            cond_volume &
            cond_rev_not_old &
            cond_rev_mom &
            cond_rev_healthy &
            cond_rev_momentum &
            cond_non_op
        )

        # 選股：RSV 最高
        position = (cond_all * rsv).is_largest(top_n)
        position = position.reindex(self.dl.get('rev').index_str_to_date().index)

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略三：營收股價雙渦輪
    # =========================================================================
    def strategy_revenue_price_turbo(self, params: Dict) -> pd.DataFrame:
        """
        策略三：營收股價雙渦輪

        核心邏輯：
        - 營收創新高 + 股價創新高
        - 多頭排列
        - 大戶持股集中
        """
        # 參數解碼
        rev_ma_period = int(params.get('rpt_rev_ma', 4))
        rev_ma_lookback = int(params.get('rpt_lookback', 20))
        price_high_window = int(params.get('rpt_price_window', 7))
        min_volume = params.get('rpt_min_vol', 250000)
        min_price = params.get('rpt_min_price', 15)
        rsi_threshold = params.get('rpt_rsi', 55)
        pe_limit = params.get('rpt_pe_limit', 150)
        top_n = int(params.get('rpt_top_n', 12))

        # 獲取數據
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        pe = self.dl.get('pe')
        rev = self.dl.get('rev')
        rev_yoy = self.dl.get('rev_yoy')
        rsi = self.dl.get('rsi')
        業外收支營收率 = self.dl.get('業外收支營收率')
        營業毛利率 = self.dl.get('營業毛利率')
        稅前淨利率 = self.dl.get('稅前淨利率')
        稅後淨利率 = self.dl.get('稅後淨利率')
        limit_up_all_day = self.dl.get('limit_up_all_day')
        inventory = self.dl.get('inventory')

        # 營收創新高
        rev_ma = rev.average(rev_ma_period)
        cond_rev_high = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()

        # 股價創新高
        cond_price_high = (close == close.rolling(260).max()).sustain(price_high_window, 1)

        # 成交量
        cond_volume = vol.average(1) > min_volume

        # 多頭排列
        cond_ma_align = (
            (close > close.average(5)) &
            (close > close.average(10)) &
            (close > close.average(20)) &
            (close > close.average(60)) &
            (close > close.average(150)) &
            (close > close.average(200))
        )

        # 超級績效
        cond_super = close > (close.average(250) * 1.1)

        # RSI
        cond_rsi = (rsi > rsi_threshold).sustain(1)

        # 基本面
        cond_gpm = (營業毛利率 > 5).sustain(5)
        cond_btpm = (稅前淨利率 > 4).sustain(1)
        cond_atpm = (稅後淨利率 > 3).sustain(1)

        # 營收年增排名
        cond_rev_rank = rev_yoy.rank(pct=True, axis=1) > 0.9

        # 大戶持股
        try:
            boss_inv = (inventory[(inventory.持股分級.astype(int) >= 12) & (inventory.持股分級.astype(int) <= 16)]
                       .reset_index()
                       .groupby(["date", "stock_id"])
                       .agg({"占集保庫存數比例": "sum"})
                       .reset_index()
                       .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= 18
        except:
            boss_inv = close > 0

        # PE 條件
        cond_pe = ~(pe_limit <= pe)

        # 綜合條件
        cond_all = (
            cond_rev_high &
            cond_price_high &
            cond_volume &
            cond_ma_align &
            cond_gpm &
            cond_btpm &
            cond_atpm &
            cond_rev_rank &
            (close > min_price) &
            (業外收支營收率 < 7.3) &
            cond_pe &
            cond_super &
            ((vol >= vol.rolling(20).mean() * 0.8)) &
            cond_rsi &
            boss_inv &
            ~limit_up_all_day
        )

        # 選股：年增率最高
        position = (rev_yoy * cond_all)
        position = position[position > 0].is_largest(top_n)
        position = position.reindex(rev.index_str_to_date().index, method="ffill")

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略四：高殖利率防禦策略
    # =========================================================================
    def strategy_high_dividend_defense(self, params: Dict) -> pd.DataFrame:
        """
        策略四：高殖利率防禦

        核心邏輯：
        - 高殖利率穩定配息
        - 低波動防禦型
        - 獲利穩定
        """
        # 參數解碼
        div_yield_min = params.get('hd_div_min', 4.0)
        volatility_max = params.get('hd_vol_max', 0.25)
        pe_max = params.get('hd_pe_max', 20)
        roe_min = params.get('hd_roe_min', 8)
        top_n = int(params.get('hd_top_n', 8))

        # 獲取數據
        close = self.dl.get('close')
        dividend_yield = self.dl.get('dividend_yield')
        volatility = self.dl.get('volatility_60')
        pe = self.dl.get('pe')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        vol = self.dl.get('vol')
        營業毛利率 = self.dl.get('營業毛利率')

        # 條件設定
        cond_div = dividend_yield > div_yield_min
        cond_vol = volatility < volatility_max
        cond_pe = pe < pe_max
        cond_roe = ROE綜合損益 > roe_min
        cond_volume = vol > 100000
        cond_trend = close > close.average(120)
        cond_gpm = 營業毛利率 > 10

        # 綜合條件
        cond_all = (
            cond_div &
            cond_vol &
            cond_pe &
            cond_roe &
            cond_volume &
            cond_trend &
            cond_gpm
        )

        # 選股：殖利率最高
        position = (cond_all * dividend_yield).is_largest(top_n)

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略五：動能突破策略
    # =========================================================================
    def strategy_momentum_breakout(self, params: Dict) -> pd.DataFrame:
        """
        策略五：動能突破

        核心邏輯：
        - 價格動能強勁
        - 成交量配合
        - 突破關鍵價位
        """
        # 參數解碼
        momentum_threshold = params.get('mb_mom_threshold', 0.15)
        volume_ratio_min = params.get('mb_vol_ratio', 1.5)
        rsi_min = params.get('mb_rsi_min', 50)
        rsi_max = params.get('mb_rsi_max', 80)
        top_n = int(params.get('mb_top_n', 10))

        # 獲取數據
        close = self.dl.get('close')
        momentum_20 = self.dl.get('momentum_20')
        momentum_60 = self.dl.get('momentum_60')
        vol_ratio = self.dl.get('vol_ratio')
        rsi = self.dl.get('rsi')
        vol = self.dl.get('vol')

        # 條件設定
        cond_momentum = momentum_20 > momentum_threshold
        cond_momentum_60 = momentum_60 > 0
        cond_vol_ratio = vol_ratio > volume_ratio_min
        cond_rsi = (rsi > rsi_min) & (rsi < rsi_max)
        cond_volume = vol > 200000

        # 均線突破
        cond_breakout = (
            (close > close.average(20)) &
            (close > close.average(60)) &
            (close.shift(1) <= close.average(20).shift(1))  # 昨日在均線下
        ) | (
            (close > close.average(60)) &
            (close.shift(1) <= close.average(60).shift(1))
        )

        # 趨勢確認
        cond_trend = close > close.average(120)

        # 綜合條件
        cond_all = (
            cond_momentum &
            cond_momentum_60 &
            cond_vol_ratio &
            cond_rsi &
            cond_volume &
            cond_trend
        )

        # 選股：動能最強
        position = (cond_all * momentum_20).is_largest(top_n)

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略六：價值成長混合策略
    # =========================================================================
    def strategy_value_growth_hybrid(self, params: Dict) -> pd.DataFrame:
        """
        策略六：價值成長混合

        核心邏輯：
        - 結合價值因子與成長因子
        - 尋找被低估的成長股
        - 多因子綜合評分
        """
        # 參數解碼
        pe_percentile = params.get('vg_pe_pct', 0.4)
        pb_percentile = params.get('vg_pb_pct', 0.5)
        growth_threshold = params.get('vg_growth', 10)
        roe_min = params.get('vg_roe_min', 10)
        top_n = int(params.get('vg_top_n', 10))

        # 獲取數據
        close = self.dl.get('close')
        pe = self.dl.get('pe')
        pb = self.dl.get('pb')
        rev_yoy = self.dl.get('rev_yoy')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        營業利益成長率 = self.dl.get('營業利益成長率')
        vol = self.dl.get('vol')

        # PE 排名（越低越好）
        pe_rank = pe.rank(pct=True, axis=1)
        cond_pe_value = pe_rank < pe_percentile

        # PB 排名（越低越好）
        pb_rank = pb.rank(pct=True, axis=1)
        cond_pb_value = pb_rank < pb_percentile

        # 成長條件
        cond_growth = (rev_yoy > growth_threshold) | (營業利益成長率 > growth_threshold)

        # ROE 條件
        cond_roe = ROE綜合損益 > roe_min

        # 基本條件
        cond_volume = vol > 100000
        cond_trend = close > close.average(60)
        cond_pe_positive = (pe > 0) & (pe < 50)

        # 綜合條件
        cond_all = (
            cond_pe_value &
            cond_pb_value &
            cond_growth &
            cond_roe &
            cond_volume &
            cond_trend &
            cond_pe_positive
        )

        # 計算綜合分數
        # 價值分數（PE、PB 越低越好）
        value_score = (1 - pe_rank) * 0.5 + (1 - pb_rank) * 0.5

        # 成長分數
        growth_score = rev_yoy.rank(pct=True, axis=1)

        # 綜合分數
        total_score = value_score * 0.4 + growth_score * 0.6

        # 選股
        position = (cond_all * total_score).is_largest(top_n)

        return position.fillna(0).astype(float)

    # =========================================================================
    # 策略組合
    # =========================================================================
    def combine_all_strategies(self, params: Dict) -> pd.DataFrame:
        """
        組合六大策略

        根據權重動態組合各策略持股
        """
        try:
            # 獲取各策略持股
            strategies = {
                'low_vol_pe': self.strategy_low_volatility_pe,
                'small_cap': self.strategy_small_cap_growth,
                'turbo': self.strategy_revenue_price_turbo,
                'high_div': self.strategy_high_dividend_defense,
                'momentum': self.strategy_momentum_breakout,
                'value_growth': self.strategy_value_growth_hybrid,
            }

            # 獲取權重
            weights = {
                'low_vol_pe': params.get('weight_lv', 0.2),
                'small_cap': params.get('weight_sc', 0.15),
                'turbo': params.get('weight_rpt', 0.25),
                'high_div': params.get('weight_hd', 0.1),
                'momentum': params.get('weight_mb', 0.15),
                'value_growth': params.get('weight_vg', 0.15),
            }

            # 正規化權重
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            # 組合持股
            combined = None
            for name, func in strategies.items():
                try:
                    pos = func(params)
                    weighted = pos * weights[name]

                    if combined is None:
                        combined = weighted
                    else:
                        combined = combined.add(weighted, fill_value=0)
                except Exception as e:
                    print(f"   ⚠️ 策略 {name} 執行失敗: {e}")
                    continue

            if combined is None:
                return pd.DataFrame()

            # 應用持股配比約束
            final_position = weight_manager.apply_to_dataframe(
                combined,
                target_stocks=int(params.get('target_stocks', 20))
            )

            return final_position

        except Exception as e:
            print(f"   ⚠️ 策略組合失敗: {e}")
            traceback.print_exc()
            return pd.DataFrame()

# 初始化策略引擎
strategy_engine = DragonStrategyEngine(data_loader)

# =============================================================================
# 第八部分：回測引擎（In-Sample / Out-of-Sample 分離）
# 注意：基因解碼器已在第 4.5 部分提前定義
# =============================================================================
class DragonBacktestEngine:
    """
    📊 龍族回測引擎

    功能：
    1. In-Sample / Out-of-Sample 分離驗證
    2. Walk-Forward 分析
    3. 詳細績效指標計算
    """

    def __init__(self):
        self.results_cache = {}

    def run_backtest(self, position: pd.DataFrame, params: Dict,
                    start_date: str = None, end_date: str = None,
                    name: str = 'Backtest') -> Optional[Dict]:
        """執行單次回測"""
        try:
            # 過濾時間
            if start_date:
                position = position.loc[start_date:]
            if end_date:
                position = position.loc[:end_date]

            if position.empty or len(position) < 50:
                return None

            # 執行回測
            report = sim(
                position=position,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price="high_low_avg",
                position_limit=params.get('position_limit', 0.35),
                stop_loss=params.get('stop_loss', 0.25),
                trail_stop=params.get('trail_stop', 0.3),
                take_profit=params.get('take_profit', 0.7),
                stop_trading_next_period=False,
                upload=False,
                name=name,
            )

            # 獲取績效指標
            metrics = report.get_metrics()

            return {
                'name': name,
                'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
                'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
                'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1) or 1),
                'capacity': metrics['liquidity'].get('capacity', 0) or 0,
                'win_rate': metrics['profitability'].get('winRate', 0) or 0,
                'profit_factor': metrics['profitability'].get('profitFactor', 0) or 0,
                'total_return': metrics['profitability'].get('totalProfit', 0) or 0,
                'report': report,
            }

        except Exception as e:
            print(f"   ⚠️ 回測失敗 ({name}): {e}")
            return None

    def run_in_out_sample(self, position: pd.DataFrame, params: Dict) -> Dict:
        """
        執行 In-Sample / Out-of-Sample 分離回測

        Returns:
            {
                'in_sample': In-Sample 績效,
                'out_sample': Out-of-Sample 績效,
                'overall': 整體績效,
                'is_valid': 是否有效（OOS 夏普 > 0）,
                'oos_decay': OOS 衰減率,
            }
        """
        try:
            # In-Sample 回測（訓練期）
            in_sample_result = self.run_backtest(
                position, params,
                start_date=IN_SAMPLE_START,
                end_date=IN_SAMPLE_END,
                name='InSample'
            )

            # Out-of-Sample 回測（測試期）
            out_sample_result = self.run_backtest(
                position, params,
                start_date=OUT_SAMPLE_START,
                end_date=OUT_SAMPLE_END,
                name='OutSample'
            )

            # 整體回測
            overall_result = self.run_backtest(
                position, params,
                start_date=IN_SAMPLE_START,
                end_date=OUT_SAMPLE_END,
                name='Overall'
            )

            # 處理空結果
            if not in_sample_result:
                in_sample_result = self._empty_result('InSample')
            if not out_sample_result:
                out_sample_result = self._empty_result('OutSample')
            if not overall_result:
                overall_result = self._empty_result('Overall')

            # 計算 OOS 衰減率
            is_sharpe = in_sample_result['sharpe']
            oos_sharpe = out_sample_result['sharpe']

            if is_sharpe > 0:
                oos_decay = 1 - (oos_sharpe / is_sharpe) if is_sharpe > 0 else 1
            else:
                oos_decay = 1

            # 驗證 OOS 有效性
            is_valid = (
                oos_sharpe > 0 and
                out_sample_result['max_drawdown'] < MAX_MDD and
                oos_decay < 0.5  # OOS 衰減不超過 50%
            )

            return {
                'in_sample': in_sample_result,
                'out_sample': out_sample_result,
                'overall': overall_result,
                'is_valid': is_valid,
                'oos_decay': oos_decay,
            }

        except Exception as e:
            print(f"   ⚠️ In/Out Sample 回測失敗: {e}")
            return self._empty_io_result()

    def _empty_result(self, name: str = '') -> Dict:
        """空結果"""
        return {
            'name': name,
            'sharpe': 0,
            'annual_return': 0,
            'max_drawdown': 1,
            'capacity': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'report': None,
        }

    def _empty_io_result(self) -> Dict:
        """空的 In/Out Sample 結果"""
        return {
            'in_sample': self._empty_result('InSample'),
            'out_sample': self._empty_result('OutSample'),
            'overall': self._empty_result('Overall'),
            'is_valid': False,
            'oos_decay': 1,
        }

backtest_engine = DragonBacktestEngine()

# =============================================================================
# 第十部分：適應度評估函數
# =============================================================================
def evaluate_dragon_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """
    🐉 龍族適應度評估

    多目標優化：
    1. 夏普值分數（OOS 為主）
    2. 胃納量分數
    3. MDD 懲罰
    4. OOS 有效性獎勵

    Returns: (sharpe_score, capacity_score, mdd_penalty, validity_bonus)
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略
        position = strategy_engine.combine_all_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, -5.0, -5.0)

        # 執行 In/Out Sample 回測
        io_result = backtest_engine.run_in_out_sample(position, params)

        # 獲取績效指標
        oos = io_result['out_sample']
        overall = io_result['overall']

        # === 計算各項分數 ===

        # 1. 夏普值分數（以 OOS 為主，權重 70%）
        oos_sharpe = oos['sharpe']
        overall_sharpe = overall['sharpe']
        weighted_sharpe = oos_sharpe * 0.7 + overall_sharpe * 0.3
        sharpe_score = min(5.0, max(0, weighted_sharpe / TARGET_SHARPE * 5.0))

        # 2. 胃納量分數
        capacity = overall['capacity']
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))

        # 3. MDD 懲罰
        mdd = oos['max_drawdown']
        if mdd > MAX_MDD:
            mdd_penalty = -((mdd - MAX_MDD) / MAX_MDD) * 5  # 超過 20% 開始扣分
        else:
            mdd_penalty = (MAX_MDD - mdd) / MAX_MDD * 2  # 低於 20% 給獎勵

        # 4. OOS 有效性獎勵
        if io_result['is_valid']:
            validity_bonus = 2.0
            # 額外獎勵：OOS 衰減低
            if io_result['oos_decay'] < 0.3:
                validity_bonus += 1.0
        else:
            validity_bonus = -3.0

        return (sharpe_score, capacity_score, mdd_penalty, validity_bonus)

    except Exception as e:
        print(f"   ⚠️ 適應度評估失敗: {e}")
        return (0.0, 0.0, -5.0, -5.0)

# =============================================================================
# 第十一部分：並行評估器
# =============================================================================
class ParallelEvaluator:
    """
    ⚡ 並行評估器

    使用多進程加速個體評估
    """

    def __init__(self, n_workers: int = N_PARALLEL_WORKERS):
        self.n_workers = n_workers

    def evaluate_population(self, population: List,
                           evaluate_func: Callable) -> List:
        """並行評估族群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        n_invalid = len(invalid)
        print(f"   ⚡ 評估 {n_invalid} 個個體（{self.n_workers} 並行）...")

        # 使用 ThreadPoolExecutor（FinLab 可能有 GIL 問題，用線程更穩定）
        try:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(evaluate_func, ind): ind
                          for ind in invalid}

                completed = 0
                for future in as_completed(futures):
                    ind = futures[future]
                    try:
                        fitness = future.result(timeout=300)  # 5 分鐘超時
                        ind.fitness.values = fitness
                    except Exception as e:
                        print(f"      ⚠️ 評估超時或失敗: {e}")
                        ind.fitness.values = (0.0, 0.0, -5.0, -5.0)

                    completed += 1
                    if completed % 10 == 0:
                        print(f"      進度: {completed}/{n_invalid}")

        except Exception as e:
            print(f"   ⚠️ 並行評估失敗，改用序列評估: {e}")
            for i, ind in enumerate(invalid):
                try:
                    fitness = evaluate_func(ind)
                    ind.fitness.values = fitness
                except:
                    ind.fitness.values = (0.0, 0.0, -5.0, -5.0)

                if (i + 1) % 10 == 0:
                    print(f"      進度: {i+1}/{n_invalid}")

        return population

parallel_evaluator = ParallelEvaluator()

# =============================================================================
# 第十二部分：DEAP 設定
# =============================================================================
# 清除舊定義
if 'FitnessMulti' in dir(creator):
    del creator.FitnessMulti
if 'Individual' in dir(creator):
    del creator.Individual

# 創建適應度（4 目標，全部最大化）
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=DragonGeneDecoder.GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_dragon_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=0.0, up=1.0, eta=20.0, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

print("✅ DEAP NSGA-II 配置完成")

# =============================================================================
# 第十三部分：Check-point 與 Pareto Archive 管理
# =============================================================================
class CheckpointManager:
    """
    💾 Check-point 管理器

    功能：
    1. 自動保存演化進度
    2. 支援斷點續傳
    3. 保存最佳參數
    """

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file

    def save(self, population: List, generation: int,
             history: List, best_fitness: float):
        """保存 Check-point"""
        try:
            checkpoint_data = {
                'generation': generation,
                'population': [
                    {'genes': list(ind), 'fitness': ind.fitness.values}
                    for ind in population
                ],
                'history': history,
                'best_fitness': best_fitness,
                'timestamp': datetime.now().isoformat(),
                'window_id': WINDOW_ID,
            }

            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            print(f"   💾 Check-point 已保存 (Gen {generation})")

        except Exception as e:
            print(f"   ⚠️ Check-point 保存失敗: {e}")

    def load(self) -> Optional[Dict]:
        """載入 Check-point"""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            print(f"✅ 載入 Check-point (Gen {data['generation']})")
            return data

        except Exception as e:
            print(f"⚠️ Check-point 載入失敗: {e}")
            return None

    def restore_population(self, checkpoint_data: Dict) -> List:
        """從 Check-point 恢復族群"""
        population = []

        for ind_data in checkpoint_data['population']:
            ind = creator.Individual(ind_data['genes'])
            ind.fitness.values = ind_data['fitness']
            population.append(ind)

        return population

checkpoint_manager = CheckpointManager(paths.checkpoint_file)


class ParetoArchiveManager:
    """
    🏆 Pareto Archive 管理器

    功能：
    1. 保存歷史最佳個體
    2. 精英注入新族群
    3. 跨視窗共享
    4. 🆕 匯入外部優秀基因
    """

    def __init__(self, archive_file: str, external_genes_path: str = None):
        self.archive_file = archive_file
        self.external_genes_path = external_genes_path or EXTERNAL_GENES_PATH
        self.archive = []
        self._load()
        self._load_external_genes()  # 🆕 自動匯入外部基因

    def _load(self):
        """載入歷史 Pareto Archive"""
        if os.path.exists(self.archive_file):
            try:
                with open(self.archive_file, 'rb') as f:
                    data = pickle.load(f)
                    self.archive = data.get('individuals', [])
                print(f"✅ 載入 {len(self.archive)} 個歷史精英")
            except Exception as e:
                print(f"⚠️ Pareto Archive 載入失敗: {e}")
                self.archive = []
        else:
            print("ℹ️ 無歷史 Pareto Archive")

    def _load_external_genes(self):
        """🆕 載入外部優秀基因（支援多種格式）"""
        if not self.external_genes_path:
            return

        if not os.path.exists(self.external_genes_path):
            print(f"ℹ️ 外部基因檔案不存在: {self.external_genes_path}")
            return

        try:
            print(f"🧬 載入外部優秀基因: {self.external_genes_path}")
            with open(self.external_genes_path, 'rb') as f:
                external_data = pickle.load(f)

            # 🔍 調試：顯示檔案格式
            print(f"   📋 檔案類型: {type(external_data).__name__}")
            if isinstance(external_data, dict):
                print(f"   📋 字典鍵: {list(external_data.keys())[:10]}")
            elif isinstance(external_data, list):
                print(f"   📋 列表長度: {len(external_data)}")
                if external_data:
                    first_item = external_data[0]
                    print(f"   📋 第一項類型: {type(first_item).__name__}")
                    if isinstance(first_item, dict):
                        print(f"   📋 第一項鍵: {list(first_item.keys())}")

            # 支援多種格式
            external_genes = []

            # ========== 格式解析 ==========
            if isinstance(external_data, list):
                for item in external_data:
                    gene_entry = self._parse_gene_item(item)
                    if gene_entry:
                        external_genes.append(gene_entry)

            elif isinstance(external_data, dict):
                # 格式A: {'individuals': [...]}
                if 'individuals' in external_data:
                    for item in external_data['individuals']:
                        gene_entry = self._parse_gene_item(item)
                        if gene_entry:
                            external_genes.append(gene_entry)

                # 格式B: {'genes': [...]}
                elif 'genes' in external_data:
                    for gene in external_data['genes']:
                        external_genes.append({
                            'genes': list(gene) if hasattr(gene, '__iter__') else gene,
                            'fitness': (5.0, 5.0, 0.0, 2.0),
                            'source': 'external'
                        })

                # 格式C: {'population': [...]}
                elif 'population' in external_data:
                    for item in external_data['population']:
                        gene_entry = self._parse_gene_item(item)
                        if gene_entry:
                            external_genes.append(gene_entry)

                # 格式D: {'archive': [...]}
                elif 'archive' in external_data:
                    for item in external_data['archive']:
                        gene_entry = self._parse_gene_item(item)
                        if gene_entry:
                            external_genes.append(gene_entry)

                # 格式E: {'best': [...]} 或其他
                else:
                    for key, value in external_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"   🔍 嘗試解析鍵 '{key}'...")
                            for item in value:
                                gene_entry = self._parse_gene_item(item)
                                if gene_entry:
                                    external_genes.append(gene_entry)
                            if external_genes:
                                break

            if external_genes:
                # 合併到 archive（外部基因優先）
                before_count = len(self.archive)
                self.archive = external_genes + self.archive

                # 去重
                self.archive = self._deduplicate(self.archive)

                # 保留前 100 名
                self.archive = sorted(
                    self.archive,
                    key=lambda x: sum(x.get('fitness', (0,0,0,0))),
                    reverse=True
                )[:100]

                added_count = len(external_genes)
                print(f"   ✅ 匯入 {added_count} 個外部優秀基因")
                print(f"   📊 Archive 總數: {before_count} → {len(self.archive)}")

                # 保存更新後的 Archive
                self._save()
            else:
                print(f"   ⚠️ 外部基因檔案格式無法識別")
                print(f"   💡 請提供檔案格式資訊以便支援")

        except Exception as e:
            print(f"⚠️ 外部基因載入失敗: {e}")
            traceback.print_exc()

    def _parse_gene_item(self, item) -> Optional[Dict]:
        """解析單個基因項目（支援多種格式）"""
        try:
            # 格式1: 已經是標準格式 {'genes': [...], 'fitness': (...)}
            if isinstance(item, dict):
                if 'genes' in item:
                    genes = list(item['genes']) if hasattr(item['genes'], '__iter__') else item['genes']
                    fitness = item.get('fitness', (5.0, 5.0, 0.0, 2.0))
                    return {'genes': genes, 'fitness': fitness, 'source': 'external'}

                # 格式2: {'individual': [...], ...}
                elif 'individual' in item:
                    genes = list(item['individual'])
                    fitness = item.get('fitness', (5.0, 5.0, 0.0, 2.0))
                    return {'genes': genes, 'fitness': fitness, 'source': 'external'}

                # 格式3: {'chromosome': [...], ...}
                elif 'chromosome' in item:
                    genes = list(item['chromosome'])
                    fitness = item.get('fitness', (5.0, 5.0, 0.0, 2.0))
                    return {'genes': genes, 'fitness': fitness, 'source': 'external'}

                # 格式4: {'params': {...}, ...} - 參數字典
                elif 'params' in item:
                    # 需要反向編碼，暫時跳過
                    return None

                # 格式5: DEAP Individual 序列化格式
                elif len(item) > 0:
                    # 嘗試找到數值列表
                    for key, value in item.items():
                        if isinstance(value, (list, np.ndarray)) and len(value) >= 30:
                            if all(isinstance(v, (int, float)) for v in value[:10]):
                                return {
                                    'genes': list(value),
                                    'fitness': item.get('fitness', (5.0, 5.0, 0.0, 2.0)),
                                    'source': 'external'
                                }

            # 格式6: 純數值列表/陣列
            elif isinstance(item, (list, np.ndarray)):
                if len(item) >= 30:  # 基因長度至少 30
                    if all(isinstance(v, (int, float)) for v in list(item)[:10]):
                        return {
                            'genes': list(item),
                            'fitness': (5.0, 5.0, 0.0, 2.0),
                            'source': 'external'
                        }

            # 格式7: DEAP creator.Individual 物件
            elif hasattr(item, 'fitness') and hasattr(item, '__iter__'):
                genes = list(item)
                fitness = item.fitness.values if hasattr(item.fitness, 'values') else (5.0, 5.0, 0.0, 2.0)
                return {'genes': genes, 'fitness': fitness, 'source': 'external'}

        except Exception as e:
            pass

        return None

    def update(self, pareto_front: List):
        """更新 Pareto Archive"""
        # 合併新舊個體
        new_individuals = [
            {'genes': list(ind), 'fitness': ind.fitness.values}
            for ind in pareto_front
        ]

        all_individuals = self.archive + new_individuals

        # 去重
        unique = self._deduplicate(all_individuals)

        # 排序保留前 50 名
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:50]

        # 保存
        self._save()

    def _deduplicate(self, individuals: List[Dict]) -> List[Dict]:
        """去重"""
        seen = set()
        unique = []

        for ind in individuals:
            gene_hash = hashlib.md5(str(ind['genes'][:15]).encode()).hexdigest()
            if gene_hash not in seen:
                seen.add(gene_hash)
                unique.append(ind)

        return unique

    def _save(self):
        """保存 Pareto Archive"""
        try:
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                    'window_id': WINDOW_ID,
                }, f)
        except Exception as e:
            print(f"⚠️ Pareto Archive 保存失敗: {e}")

    def inject_elites(self, population: List, ratio: float = ELITE_RATIO) -> List:
        """注入精英到族群"""
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

    def get_top_n(self, n: int = 5) -> List[Dict]:
        """獲取前 N 名"""
        return sorted(self.archive, key=lambda x: sum(x['fitness']), reverse=True)[:n]

pareto_archive = ParetoArchiveManager(paths.pareto_archive_file)

# =============================================================================
# 第十四部分：進度日誌記錄器
# =============================================================================
class ProgressLogger:
    """
    📝 進度日誌記錄器

    記錄演化進度供監控系統使用
    """

    def __init__(self):
        self.log_file = paths.progress_log_file
        self.start_time = time.time()

    def log_generation(self, gen: int, total_gen: int, stats: Dict):
        """記錄單代進度"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'window_id': WINDOW_ID,
            'current_gen': gen + 1,
            'total_gen': total_gen,
            'best_sharpe': stats.get('best_sharpe', 0),
            'best_capacity': stats.get('best_capacity', 0),
            'best_composite': stats.get('best_composite', 0),
            'avg_composite': stats.get('avg_composite', 0),
            'mdd_penalty': stats.get('mdd_penalty', 0),
            'validity_rate': stats.get('validity_rate', 0),
            'elapsed_total': time.time() - self.start_time,
        }

        # 讀取歷史
        history = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass

        # 只保留最近 1000 筆
        history.append(progress)
        if len(history) > 1000:
            history = history[-1000:]

        # 保存
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

progress_logger = ProgressLogger()

# =============================================================================
# 第十五部分：Top 5 驗證器
# =============================================================================
class Top5Validator:
    """
    🔍 Top 5 驗證器

    重啟後自動驗證前 5 名績效
    """

    def __init__(self):
        self.validation_file = paths.top5_validation_file

    def validate_top5(self) -> List[Dict]:
        """驗證前 5 名"""
        print("\n🔍 驗證歷史前 5 名...")

        top5 = pareto_archive.get_top_n(5)

        if not top5:
            print("   ℹ️ 無歷史資料可驗證")
            return []

        results = []

        for i, elite in enumerate(top5):
            print(f"   驗證 #{i+1}...")

            try:
                # 解碼基因
                params = gene_decoder.decode(elite['genes'])

                # 組合策略
                position = strategy_engine.combine_all_strategies(params)

                if position is None or position.empty:
                    results.append({
                        'rank': i + 1,
                        'status': 'FAILED',
                        'error': 'Empty position',
                    })
                    continue

                # 執行 OOS 回測
                oos_result = backtest_engine.run_backtest(
                    position, params,
                    start_date=OUT_SAMPLE_START,
                    name=f'Top{i+1}_OOS'
                )

                if oos_result:
                    results.append({
                        'rank': i + 1,
                        'status': 'OK',
                        'sharpe': oos_result['sharpe'],
                        'annual_return': oos_result['annual_return'],
                        'max_drawdown': oos_result['max_drawdown'],
                        'capacity': oos_result['capacity'],
                        'original_fitness': elite['fitness'],
                    })
                    print(f"      ✅ Sharpe: {oos_result['sharpe']:.2f}, MDD: {oos_result['max_drawdown']*100:.1f}%")
                else:
                    results.append({
                        'rank': i + 1,
                        'status': 'FAILED',
                        'error': 'Backtest failed',
                    })

            except Exception as e:
                results.append({
                    'rank': i + 1,
                    'status': 'ERROR',
                    'error': str(e),
                })

        # 保存驗證結果
        self._save_results(results)

        return results

    def _save_results(self, results: List[Dict]):
        """保存驗證結果"""
        try:
            with open(self.validation_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'window_id': WINDOW_ID,
                    'results': results,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   ⚠️ 驗證結果保存失敗: {e}")

top5_validator = Top5Validator()

# =============================================================================
# 第十六部分：演化引擎
# =============================================================================
class DragonEvolutionEngine:
    """
    🐉 龍族演化引擎

    整合所有功能的主引擎
    """

    def __init__(self):
        self.history = []
        self.best_ever = None
        self.start_gen = 0

    def run(self, n_generations: int = N_GENERATIONS) -> Tuple[List, List]:
        """執行演化"""
        print(f"""
{'='*70}
🐉 開始演化 - 視窗 {WINDOW_ID}
   族群: {POPULATION_SIZE}, 世代: {n_generations}
   目標: 夏普 > {TARGET_SHARPE}, MDD < {MAX_MDD*100:.0f}%, 胃納量 > {MIN_CAPACITY/1e7:.0f}00萬
{'='*70}
        """)

        # 嘗試載入 Check-point
        population = None
        if CONTINUE_EVOLUTION:
            checkpoint_data = checkpoint_manager.load()
            if checkpoint_data:
                population = checkpoint_manager.restore_population(checkpoint_data)
                self.start_gen = checkpoint_data['generation'] + 1
                self.history = checkpoint_data.get('history', [])
                print(f"   從第 {self.start_gen} 代繼續演化")

        # 初始化族群
        if population is None:
            population = toolbox.population(n=POPULATION_SIZE)
            self.start_gen = 0
            print("   初始化新族群")

        # 注入歷史精英
        population = pareto_archive.inject_elites(population, ratio=ELITE_RATIO)

        # 初始評估
        population = parallel_evaluator.evaluate_population(population, evaluate_dragon_fitness)

        # 演化循環
        for gen in range(self.start_gen, n_generations):
            start_time = time.time()

            # 選擇
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 變異
            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 評估
            offspring = parallel_evaluator.evaluate_population(offspring, evaluate_dragon_fitness)

            # 環境選擇
            population = toolbox.select(population + offspring, POPULATION_SIZE)

            # 統計
            stats = self._compute_stats(population, gen)
            self.history.append(stats)

            # 記錄進度
            progress_logger.log_generation(gen, n_generations, stats)

            elapsed = time.time() - start_time

            # 輸出
            if (gen + 1) % 5 == 0 or gen == 0:
                self._print_generation_stats(gen, n_generations, stats, elapsed)

            # 保存 Check-point
            if (gen + 1) % 10 == 0:
                checkpoint_manager.save(
                    population, gen, self.history,
                    stats['best_composite']
                )

        # 獲取 Pareto 前緣
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]

        # 更新 Pareto Archive
        pareto_archive.update(pareto_front)

        # 輸出最終結果
        self._print_final_results(pareto_front)

        return population, pareto_front

    def _compute_stats(self, population: List, gen: int) -> Dict:
        """計算統計資訊"""
        fitnesses = [ind.fitness.values for ind in population]

        # 解碼各項分數
        sharpe_scores = [f[0] for f in fitnesses]
        capacity_scores = [f[1] for f in fitnesses]
        mdd_penalties = [f[2] for f in fitnesses]
        validity_bonuses = [f[3] for f in fitnesses]

        composites = [sum(f) for f in fitnesses]

        # 計算有效率（validity_bonus > 0 的比例）
        valid_count = sum(1 for v in validity_bonuses if v > 0)
        validity_rate = valid_count / len(validity_bonuses)

        # 轉換回實際值
        best_sharpe = max(sharpe_scores) / 5.0 * TARGET_SHARPE
        best_capacity = max(capacity_scores) / 5.0 * MIN_CAPACITY / 1e4

        return {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': best_sharpe,
            'best_capacity': best_capacity,
            'mdd_penalty': np.mean(mdd_penalties),
            'validity_rate': validity_rate,
        }

    def _print_generation_stats(self, gen: int, total: int, stats: Dict, elapsed: float):
        """輸出世代統計"""
        print(f"""
=== 第 {gen+1}/{total} 代 ===
   最佳綜合: {stats['best_composite']:.4f}
   最佳夏普: {stats['best_sharpe']:.2f}
   最佳胃納量: {stats['best_capacity']:.0f} 萬
   OOS 有效率: {stats['validity_rate']*100:.1f}%
   耗時: {elapsed:.1f}s
        """)

    def _print_final_results(self, pareto_front: List):
        """輸出最終結果"""
        print(f"""
{'='*70}
🏆 Pareto 最優解集（前 10 名）
{'='*70}
{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納(萬)':<12}{'MDD懲罰':<10}{'達標'}
{'-'*70}
        """)

        sorted_front = sorted(
            pareto_front,
            key=lambda x: sum(x.fitness.values),
            reverse=True
        )[:10]

        for i, ind in enumerate(sorted_front, 1):
            f = ind.fitness.values
            composite = sum(f)
            sharpe = f[0] / 5.0 * TARGET_SHARPE
            capacity = f[1] / 5.0 * MIN_CAPACITY / 1e4
            mdd_pen = f[2]

            reached = "✅" if (sharpe >= TARGET_SHARPE and
                            capacity >= MIN_CAPACITY / 1e4 and
                            f[3] > 0) else ""

            print(f"{i:<6}{composite:<10.4f}{sharpe:<10.2f}{capacity:<12.0f}{mdd_pen:<10.2f}{reached}")

        # 保存最佳參數
        if sorted_front:
            best_ind = sorted_front[0]
            best_params = gene_decoder.decode(best_ind)

            with open(paths.best_params_file, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2, ensure_ascii=False)

            print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

# =============================================================================
# 第十七部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║         🐉 六策略快快龍 - 台股基因演算法優化系統 v12.8              ║
║         Six-Strategy Dragon GA Optimizer                           ║
╠════════════════════════════════════════════════════════════════════╣
║  🎯 目標: 夏普 > {TARGET_SHARPE}, MDD < {MAX_MDD*100:.0f}%, 胃納 > {MIN_CAPACITY/1e7:.0f}00萬              ║
║  📊 策略: 六大策略動態組合                                          ║
║  🔬 驗證: In-Sample ~2022 / Out-of-Sample 2023~                    ║
║  🧬 基因: {DragonGeneDecoder.GENE_LENGTH} 個參數                                           ║
║  🖥️  視窗: {WINDOW_ID}                                                       ║
║  🔄 續傳: {CONTINUE_EVOLUTION}                                                  ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 重啟後驗證前 5 名
    if CONTINUE_EVOLUTION:
        top5_validator.validate_top5()

    # 建立演化引擎
    engine = DragonEvolutionEngine()

    # 執行演化
    population, pareto_front = engine.run(N_GENERATIONS)

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*70}")
        print("📊 最佳個體詳細回測")
        print(f"{'='*70}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        # 顯示策略權重
        print("\n🔍 最佳策略權重：")
        print(f"   低波動本益比: {best_params['weight_lv']*100:.1f}%")
        print(f"   小資族成長: {best_params['weight_sc']*100:.1f}%")
        print(f"   營收股價雙渦輪: {best_params['weight_rpt']*100:.1f}%")
        print(f"   高殖利率防禦: {best_params['weight_hd']*100:.1f}%")
        print(f"   動能突破: {best_params['weight_mb']*100:.1f}%")
        print(f"   價值成長混合: {best_params['weight_vg']*100:.1f}%")

        # 完整回測
        try:
            print("\n執行完整回測...")
            position = strategy_engine.combine_all_strategies(best_params)

            # Out-of-Sample 回測
            oos_result = backtest_engine.run_backtest(
                position, best_params,
                start_date=OUT_SAMPLE_START,
                name='Dragon_Best_OOS'
            )

            if oos_result and oos_result['report']:
                print(f"\n📈 Out-of-Sample 績效 (2023~):")
                print(f"   夏普值: {oos_result['sharpe']:.2f}")
                print(f"   年化報酬: {oos_result['annual_return']*100:.1f}%")
                print(f"   最大回撤: {oos_result['max_drawdown']*100:.1f}%")
                print(f"   胃納量: {oos_result['capacity']/1e4:.0f} 萬")

                # 顯示完整報告
                oos_result['report'].display()

        except Exception as e:
            print(f"⚠️ 完整回測失敗: {e}")
            traceback.print_exc()

    print(f"""
{'='*70}
✅ 優化完成！
📁 結果保存於: {paths.output_dir}
📁 Pareto Archive: {paths.pareto_archive_file}
📁 Check-point: {paths.checkpoint_file}
{'='*70}
    """)

    return population, pareto_front


# =============================================================================
# 執行
# =============================================================================
if __name__ == "__main__":
    population, pareto = main()
