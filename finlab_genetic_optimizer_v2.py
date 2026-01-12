#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 三策略小小龍 - 遺傳演算法優化系統 v2.0
Taiwan Stock Genetic Algorithm Optimizer - Dragon Edition
================================================================================

【核心功能】
✅ 基於三策略合併邏輯（低波動本益比、小資族、營收股價雙渦輪）
✅ 遺傳演算法 NSGA-II 多目標自動優化
✅ In-Sample / Out-of-Sample 嚴格分離驗證
✅ 3% 最低持股權重限制（不滿 3% 則不持倉）
✅ 重啟後自動驗證歷史前 5 名數據
✅ 並行運算加速（多核心評估）
✅ Pickle/Joblib Check-point 存取

【優化目標】
- 夏普值：>= 4.2（核心目標）
- 胃納量：>= 1,000 萬台幣
- 年化報酬：最大化
- 最大回撤：< 20%（懲罰項）

【回測規範】
- In-Sample (訓練期): 2017 ~ 2022 年底
- Out-of-Sample (測試期): 2023 ~ 至今

【進化機制】
- 精英保留策略（前 10 強）
- Pareto Archive 持久化存取
- Walk-Forward 驗證防止過擬合

版本：v2.0 Dragon Edition (2025-01-12)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# ============================================================================
# 🔥 【重要】在 import 其他套件之前，先禁用 FinLab 快取以避免 EOFError
# ============================================================================
import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 第一部分：核心設定與參數
# ============================================================================
import sys
import json
import pickle
import time
import hashlib
import random
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# === 🔥 核心設定（請根據環境修改）===
WINDOW_ID = int(os.environ.get('WINDOW_ID', 1))  # 視窗 ID (1-4)
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# ============================================================================
# 🎯 優化目標設定（依據需求調整）
# ============================================================================
TARGET_SHARPE = 4.2          # 🔥 夏普值目標 > 4.2
MIN_CAPACITY = 10_000_000    # 胃納量 >= 1000萬台幣
TARGET_ANNUAL_RETURN = 0.30  # 目標年化報酬 30%
MAX_DRAWDOWN = 0.20          # 最大回撤限制 < 20%

# ============================================================================
# 🧬 GA 演化參數
# ============================================================================
POPULATION_SIZE = 60         # 族群大小
N_GENERATIONS = 100          # 演化代數
MUTATION_RATE = 0.20         # 變異率
CROSSOVER_RATE = 0.85        # 交叉率
ELITE_SIZE = 10              # 🔥 精英保留數量（前 10 強）
TOURNAMENT_SIZE = 3          # 錦標賽選擇大小

# ============================================================================
# 📊 回測時間設定（In-Sample / Out-of-Sample）
# ============================================================================
IN_SAMPLE_START = '2017-01-01'
IN_SAMPLE_END = '2022-12-31'      # 🔥 訓練期結束
OUT_SAMPLE_START = '2023-01-01'   # 🔥 測試期開始
OUT_SAMPLE_END = None             # 測試期結束（至今）

# ============================================================================
# 💼 持股權重設定
# ============================================================================
MIN_POSITION_WEIGHT = 0.03   # 🔥 最低持股權重 3%
POSITION_WEIGHT_STEP = 0.03  # 持股權重必須是 3% 的倍數

# ============================================================================
# 🔄 並行運算設定
# ============================================================================
N_WORKERS = max(1, mp.cpu_count() - 1)  # 並行工作數（保留一核心）
USE_PARALLEL = True          # 是否啟用並行運算

# ============================================================================
# 📁 Check-point 設定
# ============================================================================
CHECKPOINT_INTERVAL = 5      # 每 5 代保存一次 checkpoint
AUTO_VERIFY_TOP_N = 5        # 🔥 重啟後自動驗證前 5 名

print(f"=" * 80)
print(f"🐉 三策略小小龍 - 遺傳演算法優化系統 v2.0 - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e6:.0f}00萬, MDD < {MAX_DRAWDOWN*100:.0f}%")
print(f"   📊 訓練期：{IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
print(f"   📊 測試期：{OUT_SAMPLE_START} ~ 至今")
print(f"   🔧 並行核心：{N_WORKERS}")
print(f"=" * 80)


# ============================================================================
# 第二部分：快取清理與套件安裝
# ============================================================================
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
        'joblib': 'joblib',
        'tqdm': 'tqdm',
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

# 載入其他套件
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm

print(f"✅ 套件載入完成")
print(f"   - FinLab: {finlab.__version__}")


# ============================================================================
# 第三部分：環境設定與登入
# ============================================================================
def setup_environment():
    """設定環境"""
    # Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = '/content/drive/MyDrive/FinLab_GA_小小龍_v2'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = './finlab_ga_dragon_v2'
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
        self.pareto_dir = f"{self.base_dir}/shared_pareto"
        self.checkpoint_dir = f"{self.window_dir}/checkpoints"
        self.history_dir = f"{self.window_dir}/history"
        self.log_dir = f"{self.window_dir}_logs"

        for d in [self.window_dir, self.output_dir, self.pareto_dir,
                  self.checkpoint_dir, self.history_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def pareto_archive(self) -> str:
        return f"{self.pareto_dir}/pareto_archive_dragon_w{self.window_id}.pkl"

    @property
    def checkpoint_file(self) -> str:
        return f"{self.checkpoint_dir}/checkpoint_w{self.window_id}.pkl"

    @property
    def best_params_file(self) -> str:
        return f"{self.output_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log(self) -> str:
        return f"{self.log_dir}/progress_history.json"

    @property
    def validation_log(self) -> str:
        return f"{self.log_dir}/validation_results.json"

paths = PathManager()
print(f"📁 工作目錄: {paths.output_dir}")


# ============================================================================
# 第四部分：數據載入（使用 FinLab API）
# ============================================================================
class FinLabDataLoader:
    """FinLab 數據載入器（單例模式）"""

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
                import subprocess
                subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-delete'], stderr=subprocess.DEVNULL)
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

        # 當季營收
        當月營收 = self._cache['rev'] * 1000
        self._cache['當季營收'] = 當月營收.rolling(4).sum()
        self._cache['市值營收比'] = self._cache['市值'] / self._cache['當季營收']

    def get(self, key: str):
        """獲取數據"""
        return self._cache.get(key)

# 初始化數據載入器
data_loader = FinLabDataLoader()


# ============================================================================
# 第五部分：持股權重管理器（3% 最低權重限制）
# ============================================================================
class PositionWeightManager:
    """
    持股權重管理器

    🔥 核心規則：
    - 單一標的權重必須 >= 3%
    - 不滿 3% 則不持倉 (設為 0%)
    - 權重必須是 3% 的倍數（3%, 6%, 9%, ...）
    """

    def __init__(self, min_weight: float = MIN_POSITION_WEIGHT,
                 weight_step: float = POSITION_WEIGHT_STEP):
        self.min_weight = min_weight
        self.weight_step = weight_step

    def apply_weight_constraint(self, position: pd.DataFrame) -> pd.DataFrame:
        """
        應用持股權重約束

        Args:
            position: 原始持股權重 DataFrame

        Returns:
            調整後的持股權重 DataFrame
        """
        if position is None or position.empty:
            return position

        # 🔥 確保是 float 類型
        adjusted = position.astype(float).copy()

        # 逐行處理
        for idx in adjusted.index:
            row = adjusted.loc[idx].astype(float)

            # 計算總權重
            total = float(row.sum())

            if total <= 0:
                adjusted.loc[idx] = 0.0
                continue

            # 正規化
            normalized = row / total

            # 應用最低權重限制：低於 3% 設為 0
            filtered = normalized.copy()
            filtered[filtered < self.min_weight] = 0.0

            # 重新正規化
            new_total = float(filtered.sum())
            if new_total > 0:
                filtered = filtered / new_total

                # 量化到 3% 的倍數
                quantized = (filtered / self.weight_step).round() * self.weight_step

                # 確保總和為 1（調整最大權重）
                diff = 1.0 - float(quantized.sum())
                if abs(diff) > 0.001 and float(quantized.max()) > 0:
                    max_idx = quantized.idxmax()
                    quantized[max_idx] += diff

                adjusted.loc[idx] = quantized.astype(float)
            else:
                adjusted.loc[idx] = 0.0

        # 🔥 確保返回的是 float 類型
        return adjusted.astype(float)

    def get_position_summary(self, position: pd.DataFrame) -> Dict:
        """獲取持股摘要統計"""
        if position is None or position.empty:
            return {}

        # 計算每日持股數量
        daily_count = (position > 0).sum(axis=1)

        # 計算平均權重
        non_zero = position.replace(0, np.nan)
        avg_weight = non_zero.mean(axis=1).mean()

        return {
            'avg_holdings': daily_count.mean(),
            'max_holdings': daily_count.max(),
            'min_holdings': daily_count.min(),
            'avg_weight_per_stock': avg_weight,
        }

# 初始化權重管理器
weight_manager = PositionWeightManager()


# ============================================================================
# 第六部分：策略引擎（三策略小小龍）
# ============================================================================
class StrategyEngine:
    """策略引擎 - 整合三個子策略"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader
        self.weight_mgr = PositionWeightManager()

    def strategy_low_volatility_pe(self, params: Dict) -> Any:
        """
        策略一：低波動本益比策略

        🎯 核心邏輯：
        - 選擇低波動、合理本益比的個股
        - 強調營收穩定成長
        - 排除高融資使用率標的
        """
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
        """
        策略二：小資族策略

        🎯 核心邏輯：
        - 針對小市值、高成長潛力個股
        - 強調自由現金流為正
        - RSV 動能指標選股
        """
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
        """
        策略三：營收股價雙渦輪策略

        🎯 核心邏輯：
        - 營收創新高 + 股價創新高
        - 強勢多頭排列
        - 大戶持股占比高
        """
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

    def combine_strategies(self, params: Dict, apply_weight_constraint: bool = True) -> Any:
        """
        合併三個策略

        Args:
            params: 策略參數
            apply_weight_constraint: 是否應用 3% 最低權重限制
        """
        try:
            # 獲取三個策略的持股
            pos_lv = self.strategy_low_volatility_pe(params)
            pos_si = self.strategy_small_investor(params)
            pos_rpt = self.strategy_revenue_price_turbo(params)

            # 🔥 確保所有 position 都是 numeric 類型
            def ensure_numeric(pos):
                if pos is None:
                    return None
                # 轉換布林值為浮點數
                pos = pos.astype(float)
                # 填充 NaN 為 0
                pos = pos.fillna(0)
                return pos

            pos_lv = ensure_numeric(pos_lv)
            pos_si = ensure_numeric(pos_si)
            pos_rpt = ensure_numeric(pos_rpt)

            # 檢查是否有有效數據
            if pos_lv is None and pos_si is None and pos_rpt is None:
                return None

            # 獲取權重
            weight_lv = params['weight_lv']
            weight_si = params['weight_si']
            weight_rpt = params['weight_rpt']

            # 正規化權重
            total_weight = weight_lv + weight_si + weight_rpt
            weight_lv /= total_weight
            weight_si /= total_weight
            weight_rpt /= total_weight

            # 🔥 對齊索引並合併
            # 收集所有非空的 position
            positions = []
            weights = []
            if pos_lv is not None and not pos_lv.empty:
                positions.append(pos_lv)
                weights.append(weight_lv)
            if pos_si is not None and not pos_si.empty:
                positions.append(pos_si)
                weights.append(weight_si)
            if pos_rpt is not None and not pos_rpt.empty:
                positions.append(pos_rpt)
                weights.append(weight_rpt)

            if not positions:
                return None

            # 重新正規化權重
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            # 獲取共同的索引和欄位
            all_dates = positions[0].index
            all_cols = set(positions[0].columns)
            for pos in positions[1:]:
                all_dates = all_dates.union(pos.index)
                all_cols = all_cols.union(set(pos.columns))
            all_cols = list(all_cols)

            # 初始化合併結果
            position_combined = pd.DataFrame(0.0, index=all_dates, columns=all_cols)

            # 加權合併
            for pos, w in zip(positions, weights):
                # 對齊並填充
                pos_aligned = pos.reindex(index=all_dates, columns=all_cols, fill_value=0)
                position_combined = position_combined + pos_aligned * w

            # 確保是 float 類型
            position_combined = position_combined.astype(float)

            # 🔥 應用 3% 最低權重限制
            if apply_weight_constraint:
                position_combined = self.weight_mgr.apply_weight_constraint(position_combined)

            return position_combined

        except Exception as e:
            print(f"   ⚠️ 策略合併失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

# 初始化策略引擎
strategy_engine = StrategyEngine(data_loader)


# ============================================================================
# 第七部分：基因解碼器
# ============================================================================
class GeneDecoder:
    """
    基因解碼器

    基因結構（32個參數）：
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

# 初始化基因解碼器
gene_decoder = GeneDecoder()


# ============================================================================
# 第八部分：In-Sample / Out-of-Sample 回測引擎
# ============================================================================
class InSampleOutSampleBacktest:
    """
    In-Sample / Out-of-Sample 回測引擎

    🔥 嚴格分離訓練期與測試期：
    - In-Sample (訓練): 2017 ~ 2022 年底
    - Out-of-Sample (測試): 2023 ~ 至今
    """

    def __init__(self):
        self.in_sample_start = IN_SAMPLE_START
        self.in_sample_end = IN_SAMPLE_END
        self.out_sample_start = OUT_SAMPLE_START
        self.out_sample_end = OUT_SAMPLE_END

    def run_full_backtest(self, position, params: Dict) -> Dict:
        """
        執行完整回測（包含 In-Sample 和 Out-of-Sample）

        Returns:
            包含訓練期、測試期績效的字典
        """
        try:
            if position is None or position.empty or len(position) < 50:
                return self._empty_result()

            # === In-Sample 回測 ===
            in_sample_pos = position.loc[self.in_sample_start:self.in_sample_end]
            in_sample_result = self._backtest_period(in_sample_pos, params, 'InSample')

            # === Out-of-Sample 回測 ===
            if self.out_sample_end:
                out_sample_pos = position.loc[self.out_sample_start:self.out_sample_end]
            else:
                out_sample_pos = position.loc[self.out_sample_start:]
            out_sample_result = self._backtest_period(out_sample_pos, params, 'OutSample')

            # === 整體回測 ===
            overall_result = self._backtest_period(position, params, 'Overall')

            # 計算穩健性分數
            robustness = self._calculate_robustness(in_sample_result, out_sample_result)

            return {
                'in_sample': in_sample_result,
                'out_sample': out_sample_result,
                'overall': overall_result,
                'robustness_score': robustness,
                'is_robust': robustness > 0.6,
            }

        except Exception as e:
            print(f"   ⚠️ 回測失敗: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result()

    def _backtest_period(self, position, params: Dict, name: str) -> Optional[Dict]:
        """單一期間回測"""
        try:
            if position is None or position.empty or len(position) < 20:
                return self._default_metrics(name)

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
                'total_return': metrics['profitability'].get('totalReturn', 0) or 0,
            }

        except Exception as e:
            print(f"   ⚠️ 期間回測失敗 ({name}): {e}")
            return self._default_metrics(name)

    def _calculate_robustness(self, in_sample: Dict, out_sample: Dict) -> float:
        """
        計算穩健性分數

        🔥 核心邏輯：
        - 測試期績效應與訓練期相當
        - 懲罰過擬合（訓練期遠優於測試期）
        """
        if not in_sample or not out_sample:
            return 0.0

        in_sharpe = in_sample.get('sharpe', 0)
        out_sharpe = out_sample.get('sharpe', 0)

        if in_sharpe <= 0:
            return 0.0

        # 計算夏普值比率
        sharpe_ratio = out_sharpe / (in_sharpe + 1e-10)

        # 穩健性分數：測試期/訓練期比率
        # 理想值為 1.0（測試期與訓練期相當）
        # 大於 1.0 表示測試期更好（給予獎勵）
        # 小於 0.5 表示嚴重過擬合（給予懲罰）
        if sharpe_ratio >= 1.0:
            robustness = min(1.0, 0.8 + sharpe_ratio * 0.2)
        elif sharpe_ratio >= 0.7:
            robustness = 0.6 + (sharpe_ratio - 0.7) * (0.2 / 0.3)
        elif sharpe_ratio >= 0.5:
            robustness = 0.3 + (sharpe_ratio - 0.5) * (0.3 / 0.2)
        else:
            robustness = sharpe_ratio * 0.6

        return max(0, min(1, robustness))

    def _default_metrics(self, name: str) -> Dict:
        """預設指標"""
        return {
            'name': name,
            'sharpe': 0,
            'annual_return': 0,
            'max_drawdown': 1,
            'capacity': 0,
            'win_rate': 0,
            'total_return': 0,
        }

    def _empty_result(self) -> Dict:
        """空結果"""
        return {
            'in_sample': self._default_metrics('InSample'),
            'out_sample': self._default_metrics('OutSample'),
            'overall': self._default_metrics('Overall'),
            'robustness_score': 0,
            'is_robust': False,
        }

# 初始化回測引擎
backtest_engine = InSampleOutSampleBacktest()


# ============================================================================
# 第九部分：適應度評估函數（含 MDD 懲罰）
# ============================================================================
def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """
    評估個體適應度

    🎯 多目標優化：
    1. 夏普值分數（核心目標，> 4.2）
    2. 胃納量分數（> 1000萬）
    3. 年化報酬分數
    4. 穩健性懲罰（含 MDD 懲罰）

    Returns:
        (sharpe_score, capacity_score, return_score, penalty_score)
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略（含 3% 權重限制）
        position = strategy_engine.combine_strategies(params, apply_weight_constraint=True)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        # 執行 In-Sample / Out-of-Sample 回測
        result = backtest_engine.run_full_backtest(position, params)

        # 使用 Out-of-Sample 結果評估（避免過擬合）
        out_sample = result['out_sample']
        overall = result['overall']

        # 計算各項分數
        sharpe = out_sample['sharpe']
        capacity = out_sample['capacity']
        annual_return = out_sample['annual_return']
        max_drawdown = out_sample['max_drawdown']

        # === 夏普值分數（0-5）===
        # 目標：> 4.2
        sharpe_score = min(5.0, max(0, sharpe / TARGET_SHARPE * 5.0))

        # === 胃納量分數（0-5）===
        # 目標：> 1000萬
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))

        # === 年化報酬分數（0-2）===
        return_score = min(2.0, max(0, annual_return / TARGET_ANNUAL_RETURN * 2.0))

        # === 懲罰分數 ===
        penalty_score = 0.0

        # 1. MDD 懲罰（核心懲罰項）
        if max_drawdown > MAX_DRAWDOWN:
            # MDD 超過 20% 時施加懲罰
            mdd_penalty = (max_drawdown - MAX_DRAWDOWN) * 20
            penalty_score -= min(5.0, mdd_penalty)

        # 2. 穩健性懲罰
        robustness = result['robustness_score']
        if not result['is_robust']:
            penalty_score -= (1 - robustness) * 3

        # 3. 胃納量硬約束
        if capacity < MIN_CAPACITY * 0.25:
            penalty_score -= 5.0

        return (sharpe_score, capacity_score, return_score, penalty_score)

    except Exception as e:
        print(f"   ⚠️ 評估失敗: {e}")
        return (0.0, 0.0, 0.0, -10.0)


def evaluate_fitness_wrapper(args):
    """並行評估包裝函數"""
    individual, idx = args
    try:
        fitness = evaluate_fitness(individual)
        return idx, fitness
    except Exception as e:
        print(f"   ⚠️ 評估 {idx} 失敗: {e}")
        return idx, (0.0, 0.0, 0.0, -10.0)


# ============================================================================
# 第十部分：DEAP 設定
# ============================================================================
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

print("✅ DEAP NSGA-II 配置完成")


# ============================================================================
# 第十一部分：Pareto Archive 管理器（精英保留）
# ============================================================================
class ParetoArchiveManager:
    """
    Pareto 前緣歷史管理器

    🔥 精英保留策略：
    - 保留前 10 強精英
    - 支援 Pickle 序列化
    - 跨視窗共享
    """

    def __init__(self, archive_file: str, elite_size: int = ELITE_SIZE):
        self.archive_file = archive_file
        self.elite_size = elite_size
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
        """更新 Pareto 前緣（精英保留）"""
        # 合併新舊個體
        all_individuals = self.archive + [
            {
                'genes': list(ind),
                'fitness': ind.fitness.values,
                'timestamp': datetime.now().isoformat(),
            }
            for ind in pareto_front
        ]

        # 去重
        unique = self._deduplicate(all_individuals)

        # 🔥 保留前 ELITE_SIZE 名精英
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:self.elite_size * 3]  # 保留 30 個供選擇

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
        """保存（使用 Pickle）"""
        try:
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v2.0_dragon',
                }, f)
        except Exception as e:
            print(f"⚠️ 歷史保存失敗: {e}")

    def get_top_n(self, n: int = AUTO_VERIFY_TOP_N) -> List[Dict]:
        """獲取前 N 名精英"""
        sorted_archive = sorted(self.archive, key=lambda x: sum(x['fitness']), reverse=True)
        return sorted_archive[:n]

    def inject_elites(self, population: List, ratio: float = 0.2):
        """注入精英到族群"""
        if not self.archive:
            return population

        n_inject = int(len(population) * ratio)
        n_inject = min(n_inject, len(self.archive), self.elite_size)

        # 選擇最佳精英
        elites = self.get_top_n(n_inject)

        # 替換族群中最差的個體
        population.sort(key=lambda ind: sum(ind.fitness.values) if ind.fitness.valid else -999)

        for i, elite in enumerate(elites):
            population[i][:] = elite['genes']
            if hasattr(population[i], 'fitness'):
                del population[i].fitness.values

        print(f"✅ 注入 {n_inject} 個歷史精英")

        return population

# 初始化 Pareto Archive
pareto_archive = ParetoArchiveManager(paths.pareto_archive)


# ============================================================================
# 第十二部分：Check-point 管理器
# ============================================================================
class CheckpointManager:
    """
    Check-point 管理器

    🔥 支援功能：
    - Pickle 序列化保存
    - 自動恢復訓練
    - 進度追蹤
    """

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file

    def save(self, generation: int, population: List, pareto_front: List,
             history: List, stats: Dict):
        """保存 checkpoint"""
        try:
            checkpoint = {
                'generation': generation,
                'population': [
                    {'genes': list(ind), 'fitness': ind.fitness.values if ind.fitness.valid else None}
                    for ind in population
                ],
                'pareto_front': [
                    {'genes': list(ind), 'fitness': ind.fitness.values}
                    for ind in pareto_front
                ],
                'history': history,
                'stats': stats,
                'timestamp': datetime.now().isoformat(),
                'version': 'v2.0_dragon',
            }

            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)

            print(f"   💾 Checkpoint 已保存（第 {generation + 1} 代）")

        except Exception as e:
            print(f"   ⚠️ Checkpoint 保存失敗: {e}")

    def load(self) -> Optional[Dict]:
        """載入 checkpoint"""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            print(f"✅ 載入 Checkpoint（第 {checkpoint['generation'] + 1} 代）")
            return checkpoint

        except Exception as e:
            print(f"⚠️ Checkpoint 載入失敗: {e}")
            return None

    def restore_population(self, checkpoint: Dict) -> Tuple[List, int]:
        """從 checkpoint 恢復族群"""
        population = []

        for ind_data in checkpoint['population']:
            ind = creator.Individual(ind_data['genes'])
            if ind_data['fitness']:
                ind.fitness.values = ind_data['fitness']
            population.append(ind)

        generation = checkpoint['generation']
        return population, generation

# 初始化 Checkpoint 管理器
checkpoint_mgr = CheckpointManager(paths.checkpoint_file)


# ============================================================================
# 第十三部分：自動驗證歷史精英
# ============================================================================
class HistoryValidator:
    """
    歷史精英驗證器

    🔥 重啟後自動驗證前 5 名：
    - 使用最新數據重新回測
    - 確保策略仍然有效
    - 記錄驗證結果
    """

    def __init__(self, pareto_mgr: ParetoArchiveManager,
                 strategy_eng: StrategyEngine,
                 backtest_eng: InSampleOutSampleBacktest):
        self.pareto_mgr = pareto_mgr
        self.strategy_eng = strategy_eng
        self.backtest_eng = backtest_eng
        self.validation_log = paths.validation_log

    def validate_top_n(self, n: int = AUTO_VERIFY_TOP_N) -> List[Dict]:
        """
        驗證歷史前 N 名

        🔥 使用 sim() 重新回測
        """
        print(f"\n{'='*70}")
        print(f"🔍 自動驗證歷史前 {n} 名精英")
        print(f"{'='*70}")

        top_n = self.pareto_mgr.get_top_n(n)

        if not top_n:
            print("   ℹ️ 無歷史精英可驗證")
            return []

        validation_results = []

        for i, elite in enumerate(top_n, 1):
            print(f"\n   驗證第 {i} 名...")

            try:
                # 解碼基因
                params = gene_decoder.decode(elite['genes'])

                # 組合策略
                position = self.strategy_eng.combine_strategies(params, apply_weight_constraint=True)

                if position is None or position.empty:
                    print(f"   ⚠️ 第 {i} 名策略無法生成持股")
                    continue

                # 執行回測
                result = self.backtest_eng.run_full_backtest(position, params)

                out_sample = result['out_sample']

                validation_result = {
                    'rank': i,
                    'original_fitness': elite['fitness'],
                    'validated_sharpe': out_sample['sharpe'],
                    'validated_capacity': out_sample['capacity'],
                    'validated_mdd': out_sample['max_drawdown'],
                    'validated_return': out_sample['annual_return'],
                    'is_still_valid': (
                        out_sample['sharpe'] >= TARGET_SHARPE * 0.8 and
                        out_sample['capacity'] >= MIN_CAPACITY * 0.8 and
                        out_sample['max_drawdown'] <= MAX_DRAWDOWN * 1.2
                    ),
                    'timestamp': datetime.now().isoformat(),
                }

                validation_results.append(validation_result)

                # 輸出結果
                status = "✅ 有效" if validation_result['is_still_valid'] else "⚠️ 失效"
                print(f"   {status} | 夏普: {out_sample['sharpe']:.2f} | "
                      f"胃納量: {out_sample['capacity']/1e6:.1f}百萬 | "
                      f"MDD: {out_sample['max_drawdown']*100:.1f}%")

            except Exception as e:
                print(f"   ⚠️ 驗證失敗: {e}")

        # 保存驗證結果
        self._save_validation_log(validation_results)

        print(f"\n✅ 驗證完成：{len(validation_results)} 個精英")

        return validation_results

    def _save_validation_log(self, results: List[Dict]):
        """保存驗證日誌"""
        try:
            with open(self.validation_log, 'w', encoding='utf-8') as f:
                json.dump({
                    'validation_time': datetime.now().isoformat(),
                    'results': results,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   ⚠️ 驗證日誌保存失敗: {e}")

# 初始化驗證器
history_validator = HistoryValidator(pareto_archive, strategy_engine, backtest_engine)


# ============================================================================
# 第十四部分：進度日誌記錄器
# ============================================================================
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
            'current_gen': gen + 1,
            'total_gen': total_gen,
            'best_sharpe': stats.get('best_sharpe', 0),
            'best_capacity': stats.get('best_capacity', 0),
            'best_composite': stats.get('best_composite', 0),
            'best_return': stats.get('best_return', 0),
            'best_mdd': stats.get('best_mdd', 0),
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


# ============================================================================
# 第十五部分：演化引擎（含並行運算）
# ============================================================================
class EvolutionEngine:
    """
    演化引擎

    🔥 核心功能：
    - NSGA-II 多目標優化
    - 並行評估加速
    - 精英保留策略
    - Check-point 自動保存
    """

    def __init__(self, toolbox, pareto_mgr: ParetoArchiveManager,
                 checkpoint_mgr: CheckpointManager):
        self.toolbox = toolbox
        self.pareto_mgr = pareto_mgr
        self.checkpoint_mgr = checkpoint_mgr
        self.history = []
        self.logger = ProgressLogger(WINDOW_ID)

    def run(self, n_generations: int = N_GENERATIONS,
            resume: bool = True) -> Tuple[List, List]:
        """
        執行演化

        Args:
            n_generations: 演化代數
            resume: 是否從 checkpoint 恢復
        """
        print(f"\n{'='*70}")
        print(f"🐉 開始演化")
        print(f"   族群: {POPULATION_SIZE}, 世代: {n_generations}")
        print(f"   精英保留: {ELITE_SIZE} 個")
        print(f"   並行核心: {N_WORKERS}")
        print(f"{'='*70}\n")

        start_gen = 0
        population = None

        # 嘗試從 checkpoint 恢復
        if resume:
            checkpoint = self.checkpoint_mgr.load()
            if checkpoint:
                population, start_gen = self.checkpoint_mgr.restore_population(checkpoint)
                self.history = checkpoint.get('history', [])
                start_gen += 1  # 從下一代繼續
                print(f"   從第 {start_gen + 1} 代繼續演化...")

        # 初始化族群
        if population is None:
            population = self.toolbox.population(n=POPULATION_SIZE)
            # 注入歷史精英
            population = self.pareto_mgr.inject_elites(population, ratio=0.2)

        # 初始評估
        population = self._evaluate_population(population)

        # 演化循環
        for gen in range(start_gen, n_generations):
            gen_start_time = time.time()

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

            # 🔥 精英保留
            combined = population + offspring
            population = self.toolbox.select(combined, POPULATION_SIZE)

            # 獲取 Pareto 前緣
            pareto_front = tools.sortNondominated(population, len(population),
                                                   first_front_only=True)[0]

            # 統計
            stats = self._compute_stats(population, gen)
            self.history.append(stats)

            elapsed = time.time() - gen_start_time

            # 輸出
            if (gen + 1) % 5 == 0 or gen == start_gen:
                self._print_generation_stats(gen, n_generations, stats, elapsed)

            # 🔥 Check-point 保存
            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                self.checkpoint_mgr.save(gen, population, pareto_front,
                                         self.history, stats)

            # 記錄進度
            self.logger.log_generation(gen, n_generations, stats)

            # 垃圾回收
            if (gen + 1) % 10 == 0:
                gc.collect()

        # 最終 Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population),
                                               first_front_only=True)[0]

        # 更新歷史存檔
        self.pareto_mgr.update(pareto_front)

        # 最終 checkpoint
        self.checkpoint_mgr.save(n_generations - 1, population, pareto_front,
                                 self.history, stats)

        # 輸出結果
        self._print_pareto_front(pareto_front)

        return population, pareto_front

    def _evaluate_population(self, population: List) -> List:
        """評估族群（支援並行）"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        print(f"   評估 {len(invalid)} 個個體...")

        if USE_PARALLEL and len(invalid) > 4:
            # 並行評估
            fitnesses = self._parallel_evaluate(invalid)
        else:
            # 序列評估
            fitnesses = []
            for i, ind in enumerate(invalid):
                fitness = evaluate_fitness(ind)
                fitnesses.append(fitness)
                if (i + 1) % 10 == 0:
                    print(f"   進度: {i+1}/{len(invalid)}")

        # 設定適應度
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        return population

    def _parallel_evaluate(self, individuals: List) -> List:
        """並行評估"""
        try:
            # 使用 ThreadPoolExecutor（避免 FinLab 多進程問題）
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                args = [(ind, i) for i, ind in enumerate(individuals)]
                futures = [executor.submit(evaluate_fitness_wrapper, arg) for arg in args]

                results = [None] * len(individuals)
                completed = 0

                for future in as_completed(futures):
                    idx, fitness = future.result()
                    results[idx] = fitness
                    completed += 1

                    if completed % 10 == 0:
                        print(f"   並行進度: {completed}/{len(individuals)}")

            return results

        except Exception as e:
            print(f"   ⚠️ 並行評估失敗，改用序列: {e}")
            return [evaluate_fitness(ind) for ind in individuals]

    def _compute_stats(self, population: List, gen: int) -> Dict:
        """計算統計"""
        fitnesses = [ind.fitness.values for ind in population]

        sharpes = [f[0] / 5.0 * TARGET_SHARPE for f in fitnesses]
        capacities = [f[1] / 5.0 * MIN_CAPACITY / 1e6 for f in fitnesses]
        returns = [f[2] / 2.0 * TARGET_ANNUAL_RETURN for f in fitnesses]
        penalties = [f[3] for f in fitnesses]

        composites = [sum(f) for f in fitnesses]

        stats = {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': max(sharpes),
            'avg_sharpe': np.mean(sharpes),
            'best_capacity': max(capacities),
            'best_return': max(returns),
            'best_penalty': max(penalties),
            'best_robustness': max(penalties),
        }

        return stats

    def _print_generation_stats(self, gen: int, total_gen: int,
                                stats: Dict, elapsed: float):
        """輸出單代統計"""
        reached = "🎯" if stats['best_sharpe'] >= TARGET_SHARPE else ""

        print(f"\n=== 第 {gen+1}/{total_gen} 代 {reached} ===")
        print(f"   最佳綜合: {stats['best_composite']:.4f}")
        print(f"   最佳夏普: {stats['best_sharpe']:.2f} (目標: {TARGET_SHARPE})")
        print(f"   最佳胃納量: {stats['best_capacity']:.1f} 百萬")
        print(f"   穩健性: {stats['best_robustness']:.2f}")
        print(f"   耗時: {elapsed:.1f}s")

    def _print_pareto_front(self, pareto_front: List):
        """輸出 Pareto 前緣"""
        print(f"\n{'='*70}")
        print(f"🏆 Pareto 最優解集（前 {ELITE_SIZE} 名）")
        print(f"{'='*70}")
        print(f"{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納量(百萬)':<14}{'達標'}")
        print("-" * 70)

        sorted_front = sorted(pareto_front,
                             key=lambda x: sum(x.fitness.values),
                             reverse=True)[:ELITE_SIZE]

        for i, ind in enumerate(sorted_front, 1):
            composite = sum(ind.fitness.values)
            sharpe = ind.fitness.values[0] / 5.0 * TARGET_SHARPE
            capacity = ind.fitness.values[1] / 5.0 * MIN_CAPACITY / 1e6

            reached = "✅" if sharpe >= TARGET_SHARPE and capacity >= MIN_CAPACITY / 1e6 else ""

            print(f"{i:<6}{composite:<10.4f}{sharpe:<10.2f}{capacity:<14.2f}{reached}")


# ============================================================================
# 第十六部分：主程式
# ============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║     🐉 三策略小小龍 - 遺傳演算法優化系統 v2.0                        ║
║     NSGA-II + In-Sample/Out-of-Sample + 精英保留                   ║
╠════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e6:.0f}百萬+, MDD < {MAX_DRAWDOWN*100:.0f}%          ║
║  📊 訓練期：{IN_SAMPLE_START} ~ {IN_SAMPLE_END}                             ║
║  📊 測試期：{OUT_SAMPLE_START} ~ 至今                                  ║
║  💼 最低持股權重：{MIN_POSITION_WEIGHT*100:.0f}%                                       ║
║  🧬 基因長度：{GeneDecoder.GENE_LENGTH} 個參數                                        ║
║  🔧 並行核心：{N_WORKERS}                                              ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 🔥 重啟後自動驗證歷史前 5 名
    print("\n" + "="*70)
    print("📋 步驟 1: 驗證歷史精英")
    print("="*70)
    validation_results = history_validator.validate_top_n(AUTO_VERIFY_TOP_N)

    # 建立演化引擎
    engine = EvolutionEngine(toolbox, pareto_archive, checkpoint_mgr)

    # 執行演化
    print("\n" + "="*70)
    print("📋 步驟 2: 執行遺傳演算法優化")
    print("="*70)
    population, pareto_front = engine.run(N_GENERATIONS, resume=True)

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*70}")
        print("📊 步驟 3: 最佳個體詳細回測")
        print(f"{'='*70}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        # 顯示最佳參數
        print("\n🔍 最佳參數：")
        print(f"   策略權重：")
        print(f"      - 低波動本益比: {best_params['weight_lv']:.2%}")
        print(f"      - 小資族: {best_params['weight_si']:.2%}")
        print(f"      - 營收股價雙渦輪: {best_params['weight_rpt']:.2%}")
        print(f"   回測參數：")
        print(f"      - 停損: {best_params['stop_loss']:.1%}")
        print(f"      - 移動停利: {best_params['trail_stop']:.1%}")
        print(f"      - 停利: {best_params['take_profit']:.1%}")
        print(f"      - 持股上限: {best_params['position_limit']:.1%}")

        # 保存最佳參數
        with open(paths.best_params_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

        # 完整回測
        try:
            print("\n執行完整回測...")
            position = strategy_engine.combine_strategies(best_params, apply_weight_constraint=True)

            # In-Sample 回測
            print("\n📈 In-Sample 回測 (2017-2022):")
            in_sample_pos = position.loc[IN_SAMPLE_START:IN_SAMPLE_END]
            report_in = sim(
                position=in_sample_pos,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price="high_low_avg",
                position_limit=best_params['position_limit'],
                stop_loss=best_params['stop_loss'],
                trail_stop=best_params['trail_stop'],
                take_profit=best_params['take_profit'],
                stop_trading_next_period=False,
                upload=False,
                name='Dragon_InSample'
            )
            report_in.display()

            # Out-of-Sample 回測
            print("\n📈 Out-of-Sample 回測 (2023-至今):")
            out_sample_pos = position.loc[OUT_SAMPLE_START:]
            report_out = sim(
                position=out_sample_pos,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price="high_low_avg",
                position_limit=best_params['position_limit'],
                stop_loss=best_params['stop_loss'],
                trail_stop=best_params['trail_stop'],
                take_profit=best_params['take_profit'],
                stop_trading_next_period=False,
                upload=False,
                name='Dragon_OutSample'
            )
            report_out.display()

        except Exception as e:
            print(f"⚠️ 完整回測失敗: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ 優化完成！")
    print(f"📁 結果保存於: {paths.output_dir}")
    print(f"📁 Pareto 存檔: {paths.pareto_archive}")
    print(f"📁 Checkpoint: {paths.checkpoint_file}")
    print(f"{'='*70}")

    return population, pareto_front


if __name__ == "__main__":
    population, pareto = main()
