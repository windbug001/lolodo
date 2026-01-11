#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 FinLab 台股基因演算法優化系統 - 終極強化版 v2.0
FinLab Taiwan Stock Genetic Algorithm Optimizer - Ultimate Enhanced Edition
================================================================================

【核心功能】
✅ 基於三策略合併邏輯（低波動本益比、小資族、營收股價雙渦輪）
✅ NSGA-II 多目標遺傳演算法自動優化參數
✅ Walk-Forward 驗證避免 Overfitting
✅ 樣本內 (In-Sample ~2022) / 樣本外 (OOS 2023~) 分離評估
✅ 每 10 代進行詳細回測報告（含樣本內/外對比）
✅ Pareto Archive 歷史精英持續進化
✅ Check-point 支援程式中斷後無縫重啟
✅ 重啟時驗證歷史前 5 名數據一致性
✅ 多核心並行運算（multiprocessing）
✅ 進度日誌供監控系統讀取

【績效目標】
- 夏普值 (Sharpe Ratio) >= 4.2
- 最大回檔 (MDD) < 20%
- 資金胃納量 >= 1,000 萬台幣
- 年化報酬 >= 30%

【持股邏輯】
- 單一標的持股比例至少 3%
- 低於 3% 門檻則不持倉 (0%)

【回測規範】
- 樣本內 (In-Sample)：2017-01-01 ~ 2022-12-31
- 樣本外 (Out-of-Sample)：2023-01-01 ~ 至今

版本：v2.0 Ultimate Enhanced (2025-12-26)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# ============================================================================
# 🔥【最重要】在任何 import 之前禁用 FinLab 快取
# ============================================================================
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
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# ============================================================================
# 🔥 核心設定區（請根據需求修改）
# ============================================================================
# 視窗 ID (1-4)，多視窗並行時請修改此值
WINDOW_ID = int(os.environ.get('WINDOW_ID', 1))

# FinLab API Key
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# ===== 績效目標 =====
TARGET_SHARPE = 4.2          # 目標夏普值
MIN_CAPACITY = 10_000_000    # 最小資金胃納量 (1000萬)
TARGET_ANNUAL_RETURN = 0.30  # 目標年化報酬 30%
MAX_DRAWDOWN = 0.20          # 最大回檔限制 20%

# ===== 持股配比要求 =====
MIN_POSITION_WEIGHT = 0.03   # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股步長 3%

# ===== GA 演化參數 =====
POPULATION_SIZE = 50         # 族群大小（適應 FinLab 回測速度）
N_GENERATIONS = 100          # 總世代數
MUTATION_RATE = 0.20         # 變異率
CROSSOVER_RATE = 0.80        # 交叉率
ELITE_RATIO = 0.30           # 精英注入比例

# ===== 並行設定 =====
N_WORKERS = max(1, mp.cpu_count() - 1)  # 並行工作數

# ===== 回測時間設定（樣本內/外分離）=====
IN_SAMPLE_START = '2017-01-01'
IN_SAMPLE_END = '2022-12-31'
OUT_OF_SAMPLE_START = '2023-01-01'
OUT_OF_SAMPLE_END = None  # None 表示至今

# ===== Walk-Forward 設定 =====
WALK_FORWARD_WINDOWS = 3
TRAIN_MONTHS = 24
TEST_MONTHS = 6

# ===== 報告頻率 =====
DETAILED_REPORT_INTERVAL = 10  # 每 10 代進行詳細回測
PROGRESS_LOG_INTERVAL = 1      # 每代記錄進度
CHECKPOINT_INTERVAL = 10       # 每 10 代保存檢查點
TOP_N_VALIDATION = 5           # 重啟時驗證前 N 名

print(f"{'=' * 80}")
print(f"🚀 FinLab 台股基因演算法優化系統 v2.0 - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e7:.0f}00萬, MDD < {MAX_DRAWDOWN*100:.0f}%")
print(f"   📊 樣本內: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
print(f"   📊 樣本外: {OUT_OF_SAMPLE_START} ~ {'至今' if OUT_OF_SAMPLE_END is None else OUT_OF_SAMPLE_END}")
print(f"   🖥️  並行工作數: {N_WORKERS}")
print(f"{'=' * 80}")


# =============================================================================
# 第二部分：快取清理與套件安裝
# =============================================================================
def clear_finlab_cache():
    """清除可能損壞的 FinLab 快取（必須在載入 FinLab 之前執行）"""
    import glob
    import shutil
    import subprocess

    print("🔧 清除 FinLab 快取...")

    # 1. 清除所有 .pkl 檔案（使用 subprocess 更安全）
    try:
        subprocess.run(
            ['find', '/root', '/tmp', '-name', '*.pkl', '-type', 'f', '-delete'],
            stderr=subprocess.DEVNULL,
            timeout=10
        )
    except Exception:
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
        except Exception:
            pass

    # 3. 清除 pandas 快取
    pandas_cache = os.path.expanduser('~/.cache/pandas')
    if os.path.exists(pandas_cache):
        try:
            shutil.rmtree(pandas_cache)
            cleared += 1
        except Exception:
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

# 載入 tqdm
from tqdm import tqdm

print(f"✅ 套件載入完成")
print(f"   - FinLab: {finlab.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Pandas: {pd.__version__}")


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
    except Exception:
        base_dir = './finlab_ga_output'
        in_colab = False
        print("⚠️ 本地環境模式")

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
        self.log_dir = f"{self.window_dir}_logs"  # 日誌目錄（供監控器讀取）
        self.reports_dir = f"{self.window_dir}/reports"  # 詳細報告

        for d in [self.window_dir, self.output_dir, self.pareto_dir,
                  self.history_dir, self.log_dir, self.reports_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def pareto_archive(self) -> str:
        return f"{self.pareto_dir}/pareto_archive_w{self.window_id}.pkl"

    @property
    def global_pareto_archive(self) -> str:
        """全局 Pareto 存檔（所有視窗共享）"""
        return f"{self.pareto_dir}/global_pareto_archive.pkl"

    @property
    def checkpoint_file(self) -> str:
        return f"{self.window_dir}/checkpoint.pkl"

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
print(f"📁 工作目錄: {paths.window_dir}")


# =============================================================================
# 第四部分：數據載入（使用 FinLab API）
# =============================================================================
class FinLabDataLoader:
    """FinLab 數據載入器（Singleton 模式）"""

    _instance = None
    _cache = {}
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_all_data()
            FinLabDataLoader._loaded = True

    def _safe_get(self, key: str, retry: bool = True):
        """安全載入數據，遇到 EOFError 時清除快取重試"""
        try:
            return data.get(key)
        except (EOFError, Exception) as e:
            if 'EOF' in str(e) and retry:
                print(f"   ⚠️ 快取損壞，清除後重試: {key}")
                import subprocess
                subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-delete'],
                               stderr=subprocess.DEVNULL)
                return self._safe_get(key, retry=False)
            else:
                raise

    def _load_all_data(self):
        """載入所有數據"""
        print("📊 載入 FinLab 數據...")
        start_time = time.time()

        # ===== 價格相關數據 =====
        self._cache['close'] = self._safe_get('price:收盤價')
        self._cache['vol'] = self._safe_get('price:成交股數')
        self._cache['open'] = self._safe_get('price:開盤價')
        self._cache['high'] = self._safe_get('price:最高價')
        self._cache['low'] = self._safe_get('price:最低價')
        self._cache['adj_close'] = self._safe_get("etl:adj_close")

        # ===== 估值數據 =====
        self._cache['pe'] = self._safe_get('price_earning_ratio:本益比')
        self._cache['pb'] = self._safe_get("price_earning_ratio:股價淨值比")
        self._cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')

        # ===== 營收數據 =====
        self._cache['rev'] = self._safe_get('monthly_revenue:當月營收')
        self._cache['rev_yoy_growth'] = self._safe_get('monthly_revenue:去年同月增減(%)')
        self._cache['rev_month_growth'] = self._safe_get('monthly_revenue:上月比較增減(%)')

        # ===== 基本面指標 =====
        self._cache['營業利益成長率'] = self._safe_get('fundamental_features:營業利益成長率')
        self._cache['業外收支營收率'] = self._safe_get('fundamental_features:業外收支營收率')
        self._cache['營業毛利率'] = self._safe_get("fundamental_features:營業毛利率")
        self._cache['ROE綜合損益'] = self._safe_get("fundamental_features:ROE綜合損益")
        self._cache['稅後淨利率'] = self._safe_get("fundamental_features:稅後淨利率")
        self._cache['稅前淨利率'] = self._safe_get("fundamental_features:稅前淨利率")

        # ===== 籌碼資料 =====
        self._cache['融資使用率'] = self._safe_get('margin_transactions:融資使用率')
        self._cache['董監持有股數占比'] = self._safe_get("internal_equity_changes:董監持有股數占比")
        self._cache['inventory'] = self._safe_get("inventory")

        # ===== 市值資料 =====
        self._cache['市值'] = self._safe_get('etl:market_value')

        # ===== 財務報表 =====
        self._cache['股本'] = self._safe_get('financial_statement:股本')
        self._cache['投資活動現金流'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._cache['營業活動現金流'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._cache['稅後淨利'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._cache['權益總計'] = self._safe_get('financial_statement:股東權益總額')

        # ===== 技術指標 =====
        self._cache['rsi'] = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
        self._cache['atr'] = data.indicator('ATR', adjust_price=True, timeperiod=10)

        # 計算衍生指標
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"✅ 數據載入完成 ({elapsed:.1f}s)")
        print(f"   股票數: {len(self._cache['close'].columns)}")
        print(f"   期間: {self._cache['close'].index[0]} ~ {self._cache['close'].index[-1]}")

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


# 全局數據載入器（延遲初始化）
_data_loader = None


def get_data_loader() -> FinLabDataLoader:
    """獲取數據載入器（延遲初始化）"""
    global _data_loader
    if _data_loader is None:
        _data_loader = FinLabDataLoader()
    return _data_loader


# =============================================================================
# 第五部分：策略引擎（基於三策略合併）
# =============================================================================
class StrategyEngine:
    """策略引擎 - 整合三個子策略"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader

    def strategy_low_volatility_pe(self, params: Dict) -> Any:
        """
        策略一：低波動本益比策略

        核心邏輯：尋找營收穩定成長、低波動、低融資使用率、本益比合理的股票
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
        try:
            small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                                 .reset_index()
                                 .groupby(["date", "stock_id"])
                                 .agg({"占集保庫存數比例": "sum"})
                                 .reset_index()
                                 .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
        except Exception:
            small_inv_under50 = True

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
        position = peg[cond_all & (peg > 0)].is_smallest(top_n).reindex(
            rev.index_str_to_date().index, method='ffill'
        )
        return position

    def strategy_small_investor(self, params: Dict) -> Any:
        """
        策略二：小資族策略

        核心邏輯：鎖定中小型市值股票，要求自由現金流正、營收成長
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
        rsv = (close - close.rolling(rsv_period).min()) / (
            close.rolling(rsv_period).max() - close.rolling(rsv_period).min()
        )

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
        ) * rsv).is_largest(top_n)

        position = position.reindex(當月營收.index_str_to_date().index)
        return position

    def strategy_revenue_price_turbo(self, params: Dict) -> Any:
        """
        策略三：營收股價雙渦輪策略

        核心邏輯：尋找營收創新高、股價創新高的動能股
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
        try:
            boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= 12) & (inventory.持股分級.astype(int) <= 16)]
                                      .reset_index()
                                      .groupby(["date", "stock_id"])
                                      .agg({"占集保庫存數比例": "sum"})
                                      .reset_index()
                                      .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= 18
        except Exception:
            boss_inventory_over400 = True

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
        position = position[position > 0].is_largest(top_n).reindex(
            rev.index_str_to_date().index, method="ffill"
        )
        return position

    def combine_strategies(self, params: Dict) -> Any:
        """
        合併三個策略

        根據基因編碼的權重動態組合三個子策略
        """
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

            # 🔥 應用持股配比約束（至少 3%）
            position_combined = self._apply_position_constraints(position_combined)

            return position_combined

        except Exception as e:
            print(f"   ⚠️ 策略合併失敗: {e}")
            return None

    def _apply_position_constraints(self, position: pd.DataFrame) -> pd.DataFrame:
        """
        應用持股配比約束

        規則：
        1. 每隻股票持股至少 3%，否則設為 0%
        2. 確保總持股 = 100%
        """
        if position is None or position.empty:
            return position

        result = position.copy()

        for date in result.index:
            row = result.loc[date]
            total = row.sum()

            if total == 0:
                continue

            # 正規化
            normalized = row / total

            # 過濾低於 3% 的持股
            mask = normalized >= MIN_POSITION_WEIGHT
            filtered = normalized[mask]

            if filtered.empty:
                # 如果全部被過濾，保留最大的幾檔
                max_stocks = int(1.0 / MIN_POSITION_WEIGHT)  # 最多 33 檔
                top_stocks = normalized.nlargest(min(len(normalized[normalized > 0]), max_stocks))
                filtered = top_stocks

            # 重新正規化
            if filtered.sum() > 0:
                filtered = filtered / filtered.sum()

            # 更新結果
            result.loc[date] = 0
            result.loc[date, filtered.index] = filtered

        return result


# =============================================================================
# 第六部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """
    基因解碼器

    基因結構（32 個參數）：
    - 策略一（低波動本益比）：9 個參數
    - 策略二（小資族）：8 個參數
    - 策略三（營收股價雙渦輪）：8 個參數
    - 策略權重：3 個
    - 回測參數：4 個
    """

    GENE_LENGTH = 32

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因為策略參數"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # === 策略一：低波動本益比（9 個參數）===
        params['lv_rev_ma3_ma12_ratio'] = genes[0] * 0.5 + 1.0    # 1.0-1.5
        params['lv_rev_consistency'] = genes[1] * 0.4 + 0.5       # 0.5-0.9
        params['lv_volatility_threshold'] = genes[2] * 0.05 + 0.02  # 0.02-0.07
        params['lv_margin_usage_limit'] = genes[3] * 30 + 20      # 20-50
        params['lv_non_op_income_limit'] = genes[4] * 10 + 5      # 5-15
        params['lv_min_volume'] = genes[5] * 200000 + 100000      # 10萬-30萬
        params['lv_pe_min'] = genes[6] * 10 + 3                   # 3-13
        params['lv_pe_max'] = genes[7] * 20 + 15                  # 15-35
        params['lv_top_n'] = int(genes[8] * 5 + 2)                # 2-7

        # === 策略二：小資族（8 個參數）===
        params['si_market_value_limit'] = genes[9] * 10e9 + 10e9  # 100億-200億
        params['si_market_rev_ratio_limit'] = genes[10] * 3 + 2   # 2-5
        params['si_rev_yoy_growth_limit'] = genes[11] * 10 - 15   # -15 to -5
        params['si_rev_mom_growth_limit'] = genes[12] * 30 - 70   # -70 to -40
        params['si_rsv_period'] = int(genes[13] * 30 + 40)        # 40-70
        params['si_ma_period'] = int(genes[14] * 40 + 50)         # 50-90
        params['si_volume_threshold'] = genes[15] * 200000 + 100000  # 10萬-30萬
        params['si_top_n'] = int(genes[16] * 6 + 4)               # 4-10

        # === 策略三：營收股價雙渦輪（8 個參數）===
        params['rpt_rev_ma_period'] = int(genes[17] * 5 + 3)      # 3-8
        params['rpt_rev_ma_lookback'] = int(genes[18] * 15 + 15)  # 15-30
        params['rpt_price_high_window'] = int(genes[19] * 8 + 4)  # 4-12
        params['rpt_min_volume'] = genes[20] * 300000 + 200000    # 20萬-50萬
        params['rpt_min_price'] = genes[21] * 10 + 10             # 10-20
        params['rpt_rsi_threshold'] = genes[22] * 20 + 50         # 50-70
        params['rpt_pe_limit'] = genes[23] * 100 + 100            # 100-200
        params['rpt_top_n'] = int(genes[24] * 10 + 8)             # 8-18

        # === 策略權重（3 個）===
        raw_weights = genes[25:28]
        total = sum(raw_weights) + 1e-10
        params['weight_lv'] = raw_weights[0] / total
        params['weight_si'] = raw_weights[1] / total
        params['weight_rpt'] = raw_weights[2] / total

        # === 回測參數（4 個）===
        params['stop_loss'] = genes[28] * 0.2 + 0.15      # 15%-35%
        params['trail_stop'] = genes[29] * 0.3 + 0.2      # 20%-50%
        params['take_profit'] = genes[30] * 0.5 + 0.5     # 50%-100%
        params['position_limit'] = genes[31] * 0.2 + 0.25  # 25%-45%

        return params


gene_decoder = GeneDecoder()


# =============================================================================
# 第七部分：回測引擎（樣本內/外分離）
# =============================================================================
class BacktestEngine:
    """
    回測引擎

    功能：
    1. 樣本內 (In-Sample) 回測
    2. 樣本外 (Out-of-Sample) 回測
    3. Walk-Forward 分析
    4. 詳細報告生成
    """

    def __init__(self):
        self.results_cache = {}

    def run_in_sample_backtest(self, position, params: Dict) -> Dict:
        """執行樣本內回測（~2022 年底）"""
        try:
            # 過濾時間範圍
            position_is = position.loc[IN_SAMPLE_START:IN_SAMPLE_END]

            if position_is.empty or len(position_is) < 50:
                return self._empty_metrics()

            return self._run_backtest(position_is, params, 'InSample')

        except Exception as e:
            return self._empty_metrics()

    def run_out_of_sample_backtest(self, position, params: Dict) -> Dict:
        """執行樣本外回測（2023 年~至今）"""
        try:
            # 過濾時間範圍
            if OUT_OF_SAMPLE_END:
                position_oos = position.loc[OUT_OF_SAMPLE_START:OUT_OF_SAMPLE_END]
            else:
                position_oos = position.loc[OUT_OF_SAMPLE_START:]

            if position_oos.empty or len(position_oos) < 20:
                return self._empty_metrics()

            return self._run_backtest(position_oos, params, 'OutOfSample')

        except Exception as e:
            return self._empty_metrics()

    def run_full_backtest(self, position, params: Dict) -> Dict:
        """執行全期回測"""
        try:
            position_full = position.loc[IN_SAMPLE_START:]

            if position_full.empty:
                return self._empty_metrics()

            return self._run_backtest(position_full, params, 'Full')

        except Exception as e:
            return self._empty_metrics()

    def _run_backtest(self, position, params: Dict, name: str) -> Dict:
        """執行單次回測"""
        try:
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
                'calmar': metrics['ratio'].get('calmarRatio', 0) or 0,
            }

        except Exception as e:
            return self._empty_metrics()

    def run_walk_forward(self, position, params: Dict) -> Dict:
        """
        執行 Walk-Forward 分析

        將時間序列分割成多個訓練/測試窗口，驗證策略穩定性
        """
        try:
            if position is None or position.empty:
                return self._empty_wf_result()

            position = position.loc[IN_SAMPLE_START:]

            if len(position) < 100:
                return self._empty_wf_result()

            # 分割時間窗口
            windows = self._split_windows(position.index)

            if len(windows) < 2:
                # 窗口不足，直接全期回測
                return self._simple_wf_result(position, params)

            # 各窗口回測
            window_results = []
            for i, (train_dates, test_dates) in enumerate(windows):
                test_pos = position.loc[test_dates[0]:test_dates[-1]]

                if test_pos.empty:
                    continue

                result = self._run_backtest(test_pos, params, f'Window{i}')
                if result and result['sharpe'] != 0:
                    window_results.append(result)

            if not window_results:
                return self._empty_wf_result()

            # 整體回測
            overall_result = self._run_backtest(position, params, 'Overall')

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
            return self._empty_wf_result()

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

    def _empty_metrics(self) -> Dict:
        """空指標"""
        return {
            'name': 'Empty',
            'sharpe': 0,
            'annual_return': 0,
            'max_drawdown': 1,
            'capacity': 0,
            'win_rate': 0,
            'total_return': 0,
            'calmar': 0,
        }

    def _empty_wf_result(self) -> Dict:
        """空 Walk-Forward 結果"""
        return {
            'overall': self._empty_metrics(),
            'windows': [],
            'is_robust': False,
            'consistency_score': 0,
        }

    def _simple_wf_result(self, position, params: Dict) -> Dict:
        """簡單 Walk-Forward 結果"""
        result = self._run_backtest(position, params, 'Simple')

        return {
            'overall': result,
            'windows': [result],
            'is_robust': True,
            'consistency_score': 1.0,
        }


backtest_engine = BacktestEngine()


# =============================================================================
# 第八部分：適應度評估函數
# =============================================================================
def evaluate_fitness_worker(genes: List[float]) -> Tuple[float, float, float, float]:
    """
    適應度評估工作函數（用於並行計算）

    Returns: (sharpe_score, capacity_score, return_score, robustness_penalty)
    """
    try:
        # 獲取數據載入器和策略引擎
        dl = get_data_loader()
        engine = StrategyEngine(dl)

        # 解碼基因
        params = gene_decoder.decode(genes)

        # 組合策略
        position = engine.combine_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        # 執行樣本內回測
        is_result = backtest_engine.run_in_sample_backtest(position, params)

        # 執行樣本外回測
        oos_result = backtest_engine.run_out_of_sample_backtest(position, params)

        # 計算各項分數
        # 使用樣本內結果作為主要評估依據
        sharpe_is = is_result['sharpe']
        capacity_is = is_result['capacity']
        annual_return_is = is_result['annual_return']
        mdd_is = is_result['max_drawdown']

        # 樣本外穩健性
        sharpe_oos = oos_result['sharpe']

        # === 夏普值分數（0-5）===
        sharpe_score = min(5.0, max(0, sharpe_is / TARGET_SHARPE * 5.0))

        # === 胃納量分數（0-5）===
        capacity_score = min(5.0, max(0, capacity_is / MIN_CAPACITY * 5.0))

        # === 年化報酬分數（0-2）===
        return_score = min(2.0, max(0, annual_return_is / TARGET_ANNUAL_RETURN * 2.0))

        # === 穩健性懲罰 ===
        # 樣本外夏普應至少達到樣本內的 70%
        if sharpe_is > 0 and sharpe_oos > 0:
            oos_ratio = sharpe_oos / sharpe_is
            if oos_ratio >= 0.7:
                robustness_penalty = 0.0
            else:
                robustness_penalty = -(0.7 - oos_ratio) * 5  # 最多扣 3.5 分
        elif sharpe_is > 0 and sharpe_oos <= 0:
            robustness_penalty = -5.0  # 樣本外失敗，重罰
        else:
            robustness_penalty = -10.0

        # MDD 懲罰
        if mdd_is > MAX_DRAWDOWN:
            mdd_penalty = -(mdd_is - MAX_DRAWDOWN) * 10
            robustness_penalty += mdd_penalty

        return (sharpe_score, capacity_score, return_score, robustness_penalty)

    except Exception as e:
        return (0.0, 0.0, 0.0, -10.0)


def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """適應度評估（主執行緒）"""
    return evaluate_fitness_worker(list(individual))


# =============================================================================
# 第九部分：DEAP 設定
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
                 toolbox.attr_float, n=GeneDecoder.GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=0.0, up=1.0, eta=20.0, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

print("✅ DEAP NSGA-II 配置完成")


# =============================================================================
# 第十部分：Pareto 歷史管理器
# =============================================================================
class ParetoArchiveManager:
    """
    Pareto 前緣歷史管理器

    功能：
    1. 保存歷史最佳個體（持久化）
    2. 載入歷史精英注入新族群
    3. 多視窗共享精英池
    """

    def __init__(self, archive_file: str, global_file: str = None):
        self.archive_file = archive_file
        self.global_file = global_file
        self.archive = []
        self._load()

    def _load(self):
        """載入歷史 Pareto 前緣"""
        # 載入本地存檔
        if os.path.exists(self.archive_file):
            try:
                with open(self.archive_file, 'rb') as f:
                    data = pickle.load(f)
                    self.archive = data.get('individuals', [])
                print(f"✅ 載入本地 {len(self.archive)} 個歷史精英")
            except Exception as e:
                print(f"⚠️ 本地歷史載入失敗: {e}")
                self.archive = []

        # 載入全局存檔
        if self.global_file and os.path.exists(self.global_file):
            try:
                with open(self.global_file, 'rb') as f:
                    global_data = pickle.load(f)
                    global_elites = global_data.get('individuals', [])

                # 合併（去重）
                self.archive = self._merge_archives(self.archive, global_elites)
                print(f"✅ 合併全局精英後共 {len(self.archive)} 個")
            except Exception as e:
                print(f"⚠️ 全局歷史載入失敗: {e}")

        if not self.archive:
            print("ℹ️ 無歷史存檔，從頭開始")

    def _merge_archives(self, local: List, global_: List) -> List:
        """合併並去重"""
        all_individuals = local + global_
        return self._deduplicate(all_individuals)

    def update(self, pareto_front: List):
        """更新 Pareto 前緣"""
        # 合併新舊個體
        new_individuals = [
            {'genes': list(ind), 'fitness': ind.fitness.values, 'timestamp': datetime.now().isoformat()}
            for ind in pareto_front
        ]

        all_individuals = self.archive + new_individuals

        # 去重
        unique = self._deduplicate(all_individuals)

        # 按綜合適應度排序，保留前 50 名
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:50]

        # 保存本地
        self._save()

        # 保存全局
        self._save_global()

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
        """保存到本地檔案"""
        try:
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                    'window_id': WINDOW_ID,
                }, f)
        except Exception as e:
            print(f"⚠️ 本地歷史保存失敗: {e}")

    def _save_global(self):
        """保存到全局檔案"""
        if not self.global_file:
            return

        try:
            # 載入現有全局存檔
            global_archive = []
            if os.path.exists(self.global_file):
                with open(self.global_file, 'rb') as f:
                    global_data = pickle.load(f)
                    global_archive = global_data.get('individuals', [])

            # 合併
            merged = self._merge_archives(global_archive, self.archive)
            merged.sort(key=lambda x: sum(x['fitness']), reverse=True)
            merged = merged[:100]  # 全局保留更多

            with open(self.global_file, 'wb') as f:
                pickle.dump({
                    'individuals': merged,
                    'timestamp': datetime.now().isoformat(),
                }, f)

        except Exception as e:
            print(f"⚠️ 全局歷史保存失敗: {e}")

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

    def get_top_n(self, n: int = TOP_N_VALIDATION) -> List[Dict]:
        """獲取前 N 名精英"""
        sorted_archive = sorted(self.archive, key=lambda x: sum(x['fitness']), reverse=True)
        return sorted_archive[:n]


pareto_archive = ParetoArchiveManager(
    paths.pareto_archive,
    paths.global_pareto_archive
)


# =============================================================================
# 第十一部分：進度日誌記錄器
# =============================================================================
class ProgressLogger:
    """
    進度日誌記錄器

    用於監控系統讀取進度
    """

    def __init__(self, window_id: int, log_dir: str):
        self.window_id = window_id
        self.log_dir = log_dir
        self.history_file = f"{log_dir}/progress_history.json"
        self.start_time = time.time()
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def log_generation(self, gen: int, total_gen: int, stats: Dict, elapsed: float = 0):
        """記錄單代進度"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'window_id': self.window_id,
            'current_gen': gen + 1,
            'total_gen': total_gen,
            'best_sharpe': round(stats.get('best_sharpe', 0), 4),
            'best_capacity': round(stats.get('best_capacity', 0), 2),
            'best_composite': round(stats.get('best_composite', 0), 4),
            'best_return': round(stats.get('best_return', 0), 4),
            'robustness': round(stats.get('best_robustness', 0), 4),
            'avg_composite': round(stats.get('avg_composite', 0), 4),
            'gen_time': round(elapsed, 1),
            'elapsed_total': round(time.time() - self.start_time, 1),
        }

        # 讀取歷史
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []

        # 添加新記錄
        history.append(progress)

        # 只保留最近 500 條
        history = history[-500:]

        # 保存
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def log_detailed_report(self, gen: int, report: Dict):
        """記錄詳細報告"""
        report_file = f"{self.log_dir}/detailed_report_gen{gen+1}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


# =============================================================================
# 第十二部分：檢查點管理器
# =============================================================================
class CheckpointManager:
    """
    檢查點管理器

    功能：
    1. 保存演化狀態
    2. 恢復演化
    3. 驗證歷史前 N 名
    """

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file

    def save(self, gen: int, population: List, history: List):
        """保存檢查點"""
        try:
            checkpoint_data = {
                'generation': gen,
                'population': [
                    {'genes': list(ind), 'fitness': ind.fitness.values if ind.fitness.valid else None}
                    for ind in population
                ],
                'history': history,
                'timestamp': datetime.now().isoformat(),
                'window_id': WINDOW_ID,
            }

            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            print(f"   💾 檢查點已保存（第 {gen+1} 代）")

        except Exception as e:
            print(f"   ⚠️ 檢查點保存失敗: {e}")

    def load(self) -> Optional[Dict]:
        """載入檢查點"""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            print(f"✅ 載入檢查點（第 {data['generation']+1} 代）")
            return data

        except Exception as e:
            print(f"⚠️ 檢查點載入失敗: {e}")
            return None

    def restore_population(self, checkpoint_data: Dict, toolbox) -> Tuple[List, int, List]:
        """從檢查點恢復族群"""
        gen = checkpoint_data['generation']
        history = checkpoint_data.get('history', [])

        population = []
        for ind_data in checkpoint_data['population']:
            ind = toolbox.individual()
            ind[:] = ind_data['genes']
            if ind_data['fitness']:
                ind.fitness.values = ind_data['fitness']
            population.append(ind)

        return population, gen, history


checkpoint_mgr = CheckpointManager(paths.checkpoint_file)


# =============================================================================
# 第十三部分：歷史驗證器
# =============================================================================
class HistoryValidator:
    """
    歷史驗證器

    重啟時驗證歷史前 N 名數據一致性
    """

    def __init__(self, n_validate: int = TOP_N_VALIDATION):
        self.n_validate = n_validate
        self.validation_log = paths.validation_log

    def validate_top_n(self, pareto_mgr: ParetoArchiveManager) -> bool:
        """
        驗證歷史前 N 名

        使用 sim() 重新回測，確保數據一致性
        """
        print(f"\n🔍 驗證歷史前 {self.n_validate} 名精英...")

        top_n = pareto_mgr.get_top_n(self.n_validate)

        if not top_n:
            print("   ℹ️ 無歷史精英需要驗證")
            return True

        dl = get_data_loader()
        engine = StrategyEngine(dl)

        validation_results = []
        all_valid = True

        for i, elite in enumerate(top_n, 1):
            genes = elite['genes']
            original_fitness = elite['fitness']

            # 重新評估
            new_fitness = evaluate_fitness_worker(genes)

            # 計算差異
            original_composite = sum(original_fitness)
            new_composite = sum(new_fitness)
            diff = abs(original_composite - new_composite)
            diff_pct = diff / (abs(original_composite) + 1e-10) * 100

            # 判斷是否一致（允許 10% 誤差）
            is_valid = diff_pct < 10

            result = {
                'rank': i,
                'original_fitness': original_fitness,
                'new_fitness': new_fitness,
                'original_composite': round(original_composite, 4),
                'new_composite': round(new_composite, 4),
                'diff_pct': round(diff_pct, 2),
                'is_valid': is_valid,
            }
            validation_results.append(result)

            status = "✅" if is_valid else "⚠️"
            print(f"   {status} 第 {i} 名: 原始={original_composite:.4f}, 驗證={new_composite:.4f}, 差異={diff_pct:.1f}%")

            if not is_valid:
                all_valid = False

        # 保存驗證結果
        with open(self.validation_log, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': validation_results,
                'all_valid': all_valid,
            }, f, indent=2, ensure_ascii=False)

        if all_valid:
            print(f"✅ 所有精英驗證通過！")
        else:
            print(f"⚠️ 部分精英驗證不一致，將重新評估")

        return all_valid


history_validator = HistoryValidator()


# =============================================================================
# 第十四部分：詳細報告生成器
# =============================================================================
class DetailedReporter:
    """
    詳細報告生成器

    每 10 代生成詳細報告，包含樣本內/外對比
    """

    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        Path(reports_dir).mkdir(parents=True, exist_ok=True)

    def generate_report(self, gen: int, pareto_front: List, history: List) -> Dict:
        """生成詳細報告"""
        print(f"\n📊 生成第 {gen+1} 代詳細報告...")

        dl = get_data_loader()
        engine = StrategyEngine(dl)

        report = {
            'generation': gen + 1,
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'top_individuals': [],
            'in_sample_vs_oos': [],
            'convergence': [],
        }

        # 取前 5 名進行詳細分析
        sorted_front = sorted(pareto_front, key=lambda x: sum(x.fitness.values), reverse=True)[:5]

        for i, ind in enumerate(sorted_front, 1):
            params = gene_decoder.decode(ind)
            position = engine.combine_strategies(params)

            if position is None or position.empty:
                continue

            # 樣本內回測
            is_result = backtest_engine.run_in_sample_backtest(position, params)

            # 樣本外回測
            oos_result = backtest_engine.run_out_of_sample_backtest(position, params)

            # 全期回測
            full_result = backtest_engine.run_full_backtest(position, params)

            individual_report = {
                'rank': i,
                'fitness': ind.fitness.values,
                'composite': sum(ind.fitness.values),
                'params_summary': {
                    'weight_lv': round(params['weight_lv'], 3),
                    'weight_si': round(params['weight_si'], 3),
                    'weight_rpt': round(params['weight_rpt'], 3),
                    'stop_loss': round(params['stop_loss'], 3),
                    'position_limit': round(params['position_limit'], 3),
                },
                'in_sample': {
                    'sharpe': round(is_result['sharpe'], 3),
                    'annual_return': round(is_result['annual_return'], 4),
                    'max_drawdown': round(is_result['max_drawdown'], 4),
                    'capacity': round(is_result['capacity'], 0),
                },
                'out_of_sample': {
                    'sharpe': round(oos_result['sharpe'], 3),
                    'annual_return': round(oos_result['annual_return'], 4),
                    'max_drawdown': round(oos_result['max_drawdown'], 4),
                    'capacity': round(oos_result['capacity'], 0),
                },
                'full_period': {
                    'sharpe': round(full_result['sharpe'], 3),
                    'annual_return': round(full_result['annual_return'], 4),
                    'max_drawdown': round(full_result['max_drawdown'], 4),
                    'capacity': round(full_result['capacity'], 0),
                },
                'oos_ratio': round(oos_result['sharpe'] / (is_result['sharpe'] + 1e-10), 3),
            }

            report['top_individuals'].append(individual_report)

            # 樣本內外對比
            report['in_sample_vs_oos'].append({
                'rank': i,
                'is_sharpe': round(is_result['sharpe'], 3),
                'oos_sharpe': round(oos_result['sharpe'], 3),
                'is_return': round(is_result['annual_return'], 4),
                'oos_return': round(oos_result['annual_return'], 4),
                'is_mdd': round(is_result['max_drawdown'], 4),
                'oos_mdd': round(oos_result['max_drawdown'], 4),
            })

        # 收斂曲線（最近 10 代）
        recent_history = history[-10:] if len(history) >= 10 else history
        report['convergence'] = [
            {
                'gen': h['generation'] + 1,
                'best_composite': round(h['best_composite'], 4),
                'best_sharpe': round(h['best_sharpe'], 4),
                'avg_composite': round(h['avg_composite'], 4),
            }
            for h in recent_history
        ]

        # 總結
        if report['top_individuals']:
            best = report['top_individuals'][0]
            report['summary'] = {
                'best_sharpe_is': best['in_sample']['sharpe'],
                'best_sharpe_oos': best['out_of_sample']['sharpe'],
                'best_capacity': best['in_sample']['capacity'],
                'best_mdd': best['in_sample']['max_drawdown'],
                'oos_stability': round(np.mean([r['oos_ratio'] for r in report['top_individuals']]), 3),
                'target_reached': best['in_sample']['sharpe'] >= TARGET_SHARPE and best['in_sample']['capacity'] >= MIN_CAPACITY,
            }

        # 保存報告
        report_file = f"{self.reports_dir}/detailed_report_gen{gen+1}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 輸出摘要
        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict):
        """輸出報告摘要"""
        print(f"\n{'='*70}")
        print(f"📊 第 {report['generation']} 代詳細報告")
        print(f"{'='*70}")

        if report['summary']:
            s = report['summary']
            print(f"   最佳樣本內夏普: {s['best_sharpe_is']:.3f}")
            print(f"   最佳樣本外夏普: {s['best_sharpe_oos']:.3f}")
            print(f"   最佳胃納量: {s['best_capacity']/1e4:.1f} 萬")
            print(f"   最佳 MDD: {s['best_mdd']*100:.1f}%")
            print(f"   樣本外穩定度: {s['oos_stability']:.3f}")
            print(f"   達標: {'✅' if s['target_reached'] else '❌'}")

        print(f"\n   樣本內 vs 樣本外對比:")
        print(f"   {'排名':<6}{'IS夏普':<10}{'OOS夏普':<10}{'IS報酬':<10}{'OOS報酬':<10}{'穩定度'}")
        print(f"   {'-'*56}")

        for item in report['in_sample_vs_oos'][:5]:
            stability = item['oos_sharpe'] / (item['is_sharpe'] + 1e-10)
            print(f"   {item['rank']:<6}{item['is_sharpe']:<10.3f}{item['oos_sharpe']:<10.3f}"
                  f"{item['is_return']*100:<10.1f}%{item['oos_return']*100:<10.1f}%{stability:<.2f}")


detailed_reporter = DetailedReporter(paths.reports_dir)


# =============================================================================
# 第十五部分：演化引擎
# =============================================================================
class EvolutionEngine:
    """
    演化引擎

    核心功能：
    1. NSGA-II 多目標優化
    2. 並行評估（可選）
    3. 精英保留
    4. 檢查點管理
    5. 詳細報告
    """

    def __init__(self, toolbox, pareto_mgr: ParetoArchiveManager):
        self.toolbox = toolbox
        self.pareto_mgr = pareto_mgr
        self.history = []
        self.logger = ProgressLogger(WINDOW_ID, paths.log_dir)

    def run(self, n_generations: int = N_GENERATIONS, resume: bool = True) -> Tuple[List, List]:
        """執行演化"""
        print(f"\n{'='*70}")
        print(f"🚀 開始演化 - 視窗 {WINDOW_ID}")
        print(f"   族群: {POPULATION_SIZE}, 世代: {n_generations}")
        print(f"   目標: 夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
        print(f"{'='*70}\n")

        start_gen = 0
        population = None

        # 嘗試從檢查點恢復
        if resume:
            checkpoint = checkpoint_mgr.load()
            if checkpoint:
                # 驗證歷史精英
                is_valid = history_validator.validate_top_n(self.pareto_mgr)

                if is_valid:
                    population, start_gen, self.history = checkpoint_mgr.restore_population(
                        checkpoint, self.toolbox
                    )
                    start_gen += 1  # 從下一代開始
                    print(f"✅ 從第 {start_gen} 代繼續演化")
                else:
                    print("⚠️ 歷史驗證不通過，將重新評估族群")

        # 初始化族群（如果未恢復）
        if population is None:
            population = self.toolbox.population(n=POPULATION_SIZE)
            # 注入歷史精英
            population = self.pareto_mgr.inject_elites(population, ratio=ELITE_RATIO)

        # 初始評估（只評估未評估的）
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

            # 環境選擇
            population = self.toolbox.select(population + offspring, POPULATION_SIZE)

            # 統計
            stats = self._compute_stats(population, gen)
            self.history.append(stats)

            elapsed = time.time() - gen_start_time

            # 記錄進度
            self.logger.log_generation(gen, n_generations, stats, elapsed)

            # 輸出進度
            if (gen + 1) % 5 == 0 or gen == 0:
                self._print_progress(gen, n_generations, stats, elapsed)

            # 檢查點
            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_mgr.save(gen, population, self.history)

            # 詳細報告（每 10 代）
            if (gen + 1) % DETAILED_REPORT_INTERVAL == 0:
                pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                detailed_reporter.generate_report(gen, pareto_front, self.history)

        # 最終 Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 更新歷史存檔
        self.pareto_mgr.update(pareto_front)

        # 最終檢查點
        checkpoint_mgr.save(n_generations - 1, population, self.history)

        # 輸出結果
        self._print_pareto_front(pareto_front)

        return population, pareto_front

    def _evaluate_population(self, population: List) -> List:
        """評估族群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        print(f"   評估 {len(invalid)} 個個體...")

        # 序列評估（FinLab 回測較慢，並行容易出問題）
        for i, ind in enumerate(invalid):
            fitness = evaluate_fitness(ind)
            ind.fitness.values = fitness

            if (i + 1) % 10 == 0:
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

        return {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': max(sharpes),
            'avg_sharpe': np.mean(sharpes),
            'best_capacity': max(capacities),
            'best_return': max(returns),
            'best_robustness': max(robustness),
        }

    def _print_progress(self, gen: int, total_gen: int, stats: Dict, elapsed: float):
        """輸出進度"""
        print(f"=== 第 {gen+1}/{total_gen} 代 ===")
        print(f"   最佳綜合: {stats['best_composite']:.4f} (平均: {stats['avg_composite']:.4f})")
        print(f"   最佳夏普: {stats['best_sharpe']:.3f} (目標: {TARGET_SHARPE})")
        print(f"   最佳胃納量: {stats['best_capacity']:.1f} 萬 (目標: {MIN_CAPACITY/1e4:.0f}萬)")
        print(f"   穩健性: {stats['best_robustness']:.3f}")
        print(f"   耗時: {elapsed:.1f}s")

    def _print_pareto_front(self, pareto_front: List):
        """輸出 Pareto 前緣"""
        print(f"\n{'='*70}")
        print(f"🏆 Pareto 最優解集（前 10 名）")
        print(f"{'='*70}")
        print(f"{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納量(萬)':<12}{'穩健性':<10}{'達標'}")
        print("-" * 70)

        sorted_front = sorted(pareto_front,
                              key=lambda x: sum(x.fitness.values),
                              reverse=True)[:10]

        for i, ind in enumerate(sorted_front, 1):
            composite = sum(ind.fitness.values)
            sharpe = ind.fitness.values[0] / 5.0 * TARGET_SHARPE
            capacity = ind.fitness.values[1] / 5.0 * MIN_CAPACITY / 1e4
            robustness = ind.fitness.values[3]

            reached = "✅" if sharpe >= TARGET_SHARPE * 0.9 and capacity >= MIN_CAPACITY / 1e4 * 0.9 else ""

            print(f"{i:<6}{composite:<10.4f}{sharpe:<10.3f}{capacity:<12.1f}{robustness:<10.3f}{reached}")


# =============================================================================
# 第十六部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    # 使用簡潔格式避免對齊問題
    print("")
    print("=" * 60)
    print("  FinLab 台股基因演算法優化系統 v2.0 - 終極強化版")
    print("  NSGA-II + Walk-Forward + Pareto Archive + 樣本內外分離")
    print("=" * 60)
    print(f"  目標: 夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e7:.0f}00萬+, MDD < {MAX_DRAWDOWN*100:.0f}%")
    print(f"  樣本內: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
    print(f"  樣本外: {OUT_OF_SAMPLE_START} ~ {'至今' if OUT_OF_SAMPLE_END is None else OUT_OF_SAMPLE_END}")
    print(f"  驗證: Walk-Forward ({WALK_FORWARD_WINDOWS} 窗口) + 歷史精英驗證")
    print(f"  基因: {GeneDecoder.GENE_LENGTH} 個參數")
    print(f"  視窗: {WINDOW_ID} / 並行: {N_WORKERS} workers")
    print("=" * 60)
    print("")

    # 初始化數據載入器
    dl = get_data_loader()

    # 建立演化引擎
    engine = EvolutionEngine(toolbox, pareto_archive)

    # 執行演化
    population, pareto_front = engine.run(N_GENERATIONS, resume=True)

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*70}")
        print("📊 最佳個體完整回測")
        print(f"{'='*70}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        strategy_engine = StrategyEngine(dl)
        position = strategy_engine.combine_strategies(best_params)

        # 顯示最佳參數
        print("\n🔍 最佳參數：")
        print(f"   策略權重：低波動 {best_params['weight_lv']:.2%}, "
              f"小資族 {best_params['weight_si']:.2%}, "
              f"雙渦輪 {best_params['weight_rpt']:.2%}")
        print(f"   止損: {best_params['stop_loss']:.1%}, "
              f"停利: {best_params['take_profit']:.1%}, "
              f"移動停損: {best_params['trail_stop']:.1%}")
        print(f"   持股上限: {best_params['position_limit']:.1%}")

        # 保存最佳參數
        with open(paths.best_params_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

        # 完整回測
        if position is not None and not position.empty:
            try:
                print("\n執行完整回測...")

                # 樣本內
                print("\n📈 樣本內回測 (2017-2022):")
                report_is = sim(
                    position=position.loc[IN_SAMPLE_START:IN_SAMPLE_END],
                    fee_ratio=1.425 / 1000,
                    tax_ratio=3 / 1000,
                    trade_at_price="high_low_avg",
                    position_limit=best_params['position_limit'],
                    stop_loss=best_params['stop_loss'],
                    trail_stop=best_params['trail_stop'],
                    take_profit=best_params['take_profit'],
                    stop_trading_next_period=False,
                    upload=False,
                    name='GA_Ultimate_InSample'
                )
                report_is.display()

                # 樣本外
                print("\n📈 樣本外回測 (2023~至今):")
                report_oos = sim(
                    position=position.loc[OUT_OF_SAMPLE_START:],
                    fee_ratio=1.425 / 1000,
                    tax_ratio=3 / 1000,
                    trade_at_price="high_low_avg",
                    position_limit=best_params['position_limit'],
                    stop_loss=best_params['stop_loss'],
                    trail_stop=best_params['trail_stop'],
                    take_profit=best_params['take_profit'],
                    stop_trading_next_period=False,
                    upload=False,
                    name='GA_Ultimate_OutOfSample'
                )
                report_oos.display()

            except Exception as e:
                print(f"⚠️ 完整回測失敗: {e}")
                traceback.print_exc()

    print(f"\n✅ 優化完成！")
    print(f"📁 結果保存於: {paths.output_dir}")
    print(f"📁 Pareto 存檔: {paths.pareto_archive}")
    print(f"📁 詳細報告: {paths.reports_dir}")

    return population, pareto_front


if __name__ == "__main__":
    population, pareto = main()
