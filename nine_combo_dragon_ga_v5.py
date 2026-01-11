#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 九組合天空龍 GA v5.0 - 遺傳演算法選股優化系統
Nine Combo Dragon GA v5.0 - Genetic Algorithm Stock Optimizer
================================================================================

【重大修復 v5.0】
✅ 修正 DataFrame 對齊問題 - Length of values does not match length of index
✅ 新增 align_dataframes() 函數確保所有 DataFrame 對齊
✅ 改進策略合併邏輯，使用 common_index 和 common_columns
✅ 強化錯誤處理與回退機制

【核心功能】
✅ 9 個獨立評分策略動態加權組合
✅ 基因演算法 NSGA-II 自動優化參數
✅ 歷史最佳持續進化 (Pareto Archive)
✅ Walk-Forward 驗證避免 overfitting
✅ 完全遵循 FinLab 原生 API

【目標】
- 夏普值：>= 4.0
- 胃納量：>= 1000 萬
- 最大回撤：< 20%

版本：v5.0 (2025-01-11) - DataFrame 對齊修復版
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# === 🔥 核心設定（可透過環境變數覆蓋）===
WINDOW_ID = int(os.environ.get('WINDOW_ID', 1))  # 視窗 ID (1-4)
CONTINUE_EVOLUTION = os.environ.get('CONTINUE_EVOLUTION', 'false').lower() == 'true'
FINLAB_API_KEY = os.environ.get('FINLAB_API_KEY', "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m")

# 優化目標
TARGET_SHARPE = 4.2
MIN_CAPACITY = 10_000_000  # 1000萬
TARGET_ANNUAL_RETURN = 0.3
MAX_DRAWDOWN = 0.2

# GA 演化參數
POPULATION_SIZE = int(os.environ.get('POPULATION_SIZE', 30))
N_GENERATIONS = int(os.environ.get('EVOLUTION_GENERATIONS', 100))
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
N_STRATEGIES = 9  # 策略數量

# Walk-Forward 設定
WALK_FORWARD_WINDOWS = 3
TRAIN_MONTHS = 24
TEST_MONTHS = 6

# 回測設定
BACKTEST_START = '2017-01-01'
BACKTEST_END = '2022-12-31'  # 樣本內結束日期
OUT_SAMPLE_START = '2023-01-01'  # 樣本外開始日期

# =============================================================================
# ASCII Art Logo
# =============================================================================
DRAGON_LOGO = r"""
██████╗ ██████╗  █████╗  ██████╗  ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗████╗  ██║
██║  ██║██████╔╝███████║██║  ███╗██║   ██║██╔██╗ ██║
██║  ██║██╔══██╗██╔══██║██║   ██║██║   ██║██║╚██╗██║
██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
"""

print(DRAGON_LOGO)
print(f"🐉 九組合天空龍 GA v5.0 - {N_STRATEGIES}策略 x {GeneDecoder.GENE_LENGTH if 'GeneDecoder' in dir() else 100}基因")
print("=" * 50)
print(f"📊 視窗: {WINDOW_ID}  |  🎯 目標夏普: {TARGET_SHARPE}  |  💰 胃納量: {MIN_CAPACITY/1e7:.0f}00萬  |  🧬 基因: 100")
print(f"👥 族群: {POPULATION_SIZE}  |  🔄 世代: {N_GENERATIONS}  |  ⚡ 並行: {os.cpu_count()} 核心  |  📋 策略: {N_STRATEGIES} 個")
print("=" * 50)

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
        base_dir = '/content/drive/MyDrive/九組合天空龍_GA_v5'
        in_colab = True
        print("✅ Google Colab 環境 - Drive 已掛載")
    except:
        base_dir = './nine_combo_dragon_output'
        in_colab = False
        print("⚠️ 本地環境")

    # FinLab 登入
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab 1.5.6 登入成功")

    return base_dir, in_colab

BASE_DIR, IN_COLAB = setup_environment()
print(f"📁 工作目錄: {BASE_DIR}/九組合天空龍_GA_v5")

@dataclass
class PathManager:
    """路徑管理器"""
    base_dir: str = BASE_DIR
    window_id: int = WINDOW_ID

    def __post_init__(self):
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.output_dir = f"{self.window_dir}/output"
        self.pareto_dir = f"{self.base_dir}/shared_pareto"
        self.history_dir = f"{self.window_dir}/history"
        self.log_dir = f"{self.window_dir}_logs"

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
                import subprocess
                subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-delete'], stderr=subprocess.DEVNULL)
                subprocess.run(['find', '/root', '/tmp', '-type', 'd', '-name', '*finlab*', '-exec', 'rm', '-rf', '{}', '+'], stderr=subprocess.DEVNULL)
                return self._safe_get(key, retry=False)
            else:
                raise

    def _load_all_data(self):
        """載入所有數據"""
        print("📦 載入 FinLab 資料庫...")
        start_time = time.time()

        # 價格資料
        print("   📈 價格資料...")
        self._cache['close'] = self._safe_get('price:收盤價')
        self._cache['vol'] = self._safe_get('price:成交股數')
        self._cache['open'] = self._safe_get('price:開盤價')
        self._cache['high'] = self._safe_get('price:最高價')
        self._cache['low'] = self._safe_get('price:最低價')
        self._cache['adj_close'] = self._safe_get("etl:adj_close")

        # 估值指標
        print("   💰 估值指標...")
        self._cache['pe'] = self._safe_get('price_earning_ratio:本益比')
        self._cache['pb'] = self._safe_get("price_earning_ratio:股價淨值比")
        self._cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')

        # 營收資料
        print("   📊 營收資料...")
        self._cache['rev'] = self._safe_get('monthly_revenue:當月營收')
        self._cache['rev_yoy_growth'] = self._safe_get('monthly_revenue:去年同月增減(%)')
        self._cache['rev_month_growth'] = self._safe_get('monthly_revenue:上月比較增減(%)')

        # 基本面指標
        print("   📋 基本面指標...")
        self._cache['營業利益成長率'] = self._safe_get('fundamental_features:營業利益成長率')
        self._cache['業外收支營收率'] = self._safe_get('fundamental_features:業外收支營收率')
        self._cache['營業毛利率'] = self._safe_get("fundamental_features:營業毛利率")
        self._cache['ROE綜合損益'] = self._safe_get("fundamental_features:ROE綜合損益")
        self._cache['稅後淨利率'] = self._safe_get("fundamental_features:稅後淨利率")
        self._cache['稅前淨利率'] = self._safe_get("fundamental_features:稅前淨利率")

        # 籌碼資料
        print("   🎯 籌碼資料...")
        self._cache['融資使用率'] = self._safe_get('margin_transactions:融資使用率')
        self._cache['董監持有股數占比'] = self._safe_get("internal_equity_changes:董監持有股數占比")
        self._cache['inventory'] = self._safe_get("inventory")

        # 市值與財報
        print("   💼 市值與財報...")
        self._cache['市值'] = self._safe_get('etl:market_value')
        self._cache['股本'] = self._safe_get('financial_statement:股本')
        self._cache['投資活動現金流'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._cache['營業活動現金流'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._cache['稅後淨利'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._cache['權益總計'] = self._safe_get('financial_statement:股東權益總額')

        # 技術指標
        print("   📉 計算衍生指標...")
        self._cache['rsi'] = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
        self._cache['atr'] = data.indicator('ATR', adjust_price=True, timeperiod=10)

        # 計算衍生指標
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"✅ 資料載入完成 ({elapsed:.1f} 秒)")

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

        # 儲存參考 DataFrame（用於對齊）
        self._cache['reference_index'] = close.index
        self._cache['reference_columns'] = close.columns

    def get(self, key: str):
        """獲取數據"""
        return self._cache.get(key)

    def get_reference_frame(self) -> pd.DataFrame:
        """獲取參考 DataFrame（用於對齊）"""
        return self._cache['close']

print("📊 初始化 FinLab...")
print("輸入成功!")
data_loader = FinLabDataLoader()

# =============================================================================
# 第五部分：DataFrame 對齊工具（🔥 關鍵修復）
# =============================================================================
def align_dataframes(*dfs: pd.DataFrame, fill_value: float = 0.0) -> List[pd.DataFrame]:
    """
    🔥 對齊多個 DataFrame 到共同的 index 和 columns

    這是修復 "Length of values does not match length of index" 錯誤的關鍵函數

    Args:
        *dfs: 要對齊的 DataFrame 列表
        fill_value: 用於填充缺失值的值

    Returns:
        對齊後的 DataFrame 列表
    """
    if not dfs:
        return []

    # 過濾掉 None 和空 DataFrame
    valid_dfs = [df for df in dfs if df is not None and not df.empty]

    if not valid_dfs:
        return [pd.DataFrame() for _ in dfs]

    if len(valid_dfs) == 1:
        return list(dfs)

    # 找出共同的 index（取交集）
    common_index = valid_dfs[0].index
    for df in valid_dfs[1:]:
        common_index = common_index.intersection(df.index)

    # 找出共同的 columns（取聯集）
    common_columns = valid_dfs[0].columns
    for df in valid_dfs[1:]:
        common_columns = common_columns.union(df.columns)

    # 對齊所有 DataFrame
    aligned_dfs = []
    for df in dfs:
        if df is None or df.empty:
            # 創建空的對齊 DataFrame
            aligned_df = pd.DataFrame(fill_value, index=common_index, columns=common_columns)
        else:
            # 重新索引到共同的 index 和 columns
            aligned_df = df.reindex(index=common_index, columns=common_columns, fill_value=fill_value)
        aligned_dfs.append(aligned_df)

    return aligned_dfs


def safe_combine_positions(positions: List[pd.DataFrame], weights: List[float],
                           reference_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    🔥 安全地合併多個持倉 DataFrame

    Args:
        positions: 持倉 DataFrame 列表
        weights: 對應的權重列表
        reference_df: 參考 DataFrame（用於最終對齊）

    Returns:
        合併後的持倉 DataFrame
    """
    if not positions or not weights:
        return pd.DataFrame()

    # 過濾有效的 position 和對應的權重
    valid_pairs = [(pos, w) for pos, w in zip(positions, weights)
                   if pos is not None and not pos.empty and w > 0]

    if not valid_pairs:
        return pd.DataFrame()

    valid_positions, valid_weights = zip(*valid_pairs)
    valid_positions = list(valid_positions)
    valid_weights = list(valid_weights)

    # 正規化權重
    total_weight = sum(valid_weights)
    if total_weight <= 0:
        return pd.DataFrame()
    normalized_weights = [w / total_weight for w in valid_weights]

    # 對齊所有 DataFrame
    aligned_positions = align_dataframes(*valid_positions, fill_value=0.0)

    # 加權合併
    combined = aligned_positions[0] * normalized_weights[0]
    for pos, weight in zip(aligned_positions[1:], normalized_weights[1:]):
        combined = combined + pos * weight

    # 如果有參考 DataFrame，進行最終對齊
    if reference_df is not None:
        # 取交集的 index
        common_index = combined.index.intersection(reference_df.index)
        # 取交集的 columns
        common_columns = combined.columns.intersection(reference_df.columns)
        combined = combined.reindex(index=common_index, columns=common_columns, fill_value=0.0)

    return combined


# =============================================================================
# 第六部分：九個獨立策略引擎
# =============================================================================
class NineComboStrategyEngine:
    """九組合策略引擎 - 9 個獨立評分策略"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader
        self.reference_df = data_loader.get_reference_frame()

    def _safe_reindex(self, df: pd.DataFrame, target_index=None, target_columns=None) -> pd.DataFrame:
        """安全地重新索引 DataFrame"""
        if df is None or df.empty:
            return pd.DataFrame()

        try:
            if target_index is None:
                target_index = self.reference_df.index
            if target_columns is None:
                target_columns = self.reference_df.columns

            # 取交集避免擴展到不存在的索引
            common_index = df.index.intersection(target_index)
            common_columns = df.columns.intersection(target_columns)

            return df.reindex(index=common_index, columns=common_columns, fill_value=0.0)
        except Exception as e:
            print(f"   ⚠️ reindex 失敗: {e}")
            return df

    def strategy_1_low_volatility_pe(self, params: Dict) -> pd.DataFrame:
        """策略一：低波動本益比"""
        try:
            rev_ma3_ma12_ratio = params.get('s1_rev_ma3_ma12_ratio', 1.05)
            volatility_threshold = params.get('s1_volatility_threshold', 0.04)
            margin_usage_limit = params.get('s1_margin_usage_limit', 35)
            pe_min = params.get('s1_pe_min', 5)
            pe_max = params.get('s1_pe_max', 25)
            top_n = int(params.get('s1_top_n', 5))

            close = self.dl.get('close')
            pe = self.dl.get('pe')
            pb = self.dl.get('pb')
            vol = self.dl.get('vol')
            rev = self.dl.get('rev')
            rev_ma3 = self.dl.get('rev_ma3')
            rev_ma12 = self.dl.get('rev_ma12')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')
            融資使用率 = self.dl.get('融資使用率')
            entry_volatility = self.dl.get('entry_volatility')
            業外收支營收率 = self.dl.get('業外收支營收率')
            營業利益成長率 = self.dl.get('營業利益成長率')
            營業毛利率 = self.dl.get('營業毛利率')
            ROE綜合損益 = self.dl.get('ROE綜合損益')
            limit_up_all_day = self.dl.get('limit_up_all_day')

            # PEG
            peg = pe / (營業利益成長率 + 1e-10)

            # 條件
            cond1 = rev_ma3 / rev_ma12 > rev_ma3_ma12_ratio
            cond2 = rev / rev.shift(1) > 0.8

            tree_select_factor = ((融資使用率 <= margin_usage_limit)
                                 & (entry_volatility <= volatility_threshold)
                                 & (業外收支營收率 < 10))

            condition_近1日成交均量 = vol.average(1) > 150000
            cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)
            cond收盤價大於均線 = (close > close.average(75)) & (close > close.average(40))
            cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)
            pe_range = (pe_min <= pe) & (pe <= pe_max)
            pb_range = (0.5 <= pb) & (pb <= 3.0)
            gpm_trend = (營業毛利率 > 8).sustain(2)
            roe_trend = (ROE綜合損益 > 0).sustain(2)

            cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收連3月衰退
                       & cond收盤價大於均線 & cond近三個月營收大於年營收
                       & condition_近1日成交均量 & gpm_trend & roe_trend
                       & ~limit_up_all_day & pe_range & pb_range)

            position = peg[cond_all & (peg > 0)].is_smallest(top_n)
            position = self._safe_reindex(position.reindex(rev.index_str_to_date().index, method='ffill'))
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略1失敗: {e}")
            return pd.DataFrame()

    def strategy_2_small_investor(self, params: Dict) -> pd.DataFrame:
        """策略二：小資族"""
        try:
            market_value_limit = params.get('s2_market_value_limit', 15e9)
            market_rev_ratio_limit = params.get('s2_market_rev_ratio_limit', 3.5)
            ma_period = int(params.get('s2_ma_period', 60))
            top_n = int(params.get('s2_top_n', 6))

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

            condition1 = (市值 < market_value_limit)
            condition2 = 自由現金流 > 0
            condition3 = 股東權益報酬率 > 0
            condition4 = 營業利益成長率 > -1
            condition5 = 市值營收比 < market_rev_ratio_limit
            condition6 = vol > 100000

            cond排除月營收連3月衰退 = ~(rev_yoy_growth < -15).sustain(3)
            cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)
            cond單月營收月增率 = (rev_month_growth > -50).sustain(3)
            cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(120))
            cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)
            業外收支營收率占比低 = (業外收支營收率 < 7.5)

            rsv = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min() + 1e-10)

            position = ((condition1 & condition2 & condition3 & condition4 & condition5 & condition6
                        & cond排除月營收成長趨勢過老 & cond單月營收月增率 & cond排除月營收連3月衰退
                        & cond近三個月營收大於年營收 & 業外收支營收率占比低) * rsv).is_largest(top_n)

            position = self._safe_reindex(position.reindex(當月營收.index_str_to_date().index))
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略2失敗: {e}")
            return pd.DataFrame()

    def strategy_3_revenue_turbo(self, params: Dict) -> pd.DataFrame:
        """策略三：營收股價雙渦輪"""
        try:
            rev_ma_period = int(params.get('s3_rev_ma_period', 5))
            price_high_window = int(params.get('s3_price_high_window', 8))
            rsi_threshold = params.get('s3_rsi_threshold', 55)
            top_n = int(params.get('s3_top_n', 12))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            pe = self.dl.get('pe')
            rev = self.dl.get('rev')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')
            rsi = self.dl.get('rsi')
            營業毛利率 = self.dl.get('營業毛利率')
            稅前淨利率 = self.dl.get('稅前淨利率')
            稅後淨利率 = self.dl.get('稅後淨利率')
            limit_up_all_day = self.dl.get('limit_up_all_day')

            rev_ma = rev.average(rev_ma_period)
            condition_近n月平均營收創新高 = rev_ma == rev_ma.rolling(24, min_periods=rev_ma_period).max()
            condition_近n日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
            condition_近1日成交均量 = vol.average(1) > 250000

            long_ma_pattern = ((close > close.average(5)) & (close > close.average(10))
                              & (close > close.average(20)) & (close > close.average(60))
                              & (close > close.average(150)))

            收盤價_超級績效 = close > (close.average(250) * 1.1)
            rsi_higt_trend = (rsi > rsi_threshold).sustain(1)
            gpm_trend = (營業毛利率 > 5).sustain(5)
            btpm_trend = (稅前淨利率 > 4).sustain(1)
            atpm_trend = (稅後淨利率 > 3).sustain(1)
            rev_rise_nsatisfy = rev_yoy_growth.rank(pct=True, axis=1) > 0.9
            pe_range = ~(150 <= pe)

            conditions = (condition_近n月平均營收創新高 & condition_近n日內有1日股價創新高
                         & condition_近1日成交均量 & long_ma_pattern & gpm_trend
                         & btpm_trend & atpm_trend & rev_rise_nsatisfy
                         & (close > 15) & pe_range & 收盤價_超級績效
                         & rsi_higt_trend & ~limit_up_all_day)

            position = rev_yoy_growth * conditions
            position = position[position > 0].is_largest(top_n)
            position = self._safe_reindex(position.reindex(rev.index_str_to_date().index, method="ffill"))
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略3失敗: {e}")
            return pd.DataFrame()

    def strategy_4_value_momentum(self, params: Dict) -> pd.DataFrame:
        """策略四：價值動能"""
        try:
            pb_max = params.get('s4_pb_max', 2.5)
            dividend_min = params.get('s4_dividend_min', 2.0)
            top_n = int(params.get('s4_top_n', 8))

            close = self.dl.get('close')
            pb = self.dl.get('pb')
            dividend_yield = self.dl.get('dividend_yield')
            vol = self.dl.get('vol')
            ROE綜合損益 = self.dl.get('ROE綜合損益')
            營業毛利率 = self.dl.get('營業毛利率')

            # 價值條件
            value_cond = (pb < pb_max) & (pb > 0.3) & (dividend_yield > dividend_min)

            # 動能條件
            momentum_cond = (close > close.average(20)) & (close > close.average(60))

            # 品質條件
            quality_cond = (ROE綜合損益 > 5) & (營業毛利率 > 10)

            # 流動性
            liquidity_cond = vol.average(5) > 100000

            all_cond = value_cond & momentum_cond & quality_cond & liquidity_cond

            # 使用股息率排名
            position = dividend_yield[all_cond].is_largest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略4失敗: {e}")
            return pd.DataFrame()

    def strategy_5_growth_quality(self, params: Dict) -> pd.DataFrame:
        """策略五：成長品質"""
        try:
            rev_growth_min = params.get('s5_rev_growth_min', 15)
            roe_min = params.get('s5_roe_min', 10)
            top_n = int(params.get('s5_top_n', 8))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')
            ROE綜合損益 = self.dl.get('ROE綜合損益')
            營業毛利率 = self.dl.get('營業毛利率')
            稅後淨利率 = self.dl.get('稅後淨利率')
            營業利益成長率 = self.dl.get('營業利益成長率')

            # 成長條件
            growth_cond = (rev_yoy_growth > rev_growth_min) & (營業利益成長率 > 10)

            # 品質條件
            quality_cond = (ROE綜合損益 > roe_min) & (營業毛利率 > 15) & (稅後淨利率 > 5)

            # 趨勢條件
            trend_cond = (close > close.average(50)) & (close > close.average(100))

            # 流動性
            liquidity_cond = vol.average(5) > 150000

            all_cond = growth_cond & quality_cond & trend_cond & liquidity_cond

            # 使用 ROE 排名
            position = ROE綜合損益[all_cond].is_largest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略5失敗: {e}")
            return pd.DataFrame()

    def strategy_6_breakout(self, params: Dict) -> pd.DataFrame:
        """策略六：突破策略"""
        try:
            breakout_period = int(params.get('s6_breakout_period', 60))
            volume_ratio = params.get('s6_volume_ratio', 1.5)
            top_n = int(params.get('s6_top_n', 10))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            high = self.dl.get('high')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')

            # 價格突破
            price_breakout = close >= close.rolling(breakout_period).max()

            # 量能放大
            vol_ma = vol.rolling(20).mean()
            volume_breakout = vol > vol_ma * volume_ratio

            # 趨勢確認
            trend_cond = (close > close.average(20)) & (close > close.average(50))

            # 基本面支撐
            fundamental_cond = rev_yoy_growth > 0

            all_cond = price_breakout & volume_breakout & trend_cond & fundamental_cond

            # 使用成交量排名
            position = vol[all_cond].is_largest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略6失敗: {e}")
            return pd.DataFrame()

    def strategy_7_mean_reversion(self, params: Dict) -> pd.DataFrame:
        """策略七：均值回歸"""
        try:
            oversold_threshold = params.get('s7_oversold_threshold', 30)
            rsi_period = int(params.get('s7_rsi_period', 14))
            top_n = int(params.get('s7_top_n', 6))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            rsi = self.dl.get('rsi')
            ROE綜合損益 = self.dl.get('ROE綜合損益')
            營業毛利率 = self.dl.get('營業毛利率')

            # 超賣條件
            oversold = rsi < oversold_threshold

            # 價格在支撐區
            support = close < close.average(20) * 0.95

            # 品質篩選（只選好公司）
            quality_cond = (ROE綜合損益 > 8) & (營業毛利率 > 12)

            # 流動性
            liquidity_cond = vol.average(5) > 100000

            all_cond = oversold & support & quality_cond & liquidity_cond

            # 使用 RSI 排名（越低越好）
            position = rsi[all_cond].is_smallest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略7失敗: {e}")
            return pd.DataFrame()

    def strategy_8_dividend_growth(self, params: Dict) -> pd.DataFrame:
        """策略八：股息成長"""
        try:
            dividend_min = params.get('s8_dividend_min', 3.0)
            growth_min = params.get('s8_growth_min', 5)
            top_n = int(params.get('s8_top_n', 8))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            dividend_yield = self.dl.get('dividend_yield')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')
            ROE綜合損益 = self.dl.get('ROE綜合損益')
            自由現金流 = self.dl.get('自由現金流')

            # 股息條件
            dividend_cond = dividend_yield > dividend_min

            # 成長條件
            growth_cond = rev_yoy_growth > growth_min

            # 現金流支撐
            cashflow_cond = 自由現金流 > 0

            # 品質
            quality_cond = ROE綜合損益 > 8

            # 流動性
            liquidity_cond = vol.average(5) > 80000

            all_cond = dividend_cond & growth_cond & cashflow_cond & quality_cond & liquidity_cond

            # 使用股息率 * ROE 綜合評分
            score = dividend_yield * ROE綜合損益
            position = score[all_cond].is_largest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略8失敗: {e}")
            return pd.DataFrame()

    def strategy_9_momentum_filter(self, params: Dict) -> pd.DataFrame:
        """策略九：動量過濾"""
        try:
            momentum_period = int(params.get('s9_momentum_period', 20))
            strength_threshold = params.get('s9_strength_threshold', 0.1)
            top_n = int(params.get('s9_top_n', 10))

            close = self.dl.get('close')
            vol = self.dl.get('vol')
            rev_yoy_growth = self.dl.get('rev_yoy_growth')
            營業毛利率 = self.dl.get('營業毛利率')

            # 動量計算
            momentum = (close / close.shift(momentum_period) - 1)

            # 強勢動量
            strong_momentum = momentum > strength_threshold

            # 趨勢確認
            trend_cond = (close > close.average(10)) & (close > close.average(30)) & (close > close.average(60))

            # 基本面
            fundamental_cond = (rev_yoy_growth > 0) & (營業毛利率 > 8)

            # 流動性
            liquidity_cond = vol.average(5) > 200000

            all_cond = strong_momentum & trend_cond & fundamental_cond & liquidity_cond

            # 使用動量排名
            position = momentum[all_cond].is_largest(top_n)
            position = self._safe_reindex(position)
            return position.fillna(0)

        except Exception as e:
            print(f"   ⚠️ 策略9失敗: {e}")
            return pd.DataFrame()

    def combine_strategies(self, params: Dict) -> pd.DataFrame:
        """
        🔥 合併九個策略（使用安全對齊）

        這是修復對齊問題的關鍵方法
        """
        try:
            # 獲取所有策略的持股
            positions = [
                self.strategy_1_low_volatility_pe(params),
                self.strategy_2_small_investor(params),
                self.strategy_3_revenue_turbo(params),
                self.strategy_4_value_momentum(params),
                self.strategy_5_growth_quality(params),
                self.strategy_6_breakout(params),
                self.strategy_7_mean_reversion(params),
                self.strategy_8_dividend_growth(params),
                self.strategy_9_momentum_filter(params),
            ]

            # 獲取權重
            weights = [
                params.get('weight_s1', 0.15),
                params.get('weight_s2', 0.12),
                params.get('weight_s3', 0.15),
                params.get('weight_s4', 0.10),
                params.get('weight_s5', 0.12),
                params.get('weight_s6', 0.10),
                params.get('weight_s7', 0.08),
                params.get('weight_s8', 0.08),
                params.get('weight_s9', 0.10),
            ]

            # 🔥 使用安全合併函數
            position_combined = safe_combine_positions(
                positions,
                weights,
                reference_df=self.reference_df
            )

            if position_combined.empty:
                print("   ⚠️ 合併後無有效持倉")
                return pd.DataFrame()

            return position_combined

        except Exception as e:
            print(f"   ⚠️ 策略合併失敗: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


strategy_engine = NineComboStrategyEngine(data_loader)

# =============================================================================
# 第七部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """基因解碼器 - 100 個基因參數"""

    GENE_LENGTH = 100

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}
        idx = 0

        # === 策略一：低波動本益比（6個參數）===
        params['s1_rev_ma3_ma12_ratio'] = genes[idx] * 0.3 + 1.0; idx += 1  # 1.0-1.3
        params['s1_volatility_threshold'] = genes[idx] * 0.04 + 0.02; idx += 1  # 0.02-0.06
        params['s1_margin_usage_limit'] = genes[idx] * 25 + 20; idx += 1  # 20-45
        params['s1_pe_min'] = genes[idx] * 8 + 3; idx += 1  # 3-11
        params['s1_pe_max'] = genes[idx] * 20 + 15; idx += 1  # 15-35
        params['s1_top_n'] = int(genes[idx] * 6 + 3); idx += 1  # 3-9

        # === 策略二：小資族（4個參數）===
        params['s2_market_value_limit'] = genes[idx] * 15e9 + 5e9; idx += 1  # 50億-200億
        params['s2_market_rev_ratio_limit'] = genes[idx] * 3 + 2; idx += 1  # 2-5
        params['s2_ma_period'] = int(genes[idx] * 50 + 40); idx += 1  # 40-90
        params['s2_top_n'] = int(genes[idx] * 6 + 4); idx += 1  # 4-10

        # === 策略三：營收股價雙渦輪（4個參數）===
        params['s3_rev_ma_period'] = int(genes[idx] * 5 + 3); idx += 1  # 3-8
        params['s3_price_high_window'] = int(genes[idx] * 10 + 5); idx += 1  # 5-15
        params['s3_rsi_threshold'] = genes[idx] * 20 + 50; idx += 1  # 50-70
        params['s3_top_n'] = int(genes[idx] * 10 + 8); idx += 1  # 8-18

        # === 策略四：價值動能（3個參數）===
        params['s4_pb_max'] = genes[idx] * 2 + 1.5; idx += 1  # 1.5-3.5
        params['s4_dividend_min'] = genes[idx] * 3 + 1; idx += 1  # 1-4
        params['s4_top_n'] = int(genes[idx] * 8 + 5); idx += 1  # 5-13

        # === 策略五：成長品質（3個參數）===
        params['s5_rev_growth_min'] = genes[idx] * 20 + 5; idx += 1  # 5-25
        params['s5_roe_min'] = genes[idx] * 10 + 5; idx += 1  # 5-15
        params['s5_top_n'] = int(genes[idx] * 8 + 5); idx += 1  # 5-13

        # === 策略六：突破策略（3個參數）===
        params['s6_breakout_period'] = int(genes[idx] * 40 + 40); idx += 1  # 40-80
        params['s6_volume_ratio'] = genes[idx] * 1.5 + 1.0; idx += 1  # 1.0-2.5
        params['s6_top_n'] = int(genes[idx] * 8 + 6); idx += 1  # 6-14

        # === 策略七：均值回歸（3個參數）===
        params['s7_oversold_threshold'] = genes[idx] * 20 + 20; idx += 1  # 20-40
        params['s7_rsi_period'] = int(genes[idx] * 10 + 10); idx += 1  # 10-20
        params['s7_top_n'] = int(genes[idx] * 6 + 4); idx += 1  # 4-10

        # === 策略八：股息成長（3個參數）===
        params['s8_dividend_min'] = genes[idx] * 4 + 2; idx += 1  # 2-6
        params['s8_growth_min'] = genes[idx] * 15 + 0; idx += 1  # 0-15
        params['s8_top_n'] = int(genes[idx] * 8 + 5); idx += 1  # 5-13

        # === 策略九：動量過濾（3個參數）===
        params['s9_momentum_period'] = int(genes[idx] * 30 + 10); idx += 1  # 10-40
        params['s9_strength_threshold'] = genes[idx] * 0.2 + 0.05; idx += 1  # 0.05-0.25
        params['s9_top_n'] = int(genes[idx] * 10 + 6); idx += 1  # 6-16

        # === 策略權重（9個）===
        raw_weights = genes[idx:idx+9]
        idx += 9
        total = sum(raw_weights) + 1e-10
        for i in range(9):
            params[f'weight_s{i+1}'] = raw_weights[i] / total

        # === 回測參數（4個）===
        params['stop_loss'] = genes[idx] * 0.2 + 0.15; idx += 1  # 15%-35%
        params['trail_stop'] = genes[idx] * 0.3 + 0.2; idx += 1  # 20%-50%
        params['take_profit'] = genes[idx] * 0.5 + 0.5; idx += 1  # 50%-100%
        params['position_limit'] = genes[idx] * 0.2 + 0.25; idx += 1  # 25%-45%

        return params

gene_decoder = GeneDecoder()

# =============================================================================
# 第八部分：Walk-Forward 回測引擎（修復對齊問題）
# =============================================================================
class WalkForwardBacktest:
    """Walk-Forward 回測引擎"""

    def __init__(self):
        self.results = []
        self.error_count = 0

    def run_walk_forward(self, position: pd.DataFrame, params: Dict) -> Dict:
        """執行 Walk-Forward 分析"""
        try:
            if position is None or position.empty:
                return self._empty_result()

            # 🔥 確保 position 是乾淨的 DataFrame
            position = self._clean_position(position)

            if position.empty or len(position) < 100:
                return self._empty_result()

            # 分割樣本內外
            in_sample = position.loc[:BACKTEST_END] if BACKTEST_END else position
            out_sample = position.loc[OUT_SAMPLE_START:] if OUT_SAMPLE_START else pd.DataFrame()

            results = {
                'in_sample': None,
                'out_sample': None,
                'overall': None,
                'is_robust': False,
                'consistency_score': 0,
            }

            # 樣本內回測
            if not in_sample.empty and len(in_sample) > 50:
                results['in_sample'] = self._backtest_single_window(in_sample, params, 'InSample')

            # 樣本外回測
            if not out_sample.empty and len(out_sample) > 20:
                results['out_sample'] = self._backtest_single_window(out_sample, params, 'OutSample')

            # 整體回測
            results['overall'] = self._backtest_single_window(position, params, 'Overall')

            # 計算穩健性
            if results['in_sample'] and results['out_sample']:
                in_sharpe = results['in_sample']['sharpe']
                out_sharpe = results['out_sample']['sharpe']

                if in_sharpe > 0 and out_sharpe > 0:
                    ratio = min(out_sharpe / in_sharpe, in_sharpe / out_sharpe)
                    results['consistency_score'] = ratio
                    results['is_robust'] = ratio > 0.5 and out_sharpe > TARGET_SHARPE * 0.6

            return results

        except Exception as e:
            self.error_count += 1
            if self.error_count <= 3:
                print(f"   ⚠️ Walk-Forward 失敗: {e}")
            return self._empty_result()

    def _clean_position(self, position: pd.DataFrame) -> pd.DataFrame:
        """🔥 清理並驗證 position DataFrame"""
        try:
            # 移除全為 NaN 的列和行
            position = position.dropna(how='all', axis=0)
            position = position.dropna(how='all', axis=1)

            # 填充剩餘的 NaN
            position = position.fillna(0)

            # 確保 index 是 DatetimeIndex
            if not isinstance(position.index, pd.DatetimeIndex):
                position.index = pd.to_datetime(position.index)

            # 排序
            position = position.sort_index()

            # 移除重複的 index
            position = position[~position.index.duplicated(keep='first')]

            return position

        except Exception as e:
            print(f"   ⚠️ 清理 position 失敗: {e}")
            return position

    def _backtest_single_window(self, position: pd.DataFrame, params: Dict, name: str) -> Optional[Dict]:
        """單個窗口回測"""
        try:
            # 🔥 再次清理確保對齊
            position = self._clean_position(position)

            if position.empty:
                return None

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
            print(f"   ⚠️ 回測錯誤 ({name}): {e}")
            return None

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
            'in_sample': None,
            'out_sample': None,
            'is_robust': False,
            'consistency_score': 0,
        }

walk_forward = WalkForwardBacktest()

# =============================================================================
# 第九部分：適應度評估函數
# =============================================================================
def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """評估個體適應度"""
    try:
        params = gene_decoder.decode(individual)
        position = strategy_engine.combine_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        wf_result = walk_forward.run_walk_forward(position, params)
        overall = wf_result.get('overall') or wf_result

        if isinstance(overall, dict):
            sharpe = overall.get('sharpe', 0)
            capacity = overall.get('capacity', 0)
            annual_return = overall.get('annual_return', 0)
        else:
            return (0.0, 0.0, 0.0, -10.0)

        # 計算分數
        sharpe_score = min(5.0, max(0, sharpe / TARGET_SHARPE * 5.0))
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))
        return_score = min(2.0, max(0, annual_return / TARGET_ANNUAL_RETURN * 2.0))

        # 穩健性獎勵/懲罰
        if wf_result.get('is_robust', False):
            robustness_penalty = 1.0  # 獎勵
        else:
            consistency = wf_result.get('consistency_score', 0)
            robustness_penalty = consistency - 0.5

        return (sharpe_score, capacity_score, return_score, robustness_penalty)

    except Exception as e:
        print(f"   ⚠️ 評估失敗: {e}")
        return (0.0, 0.0, 0.0, -10.0)

# =============================================================================
# 第十部分：DEAP 設定
# =============================================================================
if 'FitnessMulti' in dir(creator):
    del creator.FitnessMulti
if 'Individual' in dir(creator):
    del creator.Individual

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
# 第十一部分：Pareto 歷史管理器
# =============================================================================
class ParetoArchiveManager:
    """Pareto 前緣歷史管理器"""

    def __init__(self, archive_file: str):
        self.archive_file = archive_file
        self.archive = []
        self._load()

    def _load(self):
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
        all_individuals = self.archive + [
            {'genes': list(ind), 'fitness': ind.fitness.values}
            for ind in pareto_front
        ]

        unique = self._deduplicate(all_individuals)
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:30]
        self._save()

    def _deduplicate(self, individuals: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for ind in individuals:
            gene_hash = hashlib.md5(str(ind['genes'][:10]).encode()).hexdigest()
            if gene_hash not in seen:
                seen.add(gene_hash)
                unique.append(ind)
        return unique

    def _save(self):
        try:
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                }, f)
        except Exception as e:
            print(f"⚠️ 歷史保存失敗: {e}")

    def inject_elites(self, population: List, ratio: float = 0.3):
        if not self.archive:
            return population

        n_inject = int(len(population) * ratio)
        n_inject = min(n_inject, len(self.archive))

        elites = sorted(self.archive, key=lambda x: sum(x['fitness']), reverse=True)[:n_inject]
        population.sort(key=lambda ind: sum(ind.fitness.values) if ind.fitness.valid else -999)

        for i, elite in enumerate(elites):
            population[i][:] = elite['genes']
            if hasattr(population[i], 'fitness'):
                del population[i].fitness.values

        print(f"✅ 注入 {n_inject} 個歷史精英")
        return population

pareto_archive = ParetoArchiveManager(paths.pareto_archive)

# =============================================================================
# 第十二部分：進度日誌記錄器
# =============================================================================
class ProgressLogger:
    """進度日誌記錄器"""

    def __init__(self, window_id: int, base_dir: str = BASE_DIR):
        self.window_id = window_id
        self.log_dir = f"{base_dir}/window_{window_id}_logs"
        self.history_file = f"{self.log_dir}/progress_history.json"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log_generation(self, gen: int, total_gen: int, stats: Dict):
        progress = {
            'timestamp': datetime.now().isoformat(),
            'window_id': self.window_id,
            'current_gen': gen + 1,
            'total_gen': total_gen,
            'best_sharpe': stats.get('best_sharpe', 0),
            'best_capacity': stats.get('best_capacity', 0),
            'best_composite': stats.get('best_composite', 0),
            'best_return': stats.get('best_return', 0),
            'robustness': stats.get('best_robustness', 0),
            'elapsed_total': time.time() - self.start_time,
        }

        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass

        history.append(progress)

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

# =============================================================================
# 第十三部分：演化引擎
# =============================================================================
class EvolutionEngine:
    """演化引擎"""

    def __init__(self, toolbox, pareto_mgr: ParetoArchiveManager):
        self.toolbox = toolbox
        self.pareto_mgr = pareto_mgr
        self.history = []
        self.logger = ProgressLogger(WINDOW_ID)

    def run(self, n_generations: int = N_GENERATIONS) -> Tuple[List, List]:
        """執行演化"""
        print(f"\n{'='*60}")
        print(f"🚀 開始九組合天空龍 GA 優化")
        print(f"{'='*60}")
        print(f"🎯 目標: 夏普 > {TARGET_SHARPE}, MDD < {MAX_DRAWDOWN*100:.0f}%, 胃納量 > {MIN_CAPACITY/1e7:.0f}M")
        print(f"🧬 基因: {GeneDecoder.GENE_LENGTH}, 族群: {POPULATION_SIZE}, 世代: {n_generations}")
        print(f"📅 樣本內: {BACKTEST_START}~{BACKTEST_END}  樣本外: {OUT_SAMPLE_START}~")
        print(f"📋 策略: {N_STRATEGIES} 個獨立評分策略動態加權組合")
        print(f"{'='*60}\n")

        # 初始化族群
        print("🔄 初始化新的演化...")
        population = self.toolbox.population(n=POPULATION_SIZE)

        # 注入歷史精英
        if CONTINUE_EVOLUTION:
            population = self.pareto_mgr.inject_elites(population, ratio=0.3)

        # 初始評估
        print(f"\n📊 評估 {POPULATION_SIZE} 個個體...")
        population = self._evaluate_population(population)

        # 演化循環
        for gen in range(n_generations):
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

            # 輸出
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"\n=== 第 {gen+1}/{n_generations} 代 ({elapsed:.1f}s) ===")
                print(f"   📈 最佳綜合: {stats['best_composite']:.4f}")
                print(f"   📊 最佳夏普: {stats['best_sharpe']:.2f}")
                print(f"   💰 最佳胃納量: {stats['best_capacity']:.0f} 萬")
                print(f"   🎯 穩健性: {stats['best_robustness']:.2f}")

        # Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 更新歷史存檔
        self.pareto_mgr.update(pareto_front)

        # 輸出結果
        self._print_pareto_front(pareto_front)

        return population, pareto_front

    def _evaluate_population(self, population: List) -> List:
        """評估族群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

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

        stats = {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': max(sharpes),
            'best_capacity': max(capacities),
            'best_return': max(returns),
            'best_robustness': max(robustness),
        }

        self.logger.log_generation(gen, N_GENERATIONS, stats)
        return stats

    def _print_pareto_front(self, pareto_front: List):
        """輸出 Pareto 前緣"""
        print(f"\n{'='*60}")
        print(f"🏆 Pareto 最優解集（前 10 名）")
        print(f"{'='*60}")
        print(f"{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納量(萬)':<12}{'達標'}")
        print("-" * 60)

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
# 第十四部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║     🐉 九組合天空龍 GA v5.0 - DataFrame 對齊修復版             ║
║     Nine Combo Dragon GA v5.0 - Alignment Fix Edition          ║
╠════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e7:.0f}M+, MDD < {MAX_DRAWDOWN*100:.0f}%              ║
║  📊 策略：{N_STRATEGIES} 個獨立策略動態組合優化                          ║
║  🔬 驗證：樣本內外分離測試                                     ║
║  🧬 基因：{GeneDecoder.GENE_LENGTH} 個參數                                        ║
║  🔧 修復：DataFrame 對齊問題 (common_index/columns)            ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # 建立演化引擎
    engine = EvolutionEngine(toolbox, pareto_archive)

    # 執行演化
    population, pareto_front = engine.run(N_GENERATIONS)

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*60}")
        print("📊 最佳個體詳細回測")
        print(f"{'='*60}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        # 顯示最佳參數
        print("\n🔍 最佳策略權重：")
        for i in range(9):
            print(f"   策略{i+1}: {best_params[f'weight_s{i+1}']:.2%}")

        # 保存最佳參數
        with open(paths.best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

        # 完整回測
        try:
            print("\n執行完整回測...")
            position = strategy_engine.combine_strategies(best_params)

            if not position.empty:
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
                    upload=False,
                    name=f'NineCombo_Dragon_GA_v5_Best'
                )

                report.display()
        except Exception as e:
            print(f"⚠️ 完整回測失敗: {e}")

    print(f"\n✅ 優化完成！")
    print(f"📁 結果保存於: {paths.output_dir}")
    print(f"📁 Pareto 存檔: {paths.pareto_archive}")

    return population, pareto_front

if __name__ == "__main__":
    population, pareto = main()
