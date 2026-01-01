#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 快快龍 v12.0 - 跨視窗協作優化系統
Kuai-Kuai-Long v12.0 - Cross-Window Collaborative Optimizer
================================================================================

【v12.0 核心改進】
✅ 從指定 checkpoint 開始演化（非歷史最佳）
✅ 跨視窗 Pareto 共享（可選）
✅ Discord 即時通知
✅ 命令行參數支持
✅ 自動快取修復機制

【使用方式】
WINDOW_ID=1 \\
CHECKPOINT_PATH='/path/to/checkpoint.pkl' \\
python genetic_algo_v12_cross_window.py --window 1

【環境變數】
- WINDOW_ID: 視窗 ID (1-4)
- FINLAB_API_KEY: FinLab API 金鑰
- DISCORD_WEBHOOK_URL: Discord 通知 URL（可選）
- GOOGLE_DRIVE_PATH: Google Drive 路徑
- CHECKPOINT_PATH: 指定從此 checkpoint 開始（可選）
- DISABLE_PARETO_INJECT: 設為 '1' 禁用歷史最佳注入

版本：v12.0 Cross-Window (2026-01-01)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# 🔥 【重要】在 import 其他套件之前，先禁用 FinLab 快取
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
import argparse
import subprocess
import glob as glob_module
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# =============================================================================
# 第二部分：環境變數與參數解析
# =============================================================================
def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='快快龍 v12.0 跨視窗優化系統')
    parser.add_argument('--window', type=int, default=1, help='視窗 ID (1-4)')
    parser.add_argument('--generations', type=int, default=100, help='演化代數')
    parser.add_argument('--population', type=int, default=50, help='族群大小')
    parser.add_argument('--checkpoint', type=str, default=None, help='指定 checkpoint 檔案路徑')
    parser.add_argument('--no-pareto-inject', action='store_true', help='禁用歷史最佳注入')
    return parser.parse_args()

args = parse_args()

# === 🔥 核心設定 ===
WINDOW_ID = int(os.environ.get('WINDOW_ID', args.window))
FINLAB_API_KEY = os.environ.get('FINLAB_API_KEY', "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m")
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
GOOGLE_DRIVE_PATH = os.environ.get('GOOGLE_DRIVE_PATH', '/content/drive/MyDrive')

# 🔥 關鍵設定：從指定 checkpoint 開始
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', args.checkpoint)
DISABLE_PARETO_INJECT = os.environ.get('DISABLE_PARETO_INJECT', '1' if args.no_pareto_inject else '0') == '1'

# 優化目標
TARGET_SHARPE = 4.2
MIN_CAPACITY = 10_000_000  # 1000萬
TARGET_ANNUAL_RETURN = 0.3
MAX_DRAWDOWN = 0.2

# GA 演化參數
POPULATION_SIZE = args.population
N_GENERATIONS = args.generations
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

# Walk-Forward 設定
WALK_FORWARD_WINDOWS = 3
TRAIN_MONTHS = 24
TEST_MONTHS = 6

# 回測設定
BACKTEST_START = '2017-01-01'
BACKTEST_END = None

print(f"=" * 80)
print(f"🐉 快快龍 v12.0 跨視窗協作優化系統 - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e7:.0f}00萬")
print(f"   📊 族群: {POPULATION_SIZE}, 代數: {N_GENERATIONS}")
if CHECKPOINT_PATH:
    print(f"   📂 從 checkpoint 開始: {CHECKPOINT_PATH}")
if DISABLE_PARETO_INJECT:
    print(f"   🚫 已禁用歷史最佳注入")
print(f"=" * 80)

# =============================================================================
# 第三部分：快取清理
# =============================================================================
def clear_finlab_cache():
    """清除可能損壞的 FinLab 快取"""
    print("🔧 清除 FinLab 快取...")

    cleared = 0

    # 1. 清除 finlab 相關目錄（但保留我們的 checkpoint）
    cache_patterns = [
        '/root/.finlab*',
        '/tmp/.finlab*',
        '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'),
        '/content/.finlab*',
    ]

    for pattern in cache_patterns:
        try:
            matches = glob_module.glob(pattern)
            for path in matches:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                cleared += 1
        except:
            pass

    # 2. 清除 pandas 快取
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

clear_finlab_cache()

# =============================================================================
# 第四部分：套件安裝與載入
# =============================================================================
def install_packages():
    """安裝必要套件"""
    required = {
        'finlab': 'finlab',
        'deap': 'deap',
        'requests': 'requests',
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

# Discord 通知
import requests

print(f"✅ 套件載入完成")
print(f"   - FinLab: {finlab.__version__}")

# =============================================================================
# 第五部分：Discord 通知
# =============================================================================
def send_discord_notification(message: str, is_important: bool = False):
    """發送 Discord 通知"""
    if not DISCORD_WEBHOOK_URL:
        return

    try:
        emoji = "🔔" if is_important else "📊"
        payload = {
            "content": f"{emoji} **視窗 {WINDOW_ID}** | {message}"
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except:
        pass

# =============================================================================
# 第六部分：環境設定與登入
# =============================================================================
def setup_environment():
    """設定環境"""
    # Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = f'{GOOGLE_DRIVE_PATH}/快快龍_v12_跨視窗優化'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = './kuaikuai_v12_output'
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
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.output_dir = f"{self.window_dir}/output"
        self.pareto_dir = f"{self.base_dir}/shared_pareto"
        self.checkpoint_dir = f"{self.window_dir}/checkpoints"
        self.log_dir = f"{self.window_dir}_logs"

        for d in [self.window_dir, self.output_dir, self.pareto_dir,
                  self.checkpoint_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def pareto_archive(self) -> str:
        return f"{self.pareto_dir}/pareto_archive_w{self.window_id}.pkl"

    @property
    def checkpoint_latest(self) -> str:
        return f"{self.checkpoint_dir}/checkpoint_window_{self.window_id}_latest.pkl"

    @property
    def best_params_file(self) -> str:
        return f"{self.output_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log(self) -> str:
        return f"{self.log_dir}/progress_history.json"

paths = PathManager()
print(f"📁 工作目錄: {paths.output_dir}")

# =============================================================================
# 第七部分：數據載入
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
        """安全載入數據"""
        try:
            return data.get(key)
        except (EOFError, Exception) as e:
            if 'EOF' in str(e) and retry:
                print(f"   ⚠️  快取損壞，清除後重試: {key}")
                clear_finlab_cache()
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

data_loader = FinLabDataLoader()

# =============================================================================
# 第八部分：策略引擎
# =============================================================================
class StrategyEngine:
    """策略引擎 - 整合三個子策略"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader

    def strategy_low_volatility_pe(self, params: Dict) -> Any:
        """策略一：低波動本益比"""
        rev_ma3_ma12_ratio = params['lv_rev_ma3_ma12_ratio']
        rev_consistency = params['lv_rev_consistency']
        volatility_threshold = params['lv_volatility_threshold']
        margin_usage_limit = params['lv_margin_usage_limit']
        non_op_income_limit = params['lv_non_op_income_limit']
        min_volume = params['lv_min_volume']
        pe_min = params['lv_pe_min']
        pe_max = params['lv_pe_max']
        top_n = int(params['lv_top_n'])

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

        peg = pe / 營業利益成長率

        cond1 = rev_ma3 / rev_ma12 > rev_ma3_ma12_ratio
        cond2 = rev / rev.shift(1) > rev_consistency

        tree_select_factor = ((融資使用率 <= margin_usage_limit)
                             & (entry_volatility <= volatility_threshold)
                             & (業外收支營收率 < non_op_income_limit))

        condition_近1日成交均量 = vol.average(1) > min_volume
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 30).sustain(12, 8)
        cond單月營收月增率 = (rev_month_growth > -54).sustain(3)
        cond收盤價大於均線 = (close > close.average(75)) & (close > close.average(40)) & (close > close.average(90))
        cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)

        pe_range = (pe_min <= pe) & (pe <= pe_max)
        pb_range = (0.5 <= pb) & (pb <= 2.8)
        gpm_trend = (營業毛利率 > 8).sustain(2)
        roe_trend = (ROE綜合損益 > 0).sustain(2)

        small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46

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

        position = peg[cond_all & (peg > 0)].is_smallest(top_n).reindex(rev.index_str_to_date().index, method='ffill')
        return position

    def strategy_small_investor(self, params: Dict) -> Any:
        """策略二：小資族"""
        market_value_limit = params['si_market_value_limit']
        market_rev_ratio_limit = params['si_market_rev_ratio_limit']
        rev_yoy_growth_limit = params['si_rev_yoy_growth_limit']
        rev_mom_growth_limit = params['si_rev_mom_growth_limit']
        rsv_period = int(params['si_rsv_period'])
        ma_period = int(params['si_ma_period'])
        volume_threshold = params['si_volume_threshold']
        top_n = int(params['si_top_n'])

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
        condition6 = vol > volume_threshold

        cond排除月營收連3月衰退 = ~(rev_yoy_growth < rev_yoy_growth_limit).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)
        cond單月營收月增率 = (rev_month_growth > rev_mom_growth_limit).sustain(3)
        cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(120)) & (close > close.average(75))
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)
        業外收支營收率占比低 = (業外收支營收率 < 7.3)

        rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

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
        rev_ma_period = int(params['rpt_rev_ma_period'])
        rev_ma_lookback = int(params['rpt_rev_ma_lookback'])
        price_high_window = int(params['rpt_price_high_window'])
        min_volume = params['rpt_min_volume']
        min_price = params['rpt_min_price']
        rsi_threshold = params['rpt_rsi_threshold']
        pe_limit = params['rpt_pe_limit']
        top_n = int(params['rpt_top_n'])

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

        rev_ma = rev.average(rev_ma_period)
        condition_近n月平均營收創新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()
        condition_近n日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
        condition_近1日成交均量 = vol.average(1) > min_volume

        long_ma_pattern = (
                        (close > close.average(5))
                        & (close > close.average(10))
                        & (close > close.average(20))
                        & (close > close.average(60))
                        & (close > close.average(150))
                        & (close > close.average(200))
                        )

        收盤價_超級績效 = close > (close.average(250) * 1.1)
        rsi_higt_trend = (rsi > rsi_threshold).sustain(1)
        gpm_trend = (營業毛利率 > 5).sustain(5)
        btpm_trend = (稅前淨利率 > 4).sustain(1)
        atpm_trend = (稅後淨利率 > 3).sustain(1)
        rev_rise_nsatisfy = rev_yoy_growth.rank(pct=True, axis=1) > 0.9

        boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= 12) & (inventory.持股分級.astype(int) <= 16)]
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= 18

        pe_range = ~(pe_limit <= pe)

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

        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(top_n).reindex(rev.index_str_to_date().index, method="ffill")
        return position

    def combine_strategies(self, params: Dict) -> Any:
        """合併三個策略"""
        try:
            pos_lv = self.strategy_low_volatility_pe(params)
            pos_si = self.strategy_small_investor(params)
            pos_rpt = self.strategy_revenue_price_turbo(params)

            weight_lv = params['weight_lv']
            weight_si = params['weight_si']
            weight_rpt = params['weight_rpt']

            total_weight = weight_lv + weight_si + weight_rpt
            weight_lv /= total_weight
            weight_si /= total_weight
            weight_rpt /= total_weight

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
# 第九部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """基因解碼器"""

    GENE_LENGTH = 32

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # === 策略一：低波動本益比（9個參數）===
        params['lv_rev_ma3_ma12_ratio'] = genes[0] * 0.5 + 1.0
        params['lv_rev_consistency'] = genes[1] * 0.4 + 0.5
        params['lv_volatility_threshold'] = genes[2] * 0.05 + 0.02
        params['lv_margin_usage_limit'] = genes[3] * 30 + 20
        params['lv_non_op_income_limit'] = genes[4] * 10 + 5
        params['lv_min_volume'] = genes[5] * 200000 + 100000
        params['lv_pe_min'] = genes[6] * 10 + 3
        params['lv_pe_max'] = genes[7] * 20 + 15
        params['lv_top_n'] = int(genes[8] * 5 + 2)

        # === 策略二：小資族（8個參數）===
        params['si_market_value_limit'] = genes[9] * 10e9 + 10e9
        params['si_market_rev_ratio_limit'] = genes[10] * 3 + 2
        params['si_rev_yoy_growth_limit'] = genes[11] * 10 - 15
        params['si_rev_mom_growth_limit'] = genes[12] * 30 - 70
        params['si_rsv_period'] = int(genes[13] * 30 + 40)
        params['si_ma_period'] = int(genes[14] * 40 + 50)
        params['si_volume_threshold'] = genes[15] * 200000 + 100000
        params['si_top_n'] = int(genes[16] * 6 + 4)

        # === 策略三：營收股價雙渦輪（8個參數）===
        params['rpt_rev_ma_period'] = int(genes[17] * 5 + 3)
        params['rpt_rev_ma_lookback'] = int(genes[18] * 15 + 15)
        params['rpt_price_high_window'] = int(genes[19] * 8 + 4)
        params['rpt_min_volume'] = genes[20] * 300000 + 200000
        params['rpt_min_price'] = genes[21] * 10 + 10
        params['rpt_rsi_threshold'] = genes[22] * 20 + 50
        params['rpt_pe_limit'] = genes[23] * 100 + 100
        params['rpt_top_n'] = int(genes[24] * 10 + 8)

        # === 策略權重（3個）===
        raw_weights = genes[25:28]
        total = sum(raw_weights) + 1e-10
        params['weight_lv'] = raw_weights[0] / total
        params['weight_si'] = raw_weights[1] / total
        params['weight_rpt'] = raw_weights[2] / total

        # === 回測參數（4個）===
        params['stop_loss'] = genes[28] * 0.2 + 0.15
        params['trail_stop'] = genes[29] * 0.3 + 0.2
        params['take_profit'] = genes[30] * 0.5 + 0.5
        params['position_limit'] = genes[31] * 0.2 + 0.25

        return params

gene_decoder = GeneDecoder()

# =============================================================================
# 第十部分：Walk-Forward 回測引擎
# =============================================================================
class WalkForwardBacktest:
    """Walk-Forward 回測引擎"""

    def run_walk_forward(self, position, params: Dict) -> Dict:
        """執行 Walk-Forward 分析"""
        try:
            if BACKTEST_START:
                position = position.loc[BACKTEST_START:]

            if position is None or position.empty or len(position) < 100:
                return self._empty_result()

            windows = self._split_windows(position.index)

            if len(windows) < 2:
                return self._simple_backtest(position, params)

            window_results = []
            for i, (train_dates, test_dates) in enumerate(windows):
                test_pos = position.loc[test_dates[0]:test_dates[-1]]

                if test_pos.empty:
                    continue

                result = self._backtest_single_window(test_pos, params, f'Window{i}')
                if result:
                    window_results.append(result)

            if not window_results:
                return self._empty_result()

            overall_result = self._backtest_single_window(position, params, 'Overall')

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
# 第十一部分：適應度評估函數
# =============================================================================
def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """評估個體適應度"""
    try:
        params = gene_decoder.decode(individual)
        position = strategy_engine.combine_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        wf_result = walk_forward.run_walk_forward(position, params)
        overall = wf_result['overall']

        sharpe = overall['sharpe']
        capacity = overall['capacity']
        annual_return = overall['annual_return']

        sharpe_score = min(5.0, max(0, sharpe / TARGET_SHARPE * 5.0))
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))
        return_score = min(2.0, max(0, annual_return / TARGET_ANNUAL_RETURN * 2.0))

        if wf_result['is_robust']:
            robustness_penalty = 0.0
        else:
            consistency = wf_result['consistency_score']
            robustness_penalty = -(1 - consistency) * 3

        return (sharpe_score, capacity_score, return_score, robustness_penalty)

    except Exception as e:
        print(f"   ⚠️ 評估失敗: {e}")
        return (0.0, 0.0, 0.0, -10.0)

# =============================================================================
# 第十二部分：DEAP 設定
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
# 第十三部分：🔥 Checkpoint 載入器（核心功能）
# =============================================================================
class CheckpointLoader:
    """
    🔥 Checkpoint 載入器

    支援從指定的 checkpoint 檔案載入種群，而非從歷史最佳開始
    """

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Optional[List[Dict]]:
        """
        載入 checkpoint 檔案

        Returns:
            List of {'genes': [...], 'fitness': (...)} or None
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint 檔案不存在: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            # 支援多種 checkpoint 格式
            if isinstance(data, dict):
                # 格式 1: {'population': [...], 'generation': ...}
                if 'population' in data:
                    population_data = data['population']
                    print(f"✅ 載入 checkpoint: {len(population_data)} 個個體")
                    print(f"   來源: {checkpoint_path}")
                    if 'generation' in data:
                        print(f"   代數: {data['generation']}")
                    return population_data

                # 格式 2: {'individuals': [...]}
                elif 'individuals' in data:
                    individuals = data['individuals']
                    print(f"✅ 載入 Pareto archive: {len(individuals)} 個個體")
                    return individuals

            # 格式 3: 直接是 list
            elif isinstance(data, list):
                print(f"✅ 載入種群列表: {len(data)} 個個體")
                return data

            print(f"⚠️ 無法識別的 checkpoint 格式")
            return None

        except Exception as e:
            print(f"⚠️ 載入 checkpoint 失敗: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def create_population_from_checkpoint(checkpoint_data: List[Dict],
                                          target_size: int) -> List:
        """
        從 checkpoint 數據創建 DEAP 族群

        Args:
            checkpoint_data: checkpoint 中的個體列表
            target_size: 目標族群大小

        Returns:
            DEAP 格式的族群
        """
        population = []

        for item in checkpoint_data[:target_size]:
            # 創建個體
            ind = creator.Individual(item['genes'] if isinstance(item, dict) else item)

            # 如果有適應度，設定之（但會強制重新評估）
            if isinstance(item, dict) and 'fitness' in item:
                # 不設定適應度，讓系統重新評估
                pass

            population.append(ind)

        # 如果 checkpoint 個體不足，補充隨機個體
        while len(population) < target_size:
            ind = toolbox.individual()
            population.append(ind)

        print(f"   📊 從 checkpoint 載入 {min(len(checkpoint_data), target_size)} 個個體")
        if len(checkpoint_data) < target_size:
            print(f"   📊 補充 {target_size - len(checkpoint_data)} 個隨機個體")

        return population

checkpoint_loader = CheckpointLoader()

# =============================================================================
# 第十四部分：Pareto 歷史管理器（可選功能）
# =============================================================================
class ParetoArchiveManager:
    """Pareto 前緣歷史管理器"""

    def __init__(self, archive_file: str, disabled: bool = False):
        self.archive_file = archive_file
        self.archive = []
        self.disabled = disabled

        if not disabled:
            self._load()

    def _load(self):
        """載入歷史"""
        if self.disabled:
            return

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
            print("ℹ️ 無歷史存檔")

    def update(self, pareto_front: List):
        """更新 Pareto 前緣"""
        all_individuals = self.archive + [
            {'genes': list(ind), 'fitness': ind.fitness.values}
            for ind in pareto_front
        ]

        unique = self._deduplicate(all_individuals)
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:30]

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
        """保存"""
        try:
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                }, f)
        except Exception as e:
            print(f"⚠️ 歷史保存失敗: {e}")

    def inject_elites(self, population: List, ratio: float = 0.3):
        """注入精英"""
        if self.disabled or not self.archive:
            print("ℹ️ 跳過歷史精英注入（已禁用或無歷史）")
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

pareto_archive = ParetoArchiveManager(paths.pareto_archive, disabled=DISABLE_PARETO_INJECT)

# =============================================================================
# 第十五部分：進度日誌記錄器
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
# 第十六部分：🔥 演化引擎（核心：支援從 checkpoint 開始）
# =============================================================================
class EvolutionEngine:
    """演化引擎（支援從指定 checkpoint 開始）"""

    def __init__(self, toolbox, pareto_mgr: ParetoArchiveManager):
        self.toolbox = toolbox
        self.pareto_mgr = pareto_mgr
        self.history = []
        self.logger = ProgressLogger(WINDOW_ID)

    def run(self, n_generations: int = N_GENERATIONS,
            checkpoint_path: str = None) -> Tuple[List, List]:
        """
        執行演化

        Args:
            n_generations: 演化代數
            checkpoint_path: 指定 checkpoint 檔案路徑（優先使用）
        """
        print(f"\n{'='*70}")
        print(f"🐉 開始演化 - 快快龍 v12.0 視窗 {WINDOW_ID}")
        print(f"   族群: {POPULATION_SIZE}, 世代: {n_generations}")
        print(f"{'='*70}\n")

        # 🔥 決定初始族群來源
        population = None

        # 優先級 1: 命令行/環境變數指定的 checkpoint
        if checkpoint_path:
            checkpoint_data = checkpoint_loader.load_checkpoint(checkpoint_path)
            if checkpoint_data:
                population = checkpoint_loader.create_population_from_checkpoint(
                    checkpoint_data, POPULATION_SIZE
                )
                send_discord_notification(f"從 checkpoint 開始演化: {len(checkpoint_data)} 個種子", True)

        # 優先級 2: 隨機初始化
        if population is None:
            print("📊 使用隨機初始化族群")
            population = self.toolbox.population(n=POPULATION_SIZE)

            # 可選：注入歷史精英（如果未禁用）
            if not DISABLE_PARETO_INJECT:
                population = self.pareto_mgr.inject_elites(population, ratio=0.3)

        # 初始評估
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
                print(f"=== 第 {gen+1}/{n_generations} 代 ===")
                print(f"   最佳綜合: {stats['best_composite']:.4f}")
                print(f"   最佳夏普: {stats['best_sharpe']:.2f}")
                print(f"   最佳胃納量: {stats['best_capacity']:.0f} 萬")
                print(f"   穩健性: {stats['best_robustness']:.2f}")
                print(f"   耗時: {elapsed:.1f}s")

                # Discord 通知（每 20 代）
                if (gen + 1) % 20 == 0:
                    send_discord_notification(
                        f"第 {gen+1}/{n_generations} 代 | 夏普: {stats['best_sharpe']:.2f} | 胃納量: {stats['best_capacity']:.0f}萬"
                    )

            # 保存 checkpoint
            if (gen + 1) % 10 == 0:
                self._save_checkpoint(population, gen)

        # Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 更新歷史存檔
        self.pareto_mgr.update(pareto_front)

        # 輸出結果
        self._print_pareto_front(pareto_front)

        # Discord 通知完成
        if pareto_front:
            best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
            best_sharpe = best_ind.fitness.values[0] / 5.0 * TARGET_SHARPE
            best_capacity = best_ind.fitness.values[1] / 5.0 * MIN_CAPACITY / 1e4
            send_discord_notification(
                f"✅ 演化完成！最佳夏普: {best_sharpe:.2f} | 胃納量: {best_capacity:.0f}萬",
                True
            )

        return population, pareto_front

    def _evaluate_population(self, population: List) -> List:
        """評估族群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        print(f"   評估 {len(invalid)} 個個體...")

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

        self.logger.log_generation(gen, N_GENERATIONS, stats)

        return stats

    def _save_checkpoint(self, population: List, gen: int):
        """保存 checkpoint"""
        try:
            checkpoint_data = {
                'generation': gen,
                'population': [
                    {'genes': list(ind), 'fitness': ind.fitness.values}
                    for ind in population
                ],
                'timestamp': datetime.now().isoformat(),
                'window_id': WINDOW_ID,
            }

            with open(paths.checkpoint_latest, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            print(f"   💾 Checkpoint 已保存: {paths.checkpoint_latest}")
        except Exception as e:
            print(f"⚠️ Checkpoint 保存失敗: {e}")

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
# 第十七部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║     🐉 快快龍 v12.0 - 跨視窗協作優化系統                        ║
║     NSGA-II + Walk-Forward + Checkpoint Resume                 ║
╠════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e7:.0f}00萬+                        ║
║  📊 策略：三策略動態組合優化                                     ║
║  🔬 驗證：Walk-Forward ({WALK_FORWARD_WINDOWS} 窗口)                             ║
║  🧬 基因：{GeneDecoder.GENE_LENGTH} 個參數                                        ║
║  🖥️  視窗：{WINDOW_ID}                                                     ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # 建立演化引擎
    engine = EvolutionEngine(toolbox, pareto_archive)

    # 🔥 執行演化（支援從指定 checkpoint 開始）
    population, pareto_front = engine.run(
        N_GENERATIONS,
        checkpoint_path=CHECKPOINT_PATH
    )

    # 最佳個體詳細回測
    if pareto_front:
        print(f"\n{'='*70}")
        print("📊 最佳個體詳細回測")
        print(f"{'='*70}")

        best_ind = max(pareto_front, key=lambda x: sum(x.fitness.values))
        best_params = gene_decoder.decode(best_ind)

        print("\n🔍 最佳參數：")
        print(f"   策略權重：低波動 {best_params['weight_lv']:.2%}, "
              f"小資族 {best_params['weight_si']:.2%}, "
              f"雙渦輪 {best_params['weight_rpt']:.2%}")

        # 保存最佳參數
        with open(paths.best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n✅ 最佳參數已保存: {paths.best_params_file}")

        # 完整回測
        try:
            print("\n執行完整回測...")
            position = strategy_engine.combine_strategies(best_params)

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
                name=f'快快龍_v12_W{WINDOW_ID}_Best'
            )

            report.display()
        except Exception as e:
            print(f"⚠️ 完整回測失敗: {e}")

    print(f"\n✅ 優化完成！")
    print(f"📁 結果保存於: {paths.output_dir}")
    print(f"📁 Checkpoint: {paths.checkpoint_latest}")

    return population, pareto_front

if __name__ == "__main__":
    population, pareto = main()
