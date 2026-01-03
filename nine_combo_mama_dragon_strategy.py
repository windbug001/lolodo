#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 九組合媽媽龍策略 - Nine Combo Mama Dragon Strategy
================================================================================

【策略架構】九大策略多元組合
┌─────────────────────────────────────────────────────────────────────────┐
│ 原六組合快快龍策略：                                                      │
│   1. 低波動本益比策略 (Low Vol PE)         - 價值投資 + 低風險           │
│   2. 小資族策略 (Small Investor)           - 小型股成長                  │
│   3. 營收股價雙渦輪策略 (Revenue Turbo)     - 營收動能                   │
│   4. 高殖利率烏龜策略 (High Yield Turtle)  - 股息穩健                    │
│   5. 低波動性指標策略 (Low Vol Index)      - 低波動大盤                  │
│   6. 藏獒外掛大盤指針 (Market Indicator)   - 趨勢突破                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 新增三策略：                                                              │
│   7. 小蝦米跟大鯨魚 (Shrimp & Whale)       - 籌碼面追蹤                  │
│   8. 純技術趨勢策略 (Tech Trend)           - 技術面動能                  │
│   9. 財報指標20大 (Fundamental 20)         - 財務基本面                  │
└─────────────────────────────────────────────────────────────────────────┘

【策略風格分布】
- 價值投資: 策略 1, 4
- 成長動能: 策略 2, 3, 7
- 技術分析: 策略 6, 8
- 基本面: 策略 5, 9

【目標】
- 夏普值：>= 4.0
- 胃納量：>= 1000 萬
- 年化報酬：最大化
- 最大回撤：< 20%

版本：v1.0 (2025-01-03)
================================================================================
"""

from __future__ import annotations

# 🔥 禁用 FinLab 快取以避免 EOFError
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# === 核心設定 ===
WINDOW_ID = 1
FINLAB_API_KEY = "YOUR_API_KEY_HERE"  # 請替換成您的 API Key

# 優化目標
TARGET_SHARPE = 4.0
MIN_CAPACITY = 10_000_000  # 1000萬
TARGET_ANNUAL_RETURN = 0.3
MAX_DRAWDOWN = 0.2

# GA 演化參數
POPULATION_SIZE = 60
N_GENERATIONS = 150
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

# 回測設定
BACKTEST_START = '2017-01-01'

print(f"=" * 80)
print(f"🐉 九組合媽媽龍策略 v1.0")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e7:.0f}00萬")
print(f"=" * 80)

# =============================================================================
# 第二部分：套件載入
# =============================================================================
def install_packages():
    """安裝必要套件"""
    required = {'finlab': 'finlab', 'deap': 'deap'}
    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"   安裝 {pkg_name}...")
            os.system(f'pip install {pkg_name} -q')

install_packages()

import finlab
from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

from deap import base, creator, tools

print(f"✅ 套件載入完成 - FinLab: {finlab.__version__}")

# =============================================================================
# 第三部分：環境設定
# =============================================================================
def setup_environment():
    """設定環境"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = '/content/drive/MyDrive/九組合媽媽龍策略'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = './mama_dragon_output'
        in_colab = False
        print("⚠️ 本地環境")

    if FINLAB_API_KEY and FINLAB_API_KEY != "YOUR_API_KEY_HERE":
        finlab.login(FINLAB_API_KEY)
        print("✅ FinLab VIP 登入成功")

    return base_dir, in_colab

BASE_DIR, IN_COLAB = setup_environment()

# =============================================================================
# 第四部分：數據載入器
# =============================================================================
class FinLabDataLoader:
    """FinLab 數據載入器 - 單例模式"""

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
                print(f"   ⚠️ 快取損壞，重試: {key}")
                return self._safe_get(key, retry=False)
            raise

    def _load_all_data(self):
        """載入所有數據"""
        print("📊 載入 FinLab 數據...")
        start_time = time.time()

        # 設定交易範圍
        data.set_universe('TSE_OTC')

        # 價格數據
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
        self._cache['營業利益率'] = self._safe_get('fundamental_features:營業利益率')

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

        # 載入財報特徵 (策略9需要)
        self._load_fundamental_features()

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

    def _load_fundamental_features(self):
        """載入財報特徵 (策略9用)"""
        feature_names = [
            'fundamental_features:流動比率',
            'fundamental_features:稅前淨利率',
            'fundamental_features:資產總額成長率',
            'fundamental_features:營收成長率',
            'fundamental_features:ROE綜合損益',
            'fundamental_features:營運資金',
            'fundamental_features:流動資產',
            'fundamental_features:營業費用率',
            'fundamental_features:負債比率',
            'fundamental_features:總負債除總淨值',
            'fundamental_features:營業利益率',
            'fundamental_features:稅前息前折舊前淨利率',
            'fundamental_features:稅後淨利成長率',
            'fundamental_features:ROE稅後',
            'fundamental_features:稅率',
            'fundamental_features:營業利益',
        ]

        self._cache['fundamental_features'] = {}
        for name in feature_names:
            try:
                key = name.split(':')[1]
                self._cache['fundamental_features'][key] = self._safe_get(name)
            except:
                pass

    def get(self, key: str):
        """獲取數據"""
        return self._cache.get(key)

# 初始化數據載入器
data_loader = FinLabDataLoader()

# =============================================================================
# 第五部分：九大策略引擎
# =============================================================================
class NineStrategyEngine:
    """九組合媽媽龍策略引擎"""

    def __init__(self, data_loader: FinLabDataLoader):
        self.dl = data_loader

    # =========================================================================
    # 策略 1：低波動本益比策略
    # =========================================================================
    def strategy_1_low_volatility_pe(self, params: Dict) -> Any:
        """策略一：低波動本益比"""
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

        cond1 = rev_ma3 / rev_ma12 > params.get('rev_ma3_ma12_ratio', 1.0)
        cond2 = rev / rev.shift(1) > params.get('rev_consistency', 0.8)

        tree_select_factor = ((融資使用率 <= params.get('margin_usage_limit', 40))
                             & (entry_volatility <= params.get('volatility_threshold', 0.04))
                             & (業外收支營收率 < params.get('non_op_income_limit', 10)))

        condition_成交量 = vol.average(1) > params.get('min_volume', 100000)
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 30).sustain(12, 8)
        cond單月營收月增率 = (rev_month_growth > -54).sustain(3)
        cond收盤價大於均線 = (close > close.average(75)) & (close > close.average(40)) & (close > close.average(90))
        cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)

        pe_range = (params.get('pe_min', 5) <= pe) & (pe <= params.get('pe_max', 25))
        pb_range = (0.5 <= pb) & (pb <= 2.8)
        gpm_trend = (營業毛利率 > 8).sustain(2)
        roe_trend = (ROE綜合損益 > 0).sustain(2)

        try:
            small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                            .reset_index()
                            .groupby(["date", "stock_id"])
                            .agg({"占集保庫存數比例": "sum"})
                            .reset_index()
                            .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
        except:
            small_inv_under50 = True

        cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老
                   & cond排除月營收連3月衰退 & cond收盤價大於均線 & cond近三個月營收大於年營收
                   & cond單月營收月增率 & ~limit_up_all_day & condition_成交量
                   & gpm_trend & roe_trend & small_inv_under50 & pe_range & pb_range)

        position = peg[cond_all & (peg > 0)].is_smallest(params.get('top_n', 5))
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 2：小資族策略
    # =========================================================================
    def strategy_2_small_investor(self, params: Dict) -> Any:
        """策略二：小資族"""
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

        condition1 = (市值 < params.get('market_value_limit', 15e9))
        condition2 = 自由現金流 > 0
        condition3 = 股東權益報酬率 > 0
        condition4 = 營業利益成長率 > -1
        condition5 = 市值營收比 < params.get('market_rev_ratio_limit', 3)
        condition6 = vol > params.get('volume_threshold', 100000)

        cond排除月營收連3月衰退 = ~(rev_yoy_growth < -10).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)
        cond單月營收月增率 = (rev_month_growth > -54).sustain(3)
        cond收盤價大於均線 = (close > close.average(60)) & (close > close.average(120)) & (close > close.average(75))
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)
        業外收支營收率占比低 = (業外收支營收率 < 7.3)

        rsv_period = params.get('rsv_period', 50)
        rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

        position = ((condition1 & condition2 & condition3 & condition4 & condition5
                    & condition6 & cond排除月營收成長趨勢過老 & cond單月營收月增率
                    & cond排除月營收連3月衰退 & cond近三個月營收大於年營收
                    & 業外收支營收率占比低) * rsv).is_largest(params.get('top_n', 6))

        position = position.reindex(當月營收.index_str_to_date().index)
        return position

    # =========================================================================
    # 策略 3：營收股價雙渦輪策略
    # =========================================================================
    def strategy_3_revenue_price_turbo(self, params: Dict) -> Any:
        """策略三：營收股價雙渦輪"""
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

        rev_ma_period = params.get('rev_ma_period', 4)
        rev_ma = rev.average(rev_ma_period)
        rev_ma_lookback = params.get('rev_ma_lookback', 20)

        condition_近n月平均營收創新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()
        condition_近n日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(params.get('price_high_window', 8), 1)
        condition_成交量 = vol.average(1) > params.get('min_volume', 200000)

        long_ma_pattern = ((close > close.average(5)) & (close > close.average(10))
                          & (close > close.average(20)) & (close > close.average(60))
                          & (close > close.average(150)) & (close > close.average(200)))

        收盤價_超級績效 = close > (close.average(250) * 1.1)
        rsi_higt_trend = (rsi > params.get('rsi_threshold', 60)).sustain(1)
        gpm_trend = (營業毛利率 > 5).sustain(5)
        btpm_trend = (稅前淨利率 > 4).sustain(1)
        atpm_trend = (稅後淨利率 > 3).sustain(1)
        rev_rise_nsatisfy = rev_yoy_growth.rank(pct=True, axis=1) > 0.9

        try:
            boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= 12) & (inventory.持股分級.astype(int) <= 16)]
                            .reset_index()
                            .groupby(["date", "stock_id"])
                            .agg({"占集保庫存數比例": "sum"})
                            .reset_index()
                            .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= 18
        except:
            boss_inventory_over400 = True

        pe_range = ~(params.get('pe_limit', 150) <= pe)

        conditions = (condition_近n月平均營收創新高 & condition_近n日內有1日股價創新高
                     & condition_成交量 & long_ma_pattern & gpm_trend & btpm_trend
                     & atpm_trend & rev_rise_nsatisfy & (close > params.get('min_price', 15))
                     & (業外收支營收率 < 7.3) & pe_range & 收盤價_超級績效
                     & ((vol >= vol.rolling(20).mean() * 0.8)) & rsi_higt_trend
                     & boss_inventory_over400 & ~limit_up_all_day)

        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(params.get('top_n', 12))
        position = position.reindex(rev.index_str_to_date().index, method="ffill")
        return position

    # =========================================================================
    # 策略 4：高殖利率烏龜策略
    # =========================================================================
    def strategy_4_high_yield_turtle(self, params: Dict) -> Any:
        """策略四：高殖利率烏龜"""
        yield_ratio = self.dl.get('dividend_yield')
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        營業利益率 = self.dl.get('營業利益率')
        董監持有股數占比 = self.dl.get('董監持有股數占比')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')

        sma20 = close.average(20)
        sma60 = close.average(60)

        cond1 = yield_ratio >= params.get('min_yield_ratio', 4)
        cond2 = (close > sma20) & (close > sma60)
        cond3 = rev.average(3) > rev.average(12)
        cond4 = 營業利益率 >= params.get('min_op_earn_ratio', 8)
        cond5 = 董監持有股數占比 >= params.get('min_boss_hold', 15)
        cond6 = (vol.average(5) >= params.get('min_volume', 100000)) & (vol.average(5) <= params.get('max_volume', 3000000))

        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        cond_all = cond_all * rev_yoy_growth
        position = cond_all[cond_all > 0].is_largest(params.get('top_n', 8))
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 5：低波動性指標策略
    # =========================================================================
    def strategy_5_low_volatility_index(self, params: Dict) -> Any:
        """策略五：低波動性指標"""
        cap = self.dl.get('市值')
        vol = self.dl.get('vol')
        close = self.dl.get('close')

        std_window = params.get('std_window', 20)
        std = close.pct_change().rolling(std_window).std().rank(axis=1, pct=True)

        position = cap[(vol.average(20) > params.get('min_volume', 100000)) &
                       (close > close.average(60)) &
                       (close > close.average(120)) &
                       (close > close.average(250)) &
                       (std < params.get('std_threshold', 0.3))].is_smallest(params.get('top_n', 15))

        position = position.reindex(close.index, method='ffill')
        return position

    # =========================================================================
    # 策略 6：藏獒外掛大盤指針策略
    # =========================================================================
    def strategy_6_market_indicator(self, params: Dict) -> Any:
        """策略六：藏獒外掛大盤指針"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')
        rev_month_growth = self.dl.get('rev_month_growth')

        vol_ma = vol.average(10)
        new_high_window = params.get('new_high_window', 120)

        cond1 = (close == close.rolling(new_high_window).max())
        cond2 = ~(rev_yoy_growth < -15).sustain(3)
        cond3 = ~(rev_yoy_growth > 60).sustain(12, 8)
        cond4 = ((rev.rolling(12).min())/(rev) < 1.2).sustain(3)
        cond5 = (rev_month_growth > -10).sustain(3)
        cond6 = vol_ma > params.get('min_volume', 100000)

        buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        buy = vol_ma * buy
        buy = buy[buy > 0]
        buy = buy.is_smallest(params.get('top_n', 10))
        position = buy.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 7：小蝦米跟大鯨魚 (新增)
    # =========================================================================
    def strategy_7_shrimp_whale(self, params: Dict) -> Any:
        """策略七：小蝦米跟大鯨魚 - 籌碼面策略"""
        inv = self.dl.get('inventory')
        rev = self.dl.get('rev')
        close = self.dl.get('close')
        interest = self.dl.get('dividend_yield')
        grow = self.dl.get('營業利益成長率')

        # 散戶持股 (1-5級)
        h1 = inv[inv.持股分級.astype(int) <= 5].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')

        # 大戶持股 (9-15級)
        h2 = inv[(inv.持股分級.astype(int) >= 9) & (inv.持股分級.astype(int) <= 15)].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')

        # 大戶持股比例
        ratio = (h2 / (h1 + h2))

        # 營收指標
        rev_mom = rev / rev.shift()
        rev_yoy2 = rev.rolling(2).mean() / rev.shift(12).rolling(2).mean()

        # 多階段篩選
        # 第一層：50檔
        p = (FinlabDataFrame(ratio).rank(axis=1, pct=True) *
             (close.notna() & (close > close.average(250)))
             + rev_yoy2.rank(axis=1, pct=True)
             + rev_mom.rank(axis=1, pct=True)).is_largest(50)

        # 第二層：35檔 (營收成長)
        p = (p * grow).is_largest(35)

        # 第三層：10檔 (殖利率)
        p = (p * interest).is_largest(10)

        # 第四層：5檔 (大戶增持)
        p = (p * ratio.diff(8)).is_largest(params.get('top_n', 5))

        p = p.reindex(rev.index_str_to_date().index, method='ffill')
        return p

    # =========================================================================
    # 策略 8：純技術趨勢策略 (新增)
    # =========================================================================
    def strategy_8_tech_trend(self, params: Dict) -> Any:
        """策略八：純技術趨勢策略"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')

        def wma(price, n):
            return price.ewm(com=n).mean()

        def zlma(price, n):
            lag = (n - 1) // 2
            series = 2 * price - price.shift(lag)
            return wma(series, n)

        # 計算技術指標
        bias_sma_250 = close / close.rolling(250).mean() - 1
        bias_zlma_120 = close / close.apply(lambda s: zlma(s, 120)) - 1
        bias_wma_60 = close / close.apply(lambda s: wma(s, 60)) - 1
        slope1_sma_60 = close.rolling(60).mean().pipe(lambda df: df / df.shift(60) - 1)
        kurtosis_250 = close.pct_change().rolling(250).kurt()
        kurtosis_120 = close.pct_change().rolling(120).kurt()

        # 條件篩選
        pos = vol.average(10)[
            (bias_sma_250 > params.get('bias_sma_threshold', 0.1)) &
            (bias_zlma_120 > params.get('bias_zlma_threshold', 0.1)) &
            (bias_wma_60 > params.get('bias_wma_threshold', 0.1)) &
            (slope1_sma_60.rank(axis=1, pct=True) > params.get('slope_percentile', 0.5)) &
            (kurtosis_250.rank(axis=1, pct=True) > params.get('kurtosis_percentile', 0.4)) &
            (vol > params.get('min_volume', 200000))
        ].is_smallest(params.get('top_n', 10))

        return pos

    # =========================================================================
    # 策略 9：財報指標20大 (新增)
    # =========================================================================
    def strategy_9_fundamental_20(self, params: Dict) -> Any:
        """策略九：財報指標20大"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        fundamental_features = self.dl.get('fundamental_features')

        if not fundamental_features:
            return pd.DataFrame()

        class Process():
            @staticmethod
            def df_max4(df):
                return df >= df.rolling(4).max()

            @staticmethod
            def df_max8(df):
                return df >= df.rolling(8).max()

            @staticmethod
            def avg4_diff4(df):
                avg4 = df.average(4)
                return avg4 / avg4.diff(4)

            @staticmethod
            def l_avg4(df):
                return df / df.average(4)

            @staticmethod
            def avg4(df):
                return df.average(4)

            @staticmethod
            def avg2_avg4(df):
                return df.rolling(2).mean() / df.average(4)

        # 處理財報特徵
        fs = []
        for key, df in fundamental_features.items():
            if df is not None:
                try:
                    # 選擇處理方法
                    if '成長率' in key:
                        processed = Process.l_avg4(df)
                    elif 'ROE' in key:
                        processed = Process.avg2_avg4(df)
                    elif '比率' in key:
                        processed = Process.df_max4(df)
                    else:
                        processed = Process.avg4(df)
                    fs.append(processed)
                except:
                    pass

        if not fs:
            return pd.DataFrame()

        # 波動率條件
        std = close.pct_change().rolling(60).std()
        cond1 = (close > close.average(60)).astype(float) * 5
        cond2 = (std.rank(pct=True, axis=1) < 0.5).astype(float) * 5
        cond3 = (vol.rolling(5).mean() > params.get('min_volume', 200000)).astype(float) * 10

        # 綜合評分
        score = sum([f.deadline().rank(axis=1, pct=True).fillna(0) for f in fs])
        final_score = score + (cond1 + cond2 + cond3).reindex(score.index, method='ffill')

        position = final_score.is_largest(params.get('top_n', 15))
        position = position.reindex(close.loc[score.index[0]:].index, method='ffill')

        return position

    # =========================================================================
    # 策略組合
    # =========================================================================
    def combine_all_strategies(self, params: Dict) -> Any:
        """合併九個策略"""
        try:
            # 獲取各策略持股
            positions = []
            weights = []

            strategy_funcs = [
                (self.strategy_1_low_volatility_pe, 'weight_1'),
                (self.strategy_2_small_investor, 'weight_2'),
                (self.strategy_3_revenue_price_turbo, 'weight_3'),
                (self.strategy_4_high_yield_turtle, 'weight_4'),
                (self.strategy_5_low_volatility_index, 'weight_5'),
                (self.strategy_6_market_indicator, 'weight_6'),
                (self.strategy_7_shrimp_whale, 'weight_7'),
                (self.strategy_8_tech_trend, 'weight_8'),
                (self.strategy_9_fundamental_20, 'weight_9'),
            ]

            for func, weight_key in strategy_funcs:
                try:
                    pos = func(params)
                    if pos is not None and not pos.empty:
                        positions.append(pos)
                        weights.append(params.get(weight_key, 1.0))
                except Exception as e:
                    print(f"   ⚠️ 策略執行失敗: {e}")

            if not positions:
                return None

            # 正規化權重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # 合併持股
            position_combined = positions[0] * weights[0]
            for pos, w in zip(positions[1:], weights[1:]):
                position_combined = position_combined.add(pos * w, fill_value=0)

            return position_combined

        except Exception as e:
            print(f"   ⚠️ 策略合併失敗: {e}")
            return None

# 初始化策略引擎
strategy_engine = NineStrategyEngine(data_loader)

# =============================================================================
# 第六部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """基因解碼器 - 九策略版本"""

    GENE_LENGTH = 50  # 擴展到50個參數

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # === 九策略權重 (0-8) ===
        raw_weights = [max(0.01, genes[i]) for i in range(9)]
        total = sum(raw_weights)
        for i in range(9):
            params[f'weight_{i+1}'] = raw_weights[i] / total

        # === 策略1參數 (9-14) ===
        params['rev_ma3_ma12_ratio'] = genes[9] * 0.5 + 1.0
        params['rev_consistency'] = genes[10] * 0.4 + 0.5
        params['volatility_threshold'] = genes[11] * 0.05 + 0.02
        params['margin_usage_limit'] = genes[12] * 30 + 20
        params['pe_min'] = genes[13] * 10 + 3
        params['pe_max'] = genes[14] * 20 + 15

        # === 策略2參數 (15-18) ===
        params['market_value_limit'] = genes[15] * 20e9 + 5e9
        params['market_rev_ratio_limit'] = genes[16] * 3 + 2
        params['rsv_period'] = int(genes[17] * 30 + 40)

        # === 策略3參數 (18-22) ===
        params['rev_ma_period'] = int(genes[18] * 5 + 3)
        params['rev_ma_lookback'] = int(genes[19] * 15 + 15)
        params['price_high_window'] = int(genes[20] * 8 + 4)
        params['rsi_threshold'] = genes[21] * 20 + 50
        params['pe_limit'] = genes[22] * 100 + 100

        # === 策略4參數 (23-26) ===
        params['min_yield_ratio'] = genes[23] * 4 + 3
        params['min_op_earn_ratio'] = genes[24] * 10 + 5
        params['min_boss_hold'] = genes[25] * 20 + 10

        # === 策略5參數 (26-28) ===
        params['std_window'] = int(genes[26] * 20 + 10)
        params['std_threshold'] = genes[27] * 0.3 + 0.2

        # === 策略6參數 (28-30) ===
        params['new_high_window'] = int(genes[28] * 140 + 60)

        # === 策略7參數 (30-32) - 小蝦米跟大鯨魚 ===
        # 使用預設參數

        # === 策略8參數 (32-38) - 純技術趨勢 ===
        params['bias_sma_threshold'] = genes[32] * 0.15 + 0.05
        params['bias_zlma_threshold'] = genes[33] * 0.15 + 0.05
        params['bias_wma_threshold'] = genes[34] * 0.15 + 0.05
        params['slope_percentile'] = genes[35] * 0.3 + 0.4
        params['kurtosis_percentile'] = genes[36] * 0.3 + 0.3

        # === 策略9參數 (38-40) - 財報指標20大 ===
        # 使用預設參數

        # === 各策略 top_n (40-48) ===
        params['top_n'] = int(genes[40] * 5 + 5)  # 通用

        # === 回測參數 (48-50) ===
        params['stop_loss'] = genes[48] * 0.2 + 0.15
        params['position_limit'] = genes[49] * 0.2 + 0.25
        params['min_volume'] = genes[40] * 200000 + 100000

        return params

gene_decoder = GeneDecoder()

# =============================================================================
# 第七部分：回測引擎
# =============================================================================
def run_backtest(position, params: Dict, name: str = "九組合媽媽龍") -> Dict:
    """執行回測"""
    try:
        if position is None or position.empty:
            return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1}

        if BACKTEST_START:
            position = position.loc[BACKTEST_START:]

        report = sim(
            position=position,
            fee_ratio=1.425 / 1000,
            tax_ratio=3 / 1000,
            trade_at_price="high_low_avg",
            position_limit=params.get('position_limit', 0.35),
            stop_loss=params.get('stop_loss', 0.25),
            stop_trading_next_period=False,
            upload=False,
            name=name,
        )

        metrics = report.get_metrics()

        return {
            'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
            'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
            'capacity': metrics['liquidity'].get('capacity', 0) or 0,
            'win_rate': metrics['profitability'].get('winRate', 0) or 0,
            'report': report,
        }
    except Exception as e:
        print(f"   ⚠️ 回測失敗: {e}")
        return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1}

# =============================================================================
# 第八部分：主程式 - 策略分析
# =============================================================================
def analyze_strategies():
    """分析各策略表現"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     🐉 九組合媽媽龍策略分析                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  策略 1: 低波動本益比     - 價值投資 + 低風險                             ║
║  策略 2: 小資族           - 小型股成長                                   ║
║  策略 3: 營收股價雙渦輪    - 營收 + 動能                                  ║
║  策略 4: 高殖利率烏龜      - 股息穩健                                    ║
║  策略 5: 低波動性指標      - 低波動大盤                                  ║
║  策略 6: 藏獒外掛大盤指針  - 趨勢突破                                    ║
║  策略 7: 小蝦米跟大鯨魚    - 籌碼面追蹤 (新)                              ║
║  策略 8: 純技術趨勢       - 技術面動能 (新)                              ║
║  策略 9: 財報指標20大     - 財務基本面 (新)                              ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # 預設參數
    default_params = gene_decoder.decode([0.5] * GeneDecoder.GENE_LENGTH)

    results = []
    strategy_names = [
        "1. 低波動本益比",
        "2. 小資族",
        "3. 營收股價雙渦輪",
        "4. 高殖利率烏龜",
        "5. 低波動性指標",
        "6. 藏獒外掛大盤指針",
        "7. 小蝦米跟大鯨魚",
        "8. 純技術趨勢",
        "9. 財報指標20大",
    ]

    strategy_funcs = [
        strategy_engine.strategy_1_low_volatility_pe,
        strategy_engine.strategy_2_small_investor,
        strategy_engine.strategy_3_revenue_price_turbo,
        strategy_engine.strategy_4_high_yield_turtle,
        strategy_engine.strategy_5_low_volatility_index,
        strategy_engine.strategy_6_market_indicator,
        strategy_engine.strategy_7_shrimp_whale,
        strategy_engine.strategy_8_tech_trend,
        strategy_engine.strategy_9_fundamental_20,
    ]

    print("\n📊 各策略獨立回測結果：")
    print("-" * 80)
    print(f"{'策略名稱':<25} {'夏普值':<10} {'年化報酬':<12} {'最大回撤':<12} {'胃納量(萬)':<12}")
    print("-" * 80)

    for name, func in zip(strategy_names, strategy_funcs):
        try:
            pos = func(default_params)
            result = run_backtest(pos, default_params, name)
            results.append(result)

            print(f"{name:<25} {result['sharpe']:<10.2f} {result['annual_return']*100:<10.1f}% "
                  f"{result['max_drawdown']*100:<10.1f}% {result['capacity']/1e4:<10.0f}")
        except Exception as e:
            print(f"{name:<25} 執行失敗: {e}")
            results.append({'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1})

    print("-" * 80)

    # 組合策略回測
    print("\n🐉 九組合媽媽龍 組合回測：")
    combined_pos = strategy_engine.combine_all_strategies(default_params)
    combined_result = run_backtest(combined_pos, default_params, "九組合媽媽龍")

    print(f"   夏普值: {combined_result['sharpe']:.2f}")
    print(f"   年化報酬: {combined_result['annual_return']*100:.1f}%")
    print(f"   最大回撤: {combined_result['max_drawdown']*100:.1f}%")
    print(f"   胃納量: {combined_result['capacity']/1e4:.0f} 萬")

    if 'report' in combined_result and combined_result['report'] is not None:
        print("\n📈 顯示詳細回測報告：")
        combined_result['report'].display()

    return results, combined_result

# =============================================================================
# 第九部分：進階回測 - 含 live performance
# =============================================================================
def run_live_backtest():
    """執行 live performance 回測"""
    print("\n" + "=" * 80)
    print("🔴 Live Performance 回測")
    print("=" * 80)

    default_params = gene_decoder.decode([0.5] * GeneDecoder.GENE_LENGTH)

    # 策略7: 小蝦米跟大鯨魚
    print("\n📊 策略7 - 小蝦米跟大鯨魚 (Live from 2022-03-01):")
    pos_7 = strategy_engine.strategy_7_shrimp_whale(default_params)
    if pos_7 is not None and not pos_7.empty:
        try:
            r7 = sim(pos_7, name='小蝦米跟大鯨魚', live_performance_start='2022-03-01')
            r7.display()
        except Exception as e:
            print(f"   回測失敗: {e}")

    # 策略8: 純技術趨勢
    print("\n📊 策略8 - 純技術趨勢 (Quarterly Resample):")
    pos_8 = strategy_engine.strategy_8_tech_trend(default_params)
    if pos_8 is not None and not pos_8.empty:
        try:
            r8 = sim(pos_8, resample='Q')
            r8.display()
        except Exception as e:
            print(f"   回測失敗: {e}")

    # 策略9: 財報指標20大
    print("\n📊 策略9 - 財報指標20大 (Live from 2022-11-27):")
    pos_9 = strategy_engine.strategy_9_fundamental_20(default_params)
    if pos_9 is not None and not pos_9.empty:
        try:
            r9 = sim(pos_9, resample='W', live_performance_start='2022-11-27')
            r9.display()
        except Exception as e:
            print(f"   回測失敗: {e}")

# =============================================================================
# 主程式入口
# =============================================================================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                   🐉 九組合媽媽龍策略系統                             ║
    ║                   Nine Combo Mama Dragon Strategy                   ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  原六組合快快龍 + 三個新策略 = 九組合媽媽龍                           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 執行策略分析
    results, combined = analyze_strategies()

    # 執行 live performance 回測
    run_live_backtest()

    print("\n✅ 九組合媽媽龍策略分析完成！")
