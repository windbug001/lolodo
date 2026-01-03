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

# 🎯 優化目標（依用戶需求設定）
TARGET_SHARPE = 4.2          # 目標夏普值 >= 4.2
MIN_CAPACITY = 10_000_000    # 最少胃納量 1000萬台幣
TARGET_ANNUAL_RETURN = 0.35  # 目標年化報酬 35%
MAX_DRAWDOWN = 0.20          # 最大回檔 -20%
MIN_POSITION_RATIO = 0.03    # 每股占比至少 3%

# GA 演化參數
POPULATION_SIZE = 60
N_GENERATIONS = 150
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

# 🔥 完整回測與斷點設定
FULL_BACKTEST_INTERVAL = 5   # 每 5 代進行完整回測
CHECKPOINT_INTERVAL = 5      # 每 5 代保存斷點

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
        """策略一：低波動本益比 (20個參數: s1_*)"""
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

        # 使用 s1_ 前綴參數
        cond1 = rev_ma3 / rev_ma12 > params.get('s1_rev_ma3_ma12_ratio', 1.0)
        cond2 = rev / rev.shift(1) > params.get('s1_rev_consistency', 0.8)

        tree_select_factor = ((融資使用率 <= params.get('s1_margin_usage_limit', 40))
                             & (entry_volatility <= params.get('s1_volatility_threshold', 0.04))
                             & (業外收支營收率 < params.get('s1_non_op_income_limit', 10)))

        condition_成交量 = vol.average(1) > params.get('s1_min_volume', 100000)
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 30).sustain(12, 8)
        cond單月營收月增率 = (rev_month_growth > -54).sustain(3)

        # 使用可調整的均線參數
        ma_short = int(params.get('s1_ma_short', 40))
        ma_mid = int(params.get('s1_ma_mid', 75))
        ma_long = int(params.get('s1_ma_long', 200))
        cond收盤價大於均線 = (close > close.average(ma_short)) & (close > close.average(ma_mid)) & (close > close.average(ma_long))
        cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)

        pe_range = (params.get('s1_pe_min', 5) <= pe) & (pe <= params.get('s1_pe_max', 25))
        pb_range = (params.get('s1_pb_min', 0.5) <= pb) & (pb <= params.get('s1_pb_max', 2.8))
        gpm_trend = (營業毛利率 > params.get('s1_min_gpm', 8)).sustain(int(params.get('s1_gpm_sustain', 2)))
        roe_trend = (ROE綜合損益 > params.get('s1_min_roe', 0)).sustain(int(params.get('s1_roe_sustain', 2)))

        try:
            inv_level_max = int(params.get('s1_inv_level_max', 8))
            inv_ratio_max = params.get('s1_inv_ratio_max', 46)
            small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= inv_level_max)]
                            .reset_index()
                            .groupby(["date", "stock_id"])
                            .agg({"占集保庫存數比例": "sum"})
                            .reset_index()
                            .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= inv_ratio_max
        except:
            small_inv_under50 = True

        cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老
                   & cond排除月營收連3月衰退 & cond收盤價大於均線 & cond近三個月營收大於年營收
                   & cond單月營收月增率 & ~limit_up_all_day & condition_成交量
                   & gpm_trend & roe_trend & small_inv_under50 & pe_range & pb_range)

        position = peg[cond_all & (peg > 0)].is_smallest(int(params.get('s1_top_n', 5)))
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 2：小資族策略
    # =========================================================================
    def strategy_2_small_investor(self, params: Dict) -> Any:
        """策略二：小資族 (20個參數: s2_*)"""
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

        # 使用 s2_ 前綴參數
        condition1 = (市值 < params.get('s2_market_value_limit', 15e9))
        condition2 = 自由現金流 > params.get('s2_min_free_cash', 0)
        condition3 = 股東權益報酬率 > params.get('s2_min_roe', 0)
        condition4 = 營業利益成長率 > params.get('s2_min_op_growth', -1)
        condition5 = 市值營收比 < params.get('s2_market_rev_ratio_limit', 3)
        condition6 = vol > params.get('s2_volume_threshold', 100000)

        # 使用可調整的衰退門檻
        rev_decline = params.get('s2_rev_decline_threshold', -10)
        sustain_period = int(params.get('s2_sustain_period', 3))
        old_period = int(params.get('s2_old_trend_period', 12))
        old_match = int(params.get('s2_old_trend_match', 8))

        cond排除月營收連3月衰退 = ~(rev_yoy_growth < rev_decline).sustain(sustain_period)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(old_period, old_match)
        cond單月營收月增率 = (rev_month_growth > params.get('s2_rev_mom_growth_limit', -54)).sustain(sustain_period)

        # 使用可調整的均線參數
        ma_short = int(params.get('s2_ma_short', 60))
        ma_mid = int(params.get('s2_ma_mid', 120))
        cond收盤價大於均線 = (close > close.average(ma_short)) & (close > close.average(ma_mid)) & (close > close.average(75))

        # 營收比較 (使用可調整窗口)
        rev_bottom_window = int(params.get('s2_rev_bottom_window', 12))
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(rev_bottom_window)
        業外收支營收率占比低 = (業外收支營收率 < params.get('s2_non_op_limit', 7.3))

        rsv_period = int(params.get('s2_rsv_period', 50))
        rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

        position = ((condition1 & condition2 & condition3 & condition4 & condition5
                    & condition6 & cond排除月營收成長趨勢過老 & cond單月營收月增率
                    & cond排除月營收連3月衰退 & cond近三個月營收大於年營收
                    & 業外收支營收率占比低) * rsv).is_largest(int(params.get('s2_top_n', 6)))

        position = position.reindex(當月營收.index_str_to_date().index)
        return position

    # =========================================================================
    # 策略 3：營收股價雙渦輪策略
    # =========================================================================
    def strategy_3_revenue_price_turbo(self, params: Dict) -> Any:
        """策略三：營收股價雙渦輪 (20個參數: s3_*)"""
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

        # 使用 s3_ 前綴參數
        rev_ma_period = int(params.get('s3_rev_ma_period', 4))
        rev_ma = rev.average(rev_ma_period)
        rev_ma_lookback = int(params.get('s3_rev_ma_lookback', 20))

        condition_近n月平均營收創新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()
        price_high_window = int(params.get('s3_price_high_window', 8))
        condition_近n日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
        condition_成交量 = vol.average(1) > params.get('s3_min_volume', 200000)

        long_ma_pattern = ((close > close.average(5)) & (close > close.average(10))
                          & (close > close.average(20)) & (close > close.average(60))
                          & (close > close.average(150)) & (close > close.average(200)))

        performance_ma = int(params.get('s3_performance_ma', 250))
        收盤價_超級績效 = close > (close.average(performance_ma) * 1.1)

        rsi_sustain = int(params.get('s3_rsi_sustain', 1))
        rsi_higt_trend = (rsi > params.get('s3_rsi_threshold', 60)).sustain(rsi_sustain)

        gpm_sustain = int(params.get('s3_gpm_sustain', 5))
        gpm_trend = (營業毛利率 > params.get('s3_min_gpm', 5)).sustain(gpm_sustain)

        btpm_sustain = int(params.get('s3_btpm_sustain', 1))
        btpm_trend = (稅前淨利率 > params.get('s3_min_btpm', 4)).sustain(btpm_sustain)

        atpm_sustain = int(params.get('s3_atpm_sustain', 1))
        atpm_trend = (稅後淨利率 > params.get('s3_min_atpm', 3)).sustain(atpm_sustain)

        rev_growth_pct = params.get('s3_rev_growth_pct', 0.9)
        rev_rise_nsatisfy = rev_yoy_growth.rank(pct=True, axis=1) > rev_growth_pct

        try:
            boss_min_level = int(params.get('s3_boss_min_level', 12))
            boss_max_level = int(params.get('s3_boss_max_level', 16))
            boss_ratio = params.get('s3_boss_ratio', 18)
            boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= boss_min_level) & (inventory.持股分級.astype(int) <= boss_max_level)]
                            .reset_index()
                            .groupby(["date", "stock_id"])
                            .agg({"占集保庫存數比例": "sum"})
                            .reset_index()
                            .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= boss_ratio
        except:
            boss_inventory_over400 = True

        pe_range = ~(params.get('s3_pe_limit', 150) <= pe)

        conditions = (condition_近n月平均營收創新高 & condition_近n日內有1日股價創新高
                     & condition_成交量 & long_ma_pattern & gpm_trend & btpm_trend
                     & atpm_trend & rev_rise_nsatisfy & (close > params.get('s3_min_price', 15))
                     & (業外收支營收率 < 7.3) & pe_range & 收盤價_超級績效
                     & ((vol >= vol.rolling(20).mean() * 0.8)) & rsi_higt_trend
                     & boss_inventory_over400 & ~limit_up_all_day)

        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(int(params.get('s3_top_n', 12)))
        position = position.reindex(rev.index_str_to_date().index, method="ffill")
        return position

    # =========================================================================
    # 策略 4：高殖利率烏龜策略
    # =========================================================================
    def strategy_4_high_yield_turtle(self, params: Dict) -> Any:
        """策略四：高殖利率烏龜 (16個參數: s4_*)"""
        yield_ratio = self.dl.get('dividend_yield')
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        營業利益率 = self.dl.get('營業利益率')
        營業毛利率 = self.dl.get('營業毛利率')
        稅後淨利率 = self.dl.get('稅後淨利率')
        pb = self.dl.get('pb')
        pe = self.dl.get('pe')
        董監持有股數占比 = self.dl.get('董監持有股數占比')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')

        # 使用 s4_ 前綴參數
        sma_short = int(params.get('s4_sma_short', 20))
        sma_long = int(params.get('s4_sma_long', 60))
        sma20 = close.average(sma_short)
        sma60 = close.average(sma_long)

        cond1 = yield_ratio >= params.get('s4_min_yield_ratio', 4)
        cond2 = (close > sma20) & (close > sma60)

        rev_period = int(params.get('s4_rev_period', 3))
        rev_compare = int(params.get('s4_rev_compare', 12))
        cond3 = rev.average(rev_period) > rev.average(rev_compare)

        cond4 = 營業利益率 >= params.get('s4_min_op_earn_ratio', 8)
        cond5 = 董監持有股數占比 >= params.get('s4_min_boss_hold', 15)
        cond6 = (vol.average(5) >= params.get('s4_min_volume', 100000)) & (vol.average(5) <= params.get('s4_max_volume', 3000000))

        # 額外財務條件
        cond7 = 營業毛利率 >= params.get('s4_min_gpm', 5)
        cond8 = 稅後淨利率 >= params.get('s4_min_npm', 3)
        cond9 = pb <= params.get('s4_pb_max', 4)
        cond10 = pe <= params.get('s4_pe_max', 30)

        sustain = int(params.get('s4_sustain', 2))
        rev_growth_cond = (rev_yoy_growth > params.get('s4_min_rev_growth', -10)).sustain(sustain)

        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9 & cond10 & rev_growth_cond
        cond_all = cond_all * rev_yoy_growth
        position = cond_all[cond_all > 0].is_largest(int(params.get('s4_top_n', 8)))
        position = position.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 5：低波動性指標策略
    # =========================================================================
    def strategy_5_low_volatility_index(self, params: Dict) -> Any:
        """策略五：低波動性指標 (16個參數: s5_*)"""
        cap = self.dl.get('市值')
        vol = self.dl.get('vol')
        close = self.dl.get('close')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        稅後淨利率 = self.dl.get('稅後淨利率')
        pe = self.dl.get('pe')
        dividend_yield = self.dl.get('dividend_yield')

        # 使用 s5_ 前綴參數
        std_window = int(params.get('s5_std_window', 20))
        std = close.pct_change().rolling(std_window).std().rank(axis=1, pct=True)

        # 均線參數
        ma_short = int(params.get('s5_ma_short', 60))
        ma_mid = int(params.get('s5_ma_mid', 120))
        ma_long = int(params.get('s5_ma_long', 250))

        # Beta 計算
        beta_window = int(params.get('s5_beta_window', 60))
        market_return = close.mean(axis=1).pct_change()
        stock_return = close.pct_change()
        beta_cond = True  # 簡化處理

        # 基本面條件
        use_cap = params.get('s5_use_cap', True)
        if use_cap:
            cap_cond = cap > params.get('s5_min_market_cap', 50e9)
        else:
            cap_cond = True

        roe_cond = ROE綜合損益 > params.get('s5_min_roe', 5) if ROE綜合損益 is not None else True
        npm_cond = 稅後淨利率 > params.get('s5_min_npm', 3) if 稅後淨利率 is not None else True
        pe_cond = pe < params.get('s5_max_pe', 40) if pe is not None else True
        yield_cond = dividend_yield > params.get('s5_min_yield', 2) if dividend_yield is not None else True

        volatility_pct = params.get('s5_volatility_pct', 0.3)

        position = cap[(vol.average(20) > params.get('s5_min_volume', 100000)) &
                       (close > close.average(ma_short)) &
                       (close > close.average(ma_mid)) &
                       (close > close.average(ma_long)) &
                       (std < params.get('s5_std_threshold', volatility_pct)) &
                       cap_cond & roe_cond & npm_cond & pe_cond & yield_cond]

        position = position.is_smallest(int(params.get('s5_top_n', 15)))
        position = position.reindex(close.index, method='ffill')
        return position

    # =========================================================================
    # 策略 6：藏獒外掛大盤指針策略
    # =========================================================================
    def strategy_6_market_indicator(self, params: Dict) -> Any:
        """策略六：藏獒外掛大盤指針 (16個參數: s6_*)"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        rev = self.dl.get('rev')
        rev_yoy_growth = self.dl.get('rev_yoy_growth')
        rev_month_growth = self.dl.get('rev_month_growth')

        # 使用 s6_ 前綴參數
        vol_ma_period = int(params.get('s6_volume_ma_period', 10))
        vol_ma = vol.average(vol_ma_period)
        new_high_window = int(params.get('s6_new_high_window', 120))

        cond1 = (close == close.rolling(new_high_window).max())

        # 營收衰退排除
        min_year_growth = params.get('s6_min_year_growth', -15)
        decline_period = int(params.get('s6_decline_period', 3))
        cond2 = ~(rev_yoy_growth < min_year_growth).sustain(decline_period)

        # 營收成長趨勢過老排除
        max_year_growth = params.get('s6_max_year_growth', 60)
        old_period = int(params.get('s6_old_period', 12))
        old_match = int(params.get('s6_old_match', 8))
        cond3 = ~(rev_yoy_growth > max_year_growth).sustain(old_period, old_match)

        # 營收底部反轉
        rev_window = int(params.get('s6_rev_window', 12))
        rev_bottom_ratio = params.get('s6_rev_bottom_ratio', 1.2)
        sustain_period = int(params.get('s6_sustain_period', 3))
        cond4 = ((rev.rolling(rev_window).min())/(rev) < rev_bottom_ratio).sustain(sustain_period)

        # 月營收成長
        min_month_growth = params.get('s6_min_month_growth', -10)
        cond5 = (rev_month_growth > min_month_growth).sustain(sustain_period)
        cond6 = vol_ma > params.get('s6_min_volume', 100000)

        # 均線條件 (可選)
        ma_short = int(params.get('s6_ma_short', 60))
        ma_long = int(params.get('s6_ma_long', 120))
        cond7 = (close > close.average(ma_short)) & (close > close.average(ma_long))

        buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7
        buy = vol_ma * buy
        buy = buy[buy > 0]

        # 選擇最小或最大
        use_smallest = params.get('s6_use_smallest', True)
        if use_smallest:
            buy = buy.is_smallest(int(params.get('s6_top_n', 10)))
        else:
            buy = buy.is_largest(int(params.get('s6_top_n', 10)))

        position = buy.reindex(rev.index_str_to_date().index, method='ffill')
        return position

    # =========================================================================
    # 策略 7：小蝦米跟大鯨魚 (新增)
    # =========================================================================
    def strategy_7_shrimp_whale(self, params: Dict) -> Any:
        """策略七：小蝦米跟大鯨魚 - 籌碼面策略 (20個參數: s7_*)"""
        inv = self.dl.get('inventory')
        rev = self.dl.get('rev')
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        interest = self.dl.get('dividend_yield')
        grow = self.dl.get('營業利益成長率')

        # 使用 s7_ 前綴參數
        small_inv_max_level = int(params.get('s7_small_inv_max_level', 5))
        big_inv_min_level = int(params.get('s7_big_inv_min_level', 9))
        big_inv_max_level = int(params.get('s7_big_inv_max_level', 15))

        # 散戶持股
        h1 = inv[inv.持股分級.astype(int) <= small_inv_max_level].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')

        # 大戶持股
        h2 = inv[(inv.持股分級.astype(int) >= big_inv_min_level) & (inv.持股分級.astype(int) <= big_inv_max_level)].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')

        # 大戶持股比例
        ratio = (h2 / (h1 + h2))

        # 營收指標
        ma_period = int(params.get('s7_ma_period', 250))
        rev_yoy_window = int(params.get('s7_rev_yoy_window', 12))
        rev_mom_window = int(params.get('s7_rev_mom_window', 1))

        rev_mom = rev / rev.shift(rev_mom_window)
        rev_yoy2 = rev.rolling(2).mean() / rev.shift(rev_yoy_window).rolling(2).mean()

        # 權重設定
        ratio_weight = params.get('s7_ratio_weight', 1.0)
        rev_weight = params.get('s7_rev_weight', 1.0)
        mom_weight = params.get('s7_mom_weight', 1.0)

        # 多階段篩選
        first_filter = int(params.get('s7_first_filter', 50))
        second_filter = int(params.get('s7_second_filter', 35))
        third_filter = int(params.get('s7_third_filter', 10))

        # 第一層
        p = (FinlabDataFrame(ratio).rank(axis=1, pct=True) * ratio_weight *
             (close.notna() & (close > close.average(ma_period)))
             + rev_yoy2.rank(axis=1, pct=True) * rev_weight
             + rev_mom.rank(axis=1, pct=True) * mom_weight).is_largest(first_filter)

        # 第二層：營收成長 (可選)
        use_grow = params.get('s7_use_grow', True)
        if use_grow:
            grow_threshold = params.get('s7_grow_threshold', -100)
            p = (p * (grow > grow_threshold) * grow).is_largest(second_filter)
        else:
            p = p.is_largest(second_filter)

        # 第三層：殖利率 (可選)
        use_interest = params.get('s7_use_interest', True)
        if use_interest:
            interest_threshold = params.get('s7_interest_threshold', 0)
            p = (p * (interest > interest_threshold) * interest).is_largest(third_filter)
        else:
            p = p.is_largest(third_filter)

        # 第四層：大戶增持
        ratio_diff_period = int(params.get('s7_ratio_diff_period', 8))
        p = (p * ratio.diff(ratio_diff_period)).is_largest(int(params.get('s7_top_n', 5)))

        # 成交量過濾
        vol_cond = vol.average(5) > params.get('s7_min_volume', 100000)
        p = p * vol_cond

        # Resample (可選)
        resample = params.get('s7_resample', None)
        if resample:
            p = p.resample(resample).last()

        p = p.reindex(rev.index_str_to_date().index, method='ffill')
        return p

    # =========================================================================
    # 策略 8：純技術趨勢策略 (新增)
    # =========================================================================
    def strategy_8_tech_trend(self, params: Dict) -> Any:
        """策略八：純技術趨勢策略 (20個參數: s8_*)"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')

        # 使用 s8_ 前綴參數
        wma_period = int(params.get('s8_wma_period', 60))
        zlma_period = int(params.get('s8_zlma_period', 120))
        bias_sma_period = int(params.get('s8_bias_sma_period', 250))
        slope_period = int(params.get('s8_slope_period', 60))
        kurtosis_period = int(params.get('s8_kurtosis_period', 250))
        skew_period = int(params.get('s8_skew_period', 180))

        def wma(price, n):
            return price.ewm(com=n).mean()

        def zlma(price, n):
            lag = (n - 1) // 2
            series = 2 * price - price.shift(lag)
            return wma(series, n)

        # 計算技術指標
        bias_sma = close / close.rolling(bias_sma_period).mean() - 1
        bias_zlma = close / close.apply(lambda s: zlma(s, zlma_period)) - 1
        bias_wma = close / close.apply(lambda s: wma(s, wma_period)) - 1
        slope1_sma = close.rolling(slope_period).mean().pipe(lambda df: df / df.shift(slope_period) - 1)
        kurtosis = close.pct_change().rolling(kurtosis_period).kurt()
        skewness = close.pct_change().rolling(skew_period).skew()

        # 均線確認 (可選)
        ma_confirm = params.get('s8_ma_confirm', True)
        if ma_confirm:
            ma_short = int(params.get('s8_ma_short', 20))
            ma_long = int(params.get('s8_ma_long', 60))
            ma_cond = (close > close.average(ma_short)) & (close > close.average(ma_long))
        else:
            ma_cond = True

        # 成交量均線
        vol_ma_period = int(params.get('s8_vol_ma_period', 10))

        # 條件篩選
        pos = vol.average(vol_ma_period)[
            (bias_sma > params.get('s8_bias_sma_threshold', 0.1)) &
            (bias_zlma > params.get('s8_bias_zlma_threshold', 0.1)) &
            (bias_wma > params.get('s8_bias_wma_threshold', 0.1)) &
            (slope1_sma.rank(axis=1, pct=True) > params.get('s8_slope_percentile', 0.5)) &
            (kurtosis.rank(axis=1, pct=True) > params.get('s8_kurtosis_percentile', 0.4)) &
            (skewness < params.get('s8_skew_threshold', 0)) &
            (vol > params.get('s8_min_volume', 200000)) &
            ma_cond
        ]

        # 選擇最小或最大
        use_smallest = params.get('s8_use_smallest', True)
        if use_smallest:
            pos = pos.is_smallest(int(params.get('s8_top_n', 10)))
        else:
            pos = pos.is_largest(int(params.get('s8_top_n', 10)))

        # Resample (可選)
        resample = params.get('s8_resample', None)
        if resample:
            pos = pos.resample(resample).last()

        return pos

    # =========================================================================
    # 策略 9：財報指標20大 (新增)
    # =========================================================================
    def strategy_9_fundamental_20(self, params: Dict) -> Any:
        """策略九：財報指標20大 (17個參數: s9_*)"""
        close = self.dl.get('close')
        vol = self.dl.get('vol')
        fundamental_features = self.dl.get('fundamental_features')
        ROE綜合損益 = self.dl.get('ROE綜合損益')
        稅後淨利率 = self.dl.get('稅後淨利率')

        if not fundamental_features:
            return pd.DataFrame()

        # 使用 s9_ 前綴參數
        std_window = int(params.get('s9_std_window', 60))
        ma_period = int(params.get('s9_ma_period', 60))
        std_percentile = params.get('s9_std_percentile', 0.5)
        feature_count = int(params.get('s9_feature_count', 20))
        deadline_shift = int(params.get('s9_deadline_shift', 0))
        rank_method = params.get('s9_rank_method', 'pct')

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
        use_roe = params.get('s9_use_roe', True)
        use_npm = params.get('s9_use_npm', True)
        use_growth = params.get('s9_use_growth', True)
        use_liquidity = params.get('s9_use_liquidity', True)
        use_leverage = params.get('s9_use_leverage', True)

        for key, df in list(fundamental_features.items())[:feature_count]:
            if df is not None:
                try:
                    # 根據參數決定是否使用某類特徵
                    if 'ROE' in key and not use_roe:
                        continue
                    if '淨利率' in key and not use_npm:
                        continue
                    if '成長率' in key and not use_growth:
                        continue
                    if '流動' in key and not use_liquidity:
                        continue
                    if '負債' in key and not use_leverage:
                        continue

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
        std = close.pct_change().rolling(std_window).std()

        cond1_weight = params.get('s9_cond1_weight', 5)
        cond2_weight = params.get('s9_cond2_weight', 5)
        cond3_weight = params.get('s9_cond3_weight', 10)

        cond1 = (close > close.average(ma_period)).astype(float) * cond1_weight
        cond2 = (std.rank(pct=True, axis=1) < std_percentile).astype(float) * cond2_weight
        cond3 = (vol.rolling(5).mean() > params.get('s9_min_volume', 200000)).astype(float) * cond3_weight

        # 綜合評分
        if rank_method == 'pct':
            score = sum([f.deadline(deadline_shift).rank(axis=1, pct=True).fillna(0) for f in fs])
        else:
            score = sum([f.deadline(deadline_shift).rank(axis=1, method='dense').fillna(0) for f in fs])

        final_score = score + (cond1 + cond2 + cond3).reindex(score.index, method='ffill')

        position = final_score.is_largest(int(params.get('s9_top_n', 15)))

        # Resample (可選)
        resample = params.get('s9_resample', None)
        if resample:
            position = position.resample(resample).last()

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
    """
    基因解碼器 - 九策略完整版 (180個參數)

    基因結構：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 0-8     : 九策略權重 (9個)                                       │
    │ 9-28    : 策略1 低波動本益比 (20個)                              │
    │ 29-48   : 策略2 小資族 (20個)                                    │
    │ 49-68   : 策略3 營收股價雙渦輪 (20個)                            │
    │ 69-84   : 策略4 高殖利率烏龜 (16個)                              │
    │ 85-100  : 策略5 低波動性指標 (16個)                              │
    │ 101-116 : 策略6 藏獒外掛大盤指針 (16個)                          │
    │ 117-136 : 策略7 小蝦米跟大鯨魚 (20個)                            │
    │ 137-156 : 策略8 純技術趨勢 (20個)                                │
    │ 157-173 : 策略9 財報指標20大 (17個)                              │
    │ 174-179 : 回測與風控參數 (6個)                                   │
    └─────────────────────────────────────────────────────────────────┘
    總計：180 個可演化參數
    """

    GENE_LENGTH = 180  # 完整180個參數

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """解碼基因為策略參數"""
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # =========================================================
        # 九策略權重 (0-8) - 9個參數
        # =========================================================
        raw_weights = [max(0.01, genes[i]) for i in range(9)]
        total = sum(raw_weights)
        for i in range(9):
            params[f'weight_{i+1}'] = raw_weights[i] / total

        # =========================================================
        # 策略1: 低波動本益比 (9-28) - 20個參數
        # =========================================================
        params['s1_rev_ma3_ma12_ratio'] = genes[9] * 0.5 + 1.0      # 1.0-1.5
        params['s1_rev_consistency'] = genes[10] * 0.4 + 0.5        # 0.5-0.9
        params['s1_volatility_threshold'] = genes[11] * 0.05 + 0.02 # 0.02-0.07
        params['s1_margin_usage_limit'] = genes[12] * 30 + 20       # 20-50
        params['s1_non_op_income_limit'] = genes[13] * 10 + 5       # 5-15
        params['s1_pe_min'] = genes[14] * 10 + 3                    # 3-13
        params['s1_pe_max'] = genes[15] * 20 + 15                   # 15-35
        params['s1_pb_min'] = genes[16] * 1.5 + 0.5                 # 0.5-2.0
        params['s1_pb_max'] = genes[17] * 2.0 + 2.0                 # 2.0-4.0
        params['s1_min_gpm'] = genes[18] * 15 + 5                   # 5-20%
        params['s1_gpm_sustain'] = int(genes[19] * 4 + 2)           # 2-6
        params['s1_min_roe'] = genes[20] * 15 + 5                   # 5-20%
        params['s1_roe_sustain'] = int(genes[21] * 4 + 2)           # 2-6
        params['s1_inv_level_max'] = int(genes[22] * 4 + 6)         # 6-10
        params['s1_inv_ratio_max'] = genes[23] * 20 + 35            # 35-55%
        params['s1_ma_short'] = int(genes[24] * 30 + 40)            # 40-70
        params['s1_ma_mid'] = int(genes[25] * 30 + 70)              # 70-100
        params['s1_ma_long'] = int(genes[26] * 50 + 200)            # 200-250
        params['s1_min_volume'] = genes[27] * 200000 + 100000       # 10萬-30萬
        params['s1_top_n'] = int(genes[28] * 8 + 3)                 # 3-11

        # =========================================================
        # 策略2: 小資族 (29-48) - 20個參數
        # =========================================================
        params['s2_market_value_limit'] = genes[29] * 15e9 + 5e9    # 50億-200億
        params['s2_market_rev_ratio_limit'] = genes[30] * 3 + 2     # 2-5
        params['s2_rev_yoy_growth_limit'] = genes[31] * 15 - 20     # -20 to -5
        params['s2_rev_mom_growth_limit'] = genes[32] * 40 - 60     # -60 to -20
        params['s2_rsv_period'] = int(genes[33] * 40 + 30)          # 30-70
        params['s2_ma_period'] = int(genes[34] * 50 + 40)           # 40-90
        params['s2_volume_threshold'] = genes[35] * 200000 + 100000 # 10萬-30萬
        params['s2_non_op_limit'] = genes[36] * 5 + 5               # 5-10
        params['s2_min_free_cash'] = genes[37] * 10 - 5             # -5 to 5
        params['s2_min_roe'] = genes[38] * 15 + 5                   # 5-20%
        params['s2_min_op_growth'] = genes[39] * 30 - 10            # -10 to 20
        params['s2_ma_short'] = int(genes[40] * 40 + 40)            # 40-80
        params['s2_ma_mid'] = int(genes[41] * 40 + 100)             # 100-140
        params['s2_sustain_period'] = int(genes[42] * 4 + 2)        # 2-6
        params['s2_rev_bottom_window'] = int(genes[43] * 8 + 8)     # 8-16
        params['s2_rev_bottom_ratio'] = genes[44] * 0.3 + 1.1       # 1.1-1.4
        params['s2_rev_decline_threshold'] = genes[45] * 20 - 35    # -35 to -15
        params['s2_old_trend_period'] = int(genes[46] * 6 + 10)     # 10-16
        params['s2_old_trend_match'] = int(genes[47] * 4 + 6)       # 6-10
        params['s2_top_n'] = int(genes[48] * 8 + 4)                 # 4-12

        # =========================================================
        # 策略3: 營收股價雙渦輪 (49-68) - 20個參數
        # =========================================================
        params['s3_rev_ma_period'] = int(genes[49] * 5 + 3)         # 3-8
        params['s3_rev_ma_lookback'] = int(genes[50] * 18 + 12)     # 12-30
        params['s3_price_high_window'] = int(genes[51] * 10 + 5)    # 5-15
        params['s3_min_volume'] = genes[52] * 300000 + 200000       # 20萬-50萬
        params['s3_min_price'] = genes[53] * 20 + 10                # 10-30
        params['s3_rsi_threshold'] = genes[54] * 20 + 50            # 50-70
        params['s3_rsi_sustain'] = int(genes[55] * 3 + 1)           # 1-4
        params['s3_pe_limit'] = genes[56] * 100 + 100               # 100-200
        params['s3_min_gpm'] = genes[57] * 10 + 5                   # 5-15%
        params['s3_gpm_sustain'] = int(genes[58] * 4 + 3)           # 3-7
        params['s3_min_btpm'] = genes[59] * 6 + 2                   # 2-8%
        params['s3_btpm_sustain'] = int(genes[60] * 2 + 1)          # 1-3
        params['s3_min_atpm'] = genes[61] * 5 + 2                   # 2-7%
        params['s3_atpm_sustain'] = int(genes[62] * 2 + 1)          # 1-3
        params['s3_rev_growth_pct'] = genes[63] * 0.2 + 0.8         # 0.8-1.0
        params['s3_boss_min_level'] = int(genes[64] * 3 + 10)       # 10-13
        params['s3_boss_max_level'] = int(genes[65] * 3 + 14)       # 14-17
        params['s3_boss_ratio'] = genes[66] * 15 + 15               # 15-30%
        params['s3_performance_ma'] = int(genes[67] * 100 + 200)    # 200-300
        params['s3_top_n'] = int(genes[68] * 12 + 6)                # 6-18

        # =========================================================
        # 策略4: 高殖利率烏龜 (69-84) - 16個參數
        # =========================================================
        params['s4_min_yield_ratio'] = genes[69] * 4 + 3            # 3-7%
        params['s4_min_op_earn_ratio'] = genes[70] * 12 + 5         # 5-17%
        params['s4_min_boss_hold'] = genes[71] * 25 + 10            # 10-35%
        params['s4_min_volume'] = genes[72] * 150000 + 50000        # 5萬-20萬
        params['s4_max_volume'] = genes[73] * 3000000 + 1000000     # 100萬-400萬
        params['s4_sma_short'] = int(genes[74] * 15 + 10)           # 10-25
        params['s4_sma_long'] = int(genes[75] * 40 + 40)            # 40-80
        params['s4_rev_period'] = int(genes[76] * 4 + 2)            # 2-6
        params['s4_rev_compare'] = int(genes[77] * 8 + 10)          # 10-18
        params['s4_min_rev_growth'] = genes[78] * 20 - 10           # -10 to 10
        params['s4_min_gpm'] = genes[79] * 10 + 5                   # 5-15%
        params['s4_min_npm'] = genes[80] * 8 + 3                    # 3-11%
        params['s4_pb_max'] = genes[81] * 2 + 2                     # 2-4
        params['s4_pe_max'] = genes[82] * 15 + 15                   # 15-30
        params['s4_sustain'] = int(genes[83] * 3 + 2)               # 2-5
        params['s4_top_n'] = int(genes[84] * 8 + 5)                 # 5-13

        # =========================================================
        # 策略5: 低波動性指標 (85-100) - 16個參數
        # =========================================================
        params['s5_min_volume'] = genes[85] * 200000 + 100000       # 10萬-30萬
        params['s5_std_window'] = int(genes[86] * 25 + 15)          # 15-40
        params['s5_std_threshold'] = genes[87] * 0.3 + 0.2          # 0.2-0.5
        params['s5_ma_short'] = int(genes[88] * 30 + 40)            # 40-70
        params['s5_ma_mid'] = int(genes[89] * 40 + 100)             # 100-140
        params['s5_ma_long'] = int(genes[90] * 50 + 200)            # 200-250
        params['s5_beta_window'] = int(genes[91] * 40 + 60)         # 60-100
        params['s5_beta_max'] = genes[92] * 0.5 + 0.5               # 0.5-1.0
        params['s5_min_market_cap'] = genes[93] * 50e9 + 50e9       # 500億-1000億
        params['s5_min_roe'] = genes[94] * 10 + 5                   # 5-15%
        params['s5_min_npm'] = genes[95] * 8 + 3                    # 3-11%
        params['s5_max_pe'] = genes[96] * 20 + 20                   # 20-40
        params['s5_min_yield'] = genes[97] * 3 + 2                  # 2-5%
        params['s5_volatility_pct'] = genes[98] * 0.3 + 0.2         # 0.2-0.5
        params['s5_use_cap'] = genes[99] > 0.5                      # True/False
        params['s5_top_n'] = int(genes[100] * 15 + 10)              # 10-25

        # =========================================================
        # 策略6: 藏獒外掛大盤指針 (101-116) - 16個參數
        # =========================================================
        params['s6_new_high_window'] = int(genes[101] * 160 + 60)   # 60-220
        params['s6_min_year_growth'] = genes[102] * 20 - 25         # -25 to -5
        params['s6_max_year_growth'] = genes[103] * 40 + 50         # 50-90
        params['s6_rev_bottom_ratio'] = genes[104] * 0.3 + 1.1      # 1.1-1.4
        params['s6_min_month_growth'] = genes[105] * 15 - 20        # -20 to -5
        params['s6_min_volume'] = genes[106] * 200000 + 100000      # 10萬-30萬
        params['s6_sustain_period'] = int(genes[107] * 3 + 2)       # 2-5
        params['s6_volume_ma_period'] = int(genes[108] * 10 + 5)    # 5-15
        params['s6_rev_window'] = int(genes[109] * 8 + 8)           # 8-16
        params['s6_decline_period'] = int(genes[110] * 3 + 2)       # 2-5
        params['s6_old_period'] = int(genes[111] * 6 + 10)          # 10-16
        params['s6_old_match'] = int(genes[112] * 4 + 6)            # 6-10
        params['s6_ma_short'] = int(genes[113] * 30 + 40)           # 40-70
        params['s6_ma_long'] = int(genes[114] * 60 + 100)           # 100-160
        params['s6_use_smallest'] = genes[115] > 0.5                # True/False
        params['s6_top_n'] = int(genes[116] * 12 + 8)               # 8-20

        # =========================================================
        # 策略7: 小蝦米跟大鯨魚 (117-136) - 20個參數
        # =========================================================
        params['s7_small_inv_max_level'] = int(genes[117] * 4 + 3)  # 3-7級
        params['s7_big_inv_min_level'] = int(genes[118] * 3 + 9)    # 9-12級
        params['s7_big_inv_max_level'] = int(genes[119] * 3 + 13)   # 13-16級
        params['s7_ma_period'] = int(genes[120] * 150 + 200)        # 200-350
        params['s7_rev_yoy_window'] = int(genes[121] * 4 + 10)      # 10-14
        params['s7_rev_mom_window'] = int(genes[122] * 3 + 1)       # 1-4
        params['s7_first_filter'] = int(genes[123] * 30 + 40)       # 40-70
        params['s7_second_filter'] = int(genes[124] * 20 + 25)      # 25-45
        params['s7_third_filter'] = int(genes[125] * 8 + 8)         # 8-16
        params['s7_ratio_diff_period'] = int(genes[126] * 6 + 6)    # 6-12
        params['s7_use_interest'] = genes[127] > 0.5                # True/False
        params['s7_use_grow'] = genes[128] > 0.5                    # True/False
        params['s7_grow_threshold'] = genes[129] * 30 - 10          # -10 to 20
        params['s7_interest_threshold'] = genes[130] * 4 + 2        # 2-6%
        params['s7_ratio_weight'] = genes[131] * 0.5 + 0.3          # 0.3-0.8
        params['s7_rev_weight'] = genes[132] * 0.5 + 0.3            # 0.3-0.8
        params['s7_mom_weight'] = genes[133] * 0.5 + 0.2            # 0.2-0.7
        params['s7_min_volume'] = genes[134] * 200000 + 100000      # 10萬-30萬
        params['s7_resample'] = 'M' if genes[135] > 0.5 else None   # M or None
        params['s7_top_n'] = int(genes[136] * 5 + 3)                # 3-8

        # =========================================================
        # 策略8: 純技術趨勢 (137-156) - 20個參數
        # =========================================================
        params['s8_bias_sma_period'] = int(genes[137] * 100 + 200)  # 200-300
        params['s8_bias_sma_threshold'] = genes[138] * 0.15 + 0.05  # 0.05-0.20
        params['s8_zlma_period'] = int(genes[139] * 60 + 90)        # 90-150
        params['s8_bias_zlma_threshold'] = genes[140] * 0.15 + 0.05 # 0.05-0.20
        params['s8_wma_period'] = int(genes[141] * 40 + 40)         # 40-80
        params['s8_bias_wma_threshold'] = genes[142] * 0.15 + 0.05  # 0.05-0.20
        params['s8_slope_period'] = int(genes[143] * 40 + 40)       # 40-80
        params['s8_slope_percentile'] = genes[144] * 0.3 + 0.4      # 0.4-0.7
        params['s8_kurtosis_period'] = int(genes[145] * 100 + 150)  # 150-250
        params['s8_kurtosis_percentile'] = genes[146] * 0.3 + 0.3   # 0.3-0.6
        params['s8_skew_period'] = int(genes[147] * 80 + 120)       # 120-200
        params['s8_skew_threshold'] = genes[148] * 1.0 - 0.5        # -0.5 to 0.5
        params['s8_min_volume'] = genes[149] * 200000 + 150000      # 15萬-35萬
        params['s8_vol_ma_period'] = int(genes[150] * 10 + 5)       # 5-15
        params['s8_use_smallest'] = genes[151] > 0.5                # True/False
        params['s8_ma_confirm'] = genes[152] > 0.5                  # True/False
        params['s8_ma_short'] = int(genes[153] * 20 + 10)           # 10-30
        params['s8_ma_long'] = int(genes[154] * 40 + 40)            # 40-80
        params['s8_resample'] = 'Q' if genes[155] > 0.5 else None   # Q or None
        params['s8_top_n'] = int(genes[156] * 10 + 8)               # 8-18

        # =========================================================
        # 策略9: 財報指標20大 (157-173) - 17個參數
        # =========================================================
        params['s9_std_window'] = int(genes[157] * 40 + 40)         # 40-80
        params['s9_ma_period'] = int(genes[158] * 40 + 40)          # 40-80
        params['s9_std_percentile'] = genes[159] * 0.3 + 0.3        # 0.3-0.6
        params['s9_min_volume'] = genes[160] * 200000 + 150000      # 15萬-35萬
        params['s9_cond1_weight'] = genes[161] * 8 + 2              # 2-10
        params['s9_cond2_weight'] = genes[162] * 8 + 2              # 2-10
        params['s9_cond3_weight'] = genes[163] * 12 + 5             # 5-17
        params['s9_use_roe'] = genes[164] > 0.5                     # True/False
        params['s9_use_npm'] = genes[165] > 0.5                     # True/False
        params['s9_use_growth'] = genes[166] > 0.5                  # True/False
        params['s9_use_liquidity'] = genes[167] > 0.5               # True/False
        params['s9_use_leverage'] = genes[168] > 0.5                # True/False
        params['s9_feature_count'] = int(genes[169] * 10 + 10)      # 10-20
        params['s9_deadline_shift'] = int(genes[170] * 3)           # 0-3
        params['s9_rank_method'] = 'pct' if genes[171] > 0.5 else 'dense'
        params['s9_resample'] = 'W' if genes[172] < 0.5 else 'M'    # W or M
        params['s9_top_n'] = int(genes[173] * 12 + 10)              # 10-22

        # =========================================================
        # 回測與風控參數 (174-179) - 6個參數
        # =========================================================
        params['stop_loss'] = genes[174] * 0.15 + 0.15              # 15%-30%
        params['trail_stop'] = genes[175] * 0.20 + 0.20             # 20%-40%
        params['take_profit'] = genes[176] * 0.40 + 0.50            # 50%-90%
        params['position_limit'] = genes[177] * 0.15 + 0.25         # 25%-40%
        params['min_volume'] = genes[178] * 200000 + 100000         # 10萬-30萬
        params['liquidity_days'] = int(genes[179] * 15 + 10)        # 10-25天

        return params

gene_decoder = GeneDecoder()

# =============================================================================
# 第七部分：回測引擎
# =============================================================================
def run_backtest(position, params: Dict, name: str = "九組合媽媽龍", full_report: bool = False) -> Dict:
    """
    執行回測

    Args:
        position: 持股部位
        params: 策略參數
        name: 策略名稱
        full_report: 是否輸出完整報告
    """
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
            position_limit=MIN_POSITION_RATIO,  # 每股占比至少 3%
            stop_loss=params.get('stop_loss', 0.25),
            stop_trading_next_period=False,
            upload=False,
            name=name,
        )

        metrics = report.get_metrics()

        result = {
            'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
            'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
            'capacity': metrics['liquidity'].get('capacity', 0) or 0,
            'win_rate': metrics['profitability'].get('winRate', 0) or 0,
            'report': report,
        }

        # 檢查是否達標
        result['meets_sharpe'] = result['sharpe'] >= TARGET_SHARPE
        result['meets_capacity'] = result['capacity'] >= MIN_CAPACITY
        result['meets_drawdown'] = result['max_drawdown'] <= MAX_DRAWDOWN
        result['meets_all'] = result['meets_sharpe'] and result['meets_capacity'] and result['meets_drawdown']

        if full_report:
            print(f"\n{'='*60}")
            print(f"📊 完整回測報告 - {name}")
            print(f"{'='*60}")
            print(f"   夏普值: {result['sharpe']:.2f} {'✅' if result['meets_sharpe'] else '❌'} (目標 >= {TARGET_SHARPE})")
            print(f"   胃納量: {result['capacity']/1e4:.0f}萬 {'✅' if result['meets_capacity'] else '❌'} (目標 >= {MIN_CAPACITY/1e4:.0f}萬)")
            print(f"   最大回撤: {result['max_drawdown']*100:.1f}% {'✅' if result['meets_drawdown'] else '❌'} (目標 <= {MAX_DRAWDOWN*100:.0f}%)")
            print(f"   年化報酬: {result['annual_return']*100:.1f}%")
            print(f"   勝率: {result['win_rate']*100:.1f}%")
            if result['meets_all']:
                print(f"   🎉 全部達標！")

        return result
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
# 第十部分：DEAP NSGA-II 基因演算法設定
# =============================================================================

# 清除舊定義（避免重複執行時出錯）
if 'FitnessMulti' in dir(creator):
    del creator.FitnessMulti
if 'Individual' in dir(creator):
    del creator.Individual

# 創建適應度類別（4目標：綜合分數、夏普值、胃納量、穩健性）
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=GeneDecoder.GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

print("✅ DEAP NSGA-II 配置完成")

# =============================================================================
# 第十一部分：適應度評估函數
# =============================================================================
def evaluate_fitness(individual: List[float]) -> Tuple[float, float, float, float]:
    """
    評估個體適應度 (多目標)

    Returns: (composite_score, sharpe_score, capacity_score, robustness_score)
    """
    try:
        # 解碼基因
        params = gene_decoder.decode(individual)

        # 組合策略
        position = strategy_engine.combine_all_strategies(params)

        if position is None or position.empty:
            return (0.0, 0.0, 0.0, -10.0)

        # 執行回測
        result = run_backtest(position, params, "GA評估")

        sharpe = result.get('sharpe', 0)
        capacity = result.get('capacity', 0)
        annual_return = result.get('annual_return', 0)
        max_drawdown = result.get('max_drawdown', 1)

        # 夏普值分數（0-5）
        sharpe_score = min(5.0, max(0, sharpe / TARGET_SHARPE * 5.0))

        # 胃納量分數（0-5）
        capacity_score = min(5.0, max(0, capacity / MIN_CAPACITY * 5.0))

        # 年化報酬分數（0-2）
        return_score = min(2.0, max(0, annual_return / TARGET_ANNUAL_RETURN * 2.0))

        # 回撤懲罰
        drawdown_penalty = 0 if max_drawdown <= MAX_DRAWDOWN else -(max_drawdown - MAX_DRAWDOWN) * 5

        # 穩健性分數
        robustness_score = return_score + drawdown_penalty

        # 綜合分數
        composite = sharpe_score * 0.4 + capacity_score * 0.3 + return_score * 0.2 + max(0, robustness_score) * 0.1

        return (composite, sharpe_score, capacity_score, robustness_score)

    except Exception as e:
        print(f"   ⚠️ 評估失敗: {e}")
        return (0.0, 0.0, 0.0, -10.0)

toolbox.register("evaluate", evaluate_fitness)

# =============================================================================
# 第十二部分：Pareto Archive 持續演進管理器
# =============================================================================
class ParetoArchiveManager:
    """Pareto 前緣歷史管理器 - 支援持續演進"""

    def __init__(self, archive_file: str = None):
        self.archive_file = archive_file or f"{BASE_DIR}/pareto_archive_9combo.pkl"
        self.archive = []
        self._load()

    def _load(self):
        """載入歷史精英"""
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
            print("ℹ️ 無歷史存檔，從頭開始演化")

    def update(self, pareto_front: List):
        """更新 Pareto 前緣"""
        # 合併新舊個體
        all_individuals = self.archive + [
            {'genes': list(ind), 'fitness': ind.fitness.values, 'timestamp': datetime.now().isoformat()}
            for ind in pareto_front
        ]

        # 去重
        unique = self._deduplicate(all_individuals)

        # 保留前 50 名
        unique.sort(key=lambda x: sum(x['fitness']), reverse=True)
        self.archive = unique[:50]

        # 保存
        self._save()

        return len(self.archive)

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
        """保存到檔案"""
        try:
            Path(os.path.dirname(self.archive_file)).mkdir(parents=True, exist_ok=True)
            with open(self.archive_file, 'wb') as f:
                pickle.dump({
                    'individuals': self.archive,
                    'timestamp': datetime.now().isoformat(),
                    'version': '9combo_mama_dragon_v1.0',
                }, f)
            print(f"   💾 已保存 {len(self.archive)} 個精英到 Pareto Archive")
        except Exception as e:
            print(f"⚠️ 保存失敗: {e}")

    def inject_elites(self, population: List, ratio: float = 0.3) -> List:
        """將歷史精英注入族群"""
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

        print(f"   🧬 注入 {n_inject} 個歷史精英")
        return population

    def get_best(self) -> Optional[Dict]:
        """獲取最佳個體"""
        if not self.archive:
            return None
        return max(self.archive, key=lambda x: sum(x['fitness']))

# =============================================================================
# 第十三部分：演化引擎
# =============================================================================
class EvolutionEngine:
    """基因演算法演化引擎 - 支援持續演進"""

    def __init__(self, pareto_mgr: ParetoArchiveManager):
        self.pareto_mgr = pareto_mgr
        self.history = []
        self.best_ever = None
        self.checkpoint_file = f"{BASE_DIR}/ga_checkpoint_9combo.pkl"

    def run(self, n_generations: int = N_GENERATIONS, resume: bool = True) -> Tuple[List, List]:
        """
        執行演化

        Args:
            n_generations: 演化世代數
            resume: 是否從斷點續傳
        """
        print(f"\n{'='*70}")
        print(f"🧬 開始基因演算法優化")
        print(f"   族群大小: {POPULATION_SIZE}")
        print(f"   演化世代: {n_generations}")
        print(f"   目標夏普: >= {TARGET_SHARPE}")
        print(f"   目標胃納量: >= {MIN_CAPACITY/1e4:.0f}萬")
        print(f"{'='*70}\n")

        start_gen = 0

        # 嘗試從斷點恢復
        if resume and os.path.exists(self.checkpoint_file):
            try:
                checkpoint = self._load_checkpoint()
                if checkpoint:
                    population = checkpoint['population']
                    start_gen = checkpoint['generation'] + 1
                    self.history = checkpoint.get('history', [])
                    print(f"✅ 從第 {start_gen} 代恢復演化")
            except:
                population = None
        else:
            population = None

        # 初始化族群
        if population is None:
            population = toolbox.population(n=POPULATION_SIZE)
            # 注入歷史精英
            population = self.pareto_mgr.inject_elites(population, ratio=0.3)

        # 初始評估
        population = self._evaluate_population(population)

        # 演化循環
        for gen in range(start_gen, n_generations):
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
            offspring = self._evaluate_population(offspring)

            # 環境選擇 (NSGA-II)
            population = toolbox.select(population + offspring, POPULATION_SIZE)

            # 統計
            stats = self._compute_stats(population, gen)
            self.history.append(stats)

            elapsed = time.time() - start_time

            # 輸出進度
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"\n=== 第 {gen+1}/{n_generations} 代 ===")
                print(f"   最佳綜合: {stats['best_composite']:.4f}")
                print(f"   最佳夏普: {stats['best_sharpe']:.2f}")
                print(f"   最佳胃納量: {stats['best_capacity']:.0f} 萬")
                print(f"   耗時: {elapsed:.1f}s")

            # 🔥 每 5 代進行完整回測
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0:
                self._run_full_backtest(population, gen)

            # 🔥 每 5 代保存斷點（Checkpoint）
            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(population, gen)
                # 同時更新 Pareto Archive
                current_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                self.pareto_mgr.update(current_front)

        # 取得 Pareto 前緣
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 更新 Pareto Archive
        n_saved = self.pareto_mgr.update(pareto_front)
        print(f"\n✅ 演化完成！保存了 {n_saved} 個 Pareto 最優解")

        # 輸出最佳結果
        self._print_best_results(pareto_front)

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

            if (i + 1) % 10 == 0:
                print(f"   進度: {i+1}/{len(invalid)}")

        return population

    def _compute_stats(self, population: List, gen: int) -> Dict:
        """計算統計"""
        fitnesses = [ind.fitness.values for ind in population]

        composites = [f[0] for f in fitnesses]
        sharpes = [f[1] / 5.0 * TARGET_SHARPE for f in fitnesses]
        capacities = [f[2] / 5.0 * MIN_CAPACITY / 1e4 for f in fitnesses]

        stats = {
            'generation': gen,
            'best_composite': max(composites),
            'avg_composite': np.mean(composites),
            'best_sharpe': max(sharpes),
            'avg_sharpe': np.mean(sharpes),
            'best_capacity': max(capacities),
            'timestamp': datetime.now().isoformat(),
        }

        return stats

    def _save_checkpoint(self, population: List, gen: int):
        """保存斷點"""
        try:
            Path(os.path.dirname(self.checkpoint_file)).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'population': population,
                'generation': gen,
                'history': self.history,
                'timestamp': datetime.now().isoformat(),
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"   💾 斷點已保存 (第 {gen+1} 代)")
        except Exception as e:
            print(f"   ⚠️ 斷點保存失敗: {e}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """載入斷點"""
        try:
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def _run_full_backtest(self, population: List, gen: int):
        """
        每 N 代進行完整回測
        找出當前最佳個體並執行詳細回測
        """
        print(f"\n{'='*60}")
        print(f"🔬 第 {gen+1} 代 - 完整回測檢查")
        print(f"{'='*60}")

        # 找出最佳個體
        best_ind = max(population, key=lambda x: sum(x.fitness.values))
        params = gene_decoder.decode(best_ind)

        # 執行完整回測
        position = strategy_engine.combine_all_strategies(params)
        result = run_backtest(position, params, f"第{gen+1}代最佳", full_report=True)

        # 記錄是否達標
        if result.get('meets_all', False):
            print(f"\n🎉 第 {gen+1} 代找到達標解！")
            # 立即保存達標解
            self._save_qualified_solution(best_ind, result, gen)

        return result

    def _save_qualified_solution(self, individual, result: Dict, gen: int):
        """保存達標解"""
        try:
            qualified_file = f"{BASE_DIR}/qualified_solutions.json"
            solutions = []

            if os.path.exists(qualified_file):
                with open(qualified_file, 'r', encoding='utf-8') as f:
                    solutions = json.load(f)

            solution = {
                'generation': gen + 1,
                'genes': list(individual),
                'sharpe': result['sharpe'],
                'capacity': result['capacity'],
                'max_drawdown': result['max_drawdown'],
                'annual_return': result['annual_return'],
                'timestamp': datetime.now().isoformat(),
            }
            solutions.append(solution)

            Path(os.path.dirname(qualified_file)).mkdir(parents=True, exist_ok=True)
            with open(qualified_file, 'w', encoding='utf-8') as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)

            print(f"   💾 達標解已保存！(共 {len(solutions)} 個)")
        except Exception as e:
            print(f"   ⚠️ 保存達標解失敗: {e}")

    def _print_best_results(self, pareto_front: List):
        """輸出最佳結果"""
        print(f"\n{'='*70}")
        print(f"🏆 Pareto 最優解集（前 10 名）")
        print(f"{'='*70}")
        print(f"{'排名':<6}{'綜合':<10}{'夏普':<10}{'胃納量(萬)':<12}{'達標'}")
        print("-" * 70)

        sorted_front = sorted(pareto_front, key=lambda x: sum(x.fitness.values), reverse=True)[:10]

        for i, ind in enumerate(sorted_front, 1):
            composite = ind.fitness.values[0]
            sharpe = ind.fitness.values[1] / 5.0 * TARGET_SHARPE
            capacity = ind.fitness.values[2] / 5.0 * MIN_CAPACITY / 1e4

            reached = "✅" if sharpe >= TARGET_SHARPE * 0.9 and capacity >= MIN_CAPACITY / 1e4 * 0.9 else ""

            print(f"{i:<6}{composite:<10.4f}{sharpe:<10.2f}{capacity:<12.2f}{reached}")

# =============================================================================
# 第十四部分：完整回測最佳個體
# =============================================================================
def backtest_best_individual(pareto_mgr: ParetoArchiveManager):
    """對最佳個體進行完整回測"""
    best = pareto_mgr.get_best()

    if not best:
        print("⚠️ 無歷史最佳個體")
        return None

    print(f"\n{'='*70}")
    print("📊 最佳個體完整回測")
    print(f"{'='*70}")

    params = gene_decoder.decode(best['genes'])

    # 顯示策略權重
    print("\n🎯 策略權重配置：")
    strategy_names = ['低波動本益比', '小資族', '營收雙渦輪', '高殖利率烏龜',
                      '低波動指標', '大盤指針', '小蝦米大鯨魚', '純技術趨勢', '財報20大']
    for i, name in enumerate(strategy_names):
        weight = params.get(f'weight_{i+1}', 0)
        bar = '█' * int(weight * 50)
        print(f"   {name:<12}: {weight*100:5.1f}% {bar}")

    # 執行回測
    position = strategy_engine.combine_all_strategies(params)

    if position is not None and not position.empty:
        print("\n📈 執行完整回測...")

        report = sim(
            position=position,
            fee_ratio=1.425 / 1000,
            tax_ratio=3 / 1000,
            trade_at_price="high_low_avg",
            position_limit=params.get('position_limit', 0.35),
            stop_loss=params.get('stop_loss', 0.25),
            stop_trading_next_period=False,
            upload=False,
            name='九組合媽媽龍_最佳化',
            live_performance_start='2022-01-01',
        )

        report.display()

        # 保存最佳參數
        params_file = f"{BASE_DIR}/best_params_9combo.json"
        try:
            Path(os.path.dirname(params_file)).mkdir(parents=True, exist_ok=True)
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 最佳參數已保存: {params_file}")
        except:
            pass

        return report

    return None

# =============================================================================
# 第十五部分：主程式入口
# =============================================================================
def main():
    """主程式入口"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           🐉 九組合媽媽龍策略 - 基因演算法優化系統                    ║
    ║           Nine Combo Mama Dragon - GA Optimization                  ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  🧬 NSGA-II 多目標優化                                               ║
    ║  📊 Pareto Archive 持續演進                                          ║
    ║  💾 斷點續傳支援                                                     ║
    ║  🎯 目標：夏普 >= 4.0, 胃納量 >= 1000萬                              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 選擇執行模式
    print("\n請選擇執行模式：")
    print("  1. 策略分析（快速）")
    print("  2. 基因演算法優化（完整）")
    print("  3. 回測最佳個體")
    print("  4. Live Performance 回測")

    # 預設執行完整優化
    mode = 2

    if mode == 1:
        # 策略分析
        results, combined = analyze_strategies()

    elif mode == 2:
        # 基因演算法優化
        pareto_mgr = ParetoArchiveManager()
        engine = EvolutionEngine(pareto_mgr)

        # 執行演化（支援斷點續傳）
        population, pareto_front = engine.run(n_generations=N_GENERATIONS, resume=True)

        # 回測最佳個體
        backtest_best_individual(pareto_mgr)

    elif mode == 3:
        # 回測最佳個體
        pareto_mgr = ParetoArchiveManager()
        backtest_best_individual(pareto_mgr)

    elif mode == 4:
        # Live Performance 回測
        run_live_backtest()

    print("\n✅ 九組合媽媽龍策略系統執行完成！")

if __name__ == "__main__":
    main()
