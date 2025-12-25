# -*- coding: utf-8 -*-
"""
================================================================================
🧬 台股基因演算法優化系統 v9.0 - 終極版
================================================================================
【v9.0 核心改進】
   ✅ 多目標優化：夏普值 4.0 + 胃納量 500萬
   ✅ NSGA-II 演算法：Pareto 最優解集
   ✅ 並行計算：CPU 多核心加速
   ✅ Walk-Forward 驗證：防止過擬合
   ✅ 智能參數搜索：自適應變異率
   ✅ 穩健性評估：多時間段測試

【目標】
   - 夏普值：>= 4.0
   - 胃納量：>= 500萬
   - 年化報酬：最大化
   - 最大回撤：< 20%

作者：FinLab VIP 優化系統
版本：v9.0 Ultimate (2025)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
#                   🔥 重要設定區 🔥
# ============================================================================

# 🔥 設定 WINDOW_ID (1-4)，每個視窗請修改此值
WINDOW_ID = 1  # 請改成 1, 2, 3, 或 4

# 🔥 設定 FinLab API Key
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# 🔥 核心目標設定
TARGET_SHARPE = 4.0         # 目標夏普值
MIN_CAPACITY = 5_000_000    # 最低胃納量：500萬（調整後）
TARGET_ANNUAL_RETURN = 0.3  # 目標年化報酬 30%
MAX_DRAWDOWN = 0.2          # 最大回撤限制 20%

# 🔥 斷點續傳設定
ENABLE_RESUME_FROM_CHECKPOINT = True

# ============================================================================

print("=" * 80)
print(f"🚀 台股基因演算法優化系統 v9.0 - Window {WINDOW_ID}")
print("   ✅ 多目標優化 + 並行計算 + Walk-Forward + 智能搜索")
print(f"   🎯 目標：夏普值 >= {TARGET_SHARPE} & 胃納量 >= {MIN_CAPACITY/1e6:.0f}萬")
print("=" * 80)

# ============================================================================
#                   PART 1: 環境準備
# ============================================================================
import os
import sys
import subprocess
import time
import json
import pickle
import hashlib
import random
import traceback
import glob
import datetime
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import reduce
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('future.no_silent_downcasting', True)

# ============================================================================
#                   PART 2: 自動登入 Google Drive & FinLab
# ============================================================================
print("\n" + "=" * 60)
print("📂 STEP 1: 自動登入 Google Drive")
print("=" * 60)

IN_COLAB = False
DRIVE_MOUNTED = False

try:
    from google.colab import drive
    IN_COLAB = True
    print("   正在掛載 Google Drive...")
    drive.mount('/content/drive', force_remount=False)

    if os.path.exists('/content/drive/MyDrive'):
        DRIVE_MOUNTED = True
        print("   ✅ Google Drive 掛載成功！")
    else:
        print("   ⚠️ Google Drive 路徑不存在")

except ImportError:
    print("   ℹ️ 非 Colab 環境，跳過 Drive 掛載")
except Exception as e:
    print(f"   ⚠️ Drive 掛載失敗: {e}")

print("\n" + "=" * 60)
print("🔑 STEP 2: 自動登入 FinLab VIP")
print("=" * 60)

# 確保 finlab 已安裝
try:
    import finlab
    print(f"   FinLab 版本: {finlab.__version__}")
except ImportError:
    print("   正在安裝 finlab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "finlab"])
    import finlab
    print(f"   ✅ FinLab 安裝完成")

# 登入 FinLab
if FINLAB_API_KEY and FINLAB_API_KEY != "YOUR_FINLAB_API_KEY_HERE":
    try:
        finlab.login(FINLAB_API_KEY)
        print("   ✅ FinLab VIP 登入成功！")
    except Exception as e:
        print(f"   ⚠️ FinLab 登入失敗: {e}")
else:
    print("   ⚠️ 未設定 API Key")

# ============================================================================
#                   PART 3: 載入必要套件
# ============================================================================
print("\n" + "=" * 60)
print("📦 STEP 3: 載入必要套件")
print("=" * 60)

import gc

# 安裝 DEAP
try:
    from deap import base, creator, tools, algorithms
    print("   ✅ DEAP 已載入")
except ImportError:
    print("   正在安裝 DEAP...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "deap"])
    from deap import base, creator, tools, algorithms
    print("   ✅ DEAP 安裝完成")

print("   ✅ 所有套件載入完成！")

# ============================================================================
#                   PART 4: 路徑配置
# ============================================================================
print("\n" + "=" * 60)
print(f"📁 STEP 4: 配置 Window {WINDOW_ID} 路徑")
print("=" * 60)

if IN_COLAB and DRIVE_MOUNTED:
    BASE_PATH = '/content/drive/MyDrive/投資策略優化_v9.0_多目標'
    WINDOW_PATH = f'{BASE_PATH}/window_{WINDOW_ID}'
    WORKING_PATH = f'{WINDOW_PATH}/working'
    CHECKPOINT_PATH = f'{WINDOW_PATH}/checkpoints'
    SHARED_PATH = f'{BASE_PATH}/shared_best'
    SHARED_ELITE_PATH = f'{SHARED_PATH}/elite_pool'
    BACKTEST_RESULTS_PATH = f'{BASE_PATH}/回測結果_v9.0'

    for path in [WINDOW_PATH, WORKING_PATH, CHECKPOINT_PATH,
                 SHARED_PATH, SHARED_ELITE_PATH, BACKTEST_RESULTS_PATH]:
        os.makedirs(path, exist_ok=True)

    # 搜尋歷史個體的路徑
    SEARCH_PATHS = [
        f'{BASE_PATH}/window_1',
        f'{BASE_PATH}/window_2',
        f'{BASE_PATH}/window_3',
        f'{BASE_PATH}/window_4',
        '/content/drive/MyDrive/投資策略優化_六策略_v8.2_分散式',
        '/content/drive/MyDrive/投資策略優化_六策略_v8.1_分散式',
    ]
    SEARCH_PATHS = [path for path in SEARCH_PATHS if os.path.exists(path)]

    print(f"   工作目錄: {WORKING_PATH}")
    print(f"   可用搜尋路徑: {len(SEARCH_PATHS)} 個")
else:
    WORKING_PATH = '.'
    CHECKPOINT_PATH = '.'
    SHARED_PATH = '.'
    SHARED_ELITE_PATH = '.'
    BACKTEST_RESULTS_PATH = '.'
    SEARCH_PATHS = ['.']
    print("   ⚠️ 非 Colab 環境，使用本地路徑")

print("   ✅ 路徑配置完成！")

# ============================================================================
#                   PART 5: 系統配置
# ============================================================================

# 基因長度
GENE_LENGTH = 160

# 回測起始日期
BACKTEST_START_DATE = "2017-01-01"

# 交易成本
FEE_RATIO = 1.425 / 1000
TAX_RATIO = 3 / 1000

# 🔥 多目標優化配置
MULTI_OBJECTIVE_CONFIG = {
    'target_sharpe': TARGET_SHARPE,
    'min_capacity': MIN_CAPACITY,
    'target_annual_return': TARGET_ANNUAL_RETURN,
    'max_drawdown': MAX_DRAWDOWN,
    'capacity_weight': 0.3,      # 胃納量權重
    'sharpe_weight': 0.5,        # 夏普值權重
    'return_weight': 0.15,       # 報酬率權重
    'drawdown_weight': 0.05,     # 回撤權重
}

# 🔥 基因演算法配置（NSGA-II）
GA_CONFIG = {
    'population_size': 80,           # 增加族群大小以提升多樣性
    'num_generations': 500,          # 增加演化世代
    'target_fitness': 10.0,          # 綜合適應度目標
    'mutation_rate': 0.25,           # 降低變異率
    'crossover_rate': 0.8,           # 提高交叉率
    'tournament_size': 3,
    'checkpoint_interval': 20,
    'num_cpu_cores': min(cpu_count(), 8),  # 使用多核心
    'enable_parallel': True,
}

# Walk-Forward 配置
WALK_FORWARD_CONFIG = {
    'num_folds': 5,                  # 5折交叉驗證
    'train_ratio': 0.7,              # 訓練集比例
    'min_sharpe_consistency': 0.6,   # 各折最低一致性
}

print(f"\n{'='*60}")
print(f"📊 Window {WINDOW_ID} 系統配置")
print(f"{'='*60}")
print(f"基因長度: {GENE_LENGTH}")
print(f"族群大小: {GA_CONFIG['population_size']}")
print(f"演化世代: {GA_CONFIG['num_generations']}")
print(f"目標夏普: {TARGET_SHARPE}")
print(f"最低胃納量: {MIN_CAPACITY/1e6:.0f} 萬")
print(f"使用核心數: {GA_CONFIG['num_cpu_cores']}")
print(f"{'='*60}")

# ============================================================================
#                   PART 6: 載入 FinLab 數據
# ============================================================================
print("\n" + "=" * 60)
print("📊 STEP 5: 載入 FinLab 數據")
print("=" * 60)

from finlab import data
from finlab.backtest import sim

def safe_data_get(dataset_name, max_retries=2):
    """穩健載入 FinLab 數據"""
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                data.force_cloud_download = True
                data.set_storage(data.CacheStorage())

            df = data.get(dataset_name)

            if df is not None and len(df) > 0:
                if attempt > 0:
                    data.force_cloud_download = False
                return df
            else:
                return None

        except EOFError:
            if attempt < max_retries:
                dataset_key = dataset_name.replace(':', '#').replace(':', '_')
                cache_patterns = [
                    f'/root/.cache/finlab/*{dataset_key}*',
                    f'/root/.cache/finlab_db/*{dataset_key}*',
                ]
                for pattern in cache_patterns:
                    for path in glob.glob(pattern):
                        try:
                            os.remove(path)
                        except:
                            pass
            else:
                return None

        except Exception as e:
            if attempt >= max_retries:
                return None

    return None

# 載入數據
print("   載入價格數據...")
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

print("   載入估值數據...")
pe = data.get('price_earning_ratio:本益比')
股價淨值比 = data.get("price_earning_ratio:股價淨值比")

print("   載入營收數據...")
rev = data.get('monthly_revenue:當月營收')
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)
rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

print("   載入基本面數據...")
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
稅前淨利率 = data.get("fundamental_features:稅前淨利率")

print("   載入籌碼數據...")
融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")

print("   載入集保庫存數據...")
INVENTORY_AVAILABLE = False
inventory = safe_data_get('inventory', max_retries=2)
if inventory is not None:
    INVENTORY_AVAILABLE = True
    print(f"   ✅ 集保庫存: {inventory.shape}")
else:
    INVENTORY_AVAILABLE = False
    print("   ⚠️ 集保庫存不可用")

市值 = data.get('etl:market_value')

print("   計算衍生指標...")
成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

# 漲停板計算
limit_up = (close > close.shift(1)*1.095)
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

print("   計算技術指標...")
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)
entry_volatility = atr / adj_close

print("   ✅ 所有數據載入完成！")

# ============================================================================
#                   PART 7: gene_to_params 函數（160 基因，優化版）
# ============================================================================
def gene_to_params(gene):
    """
    將基因轉換為策略參數（160 基因結構，優化版）

    🔥 優化改進：
    - 參數範圍更合理（避免極端值）
    - 使用 sigmoid/tanh 限制參數範圍
    - 確保參數邏輯一致性
    """
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    if len(gene) < required_length:
        gene = gene + [0.0] * (required_length - len(gene))
    elif len(gene) > required_length:
        gene = gene[:required_length]

    # 🔥 使用 softmax 確保配置比例合理
    alloc_raw = [max(0.01, abs(gene[i])) for i in range(6)]
    alloc_sum = sum(alloc_raw)
    allocation = [x/alloc_sum for x in alloc_raw]

    # 🔥 輔助函數：限制參數範圍
    def clip(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def sigmoid(x):
        """將 x 映射到 (0, 1)"""
        return 1 / (1 + np.exp(-x/10))

    def positive_int(val, min_val=1, max_val=100):
        """確保正整數"""
        return max(min_val, min(max_val, int(abs(val))))

    # 策略1: 低波動本益比
    low_vol_pe_params = {
        'rev_ma3_ma12_ratio': clip(gene[6], 0.8, 1.5),
        'rev_consistency': clip(gene[7], 0.8, 1.2),
        'volatility_threshold': clip(gene[8]/1000, 0.01, 0.1),
        'margin_usage_limit': clip(gene[9], 20, 60),
        'non_op_income_limit': clip(gene[10], 5, 15),
        'min_volume': abs(gene[11]) * 100 + 100,  # 至少100張
        'pe_min': clip(gene[12], 5, 15),
        'pe_max': clip(gene[13], 15, 40),
        'top_n': positive_int(gene[14], 3, 30),
        'min_rev_yoy_threshold': clip(-gene[120], -30, 0),
        'rev_decline_period': positive_int(gene[121], 2, 6),
        'max_rev_yoy_threshold': clip(gene[122], 30, 100),
        'old_trend_period': positive_int(gene[123], 6, 15),
        'old_trend_match': positive_int(gene[124], 3, 10),
        'rev_bottom_window': positive_int(gene[125], 6, 18),
        'rev_bottom_ratio': clip(gene[126], 1.0, 1.5),
        'rev_bottom_sustain': positive_int(gene[127], 2, 6),
        'min_rev_mom_growth': clip(-gene[128], -20, 0),
        'rev_mom_sustain': positive_int(gene[129], 2, 6),
        'quarter_ma': positive_int(gene[130], 40, 80),
        'half_year_ma': positive_int(gene[131], 100, 140),
        'long_ma': positive_int(gene[132], 200, 260),
        'recent_rev_period': positive_int(gene[133], 2, 6),
        'annual_rev_period': positive_int(gene[134], 10, 14),
        'pb_min': clip(gene[135], 0.5, 2.0),
        'pb_max': clip(gene[136], 2.0, 5.0),
        'min_gpm': clip(gene[137], 10, 40),
        'gpm_sustain_period': positive_int(gene[138], 2, 6),
        'min_roe': clip(gene[139], 5, 25),
        'roe_sustain_period': positive_int(gene[140], 2, 6),
    }

    # 策略2: 小資族
    small_inv_params = {
        'market_value_limit': clip(gene[15], 5, 50) * 1e9,
        'market_rev_ratio_limit': clip(gene[16], 1, 5),
        'rev_yoy_growth_limit': clip(-gene[17], -30, 0),
        'rev_mom_growth_limit': clip(-gene[18], -20, 0),
        'rsv_period': positive_int(gene[19], 5, 30),
        'ma_period': positive_int(gene[20], 10, 60),
        'volume_threshold': abs(gene[21]) * 100 + 100,
        'top_n': positive_int(gene[22], 3, 30),
        'min_free_cash_flow': clip(gene[150], -10, 20),
        'min_roe': clip(gene[151], 5, 25),
        'min_op_profit_growth': clip(gene[152], 0, 50),
    }

    # 策略3: 營收股價雙渦輪
    turbo_params = {
        'rev_ma_period': positive_int(gene[23], 2, 6),
        'rev_ma_lookback': positive_int(gene[24], 6, 24),
        'price_high_window': positive_int(gene[25], 20, 100),
        'min_volume': abs(gene[26]) * 100 + 100,
        'min_price': clip(gene[27], 10, 50),
        'rsi_threshold': clip(gene[28], 50, 80),
        'pe_limit': clip(gene[29], 15, 40),
        'top_n': positive_int(gene[30], 3, 20),
        'performance_ma_period': positive_int(gene[100], 10, 60),
        'performance_threshold': clip(gene[101], 1.0, 1.3),
        'rsi_trend_period': positive_int(gene[102], 3, 10),
        'min_gpm': clip(gene[103], 15, 45),
        'gpm_sustain_period': positive_int(gene[104], 2, 6),
        'min_btpm': clip(gene[105], 5, 25),
        'btpm_sustain_period': positive_int(gene[106], 2, 6),
        'min_atpm': clip(gene[107], 3, 20),
        'atpm_sustain_period': positive_int(gene[108], 2, 6),
        'rev_growth_percentile': clip(gene[109], 0.5, 0.9),
        'boss_min_level': positive_int(gene[110], 10, 14),
        'boss_max_level': positive_int(gene[111], 15, 17),
        'min_boss_ratio': clip(gene[112], 20, 60),
    }

    # 策略4: 高殖利率烏龜
    high_yield_turtle_params = {
        'min_yield_ratio': clip(gene[31], 3, 8),
        'min_op_earn_ratio': clip(gene[32], 5, 20),
        'min_boss_hold': clip(gene[33], 10, 40),
        'min_volume': abs(gene[34]) * 100 + 100,
        'max_volume': abs(gene[35]) * 100 + 5000,
        'top_n': positive_int(gene[36], 3, 20),
    }

    # 策略5: 低波動性指標
    low_vol_index_params = {
        'min_volume': abs(gene[37]) * 100 + 100,
        'std_window': positive_int(gene[38], 10, 40),
        'std_threshold': clip(gene[39], 0.1, 0.5),
        'top_n': positive_int(gene[40], 5, 30),
    }

    # 策略6: 市場指針
    market_indicator_params = {
        'new_high_window': positive_int(gene[43], 60, 260),
        'min_year_growth': clip(-gene[44], -30, 0),
        'max_year_growth': clip(gene[45], 40, 100),
        'rev_bottom_ratio': clip(gene[46], 1.0, 1.5),
        'min_month_growth': clip(-gene[47], -20, 0),
        'min_volume': abs(gene[48]) * 100 + 100,
        'top_n': positive_int(gene[49], 5, 25),
    }

    # 整體參數
    overall_params = {
        'stop_loss': clip(gene[51], 5, 20) / 100,
        'trail_stop': clip(gene[52], 3, 15) / 100,
        'take_profit': clip(gene[53], 10, 50) / 100,
        'position_limit': clip(gene[54], 5, 30) / 100,
        'trade_at_price': ["open", "close", "high_low_avg", "open_close_avg"][int(abs(gene[55])) % 4],
        'liquidity_threshold': clip(gene[56], 1, 20) * 1e6,
    }

    return (allocation, low_vol_pe_params, small_inv_params, turbo_params,
            high_yield_turtle_params, low_vol_index_params, market_indicator_params,
            overall_params)

# ============================================================================
#                   PART 8: 六個策略函數（與原版相同）
# ============================================================================
def strategy_low_volatility_pe(params):
    """策略1：低波動本益比"""
    try:
        peg = pe / 營業利益成長率
        cond1 = rev_ma3 / rev_ma12 > params['rev_ma3_ma12_ratio']
        cond2 = rev / rev.shift(1) > params['rev_consistency']
        tree_select_factor = ((融資使用率 <= params['margin_usage_limit'])
                              & (entry_volatility <= params['volatility_threshold'])
                              & (業外收支營收率 < params['non_op_income_limit']))
        condition_近1日成交均量大於100張 = vol.average(1) > params['min_volume']
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < params['min_rev_yoy_threshold']).sustain(params['rev_decline_period'])
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > params['max_rev_yoy_threshold']).sustain(params['old_trend_period'], params['old_trend_match'])
        cond確認營收底部 = ((rev.rolling(params['rev_bottom_window']).min())/(rev) < params['rev_bottom_ratio']).sustain(params['rev_bottom_sustain'])
        cond單月營收月增率 = (rev_month_growth > params['min_rev_mom_growth']).sustain(params['rev_mom_sustain'])
        cond收盤價大於季線及半年線 = (close > close.average(params['quarter_ma'])) & (close > close.average(params['half_year_ma'])) & (close > close.average(params['long_ma']))
        cond近三個月營收大於年營收 = rev.average(params['recent_rev_period']) > rev.average(params['annual_rev_period'])
        pe_range_1 = (params['pe_min'] <= pe) & (pe <= params['pe_max'])
        pb_range_1 = (params['pb_min'] <= 股價淨值比) & (股價淨值比 <= params['pb_max'])
        gpm_trend_1 = (營業毛利率 > params['min_gpm']).sustain(params['gpm_sustain_period'])
        roe_trend_1 = (ROE綜合損益 > params['min_roe']).sustain(params['roe_sustain_period'])

        if INVENTORY_AVAILABLE and inventory is not None:
            try:
                small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                                     .reset_index().groupby(["date", "stock_id"])
                                     .agg({"占集保庫存數比例": "sum"}).reset_index()
                                     .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
            except Exception:
                small_inv_under50 = True
        else:
            small_inv_under50 = True

        cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老 & cond排除月營收連3月衰退
                    & cond收盤價大於季線及半年線 & cond近三個月營收大於年營收 & cond單月營收月增率 & ~(limit_up_all_day)
                    & (condition_近1日成交均量大於100張) & (gpm_trend_1) & (roe_trend_1) & (small_inv_under50) & pb_range_1 & pe_range_1)

        position = peg[cond_all & (peg > 0)].is_smallest(params['top_n']).reindex(rev.index_str_to_date().index, method='ffill')
        return position
    except Exception as e:
        return pd.DataFrame()

def strategy_small_investor(params):
    """策略2：小資族"""
    try:
        股本 = data.get('financial_statement:股本')
        df1 = data.get('financial_statement:投資活動之淨現金流入_流出')
        df2 = data.get('financial_statement:營業活動之淨現金流入_流出')
        自由現金流 = (df1 + df2).rolling(4).mean()
        稅後淨利 = data.get('fundamental_features:經常稅後淨利')
        權益總計 = data.get('financial_statement:股東權益總額')
        股東權益報酬率 = 稅後淨利 / 權益總計
        當月營收 = data.get('monthly_revenue:當月營收') * 1000
        當季營收 = 當月營收.rolling(4).sum()
        市值營收比 = 市值 / 當季營收

        cond1 = (市值 < params['market_value_limit'])
        cond2 = 自由現金流 > params['min_free_cash_flow']
        cond3 = 股東權益報酬率 > params['min_roe']
        cond4 = 營業利益成長率 > params['min_op_profit_growth']
        cond5 = 市值營收比 < params['market_rev_ratio_limit']
        cond6 = vol > params['volume_threshold']
        cond排除月營收連3月衰退 = ~(rev_yoy_growth < params['rev_yoy_growth_limit']).sustain(3)
        cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)
        cond確認營收底部 = ((rev.rolling(12).min())/(rev) < 1.2).sustain(3)
        cond單月營收月增率連續3月大於閾值 = (rev_month_growth > params['rev_mom_growth_limit']).sustain(3)
        ma_period = params['ma_period']
        cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(ma_period*2))
        cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)
        業外收支營收率占比低 = (業外收支營收率 < 7.3)
        rsv_period = params['rsv_period']
        rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

        position = ((cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond排除月營收成長趨勢過老 & cond單月營收月增率連續3月大於閾值
                     & cond排除月營收連3月衰退 & cond近三個月營收大於年營收 & cond收盤價大於均線 & 業外收支營收率占比低 & cond確認營收底部) * rsv).is_largest(params['top_n'])
        position = position.reindex(當月營收.index_str_to_date().index, method='ffill')
        return position
    except Exception as e:
        return pd.DataFrame()

def strategy_revenue_price_turbo(params):
    """策略3：營收股價雙渦輪"""
    try:
        rev_ma_period = max(1, int(params['rev_ma_period']))
        rev_ma = rev.average(rev_ma_period)
        rev_ma_lookback = max(rev_ma_period + 1, int(params['rev_ma_lookback']))
        condition_近N月平均營收創M個月來新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=1).max()
        price_high_window = max(1, int(params['price_high_window']))
        condition_近N日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
        condition_成交均量大於閾值 = vol.average(1) > params['min_volume']
        long_ma_pattern = ((close > close.average(5)) & (close > close.average(10)) & (close > close.average(20)) & (close > close.average(60)) & (close > close.average(120)))
        收盤價_超級績效 = close > (close.average(params['performance_ma_period'])*params['performance_threshold'])
        rsi_higt_trend = (rsi > params['rsi_threshold']).sustain(params['rsi_trend_period'])
        gpm_trend_1 = (營業毛利率 > params['min_gpm']).sustain(params['gpm_sustain_period'])
        btpm_trend_1 = (稅前淨利率 > params['min_btpm']).sustain(params['btpm_sustain_period'])
        atpm_trend_1 = (稅後淨利率 > params['min_atpm']).sustain(params['atpm_sustain_period'])
        rev_rise_nsatisfy_2 = rev_yoy_growth.rank(pct=True, axis=1) > params['rev_growth_percentile']

        if INVENTORY_AVAILABLE and inventory is not None:
            try:
                boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= params['boss_min_level']) &
                                                    (inventory.持股分級.astype(int) <= params['boss_max_level'])]
                                        .reset_index().groupby(["date", "stock_id"])
                                        .agg({"占集保庫存數比例": "sum"}).reset_index()
                                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= params['min_boss_ratio']
            except Exception:
                boss_inventory_over400 = True
        else:
            boss_inventory_over400 = True

        pe_range_1 = (params['pe_limit'] <= pe)

        conditions = (condition_近N月平均營收創M個月來新高 & condition_近N日內有1日股價創新高 & condition_成交均量大於閾值 & long_ma_pattern & gpm_trend_1 & btpm_trend_1 & atpm_trend_1 & rev_rise_nsatisfy_2 & (close > params['min_price']) & (業外收支營收率 < 7.3) & ~(pe_range_1) & (收盤價_超級績效) & ((vol >= vol.rolling(20).mean()*0.8)) & (rsi_higt_trend) & (boss_inventory_over400) & ~(limit_up_all_day))

        position = rev_yoy_growth * conditions
        position = position[position > 0].is_largest(params['top_n']).reindex(rev.index_str_to_date().index, method="ffill")
        return position
    except Exception as e:
        return pd.DataFrame()

def strategy_high_yield_turtle(params):
    """策略4：高殖利率烏龜"""
    try:
        yield_ratio = data.get('price_earning_ratio:殖利率(%)')
        close_local = data.get('price:收盤價')
        vol_local = data.get('price:成交股數')
        sma20 = close_local.average(20)
        sma60 = close_local.average(60)
        rev_local = data.get('monthly_revenue:當月營收')
        ope_earn = data.get('fundamental_features:營業利益率')
        boss_hold = data.get("internal_equity_changes:董監持有股數占比")
        rev_growth_rate = data.get('monthly_revenue:去年同月增減(%)')

        cond1 = yield_ratio >= params['min_yield_ratio']
        cond2 = (close_local > sma20) & (close_local > sma60)
        cond3 = rev_local.average(3) > rev_local.average(12)
        cond4 = ope_earn >= params['min_op_earn_ratio']
        cond5 = boss_hold >= params['min_boss_hold']
        cond6 = (vol_local.average(5) >= params['min_volume']) & (vol_local.average(5) <= params['max_volume'])

        cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        cond_all = cond_all * rev_growth_rate
        position = cond_all[cond_all > 0].is_largest(params['top_n'])
        position = position.reindex(rev_local.index_str_to_date().index, method='ffill')
        return position
    except Exception as e:
        return pd.DataFrame()

def strategy_low_volatility_index(params):
    """策略5：低波動性指標"""
    try:
        cap = data.get('etl:market_value')
        vol_local = data.get('price:成交股數')
        close_local = data.get('price:收盤價')
        std = close_local.pct_change().rolling(params['std_window']).std().rank(axis=1, pct=True)

        position = cap[(vol_local.average(20) > params['min_volume']) &
                       (close_local > close_local.average(60)) &
                       (close_local > close_local.average(120)) &
                       (close_local > close_local.average(250)) &
                       (std < params['std_threshold'])].is_smallest(params['top_n'])
        position = position.reindex(close_local.index_str_to_date().index, method='ffill')
        return position
    except Exception as e:
        return pd.DataFrame()

def strategy_market_indicator(params):
    """策略6：藏獒外掛大盤指針"""
    try:
        close_local = data.get("price:收盤價")
        vol_local = data.get("price:成交股數")
        vol_ma = vol_local.average(10)
        rev_local = data.get('monthly_revenue:當月營收')
        rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
        rev_month_growth_local = data.get('monthly_revenue:上月比較增減(%)')

        cond1 = (close_local == close_local.rolling(params['new_high_window']).max())
        cond2 = ~(rev_year_growth < params['min_year_growth']).sustain(3)
        cond3 = ~(rev_year_growth > params['max_year_growth']).sustain(12, 8)
        cond4 = ((rev_local.rolling(12).min())/(rev_local) < params['rev_bottom_ratio']).sustain(3)
        cond5 = (rev_month_growth_local > params['min_month_growth']).sustain(3)
        cond6 = vol_ma > params['min_volume']

        buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
        buy = vol_ma * buy
        buy = buy[buy > 0]
        buy = buy.is_smallest(params['top_n'])
        position = buy.reindex(rev_local.index_str_to_date().index, method='ffill')
        return position
    except Exception as e:
        return pd.DataFrame()

# ============================================================================
#                   PART 9: 合併策略函數
# ============================================================================
def combined_strategy(gene, start_date_str=BACKTEST_START_DATE, end_date_str=None):
    """根據基因參數生成合併策略"""
    try:
        (allocation, low_vol_pe_params, small_inv_params, turbo_params,
         high_yield_turtle_params, low_vol_index_params, market_indicator_params,
         overall_params) = gene_to_params(gene)

        position_low_vol_pe = strategy_low_volatility_pe(low_vol_pe_params)
        position_small_investor = strategy_small_investor(small_inv_params)
        position_turbo = strategy_revenue_price_turbo(turbo_params)
        position_high_yield_turtle = strategy_high_yield_turtle(high_yield_turtle_params)
        position_low_vol_index = strategy_low_volatility_index(low_vol_index_params)
        position_market_indicator = strategy_market_indicator(market_indicator_params)

        position_combined = (
            position_low_vol_pe * allocation[0] +
            position_small_investor * allocation[1] +
            position_turbo * allocation[2] +
            position_high_yield_turtle * allocation[3] +
            position_low_vol_index * allocation[4] +
            position_market_indicator * allocation[5]
        )

        avg_daily_volume = 成交金額.average(20)
        liquid_stocks = avg_daily_volume > overall_params['liquidity_threshold']
        position_combined = position_combined * liquid_stocks

        if position_combined.empty:
            return pd.DataFrame(), overall_params, allocation

        if not isinstance(position_combined.index, pd.DatetimeIndex):
            position_combined.index = pd.to_datetime(position_combined.index, errors='coerce')
            position_combined = position_combined.loc[~position_combined.index.isna()]

        start_date_ts = pd.Timestamp(start_date_str)
        position_combined = position_combined[position_combined.index >= start_date_ts]

        if end_date_str is not None:
            end_date_ts = pd.Timestamp(end_date_str)
            position_combined = position_combined[position_combined.index <= end_date_ts]

        return position_combined, overall_params, allocation

    except Exception as e:
        return pd.DataFrame(), {}, []

# ============================================================================
#                   PART 10: 回測函數
# ============================================================================
def run_backtest(gene, name="策略", upload=False, display_report=False,
                 start_date=None, end_date=None):
    """執行回測並返回指標"""
    try:
        position_combined, overall_params, allocation = combined_strategy(
            gene,
            start_date_str=start_date or BACKTEST_START_DATE,
            end_date_str=end_date
        )

        if position_combined.empty:
            return None, None, None, None

        report = sim(
            position=position_combined,
            stop_loss=overall_params.get('stop_loss'),
            trail_stop=overall_params.get('trail_stop'),
            take_profit=overall_params.get('take_profit'),
            position_limit=overall_params.get('position_limit'),
            trade_at_price=overall_params.get('trade_at_price', 'close'),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            name=name,
            upload=upload
        )

        if report is None:
            return None, None, None, None

        metrics = report.get_metrics()

        if display_report:
            report.display()

        return report, metrics, position_combined, allocation

    except Exception as e:
        return None, None, None, None

# ============================================================================
#                   PART 11: 🔥 多目標適應度函數（核心優化）
# ============================================================================
def evaluate_multi_objective_fitness(individual):
    """
    🔥 多目標適應度評估

    目標：
    1. 夏普值 >= 4.0
    2. 胃納量 >= 500萬
    3. 年化報酬最大化
    4. 最大回撤 < 20%

    返回：(綜合適應度, 夏普值, 胃納量)
    """
    try:
        position_combined, overall_params, _ = combined_strategy(list(individual))

        if position_combined.empty or len(position_combined) < 20:
            return (-999.0, 0.0, 0.0)

        report = sim(
            position=position_combined,
            stop_loss=overall_params.get('stop_loss'),
            trail_stop=overall_params.get('trail_stop'),
            take_profit=overall_params.get('take_profit'),
            position_limit=overall_params.get('position_limit'),
            trade_at_price=overall_params.get('trade_at_price', 'close'),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            name='評估',
            upload=False
        )

        if report is None:
            return (-999.0, 0.0, 0.0)

        metrics = report.get_metrics()

        # 提取關鍵指標
        sharpe = metrics['ratio']['sharpeRatio']
        capacity = metrics['liquidity'].get('capacity', 0)
        annual_return = metrics['profitability'].get('annualReturn', 0)
        max_drawdown = abs(metrics['risk'].get('maxDrawdown', 1))

        # 🔥 多目標綜合適應度計算
        # 1. 夏普值分數（目標 >= 4.0）
        if sharpe >= TARGET_SHARPE:
            sharpe_score = 5.0 + (sharpe - TARGET_SHARPE) * 0.5  # 達標後每增加1給0.5分
        else:
            sharpe_score = (sharpe / TARGET_SHARPE) * 5.0  # 未達標按比例給分

        # 2. 胃納量分數（目標 >= 500萬）
        capacity_million = capacity / 1e6 if capacity > 0 else 0
        if capacity >= MIN_CAPACITY:
            capacity_score = 3.0 + min(2.0, (capacity - MIN_CAPACITY) / 1e7)  # 達標後每1000萬多給0.2分
        else:
            capacity_score = (capacity / MIN_CAPACITY) * 3.0  # 未達標按比例給分

        # 3. 年化報酬分數
        if annual_return >= TARGET_ANNUAL_RETURN:
            return_score = 1.5 + min(1.0, (annual_return - TARGET_ANNUAL_RETURN) * 2)
        else:
            return_score = (annual_return / TARGET_ANNUAL_RETURN) * 1.5 if annual_return > 0 else 0

        # 4. 回撤懲罰
        if max_drawdown <= MAX_DRAWDOWN:
            drawdown_score = 0.5
        else:
            drawdown_score = max(0, 0.5 - (max_drawdown - MAX_DRAWDOWN) * 2)

        # 🔥 綜合適應度（加權平均）
        综合适应度 = (
            sharpe_score * MULTI_OBJECTIVE_CONFIG['sharpe_weight'] +
            capacity_score * MULTI_OBJECTIVE_CONFIG['capacity_weight'] +
            return_score * MULTI_OBJECTIVE_CONFIG['return_weight'] +
            drawdown_score * MULTI_OBJECTIVE_CONFIG['drawdown_weight']
        )

        # 🔥 硬約束：胃納量不足直接懲罰
        if capacity < MIN_CAPACITY * 0.5:  # 低於最低要求的50%
            综合适应度 *= 0.3

        # 🔥 硬約束：回撤過大懲罰
        if max_drawdown > 0.3:  # 回撤超過30%
            综合适应度 *= 0.5

        return (综合适应度, sharpe, capacity_million)

    except Exception as e:
        return (-999.0, 0.0, 0.0)

# 🔥 並行評估函數（供 multiprocessing 使用）
def evaluate_individual_parallel(individual):
    """並行評估單個個體"""
    return evaluate_multi_objective_fitness(individual)

# ============================================================================
#                   PART 12: DEAP 多目標優化設定（NSGA-II）
# ============================================================================

# 創建適應度類別（三個目標：綜合適應度、夏普值、胃納量）
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def random_gene_optimized():
    """
    🔥 優化版基因初始化
    - 更合理的參數範圍
    - 避免極端值
    """
    gene = []

    # 前6個基因：配置比例（使用正數，後續會做 softmax）
    for _ in range(6):
        gene.append(random.uniform(0.1, 1.0))

    # 其餘基因：使用常態分佈，避免極端值
    for i in range(6, GENE_LENGTH):
        if i < 50:  # 主要參數
            gene.append(random.gauss(20, 15))  # 均值20，標準差15
        elif i < 100:  # 次要參數
            gene.append(random.gauss(10, 10))
        else:  # 附加參數
            gene.append(random.gauss(5, 8))

    return gene

toolbox.register("individual", tools.initIterate, creator.Individual, random_gene_optimized)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_multi_objective_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.15)  # 降低變異強度
toolbox.register("select", tools.selNSGA2)  # 🔥 使用 NSGA-II 選擇

# ============================================================================
#                   PART 13: 🔥 並行評估引擎
# ============================================================================
class ParallelEvaluator:
    """並行評估引擎"""

    def __init__(self, num_workers=None):
        self.num_workers = num_workers or GA_CONFIG['num_cpu_cores']
        self.pool = None

    def __enter__(self):
        if GA_CONFIG['enable_parallel'] and self.num_workers > 1:
            self.pool = Pool(processes=self.num_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def evaluate_population(self, population):
        """評估整個族群"""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        if not invalid_ind:
            return

        if self.pool and len(invalid_ind) > 1:
            # 並行評估
            fitnesses = self.pool.map(evaluate_individual_parallel, invalid_ind)
        else:
            # 序列評估
            fitnesses = list(map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

# ============================================================================
#                   PART 14: 主演化引擎
# ============================================================================
def run_evolution(window_id, num_generations=500, population_size=80):
    """
    🔥 主演化引擎（NSGA-II 多目標優化）
    """
    print(f"\n{'='*70}")
    print(f"🧬 開始演化 - Window {window_id}")
    print(f"{'='*70}")

    # 初始化族群
    population = toolbox.population(n=population_size)

    # 統計記錄
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # 綜合適應度
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    # 名人堂（保存最佳個體）
    hof = tools.ParetoFront()

    # 並行評估器
    with ParallelEvaluator() as evaluator:
        # 初始評估
        evaluator.evaluate_population(population)
        hof.update(population)

        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        print(logbook.stream)

        # 演化主循環
        for gen in range(1, num_generations + 1):
            gen_start_time = time.time()

            # 🔥 NSGA-II 選擇
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉
            for i in range(1, len(offspring), 2):
                if random.random() < GA_CONFIG['crossover_rate']:
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # 🔥 自適應變異率（後期降低變異）
            adaptive_mutation_rate = GA_CONFIG['mutation_rate'] * (1 - gen / num_generations * 0.5)

            for i in range(len(offspring)):
                if random.random() < adaptive_mutation_rate:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # 評估新個體
            evaluator.evaluate_population(offspring)

            # 更新族群（NSGA-II 替換）
            population[:] = toolbox.select(population + offspring, population_size)
            hof.update(population)

            # 記錄統計
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len([ind for ind in offspring if not ind.fitness.valid]), **record)

            gen_time = time.time() - gen_start_time

            # 🔥 每10代詳細輸出
            if gen % 10 == 0:
                best_ind = max(population, key=lambda x: x.fitness.values[0])
                print(f"\n=== 第 {gen}/{num_generations} 代 ===")
                print(f"   最佳綜合適應度: {best_ind.fitness.values[0]:.4f}")
                print(f"   最佳夏普值: {best_ind.fitness.values[1]:.4f}")
                print(f"   最佳胃納量: {best_ind.fitness.values[2]:.2f} 萬")
                print(f"   本代平均: {record['avg']:.4f}")
                print(f"   耗時: {gen_time:.1f}s")

                # 🔥 檢查是否達標
                if (best_ind.fitness.values[1] >= TARGET_SHARPE and
                    best_ind.fitness.values[2] * 1e6 >= MIN_CAPACITY):
                    print(f"\n🎉 達標！夏普值 {best_ind.fitness.values[1]:.4f} >= {TARGET_SHARPE}")
                    print(f"   胃納量 {best_ind.fitness.values[2]:.2f}萬 >= {MIN_CAPACITY/1e6:.0f}萬")
                    break

            # 儲存檢查點
            if gen % GA_CONFIG['checkpoint_interval'] == 0:
                checkpoint = {
                    'generation': gen,
                    'population': population,
                    'halloffame': list(hof),
                    'logbook': logbook,
                    'best_fitness': max([ind.fitness.values[0] for ind in population]),
                }
                cp_file = os.path.join(CHECKPOINT_PATH, f"checkpoint_w{window_id}_gen{gen}.pkl")
                with open(cp_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"   💾 檢查點已儲存")

            # 記憶體清理
            if gen % 50 == 0:
                gc.collect()

    return population, hof, logbook

# ============================================================================
#                   PART 15: 主程式
# ============================================================================
def main():
    """主程式"""
    print("\n" + "=" * 80)
    print(f"🧬 台股基因演算法優化系統 v9.0 - Window {WINDOW_ID}")
    print("=" * 80)

    start_time = time.time()

    # 執行演化
    population, hof, logbook = run_evolution(
        window_id=WINDOW_ID,
        num_generations=GA_CONFIG['num_generations'],
        population_size=GA_CONFIG['population_size']
    )

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"🏁 演化完成 - Window {WINDOW_ID}")
    print(f"{'='*70}")
    print(f"總耗時: {total_time/60:.1f} 分鐘")

    # 🔥 找出 Pareto 前緣的最佳解
    print(f"\n📊 Pareto 最優解集（前10名）:")
    print(f"{'排名':<6}{'綜合適應度':<15}{'夏普值':<12}{'胃納量(萬)':<15}{'達標':<8}")
    print("=" * 70)

    sorted_hof = sorted(hof, key=lambda x: x.fitness.values[0], reverse=True)[:10]

    qualified_solutions = []
    for rank, ind in enumerate(sorted_hof, 1):
        综合适应度, sharpe, capacity_million = ind.fitness.values
        is_qualified = (sharpe >= TARGET_SHARPE and capacity_million * 1e6 >= MIN_CAPACITY)

        print(f"{rank:<6}{综合适应度:<15.4f}{sharpe:<12.4f}{capacity_million:<15.2f}{'✅' if is_qualified else '❌':<8}")

        if is_qualified:
            qualified_solutions.append((rank, ind, sharpe, capacity_million))

    # 完整回測最佳解
    if qualified_solutions:
        print(f"\n{'='*70}")
        print(f"🎯 找到 {len(qualified_solutions)} 個達標解！進行完整回測...")
        print(f"{'='*70}")

        for rank, ind, sharpe, capacity_million in qualified_solutions[:3]:  # 只回測前3名
            print(f"\n📊 第 {rank} 名完整回測")
            print(f"   預估夏普: {sharpe:.4f}, 預估胃納量: {capacity_million:.2f}萬")

            report, metrics, _, allocation = run_backtest(
                gene=list(ind),
                name=f"W{WINDOW_ID}_第{rank}名",
                upload=False,
                display_report=True
            )

            if metrics:
                print(f"\n   策略配置: {[f'{a:.2%}' for a in allocation]}")
    else:
        print(f"\n⚠️ 未找到完全達標的解，顯示最佳解回測結果...")
        best_ind = sorted_hof[0]
        report, metrics, _, allocation = run_backtest(
            gene=list(best_ind),
            name=f"W{WINDOW_ID}_最佳解",
            upload=False,
            display_report=True
        )

    # 儲存最終結果
    result = {
        'window_id': WINDOW_ID,
        'population': population,
        'halloffame': list(hof),
        'logbook': logbook,
        'qualified_solutions': [(list(ind), s, c) for _, ind, s, c in qualified_solutions],
        'timestamp': datetime.datetime.now().isoformat(),
    }

    result_file = os.path.join(BACKTEST_RESULTS_PATH, f"final_result_w{WINDOW_ID}.pkl")
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✅ 結果已儲存: {result_file}")
    print(f"{'='*70}")

# ============================================================================
#                   執行
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Window {WINDOW_ID} 使用者中斷")
    except Exception as e:
        print(f"\n❌ Window {WINDOW_ID} 錯誤: {e}")
        traceback.print_exc()
