# =============================================================================
# 🐉 九組合天空龍策略 - 純夏普優化版 v4.0
# =============================================================================
# v4.0 核心改進：
# ✅ 1. 純夏普優先 - 單目標優化，夏普值永遠第一
# ✅ 2. 分數疊加機制 - 9策略評分疊加，產生差異化持倉比例
# ✅ 3. 固定 ~15 檔持股 - 每檔至少 3%，最多 20 檔
# ✅ 4. 全股評分 - 每個策略對所有股票評分，不是選 top N
# =============================================================================
# 關鍵邏輯：
# - 每個策略返回 0-1 分數（不是布林值）
# - 同一支股票被多策略評高分 = 總分更高 = 持倉比例更高
# - 最終選分數最高的 ~15 檔，持倉比例依分數分配
# =============================================================================

#%% ========== 安裝套件 ==========
import subprocess
import sys
try:
    import finlab
    import deap
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "finlab", "deap", "-q"])

#%% ========== 環境變數與設定 ==========
import os
import sys
import json
import pickle
import random
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import reduce
import multiprocessing as mp

# ===== 環境變數設定 =====
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))
CONTINUE_EVOLUTION = os.environ.get('CONTINUE_EVOLUTION', 'false').lower() == 'true'
EVOLUTION_GENERATIONS = int(os.environ.get('EVOLUTION_GENERATIONS', '200'))
EVALUATE_MODE = os.environ.get('EVALUATE_MODE', 'false').lower() == 'true'
BASE_DIR = os.environ.get('BASE_DIR', '/content/drive/MyDrive/九組合天空龍_GA_v4')

os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

from deap import base, creator, tools

# 🔐 API Key 取得順序：1. 環境變數 2. Colab Secrets 3. 空值
def get_finlab_api_key():
    api_key = os.environ.get('FINLAB_API_KEY', '')
    if api_key:
        return api_key
    try:
        from google.colab import userdata
        api_key = userdata.get('FINLAB_API_KEY')
        if api_key:
            return api_key
    except:
        pass
    return ""

FINLAB_API_KEY = get_finlab_api_key()

import finlab
if FINLAB_API_KEY:
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")
else:
    try:
        from finlab import data
        _ = data.get('price:收盤價')
        print("✅ 使用已登入的 FinLab session")
    except:
        print("⚠️ 請設定 FINLAB_API_KEY 環境變數或 Colab Secrets")

from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🐉 九組合天空龍策略 - 純夏普優化版 v4.0                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔥 核心改進: 純夏普優先 + 分數疊加 + 差異化持倉比例                           ║
║  🔥 目標持股: ~15 檔，每檔至少 3%                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FinLab: {finlab.__version__:<10}  CPU 核心: {mp.cpu_count():<5}                                    ║
║  WINDOW_ID: {WINDOW_ID:<5}  CONTINUE: {CONTINUE_EVOLUTION}  GENERATIONS: {EVOLUTION_GENERATIONS:<5}             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

#%% ========== 核心參數設定 ==========
# 🎯 純夏普目標
TARGET_SHARPE = 4.0
MAX_DRAWDOWN = 0.25           # 25% 硬性回撤限制
MIN_CAPACITY = 5_000_000      # 500萬最低胃納

# ⚠️ 核心參數：持股約 15 檔
TARGET_STOCKS = 15            # 目標持股數
MIN_POSITION_RATIO = 0.03     # 每股至少 3%
MAX_STOCKS = int(1 / MIN_POSITION_RATIO)  # 最多 33 檔

# GA 設定
POPULATION_SIZE = 50
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
ELITE_SIZE = 8
CHECKPOINT_INTERVAL = 5
DETAILED_BACKTEST_INTERVAL = 10

# 回測時間設定
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
BACKTEST_START = '2017-01-01'

# 基因長度 (精簡版)
GENE_LENGTH = 100

# 儲存路徑
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✅ Google Drive 已掛載")
except:
    print("⚠️ 本地模式")

Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = f"{BASE_DIR}/checkpoint_v4_w{WINDOW_ID}.pkl"
PARETO_FILE = f"{BASE_DIR}/pareto_v4_w{WINDOW_ID}.pkl"
BEST_PARAMS_FILE = f"{BASE_DIR}/best_params_v4_w{WINDOW_ID}.json"

print(f"📁 儲存路徑: {BASE_DIR}")
print(f"📄 Checkpoint: checkpoint_v4_w{WINDOW_ID}.pkl")

#%% ========== 載入數據 ==========
print("\n📊 載入 FinLab 數據...")
data.set_universe('TSE_OTC')

# 價格數據
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

# 基本面
pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')
rev = data.get('monthly_revenue:當月營收')
rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
rev_mom = data.get('monthly_revenue:上月比較增減(%)')

# 財務指標
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
營業利益率 = data.get('fundamental_features:營業利益率')

# 籌碼
融資使用率 = data.get('margin_transactions:融資使用率')
董監持股 = data.get("internal_equity_changes:董監持有股數占比")

# 其他
市值 = data.get('etl:market_value')
投資現金流 = data.get('financial_statement:投資活動之淨現金流入_流出')
營業現金流 = data.get('financial_statement:營業活動之淨現金流入_流出')
稅後淨利 = data.get('fundamental_features:經常稅後淨利')
權益總計 = data.get('financial_statement:股東權益總額')

# 技術指標
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)

# 衍生指標
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)
volatility = atr / adj_close
自由現金流 = (投資現金流 + 營業現金流).rolling(4).mean()
ROE_calc = 稅後淨利 / 權益總計
當月營收 = rev * 1000
市值營收比 = 市值 / 當月營收.rolling(4).sum()

# 漲停鎖死判斷
limit_up = (close > close.shift(1) * 1.095)
limit_up_all_day = ((close * close).replace(0.0, np.nan) == (close * high).replace(0.0, np.nan)).replace(False, np.nan)
limit_up_all_day = (limit_up_all_day == limit_up).fillna(False)

# 基礎流動性過濾
MIN_VOL_FILTER = 100000  # 每日至少 10 萬股成交量
liquid_mask = vol.average(20) > MIN_VOL_FILTER

print("✅ 數據載入完成")

#%% ========== 基因解碼器 v4.0 ==========
class GeneDecoder:
    """
    v4.0 精簡版基因解碼器

    基因配置 (100 個):
    - 0-8:   策略權重 (9 個)
    - 9-17:  各策略篩選強度 (9 個)
    - 18-26: 各策略評分指標權重 (9 個)
    - 27-35: 技術面參數 (9 個)
    - 36-44: 基本面參數 (9 個)
    - 45-53: 籌碼面參數 (9 個)
    - 54-62: 流動性參數 (9 個)
    - 63-71: 動能參數 (9 個)
    - 72-80: 價值參數 (9 個)
    - 81-89: 風控參數 (9 個)
    - 90-99: 全局參數 (10 個)
    """

    @staticmethod
    def decode(genes: List[float]) -> Dict[str, Any]:
        genes = list(genes) + [0.5] * (GENE_LENGTH - len(genes))
        genes = genes[:GENE_LENGTH]
        p = {}

        # 策略權重 (0-8) - 確保加總為 1
        raw_w = [max(0.05, genes[i]) for i in range(9)]  # 最低 5% 權重
        total = sum(raw_w)
        for i in range(9):
            p[f'w{i+1}'] = raw_w[i] / total

        # 各策略篩選強度 (9-17) - 控制每個策略選股的嚴格程度
        for i in range(9):
            p[f's{i+1}_strictness'] = genes[9+i] * 0.8 + 0.1  # 10%-90%

        # 各策略評分指標權重 (18-26)
        for i in range(9):
            p[f's{i+1}_score_weight'] = genes[18+i] * 2.0 + 0.5  # 0.5-2.5

        # 技術面參數 (27-35)
        p['ma_short'] = int(genes[27] * 40 + 5)       # 5-45 日
        p['ma_mid'] = int(genes[28] * 100 + 20)      # 20-120 日
        p['ma_long'] = int(genes[29] * 200 + 60)     # 60-260 日
        p['rsi_low'] = genes[30] * 30 + 20           # 20-50
        p['rsi_high'] = genes[31] * 30 + 60          # 60-90
        p['atr_mult'] = genes[32] * 2.0 + 0.5        # 0.5-2.5
        p['price_momentum'] = int(genes[33] * 60 + 20)  # 20-80 日動能
        p['vol_ma'] = int(genes[34] * 20 + 5)        # 5-25 日均量
        p['breakout_period'] = int(genes[35] * 200 + 60)  # 60-260 日突破

        # 基本面參數 (36-44)
        p['pe_min'] = genes[36] * 10 + 3             # 3-13
        p['pe_max'] = genes[37] * 30 + 15            # 15-45
        p['pb_max'] = genes[38] * 5 + 1              # 1-6
        p['roe_min'] = genes[39] * 20                # 0-20%
        p['gpm_min'] = genes[40] * 30                # 0-30%
        p['rev_growth_min'] = genes[41] * 50 - 20    # -20% ~ 30%
        p['opm_min'] = genes[42] * 20                # 0-20%
        p['fcf_positive'] = genes[43] > 0.5         # 要求正自由現金流
        p['dividend_min'] = genes[44] * 5            # 0-5%

        # 籌碼面參數 (45-53)
        p['margin_max'] = genes[45] * 40 + 10        # 10-50%
        p['director_min'] = genes[46] * 30           # 0-30%
        p['foreign_trend'] = int(genes[47] * 20 + 5)  # 5-25 日外資趨勢
        p['trust_trend'] = int(genes[48] * 20 + 5)   # 5-25 日投信趨勢
        p['chip_score_weight'] = genes[49] * 1.5 + 0.5  # 0.5-2.0
        for i in range(50, 54):
            p[f'chip_p{i-49}'] = genes[i]

        # 流動性參數 (54-62)
        p['min_vol'] = genes[54] * 300000 + 50000    # 5-35 萬股
        p['min_cap'] = genes[55] * 5e9 + 1e9         # 10-60 億
        p['max_cap'] = genes[56] * 200e9 + 10e9      # 100-2100 億
        p['vol_spike'] = genes[57] * 2.0 + 1.0       # 1-3 倍量比
        p['liquidity_score_weight'] = genes[58] * 1.5 + 0.5
        for i in range(59, 63):
            p[f'liq_p{i-58}'] = genes[i]

        # 動能參數 (63-71)
        p['momentum_window'] = int(genes[63] * 60 + 20)  # 20-80 日
        p['momentum_threshold'] = genes[64] * 0.3       # 0-30%
        p['trend_strength'] = genes[65] * 0.5 + 0.3     # 30-80%
        p['volatility_max'] = genes[66] * 0.1 + 0.02    # 2-12%
        p['momentum_score_weight'] = genes[67] * 1.5 + 0.5
        for i in range(68, 72):
            p[f'mom_p{i-67}'] = genes[i]

        # 價值參數 (72-80)
        p['value_pe_weight'] = genes[72] * 1.5 + 0.5
        p['value_pb_weight'] = genes[73] * 1.5 + 0.5
        p['value_div_weight'] = genes[74] * 1.5 + 0.5
        p['value_fcf_weight'] = genes[75] * 1.5 + 0.5
        p['value_growth_weight'] = genes[76] * 1.5 + 0.5
        p['value_score_weight'] = genes[77] * 1.5 + 0.5
        for i in range(78, 81):
            p[f'val_p{i-77}'] = genes[i]

        # 風控參數 (81-89)
        p['stop_loss'] = genes[81] * 0.15 + 0.10      # 10-25% 停損
        p['trail_stop'] = genes[82] * 0.25 + 0.15     # 15-40% 移動停損
        p['take_profit'] = genes[83] * 0.5 + 0.3      # 30-80% 停利
        p['position_limit'] = genes[84] * 0.20 + 0.15 # 15-35% 單股上限
        p['drawdown_exit'] = genes[85] * 0.15 + 0.15  # 15-30% 總回撤
        p['correlation_limit'] = genes[86] * 0.3 + 0.5  # 50-80% 相關性上限
        for i in range(87, 90):
            p[f'risk_p{i-86}'] = genes[i]

        # 全局參數 (90-99)
        p['n_stocks'] = int(genes[90] * 10 + 10)       # 10-20 檔 (目標 ~15)
        p['rebalance_threshold'] = genes[91] * 0.15 + 0.05  # 5-20% 調整門檻
        p['score_decay'] = genes[92] * 0.3 + 0.7      # 70-100% 分數衰減
        p['overlap_bonus'] = genes[93] * 1.5 + 1.0    # 1-2.5 倍重疊加分
        p['diversity_weight'] = genes[94] * 0.5       # 0-50% 多樣性權重
        for i in range(95, 100):
            p[f'global_p{i-94}'] = genes[i]

        return p

#%% ========== 九大策略評分函數 v4.0 ==========
# 🔥 關鍵改變：每個策略返回 0-1 分數，不是布林值
# 這樣多策略疊加才能產生差異化

def score_strategy_1(p):
    """
    策略1: 低波動價值股
    評分邏輯: PE 低 + 波動低 + 營收成長 = 高分
    """
    try:
        # 基礎條件過濾
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol']) &
            ~limit_up_all_day
        )

        # 評分組件 (各 0-1 分)
        # 1. PE 分數 (越低越好，在合理範圍內)
        pe_valid = (pe > p['pe_min']) & (pe < p['pe_max'])
        pe_score = (1 - (pe - p['pe_min']) / (p['pe_max'] - p['pe_min'])).clip(0, 1)
        pe_score = pe_score.where(pe_valid, 0)

        # 2. 波動率分數 (越低越好)
        vol_score = (1 - volatility / p['volatility_max']).clip(0, 1)

        # 3. 營收成長分數
        rev_growth = rev_ma3 / rev_ma12
        rev_score = ((rev_growth - 1) / 0.5 + 0.5).clip(0, 1)

        # 4. 毛利率分數
        gpm_score = (營業毛利率 / 50).clip(0, 1)

        # 5. ROE 分數
        roe_score = (ROE / 30).clip(0, 1)

        # 加權總分
        total_score = (
            pe_score * 0.25 +
            vol_score * 0.25 +
            rev_score * 0.20 +
            gpm_score * 0.15 +
            roe_score * 0.15
        ) * p['s1_score_weight']

        # 套用基礎條件
        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_2(p):
    """
    策略2: 小型成長股
    評分邏輯: 市值小 + 自由現金流正 + 價值被低估 = 高分
    """
    try:
        base_cond = (
            liquid_mask &
            (市值 < p['max_cap']) &
            (vol.average(20) > p['min_vol'])
        )

        # 1. 市值分數 (越小越好，但不要太小)
        cap_score = (1 - 市值 / p['max_cap']).clip(0, 1)
        cap_score = cap_score.where(市值 > p['min_cap'], 0)

        # 2. 自由現金流分數
        fcf_score = (自由現金流 > 0).astype(float)
        if p['fcf_positive']:
            fcf_score = fcf_score * 1.5

        # 3. 市值營收比分數 (越低越好)
        ps_score = (1 - 市值營收比 / 5).clip(0, 1)

        # 4. ROE 分數
        roe_score = (ROE_calc * 10).clip(0, 1)

        # 5. 動能分數 (RSV)
        rsv = (close - close.rolling(p['momentum_window']).min()) / \
              (close.rolling(p['momentum_window']).max() - close.rolling(p['momentum_window']).min())
        momentum_score = rsv.clip(0, 1)

        total_score = (
            cap_score * 0.20 +
            fcf_score * 0.25 +
            ps_score * 0.20 +
            roe_score * 0.20 +
            momentum_score * 0.15
        ) * p['s2_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_3(p):
    """
    策略3: 營收雙渦輪
    評分邏輯: 營收創高 + 股價創高 + 毛利穩定 = 高分
    """
    try:
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol']) &
            ~limit_up_all_day
        )

        # 1. 營收創高分數
        rev_ma = rev.average(3)
        rev_at_high = rev_ma == rev_ma.rolling(12, min_periods=3).max()
        rev_high_score = rev_at_high.astype(float)

        # 2. 股價創高分數
        price_at_high = close == close.rolling(p['breakout_period']).max()
        price_high_score = price_at_high.astype(float) * 0.8 + \
                          (close > close.average(p['ma_long'])).astype(float) * 0.2

        # 3. 營收年增率分數
        rev_yoy_score = (rev_yoy / 100 + 0.5).clip(0, 1)

        # 4. 毛利率穩定分數
        gpm_stable = 營業毛利率.rolling(4).std() < 5
        gpm_score = gpm_stable.astype(float) * 0.5 + (營業毛利率 / 40).clip(0, 0.5)

        # 5. RSI 趨勢分數
        rsi_score = ((rsi - 30) / 40).clip(0, 1)

        total_score = (
            rev_high_score * 0.30 +
            price_high_score * 0.25 +
            rev_yoy_score * 0.20 +
            gpm_score * 0.15 +
            rsi_score * 0.10
        ) * p['s3_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_4(p):
    """
    策略4: 高殖利率價值股
    評分邏輯: 殖利率高 + 董監持股高 + 營益率佳 = 高分
    """
    try:
        base_cond = (
            liquid_mask &
            (dividend_yield > 0) &
            (vol.average(20) > p['min_vol'])
        )

        # 1. 殖利率分數
        div_score = (dividend_yield / 10).clip(0, 1)

        # 2. 董監持股分數
        director_score = (董監持股 / 50).clip(0, 1)

        # 3. 營業利益率分數
        opm_score = (營業利益率 / 30).clip(0, 1)

        # 4. 營收趨勢分數
        rev_trend = rev.average(3) > rev.average(12)
        rev_trend_score = rev_trend.astype(float)

        # 5. 價格趨勢分數
        ma_trend = (close > close.average(p['ma_short'])) & \
                   (close > close.average(p['ma_mid']))
        ma_trend_score = ma_trend.astype(float)

        total_score = (
            div_score * 0.30 +
            director_score * 0.20 +
            opm_score * 0.20 +
            rev_trend_score * 0.15 +
            ma_trend_score * 0.15
        ) * p['s4_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_5(p):
    """
    策略5: 低波動穩健股
    評分邏輯: 波動低 + 均線多頭 + 市值偏小 = 高分
    """
    try:
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol'])
        )

        # 1. 波動率分數 (標準差排名)
        std = close.pct_change().rolling(60).std()
        std_rank = std.rank(axis=1, pct=True)
        vol_score = (1 - std_rank).clip(0, 1)

        # 2. 均線多頭分數
        ma_bull = (
            (close > close.average(p['ma_short'])).astype(float) * 0.3 +
            (close > close.average(p['ma_mid'])).astype(float) * 0.3 +
            (close > close.average(p['ma_long'])).astype(float) * 0.4
        )

        # 3. 市值分數 (偏小但不要太小)
        cap_rank = 市值.rank(axis=1, pct=True)
        cap_score = (1 - cap_rank).clip(0, 1)
        cap_score = cap_score.where(市值 > p['min_cap'], 0)

        # 4. 融資使用率分數 (越低越好)
        margin_score = (1 - 融資使用率 / p['margin_max']).clip(0, 1)

        # 5. 成交量穩定分數
        vol_std = vol.rolling(20).std() / vol.rolling(20).mean()
        vol_stable_score = (1 - vol_std / 2).clip(0, 1)

        total_score = (
            vol_score * 0.30 +
            ma_bull * 0.25 +
            cap_score * 0.20 +
            margin_score * 0.15 +
            vol_stable_score * 0.10
        ) * p['s5_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_6(p):
    """
    策略6: 創高突破股
    評分邏輯: 價格創高 + 量能配合 + 營收不差 = 高分
    """
    try:
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol']) &
            ~limit_up_all_day
        )

        # 1. 創高分數 (距離新高的程度)
        high_260 = close.rolling(260).max()
        dist_to_high = close / high_260
        high_score = dist_to_high.clip(0.8, 1) - 0.8  # 0-0.2 -> 0-1
        high_score = (high_score / 0.2).clip(0, 1)

        # 2. 量比分數
        vol_ma = vol.average(p['vol_ma'])
        vol_ratio = vol / vol_ma
        vol_score = ((vol_ratio - 0.5) / 2).clip(0, 1)

        # 3. 營收月增分數
        rev_mom_score = ((rev_mom + 20) / 40).clip(0, 1)

        # 4. 營收年增分數
        rev_yoy_score = ((rev_yoy + 30) / 60).clip(0, 1)

        # 5. 均線支撐分數
        ma_support = (close > close.average(60)).astype(float)

        total_score = (
            high_score * 0.35 +
            vol_score * 0.20 +
            rev_mom_score * 0.15 +
            rev_yoy_score * 0.15 +
            ma_support * 0.15
        ) * p['s6_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_7(p):
    """
    策略7: 財務品質股
    評分邏輯: 多重財務指標綜合評分
    """
    try:
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol'])
        )

        # 財務指標排名分數
        scores = []

        # 1. 營業利益成長率
        if 營業利益成長率 is not None:
            og_score = 營業利益成長率.rank(axis=1, pct=True).fillna(0)
            scores.append(og_score)

        # 2. ROE
        if ROE is not None:
            roe_score = ROE.rank(axis=1, pct=True).fillna(0)
            scores.append(roe_score)

        # 3. 毛利率
        if 營業毛利率 is not None:
            gpm_score = 營業毛利率.rank(axis=1, pct=True).fillna(0)
            scores.append(gpm_score)

        # 4. 稅後淨利率
        if 稅後淨利率 is not None:
            npm_score = 稅後淨利率.rank(axis=1, pct=True).fillna(0)
            scores.append(npm_score)

        # 5. 營業利益率
        if 營業利益率 is not None:
            opm_score = 營業利益率.rank(axis=1, pct=True).fillna(0)
            scores.append(opm_score)

        if not scores:
            return pd.DataFrame()

        # 綜合財務分數
        finance_score = sum(scores) / len(scores)

        # 波動率調整 (低波動加分)
        std = close.pct_change().rolling(60).std()
        vol_adj = (1 - std.rank(axis=1, pct=True)).fillna(0.5) * 0.3

        # 均線加分
        ma_adj = (close > close.average(p['ma_mid'])).astype(float) * 0.2

        total_score = (finance_score * 0.7 + vol_adj + ma_adj) * p['s7_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_8(p):
    """
    策略8: 技術動能股
    評分邏輯: 均線趨勢 + 動能指標 + 量能趨勢
    """
    try:
        base_cond = (
            liquid_mask &
            (vol > p['min_vol'])
        )

        # 1. 均線趨勢分數
        def wma(x, n):
            return x.ewm(com=n).mean()

        sma_trend = close / close.rolling(p['ma_long']).mean() - 1
        sma_score = (sma_trend / 0.3 + 0.5).clip(0, 1)

        # 2. 動能分數 (價格漲幅排名)
        momentum = close / close.shift(p['momentum_window']) - 1
        mom_rank = momentum.rank(axis=1, pct=True).fillna(0.5)

        # 3. 均線斜率分數
        ma60 = close.rolling(60).mean()
        ma_slope = ma60 / ma60.shift(60) - 1
        slope_score = (ma_slope / 0.3 + 0.5).clip(0, 1)

        # 4. 量能趨勢分數
        vol_trend = vol.rolling(20).mean() / vol.rolling(60).mean()
        vol_trend_score = (vol_trend - 0.5).clip(0, 1)

        # 5. 波動率適中分數 (不要太高也不要太低)
        std = close.pct_change().rolling(60).std()
        std_rank = std.rank(axis=1, pct=True)
        vol_mid_score = 1 - abs(std_rank - 0.5) * 2  # 中間最高

        total_score = (
            sma_score * 0.25 +
            mom_rank * 0.25 +
            slope_score * 0.20 +
            vol_trend_score * 0.15 +
            vol_mid_score * 0.15
        ) * p['s8_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_9(p):
    """
    策略9: 綜合品質動能股
    評分邏輯: 財務品質 + 技術動能 + 籌碼安全
    """
    try:
        base_cond = (
            liquid_mask &
            (vol.average(20) > p['min_vol'])
        )

        # 1. 財務品質分數 (多指標平均)
        fs = []
        for f in [營業利益成長率, ROE, 營業毛利率, 稅後淨利率, 營業利益率]:
            if f is not None:
                f_score = f.average(4).rank(axis=1, pct=True).fillna(0)
                fs.append(f_score)

        finance_score = sum(fs) / len(fs) if fs else pd.DataFrame(0.5, index=close.index, columns=close.columns)

        # 2. 技術趨勢分數
        ma_bull = (
            (close > close.average(20)).astype(float) * 0.2 +
            (close > close.average(60)).astype(float) * 0.3 +
            (close > close.average(120)).astype(float) * 0.5
        )

        # 3. 波動率低分數
        std = close.pct_change().rolling(60).std()
        vol_low_score = (1 - std.rank(axis=1, pct=True)).fillna(0.5)

        # 4. 流動性分數
        vol_rank = vol.rolling(20).mean().rank(axis=1, pct=True).fillna(0.5)

        # 5. 籌碼安全分數
        margin_safe = (融資使用率 < p['margin_max']).astype(float)

        total_score = (
            finance_score * 0.30 +
            ma_bull * 0.25 +
            vol_low_score * 0.20 +
            vol_rank * 0.15 +
            margin_safe * 0.10
        ) * p['s9_score_weight']

        total_score = total_score.where(base_cond, 0)

        return total_score.fillna(0)
    except:
        return pd.DataFrame()

#%% ========== 組合策略與回測 v4.0 ==========
def combine_strategies_v4(params: Dict) -> pd.DataFrame:
    """
    🔥 v4.0 核心：分數疊加產生差異化持倉

    關鍵邏輯：
    1. 每個策略對所有股票評分 (0-1)
    2. 用策略權重加權疊加分數
    3. 同一支股票被多策略評高分 = 總分更高
    4. 選總分最高的 N 檔
    5. 用分數比例作為持倉比例 (產生差異化)
    """
    strategies = [
        score_strategy_1, score_strategy_2, score_strategy_3,
        score_strategy_4, score_strategy_5, score_strategy_6,
        score_strategy_7, score_strategy_8, score_strategy_9
    ]

    scores = []

    for i, func in enumerate(strategies):
        try:
            score = func(params)
            if score is not None and not score.empty:
                weight = params.get(f'w{i+1}', 1/9)
                weighted_score = score * weight
                scores.append(weighted_score)
        except:
            pass

    if not scores:
        return pd.DataFrame()

    # 🔥 關鍵：疊加所有策略分數
    combined_score = reduce(lambda a, b: a.add(b, fill_value=0), scores)

    # 確保數值型態
    combined_score = combined_score.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 目標持股數量
    n_stocks = params.get('n_stocks', TARGET_STOCKS)
    n_stocks = min(n_stocks, MAX_STOCKS)  # 不超過 33 檔

    # 重疊加分係數
    overlap_bonus = params.get('overlap_bonus', 1.5)

    def normalize_to_position(row):
        """
        🔥 v4.0: 分數轉持倉比例

        步驟：
        1. 選分數最高的 N 檔
        2. 用分數比例分配持倉
        3. 迭代移除 < 3% 的股票
        4. 最終產生差異化持倉比例
        """
        row_numeric = pd.to_numeric(row, errors='coerce').fillna(0)
        valid = row_numeric[row_numeric > 0]

        if len(valid) == 0:
            return pd.Series(0.0, index=row.index)

        # 選分數最高的 N 檔 (保留原始分數)
        top_n = valid.nlargest(min(n_stocks, len(valid)))

        # 迭代過濾：移除 < 3% 且重新分配
        for _ in range(10):
            total = top_n.sum()
            if total <= 0:
                return pd.Series(0.0, index=row.index)

            # 用分數比例作為持倉比例 (這會產生差異化!)
            normalized = top_n / total

            # 檢查 < 3% 的股票
            below_min = normalized < MIN_POSITION_RATIO
            if not below_min.any():
                # 全部 >= 3%，完成
                result = pd.Series(0.0, index=row.index)
                result[normalized.index] = normalized.values
                return result

            # 移除 < 3% 的股票
            top_n = top_n[~below_min]

            if len(top_n) == 0:
                return pd.Series(0.0, index=row.index)

        # 最終結果
        total = top_n.sum()
        if total > 0:
            normalized = top_n / total
            result = pd.Series(0.0, index=row.index)
            result[normalized.index] = normalized.values
            return result

        return pd.Series(0.0, index=row.index)

    # 應用正規化
    position = combined_score.apply(normalize_to_position, axis=1)

    return position

def run_backtest_v4(position, params, start_date=None, end_date=None, name="Strategy", upload=False):
    """執行回測 v4.0"""
    try:
        if position is None or position.empty:
            return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1, 'win_rate': 0}

        if start_date:
            position = position.loc[start_date:]
        if end_date:
            position = position.loc[:end_date]

        if position.empty:
            return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1, 'win_rate': 0}

        report = sim(
            position,
            fee_ratio=1.425/1000,
            tax_ratio=3/1000,
            trade_at_price="high_low_avg",
            stop_loss=params.get('stop_loss', 0.20),
            trail_stop=params.get('trail_stop', 0.30),
            take_profit=params.get('take_profit', 0.60),
            position_limit=params.get('position_limit', 0.25),
            stop_trading_next_period=False,
            upload=upload,
            name=name
        )

        m = report.get_metrics()
        return {
            'sharpe': m['ratio'].get('sharpeRatio', 0) or 0,
            'capacity': m['liquidity'].get('capacity', 0) or 0,
            'annual_return': m['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(m['risk'].get('maxDrawdown', 1) or 1),
            'win_rate': m['profitability'].get('winRate', 0) or 0,
            'report': report
        }
    except Exception as e:
        print(f"   ⚠️ 回測錯誤: {e}")
        return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1, 'win_rate': 0}

#%% ========== 純夏普適應度函數 v4.0 ==========
def evaluate_pure_sharpe(individual):
    """
    🔥 v4.0 核心：純夏普優先適應度函數

    邏輯：
    1. 夏普值是唯一目標
    2. 回撤超標 -> 懲罰
    3. 胃納不足 -> 懲罰
    4. 持股太少 -> 懲罰
    """
    try:
        params = GeneDecoder.decode(individual)
        position = combine_strategies_v4(params)

        if position is None or position.empty:
            return (0.0,)

        # 檢查持股數量
        avg_stocks = position[position > 0].count(axis=1).mean()
        if avg_stocks < 5:
            return (0.0,)  # 持股太少，直接淘汰

        # 訓練期回測
        result = run_backtest_v4(position, params, start_date=BACKTEST_START, end_date=TRAIN_END, name="train")

        sharpe = result['sharpe']
        capacity = result['capacity']
        max_dd = result['max_drawdown']

        # 🔥 純夏普，但有硬性限制
        fitness = sharpe

        # 回撤懲罰 (超過 25% 開始懲罰)
        if max_dd > MAX_DRAWDOWN:
            penalty = (max_dd - MAX_DRAWDOWN) * 5
            fitness = fitness - penalty

        # 胃納懲罰 (低於 500 萬開始懲罰)
        if capacity < MIN_CAPACITY:
            penalty = (1 - capacity / MIN_CAPACITY) * 1.0
            fitness = fitness - penalty

        # 持股數量獎勵 (接近 15 檔加分)
        target_diff = abs(avg_stocks - TARGET_STOCKS)
        if target_diff < 3:
            fitness += 0.1  # 接近目標數量加分
        elif target_diff > 10:
            fitness -= 0.2  # 偏離太多扣分

        # 確保非負
        fitness = max(0.0, fitness)

        return (fitness,)
    except:
        return (0.0,)

#%% ========== DEAP 設定 v4.0 ==========
if 'FitnessSingle' in dir(creator):
    del creator.FitnessSingle
if 'Individual' in dir(creator):
    del creator.Individual

# 🔥 v4.0: 單目標優化 (純夏普)
creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessSingle)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_pure_sharpe)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1.0/GENE_LENGTH)
toolbox.register("select", tools.selTournament, tournsize=3)  # 錦標賽選擇

#%% ========== 精英存檔 v4.0 ==========
class EliteArchive:
    """精英存檔 (按夏普排序)"""

    def __init__(self, max_size=30):
        self.archive = []
        self.max_size = max_size
        self.load()

    def update(self, population):
        combined = self.archive + [ind for ind in population if ind.fitness.valid]
        if combined:
            # 按夏普排序
            combined.sort(key=lambda x: x.fitness.values[0], reverse=True)
            self.archive = combined[:self.max_size]

    def save(self):
        save_data = [(list(ind), ind.fitness.values) for ind in self.archive]
        with open(PARETO_FILE, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self):
        try:
            if os.path.exists(PARETO_FILE):
                with open(PARETO_FILE, 'rb') as f:
                    save_data = pickle.load(f)
                self.archive = []
                for genes, fitness in save_data:
                    ind = creator.Individual(genes)
                    ind.fitness.values = fitness
                    self.archive.append(ind)
                print(f"   📂 載入精英存檔 ({len(self.archive)} 個)")
        except:
            self.archive = []

    def get_best(self, n=5):
        return self.archive[:n]

#%% ========== 演化引擎 v4.0 ==========
class EvolutionEngineV4:
    """v4.0 演化引擎 - 純夏普優化"""

    def __init__(self):
        self.elite = EliteArchive()
        self.history = []
        self.start_gen = 0
        self.population = None
        self.load_checkpoint()

    def save_checkpoint(self, gen, population):
        checkpoint = {
            'generation': gen,
            'population': [(list(ind), ind.fitness.values if ind.fitness.valid else None) for ind in population],
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.elite.save()

        # 儲存最佳參數
        if self.elite.archive:
            best_genes = list(self.elite.get_best(1)[0])
            best_params = GeneDecoder.decode(best_genes)
            with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'genes': best_genes,
                    'params': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                              for k, v in best_params.items()}
                }, f, indent=2, ensure_ascii=False)

        print(f"   💾 Checkpoint 已儲存 (第 {gen} 代)")

    def load_checkpoint(self):
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.start_gen = checkpoint['generation'] + 1
                self.history = checkpoint.get('history', [])
                self.population = []
                for genes, fitness in checkpoint['population']:
                    ind = creator.Individual(genes)
                    if fitness:
                        ind.fitness.values = fitness
                    self.population.append(ind)
                print(f"   📂 載入 Checkpoint (從第 {self.start_gen} 代繼續)")
                return True
        except Exception as e:
            print(f"   ⚠️ 無法載入 Checkpoint: {e}")
        return False

    def detailed_backtest(self, gen):
        """詳細回測前 5 名"""
        print(f"\n{'='*70}")
        print(f"📊 第 {gen} 代 - 詳細回測 (純夏普優化)")
        print('='*70)

        best_individuals = self.elite.get_best(5)

        for rank, ind in enumerate(best_individuals, 1):
            params = GeneDecoder.decode(ind)
            position = combine_strategies_v4(params)

            if position is None or position.empty:
                continue

            # 檢查持股比例
            avg_stocks = position[position > 0].count(axis=1).mean()

            # 檢查權重分布
            weights = position[position > 0].mean()
            if len(weights) > 0:
                max_weight = weights.max()
                min_weight = weights[weights > 0].min() if len(weights[weights > 0]) > 0 else 0
            else:
                max_weight = 0
                min_weight = 0

            train_result = run_backtest_v4(position, params, start_date=BACKTEST_START, end_date=TRAIN_END, name=f"R{rank}_Train")
            test_result = run_backtest_v4(position, params, start_date=TEST_START, name=f"R{rank}_Test")

            print(f"\n🏆 第 {rank} 名 (夏普: {ind.fitness.values[0]:.2f})")
            print(f"   持股: {avg_stocks:.1f} 檔 | 權重: {min_weight*100:.1f}%~{max_weight*100:.1f}%")
            print(f"   [訓練期] 夏普: {train_result['sharpe']:.2f}, 年化: {train_result['annual_return']*100:.1f}%, 回撤: {train_result['max_drawdown']*100:.1f}%")
            print(f"   [測試期] 夏普: {test_result['sharpe']:.2f}, 年化: {test_result['annual_return']*100:.1f}%, 回撤: {test_result['max_drawdown']*100:.1f}%")

            # 顯示策略權重
            print(f"   策略權重: ", end="")
            for i in range(9):
                w = params.get(f'w{i+1}', 1/9)
                print(f"S{i+1}={w*100:.0f}% ", end="")
            print()

        print('='*70)

    def run(self, n_generations):
        """執行演化"""
        print(f"\n{'='*70}")
        print(f"🧬 純夏普優化演化 v4.0 (Window {WINDOW_ID})")
        print(f"   目標: 夏普>={TARGET_SHARPE}, 持股~{TARGET_STOCKS}檔, 每檔>={MIN_POSITION_RATIO*100:.0f}%")
        print('='*70)

        if self.population is None:
            print("\n📦 初始化族群...")
            self.population = toolbox.population(n=POPULATION_SIZE)
            if self.elite.archive:
                n_inject = min(ELITE_SIZE, len(self.elite.archive))
                for i in range(n_inject):
                    self.population[i] = creator.Individual(self.elite.archive[i])
                print(f"   注入 {n_inject} 個歷史精英")

        print("\n🔄 評估初始族群...")
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = evaluate_pure_sharpe(ind)

        self.elite.update(self.population)

        for gen in range(self.start_gen, n_generations):
            start_time = time.time()

            # 錦標賽選擇
            offspring = toolbox.select(self.population, POPULATION_SIZE)
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
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = evaluate_pure_sharpe(ind)

            # 精英保留
            elite = tools.selBest(self.population, ELITE_SIZE)
            offspring.extend(elite)

            # 選擇下一代
            self.population = tools.selBest(offspring, POPULATION_SIZE)
            self.elite.update(self.population)

            # 統計
            fits = [ind.fitness.values[0] for ind in self.population]
            elapsed = time.time() - start_time

            self.history.append({
                'gen': gen,
                'best_sharpe': max(fits),
                'avg_sharpe': np.mean(fits),
                'std_sharpe': np.std(fits)
            })

            print(f"[Gen {gen:3d}] 夏普: {max(fits):5.2f} (avg {np.mean(fits):.2f}, std {np.std(fits):.2f}) | {elapsed:.1f}s")

            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(gen, self.population)

            if (gen + 1) % DETAILED_BACKTEST_INTERVAL == 0:
                self.detailed_backtest(gen)

        self.save_checkpoint(n_generations - 1, self.population)
        print(f"\n✅ 演化完成！共 {n_generations} 代")
        return self.population, self.elite.archive

#%% ========== 主程式 ==========
if EVALUATE_MODE:
    print("\n📊 評估模式：回測歷史前 5 名")
    elite = EliteArchive()
    if elite.archive:
        engine = EvolutionEngineV4()
        engine.detailed_backtest(0)

        best_ind = elite.get_best(1)[0]
        params = GeneDecoder.decode(best_ind)
        position = combine_strategies_v4(params)

        print("\n📈 最佳策略完整回測 (2017~至今):")
        result = run_backtest_v4(position, params, start_date=BACKTEST_START, name="天空龍_v4_Best", upload=True)
        if 'report' in result and result['report']:
            result['report'].display()
    else:
        print("⚠️ 找不到精英存檔")

else:
    # 演化模式
    elite = EliteArchive()
    if elite.archive:
        print("\n" + "="*70)
        print("📊 先完整回測歷史最佳基因...")
        print("="*70)

        best_ind = elite.get_best(1)[0]
        print(f"   歷史最佳夏普: {best_ind.fitness.values[0]:.4f}")

        try:
            params = GeneDecoder.decode(best_ind)
            position = combine_strategies_v4(params)

            print(f"\n📈 歷史最佳完整回測:")
            result = run_backtest_v4(
                position, params,
                start_date=BACKTEST_START,
                name=f"天空龍_v4_視窗{WINDOW_ID}_歷史最佳",
                upload=True
            )
            if result and result.get('report'):
                result['report'].display()
        except Exception as e:
            print(f"   ⚠️ 歷史最佳回測失敗: {e}")

        print("\n" + "="*70)

    engine = EvolutionEngineV4()
    population, elite_archive = engine.run(n_generations=EVOLUTION_GENERATIONS)

    # 最終回測
    engine.detailed_backtest(EVOLUTION_GENERATIONS)

    # 顯示最佳策略配置
    if engine.elite.archive:
        best_ind = engine.elite.get_best(1)[0]
        params = GeneDecoder.decode(best_ind)

        print("\n📊 最佳策略配置:")
        names = ['低波動價值', '小型成長', '營收雙渦輪', '高殖利率', '低波動穩健', '創高突破', '財務品質', '技術動能', '綜合品質']
        for i, name in enumerate(names):
            w = params.get(f'w{i+1}', 0)
            bar = '█' * int(w * 40)
            print(f"   {name:<12}: {w*100:5.1f}% {bar}")

        print(f"\n   目標持股: {params.get('n_stocks', TARGET_STOCKS)} 檔")
        print(f"   停損: {params.get('stop_loss', 0.2)*100:.0f}%")
        print(f"   移動停損: {params.get('trail_stop', 0.3)*100:.0f}%")
        print(f"   停利: {params.get('take_profit', 0.6)*100:.0f}%")

print(f"\n✅ 九組合天空龍 純夏普優化 v4.0 完成！")
print(f"📁 結果儲存: {BASE_DIR}")
