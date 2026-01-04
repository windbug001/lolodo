# =============================================================================
# 🐉 九組合媽媽龍策略 - 基因演算法持續優化版 v2.0
# =============================================================================
# 持股比例至少 3%，否則 0%
# 使用環境變數控制執行模式
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
BASE_DIR = os.environ.get('BASE_DIR', '/content/drive/MyDrive/九組合媽媽龍_GA')

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
    # 優先從環境變數取得
    api_key = os.environ.get('FINLAB_API_KEY', '')
    if api_key:
        return api_key
    # 嘗試從 Colab Secrets 取得
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
    # 嘗試使用 finlab 已登入的 session
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
║              🐉 九組合媽媽龍策略 - 基因演算法 v2.0                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FinLab: {finlab.__version__:<10}  CPU 核心: {mp.cpu_count():<5}                                    ║
║  WINDOW_ID: {WINDOW_ID:<5}  CONTINUE: {CONTINUE_EVOLUTION}  GENERATIONS: {EVOLUTION_GENERATIONS:<5}             ║
║  BASE_DIR: {BASE_DIR[:50]:<50}  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

#%% ========== 核心參數設定 ==========
# 🎯 2026 目標
TARGET_SHARPE = 4.2
TARGET_CAPACITY = 10_000_000  # 1000萬台幣
MAX_DRAWDOWN = 0.20           # 20%

# ⚠️ 重要：持股比例至少 3%，否則 0%
MIN_POSITION_RATIO = 0.03     # 每股至少 3%

# GA 設定
POPULATION_SIZE = 60
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.7
ELITE_SIZE = 10
CHECKPOINT_INTERVAL = 5       # 每 5 代存檔
DETAILED_BACKTEST_INTERVAL = 10  # 每 10 代詳細回測

# 回測時間設定
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
BACKTEST_START = '2017-01-01'

# 基因長度
GENE_LENGTH = 180

# 儲存路徑
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✅ Google Drive 已掛載")
except:
    print("⚠️ 本地模式")

Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = f"{BASE_DIR}/checkpoint_w{WINDOW_ID}.pkl"
PARETO_FILE = f"{BASE_DIR}/pareto_archive_w{WINDOW_ID}.pkl"
BEST_PARAMS_FILE = f"{BASE_DIR}/best_params_w{WINDOW_ID}.json"
HISTORY_FILE = f"{BASE_DIR}/history_w{WINDOW_ID}.json"

print(f"📁 儲存路徑: {BASE_DIR}")
print(f"📄 Checkpoint: checkpoint_w{WINDOW_ID}.pkl")

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
inventory = data.get("inventory")

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

print("✅ 數據載入完成")

#%% ========== 基因解碼器 ==========
class GeneDecoder:
    """將 180 個基因解碼為策略參數"""

    @staticmethod
    def decode(genes: List[float]) -> Dict[str, Any]:
        """解碼基因為參數字典"""
        genes = list(genes) + [0.5] * (GENE_LENGTH - len(genes))
        genes = genes[:GENE_LENGTH]
        p = {}

        # 策略權重 (0-8) - 確保加總為 1
        raw_w = [max(0.01, genes[i]) for i in range(9)]
        total = sum(raw_w)
        for i in range(9):
            p[f'w{i+1}'] = raw_w[i] / total

        # 策略1: 低波動本益比 (9-28) - 20個參數
        p['s1_rev_ratio'] = genes[9] * 0.5 + 1.0
        p['s1_volatility'] = genes[10] * 0.05 + 0.02
        p['s1_margin'] = genes[11] * 30 + 20
        p['s1_pe_min'] = genes[12] * 10 + 3
        p['s1_pe_max'] = genes[13] * 20 + 15
        p['s1_gpm'] = genes[14] * 15 + 5
        p['s1_roe'] = genes[15] * 15
        p['s1_ma1'] = int(genes[16] * 40 + 20)
        p['s1_ma2'] = int(genes[17] * 60 + 40)
        p['s1_ma3'] = int(genes[18] * 100 + 100)
        p['s1_vol'] = genes[19] * 200000 + 50000
        p['s1_top'] = int(genes[20] * 8 + 3)
        for i in range(21, 29):
            p[f's1_p{i-20}'] = genes[i]

        # 策略2: 小資族 (29-48) - 20個參數
        p['s2_cap'] = genes[29] * 20e9 + 5e9
        p['s2_ps'] = genes[30] * 4 + 1
        p['s2_rsv'] = int(genes[31] * 50 + 20)
        p['s2_ma1'] = int(genes[32] * 60 + 30)
        p['s2_ma2'] = int(genes[33] * 80 + 60)
        p['s2_vol'] = genes[34] * 200000 + 50000
        p['s2_top'] = int(genes[35] * 8 + 3)
        for i in range(36, 49):
            p[f's2_p{i-35}'] = genes[i]

        # 策略3: 營收雙渦輪 (49-68) - 20個參數
        p['s3_rev_ma'] = int(genes[49] * 6 + 2)
        p['s3_lookback'] = int(genes[50] * 20 + 10)
        p['s3_price_win'] = int(genes[51] * 15 + 5)
        p['s3_vol'] = genes[52] * 300000 + 100000
        p['s3_rsi'] = genes[53] * 30 + 50
        p['s3_gpm'] = genes[54] * 10 + 3
        p['s3_top'] = int(genes[55] * 15 + 5)
        for i in range(56, 69):
            p[f's3_p{i-55}'] = genes[i]

        # 策略4: 高殖利率 (69-84) - 16個參數
        p['s4_yield'] = genes[69] * 4 + 2
        p['s4_op'] = genes[70] * 15 + 5
        p['s4_boss'] = genes[71] * 25 + 10
        p['s4_sma1'] = int(genes[72] * 20 + 10)
        p['s4_sma2'] = int(genes[73] * 50 + 30)
        p['s4_vol'] = genes[74] * 200000 + 50000
        p['s4_top'] = int(genes[75] * 10 + 5)
        for i in range(76, 85):
            p[f's4_p{i-75}'] = genes[i]

        # 策略5: 低波動 (85-100) - 16個參數
        p['s5_std'] = int(genes[85] * 30 + 10)
        p['s5_pct'] = genes[86] * 0.4 + 0.1
        p['s5_ma1'] = int(genes[87] * 50 + 40)
        p['s5_ma2'] = int(genes[88] * 100 + 100)
        p['s5_vol'] = genes[89] * 200000 + 50000
        p['s5_top'] = int(genes[90] * 20 + 10)
        for i in range(91, 101):
            p[f's5_p{i-90}'] = genes[i]

        # 策略6: 大盤指針 (101-116) - 16個參數
        p['s6_high'] = int(genes[101] * 200 + 60)
        p['s6_vol_ma'] = int(genes[102] * 15 + 5)
        p['s6_vol'] = genes[103] * 200000 + 50000
        p['s6_top'] = int(genes[104] * 15 + 5)
        for i in range(105, 117):
            p[f's6_p{i-104}'] = genes[i]

        # 策略7: 小蝦米大鯨魚 (117-136) - 20個參數
        p['s7_small'] = int(genes[117] * 5 + 3)
        p['s7_big_min'] = int(genes[118] * 4 + 8)
        p['s7_big_max'] = int(genes[119] * 4 + 12)
        p['s7_ma'] = int(genes[120] * 200 + 150)
        p['s7_f1'] = int(genes[121] * 40 + 30)
        p['s7_f2'] = int(genes[122] * 30 + 20)
        p['s7_f3'] = int(genes[123] * 15 + 5)
        p['s7_top'] = int(genes[124] * 8 + 3)
        for i in range(125, 137):
            p[f's7_p{i-124}'] = genes[i]

        # 策略8: 技術趨勢 (137-156) - 20個參數
        p['s8_sma'] = int(genes[137] * 150 + 150)
        p['s8_zlma'] = int(genes[138] * 80 + 60)
        p['s8_wma'] = int(genes[139] * 50 + 30)
        p['s8_bias'] = genes[140] * 0.2 + 0.05
        p['s8_slope'] = genes[141] * 0.4 + 0.3
        p['s8_kurt'] = genes[142] * 0.4 + 0.2
        p['s8_vol'] = genes[143] * 300000 + 100000
        p['s8_top'] = int(genes[144] * 15 + 5)
        for i in range(145, 157):
            p[f's8_p{i-144}'] = genes[i]

        # 策略9: 財報20大 (157-173) - 17個參數
        p['s9_std'] = int(genes[157] * 50 + 30)
        p['s9_ma'] = int(genes[158] * 50 + 30)
        p['s9_pct'] = genes[159] * 0.4 + 0.2
        p['s9_vol'] = genes[160] * 300000 + 100000
        p['s9_top'] = int(genes[161] * 20 + 10)
        for i in range(162, 174):
            p[f's9_p{i-161}'] = genes[i]

        # 風控參數 (174-179) - 6個參數
        p['stop_loss'] = genes[174] * 0.2 + 0.1
        p['trail_stop'] = genes[175] * 0.3 + 0.15
        p['take_profit'] = genes[176] * 0.5 + 0.3
        p['min_vol'] = genes[177] * 200000 + 50000
        p['liq_days'] = int(genes[178] * 20 + 5)
        # 持股數量 (決定每股比例)
        # position_limit = 1/n，如果 n=33 則每股 ~3%
        p['n_stocks'] = int(genes[179] * 20 + 10)  # 10-30 檔

        return p

#%% ========== 九大策略函數 ==========
def strategy_1(p):
    """低波動本益比"""
    try:
        peg = pe / 營業利益成長率
        cond = ((rev_ma3/rev_ma12 > p['s1_rev_ratio']) &
                (volatility <= p['s1_volatility']) &
                (融資使用率 <= p['s1_margin']) &
                (vol.average(1) > p['s1_vol']) &
                (close > close.average(p['s1_ma1'])) &
                (close > close.average(p['s1_ma2'])) &
                (p['s1_pe_min'] <= pe) & (pe <= p['s1_pe_max']) &
                (營業毛利率 > p['s1_gpm']).sustain(2) &
                (ROE > p['s1_roe']).sustain(2) &
                ~limit_up_all_day)
        pos = peg[cond & (peg > 0)].is_smallest(p['s1_top'])
        return pos.reindex(rev.index_str_to_date().index, method='ffill')
    except:
        return pd.DataFrame()

def strategy_2(p):
    """小資族"""
    try:
        cond = ((市值 < p['s2_cap']) &
                (自由現金流 > 0) &
                (ROE_calc > 0) &
                (市值營收比 < p['s2_ps']) &
                (vol > p['s2_vol']) &
                (close > close.average(p['s2_ma1'])) &
                (close > close.average(p['s2_ma2'])) &
                (rev.average(3) > rev.average(12)))
        rsv = (close - close.rolling(p['s2_rsv']).min()) / (close.rolling(p['s2_rsv']).max() - close.rolling(p['s2_rsv']).min())
        pos = (cond * rsv).is_largest(p['s2_top'])
        return pos.reindex(當月營收.index_str_to_date().index)
    except:
        return pd.DataFrame()

def strategy_3(p):
    """營收雙渦輪"""
    try:
        rev_ma = rev.average(p['s3_rev_ma'])
        cond = ((rev_ma == rev_ma.rolling(p['s3_lookback'], min_periods=p['s3_rev_ma']).max()) &
                (close == close.rolling(260).max()).sustain(p['s3_price_win'], 1) &
                (vol.average(1) > p['s3_vol']) &
                (close > close.average(5)) & (close > close.average(20)) &
                (close > close.average(60)) & (close > close.average(200)) &
                (營業毛利率 > p['s3_gpm']).sustain(3) &
                (rsi > p['s3_rsi']).sustain(1) &
                ~limit_up_all_day)
        pos = (rev_yoy * cond)
        pos = pos[pos > 0].is_largest(p['s3_top'])
        return pos.reindex(rev.index_str_to_date().index, method="ffill")
    except:
        return pd.DataFrame()

def strategy_4(p):
    """高殖利率"""
    try:
        cond = ((dividend_yield >= p['s4_yield']) &
                (close > close.average(p['s4_sma1'])) &
                (close > close.average(p['s4_sma2'])) &
                (rev.average(3) > rev.average(12)) &
                (營業利益率 >= p['s4_op']) &
                (董監持股 >= p['s4_boss']) &
                (vol.average(5) >= p['s4_vol']))
        pos = ((cond * rev_yoy)[cond * rev_yoy > 0]).is_largest(p['s4_top'])
        return pos.reindex(rev.index_str_to_date().index, method='ffill')
    except:
        return pd.DataFrame()

def strategy_5(p):
    """低波動"""
    try:
        std = close.pct_change().rolling(p['s5_std']).std().rank(axis=1, pct=True)
        cond = ((vol.average(20) > p['s5_vol']) &
                (close > close.average(p['s5_ma1'])) &
                (close > close.average(p['s5_ma2'])) &
                (std < p['s5_pct']))
        pos = 市值[cond].is_smallest(p['s5_top'])
        return pos.reindex(close.index, method='ffill')
    except:
        return pd.DataFrame()

def strategy_6(p):
    """大盤指針"""
    try:
        vol_ma = vol.average(p['s6_vol_ma'])
        cond = ((close == close.rolling(p['s6_high']).max()) &
                ~(rev_yoy < -15).sustain(3) &
                ~(rev_yoy > 60).sustain(12, 8) &
                ((rev.rolling(12).min()/rev < 1.2).sustain(3)) &
                (rev_mom > -10).sustain(3) &
                (vol_ma > p['s6_vol']))
        pos = ((cond * vol_ma)[cond * vol_ma > 0]).is_smallest(p['s6_top'])
        return pos.reindex(rev.index_str_to_date().index, method='ffill')
    except:
        return pd.DataFrame()

def strategy_7(p):
    """小蝦米大鯨魚"""
    try:
        h1 = inventory[inventory.持股分級.astype(int) <= p['s7_small']].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
        h2 = inventory[(inventory.持股分級.astype(int) >= p['s7_big_min']) & (inventory.持股分級.astype(int) <= p['s7_big_max'])].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
        ratio = h2 / (h1 + h2)
        pos = (FinlabDataFrame(ratio).rank(axis=1, pct=True) * (close.notna() & (close > close.average(p['s7_ma']))) + (rev / rev.shift(12).rolling(2).mean()).rank(axis=1, pct=True) + (rev / rev.shift()).rank(axis=1, pct=True)).is_largest(p['s7_f1'])
        pos = (pos * 營業利益成長率).is_largest(p['s7_f2'])
        pos = (pos * dividend_yield).is_largest(p['s7_f3'])
        pos = (pos * ratio.diff(8)).is_largest(p['s7_top'])
        return pos.reindex(rev.index_str_to_date().index, method='ffill')
    except:
        return pd.DataFrame()

def strategy_8(p):
    """技術趨勢"""
    try:
        def wma(x, n): return x.ewm(com=n).mean()
        def zlma(x, n): return wma(2*x - x.shift((n-1)//2), n)
        cond = ((close/close.rolling(p['s8_sma']).mean()-1 > p['s8_bias']) &
                (close/close.apply(lambda s: zlma(s, p['s8_zlma']))-1 > p['s8_bias']) &
                (close/close.apply(lambda s: wma(s, p['s8_wma']))-1 > p['s8_bias']) &
                (close.rolling(60).mean().pipe(lambda df: df/df.shift(60)-1).rank(axis=1, pct=True) > p['s8_slope']) &
                (close.pct_change().rolling(250).kurt().rank(axis=1, pct=True) > p['s8_kurt']) &
                (vol > p['s8_vol']))
        return vol.average(10)[cond].is_smallest(p['s8_top'])
    except:
        return pd.DataFrame()

def strategy_9(p):
    """財報20大"""
    try:
        fs = [f.average(4).rank(axis=1, pct=True).fillna(0) for f in [營業利益成長率, ROE, 營業毛利率, 稅後淨利率, 營業利益率] if f is not None]
        if not fs: return pd.DataFrame()
        std = close.pct_change().rolling(p['s9_std']).std()
        score = sum(fs) + ((close > close.average(p['s9_ma'])).astype(float)*5 + (std.rank(pct=True, axis=1) < p['s9_pct']).astype(float)*5 + (vol.rolling(5).mean() > p['s9_vol']).astype(float)*10).reindex(sum(fs).index, method='ffill')
        pos = score.is_largest(p['s9_top'])
        return pos.reindex(close.loc[score.index[0]:].index, method='ffill')
    except:
        return pd.DataFrame()

#%% ========== 組合策略與回測 ==========
def combine_strategies(params: Dict) -> pd.DataFrame:
    """組合九個策略，確保持股比例 >= 3%"""
    strategies = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5,
                  strategy_6, strategy_7, strategy_8, strategy_9]
    positions = []

    for i, func in enumerate(strategies):
        try:
            pos = func(params)
            if pos is not None and not pos.empty:
                weight = params.get(f'w{i+1}', 1/9)
                positions.append(pos * weight)
        except:
            pass

    if not positions:
        return pd.DataFrame()

    combined = sum(positions)

    # ⚠️ 關鍵：確保每股至少 3%，否則設為 0（但保留原始相對權重）
    n_stocks = params.get('n_stocks', 20)
    max_stocks = int(1 / MIN_POSITION_RATIO)  # 最多 33 檔 (3% 下限)
    actual_n = min(n_stocks, max_stocks)

    def normalize_with_min_position(row):
        """
        保留原始權重，過濾掉低於 3% 的股票，重新正規化
        - 先選前 N 名
        - 正規化為 100%
        - 過濾掉 < 3% 的股票
        - 重新正規化，迭代直到所有股票都 >= 3%
        """
        row_numeric = pd.to_numeric(row, errors='coerce').fillna(0)
        valid = row_numeric[row_numeric > 0]

        if len(valid) == 0:
            return pd.Series(0.0, index=row.index)

        # 選前 N 名
        valid = valid.nlargest(min(actual_n, len(valid)))

        # 迭代過濾：移除 < 3% 的股票並重新正規化
        for _ in range(10):  # 最多迭代 10 次
            total = valid.sum()
            if total <= 0:
                return pd.Series(0.0, index=row.index)

            # 正規化
            normalized = valid / total

            # 檢查是否有 < 3% 的股票
            below_min = normalized < MIN_POSITION_RATIO
            if not below_min.any():
                # 所有股票都 >= 3%，完成
                result = pd.Series(0.0, index=row.index)
                result[normalized.index] = normalized.values
                return result

            # 移除 < 3% 的股票
            valid = valid[~below_min]

            if len(valid) == 0:
                return pd.Series(0.0, index=row.index)

        # 迭代完成後，返回結果
        total = valid.sum()
        if total > 0:
            normalized = valid / total
            result = pd.Series(0.0, index=row.index)
            result[normalized.index] = normalized.values
            return result
        return pd.Series(0.0, index=row.index)

    # 確保 combined 是數值型態
    combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0)
    combined = combined[combined > 0]
    if combined.empty:
        return pd.DataFrame()

    # 應用持股權重正規化（保留相對權重，過濾 < 3%）
    position = combined.apply(normalize_with_min_position, axis=1)

    return position

def run_backtest(position, params, start_date=None, end_date=None, name="Strategy", upload=False):
    """執行回測"""
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
            position_limit=MIN_POSITION_RATIO,  # ⚠️ 每股至少 3%
            stop_loss=params.get('stop_loss', 0.25),
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

#%% ========== 適應度函數 ==========
def evaluate(individual):
    """評估個體適應度 (多目標優化)"""
    try:
        params = GeneDecoder.decode(individual)
        position = combine_strategies(params)

        # 訓練期回測
        result = run_backtest(position, params, start_date=BACKTEST_START, end_date=TRAIN_END, name="train")

        sharpe = result['sharpe']
        capacity = result['capacity']
        max_dd = result['max_drawdown']

        # 多目標適應度
        f1 = sharpe
        f2 = np.log10(max(capacity, 1))
        f3 = 1 - max_dd

        return (f1, f2, f3)
    except:
        return (0, 0, 0)

#%% ========== DEAP 設定 ==========
if 'FitnessMulti' in dir(creator):
    del creator.FitnessMulti
if 'Individual' in dir(creator):
    del creator.Individual

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20, indpb=1.0/GENE_LENGTH)
toolbox.register("select", tools.selNSGA2)

#%% ========== Pareto Archive ==========
class ParetoArchive:
    """Pareto 前沿存檔"""

    def __init__(self, max_size=50):
        self.archive = []
        self.max_size = max_size
        self.load()

    def update(self, population):
        combined = self.archive + [ind for ind in population if ind.fitness.valid]
        if combined:
            self.archive = tools.selNSGA2(combined, min(len(combined), self.max_size))

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
                print(f"   📂 載入 Pareto Archive ({len(self.archive)} 個)")
        except:
            self.archive = []

    def get_best(self, n=5):
        sorted_archive = sorted(self.archive, key=lambda x: x.fitness.values[0], reverse=True)
        return sorted_archive[:n]

#%% ========== 演化引擎 ==========
class EvolutionEngine:
    """基因演算法演化引擎"""

    def __init__(self):
        self.pareto = ParetoArchive()
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
        self.pareto.save()

        # 儲存最佳參數
        if self.pareto.archive:
            best_genes = list(self.pareto.get_best(1)[0])
            best_params = GeneDecoder.decode(best_genes)
            with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump({'genes': best_genes, 'params': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in best_params.items()}}, f, indent=2, ensure_ascii=False)

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
        print(f"📊 第 {gen} 代 - 詳細回測 (樣本內 + 樣本外)")
        print('='*70)

        best_individuals = self.pareto.get_best(5)

        for rank, ind in enumerate(best_individuals, 1):
            params = GeneDecoder.decode(ind)
            position = combine_strategies(params)

            if position is None or position.empty:
                continue

            # 檢查持股比例
            avg_stocks = position[position > 0].count(axis=1).mean()
            avg_weight = position[position > 0].mean().mean()

            train_result = run_backtest(position, params, start_date=BACKTEST_START, end_date=TRAIN_END, name=f"R{rank}_Train")
            test_result = run_backtest(position, params, start_date=TEST_START, name=f"R{rank}_Test")

            print(f"\n🏆 第 {rank} 名 (平均持股 {avg_stocks:.0f} 檔, 每股 {avg_weight*100:.1f}%):")
            print(f"   [訓練期 ~{TRAIN_END}] 夏普: {train_result['sharpe']:.2f}, 年化: {train_result['annual_return']*100:.1f}%, 回撤: {train_result['max_drawdown']*100:.1f}%, 胃納: {train_result['capacity']/1e4:.0f}萬")
            print(f"   [測試期 {TEST_START}~] 夏普: {test_result['sharpe']:.2f}, 年化: {test_result['annual_return']*100:.1f}%, 回撤: {test_result['max_drawdown']*100:.1f}%, 胃納: {test_result['capacity']/1e4:.0f}萬")

            if (test_result['sharpe'] >= TARGET_SHARPE and
                test_result['capacity'] >= TARGET_CAPACITY and
                test_result['max_drawdown'] <= MAX_DRAWDOWN):
                print(f"   🎉 達到 2026 目標！")

        print('='*70)

    def run(self, n_generations):
        """執行演化"""
        print(f"\n{'='*70}")
        print(f"🧬 基因演算法演化 (Window {WINDOW_ID})")
        print(f"   目標: 夏普>={TARGET_SHARPE}, 胃納>={TARGET_CAPACITY/1e4:.0f}萬, 回撤<={MAX_DRAWDOWN*100:.0f}%")
        print(f"   族群: {POPULATION_SIZE}, 代數: {n_generations}, 持股下限: {MIN_POSITION_RATIO*100:.0f}%")
        print('='*70)

        if self.population is None:
            print("\n📦 初始化族群...")
            self.population = toolbox.population(n=POPULATION_SIZE)
            if self.pareto.archive:
                n_inject = min(ELITE_SIZE, len(self.pareto.archive))
                for i in range(n_inject):
                    self.population[i] = creator.Individual(self.pareto.archive[i])
                print(f"   注入 {n_inject} 個歷史精英")

        print("\n🔄 評估初始族群...")
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = evaluate(ind)

        self.pareto.update(self.population)

        for gen in range(self.start_gen, n_generations):
            start_time = time.time()

            offspring = toolbox.select(self.population, POPULATION_SIZE)
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = evaluate(ind)

            elite = tools.selBest(self.population, ELITE_SIZE)
            offspring.extend(elite)
            self.population = toolbox.select(offspring, POPULATION_SIZE)
            self.pareto.update(self.population)

            fits = [ind.fitness.values for ind in self.population]
            sharpes = [f[0] for f in fits]
            capacities = [10**f[1] for f in fits]
            drawdowns = [1-f[2] for f in fits]

            elapsed = time.time() - start_time

            self.history.append({
                'gen': gen,
                'best_sharpe': max(sharpes),
                'avg_sharpe': np.mean(sharpes),
                'best_capacity': max(capacities),
                'min_drawdown': min(drawdowns)
            })

            print(f"[Gen {gen:3d}] 夏普: {max(sharpes):5.2f} (avg {np.mean(sharpes):.2f}) | 胃納: {max(capacities)/1e4:5.0f}萬 | 回撤: {min(drawdowns)*100:4.1f}% | {elapsed:.1f}s")

            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(gen, self.population)

            if (gen + 1) % DETAILED_BACKTEST_INTERVAL == 0:
                self.detailed_backtest(gen)

        self.save_checkpoint(n_generations - 1, self.population)
        print(f"\n✅ 演化完成！共 {n_generations} 代")
        return self.population, self.pareto.archive

#%% ========== 主程式 ==========
if EVALUATE_MODE:
    # 評估模式：回測歷史前 5 名
    print("\n📊 評估模式：回測歷史前 5 名")
    pareto = ParetoArchive()
    if pareto.archive:
        engine = EvolutionEngine()
        engine.detailed_backtest(0)

        # 顯示最佳策略完整回測
        best_ind = pareto.get_best(1)[0]
        params = GeneDecoder.decode(best_ind)
        position = combine_strategies(params)

        print("\n📈 最佳策略完整回測 (2017~至今):")
        result = run_backtest(position, params, start_date=BACKTEST_START, name="九組合媽媽龍_Best", upload=True)
        if 'report' in result and result['report']:
            result['report'].display()
    else:
        print("⚠️ 找不到 Pareto Archive")

else:
    # 演化模式

    # 🔥 先完整回測歷史最佳
    pareto = ParetoArchive()
    if pareto.archive:
        print("\n" + "="*70)
        print("📊 先完整回測歷史最佳基因...")
        print("="*70)

        best_ind = pareto.get_best(1)[0]
        best_fitness = sum(best_ind.fitness.values) if hasattr(best_ind, 'fitness') else 0
        print(f"   歷史最佳適應度: {best_fitness:.4f}")

        try:
            params = GeneDecoder.decode(best_ind)
            position = combine_strategies(params)

            # 🔥 上傳到 FinLab 並顯示官方報告
            print(f"\n📈 歷史最佳完整回測 (已上傳到 FinLab):")
            result = run_backtest(
                position, params,
                start_date=BACKTEST_START,
                name=f"媽媽龍_視窗{WINDOW_ID}_歷史最佳",
                upload=True
            )
            if result and result.get('report'):
                result['report'].display()  # 🔥 顯示官方樣式報告
                print(f"\n✅ 已上傳: 媽媽龍_視窗{WINDOW_ID}_歷史最佳")
        except Exception as e:
            print(f"   ⚠️ 歷史最佳回測失敗: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*70)
        print("🚀 開始演化優化...")
        print("="*70 + "\n")

    engine = EvolutionEngine()
    population, pareto_front = engine.run(n_generations=EVOLUTION_GENERATIONS)

    # 最終回測
    engine.detailed_backtest(EVOLUTION_GENERATIONS)

    # 顯示最佳策略權重
    if engine.pareto.archive:
        best_ind = engine.pareto.get_best(1)[0]
        params = GeneDecoder.decode(best_ind)
        print("\n📊 最佳策略權重配置:")
        names = ['低波動PE', '小資族', '營收雙渦輪', '高殖利率', '低波動', '大盤指針', '小蝦米鯨魚', '技術趨勢', '財報20大']
        for i, name in enumerate(names):
            w = params.get(f'w{i+1}', 0)
            bar = '█' * int(w * 40)
            print(f"   {name:<12}: {w*100:5.1f}% {bar}")

        print(f"\n   持股數量: {params.get('n_stocks', 20)} 檔 (每股 >= {100/params.get('n_stocks', 20):.1f}%)")

print(f"\n✅ 九組合媽媽龍 GA v2.0 完成！")
print(f"📁 結果儲存: {BASE_DIR}")
