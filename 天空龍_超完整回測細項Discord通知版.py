# =============================================================================
# 🐉 九組合天空龍 v4.0 - 超完整回測細項 Discord 通知版
# =============================================================================
# 功能：
# ✅ 1. 載入歷史最佳基因，執行完整回測
# ✅ 2. 詳細回測報告（月報酬、季報酬、年報酬）
# ✅ 3. 持股明細與權重分布
# ✅ 4. Discord Webhook 通知
# ✅ 5. 訓練期 vs 測試期比較
# =============================================================================

#%% ========== 安裝套件 ==========
import subprocess
import sys

try:
    import finlab
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "finlab", "requests", "-q"])

#%% ========== 環境設定 ==========
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('future.no_silent_downcasting', True)

# ===== 環境變數 =====
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))
BASE_DIR = os.environ.get('BASE_DIR', '/content/drive/MyDrive/九組合天空龍_GA_v4')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
UPLOAD_TO_FINLAB = os.environ.get('UPLOAD_TO_FINLAB', 'true').lower() == 'true'

os.environ['FINLAB_DISABLE_CACHE'] = '1'

# 檔案路徑
PARETO_FILE = f"{BASE_DIR}/pareto_v4_w{WINDOW_ID}.pkl"
BEST_PARAMS_FILE = f"{BASE_DIR}/best_params_v4_w{WINDOW_ID}.json"

#%% ========== FinLab 登入 ==========
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
        print("⚠️ 請設定 FINLAB_API_KEY")

from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        🐉 九組合天空龍 v4.0 - 超完整回測細項 Discord 通知版                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 完整回測 + 詳細報告 + Discord 通知                                        ║
║  📁 資料來源: {BASE_DIR:<52} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

#%% ========== 回測時間設定 ==========
TRAIN_START = '2017-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = datetime.now().strftime('%Y-%m-%d')

# 持股下限
MIN_POSITION_RATIO = 0.03
TARGET_STOCKS = 15

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

# 漲停判斷
limit_up = (close > close.shift(1) * 1.095)
limit_up_all_day = ((close * close).replace(0.0, np.nan) == (close * high).replace(0.0, np.nan)).replace(False, np.nan)
limit_up_all_day = (limit_up_all_day == limit_up).fillna(False)

# 流動性過濾
MIN_VOL_FILTER = 100000
liquid_mask = vol.average(20) > MIN_VOL_FILTER

print("✅ 數據載入完成")

#%% ========== 基因解碼器 ==========
GENE_LENGTH = 100

class GeneDecoder:
    @staticmethod
    def decode(genes: List[float]) -> Dict[str, Any]:
        genes = list(genes) + [0.5] * (GENE_LENGTH - len(genes))
        genes = genes[:GENE_LENGTH]
        p = {}

        # 策略權重
        raw_w = [max(0.05, genes[i]) for i in range(9)]
        total = sum(raw_w)
        for i in range(9):
            p[f'w{i+1}'] = raw_w[i] / total

        # 篩選強度
        for i in range(9):
            p[f's{i+1}_strictness'] = genes[9+i] * 0.8 + 0.1
            p[f's{i+1}_score_weight'] = genes[18+i] * 2.0 + 0.5

        # 技術面參數
        p['ma_short'] = int(genes[27] * 40 + 5)
        p['ma_mid'] = int(genes[28] * 100 + 20)
        p['ma_long'] = int(genes[29] * 200 + 60)
        p['rsi_low'] = genes[30] * 30 + 20
        p['rsi_high'] = genes[31] * 30 + 60
        p['atr_mult'] = genes[32] * 2.0 + 0.5
        p['price_momentum'] = int(genes[33] * 60 + 20)
        p['vol_ma'] = int(genes[34] * 20 + 5)
        p['breakout_period'] = int(genes[35] * 200 + 60)

        # 基本面參數
        p['pe_min'] = genes[36] * 10 + 3
        p['pe_max'] = genes[37] * 30 + 15
        p['pb_max'] = genes[38] * 5 + 1
        p['roe_min'] = genes[39] * 20
        p['gpm_min'] = genes[40] * 30
        p['rev_growth_min'] = genes[41] * 50 - 20
        p['opm_min'] = genes[42] * 20
        p['fcf_positive'] = genes[43] > 0.5
        p['dividend_min'] = genes[44] * 5

        # 籌碼面參數
        p['margin_max'] = genes[45] * 40 + 10
        p['director_min'] = genes[46] * 30
        p['foreign_trend'] = int(genes[47] * 20 + 5)
        p['trust_trend'] = int(genes[48] * 20 + 5)
        p['chip_score_weight'] = genes[49] * 1.5 + 0.5

        # 流動性參數
        p['min_vol'] = genes[54] * 300000 + 50000
        p['min_cap'] = genes[55] * 5e9 + 1e9
        p['max_cap'] = genes[56] * 200e9 + 10e9
        p['vol_spike'] = genes[57] * 2.0 + 1.0
        p['liquidity_score_weight'] = genes[58] * 1.5 + 0.5

        # 動能參數
        p['momentum_window'] = int(genes[63] * 60 + 20)
        p['momentum_threshold'] = genes[64] * 0.3
        p['trend_strength'] = genes[65] * 0.5 + 0.3
        p['volatility_max'] = genes[66] * 0.1 + 0.02
        p['momentum_score_weight'] = genes[67] * 1.5 + 0.5

        # 價值參數
        p['value_pe_weight'] = genes[72] * 1.5 + 0.5
        p['value_pb_weight'] = genes[73] * 1.5 + 0.5
        p['value_div_weight'] = genes[74] * 1.5 + 0.5
        p['value_fcf_weight'] = genes[75] * 1.5 + 0.5
        p['value_growth_weight'] = genes[76] * 1.5 + 0.5
        p['value_score_weight'] = genes[77] * 1.5 + 0.5

        # 風控參數
        p['stop_loss'] = genes[81] * 0.15 + 0.10
        p['trail_stop'] = genes[82] * 0.25 + 0.15
        p['take_profit'] = genes[83] * 0.5 + 0.3
        p['position_limit'] = genes[84] * 0.20 + 0.15
        p['drawdown_exit'] = genes[85] * 0.15 + 0.15
        p['correlation_limit'] = genes[86] * 0.3 + 0.5

        # 全局參數
        p['n_stocks'] = int(genes[90] * 10 + 10)
        p['rebalance_threshold'] = genes[91] * 0.15 + 0.05
        p['score_decay'] = genes[92] * 0.3 + 0.7
        p['overlap_bonus'] = genes[93] * 1.5 + 1.0
        p['diversity_weight'] = genes[94] * 0.5

        return p

#%% ========== 九大評分策略 ==========
def score_strategy_1(p):
    """低波動價值股"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol']) & ~limit_up_all_day
        pe_valid = (pe > p['pe_min']) & (pe < p['pe_max'])
        pe_score = (1 - (pe - p['pe_min']) / (p['pe_max'] - p['pe_min'])).clip(0, 1).where(pe_valid, 0)
        vol_score = (1 - volatility / p['volatility_max']).clip(0, 1)
        rev_score = ((rev_ma3 / rev_ma12 - 1) / 0.5 + 0.5).clip(0, 1)
        gpm_score = (營業毛利率 / 50).clip(0, 1)
        roe_score = (ROE / 30).clip(0, 1)
        total = (pe_score * 0.25 + vol_score * 0.25 + rev_score * 0.20 + gpm_score * 0.15 + roe_score * 0.15) * p['s1_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_2(p):
    """小型成長股"""
    try:
        base_cond = liquid_mask & (市值 < p['max_cap']) & (vol.average(20) > p['min_vol'])
        cap_score = (1 - 市值 / p['max_cap']).clip(0, 1).where(市值 > p['min_cap'], 0)
        fcf_score = (自由現金流 > 0).astype(float) * (1.5 if p['fcf_positive'] else 1.0)
        ps_score = (1 - 市值營收比 / 5).clip(0, 1)
        roe_score = (ROE_calc * 10).clip(0, 1)
        rsv = (close - close.rolling(p['momentum_window']).min()) / (close.rolling(p['momentum_window']).max() - close.rolling(p['momentum_window']).min())
        total = (cap_score * 0.20 + fcf_score * 0.25 + ps_score * 0.20 + roe_score * 0.20 + rsv.clip(0, 1) * 0.15) * p['s2_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_3(p):
    """營收雙渦輪"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol']) & ~limit_up_all_day
        rev_ma = rev.average(3)
        rev_high = (rev_ma == rev_ma.rolling(12, min_periods=3).max()).astype(float)
        price_high = (close == close.rolling(p['breakout_period']).max()).astype(float) * 0.8 + (close > close.average(p['ma_long'])).astype(float) * 0.2
        rev_yoy_score = (rev_yoy / 100 + 0.5).clip(0, 1)
        gpm_score = (營業毛利率.rolling(4).std() < 5).astype(float) * 0.5 + (營業毛利率 / 40).clip(0, 0.5)
        rsi_score = ((rsi - 30) / 40).clip(0, 1)
        total = (rev_high * 0.30 + price_high * 0.25 + rev_yoy_score * 0.20 + gpm_score * 0.15 + rsi_score * 0.10) * p['s3_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_4(p):
    """高殖利率價值股"""
    try:
        base_cond = liquid_mask & (dividend_yield > 0) & (vol.average(20) > p['min_vol'])
        div_score = (dividend_yield / 10).clip(0, 1)
        director_score = (董監持股 / 50).clip(0, 1)
        opm_score = (營業利益率 / 30).clip(0, 1)
        rev_trend = (rev.average(3) > rev.average(12)).astype(float)
        ma_trend = ((close > close.average(p['ma_short'])) & (close > close.average(p['ma_mid']))).astype(float)
        total = (div_score * 0.30 + director_score * 0.20 + opm_score * 0.20 + rev_trend * 0.15 + ma_trend * 0.15) * p['s4_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_5(p):
    """低波動穩健股"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol'])
        std_rank = close.pct_change().rolling(60).std().rank(axis=1, pct=True)
        vol_score = (1 - std_rank).clip(0, 1)
        ma_bull = (close > close.average(p['ma_short'])).astype(float) * 0.3 + (close > close.average(p['ma_mid'])).astype(float) * 0.3 + (close > close.average(p['ma_long'])).astype(float) * 0.4
        cap_score = (1 - 市值.rank(axis=1, pct=True)).clip(0, 1).where(市值 > p['min_cap'], 0)
        margin_score = (1 - 融資使用率 / p['margin_max']).clip(0, 1)
        vol_std = vol.rolling(20).std() / vol.rolling(20).mean()
        vol_stable = (1 - vol_std / 2).clip(0, 1)
        total = (vol_score * 0.30 + ma_bull * 0.25 + cap_score * 0.20 + margin_score * 0.15 + vol_stable * 0.10) * p['s5_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_6(p):
    """創高突破股"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol']) & ~limit_up_all_day
        high_260 = close.rolling(260).max()
        high_score = ((close / high_260).clip(0.8, 1) - 0.8) / 0.2
        vol_ma = vol.average(p['vol_ma'])
        vol_score = ((vol / vol_ma - 0.5) / 2).clip(0, 1)
        rev_mom_score = ((rev_mom + 20) / 40).clip(0, 1)
        rev_yoy_score = ((rev_yoy + 30) / 60).clip(0, 1)
        ma_support = (close > close.average(60)).astype(float)
        total = (high_score * 0.35 + vol_score * 0.20 + rev_mom_score * 0.15 + rev_yoy_score * 0.15 + ma_support * 0.15) * p['s6_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_7(p):
    """財務品質股"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol'])
        scores = []
        for f in [營業利益成長率, ROE, 營業毛利率, 稅後淨利率, 營業利益率]:
            if f is not None:
                scores.append(f.rank(axis=1, pct=True).fillna(0))
        if not scores:
            return pd.DataFrame()
        finance = sum(scores) / len(scores)
        vol_adj = (1 - close.pct_change().rolling(60).std().rank(axis=1, pct=True)).fillna(0.5) * 0.3
        ma_adj = (close > close.average(p['ma_mid'])).astype(float) * 0.2
        total = (finance * 0.7 + vol_adj + ma_adj) * p['s7_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_8(p):
    """技術動能股"""
    try:
        base_cond = liquid_mask & (vol > p['min_vol'])
        sma_score = (close / close.rolling(p['ma_long']).mean() - 1).pipe(lambda x: (x / 0.3 + 0.5).clip(0, 1))
        mom_rank = (close / close.shift(p['momentum_window']) - 1).rank(axis=1, pct=True).fillna(0.5)
        ma60 = close.rolling(60).mean()
        slope_score = ((ma60 / ma60.shift(60) - 1) / 0.3 + 0.5).clip(0, 1)
        vol_trend = (vol.rolling(20).mean() / vol.rolling(60).mean() - 0.5).clip(0, 1)
        std_rank = close.pct_change().rolling(60).std().rank(axis=1, pct=True)
        vol_mid = 1 - abs(std_rank - 0.5) * 2
        total = (sma_score * 0.25 + mom_rank * 0.25 + slope_score * 0.20 + vol_trend * 0.15 + vol_mid * 0.15) * p['s8_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

def score_strategy_9(p):
    """綜合品質動能股"""
    try:
        base_cond = liquid_mask & (vol.average(20) > p['min_vol'])
        fs = []
        for f in [營業利益成長率, ROE, 營業毛利率, 稅後淨利率, 營業利益率]:
            if f is not None:
                fs.append(f.average(4).rank(axis=1, pct=True).fillna(0))
        finance = sum(fs) / len(fs) if fs else pd.DataFrame(0.5, index=close.index, columns=close.columns)
        ma_bull = (close > close.average(20)).astype(float) * 0.2 + (close > close.average(60)).astype(float) * 0.3 + (close > close.average(120)).astype(float) * 0.5
        vol_low = (1 - close.pct_change().rolling(60).std().rank(axis=1, pct=True)).fillna(0.5)
        vol_rank = vol.rolling(20).mean().rank(axis=1, pct=True).fillna(0.5)
        margin_safe = (融資使用率 < p['margin_max']).astype(float)
        total = (finance * 0.30 + ma_bull * 0.25 + vol_low * 0.20 + vol_rank * 0.15 + margin_safe * 0.10) * p['s9_score_weight']
        return total.where(base_cond, 0).fillna(0)
    except:
        return pd.DataFrame()

#%% ========== 組合策略 ==========
def combine_strategies(params: Dict) -> pd.DataFrame:
    strategies = [score_strategy_1, score_strategy_2, score_strategy_3,
                  score_strategy_4, score_strategy_5, score_strategy_6,
                  score_strategy_7, score_strategy_8, score_strategy_9]
    scores = []
    for i, func in enumerate(strategies):
        try:
            score = func(params)
            if score is not None and not score.empty:
                weight = params.get(f'w{i+1}', 1/9)
                scores.append(score * weight)
        except:
            pass

    if not scores:
        return pd.DataFrame()

    combined = reduce(lambda a, b: a.add(b, fill_value=0), scores)
    combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0)

    n_stocks = params.get('n_stocks', TARGET_STOCKS)
    max_stocks = int(1 / MIN_POSITION_RATIO)
    n_stocks = min(n_stocks, max_stocks)

    def normalize(row):
        row_numeric = pd.to_numeric(row, errors='coerce').fillna(0)
        valid = row_numeric[row_numeric > 0]
        if len(valid) == 0:
            return pd.Series(0.0, index=row.index)
        top_n = valid.nlargest(min(n_stocks, len(valid)))
        for _ in range(10):
            total = top_n.sum()
            if total <= 0:
                return pd.Series(0.0, index=row.index)
            normalized = top_n / total
            below_min = normalized < MIN_POSITION_RATIO
            if not below_min.any():
                result = pd.Series(0.0, index=row.index)
                result[normalized.index] = normalized.values
                return result
            top_n = top_n[~below_min]
            if len(top_n) == 0:
                return pd.Series(0.0, index=row.index)
        total = top_n.sum()
        if total > 0:
            result = pd.Series(0.0, index=row.index)
            result[top_n.index] = (top_n / total).values
            return result
        return pd.Series(0.0, index=row.index)

    return combined.apply(normalize, axis=1)

#%% ========== Discord 通知 ==========
def send_discord_notification(content: str, embed: dict = None):
    """發送 Discord 通知"""
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ 未設定 DISCORD_WEBHOOK_URL，跳過通知")
        return False

    try:
        import requests
        payload = {"content": content}
        if embed:
            payload["embeds"] = [embed]

        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 204:
            print("✅ Discord 通知已發送")
            return True
        else:
            print(f"⚠️ Discord 通知失敗: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Discord 通知錯誤: {e}")
        return False

def format_backtest_embed(title: str, result: dict, position: pd.DataFrame, params: dict) -> dict:
    """格式化回測結果為 Discord Embed"""

    # 計算持股統計
    avg_stocks = position[position > 0].count(axis=1).mean()
    weights = position[position > 0].mean()
    max_weight = weights.max() if len(weights) > 0 else 0
    min_weight = weights[weights > 0].min() if len(weights[weights > 0]) > 0 else 0

    # 顏色依夏普值決定
    sharpe = result.get('sharpe', 0)
    if sharpe >= 2.0:
        color = 0x00FF00  # 綠色
    elif sharpe >= 1.0:
        color = 0xFFFF00  # 黃色
    else:
        color = 0xFF0000  # 紅色

    # 策略權重文字
    strategy_names = ['低波動價值', '小型成長', '營收雙渦輪', '高殖利率', '低波動穩健', '創高突破', '財務品質', '技術動能', '綜合品質']
    weight_text = "\n".join([f"S{i+1} {strategy_names[i]}: {params.get(f'w{i+1}', 0)*100:.1f}%" for i in range(9)])

    embed = {
        "title": f"🐉 {title}",
        "color": color,
        "fields": [
            {"name": "📈 夏普值", "value": f"{sharpe:.2f}", "inline": True},
            {"name": "💰 年化報酬", "value": f"{result.get('annual_return', 0)*100:.1f}%", "inline": True},
            {"name": "📉 最大回撤", "value": f"{result.get('max_drawdown', 0)*100:.1f}%", "inline": True},
            {"name": "🎯 勝率", "value": f"{result.get('win_rate', 0)*100:.1f}%", "inline": True},
            {"name": "💼 胃納量", "value": f"{result.get('capacity', 0)/1e4:.0f}萬", "inline": True},
            {"name": "📊 平均持股", "value": f"{avg_stocks:.1f} 檔", "inline": True},
            {"name": "⚖️ 權重範圍", "value": f"{min_weight*100:.1f}% ~ {max_weight*100:.1f}%", "inline": True},
            {"name": "🛡️ 停損", "value": f"{params.get('stop_loss', 0)*100:.0f}%", "inline": True},
            {"name": "🎯 停利", "value": f"{params.get('take_profit', 0)*100:.0f}%", "inline": True},
            {"name": "📋 策略權重", "value": f"```\n{weight_text}\n```", "inline": False},
        ],
        "footer": {"text": f"回測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
    }

    return embed

#%% ========== 載入最佳基因 ==========
def load_best_genes():
    """載入歷史最佳基因"""

    # 方法1: 從 pareto 檔案載入
    if os.path.exists(PARETO_FILE):
        try:
            with open(PARETO_FILE, 'rb') as f:
                save_data = pickle.load(f)
            if save_data:
                genes, fitness = save_data[0]
                print(f"✅ 從 Pareto 檔案載入最佳基因 (夏普: {fitness[0]:.2f})")
                return genes
        except Exception as e:
            print(f"⚠️ 載入 Pareto 失敗: {e}")

    # 方法2: 從 best_params 檔案載入
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'genes' in data:
                print("✅ 從 best_params 檔案載入最佳基因")
                return data['genes']
        except Exception as e:
            print(f"⚠️ 載入 best_params 失敗: {e}")

    print("⚠️ 找不到歷史最佳基因，使用預設值")
    return [0.5] * GENE_LENGTH

#%% ========== 回測執行 ==========
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
        print(f"⚠️ 回測錯誤: {e}")
        return {'sharpe': 0, 'capacity': 0, 'annual_return': 0, 'max_drawdown': 1, 'win_rate': 0}

#%% ========== 詳細報告生成 ==========
def generate_detailed_report(report, position, params):
    """生成詳細回測報告"""

    print("\n" + "="*80)
    print("📊 詳細回測報告")
    print("="*80)

    # 1. 基本績效指標
    m = report.get_metrics()
    print("\n📈 基本績效指標:")
    print(f"   夏普值: {m['ratio'].get('sharpeRatio', 0):.2f}")
    print(f"   年化報酬: {m['profitability'].get('annualReturn', 0)*100:.1f}%")
    print(f"   最大回撤: {abs(m['risk'].get('maxDrawdown', 0))*100:.1f}%")
    print(f"   勝率: {m['profitability'].get('winRate', 0)*100:.1f}%")
    print(f"   胃納量: {m['liquidity'].get('capacity', 0)/1e4:.0f} 萬")

    # 2. 持股統計
    print("\n📋 持股統計:")
    avg_stocks = position[position > 0].count(axis=1).mean()
    weights = position[position > 0].mean()
    print(f"   平均持股: {avg_stocks:.1f} 檔")
    if len(weights) > 0:
        print(f"   權重範圍: {weights[weights > 0].min()*100:.1f}% ~ {weights.max()*100:.1f}%")
        print(f"   平均權重: {weights.mean()*100:.1f}%")

    # 3. 策略權重
    print("\n🎯 策略權重配置:")
    strategy_names = ['低波動價值', '小型成長', '營收雙渦輪', '高殖利率', '低波動穩健', '創高突破', '財務品質', '技術動能', '綜合品質']
    for i, name in enumerate(strategy_names):
        w = params.get(f'w{i+1}', 0)
        bar = '█' * int(w * 40)
        print(f"   S{i+1} {name:<10}: {w*100:5.1f}% {bar}")

    # 4. 風控參數
    print("\n🛡️ 風控參數:")
    print(f"   停損: {params.get('stop_loss', 0)*100:.0f}%")
    print(f"   移動停損: {params.get('trail_stop', 0)*100:.0f}%")
    print(f"   停利: {params.get('take_profit', 0)*100:.0f}%")
    print(f"   單股上限: {params.get('position_limit', 0)*100:.0f}%")

    # 5. 最近持股
    print("\n📋 最近一期持股明細:")
    last_pos = position.iloc[-1]
    holdings = last_pos[last_pos > 0].sort_values(ascending=False)
    if len(holdings) > 0:
        for i, (stock, weight) in enumerate(holdings.items(), 1):
            print(f"   {i:2d}. {stock}: {weight*100:.1f}%")
    else:
        print("   (無持股)")

    # 6. 月度報酬
    try:
        equity = report.daily_df['profit']
        monthly_returns = equity.resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 0:
            print("\n📅 月度報酬統計:")
            print(f"   正報酬月數: {(monthly_returns > 0).sum()}/{len(monthly_returns)} ({(monthly_returns > 0).mean()*100:.1f}%)")
            print(f"   平均月報酬: {monthly_returns.mean()*100:.2f}%")
            print(f"   最佳月份: {monthly_returns.max()*100:.2f}%")
            print(f"   最差月份: {monthly_returns.min()*100:.2f}%")

            # 最近 12 個月
            print("\n   最近 12 個月報酬:")
            recent = monthly_returns.tail(12)
            for date, ret in recent.items():
                color = "🟢" if ret > 0 else "🔴"
                print(f"   {color} {date.strftime('%Y-%m')}: {ret*100:+.1f}%")
    except Exception as e:
        print(f"   ⚠️ 無法計算月度報酬: {e}")

    print("\n" + "="*80)

#%% ========== 主程式 ==========
def main():
    print("\n🚀 開始完整回測...")

    # 載入最佳基因
    genes = load_best_genes()
    params = GeneDecoder.decode(genes)

    # 生成持倉
    print("\n📊 生成策略持倉...")
    position = combine_strategies(params)

    if position is None or position.empty:
        print("❌ 無法生成持倉")
        return

    # 持股統計
    avg_stocks = position[position > 0].count(axis=1).mean()
    print(f"   平均持股: {avg_stocks:.1f} 檔")

    # ========== 訓練期回測 ==========
    print(f"\n{'='*60}")
    print(f"📈 訓練期回測 ({TRAIN_START} ~ {TRAIN_END})")
    print('='*60)

    train_result = run_backtest(
        position, params,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        name="天空龍v4_訓練期",
        upload=False
    )

    print(f"   夏普: {train_result['sharpe']:.2f}")
    print(f"   年化: {train_result['annual_return']*100:.1f}%")
    print(f"   回撤: {train_result['max_drawdown']*100:.1f}%")
    print(f"   胃納: {train_result['capacity']/1e4:.0f}萬")

    # ========== 測試期回測 ==========
    print(f"\n{'='*60}")
    print(f"📈 測試期回測 ({TEST_START} ~ {TEST_END})")
    print('='*60)

    test_result = run_backtest(
        position, params,
        start_date=TEST_START,
        end_date=TEST_END,
        name="天空龍v4_測試期",
        upload=False
    )

    print(f"   夏普: {test_result['sharpe']:.2f}")
    print(f"   年化: {test_result['annual_return']*100:.1f}%")
    print(f"   回撤: {test_result['max_drawdown']*100:.1f}%")
    print(f"   胃納: {test_result['capacity']/1e4:.0f}萬")

    # ========== 完整回測 ==========
    print(f"\n{'='*60}")
    print(f"📈 完整回測 ({TRAIN_START} ~ {TEST_END})")
    print('='*60)

    full_result = run_backtest(
        position, params,
        start_date=TRAIN_START,
        end_date=TEST_END,
        name="天空龍v4_完整",
        upload=UPLOAD_TO_FINLAB
    )

    print(f"   夏普: {full_result['sharpe']:.2f}")
    print(f"   年化: {full_result['annual_return']*100:.1f}%")
    print(f"   回撤: {full_result['max_drawdown']*100:.1f}%")
    print(f"   胃納: {full_result['capacity']/1e4:.0f}萬")

    # 顯示官方報告
    if full_result.get('report'):
        full_result['report'].display()

    # 詳細報告
    if full_result.get('report'):
        generate_detailed_report(full_result['report'], position, params)

    # ========== Discord 通知 ==========
    if DISCORD_WEBHOOK_URL:
        print("\n📤 發送 Discord 通知...")

        # 發送訓練期結果
        train_embed = format_backtest_embed(
            f"天空龍 v4.0 訓練期 ({TRAIN_START}~{TRAIN_END})",
            train_result, position, params
        )
        send_discord_notification("", train_embed)

        # 發送測試期結果
        test_embed = format_backtest_embed(
            f"天空龍 v4.0 測試期 ({TEST_START}~{TEST_END})",
            test_result, position, params
        )
        send_discord_notification("", test_embed)

        # 發送完整期結果
        full_embed = format_backtest_embed(
            f"天空龍 v4.0 完整期 ({TRAIN_START}~{TEST_END})",
            full_result, position, params
        )
        send_discord_notification("", full_embed)

    # ========== 結果比較 ==========
    print(f"\n{'='*60}")
    print("📊 訓練期 vs 測試期 比較")
    print('='*60)
    print(f"{'指標':<15} {'訓練期':>10} {'測試期':>10} {'差異':>10}")
    print("-"*50)
    print(f"{'夏普值':<15} {train_result['sharpe']:>10.2f} {test_result['sharpe']:>10.2f} {test_result['sharpe']-train_result['sharpe']:>+10.2f}")
    print(f"{'年化報酬':<15} {train_result['annual_return']*100:>9.1f}% {test_result['annual_return']*100:>9.1f}% {(test_result['annual_return']-train_result['annual_return'])*100:>+9.1f}%")
    print(f"{'最大回撤':<15} {train_result['max_drawdown']*100:>9.1f}% {test_result['max_drawdown']*100:>9.1f}% {(test_result['max_drawdown']-train_result['max_drawdown'])*100:>+9.1f}%")
    print(f"{'胃納量(萬)':<15} {train_result['capacity']/1e4:>10.0f} {test_result['capacity']/1e4:>10.0f} {(test_result['capacity']-train_result['capacity'])/1e4:>+10.0f}")

    # 過擬合評估
    sharpe_decay = (train_result['sharpe'] - test_result['sharpe']) / train_result['sharpe'] * 100 if train_result['sharpe'] > 0 else 0
    print(f"\n🔍 過擬合評估:")
    print(f"   夏普衰減: {sharpe_decay:.1f}%")
    if sharpe_decay < 20:
        print("   ✅ 過擬合程度低，策略穩健")
    elif sharpe_decay < 40:
        print("   ⚠️ 有一定程度過擬合")
    else:
        print("   ❌ 過擬合嚴重，需要調整")

    print("\n" + "="*60)
    print("✅ 回測完成！")
    print("="*60)

if __name__ == "__main__":
    main()
