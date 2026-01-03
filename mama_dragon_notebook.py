# =============================================================================
# 🐉 九組合媽媽龍策略 - Notebook 單 Cell 執行版
# =============================================================================
# 直接複製到 Jupyter/Colab 執行即可

#%% ========== 第一步：安裝套件 ==========
!pip install finlab deap -q

#%% ========== 第二步：設定 ==========
import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'

# 🔑 請在這裡填入您的 FinLab API Key
FINLAB_API_KEY = "YOUR_API_KEY_HERE"  # ← 換成您的 Key

# 選擇執行模式
# 1 = 策略分析（快速）
# 2 = 基因演算法優化（完整）
# 3 = 單策略測試
RUN_MODE = 1

# 如果選擇模式 3，指定要測試的策略 (1-9)
TEST_STRATEGY = 7  # 小蝦米跟大鯨魚

#%% ========== 第三步：執行 ==========
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# 登入 FinLab
import finlab
if FINLAB_API_KEY and FINLAB_API_KEY != "YOUR_API_KEY_HERE":
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")
else:
    print("⚠️ 請設定 FINLAB_API_KEY")

from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

print(f"✅ FinLab 版本: {finlab.__version__}")
print("📊 載入數據中...")

# 設定交易範圍
data.set_universe('TSE_OTC')

#%% ========== 數據載入 ==========
print("📦 載入價格數據...")
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
high = data.get('price:最高價')
low = data.get('price:最低價')
open_ = data.get('price:開盤價')
adj_close = data.get("etl:adj_close")

print("📦 載入財務數據...")
pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')

print("📦 載入營收數據...")
rev = data.get('monthly_revenue:當月營收')
rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

print("📦 載入基本面指標...")
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
稅前淨利率 = data.get("fundamental_features:稅前淨利率")
營業利益率 = data.get('fundamental_features:營業利益率')

print("📦 載入籌碼資料...")
融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")
inventory = data.get("inventory")

print("📦 載入其他數據...")
市值 = data.get('etl:market_value')
股本 = data.get('financial_statement:股本')
投資活動現金流 = data.get('financial_statement:投資活動之淨現金流入_流出')
營業活動現金流 = data.get('financial_statement:營業活動之淨現金流入_流出')
稅後淨利 = data.get('fundamental_features:經常稅後淨利')
權益總計 = data.get('financial_statement:股東權益總額')

# 技術指標
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)

# 衍生指標
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)
entry_volatility = atr / adj_close
自由現金流 = (投資活動現金流 + 營業活動現金流).rolling(4).mean()
股東權益報酬率 = 稅後淨利 / 權益總計
當月營收 = rev * 1000
當季營收 = 當月營收.rolling(4).sum()
市值營收比 = 市值 / 當季營收

# 漲停判斷
limit_up = (close > close.shift(1) * 1.095)
entry_close = (close * close).replace(0.0, np.nan)
entry_high = (close * high).replace(0.0, np.nan)
close_vs_high = (entry_close == entry_high).replace(False, np.nan)
limit_up_all_day = (close_vs_high == limit_up).fillna(False)

print("✅ 數據載入完成！")

#%% ========== 九大策略定義 ==========
def strategy_1_low_volatility_pe():
    """策略一：低波動本益比"""
    peg = pe / 營業利益成長率
    cond1 = rev_ma3 / rev_ma12 > 1.0
    cond2 = rev / rev.shift(1) > 0.8
    tree_select = ((融資使用率 <= 40) & (entry_volatility <= 0.04) & (業外收支營收率 < 10))
    cond_vol = vol.average(1) > 100000
    cond_ma = (close > close.average(75)) & (close > close.average(40))
    pe_range = (5 <= pe) & (pe <= 25)
    gpm_trend = (營業毛利率 > 8).sustain(2)
    roe_trend = (ROE綜合損益 > 0).sustain(2)
    cond_all = cond1 & cond2 & tree_select & cond_vol & cond_ma & pe_range & gpm_trend & roe_trend & ~limit_up_all_day
    position = peg[cond_all & (peg > 0)].is_smallest(5)
    return position.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_2_small_investor():
    """策略二：小資族"""
    cond1 = 市值 < 15e9
    cond2 = 自由現金流 > 0
    cond3 = 股東權益報酬率 > 0
    cond4 = 營業利益成長率 > -1
    cond5 = 市值營收比 < 3
    cond6 = vol > 100000
    cond_ma = (close > close.average(60)) & (close > close.average(120))
    cond_rev = rev.average(3) > rev.average(12)
    rsv = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min())
    position = ((cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond_ma & cond_rev) * rsv).is_largest(6)
    return position.reindex(當月營收.index_str_to_date().index)

def strategy_3_revenue_price_turbo():
    """策略三：營收股價雙渦輪"""
    rev_ma = rev.average(4)
    cond_rev_high = rev_ma == rev_ma.rolling(20, min_periods=4).max()
    cond_price_high = (close == close.rolling(260).max()).sustain(8, 1)
    cond_vol = vol.average(1) > 200000
    long_ma = (close > close.average(5)) & (close > close.average(20)) & (close > close.average(60)) & (close > close.average(200))
    rsi_trend = (rsi > 60).sustain(1)
    gpm_trend = (營業毛利率 > 5).sustain(5)
    conditions = cond_rev_high & cond_price_high & cond_vol & long_ma & gpm_trend & rsi_trend & ~limit_up_all_day
    position = (rev_yoy_growth * conditions)
    position = position[position > 0].is_largest(12)
    return position.reindex(rev.index_str_to_date().index, method="ffill")

def strategy_4_high_yield_turtle():
    """策略四：高殖利率烏龜"""
    cond1 = dividend_yield >= 4
    cond2 = (close > close.average(20)) & (close > close.average(60))
    cond3 = rev.average(3) > rev.average(12)
    cond4 = 營業利益率 >= 8
    cond5 = 董監持有股數占比 >= 15
    cond6 = (vol.average(5) >= 100000) & (vol.average(5) <= 3000000)
    cond_all = (cond1 & cond2 & cond3 & cond4 & cond5 & cond6) * rev_yoy_growth
    position = cond_all[cond_all > 0].is_largest(8)
    return position.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_5_low_volatility_index():
    """策略五：低波動性指標"""
    std = close.pct_change().rolling(20).std().rank(axis=1, pct=True)
    position = 市值[(vol.average(20) > 100000) & (close > close.average(60)) & (close > close.average(250)) & (std < 0.3)].is_smallest(15)
    return position.reindex(close.index, method='ffill')

def strategy_6_market_indicator():
    """策略六：藏獒外掛大盤指針"""
    vol_ma = vol.average(10)
    cond1 = close == close.rolling(120).max()
    cond2 = ~(rev_yoy_growth < -15).sustain(3)
    cond3 = ~(rev_yoy_growth > 60).sustain(12, 8)
    cond4 = ((rev.rolling(12).min()) / rev < 1.2).sustain(3)
    cond5 = (rev_month_growth > -10).sustain(3)
    cond6 = vol_ma > 100000
    buy = (cond1 & cond2 & cond3 & cond4 & cond5 & cond6) * vol_ma
    buy = buy[buy > 0].is_smallest(10)
    return buy.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_7_shrimp_whale():
    """策略七：小蝦米跟大鯨魚"""
    h1 = inventory[inventory.持股分級.astype(int) <= 5].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
    h2 = inventory[(inventory.持股分級.astype(int) >= 9) & (inventory.持股分級.astype(int) <= 15)].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
    ratio = h2 / (h1 + h2)
    rev_mom = rev / rev.shift()
    rev_yoy2 = rev.rolling(2).mean() / rev.shift(12).rolling(2).mean()
    p = (FinlabDataFrame(ratio).rank(axis=1, pct=True) * (close.notna() & (close > close.average(250))) + rev_yoy2.rank(axis=1, pct=True) + rev_mom.rank(axis=1, pct=True)).is_largest(50)
    p = (p * 營業利益成長率).is_largest(35)
    p = (p * dividend_yield).is_largest(10)
    p = (p * ratio.diff(8)).is_largest(5)
    return p.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_8_tech_trend():
    """策略八：純技術趨勢"""
    def wma(price, n): return price.ewm(com=n).mean()
    def zlma(price, n):
        lag = (n - 1) // 2
        return wma(2 * price - price.shift(lag), n)
    bias_sma = close / close.rolling(250).mean() - 1
    bias_zlma = close / close.apply(lambda s: zlma(s, 120)) - 1
    bias_wma = close / close.apply(lambda s: wma(s, 60)) - 1
    slope = close.rolling(60).mean().pipe(lambda df: df / df.shift(60) - 1)
    kurtosis = close.pct_change().rolling(250).kurt()
    pos = vol.average(10)[(bias_sma > 0.1) & (bias_zlma > 0.1) & (bias_wma > 0.1) & (slope.rank(axis=1, pct=True) > 0.5) & (kurtosis.rank(axis=1, pct=True) > 0.4) & (vol > 200000)].is_smallest(10)
    return pos

def strategy_9_fundamental_20():
    """策略九：財報指標20大"""
    features = [營業利益成長率, ROE綜合損益, 營業毛利率, 稅後淨利率, 營業利益率]
    fs = [f.average(4).rank(axis=1, pct=True).fillna(0) for f in features if f is not None]
    if not fs: return pd.DataFrame()
    std = close.pct_change().rolling(60).std()
    cond1 = (close > close.average(60)).astype(float) * 5
    cond2 = (std.rank(pct=True, axis=1) < 0.5).astype(float) * 5
    cond3 = (vol.rolling(5).mean() > 200000).astype(float) * 10
    score = sum(fs)
    final_score = score + (cond1 + cond2 + cond3).reindex(score.index, method='ffill')
    position = final_score.is_largest(15)
    return position.reindex(close.loc[score.index[0]:].index, method='ffill')

#%% ========== 執行回測 ==========
strategies = {
    1: ("低波動本益比", strategy_1_low_volatility_pe),
    2: ("小資族", strategy_2_small_investor),
    3: ("營收股價雙渦輪", strategy_3_revenue_price_turbo),
    4: ("高殖利率烏龜", strategy_4_high_yield_turtle),
    5: ("低波動性指標", strategy_5_low_volatility_index),
    6: ("藏獒外掛大盤指針", strategy_6_market_indicator),
    7: ("小蝦米跟大鯨魚", strategy_7_shrimp_whale),
    8: ("純技術趨勢", strategy_8_tech_trend),
    9: ("財報指標20大", strategy_9_fundamental_20),
}

def run_backtest(position, name):
    """執行回測"""
    if position is None or position.empty:
        print(f"⚠️ {name}: 無持股訊號")
        return None
    position = position.loc['2017-01-01':]
    report = sim(position, fee_ratio=1.425/1000, tax_ratio=3/1000, trade_at_price="high_low_avg", position_limit=0.03, stop_loss=0.25, upload=False, name=name)
    return report

if RUN_MODE == 1:
    # 策略分析
    print("\n" + "="*80)
    print("🐉 九組合媽媽龍 - 策略分析")
    print("="*80)
    print(f"{'策略':<20} {'夏普值':<10} {'年化報酬':<12} {'最大回撤':<12} {'胃納量(萬)':<12}")
    print("-"*80)

    results = []
    for i, (name, func) in strategies.items():
        try:
            pos = func()
            report = run_backtest(pos, name)
            if report:
                m = report.get_metrics()
                sharpe = m['ratio'].get('sharpeRatio', 0) or 0
                annual = m['profitability'].get('annualReturn', 0) or 0
                mdd = abs(m['risk'].get('maxDrawdown', 0) or 0)
                cap = m['liquidity'].get('capacity', 0) or 0
                print(f"{i}. {name:<17} {sharpe:<10.2f} {annual*100:<10.1f}% {mdd*100:<10.1f}% {cap/1e4:<10.0f}")
                results.append({'name': name, 'sharpe': sharpe, 'report': report})
        except Exception as e:
            print(f"{i}. {name:<17} ❌ 錯誤: {e}")

    print("-"*80)

    # 組合回測
    print("\n🐉 九組合媽媽龍 - 組合回測")
    all_positions = []
    for i, (name, func) in strategies.items():
        try:
            pos = func()
            if pos is not None and not pos.empty:
                all_positions.append(pos * (1/9))
        except:
            pass

    if all_positions:
        combined = sum(all_positions)
        combined = combined[combined > 0]
        report = run_backtest(combined, "九組合媽媽龍")
        if report:
            report.display()

elif RUN_MODE == 3:
    # 單策略測試
    name, func = strategies[TEST_STRATEGY]
    print(f"\n🔍 測試策略 {TEST_STRATEGY}: {name}")
    print("="*60)
    pos = func()
    if pos is not None and not pos.empty:
        print(f"持股天數: {len(pos)}")
        print(f"平均持股數: {pos.sum(axis=1).mean():.1f}")
        report = run_backtest(pos, name)
        if report:
            report.display()

print("\n✅ 測試完成！")
