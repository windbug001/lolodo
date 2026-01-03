# =============================================================================
# 🐉 九組合媽媽龍策略 - Colab Pro+ 直接執行版
# =============================================================================
# 複製整個 cell 到 Colab 直接執行

#%% 安裝與設定
!pip install finlab deap -q

import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# 🔑 FinLab API Key
FINLAB_API_KEY = "YOUR_API_KEY_HERE"  # ← 換成您的 Key

import finlab
if FINLAB_API_KEY != "YOUR_API_KEY_HERE":
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")

from finlab import data
from finlab.backtest import sim
from finlab.dataframe import FinlabDataFrame

print(f"✅ FinLab {finlab.__version__}")

#%% 載入數據
print("📊 載入數據...")
data.set_universe('TSE_OTC')

close = data.get('price:收盤價')
vol = data.get('price:成交股數')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")
pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')
rev = data.get('monthly_revenue:當月營收')
rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
營業利益率 = data.get('fundamental_features:營業利益率')
融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")
inventory = data.get("inventory")
市值 = data.get('etl:market_value')
投資活動現金流 = data.get('financial_statement:投資活動之淨現金流入_流出')
營業活動現金流 = data.get('financial_statement:營業活動之淨現金流入_流出')
稅後淨利 = data.get('fundamental_features:經常稅後淨利')
權益總計 = data.get('financial_statement:股東權益總額')
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
limit_up = (close > close.shift(1) * 1.095)
limit_up_all_day = ((close * close).replace(0.0, np.nan) == (close * high).replace(0.0, np.nan)).replace(False, np.nan)
limit_up_all_day = (limit_up_all_day == limit_up).fillna(False)

print("✅ 數據載入完成")

#%% 九大策略
def s1(): # 低波動本益比
    peg = pe / 營業利益成長率
    cond = (rev_ma3/rev_ma12 > 1) & (rev/rev.shift(1) > 0.8) & (融資使用率 <= 40) & (entry_volatility <= 0.04) & (業外收支營收率 < 10) & (vol.average(1) > 100000) & (close > close.average(75)) & (close > close.average(40)) & ((5 <= pe) & (pe <= 25)) & (營業毛利率 > 8).sustain(2) & (ROE綜合損益 > 0).sustain(2) & ~limit_up_all_day
    return peg[cond & (peg > 0)].is_smallest(5).reindex(rev.index_str_to_date().index, method='ffill')

def s2(): # 小資族
    cond = (市值 < 15e9) & (自由現金流 > 0) & (股東權益報酬率 > 0) & (營業利益成長率 > -1) & (市值營收比 < 3) & (vol > 100000) & (close > close.average(60)) & (close > close.average(120)) & (rev.average(3) > rev.average(12))
    rsv = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min())
    return (cond * rsv).is_largest(6).reindex(當月營收.index_str_to_date().index)

def s3(): # 營收股價雙渦輪
    rev_ma = rev.average(4)
    cond = (rev_ma == rev_ma.rolling(20, min_periods=4).max()) & (close == close.rolling(260).max()).sustain(8, 1) & (vol.average(1) > 200000) & (close > close.average(5)) & (close > close.average(20)) & (close > close.average(60)) & (close > close.average(200)) & (營業毛利率 > 5).sustain(5) & (rsi > 60).sustain(1) & ~limit_up_all_day
    pos = (rev_yoy_growth * cond)
    return pos[pos > 0].is_largest(12).reindex(rev.index_str_to_date().index, method="ffill")

def s4(): # 高殖利率烏龜
    cond = (dividend_yield >= 4) & (close > close.average(20)) & (close > close.average(60)) & (rev.average(3) > rev.average(12)) & (營業利益率 >= 8) & (董監持有股數占比 >= 15) & (vol.average(5) >= 100000) & (vol.average(5) <= 3000000)
    return ((cond * rev_yoy_growth)[cond * rev_yoy_growth > 0]).is_largest(8).reindex(rev.index_str_to_date().index, method='ffill')

def s5(): # 低波動性指標
    std = close.pct_change().rolling(20).std().rank(axis=1, pct=True)
    return 市值[(vol.average(20) > 100000) & (close > close.average(60)) & (close > close.average(250)) & (std < 0.3)].is_smallest(15).reindex(close.index, method='ffill')

def s6(): # 藏獒外掛大盤指針
    vol_ma = vol.average(10)
    cond = (close == close.rolling(120).max()) & ~(rev_yoy_growth < -15).sustain(3) & ~(rev_yoy_growth > 60).sustain(12, 8) & ((rev.rolling(12).min()/rev < 1.2).sustain(3)) & (rev_month_growth > -10).sustain(3) & (vol_ma > 100000)
    return ((cond * vol_ma)[cond * vol_ma > 0]).is_smallest(10).reindex(rev.index_str_to_date().index, method='ffill')

def s7(): # 小蝦米跟大鯨魚
    h1 = inventory[inventory.持股分級.astype(int) <= 5].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
    h2 = inventory[(inventory.持股分級.astype(int) >= 9) & (inventory.持股分級.astype(int) <= 15)].reset_index().groupby(['date', 'stock_id']).agg({'持有股數': 'sum'}).reset_index().pivot(index='date', columns='stock_id', values='持有股數')
    ratio = h2 / (h1 + h2)
    p = (FinlabDataFrame(ratio).rank(axis=1, pct=True) * (close.notna() & (close > close.average(250))) + (rev / rev.shift(12).rolling(2).mean()).rank(axis=1, pct=True) + (rev / rev.shift()).rank(axis=1, pct=True)).is_largest(50)
    p = (p * 營業利益成長率).is_largest(35)
    p = (p * dividend_yield).is_largest(10)
    return (p * ratio.diff(8)).is_largest(5).reindex(rev.index_str_to_date().index, method='ffill')

def s8(): # 純技術趨勢
    def wma(x, n): return x.ewm(com=n).mean()
    def zlma(x, n): return wma(2*x - x.shift((n-1)//2), n)
    cond = (close/close.rolling(250).mean()-1 > 0.1) & (close/close.apply(lambda s: zlma(s,120))-1 > 0.1) & (close/close.apply(lambda s: wma(s,60))-1 > 0.1) & (close.rolling(60).mean().pipe(lambda df: df/df.shift(60)-1).rank(axis=1, pct=True) > 0.5) & (close.pct_change().rolling(250).kurt().rank(axis=1, pct=True) > 0.4) & (vol > 200000)
    return vol.average(10)[cond].is_smallest(10)

def s9(): # 財報指標20大
    fs = [f.average(4).rank(axis=1, pct=True).fillna(0) for f in [營業利益成長率, ROE綜合損益, 營業毛利率, 稅後淨利率, 營業利益率] if f is not None]
    if not fs: return pd.DataFrame()
    std = close.pct_change().rolling(60).std()
    score = sum(fs) + ((close > close.average(60)).astype(float)*5 + (std.rank(pct=True, axis=1) < 0.5).astype(float)*5 + (vol.rolling(5).mean() > 200000).astype(float)*10).reindex(sum(fs).index, method='ffill')
    return score.is_largest(15).reindex(close.loc[score.index[0]:].index, method='ffill')

#%% 執行回測
strategies = [(1,"低波動本益比",s1), (2,"小資族",s2), (3,"營收股價雙渦輪",s3), (4,"高殖利率烏龜",s4), (5,"低波動性指標",s5), (6,"藏獒外掛大盤指針",s6), (7,"小蝦米跟大鯨魚",s7), (8,"純技術趨勢",s8), (9,"財報指標20大",s9)]

print("\n" + "="*80)
print("🐉 九組合媽媽龍 - 策略分析")
print("="*80)
print(f"{'策略':<20} {'夏普值':<10} {'年化報酬':<12} {'最大回撤':<12} {'胃納量(萬)':<12}")
print("-"*80)

all_pos = []
for i, name, func in strategies:
    try:
        pos = func()
        if pos is not None and not pos.empty:
            pos = pos.loc['2017-01-01':]
            report = sim(pos, fee_ratio=1.425/1000, tax_ratio=3/1000, trade_at_price="high_low_avg", position_limit=0.03, stop_loss=0.25, upload=False, name=name)
            m = report.get_metrics()
            sharpe = m['ratio'].get('sharpeRatio', 0) or 0
            annual = m['profitability'].get('annualReturn', 0) or 0
            mdd = abs(m['risk'].get('maxDrawdown', 0) or 0)
            cap = m['liquidity'].get('capacity', 0) or 0
            print(f"{i}. {name:<17} {sharpe:<10.2f} {annual*100:<10.1f}% {mdd*100:<10.1f}% {cap/1e4:<10.0f}")
            all_pos.append(pos * (1/9))
    except Exception as e:
        print(f"{i}. {name:<17} ❌ {e}")

print("-"*80)

#%% 組合回測
print("\n🐉 九組合媽媽龍 - 組合策略回測")
print("="*80)

if all_pos:
    combined = sum(all_pos)
    combined = combined[combined > 0].loc['2017-01-01':]

    report = sim(
        combined,
        fee_ratio=1.425/1000,
        tax_ratio=3/1000,
        trade_at_price="high_low_avg",
        position_limit=0.03,
        stop_loss=0.25,
        upload=False,
        name="九組合媽媽龍",
        live_performance_start='2023-01-01'
    )

    report.display()

    # 輸出績效指標
    m = report.get_metrics()
    print("\n📊 關鍵績效指標：")
    print(f"   夏普值: {m['ratio'].get('sharpeRatio', 0):.2f}")
    print(f"   年化報酬: {m['profitability'].get('annualReturn', 0)*100:.1f}%")
    print(f"   最大回撤: {abs(m['risk'].get('maxDrawdown', 0))*100:.1f}%")
    print(f"   胃納量: {m['liquidity'].get('capacity', 0)/1e4:.0f} 萬")
    print(f"   勝率: {m['profitability'].get('winRate', 0)*100:.1f}%")

print("\n✅ 九組合媽媽龍策略測試完成！")
