# 小小龍_超完整回測細項Discord通知版
# 仿照六組合快快龍回測模式，針對三策略組合進行篩選

# ============================================================================
# 第一部分：導入套件
# ============================================================================
from finlab import data
from finlab.backtest import sim
from finlab.data import indicator
import random
import numpy as np
import pandas as pd
from functools import reduce
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import sys
import datetime
import requests
import time
import glob

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# ============================================================================
# 第二部分：gene_to_params 函數（小小龍版 - 32參數）
# ============================================================================
def gene_to_params(gene):
    """將基因轉換為策略參數（小小龍版：3策略32參數）"""
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 32
    current_length = len(gene)

    if current_length < required_length:
        extension = [0.5] * (required_length - current_length)
        gene = gene + extension
    elif current_length > required_length:
        gene = gene[:required_length]

    params = {}

    # === 策略一：低波動本益比（9個參數）===
    params['lv_rev_ma3_ma12_ratio'] = gene[0] * 0.5 + 1.0  # 1.0-1.5
    params['lv_rev_consistency'] = gene[1] * 0.4 + 0.5  # 0.5-0.9
    params['lv_volatility_threshold'] = gene[2] * 0.05 + 0.02  # 0.02-0.07
    params['lv_margin_usage_limit'] = gene[3] * 30 + 20  # 20-50
    params['lv_non_op_income_limit'] = gene[4] * 10 + 5  # 5-15
    params['lv_min_volume'] = gene[5] * 200000 + 100000  # 10萬-30萬
    params['lv_pe_min'] = gene[6] * 10 + 3  # 3-13
    params['lv_pe_max'] = gene[7] * 20 + 15  # 15-35
    params['lv_top_n'] = int(gene[8] * 5 + 2)  # 2-7

    # === 策略二：小資族（8個參數）===
    params['si_market_value_limit'] = gene[9] * 10e9 + 10e9  # 100億-200億
    params['si_market_rev_ratio_limit'] = gene[10] * 3 + 2  # 2-5
    params['si_rev_yoy_growth_limit'] = gene[11] * 10 - 15  # -15 to -5
    params['si_rev_mom_growth_limit'] = gene[12] * 30 - 70  # -70 to -40
    params['si_rsv_period'] = int(gene[13] * 30 + 40)  # 40-70
    params['si_ma_period'] = int(gene[14] * 40 + 50)  # 50-90
    params['si_volume_threshold'] = gene[15] * 200000 + 100000  # 10萬-30萬
    params['si_top_n'] = int(gene[16] * 6 + 4)  # 4-10

    # === 策略三：營收股價雙渦輪（8個參數）===
    params['rpt_rev_ma_period'] = int(gene[17] * 5 + 3)  # 3-8
    params['rpt_rev_ma_lookback'] = int(gene[18] * 15 + 15)  # 15-30
    params['rpt_price_high_window'] = int(gene[19] * 8 + 4)  # 4-12
    params['rpt_min_volume'] = gene[20] * 300000 + 200000  # 20萬-50萬
    params['rpt_min_price'] = gene[21] * 10 + 10  # 10-20
    params['rpt_rsi_threshold'] = gene[22] * 20 + 50  # 50-70
    params['rpt_pe_limit'] = gene[23] * 100 + 100  # 100-200
    params['rpt_top_n'] = int(gene[24] * 10 + 8)  # 8-18

    # === 策略權重（3個）===
    raw_weights = [max(0.01, gene[25]), max(0.01, gene[26]), max(0.01, gene[27])]
    total = sum(raw_weights)
    params['weight_lv'] = raw_weights[0] / total
    params['weight_si'] = raw_weights[1] / total
    params['weight_rpt'] = raw_weights[2] / total

    # === 回測參數（4個）===
    params['stop_loss'] = gene[28] * 0.2 + 0.15  # 15%-35%
    params['trail_stop'] = gene[29] * 0.3 + 0.2  # 20%-50%
    params['take_profit'] = gene[30] * 0.5 + 0.5  # 50%-100%
    params['position_limit'] = gene[31] * 0.2 + 0.25  # 25%-45%

    return params

# ============================================================================
# 第三部分：Discord Webhook 通知系統
# ============================================================================
class DiscordNotifier:
    """Discord Webhook 通知系統"""

    def __init__(self, webhook_url=None, enabled=True):
        self.enabled = enabled and webhook_url is not None
        self.webhook_url = webhook_url
        self.send_delay = 1.0

        if self.enabled:
            test_result = self.send("✅ 小小龍回測通知系統已啟動", test=True)
            if test_result:
                print("✅ Discord 通知系統已連接")
            else:
                print("⚠️ Discord 通知系統測試失敗")
                self.enabled = False
        else:
            if webhook_url is None:
                print("ℹ️ Discord 通知功能未啟用")

    def send(self, message, test=False):
        if not self.enabled:
            return False
        try:
            data_payload = {"content": message, "username": "小小龍回測機器人"}
            response = requests.post(self.webhook_url, json=data_payload, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            if not test:
                print(f"⚠️ Discord 通知發送失敗: {e}")
            return False

    def send_embed(self, title, description, color=None, fields=None):
        if not self.enabled:
            return False
        try:
            color_map = {'success': 3066993, 'error': 15158332, 'warning': 16776960, 'info': 3447003}
            embed_color = color_map.get(color, 3447003)
            embed = {
                "title": title,
                "description": description,
                "color": embed_color,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            if fields:
                embed["fields"] = fields
            data_payload = {"embeds": [embed], "username": "小小龍回測機器人"}
            response = requests.post(self.webhook_url, json=data_payload, timeout=10)
            return response.status_code in [200, 204]
        except Exception as e:
            print(f"⚠️ Discord Embed 發送失敗: {e}")
            return False

    def send_milestone(self, message):
        result = self.send_embed(title="🎉 重要里程碑！", description=message, color='success')
        time.sleep(self.send_delay)
        return result

    def send_found_strategy(self, rank, candidate_idx, sharpe, annual_return, capacity):
        fields = [
            {"name": "📊 排名", "value": f"`第 {rank} 名`", "inline": True},
            {"name": "🏆 夏普值", "value": f"`{sharpe:.4f}`", "inline": True},
            {"name": "💰 年化收益", "value": f"`{annual_return*100:.1f}%`", "inline": True},
            {"name": "🎯 胃納量", "value": f"`{capacity/10000:.1f}萬`", "inline": True},
            {"name": "📌 候選編號", "value": f"`{candidate_idx}`", "inline": True}
        ]
        result = self.send_embed(title=f"💎 發現合格策略！（第 {rank} 個）", description="", color='success', fields=fields)
        time.sleep(self.send_delay)
        return result

    def send_complete_performance(self, rank, metrics, position_combined, params):
        if not self.enabled:
            return False
        print(f"\n📤 開始發送第 {rank} 名的完整績效報告...")
        try:
            backtest = metrics.get('backtest', {})
            profitability = metrics.get('profitability', {})
            risk = metrics.get('risk', {})
            ratio = metrics.get('ratio', {})
            winrate = metrics.get('winrate', {})
            liquidity = metrics.get('liquidity', {})

            results = {}
            print(f"  ├─ 發送回測基本資訊...")
            results['backtest'] = self._send_backtest_info(rank, backtest)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送盈利能力指標...")
            results['profitability'] = self._send_profitability_metrics(rank, profitability)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送風險指標...")
            results['risk'] = self._send_risk_metrics(rank, risk)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送比率指標...")
            results['ratio'] = self._send_ratio_metrics(rank, ratio)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送勝率指標...")
            results['winrate'] = self._send_winrate_metrics(rank, winrate)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送流動性指標...")
            results['liquidity'] = self._send_liquidity_metrics(rank, liquidity)
            time.sleep(self.send_delay)

            print(f"  ├─ 發送選股明細...")
            results['holdings'] = self._send_holdings_analysis(rank, position_combined)
            time.sleep(self.send_delay)

            print(f"  └─ 發送策略配置...")
            results['allocation'] = self._send_strategy_allocation(rank, params)
            time.sleep(self.send_delay)

            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            print(f"✅ 完整績效報告發送完成: {success_count}/{total_count} 成功")
            return success_count == total_count
        except Exception as e:
            print(f"⚠️ 發送完整績效失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _send_backtest_info(self, rank, backtest):
        try:
            fields = [
                {"name": "📅 開始日期", "value": f"`{backtest.get('startDate', 'N/A')}`", "inline": True},
                {"name": "📅 結束日期", "value": f"`{backtest.get('endDate', 'N/A')}`", "inline": True},
                {"name": "💵 手續費率", "value": f"`{backtest.get('feeRatio', 0)*100:.3f}%`", "inline": True},
                {"name": "💰 稅率", "value": f"`{backtest.get('taxRatio', 0)*100:.2f}%`", "inline": True},
                {"name": "🎯 交易時點", "value": f"`{backtest.get('tradeAt', 'N/A')}`", "inline": True},
            ]
            return self.send_embed(title=f"📋 第{rank}名 - 回測基本資訊", description="", color='info', fields=fields)
        except:
            return False

    def _send_profitability_metrics(self, rank, profitability):
        try:
            fields = [
                {"name": "📈 年回報率", "value": f"`{profitability.get('annualReturn', 0)*100:.2f}%`", "inline": True},
                {"name": "🎯 Alpha", "value": f"`{profitability.get('alpha', 0)*100:.2f}%`", "inline": True},
                {"name": "🔵 Beta", "value": f"`{profitability.get('beta', 0):.3f}`", "inline": True},
            ]
            return self.send_embed(title=f"💰 第{rank}名 - 盈利能力指標", description="", color='success', fields=fields)
        except:
            return False

    def _send_risk_metrics(self, rank, risk):
        try:
            fields = [
                {"name": "📉 最大回撤", "value": f"`{risk.get('maxDrawdown', 0)*100:.2f}%`", "inline": True},
                {"name": "📊 平均回撤", "value": f"`{risk.get('avgDrawdown', 0)*100:.2f}%`", "inline": True},
            ]
            return self.send_embed(title=f"⚠️ 第{rank}名 - 風險指標", description="", color='error', fields=fields)
        except:
            return False

    def _send_ratio_metrics(self, rank, ratio):
        try:
            fields = [
                {"name": "🏆 Sharpe Ratio", "value": f"`{ratio.get('sharpeRatio', 0):.4f}`", "inline": True},
                {"name": "📊 Sortino Ratio", "value": f"`{ratio.get('sortinoRatio', 0):.4f}`", "inline": True},
                {"name": "📐 Calmar Ratio", "value": f"`{ratio.get('calmarRatio', 0):.4f}`", "inline": True},
            ]
            return self.send_embed(title=f"📊 第{rank}名 - 比率指標", description="", color='info', fields=fields)
        except:
            return False

    def _send_winrate_metrics(self, rank, winrate):
        try:
            fields = [
                {"name": "✅ Win Rate", "value": f"`{winrate.get('winRate', 0)*100:.2f}%`", "inline": True},
                {"name": "🎯 Expectancy", "value": f"`{winrate.get('expectancy', 0)*100:.2f}%`", "inline": True},
            ]
            return self.send_embed(title=f"🎲 第{rank}名 - 勝率指標", description="", color='success', fields=fields)
        except:
            return False

    def _send_liquidity_metrics(self, rank, liquidity):
        try:
            fields = [
                {"name": "💰 Capacity", "value": f"`${liquidity.get('capacity', 0):,.0f}`", "inline": True},
            ]
            return self.send_embed(title=f"💧 第{rank}名 - 流動性指標", description="", color='warning', fields=fields)
        except:
            return False

    def _send_holdings_analysis(self, rank, position_combined):
        try:
            if position_combined.empty:
                return self.send(f"⚠️ 第{rank}名策略目前無持股")

            last_positions = position_combined.iloc[-1]
            holdings = last_positions[last_positions > 0].sort_values(ascending=False)

            if len(holdings) == 0:
                return self.send(f"⚠️ 第{rank}名策略目前無持股")

            total_weight = holdings.sum()
            normalized_holdings = (holdings / total_weight) if total_weight > 0 else holdings
            top10 = normalized_holdings.head(10)

            holdings_text = ""
            for i, (stock_id, weight) in enumerate(top10.items(), 1):
                holdings_text += f"{i}. `{stock_id}` - {weight*100:.1f}%\n"

            description = f"**🏆 前10大持股**\n{holdings_text}\n**📊 總持股數**: `{len(holdings)}` 檔"
            return self.send_embed(title=f"📋 第{rank}名 - 選股明細", description=description, color='success')
        except Exception as e:
            print(f"⚠️ 發送選股明細失敗: {e}")
            return False

    def _send_strategy_allocation(self, rank, params):
        try:
            strategies = ["低波動本益比", "小資族", "營收股價雙渦輪"]
            weights = [params['weight_lv'], params['weight_si'], params['weight_rpt']]

            active_strategies = [(name, weight) for name, weight in zip(strategies, weights) if weight > 0.01]
            active_strategies.sort(key=lambda x: x[1], reverse=True)

            allocation_text = "\n".join([
                f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} **{name}**: `{weight*100:.1f}%`"
                for i, (name, weight) in enumerate(active_strategies)
            ])

            description = f"**💼 資金配置**\n{allocation_text}"
            return self.send_embed(title=f"🎯 第{rank}名 - 策略配置", description=description, color='warning')
        except Exception as e:
            print(f"⚠️ 發送策略配置失敗: {e}")
            return False

    def send_progress(self, qualified_count, tested_count, target_count):
        progress = qualified_count / target_count
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        fields = [
            {"name": "✅ 已找到", "value": f"`{qualified_count}` 個", "inline": True},
            {"name": "🔍 已測試", "value": f"`{tested_count}` 個", "inline": True},
            {"name": "🎯 目標", "value": f"`{target_count}` 個", "inline": True},
            {"name": "📊 進度", "value": f"`{progress_bar}` {progress*100:.1f}%", "inline": False}
        ]
        result = self.send_embed(title="📈 篩選進度更新", description="", color='info', fields=fields)
        time.sleep(self.send_delay)
        return result

# Discord Webhook 設定
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"
notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# ============================================================================
# 第四部分：環境設定
# ============================================================================
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')

    BASE_PATH = '/content/drive/MyDrive'
    BACKTEST_RESULTS_PATH = f'{BASE_PATH}/FinLab_GA_優化_終極版/回測結果'
    os.makedirs(BACKTEST_RESULTS_PATH, exist_ok=True)
    os.chdir(BACKTEST_RESULTS_PATH)

    print(f"✅ 工作目錄已設定為: {os.getcwd()}")
    print(f"✅ 回測結果將儲存至: {BACKTEST_RESULTS_PATH}")

    # 小小龍相關的搜尋路徑
    SEARCH_PATHS = [
        f'{BASE_PATH}/FinLab_GA_優化_終極版',
        f'{BASE_PATH}/FinLab_GA_優化_終極版/window_1',
        f'{BASE_PATH}/FinLab_GA_優化_終極版/window_2',
        f'{BASE_PATH}/FinLab_GA_優化_終極版/window_3',
        f'{BASE_PATH}/FinLab_GA_優化_終極版/shared_pareto',
        f'{BASE_PATH}/FinLab_GA_優化_終極版/output',
    ]

    # 過濾出實際存在的路徑
    SEARCH_PATHS = [path for path in SEARCH_PATHS if os.path.exists(path)]

    print(f"\n📁 可用的搜尋路徑 ({len(SEARCH_PATHS)} 個):")
    for i, path in enumerate(SEARCH_PATHS, 1):
        print(f"   {i}. {os.path.basename(path)}")

else:
    BACKTEST_RESULTS_PATH = '.'
    SEARCH_PATHS = ['.']
    print(f"⚠️ 非 Colab 環境，工作目錄: {os.getcwd()}")

FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000

# ============================================================================
# 第五部分：載入數據
# ============================================================================
print("正在載入數據...")
notifier.send("📊 開始載入市場數據...")

close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
rev = data.get('monthly_revenue:當月營收')
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)
rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
稅前淨利率 = data.get("fundamental_features:稅前淨利率")

融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")
inventory = data.get("inventory")
市值 = data.get('etl:market_value')

# 財務報表數據
投資活動現金流 = data.get('financial_statement:投資活動之淨現金流入_流出')
營業活動現金流 = data.get('financial_statement:營業活動之淨現金流入_流出')
自由現金流 = (投資活動現金流 + 營業活動現金流).rolling(4).mean()
稅後淨利 = data.get('fundamental_features:經常稅後淨利')
權益總計 = data.get('financial_statement:股東權益總額')
股東權益報酬率 = 稅後淨利 / 權益總計
當月營收 = data.get('monthly_revenue:當月營收') * 1000
當季營收 = 當月營收.rolling(4).sum()
市值營收比 = 市值 / 當季營收

成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

print("數據載入完成，進行共通計算...")

# ============================================================================
# 第六部分：共通計算
# ============================================================================
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

rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)
entry_volatility = atr/adj_close

print("共通計算完成。")
notifier.send("✅ 數據載入完成，準備策略篩選...")

# ============================================================================
# 第七部分：策略函數定義（小小龍版：3策略）
# ============================================================================
def strategy_low_volatility_pe(params):
    """策略一：低波動本益比"""
    peg = pe / 營業利益成長率

    cond1 = rev_ma3 / rev_ma12 > params['lv_rev_ma3_ma12_ratio']
    cond2 = rev / rev.shift(1) > params['lv_rev_consistency']

    tree_select_factor = ((融資使用率 <= params['lv_margin_usage_limit'])
                          & (entry_volatility <= params['lv_volatility_threshold'])
                          & (業外收支營收率 < params['lv_non_op_income_limit']))

    condition_近1日成交均量 = vol.average(1) > params['lv_min_volume']
    cond排除月營收連3月衰退 = ~(rev_yoy_growth < -30).sustain(3)
    cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 30).sustain(12, 8)
    cond單月營收月增率 = (rev_month_growth > -54).sustain(3)
    cond收盤價大於均線 = (close > close.average(75)) & (close > close.average(40)) & (close > close.average(90))
    cond近三個月營收大於年營收 = rev.average(4) > rev.average(12)

    pe_range = (params['lv_pe_min'] <= pe) & (pe <= params['lv_pe_max'])
    pb_range = (0.5 <= pb) & (pb <= 2.8)
    gpm_trend = (營業毛利率 > 8).sustain(2)
    roe_trend = (ROE綜合損益 > 0).sustain(2)

    small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                         .reset_index()
                         .groupby(["date", "stock_id"])
                         .agg({"占集保庫存數比例": "sum"})
                         .reset_index()
                         .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46

    cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老 & cond排除月營收連3月衰退
                & cond收盤價大於均線 & cond近三個月營收大於年營收 & cond單月營收月增率 & ~(limit_up_all_day)
                & condition_近1日成交均量 & gpm_trend & roe_trend & small_inv_under50 & pe_range & pb_range)

    position = peg[cond_all & (peg > 0)].is_smallest(params['lv_top_n']).reindex(rev.index_str_to_date().index, method='ffill')
    return position

def strategy_small_investor(params):
    """策略二：小資族"""
    cond1 = (市值 < params['si_market_value_limit'])
    cond2 = 自由現金流 > 0
    cond3 = 股東權益報酬率 > 0
    cond4 = 營業利益成長率 > -1
    cond5 = 市值營收比 < params['si_market_rev_ratio_limit']
    cond6 = vol > params['si_volume_threshold']

    cond排除月營收連3月衰退 = ~(rev_yoy_growth < params['si_rev_yoy_growth_limit']).sustain(3)
    cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12, 8)
    cond單月營收月增率 = (rev_month_growth > params['si_rev_mom_growth_limit']).sustain(3)

    ma_period = params['si_ma_period']
    cond收盤價大於均線 = (close > close.average(ma_period)) & (close > close.average(120)) & (close > close.average(75))
    cond近三個月營收大於年營收 = rev.average(3) > rev.average(12)
    業外收支營收率占比低 = (業外收支營收率 < 7.3)

    rsv_period = params['si_rsv_period']
    rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())

    position = ((cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond排除月營收成長趨勢過老 & cond單月營收月增率
                 & cond排除月營收連3月衰退 & cond近三個月營收大於年營收 & cond收盤價大於均線 & 業外收支營收率占比低)
                * rsv).is_largest(params['si_top_n'])

    position = position.reindex(當月營收.index_str_to_date().index, method='ffill')
    return position

def strategy_revenue_price_turbo(params):
    """策略三：營收股價雙渦輪"""
    rev_ma_period = max(1, int(params['rpt_rev_ma_period']))
    rev_ma = rev.average(rev_ma_period)
    rev_ma_lookback = max(rev_ma_period + 1, int(params['rpt_rev_ma_lookback']))

    condition_近N月平均營收創M個月來新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=rev_ma_period).max()
    price_high_window = max(1, int(params['rpt_price_high_window']))
    condition_近N日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
    condition_成交均量大於閾值 = vol.average(1) > params['rpt_min_volume']

    long_ma_pattern = ((close > close.average(5)) & (close > close.average(10)) & (close > close.average(20))
                       & (close > close.average(60)) & (close > close.average(150)) & (close > close.average(200)))

    收盤價_超級績效 = close > (close.average(250) * 1.1)
    rsi_higt_trend = (rsi > params['rpt_rsi_threshold']).sustain(1)
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

    pe_range = ~(params['rpt_pe_limit'] <= pe)
    min_price = params['rpt_min_price']

    conditions = (condition_近N月平均營收創M個月來新高 & condition_近N日內有1日股價創新高 & condition_成交均量大於閾值
                  & long_ma_pattern & gpm_trend & btpm_trend & atpm_trend & rev_rise_nsatisfy & (close > min_price)
                  & (業外收支營收率 < 7.3) & pe_range & 收盤價_超級績效 & ((vol >= vol.rolling(20).mean()*0.8))
                  & rsi_higt_trend & boss_inventory_over400 & ~(limit_up_all_day))

    position = rev_yoy_growth * conditions
    position = position[position > 0].is_largest(params['rpt_top_n']).reindex(rev.index_str_to_date().index, method="ffill")
    return position

# ============================================================================
# 第八部分：合併策略函數
# ============================================================================
def combined_strategy(gene, start_date_str="2017-01-01"):
    """根據基因參數生成合併策略（小小龍版：3策略）"""
    try:
        params = gene_to_params(gene)

        position_lv = strategy_low_volatility_pe(params)
        position_si = strategy_small_investor(params)
        position_rpt = strategy_revenue_price_turbo(params)

        position_combined = (position_lv * params['weight_lv']
                            + position_si * params['weight_si']
                            + position_rpt * params['weight_rpt'])

        # 流動性過濾
        liquidity_threshold = 1e6  # 100萬
        liquid_stocks = 平均成交金額 > liquidity_threshold
        position_combined = position_combined * liquid_stocks

        try:
            if position_combined.empty:
                return pd.DataFrame(), params
            if not isinstance(position_combined.index, pd.DatetimeIndex):
                position_combined.index = pd.to_datetime(position_combined.index, errors='coerce')
                position_combined = position_combined.loc[~position_combined.index.isna()]
            start_date_ts = pd.Timestamp(start_date_str)
            position_combined = position_combined[position_combined.index >= start_date_ts]
            if 'safe_end_date' in globals() and globals()['safe_end_date'] is not None:
                safe_end_date = globals()['safe_end_date']
                if not isinstance(safe_end_date, pd.Timestamp):
                    safe_end_date = pd.Timestamp(safe_end_date)
                position_combined = position_combined[position_combined.index <= safe_end_date]
        except Exception as e:
            print(f"處理持倉資料的時間範圍時發生錯誤: {e}")

        return position_combined, params
    except Exception as e:
        print(f"合併策略函數發生錯誤: {e}")
        return pd.DataFrame(), {}

# ============================================================================
# 第九部分：主程式
# ============================================================================
if __name__ == "__main__":
    try:
        # 篩選參數設定
        TARGET_COUNT = 5
        MIN_ANNUAL_RETURN = 0.3
        MIN_CAPACITY = 5000000
        SHARPE_MIN = 2.0
        SHARPE_MAX = 5.0
        MAX_CANDIDATES = 500

        print(f"{'='*100}")
        print(f"{'小小龍：三策略組合精準篩選（找前5名）':^100}")
        print(f"{'='*100}\n")

        notifier.send(
            f"🚀 **小小龍回測啟動**\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📌 **篩選條件**\n"
            f"• 夏普值: {SHARPE_MIN:.2f} ~ {SHARPE_MAX:.2f}\n"
            f"• 年化收益 ≥ {MIN_ANNUAL_RETURN*100:.0f}%\n"
            f"• 胃納量 > {MIN_CAPACITY/10000:.0f}萬\n"
            f"🎯 目標: 找到 {TARGET_COUNT} 個合格策略\n"
            f"📁 搜尋路徑: {len(SEARCH_PATHS)} 個"
        )

        print(f"📌 篩選條件：")
        print(f"   1. 歷史夏普值: {SHARPE_MIN:.2f} ~ {SHARPE_MAX:.2f}")
        print(f"   2. 年化收益率 ≥ {MIN_ANNUAL_RETURN*100:.0f}%")
        print(f"   3. 胃納量 > {MIN_CAPACITY/10000:.0f}萬")
        print(f"📌 目標數量：{TARGET_COUNT} 個合格策略\n")

        print("【步驟1】設定回測期間...")
        safe_end_date_data = pd.Timestamp(datetime.datetime.now().strftime('%Y-%m-%d'))
        effective_start_date = pd.Timestamp("2017-01-01")
        effective_end_date = safe_end_date_data
        globals()['safe_end_date'] = effective_end_date
        globals()['safe_start_date'] = effective_start_date
        print(f"✓ 回測期間: {effective_start_date.strftime('%Y-%m-%d')} 至 {effective_end_date.strftime('%Y-%m-%d')}")

        print(f"\n【步驟2】搜尋夏普值 {SHARPE_MIN:.2f}~{SHARPE_MAX:.2f} 範圍內的策略...")
        print(f"   搜尋範圍: {len(SEARCH_PATHS)} 個資料夾")

        all_individuals = []
        total_files_found = 0

        for search_idx, search_path in enumerate(SEARCH_PATHS, 1):
            print(f"\n   [{search_idx}/{len(SEARCH_PATHS)}] 搜尋: {os.path.basename(search_path)}")
            checkpoint_pattern = os.path.join(search_path, "**", "*.pkl")
            matching_files = glob.glob(checkpoint_pattern, recursive=True)
            total_files_found += len(matching_files)
            print(f"       找到 {len(matching_files)} 個 .pkl 檔案")

            files_loaded = 0
            for file in matching_files:
                try:
                    with open(file, 'rb') as f:
                        cp = pickle.load(f)

                    # 嘗試從不同格式的存檔中讀取個體
                    individuals_to_check = []

                    # 格式1: halloffame
                    if "halloffame" in cp and cp["halloffame"]:
                        individuals_to_check.extend(cp["halloffame"])

                    # 格式2: population
                    if "population" in cp and cp["population"]:
                        individuals_to_check.extend(cp["population"])

                    # 格式3: individuals (Pareto archive 格式)
                    if "individuals" in cp and cp["individuals"]:
                        for ind_data in cp["individuals"]:
                            if isinstance(ind_data, dict) and 'genes' in ind_data and 'fitness' in ind_data:
                                # 創建一個簡單對象來存儲
                                class SimpleInd:
                                    def __init__(self, genes, fitness):
                                        self.genes = genes
                                        self.fitness_values = fitness if isinstance(fitness, tuple) else (sum(fitness) if isinstance(fitness, (list, tuple)) else fitness,)
                                ind = SimpleInd(ind_data['genes'], ind_data['fitness'])
                                sharpe = ind.fitness_values[0] if isinstance(ind.fitness_values, tuple) else ind.fitness_values
                                if SHARPE_MIN <= sharpe <= SHARPE_MAX:
                                    all_individuals.append({
                                        'gene': ind_data['genes'],
                                        'sharpe': sharpe,
                                        'source': os.path.basename(file),
                                        'path': os.path.dirname(file)
                                    })
                                    files_loaded += 1

                    # 處理標準格式的個體
                    for ind in individuals_to_check:
                        if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'valid') and ind.fitness.valid:
                            # 小小龍的適應度是 4 個值的元組
                            fitness_values = ind.fitness.values
                            # 計算夏普值：第一個值除以5再乘以目標夏普值
                            sharpe = fitness_values[0] / 5.0 * 4.0 if len(fitness_values) >= 1 else 0
                            if SHARPE_MIN <= sharpe <= SHARPE_MAX:
                                all_individuals.append({
                                    'gene': list(ind),
                                    'sharpe': sharpe,
                                    'source': os.path.basename(file),
                                    'path': os.path.dirname(file)
                                })
                                files_loaded += 1
                except Exception as e:
                    continue
            print(f"       成功載入 {files_loaded} 個有效個體")

        print(f"\n📊 搜尋統計：")
        print(f"   - 搜尋資料夾數: {len(SEARCH_PATHS)}")
        print(f"   - 找到檔案總數: {total_files_found}")
        print(f"   - 符合夏普範圍: {len(all_individuals)} 個")

        # 去重
        seen_sharpes = set()
        candidates = []
        for ind in sorted(all_individuals, key=lambda x: x['sharpe'], reverse=True):
            sharpe_rounded = round(ind['sharpe'], 4)
            if sharpe_rounded not in seen_sharpes:
                seen_sharpes.add(sharpe_rounded)
                candidates.append(ind)
                if len(candidates) >= MAX_CANDIDATES:
                    break

        if not candidates:
            error_msg = f"❌ 未找到夏普值在 {SHARPE_MIN:.2f}~{SHARPE_MAX:.2f} 範圍內的策略"
            print(error_msg)
            notifier.send(error_msg)
            exit(1)

        print(f"✓ 去重後保留 {len(candidates)} 個候選個體")

        sharpe_values = [c['sharpe'] for c in candidates]
        notifier.send(
            f"✅ **步驟2完成** - 候選搜尋\n"
            f"📊 找到 `{len(candidates)}` 個候選策略\n"
            f"🏆 夏普值分布:\n"
            f"  • 最高: `{max(sharpe_values):.4f}`\n"
            f"  • 最低: `{min(sharpe_values):.4f}`\n"
            f"  • 平均: `{sum(sharpe_values)/len(sharpe_values):.4f}`"
        )

        print(f"\n【步驟3】開始回測並篩選符合條件的策略...")
        notifier.send("🔍 **步驟3開始** - 回測篩選中...")

        qualified_results = []
        rejected_count = 0
        tested_count = 0

        for idx, ind_data in enumerate(candidates, 1):
            if len(qualified_results) >= TARGET_COUNT:
                print(f"\n✅ 已找到 {TARGET_COUNT} 個符合條件的策略，停止搜尋")
                notifier.send(f"🎉 **目標達成！** 已找到 {TARGET_COUNT} 個合格策略")
                break

            print(f"\n{'='*80}")
            print(f"測試候選 {idx}/{len(candidates)} (夏普值: {ind_data['sharpe']:.4f})")
            print(f"來源: {ind_data['source']}")
            print(f"{'='*80}")

            try:
                gene = ind_data['gene']
                position_combined, params = combined_strategy(gene)

                if position_combined.empty:
                    print(f"   ⚠️ 持倉為空，跳過")
                    continue

                report = sim(
                    position=position_combined,
                    stop_loss=params.get('stop_loss', 0.25),
                    trail_stop=params.get('trail_stop', 0.35),
                    fee_ratio=FEE_RATIO,
                    tax_ratio=TAX_RATIO,
                    trade_at_price="high_low_avg",
                    position_limit=params.get('position_limit', 0.35),
                    take_profit=params.get('take_profit', 0.7),
                    name=f"候選{idx}_夏普{ind_data['sharpe']:.3f}",
                    upload=False
                )

                tested_count += 1

                if report is not None:
                    metrics = report.get_metrics()
                    actual_sharpe = metrics['ratio']['sharpeRatio']
                    max_drawdown = metrics['risk']['maxDrawdown']
                    annual_return = metrics['profitability']['annualReturn']
                    capacity = metrics.get('liquidity', {}).get('capacity', 0)

                    print(f"\n📊 回測結果:")
                    print(f"   實際夏普值: {actual_sharpe:.4f}")
                    print(f"   年化收益率: {annual_return*100:.2f}%")
                    print(f"   胃納量: {capacity/10000:.2f}萬")

                    meets_return = annual_return >= MIN_ANNUAL_RETURN
                    meets_capacity = capacity > MIN_CAPACITY

                    if meets_return and meets_capacity:
                        print(f"   ✅ 符合所有條件！")
                        rank = len(qualified_results) + 1

                        notifier.send_found_strategy(rank, idx, actual_sharpe, annual_return, capacity)
                        notifier.send_complete_performance(rank, metrics, position_combined, params)

                        report_final = sim(
                            position=position_combined,
                            stop_loss=params.get('stop_loss', 0.25),
                            trail_stop=params.get('trail_stop', 0.35),
                            fee_ratio=FEE_RATIO,
                            tax_ratio=TAX_RATIO,
                            trade_at_price="high_low_avg",
                            position_limit=params.get('position_limit', 0.35),
                            take_profit=params.get('take_profit', 0.7),
                            name=f"合格第{rank}名_夏普{ind_data['sharpe']:.3f}",
                            upload=True
                        )

                        if report_final:
                            report_final.display()

                        result = {
                            'rank': rank,
                            'candidate_idx': idx,
                            'stored_sharpe': ind_data['sharpe'],
                            'actual_sharpe': actual_sharpe,
                            'annual_return': annual_return,
                            'capacity': capacity,
                            'gene': gene,
                            'params': params,
                            'source': ind_data['source'],
                            'source_path': ind_data['path'],
                            'report': report_final if report_final else report,
                            'metrics': metrics
                        }
                        qualified_results.append(result)

                        print(f"\n📌 目前已找到 {len(qualified_results)}/{TARGET_COUNT} 個合格策略")
                    else:
                        print(f"   ❌ 不符合條件")
                        rejected_count += 1

            except Exception as e:
                print(f"❌ 回測候選 {idx} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()

        # 儲存結果
        print("\n" + "="*100)
        print(f"{'篩選總結報告':^100}")
        print("="*100)

        if qualified_results:
            result_folder = os.path.join(
                BACKTEST_RESULTS_PATH,
                f"小小龍_前5名_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            os.makedirs(result_folder, exist_ok=True)

            # 儲存基因供後續演化使用
            genes_path = os.path.join(result_folder, "top5_genes_for_evolution.pkl")
            checkpoint_data = {
                'population': [{'genes': list(r['gene']), 'fitness': r['actual_sharpe'], 'params': r['params']} for r in qualified_results],
                'generation': 0,
                'halloffame': [{'genes': list(r['gene']), 'fitness': r['actual_sharpe'], 'params': r['params']} for r in qualified_results],
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(genes_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            print(f"\n🔥 小小龍演化起點檔案已儲存:")
            print(f"   {genes_path}")
            print(f"\n📝 使用方式:")
            print(f"   os.environ['CHECKPOINT_PATH'] = '{genes_path}'")

            notifier.send(
                f"💾 **前5名基因已儲存**\n"
                f"📂 `{genes_path}`\n"
                f"可作為小小龍演化起點"
            )

            # 顯示結果
            print("\n【合格策略詳細資訊】")
            print(f"{'排名':^6} | {'夏普值':^10} | {'年化收益':^10} | {'胃納量(萬)':^12} | {'策略配置'}")
            print("-"*80)
            for result in qualified_results:
                params = result['params']
                allocation = f"低波{params['weight_lv']*100:.0f}%/小資{params['weight_si']*100:.0f}%/渦輪{params['weight_rpt']*100:.0f}%"
                print(f"{result['rank']:^6} | {result['actual_sharpe']:^10.4f} | "
                      f"{result['annual_return']*100:^10.1f}% | {result['capacity']/10000:^12.1f} | {allocation}")

            notifier.send(
                f"🏆 **小小龍篩選完成！**\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ 找到 `{len(qualified_results)}` 個合格策略\n"
                f"📊 測試總數: `{tested_count}`\n"
                f"❌ 不合格: `{rejected_count}`"
            )
        else:
            print(f"\n⚠️ 未找到符合條件的策略")
            notifier.send(f"⚠️ **未找到合格策略**\n測試了 {tested_count} 個候選")

        print(f"\n{'='*100}")
        print(f"{'小小龍篩選完成！':^100}")
        print(f"{'='*100}")

    except Exception as e:
        import traceback
        print(f"❌ 執行過程中發生錯誤: {e}")
        traceback.print_exc()
        notifier.send(f"❌ **執行錯誤**\n{str(e)[:200]}")
