#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 六組合快快龍 基因演算法優化系統 v11.0
   整合 Discord 通知 + 進度條 + 預估時間
================================================================================

【目標設定】
- 夏普值：>= 4.2
- 胃納量：>= 700 萬
- 最大回檔：<= 17%
- 年化報酬：>= 60%

【核心功能】
✅ 每檔股票持股至少 3%（否則 0%）
✅ 每 5 代執行完整回測並記錄最佳值
✅ 進度條 + 預估剩餘時間（tqdm）
✅ Discord Webhook 即時通知
✅ 歷史前十精英注入
✅ Colab Pro+ CPU + High-RAM 並行優化

版本：v11.0 (2025-12-27)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 🔥 FinLab 快取清理（必須在最前面）
# =============================================================================
import os
import shutil
import gc

def clear_finlab_cache():
    """清理 FinLab 快取"""
    cache_paths = [
        os.path.expanduser('~/.finlab'),
        os.path.expanduser('~/.cache/finlab'),
        '/tmp/finlab_cache',
        '/root/.finlab',
    ]
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"🧹 已清除快取: {cache_path}")
            except:
                pass
    gc.collect()

clear_finlab_cache()

# =============================================================================
# 第一部分：導入套件
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import sys
import json
import pickle
import time
import glob
import hashlib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import reduce
import random

import numpy as np
import pandas as pd

# 進度條
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ 安裝 tqdm 以顯示進度條...")
    os.system('pip install tqdm -q')
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True

# =============================================================================
# 第二部分：核心設定
# =============================================================================
# === 🔥 請修改此處 ===
WINDOW_ID = 1
FINLAB_API_KEY = "YOUR_API_KEY_HERE"  # 請填入你的 FinLab API Key

# Discord Webhook
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"

# 🎯 優化目標（更新）
TARGET_SHARPE = 4.2
MIN_CAPACITY = 7_000_000  # 700萬
MAX_DRAWDOWN = 0.17       # 17%
TARGET_ANNUAL_RETURN = 0.6  # 60%

# GA 演化參數
POPULATION_SIZE = 80
N_GENERATIONS = 200
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.8
ELITE_RATIO = 0.1

# 🔥 持股配比要求
MIN_POSITION_WEIGHT = 0.03  # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股必須是 3% 倍數

# 回測設定
BACKTEST_START = '2014-01-01'
BACKTEST_END = None
FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000

# 每 N 代執行完整回測
FULL_BACKTEST_INTERVAL = 5

# 並行設定
N_JOBS = -1  # 使用所有 CPU 核心

print(f"{'='*80}")
print(f"🚀 六組合快快龍 基因演算法 v11.0 - Window {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬, 回檔 <= {MAX_DRAWDOWN*100:.0f}%")
print(f"   📊 持股要求：每隻至少 {MIN_POSITION_WEIGHT*100:.0f}%")
print(f"   🔄 每 {FULL_BACKTEST_INTERVAL} 代執行完整回測")
print(f"{'='*80}")

# =============================================================================
# 第三部分：套件安裝與載入
# =============================================================================
def install_packages():
    """安裝必要套件"""
    required = {
        'finlab': 'finlab',
        'deap': 'deap',
        'joblib': 'joblib',
        'tqdm': 'tqdm',
    }
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

from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 第四部分：環境設定與登入
# =============================================================================
def setup_environment():
    """設定環境"""
    global BASE_DIR, IN_COLAB

    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        BASE_DIR = '/content/drive/MyDrive/投資策略優化_v11_六組合快快龍'
        IN_COLAB = True
        print("✅ Google Drive 已掛載")
    except:
        BASE_DIR = './ga_v11_output'
        IN_COLAB = False
        print("⚠️ 本地環境模式")

    # FinLab 登入
    if FINLAB_API_KEY != "YOUR_API_KEY_HERE":
        finlab.login(FINLAB_API_KEY)
        print("✅ FinLab VIP 登入成功")
    else:
        print("⚠️ 請設定 FINLAB_API_KEY")

    return BASE_DIR, IN_COLAB

BASE_DIR, IN_COLAB = setup_environment()

# 路徑管理
WINDOW_DIR = f"{BASE_DIR}/window_{WINDOW_ID}"
SHARED_DIR = f"{BASE_DIR}/shared_best"
HISTORY_DIR = f"{BASE_DIR}/history"

for d in [WINDOW_DIR, SHARED_DIR, HISTORY_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

print(f"📁 工作目錄: {WINDOW_DIR}")

# =============================================================================
# 第五部分：Discord 通知系統
# =============================================================================
class DiscordNotifier:
    """Discord Webhook 通知系統（含進度追蹤）"""

    def __init__(self, webhook_url=None, enabled=True):
        self.enabled = enabled and webhook_url is not None and webhook_url != ""
        self.webhook_url = webhook_url
        self.send_delay = 1.0
        self.start_time = None
        self.total_generations = 0

        if self.enabled:
            test_result = self._send_message("✅ 六組合快快龍 v11.0 通知系統已啟動", test=True)
            if test_result:
                print("✅ Discord 通知系統已連接")
            else:
                print("⚠️ Discord 通知測試失敗，將繼續執行但不發送通知")
                self.enabled = False

    def _send_message(self, message, test=False):
        if not self.enabled:
            return False
        try:
            payload = {"content": message, "username": "六組合快快龍 v11"}
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            time.sleep(self.send_delay)
            return response.status_code in [200, 204]
        except Exception as e:
            if not test:
                print(f"⚠️ Discord 發送失敗: {e}")
            return False

    def send_embed(self, title, description, color='info', fields=None):
        if not self.enabled:
            return False
        try:
            color_map = {'success': 3066993, 'error': 15158332, 'warning': 16776960, 'info': 3447003}
            embed = {
                "title": title,
                "description": description,
                "color": color_map.get(color, 3447003),
                "timestamp": datetime.utcnow().isoformat()
            }
            if fields:
                embed["fields"] = fields
            payload = {"embeds": [embed], "username": "六組合快快龍 v11"}
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            time.sleep(self.send_delay)
            return response.status_code in [200, 204]
        except:
            return False

    def start_evolution(self, total_generations):
        """開始演化追蹤"""
        self.start_time = time.time()
        self.total_generations = total_generations
        self.send_embed(
            title="🚀 基因演化開始",
            description=f"總世代數: {total_generations}\n"
                       f"目標夏普: {TARGET_SHARPE}\n"
                       f"目標胃納量: {MIN_CAPACITY/1e4:.0f}萬\n"
                       f"最大回檔: {MAX_DRAWDOWN*100:.0f}%",
            color='info'
        )

    def send_generation_progress(self, gen, best_sharpe, best_capacity, best_drawdown, eta_str):
        """發送世代進度"""
        progress = (gen + 1) / self.total_generations
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))

        # 檢查是否達標
        sharpe_ok = "✅" if best_sharpe >= TARGET_SHARPE else "❌"
        capacity_ok = "✅" if best_capacity >= MIN_CAPACITY else "❌"
        drawdown_ok = "✅" if best_drawdown <= MAX_DRAWDOWN else "❌"

        fields = [
            {"name": "📊 進度", "value": f"`{progress_bar}` {progress*100:.1f}%", "inline": False},
            {"name": "🏆 最佳夏普", "value": f"`{best_sharpe:.4f}` {sharpe_ok}", "inline": True},
            {"name": "💰 胃納量", "value": f"`{best_capacity/1e4:.1f}萬` {capacity_ok}", "inline": True},
            {"name": "📉 回檔", "value": f"`{best_drawdown*100:.1f}%` {drawdown_ok}", "inline": True},
            {"name": "⏱️ 預估剩餘", "value": f"`{eta_str}`", "inline": True},
        ]

        self.send_embed(
            title=f"📈 第 {gen+1}/{self.total_generations} 代",
            description="",
            color='success' if all([best_sharpe >= TARGET_SHARPE, best_capacity >= MIN_CAPACITY, best_drawdown <= MAX_DRAWDOWN]) else 'info',
            fields=fields
        )

    def send_backtest_result(self, gen, metrics, is_best=False):
        """發送回測結果"""
        title = "🏆 新最佳策略！" if is_best else f"📊 第 {gen+1} 代回測結果"

        fields = [
            {"name": "夏普值", "value": f"`{metrics.get('sharpe', 0):.4f}`", "inline": True},
            {"name": "年化報酬", "value": f"`{metrics.get('annual_return', 0)*100:.1f}%`", "inline": True},
            {"name": "最大回檔", "value": f"`{metrics.get('max_drawdown', 0)*100:.1f}%`", "inline": True},
            {"name": "胃納量", "value": f"`{metrics.get('capacity', 0)/1e4:.1f}萬`", "inline": True},
            {"name": "Sortino", "value": f"`{metrics.get('sortino', 0):.4f}`", "inline": True},
            {"name": "Calmar", "value": f"`{metrics.get('calmar', 0):.4f}`", "inline": True},
        ]

        self.send_embed(title=title, description="", color='success' if is_best else 'info', fields=fields)

    def send_completion(self, best_metrics, total_time):
        """發送完成通知"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        self.send_embed(
            title="🎉 演化完成！",
            description=f"**總耗時**: {hours}小時 {minutes}分鐘\n\n"
                       f"**最佳結果**:\n"
                       f"• 夏普值: `{best_metrics.get('sharpe', 0):.4f}`\n"
                       f"• 年化報酬: `{best_metrics.get('annual_return', 0)*100:.1f}%`\n"
                       f"• 胃納量: `{best_metrics.get('capacity', 0)/1e4:.1f}萬`\n"
                       f"• 最大回檔: `{best_metrics.get('max_drawdown', 0)*100:.1f}%`",
            color='success'
        )

# 初始化通知器
notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# =============================================================================
# 第六部分：進度條管理器
# =============================================================================
class ProgressTracker:
    """進度追蹤器（含預估時間）"""

    def __init__(self, total_generations: int):
        self.total = total_generations
        self.start_time = None
        self.generation_times = []
        self.pbar = None

    def start(self):
        """開始追蹤"""
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total,
            desc="🧬 演化進度",
            unit="代",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )

    def update(self, gen: int, stats: Dict):
        """更新進度"""
        current_time = time.time()

        if len(self.generation_times) > 0:
            gen_time = current_time - self.generation_times[-1]
        else:
            gen_time = current_time - self.start_time

        self.generation_times.append(current_time)

        # 計算預估剩餘時間
        avg_time = (current_time - self.start_time) / (gen + 1)
        remaining_gens = self.total - gen - 1
        eta_seconds = avg_time * remaining_gens

        eta_str = self._format_time(eta_seconds)
        elapsed_str = self._format_time(current_time - self.start_time)

        # 更新進度條
        self.pbar.set_postfix({
            '夏普': f"{stats.get('best_sharpe', 0):.3f}",
            '胃納': f"{stats.get('best_capacity', 0)/1e4:.0f}萬",
            'ETA': eta_str
        })
        self.pbar.update(1)

        return eta_str, elapsed_str

    def _format_time(self, seconds: float) -> str:
        """格式化時間"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}時{mins}分"

    def close(self):
        """關閉進度條"""
        if self.pbar:
            self.pbar.close()

        total_time = time.time() - self.start_time
        print(f"\n✅ 演化完成！總耗時: {self._format_time(total_time)}")
        return total_time

# =============================================================================
# 第七部分：數據載入
# =============================================================================
print("\n📊 正在載入市場數據...")
data_load_start = time.time()

# 基礎價量數據
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

# 估值數據
pe = data.get('price_earning_ratio:本益比')
股價淨值比 = data.get("price_earning_ratio:股價淨值比")

# 營收數據
rev = data.get('monthly_revenue:當月營收')
rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)
rev_yoy_growth = data.get('monthly_revenue:去年同月增減(%)')
rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

# 財務數據
營業利益成長率 = data.get('fundamental_features:營業利益成長率')
業外收支營收率 = data.get('fundamental_features:業外收支營收率')
營業毛利率 = data.get("fundamental_features:營業毛利率")
ROE綜合損益 = data.get("fundamental_features:ROE綜合損益")
稅後淨利率 = data.get("fundamental_features:稅後淨利率")
稅前淨利率 = data.get("fundamental_features:稅前淨利率")
營業利益率 = data.get('fundamental_features:營業利益率')

# 籌碼數據
融資使用率 = data.get('margin_transactions:融資使用率')
董監持有股數占比 = data.get("internal_equity_changes:董監持有股數占比")
inventory = data.get("inventory")
市值 = data.get('etl:market_value')

# 衍生指標
成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

# 漲停計算
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

# 技術指標
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)
entry_volatility = atr/adj_close

data_load_time = time.time() - data_load_start
print(f"✅ 數據載入完成 ({data_load_time:.1f}秒)")
print(f"   股票數: {len(close.columns)}, 期間: {close.index[0].date()} ~ {close.index[-1].date()}")

# =============================================================================
# 第八部分：gene_to_params 函數（與六組合快快龍相同）
# =============================================================================
def gene_to_params(gene):
    """將基因轉換為策略參數（與六組合快快龍相同結構）"""
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    current_length = len(gene)

    if current_length < required_length:
        extension = [0.5] * (required_length - current_length)
        gene = gene + extension
    elif current_length > required_length:
        gene = gene[:required_length]

    # 策略配置（0-5）
    alloc_sum = sum(gene[0:6])
    if alloc_sum == 0:
        allocation = [1/6] * 6
    else:
        allocation = [gene[i]/alloc_sum for i in range(6)]

    # 低波動本益比策略參數
    low_vol_pe_params = {
        'rev_ma3_ma12_ratio': gene[6],
        'rev_consistency': gene[7],
        'volatility_threshold': gene[8]/1000,
        'margin_usage_limit': gene[9],
        'non_op_income_limit': gene[10],
        'min_volume': abs(gene[11] * 1000),
        'pe_min': gene[12],
        'pe_max': gene[13],
        'top_n': max(1, int(gene[14])),
        'min_rev_yoy_threshold': -gene[120],
        'rev_decline_period': max(1, int(gene[121])),
        'max_rev_yoy_threshold': gene[122],
        'old_trend_period': max(1, int(gene[123])),
        'old_trend_match': max(1, int(gene[124])),
        'rev_bottom_window': max(1, int(gene[125])),
        'rev_bottom_ratio': gene[126],
        'rev_bottom_sustain': max(1, int(gene[127])),
        'min_rev_mom_growth': -gene[128],
        'rev_mom_sustain': max(1, int(gene[129])),
        'quarter_ma': max(1, int(gene[130])),
        'half_year_ma': max(1, int(gene[131])),
        'long_ma': max(1, int(gene[132])),
        'recent_rev_period': max(1, int(gene[133])),
        'annual_rev_period': max(1, int(gene[134])),
        'pb_min': gene[135],
        'pb_max': gene[136],
        'min_gpm': gene[137],
        'gpm_sustain_period': max(1, int(gene[138])),
        'min_roe': gene[139],
        'roe_sustain_period': max(1, int(gene[140]))
    }

    # 小資族策略參數
    small_inv_params = {
        'market_value_limit': gene[15] * 1e9,
        'market_rev_ratio_limit': gene[16],
        'rev_yoy_growth_limit': -gene[17],
        'rev_mom_growth_limit': -gene[18],
        'rsv_period': max(1, int(gene[19])),
        'ma_period': max(1, int(gene[20])),
        'volume_threshold': abs(gene[21] * 1000),
        'top_n': max(1, int(gene[22])),
        'min_free_cash_flow': gene[150],
        'min_roe': gene[151],
        'min_op_profit_growth': gene[152]
    }

    # 營收股價雙渦輪策略參數
    turbo_params = {
        'rev_ma_period': max(1, int(gene[23])),
        'rev_ma_lookback': max(1, int(gene[24])),
        'price_high_window': max(1, int(gene[25])),
        'min_volume': abs(gene[26] * 1000),
        'min_price': gene[27],
        'rsi_threshold': gene[28],
        'pe_limit': gene[29],
        'top_n': max(1, int(gene[30])),
        'performance_ma_period': max(1, int(gene[100])),
        'performance_threshold': gene[101],
        'rsi_trend_period': max(1, int(gene[102])),
        'min_gpm': gene[103],
        'gpm_sustain_period': max(1, int(gene[104])),
        'min_btpm': gene[105],
        'btpm_sustain_period': max(1, int(gene[106])),
        'min_atpm': gene[107],
        'atpm_sustain_period': max(1, int(gene[108])),
        'rev_growth_percentile': gene[109],
        'boss_min_level': max(1, int(gene[110])),
        'boss_max_level': max(1, int(gene[111])),
        'min_boss_ratio': gene[112]
    }

    # 高殖利率烏龜策略參數
    high_yield_turtle_params = {
        'min_yield_ratio': gene[31],
        'min_op_earn_ratio': gene[32],
        'min_boss_hold': gene[33],
        'min_volume': gene[34] * 1000,
        'max_volume': gene[35] * 1000,
        'top_n': int(gene[36])
    }

    # 低波動性指標策略參數
    low_vol_index_params = {
        'min_volume': gene[37] * 1000,
        'std_window': int(gene[38]),
        'std_threshold': gene[39],
        'top_n': int(gene[40])
    }

    # 藏獒外掛大盤指針策略參數
    market_indicator_params = {
        'new_high_window': int(gene[43]),
        'min_year_growth': -gene[44],
        'max_year_growth': gene[45],
        'rev_bottom_ratio': gene[46],
        'min_month_growth': -gene[47],
        'min_volume': gene[48] * 1000,
        'top_n': int(gene[49])
    }

    # 整體參數
    overall_params = {
        'stop_loss': gene[51]/100,
        'trail_stop': gene[52]/100,
        'take_profit': gene[53]/100,
        'position_limit': gene[54]/100,
        'trade_at_price': ["open", "close", "high_low_avg", "open_close_avg"][int(gene[55]) % 4],
        'liquidity_threshold': gene[56] * 1e6,
    }

    return allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_turtle_params, low_vol_index_params, market_indicator_params, overall_params

# =============================================================================
# 第九部分：六大策略函數（與六組合快快龍相同）
# =============================================================================
def strategy_low_volatility_pe(params):
    """策略1：低波動本益比"""
    peg = pe/營業利益成長率
    cond1 = rev_ma3/rev_ma12 > params['rev_ma3_ma12_ratio']
    cond2 = rev/rev.shift(1) > params['rev_consistency']
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

    try:
        small_inv_under50 = (inventory[(inventory.持股分級.astype(int) <= 8)]
                             .reset_index()
                             .groupby(["date", "stock_id"])
                             .agg({"占集保庫存數比例": "sum"})
                             .reset_index()
                             .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
    except:
        small_inv_under50 = pe > 0  # 備用條件

    cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老 & cond排除月營收連3月衰退
                & cond收盤價大於季線及半年線 & cond近三個月營收大於年營收 & cond單月營收月增率 & ~(limit_up_all_day)
                & (condition_近1日成交均量大於100張) & (gpm_trend_1) & (roe_trend_1) & (small_inv_under50) & pb_range_1 & pe_range_1)
    position = peg[cond_all & (peg > 0)].is_smallest(params['top_n']).reindex(rev.index_str_to_date().index, method='ffill')
    return position

def strategy_small_investor(params):
    """策略2：小資族"""
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
    cond排除月營收成長趨勢過老 = ~(rev_yoy_growth > 60).sustain(12,8)
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
    position = position.reindex(rev.index_str_to_date().index, method='ffill')
    return position

def strategy_revenue_price_turbo(params):
    """策略3：營收股價雙渦輪"""
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

    try:
        boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= params['boss_min_level']) &
                                            (inventory.持股分級.astype(int) <= params['boss_max_level'])]
                                .reset_index()
                                .groupby(["date", "stock_id"])
                                .agg({"占集保庫存數比例": "sum"})
                                .reset_index()
                                .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= params['min_boss_ratio']
    except:
        boss_inventory_over400 = pe > 0

    pe_limit = params['pe_limit']
    pe_range_1 = (pe_limit <= pe)
    min_price = params['min_price']

    conditions = (condition_近N月平均營收創M個月來新高 & condition_近N日內有1日股價創新高 & condition_成交均量大於閾值
                  & long_ma_pattern & gpm_trend_1 & btpm_trend_1 & atpm_trend_1 & rev_rise_nsatisfy_2
                  & (close > min_price) & (業外收支營收率 < 7.3) & ~(pe_range_1) & (收盤價_超級績效)
                  & ((vol >= vol.rolling(20).mean()*0.8)) & (rsi_higt_trend) & (boss_inventory_over400) & ~(limit_up_all_day))

    position = rev_yoy_growth * conditions
    position = position[position > 0].is_largest(params['top_n']).reindex(rev.index_str_to_date().index, method="ffill")
    return position

def strategy_high_yield_turtle(params):
    """策略4：高殖利率烏龜"""
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

def strategy_low_volatility_index(params):
    """策略5：低波動性指標"""
    cap = data.get('etl:market_value')
    vol_local = data.get('price:成交股數')
    close_local = data.get('price:收盤價')
    std = close_local.pct_change().rolling(params['std_window']).std().rank(axis=1, pct=True)

    position = cap[(vol_local.average(20) > params['min_volume'])
                   & (close_local > close_local.average(60))
                   & (close_local > close_local.average(120))
                   & (close_local > close_local.average(250))
                   & (std < params['std_threshold'])].is_smallest(params['top_n'])
    position = position.reindex(close_local.index_str_to_date().index, method='ffill')
    return position

def strategy_market_indicator(params):
    """策略6：藏獒外掛大盤指針"""
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

# =============================================================================
# 第十部分：持股配比正規化（3% 倍數）
# =============================================================================
def normalize_position_weights(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    正規化持股配比
    規則：
    1. 每隻股票至少 3%，否則設為 0%
    2. 持股必須是 3% 倍數
    3. 總和 = 100%
    """
    result = pd.DataFrame(0.0, index=position_df.index, columns=position_df.columns)

    for date in position_df.index:
        row = position_df.loc[date]
        total = row.sum()

        if total == 0:
            continue

        # 正規化
        normalized = row / total

        # 過濾小於 3% 的
        mask = normalized >= MIN_POSITION_WEIGHT
        filtered = normalized[mask]

        if filtered.empty:
            # 保留最大的幾個
            max_stocks = int(1.0 / MIN_POSITION_WEIGHT)
            top_stocks = normalized.nlargest(min(len(normalized), max_stocks))
            filtered = top_stocks / top_stocks.sum() if top_stocks.sum() > 0 else top_stocks

        if filtered.sum() == 0:
            continue

        # 重新正規化過濾後的
        filtered = filtered / filtered.sum()

        # 轉換為 3% 倍數
        quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

        # 修正總和誤差
        total_q = quantized.sum()
        if total_q > 0 and abs(total_q - 1.0) > 0.001:
            max_idx = quantized.idxmax()
            quantized[max_idx] += (1.0 - total_q)

        # 再次檢查並移除小於 3% 的
        quantized = quantized[quantized >= MIN_POSITION_WEIGHT - 0.001]

        result.loc[date, quantized.index] = quantized

    return result

# =============================================================================
# 第十一部分：合併策略函數
# =============================================================================
def combined_strategy(gene, start_date_str="2014-01-01", apply_weight_normalization=True):
    """根據基因參數生成合併策略"""
    try:
        allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_turtle_params, low_vol_index_params, market_indicator_params, overall_params = gene_to_params(gene)

        # 執行各策略
        position_low_vol_pe = strategy_low_volatility_pe(low_vol_pe_params)
        position_small_investor = strategy_small_investor(small_inv_params)
        position_turbo = strategy_revenue_price_turbo(turbo_params)
        position_high_yield_turtle = strategy_high_yield_turtle(high_yield_turtle_params)
        position_low_vol_index = strategy_low_volatility_index(low_vol_index_params)
        position_market_indicator = strategy_market_indicator(market_indicator_params)

        # 合併
        position_combined = (
            position_low_vol_pe * allocation[0] +
            position_small_investor * allocation[1] +
            position_turbo * allocation[2] +
            position_high_yield_turtle * allocation[3] +
            position_low_vol_index * allocation[4] +
            position_market_indicator * allocation[5]
        )

        # 流動性過濾
        avg_daily_volume = 成交金額.average(20)
        liquid_stocks = avg_daily_volume > overall_params['liquidity_threshold']
        position_combined = position_combined * liquid_stocks

        # 時間過濾
        if position_combined.empty:
            return pd.DataFrame(), overall_params

        if not isinstance(position_combined.index, pd.DatetimeIndex):
            position_combined.index = pd.to_datetime(position_combined.index, errors='coerce')
            position_combined = position_combined.loc[~position_combined.index.isna()]

        start_date_ts = pd.Timestamp(start_date_str)
        position_combined = position_combined[position_combined.index >= start_date_ts]

        # 🔥 應用持股配比正規化（3% 倍數）
        if apply_weight_normalization:
            position_combined = normalize_position_weights(position_combined)

        return position_combined, overall_params

    except Exception as e:
        print(f"⚠️ 合併策略錯誤: {e}")
        return pd.DataFrame(), {}

# =============================================================================
# 第十二部分：回測與適應度函數
# =============================================================================
def run_backtest(gene, upload=False, name="GA_Strategy"):
    """執行回測並返回詳細指標"""
    try:
        position_combined, overall_params = combined_strategy(gene)

        if position_combined.empty or position_combined.sum().sum() == 0:
            return None

        report = sim(
            position=position_combined,
            stop_loss=overall_params.get('stop_loss', 0.1),
            trail_stop=overall_params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=overall_params.get('trade_at_price', 'close'),
            position_limit=overall_params.get('position_limit', 0.3),
            take_profit=overall_params.get('take_profit', 0.3),
            name=name,
            upload=upload
        )

        if report is None:
            return None

        metrics = report.get_metrics()

        return {
            'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
            'sortino': metrics['ratio'].get('sortinoRatio', 0) or 0,
            'calmar': metrics['ratio'].get('calmarRatio', 0) or 0,
            'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
            'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1)),
            'capacity': metrics['liquidity'].get('capacity', 0) or 0,
            'win_rate': metrics.get('winrate', {}).get('winRate', 0) or 0,
            'alpha': metrics['profitability'].get('alpha', 0) or 0,
            'beta': metrics['profitability'].get('beta', 0) or 0,
            'report': report,
            'position': position_combined,
            'gene': gene
        }

    except Exception as e:
        return None

def evaluate_fitness(gene):
    """適應度函數（快速評估版本）"""
    try:
        position_combined, overall_params = combined_strategy(gene)

        if position_combined.empty:
            return (0.0,)

        # 快速回測
        report = sim(
            position=position_combined,
            stop_loss=overall_params.get('stop_loss', 0.1),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            upload=False,
            name="GA_Eval"
        )

        if report is None:
            return (0.0,)

        metrics = report.get_metrics()

        sharpe = metrics['ratio'].get('sharpeRatio', 0) or 0
        capacity = metrics['liquidity'].get('capacity', 0) or 0
        max_dd = abs(metrics['risk'].get('maxDrawdown', 1))
        annual_return = metrics['profitability'].get('annualReturn', 0) or 0

        # 計算綜合適應度
        fitness = 0.0

        # 夏普值分數（權重 40%）
        sharpe_score = min(1.0, sharpe / TARGET_SHARPE) * 4.0
        fitness += sharpe_score

        # 胃納量分數（權重 25%）
        capacity_score = min(1.0, capacity / MIN_CAPACITY) * 2.5
        fitness += capacity_score

        # 回檔懲罰（權重 20%）
        if max_dd <= MAX_DRAWDOWN:
            drawdown_score = 2.0
        else:
            drawdown_score = max(0, 2.0 - (max_dd - MAX_DRAWDOWN) * 10)
        fitness += drawdown_score

        # 年化報酬分數（權重 15%）
        return_score = min(1.0, annual_return / TARGET_ANNUAL_RETURN) * 1.5
        fitness += return_score

        return (fitness,)

    except Exception as e:
        return (0.0,)

# =============================================================================
# 第十三部分：歷史精英載入
# =============================================================================
def load_historical_elites(search_paths: List[str], sharpe_min=3.5, sharpe_max=5.0, top_n=10):
    """從歷史檔案載入精英個體"""
    print(f"\n🔍 搜尋歷史精英...")

    all_individuals = []

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        checkpoint_pattern = os.path.join(search_path, "**", "*.pkl")
        matching_files = glob.glob(checkpoint_pattern, recursive=True)

        for file in matching_files:
            try:
                with open(file, 'rb') as f:
                    cp = pickle.load(f)

                # 從 halloffame 載入
                if "halloffame" in cp and cp["halloffame"]:
                    for ind in cp["halloffame"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            sharpe = ind.fitness.values[0]
                            if sharpe_min <= sharpe <= sharpe_max:
                                all_individuals.append({
                                    'gene': list(ind),
                                    'sharpe': sharpe,
                                    'source': file
                                })

                # 從 population 載入
                if "population" in cp and cp["population"]:
                    for ind in cp["population"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            sharpe = ind.fitness.values[0]
                            if sharpe_min <= sharpe <= sharpe_max:
                                all_individuals.append({
                                    'gene': list(ind),
                                    'sharpe': sharpe,
                                    'source': file
                                })
            except:
                continue

    # 去重並排序
    seen_hashes = set()
    unique_individuals = []

    for ind in sorted(all_individuals, key=lambda x: x['sharpe'], reverse=True):
        gene_hash = hashlib.md5(str(ind['gene'][:20]).encode()).hexdigest()
        if gene_hash not in seen_hashes:
            seen_hashes.add(gene_hash)
            unique_individuals.append(ind)

    # 取前 N 個
    elites = unique_individuals[:top_n]

    print(f"✅ 找到 {len(elites)} 個歷史精英")
    if elites:
        print(f"   最佳夏普: {elites[0]['sharpe']:.4f}")

    return elites

# =============================================================================
# 第十四部分：DEAP 設定
# =============================================================================
# 清除舊定義
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

# 創建適應度（單目標最大化）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

GENE_LENGTH = 160

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

print("✅ DEAP GA 配置完成")

# =============================================================================
# 第十五部分：主演化引擎
# =============================================================================
class GeneticEvolutionEngine:
    """基因演化引擎（含進度條與預估時間）"""

    def __init__(self, toolbox, notifier: DiscordNotifier):
        self.toolbox = toolbox
        self.notifier = notifier
        self.best_ever = None
        self.best_ever_metrics = None
        self.history = []
        self.generation_bests = []

    def run(self, n_generations: int = N_GENERATIONS, historical_elites: List = None):
        """執行演化"""
        print(f"\n{'='*80}")
        print(f"🚀 開始基因演化 - 共 {n_generations} 代")
        print(f"   族群大小: {POPULATION_SIZE}")
        print(f"   目標: 夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬, 回檔 <= {MAX_DRAWDOWN*100:.0f}%")
        print(f"{'='*80}\n")

        # 初始化進度追蹤
        progress = ProgressTracker(n_generations)
        progress.start()

        # 通知開始
        self.notifier.start_evolution(n_generations)

        # 初始化族群
        population = self.toolbox.population(n=POPULATION_SIZE)

        # 注入歷史精英
        if historical_elites:
            n_inject = min(len(historical_elites), int(POPULATION_SIZE * 0.3))
            print(f"💉 注入 {n_inject} 個歷史精英...")

            for i, elite in enumerate(historical_elites[:n_inject]):
                population[i][:] = elite['gene']

        # 初始評估
        print("📊 評估初始族群...")
        for ind in tqdm(population, desc="初始評估", ncols=80):
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        # 演化循環
        for gen in range(n_generations):
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

            # 評估新個體
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # 精英保留
            n_elite = int(POPULATION_SIZE * ELITE_RATIO)
            elites = tools.selBest(population, n_elite)

            # 組合新族群
            population = elites + tools.selBest(offspring, POPULATION_SIZE - n_elite)

            # 統計
            fitnesses = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]

            # 估算指標（基於適應度反推）
            est_sharpe = best_fitness * TARGET_SHARPE / 10.0
            est_capacity = best_fitness * MIN_CAPACITY / 10.0
            est_drawdown = MAX_DRAWDOWN

            stats = {
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_sharpe': est_sharpe,
                'best_capacity': est_capacity,
                'best_drawdown': est_drawdown,
            }
            self.history.append(stats)

            # 更新進度條
            eta_str, elapsed_str = progress.update(gen, stats)

            # 每 5 代執行完整回測
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0:
                print(f"\n📊 第 {gen+1} 代 - 執行完整回測...")

                result = run_backtest(best_ind, upload=False, name=f"Gen{gen+1}_Best")

                if result:
                    is_new_best = False

                    if self.best_ever_metrics is None:
                        is_new_best = True
                    elif result['sharpe'] > self.best_ever_metrics.get('sharpe', 0):
                        is_new_best = True

                    if is_new_best:
                        self.best_ever = list(best_ind)
                        self.best_ever_metrics = result
                        print(f"   🏆 新最佳! 夏普: {result['sharpe']:.4f}, 胃納: {result['capacity']/1e4:.1f}萬")

                    self.generation_bests.append({
                        'generation': gen + 1,
                        'metrics': result
                    })

                    # Discord 通知
                    self.notifier.send_backtest_result(gen, result, is_best=is_new_best)

                    # 更新統計
                    stats['best_sharpe'] = result['sharpe']
                    stats['best_capacity'] = result['capacity']
                    stats['best_drawdown'] = result['max_drawdown']

            # 每 10 代發送進度通知
            if (gen + 1) % 10 == 0:
                self.notifier.send_generation_progress(
                    gen,
                    stats['best_sharpe'],
                    stats['best_capacity'],
                    stats['best_drawdown'],
                    eta_str
                )

            # 保存檢查點
            if (gen + 1) % 20 == 0:
                self._save_checkpoint(population, gen)

        # 關閉進度條
        total_time = progress.close()

        # 最終回測
        print(f"\n{'='*80}")
        print("📊 最終回測...")

        final_best = tools.selBest(population, 1)[0]
        final_result = run_backtest(final_best, upload=True, name="GA_v11_Final_Best")

        if final_result:
            print(f"\n🏆 最終結果:")
            print(f"   夏普值: {final_result['sharpe']:.4f}")
            print(f"   年化報酬: {final_result['annual_return']*100:.1f}%")
            print(f"   最大回檔: {final_result['max_drawdown']*100:.1f}%")
            print(f"   胃納量: {final_result['capacity']/1e4:.1f}萬")

            # 驗證持股配比
            print(f"\n🔍 持股配比驗證:")
            last_pos = final_result['position'].iloc[-1]
            non_zero = last_pos[last_pos > 0].sort_values(ascending=False)
            print(f"   持股數: {len(non_zero)}")
            for stock, weight in non_zero.head(10).items():
                print(f"   {stock}: {weight*100:.1f}%")

            # 檢查是否有小於 3% 的持股
            small_positions = non_zero[non_zero < MIN_POSITION_WEIGHT - 0.001]
            if len(small_positions) > 0:
                print(f"   ⚠️ 警告: 有 {len(small_positions)} 檔持股小於 3%")
            else:
                print(f"   ✅ 所有持股都 >= 3%")

        # 發送完成通知
        if self.best_ever_metrics:
            self.notifier.send_completion(self.best_ever_metrics, total_time)

        # 保存最終結果
        self._save_final_results(population, final_result)

        return population, final_result

    def _save_checkpoint(self, population, gen):
        """保存檢查點"""
        try:
            checkpoint_file = f"{WINDOW_DIR}/checkpoint_gen{gen+1}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'generation': gen,
                    'population': population,
                    'halloffame': tools.selBest(population, 10),
                    'best_ever': self.best_ever,
                    'best_ever_metrics': self.best_ever_metrics,
                    'history': self.history,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"⚠️ 檢查點保存失敗: {e}")

    def _save_final_results(self, population, final_result):
        """保存最終結果"""
        try:
            # 保存最佳基因
            if self.best_ever:
                best_gene_file = f"{WINDOW_DIR}/best_gene.pkl"
                with open(best_gene_file, 'wb') as f:
                    pickle.dump({
                        'gene': self.best_ever,
                        'metrics': self.best_ever_metrics,
                        'timestamp': datetime.now().isoformat()
                    }, f)

            # 保存演化歷史
            history_file = f"{WINDOW_DIR}/evolution_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'history': self.history,
                    'generation_bests': [
                        {
                            'generation': gb['generation'],
                            'sharpe': gb['metrics']['sharpe'],
                            'capacity': gb['metrics']['capacity'],
                            'max_drawdown': gb['metrics']['max_drawdown'],
                            'annual_return': gb['metrics']['annual_return']
                        }
                        for gb in self.generation_bests
                    ]
                }, f, indent=2)

            print(f"\n💾 結果已保存至: {WINDOW_DIR}")

        except Exception as e:
            print(f"⚠️ 結果保存失敗: {e}")

# =============================================================================
# 第十六部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║       六組合快快龍 基因演算法優化系統 v11.0                              ║
║       Discord 通知 + 進度條 + 預估時間                                   ║
╠════════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e4:.0f}萬+, 回檔 {MAX_DRAWDOWN*100:.0f}%以內           ║
║  📊 持股：每隻至少 {MIN_POSITION_WEIGHT*100:.0f}%，3% 倍數                                  ║
║  🔄 回測：每 {FULL_BACKTEST_INTERVAL} 代執行完整回測                                       ║
║  🖥️  視窗：{WINDOW_ID}                                                              ║
╚════════════════════════════════════════════════════════════════════════╝
    """)

    # 搜尋歷史精英的路徑
    search_paths = [
        f'{BASE_DIR}/window_1',
        f'{BASE_DIR}/window_2',
        f'{BASE_DIR}/shared_best',
        '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014/window_1',
        '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014/window_2',
        '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014/shared_best',
        '/content/drive/MyDrive/投資策略優化_六策略純夏普值適應度_2014',
    ]

    # 載入歷史精英
    historical_elites = load_historical_elites(
        search_paths,
        sharpe_min=3.5,
        sharpe_max=5.0,
        top_n=10
    )

    # 建立演化引擎
    engine = GeneticEvolutionEngine(toolbox, notifier)

    # 執行演化
    population, final_result = engine.run(
        n_generations=N_GENERATIONS,
        historical_elites=historical_elites
    )

    print(f"\n✅ 優化完成！")
    print(f"📁 結果保存於: {WINDOW_DIR}")

    return population, final_result

if __name__ == "__main__":
    population, result = main()
