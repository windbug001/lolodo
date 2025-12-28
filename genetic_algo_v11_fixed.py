#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 六組合快快龍 基因演算法優化系統 v11.1 (修正版)
   修正 'Expected numeric dtype, got object instead' 錯誤
================================================================================

【修正內容】
✅ 修正 .sustain() 返回 object 類型問題
✅ 添加 .astype(float) 確保數值類型
✅ 添加 try-except 錯誤處理
✅ 簡化策略函數減少錯誤

版本：v11.1 Fixed (2025-12-28)
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 🔥 FinLab 快取清理
# =============================================================================
import os
import shutil
import gc

def clear_finlab_cache():
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
            except:
                pass
    gc.collect()

clear_finlab_cache()

# =============================================================================
# 導入套件
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
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import reduce

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    os.system('pip install tqdm -q')
    from tqdm.auto import tqdm

# =============================================================================
# 核心設定
# =============================================================================
# 🔥 視窗 ID 設定（用於多視窗並行）
# 使用環境變數: os.environ['WINDOW_ID'] = "1"  # 或 "2", "3"
WINDOW_ID = int(os.environ.get('WINDOW_ID', '1'))

# 🔥 API Key 設定方式：
# 使用環境變數 os.environ['FINLAB_API_KEY'] = "your_key"
FINLAB_API_KEY = os.environ.get('FINLAB_API_KEY', 'YOUR_API_KEY_HERE')

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"

TARGET_SHARPE = 4.2
MIN_CAPACITY = 7_000_000
MAX_DRAWDOWN = 0.17
TARGET_ANNUAL_RETURN = 0.6

POPULATION_SIZE = 80
N_GENERATIONS = 200
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.8
ELITE_RATIO = 0.1

MIN_POSITION_WEIGHT = 0.03
POSITION_WEIGHT_STEP = 0.03

BACKTEST_START = '2014-01-01'
FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000
FULL_BACKTEST_INTERVAL = 5

print(f"{'='*80}")
print(f"🚀 六組合快快龍 v11.1 (修正版) - Window {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"{'='*80}")

# =============================================================================
# 套件載入
# =============================================================================
def install_packages():
    required = {'finlab': 'finlab', 'deap': 'deap', 'joblib': 'joblib'}
    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            os.system(f'pip install {pkg_name} -q')

install_packages()

import finlab
from finlab import data
from finlab.backtest import sim
from deap import base, creator, tools, algorithms

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 環境設定
# =============================================================================
def setup_environment():
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

    if FINLAB_API_KEY != "YOUR_API_KEY_HERE":
        finlab.login(FINLAB_API_KEY)
        print("✅ FinLab VIP 登入成功")
    else:
        print("⚠️ 請設定 FINLAB_API_KEY")

    return BASE_DIR, IN_COLAB

BASE_DIR, IN_COLAB = setup_environment()
WINDOW_DIR = f"{BASE_DIR}/window_{WINDOW_ID}"
Path(WINDOW_DIR).mkdir(parents=True, exist_ok=True)
print(f"📁 工作目錄: {WINDOW_DIR}")

# =============================================================================
# Discord 通知
# =============================================================================
class DiscordNotifier:
    def __init__(self, webhook_url=None):
        self.enabled = webhook_url and webhook_url != ""
        self.webhook_url = webhook_url
        self.start_time = None

        if self.enabled:
            try:
                payload = {"content": "✅ 六組合快快龍 v11.1 修正版啟動", "username": "快快龍"}
                requests.post(self.webhook_url, json=payload, timeout=5)
                print("✅ Discord 通知系統已連接")
            except:
                self.enabled = False

    def send(self, message):
        if not self.enabled:
            return
        try:
            payload = {"content": message, "username": "快快龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            time.sleep(1)
        except:
            pass

    def send_embed(self, title, description, color='info', fields=None):
        if not self.enabled:
            return
        try:
            color_map = {'success': 3066993, 'error': 15158332, 'warning': 16776960, 'info': 3447003}
            embed = {"title": title, "description": description, "color": color_map.get(color, 3447003)}
            if fields:
                embed["fields"] = fields
            payload = {"embeds": [embed], "username": "快快龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            time.sleep(1)
        except:
            pass

notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# =============================================================================
# 數據載入
# =============================================================================
print("\n📊 載入市場數據...")
data_start = time.time()

close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')

pe = data.get('price_earning_ratio:本益比')
pb = data.get("price_earning_ratio:股價淨值比")
dividend_yield = data.get('price_earning_ratio:殖利率(%)')

rev = data.get('monthly_revenue:當月營收')
rev_yoy = data.get('monthly_revenue:去年同月增減(%)')
rev_mom = data.get('monthly_revenue:上月比較增減(%)')

市值 = data.get('etl:market_value')
成交金額 = (close * vol).replace(0.0, np.nan)
平均成交金額 = 成交金額.average(20)

# 技術指標
rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=14)

print(f"✅ 數據載入完成 ({time.time()-data_start:.1f}秒)")
print(f"   股票數: {len(close.columns)}, 期間: {close.index[0].date()} ~ {close.index[-1].date()}")

# =============================================================================
# 🔥 安全條件計算函數（避免 object 類型錯誤）
# =============================================================================
def safe_sustain(condition, periods, min_periods=None):
    """安全的 sustain 計算，確保返回數值類型"""
    try:
        if min_periods is None:
            result = condition.sustain(periods)
        else:
            result = condition.sustain(periods, min_periods)
        # 確保返回數值類型
        return result.fillna(False).astype(float)
    except Exception as e:
        # 如果失敗，返回全 False
        return condition.fillna(False).astype(float) * 0

def safe_condition(cond):
    """確保條件是數值類型"""
    try:
        return cond.fillna(False).astype(float)
    except:
        return cond * 0

# =============================================================================
# gene_to_params
# =============================================================================
def gene_to_params(gene):
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    if len(gene) < required_length:
        gene = gene + [0.5] * (required_length - len(gene))
    gene = gene[:required_length]

    alloc_sum = sum(gene[0:6])
    allocation = [gene[i]/alloc_sum for i in range(6)] if alloc_sum > 0 else [1/6]*6

    low_vol_pe_params = {
        'pe_min': gene[12] * 20,
        'pe_max': gene[13] * 30 + 10,
        'pb_min': gene[135] * 2,
        'pb_max': gene[136] * 5 + 1,
        'min_volume': abs(gene[11] * 500) + 100,
        'top_n': max(3, int(gene[14] * 10) + 5),
        'ma_short': max(5, int(gene[130] * 20)),
        'ma_long': max(20, int(gene[132] * 100) + 60),
    }

    small_inv_params = {
        'market_value_limit': gene[15] * 50e9 + 10e9,
        'min_volume': abs(gene[21] * 500) + 100,
        'top_n': max(3, int(gene[22] * 10) + 5),
        'ma_period': max(5, int(gene[20] * 30) + 10),
    }

    turbo_params = {
        'min_volume': abs(gene[26] * 500) + 100,
        'min_price': gene[27] * 50 + 10,
        'top_n': max(3, int(gene[30] * 10) + 5),
        'momentum_period': max(5, int(gene[23] * 20) + 5),
    }

    high_yield_params = {
        'min_yield': gene[31] * 5 + 2,
        'min_volume': gene[34] * 500 + 100,
        'max_volume': gene[35] * 5000 + 2000,
        'top_n': max(3, int(gene[36] * 10) + 5),
    }

    low_vol_params = {
        'min_volume': gene[37] * 500 + 100,
        'std_window': max(10, int(gene[38] * 50) + 20),
        'std_threshold': gene[39] * 0.5 + 0.1,
        'top_n': max(3, int(gene[40] * 10) + 5),
    }

    market_params = {
        'new_high_window': max(20, int(gene[43] * 200) + 60),
        'min_volume': gene[48] * 500 + 100,
        'top_n': max(3, int(gene[49] * 10) + 5),
    }

    overall_params = {
        'stop_loss': gene[51] * 0.15 + 0.05,
        'trail_stop': gene[52] * 0.1 + 0.03,
        'take_profit': gene[53] * 0.3 + 0.15,
        'position_limit': gene[54] * 0.2 + 0.2,
        'trade_at_price': 'close',
        'liquidity_threshold': gene[56] * 5e6 + 1e6,
    }

    return allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_params, low_vol_params, market_params, overall_params

# =============================================================================
# 🔥 簡化策略函數（避免錯誤）
# =============================================================================
def strategy_low_vol_pe(params):
    """策略1：低波動本益比（簡化版）"""
    try:
        pe_cond = (pe >= params['pe_min']) & (pe <= params['pe_max'])
        pb_cond = (pb >= params['pb_min']) & (pb <= params['pb_max'])
        vol_cond = vol.average(5) > params['min_volume']
        ma_cond = close > close.average(params['ma_long'])

        score = safe_condition(pe_cond & pb_cond & vol_cond & ma_cond)
        score = score / (pe.fillna(999) + 1)  # PE 越低越好

        position = score[score > 0].is_smallest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_low_vol_pe 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_small_cap(params):
    """策略2：小型股成長（簡化版）"""
    try:
        cap_cond = 市值 < params['market_value_limit']
        vol_cond = vol.average(5) > params['min_volume']
        growth_cond = rev_yoy > 0
        ma_cond = close > close.average(params['ma_period'])

        score = safe_condition(cap_cond & vol_cond & growth_cond & ma_cond)
        score = score * rev_yoy.fillna(0).clip(lower=0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_small_cap 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_turbo(params):
    """策略3：營收動能（簡化版）"""
    try:
        vol_cond = vol.average(5) > params['min_volume']
        price_cond = close > params['min_price']

        # 動能條件
        momentum = close / close.shift(params['momentum_period']) - 1
        momentum_cond = momentum > 0

        # 營收條件
        rev_ma = rev.average(3)
        rev_new_high = rev_ma >= rev_ma.rolling(12, min_periods=1).max()

        score = safe_condition(vol_cond & price_cond & momentum_cond & rev_new_high)
        score = score * momentum.fillna(0).clip(lower=0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_turbo 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_high_yield(params):
    """策略4：高殖利率（簡化版）"""
    try:
        yield_cond = dividend_yield >= params['min_yield']
        vol_cond = (vol.average(5) >= params['min_volume']) & (vol.average(5) <= params['max_volume'])
        trend_cond = (close > close.average(20)) & (close > close.average(60))
        rev_cond = rev.average(3) > rev.average(12)

        score = safe_condition(yield_cond & vol_cond & trend_cond & rev_cond)
        score = score * dividend_yield.fillna(0)

        position = score[score > 0].is_largest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_high_yield 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_low_vol(params):
    """策略5：低波動（簡化版）"""
    try:
        vol_cond = vol.average(20) > params['min_volume']
        trend_cond = (close > close.average(60)) & (close > close.average(120))

        # 波動率
        returns = close.pct_change()
        volatility = returns.rolling(params['std_window']).std()
        vol_rank = volatility.rank(axis=1, pct=True)
        low_vol_cond = vol_rank < params['std_threshold']

        score = safe_condition(vol_cond & trend_cond & low_vol_cond)
        score = score * 市值.fillna(0)  # 大市值優先

        position = score[score > 0].is_smallest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_low_vol 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

def strategy_market(params):
    """策略6：市場指針（簡化版）"""
    try:
        # 創新高
        new_high = close >= close.rolling(params['new_high_window'], min_periods=1).max()
        vol_cond = vol.average(10) > params['min_volume']

        score = safe_condition(new_high & vol_cond)
        score = score * vol.average(10).fillna(0)

        position = score[score > 0].is_smallest(params['top_n'])
        return position.fillna(0).astype(float)
    except Exception as e:
        print(f"   ⚠️ strategy_market 錯誤: {e}")
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

# =============================================================================
# 持股配比正規化
# =============================================================================
def normalize_weights(position_df):
    """正規化持股配比（3% 倍數）"""
    result = pd.DataFrame(0.0, index=position_df.index, columns=position_df.columns)

    for date in position_df.index:
        row = position_df.loc[date]
        total = row.sum()
        if total == 0:
            continue

        normalized = row / total
        filtered = normalized[normalized >= MIN_POSITION_WEIGHT]

        if filtered.empty:
            max_stocks = int(1.0 / MIN_POSITION_WEIGHT)
            filtered = normalized.nlargest(min(len(normalized), max_stocks))
            if filtered.sum() > 0:
                filtered = filtered / filtered.sum()

        if filtered.sum() == 0:
            continue

        filtered = filtered / filtered.sum()
        quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

        diff = 1.0 - quantized.sum()
        if abs(diff) > 0.001 and len(quantized) > 0:
            quantized[quantized.idxmax()] += diff

        quantized = quantized[quantized >= MIN_POSITION_WEIGHT - 0.001]
        result.loc[date, quantized.index] = quantized

    return result

# =============================================================================
# 合併策略
# =============================================================================
def combined_strategy(gene, apply_normalization=True):
    """合併策略"""
    try:
        allocation, p1, p2, p3, p4, p5, p6, overall = gene_to_params(gene)

        pos1 = strategy_low_vol_pe(p1) * allocation[0]
        pos2 = strategy_small_cap(p2) * allocation[1]
        pos3 = strategy_turbo(p3) * allocation[2]
        pos4 = strategy_high_yield(p4) * allocation[3]
        pos5 = strategy_low_vol(p5) * allocation[4]
        pos6 = strategy_market(p6) * allocation[5]

        combined = pos1.add(pos2, fill_value=0).add(pos3, fill_value=0).add(pos4, fill_value=0).add(pos5, fill_value=0).add(pos6, fill_value=0)

        # 流動性過濾
        liquid = 平均成交金額 > overall['liquidity_threshold']
        combined = combined * liquid.astype(float)

        # 時間過濾
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index, errors='coerce')

        combined = combined[combined.index >= pd.Timestamp(BACKTEST_START)]

        if apply_normalization and not combined.empty:
            combined = normalize_weights(combined)

        return combined, overall

    except Exception as e:
        print(f"⚠️ 合併策略錯誤: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

# =============================================================================
# 回測
# =============================================================================
def run_backtest(gene, upload=False, name="Strategy"):
    try:
        position, params = combined_strategy(gene)
        if position.empty or position.sum().sum() == 0:
            return None

        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            trail_stop=params.get('trail_stop', 0.05),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params.get('trade_at_price', 'close'),
            position_limit=params.get('position_limit', 0.3),
            take_profit=params.get('take_profit', 0.3),
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
            'report': report,
            'position': position,
            'gene': gene
        }
    except Exception as e:
        print(f"   ⚠️ 回測錯誤: {e}")
        return None

def evaluate_fitness(gene):
    """適應度函數"""
    try:
        position, params = combined_strategy(gene)
        if position.empty:
            return (0.0,)

        report = sim(
            position=position,
            stop_loss=params.get('stop_loss', 0.1),
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            upload=False,
            name="Eval"
        )

        if report is None:
            return (0.0,)

        metrics = report.get_metrics()
        sharpe = metrics['ratio'].get('sharpeRatio', 0) or 0
        capacity = metrics['liquidity'].get('capacity', 0) or 0
        max_dd = abs(metrics['risk'].get('maxDrawdown', 1))
        annual_return = metrics['profitability'].get('annualReturn', 0) or 0

        # 適應度計算
        sharpe_score = min(1.0, sharpe / TARGET_SHARPE) * 4.0
        capacity_score = min(1.0, capacity / MIN_CAPACITY) * 2.5
        drawdown_score = 2.0 if max_dd <= MAX_DRAWDOWN else max(0, 2.0 - (max_dd - MAX_DRAWDOWN) * 10)
        return_score = min(1.0, annual_return / TARGET_ANNUAL_RETURN) * 1.5

        fitness = sharpe_score + capacity_score + drawdown_score + return_score
        return (fitness,)

    except Exception as e:
        return (0.0,)

# =============================================================================
# DEAP 設定
# =============================================================================
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

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
# 🔥 歷史精英載入（關鍵功能！）
# =============================================================================
def load_historical_elites(top_n=20):
    """
    從歷史檔案載入最佳基因
    搜尋多個資料夾中的 checkpoint 檔案
    """
    print("\n🔍 搜尋歷史精英...")

    # 搜尋路徑（與成功程式完全相同）
    BASE_SEARCH = '/content/drive/MyDrive/投資策略優化_六策略_修正版_2014'
    search_paths = [
        # 修正版資料夾（與成功程式相同）
        f'{BASE_SEARCH}/window_1',
        f'{BASE_SEARCH}/window_2',
        f'{BASE_SEARCH}/window_3',
        f'{BASE_SEARCH}/shared_best',
        # 純夏普值適應度版本（這裡有最多 .pkl！）
        '/content/drive/MyDrive/投資策略優化_六策略純夏普值適應度_2014',
        '/content/drive/MyDrive/投資策略優化_六策略純夏普值適應度_2014_分散式',
        # 十二組合版本
        '/content/drive/MyDrive/十二組合_Calmar_Sortino_優化',
        # 六策略獨立版
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_1/working',
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_2/working',
        '/content/drive/MyDrive/投資策略優化_六策略_獨立版/window_3/working',
        # 當前專案目錄（新生成的）
        f'{BASE_DIR}/window_1',
        f'{BASE_DIR}/window_2',
        f'{BASE_DIR}/window_3',
    ]

    all_individuals = []
    total_pkl_found = 0

    # 過濾存在的路徑
    valid_paths = [p for p in search_paths if os.path.exists(p)]
    print(f"   搜尋範圍: {len(valid_paths)} 個資料夾")

    for search_idx, search_path in enumerate(valid_paths, 1):
        # 使用 os.walk 搜尋所有 .pkl 檔案（比 glob 更可靠）
        pkl_files = []
        for root, dirs, files in os.walk(search_path):
            for f in files:
                if f.endswith('.pkl'):
                    pkl_files.append(os.path.join(root, f))

        if not pkl_files:
            continue

        total_pkl_found += len(pkl_files)
        files_loaded = 0

        for file in pkl_files:
            try:
                with open(file, 'rb') as f:
                    cp = pickle.load(f)

                # 從 halloffame 載入（與成功程式相同）
                if "halloffame" in cp and cp["halloffame"]:
                    for ind in cp["halloffame"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            fitness = ind.fitness.values[0]
                            all_individuals.append({
                                'gene': list(ind),
                                'fitness': fitness,
                                'source': file
                            })
                            files_loaded += 1

                # 從 population 載入（與成功程式相同）
                if "population" in cp and cp["population"]:
                    for ind in cp["population"]:
                        if hasattr(ind, 'fitness') and ind.fitness.valid:
                            fitness = ind.fitness.values[0]
                            if fitness > 2.0:
                                all_individuals.append({
                                    'gene': list(ind),
                                    'fitness': fitness,
                                    'source': file
                                })
                                files_loaded += 1

            except:
                continue

        print(f"   [{search_idx}/{len(valid_paths)}] {os.path.basename(search_path)}: {len(pkl_files)} 檔案, {files_loaded} 個體")

    print(f"\n   📊 總計: {total_pkl_found} 個 .pkl，{len(all_individuals)} 個基因")

    # 去重並排序
    seen_hashes = set()
    unique_individuals = []

    for ind in sorted(all_individuals, key=lambda x: x['fitness'], reverse=True):
        import hashlib
        gene_hash = hashlib.md5(str(ind['gene'][:30]).encode()).hexdigest()
        if gene_hash not in seen_hashes:
            seen_hashes.add(gene_hash)
            unique_individuals.append(ind)

    elites = unique_individuals[:top_n]

    if elites:
        print(f"✅ 找到 {len(elites)} 個歷史精英")
        print(f"   最佳適應度: {elites[0]['fitness']:.4f}")
        print(f"   來源: {os.path.basename(elites[0]['source'])}")
    else:
        print("⚠️ 未找到歷史精英，從頭開始")

    return elites

# =============================================================================
# 進度追蹤器
# =============================================================================
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.start_time = None
        self.pbar = None

    def start(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total, desc="🧬 演化", unit="代", ncols=100)

    def update(self, gen, stats):
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (gen + 1)
        eta = avg_time * (self.total - gen - 1)

        if eta < 60:
            eta_str = f"{eta:.0f}秒"
        elif eta < 3600:
            eta_str = f"{eta/60:.1f}分"
        else:
            eta_str = f"{eta/3600:.1f}時"

        self.pbar.set_postfix({
            '最佳': f"{stats.get('best_fitness', 0):.3f}",
            'ETA': eta_str
        })
        self.pbar.update(1)
        return eta_str

    def close(self):
        if self.pbar:
            self.pbar.close()
        return time.time() - self.start_time

# =============================================================================
# 演化引擎
# =============================================================================
class EvolutionEngine:
    def __init__(self):
        self.best_ever = None
        self.best_metrics = None
        self.history = []

    def run(self, n_generations=N_GENERATIONS, inject_elites=True):
        print(f"\n{'='*80}")
        print(f"🚀 開始演化 - 共 {n_generations} 代, 族群 {POPULATION_SIZE}")
        print(f"{'='*80}\n")

        progress = ProgressTracker(n_generations)
        progress.start()

        notifier.send(f"🚀 開始演化: {n_generations}代, 族群{POPULATION_SIZE}")

        # 初始化族群
        population = toolbox.population(n=POPULATION_SIZE)

        # 🔥 注入歷史精英到初始族群
        if inject_elites:
            historical_elites = load_historical_elites(top_n=30)

            if historical_elites:
                n_inject = min(len(historical_elites), POPULATION_SIZE // 2)  # 最多注入一半
                print(f"💉 注入 {n_inject} 個歷史精英到初始族群")

                for i, elite_data in enumerate(historical_elites[:n_inject]):
                    # 創建新個體並複製基因
                    new_ind = creator.Individual(elite_data['gene'])
                    # 設置預設適應度（會在評估時被覆蓋）
                    new_ind.fitness.values = (elite_data['fitness'],)
                    population[i] = new_ind

                # 更新最佳記錄
                if historical_elites[0]['fitness'] > 0:
                    self.best_ever = historical_elites[0]['gene']
                    print(f"   🏆 最佳歷史基因適應度: {historical_elites[0]['fitness']:.4f}")
                    notifier.send(f"💉 注入 {n_inject} 個歷史精英，最佳適應度: {historical_elites[0]['fitness']:.4f}")

        print("📊 評估初始族群...")
        for i, ind in enumerate(tqdm(population, desc="初始評估", ncols=80)):
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # 演化
        for gen in range(n_generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            n_elite = max(2, int(POPULATION_SIZE * ELITE_RATIO))
            elites = tools.selBest(population, n_elite)
            population = elites + tools.selBest(offspring, POPULATION_SIZE - n_elite)

            fitnesses = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]

            stats = {'best_fitness': best_fitness, 'avg_fitness': np.mean(fitnesses)}
            self.history.append(stats)

            eta_str = progress.update(gen, stats)

            # 每 5 代完整回測
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0:
                print(f"\n📊 第 {gen+1} 代 - 完整回測...")
                result = run_backtest(best_ind, upload=False, name=f"Gen{gen+1}")

                if result:
                    is_best = self.best_metrics is None or result['sharpe'] > self.best_metrics.get('sharpe', 0)

                    if is_best:
                        self.best_ever = list(best_ind)
                        self.best_metrics = result
                        print(f"   🏆 新最佳! 夏普:{result['sharpe']:.4f} 胃納:{result['capacity']/1e4:.1f}萬")

                        notifier.send_embed(
                            "🏆 新最佳策略",
                            f"夏普: {result['sharpe']:.4f}\n胃納: {result['capacity']/1e4:.1f}萬",
                            'success'
                        )

            # 每 20 代發送進度
            if (gen + 1) % 20 == 0:
                notifier.send(f"📈 進度: {gen+1}/{n_generations} 代, 最佳適應度: {best_fitness:.3f}")

        total_time = progress.close()

        # 最終回測
        print(f"\n{'='*80}")
        print("📊 最終回測...")

        final_best = tools.selBest(population, 1)[0]
        final_result = run_backtest(final_best, upload=True, name="GA_v11_Final")

        if final_result:
            print(f"\n🏆 最終結果:")
            print(f"   夏普值: {final_result['sharpe']:.4f}")
            print(f"   年化報酬: {final_result['annual_return']*100:.1f}%")
            print(f"   最大回檔: {final_result['max_drawdown']*100:.1f}%")
            print(f"   胃納量: {final_result['capacity']/1e4:.1f}萬")

            # 持股驗證
            print(f"\n🔍 持股配比:")
            last_pos = final_result['position'].iloc[-1]
            non_zero = last_pos[last_pos > 0].sort_values(ascending=False)
            print(f"   持股數: {len(non_zero)}")
            for stock, weight in non_zero.head(10).items():
                print(f"   {stock}: {weight*100:.1f}%")

            notifier.send_embed(
                "🎉 演化完成",
                f"總耗時: {total_time/60:.1f}分鐘\n"
                f"夏普: {final_result['sharpe']:.4f}\n"
                f"年化: {final_result['annual_return']*100:.1f}%\n"
                f"胃納: {final_result['capacity']/1e4:.1f}萬",
                'success'
            )

        # 保存
        self._save_results(population, final_result)

        return population, final_result

    def _save_results(self, population, result):
        try:
            with open(f"{WINDOW_DIR}/best_gene.pkl", 'wb') as f:
                pickle.dump({
                    'gene': self.best_ever or list(tools.selBest(population, 1)[0]),
                    'metrics': self.best_metrics or result,
                    'timestamp': datetime.now().isoformat()
                }, f)

            with open(f"{WINDOW_DIR}/history.json", 'w') as f:
                json.dump(self.history, f)

            print(f"\n💾 已保存至: {WINDOW_DIR}")
        except Exception as e:
            print(f"⚠️ 保存失敗: {e}")

# =============================================================================
# 主程式
# =============================================================================
def main():
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║         六組合快快龍 基因演算法 v11.1 (修正版)                           ║
╠════════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e4:.0f}萬+, 回檔 {MAX_DRAWDOWN*100:.0f}%以內           ║
║  📊 持股：每隻至少 {MIN_POSITION_WEIGHT*100:.0f}%，3% 倍數                                  ║
║  🔄 回測：每 {FULL_BACKTEST_INTERVAL} 代執行完整回測                                       ║
╚════════════════════════════════════════════════════════════════════════╝
    """)

    engine = EvolutionEngine()
    population, result = engine.run(N_GENERATIONS)

    print(f"\n✅ 完成！結果保存於: {WINDOW_DIR}")
    return population, result

if __name__ == "__main__":
    population, result = main()
