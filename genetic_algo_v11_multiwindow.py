#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 六組合快快龍 基因演算法 v11.0 - 多視窗並行版
   Multi-Window Parallel Genetic Algorithm
================================================================================

【多視窗並行特色】
✅ 同時運行 N 個獨立視窗（預設 4 個）
✅ 視窗間共享最佳基因（精英遷移）
✅ 每個視窗可設定不同搜索策略
✅ 自動負載平衡與資源管理
✅ 統一進度顯示與 Discord 通知

【目標設定】
- 夏普值：>= 4.2
- 胃納量：>= 700 萬
- 最大回檔：<= 17%

【環境需求】
- Google Colab Pro+ (High-RAM)
- CPU 多核心並行

版本：v11.0 Multi-Window (2025-12-27)
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
import random
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

# 進度條
try:
    from tqdm.auto import tqdm
except ImportError:
    os.system('pip install tqdm -q')
    from tqdm.auto import tqdm

# =============================================================================
# 第二部分：核心設定
# =============================================================================
# === 🔥 請修改此處 ===
FINLAB_API_KEY = "YOUR_API_KEY_HERE"

# Discord Webhook
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"

# 🎯 優化目標
TARGET_SHARPE = 4.2
MIN_CAPACITY = 7_000_000  # 700萬
MAX_DRAWDOWN = 0.17       # 17%
TARGET_ANNUAL_RETURN = 0.6

# 🔥 多視窗設定
NUM_WINDOWS = 4           # 並行視窗數量
POPULATION_PER_WINDOW = 50  # 每視窗族群大小
N_GENERATIONS = 100       # 每視窗世代數
MIGRATION_INTERVAL = 10   # 精英遷移間隔（代）
MIGRATION_SIZE = 5        # 每次遷移精英數量

# GA 參數
MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.8
ELITE_RATIO = 0.1

# 持股配比
MIN_POSITION_WEIGHT = 0.03
POSITION_WEIGHT_STEP = 0.03

# 回測設定
BACKTEST_START = '2014-01-01'
FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000
FULL_BACKTEST_INTERVAL = 5

# 視窗搜索策略（每個視窗不同的重點）
WINDOW_STRATEGIES = {
    1: {'name': '高夏普探索', 'sharpe_weight': 1.5, 'capacity_weight': 0.8, 'mutation_rate': 0.2},
    2: {'name': '高胃納探索', 'sharpe_weight': 0.8, 'capacity_weight': 1.5, 'mutation_rate': 0.2},
    3: {'name': '均衡探索', 'sharpe_weight': 1.0, 'capacity_weight': 1.0, 'mutation_rate': 0.25},
    4: {'name': '高變異探索', 'sharpe_weight': 1.0, 'capacity_weight': 1.0, 'mutation_rate': 0.35},
}

print(f"{'='*80}")
print(f"🚀 六組合快快龍 v11.0 - 多視窗並行版")
print(f"   🖥️  並行視窗: {NUM_WINDOWS} 個")
print(f"   👥 每視窗族群: {POPULATION_PER_WINDOW}")
print(f"   🔄 精英遷移: 每 {MIGRATION_INTERVAL} 代")
print(f"   🎯 目標: 夏普 >= {TARGET_SHARPE}, 胃納 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"{'='*80}")

# =============================================================================
# 第三部分：套件載入
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
from joblib import Parallel, delayed

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 第四部分：環境設定
# =============================================================================
def setup_environment():
    global BASE_DIR, IN_COLAB

    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        BASE_DIR = '/content/drive/MyDrive/投資策略優化_v11_多視窗並行'
        IN_COLAB = True
        print("✅ Google Drive 已掛載")
    except:
        BASE_DIR = './ga_v11_multiwindow'
        IN_COLAB = False
        print("⚠️ 本地環境模式")

    if FINLAB_API_KEY != "YOUR_API_KEY_HERE":
        finlab.login(FINLAB_API_KEY)
        print("✅ FinLab VIP 登入成功")

    return BASE_DIR, IN_COLAB

BASE_DIR, IN_COLAB = setup_environment()

# 創建各視窗目錄
for i in range(1, NUM_WINDOWS + 1):
    Path(f"{BASE_DIR}/window_{i}").mkdir(parents=True, exist_ok=True)
Path(f"{BASE_DIR}/shared_elite").mkdir(parents=True, exist_ok=True)
Path(f"{BASE_DIR}/results").mkdir(parents=True, exist_ok=True)

# =============================================================================
# 第五部分：共享精英池（線程安全）
# =============================================================================
class SharedElitePool:
    """
    共享精英池 - 各視窗間交換最佳基因
    線程安全，支持並行讀寫
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.elites = []
        self.lock = threading.Lock()
        self.best_ever = None
        self.best_ever_fitness = 0

    def add_elite(self, gene: List[float], fitness: float, window_id: int, metrics: Dict = None):
        """添加精英到池中"""
        with self.lock:
            elite = {
                'gene': list(gene),
                'fitness': fitness,
                'window_id': window_id,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            # 更新最佳
            if fitness > self.best_ever_fitness:
                self.best_ever = elite
                self.best_ever_fitness = fitness
                print(f"\n🏆 全域新最佳! Window {window_id}, Fitness: {fitness:.4f}")

            # 檢查是否已存在相似基因
            gene_hash = hashlib.md5(str(gene[:20]).encode()).hexdigest()
            for e in self.elites:
                if hashlib.md5(str(e['gene'][:20]).encode()).hexdigest() == gene_hash:
                    if fitness > e['fitness']:
                        e['fitness'] = fitness
                        e['metrics'] = metrics
                    return

            # 添加新精英
            self.elites.append(elite)

            # 保持池大小
            if len(self.elites) > self.max_size:
                self.elites.sort(key=lambda x: x['fitness'], reverse=True)
                self.elites = self.elites[:self.max_size]

    def get_migrants(self, n: int, exclude_window: int = None) -> List[Dict]:
        """獲取遷移精英（排除來源視窗）"""
        with self.lock:
            candidates = [e for e in self.elites if e['window_id'] != exclude_window]
            candidates.sort(key=lambda x: x['fitness'], reverse=True)
            return candidates[:n]

    def get_best(self) -> Optional[Dict]:
        """獲取全域最佳"""
        with self.lock:
            return self.best_ever

    def get_stats(self) -> Dict:
        """獲取統計"""
        with self.lock:
            if not self.elites:
                return {'count': 0, 'best': 0, 'avg': 0}
            fitnesses = [e['fitness'] for e in self.elites]
            return {
                'count': len(self.elites),
                'best': max(fitnesses),
                'avg': np.mean(fitnesses),
                'by_window': {i: len([e for e in self.elites if e['window_id'] == i])
                              for i in range(1, NUM_WINDOWS + 1)}
            }

    def save(self, filepath: str):
        """保存到檔案"""
        with self.lock:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'elites': self.elites,
                    'best_ever': self.best_ever,
                    'timestamp': datetime.now().isoformat()
                }, f)

    def load(self, filepath: str):
        """從檔案載入"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    with self.lock:
                        self.elites = data.get('elites', [])
                        self.best_ever = data.get('best_ever')
                        if self.best_ever:
                            self.best_ever_fitness = self.best_ever['fitness']
                    print(f"✅ 載入 {len(self.elites)} 個歷史精英")
            except:
                pass

# 全域共享精英池
shared_pool = SharedElitePool(max_size=100)

# =============================================================================
# 第六部分：Discord 通知（多視窗版）
# =============================================================================
class MultiWindowNotifier:
    """多視窗 Discord 通知"""

    def __init__(self, webhook_url: str):
        self.enabled = webhook_url and webhook_url != ""
        self.webhook_url = webhook_url
        self.send_delay = 1.5
        self.lock = threading.Lock()

        if self.enabled:
            self._send("✅ 六組合快快龍 v11.0 多視窗並行系統啟動")

    def _send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            with self.lock:
                payload = {"content": message, "username": "六組合快快龍 多視窗"}
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                time.sleep(self.send_delay)
                return response.status_code in [200, 204]
        except:
            return False

    def send_embed(self, title: str, description: str, color: str = 'info', fields: List = None) -> bool:
        if not self.enabled:
            return False
        try:
            with self.lock:
                color_map = {'success': 3066993, 'error': 15158332, 'warning': 16776960, 'info': 3447003}
                embed = {
                    "title": title,
                    "description": description,
                    "color": color_map.get(color, 3447003),
                    "timestamp": datetime.utcnow().isoformat()
                }
                if fields:
                    embed["fields"] = fields
                payload = {"embeds": [embed], "username": "六組合快快龍 多視窗"}
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                time.sleep(self.send_delay)
                return response.status_code in [200, 204]
        except:
            return False

    def send_multiwindow_progress(self, window_stats: Dict[int, Dict], global_best: Dict, eta_str: str):
        """發送多視窗進度"""
        fields = []

        for wid, stats in window_stats.items():
            strategy = WINDOW_STRATEGIES.get(wid, {}).get('name', f'視窗{wid}')
            fields.append({
                "name": f"🖥️ {strategy}",
                "value": f"代數: `{stats['gen']}`\n"
                        f"最佳: `{stats['best_fitness']:.3f}`\n"
                        f"夏普: `{stats.get('best_sharpe', 0):.2f}`",
                "inline": True
            })

        # 全域最佳
        if global_best:
            fields.append({
                "name": "🏆 全域最佳",
                "value": f"Fitness: `{global_best['fitness']:.4f}`\n"
                        f"來源: 視窗 `{global_best['window_id']}`",
                "inline": False
            })

        fields.append({
            "name": "⏱️ 預估剩餘",
            "value": f"`{eta_str}`",
            "inline": True
        })

        self.send_embed(
            title="📊 多視窗演化進度",
            description="",
            color='info',
            fields=fields
        )

    def send_migration_event(self, from_windows: List[int], to_window: int, n_migrants: int):
        """發送遷移事件"""
        self._send(f"🔄 精英遷移: {n_migrants} 個精英 → 視窗 {to_window}")

    def send_new_global_best(self, metrics: Dict, window_id: int):
        """發送新全域最佳"""
        self.send_embed(
            title="🏆 發現新全域最佳！",
            description=f"來源: **視窗 {window_id}** ({WINDOW_STRATEGIES.get(window_id, {}).get('name', '')})",
            color='success',
            fields=[
                {"name": "夏普值", "value": f"`{metrics.get('sharpe', 0):.4f}`", "inline": True},
                {"name": "胃納量", "value": f"`{metrics.get('capacity', 0)/1e4:.1f}萬`", "inline": True},
                {"name": "回檔", "value": f"`{metrics.get('max_drawdown', 0)*100:.1f}%`", "inline": True},
                {"name": "年化報酬", "value": f"`{metrics.get('annual_return', 0)*100:.1f}%`", "inline": True},
            ]
        )

    def send_completion(self, best_metrics: Dict, total_time: float, window_contributions: Dict):
        """發送完成通知"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        contrib_text = "\n".join([
            f"• 視窗 {wid}: {count} 個精英"
            for wid, count in window_contributions.items()
        ])

        self.send_embed(
            title="🎉 多視窗演化完成！",
            description=f"**總耗時**: {hours}小時 {minutes}分鐘\n\n"
                       f"**各視窗貢獻**:\n{contrib_text}\n\n"
                       f"**最佳結果**:\n"
                       f"• 夏普值: `{best_metrics.get('sharpe', 0):.4f}`\n"
                       f"• 胃納量: `{best_metrics.get('capacity', 0)/1e4:.1f}萬`\n"
                       f"• 回檔: `{best_metrics.get('max_drawdown', 0)*100:.1f}%`",
            color='success'
        )

notifier = MultiWindowNotifier(DISCORD_WEBHOOK_URL)

# =============================================================================
# 第七部分：多視窗進度追蹤器
# =============================================================================
class MultiWindowProgressTracker:
    """多視窗進度追蹤器"""

    def __init__(self, num_windows: int, generations_per_window: int):
        self.num_windows = num_windows
        self.generations = generations_per_window
        self.total_work = num_windows * generations_per_window
        self.window_progress = {i: 0 for i in range(1, num_windows + 1)}
        self.window_stats = {i: {} for i in range(1, num_windows + 1)}
        self.start_time = None
        self.lock = threading.Lock()
        self.pbar = None

    def start(self):
        """開始追蹤"""
        self.start_time = time.time()
        self.pbar = tqdm(
            total=self.total_work,
            desc="🧬 多視窗演化",
            unit="代",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=100
        )

    def update_window(self, window_id: int, gen: int, stats: Dict):
        """更新視窗進度"""
        with self.lock:
            old_progress = self.window_progress[window_id]
            self.window_progress[window_id] = gen + 1
            self.window_stats[window_id] = {**stats, 'gen': gen + 1}

            # 更新總進度
            increment = (gen + 1) - old_progress
            if increment > 0 and self.pbar:
                self.pbar.update(increment)

    def get_eta(self) -> str:
        """獲取預估剩餘時間"""
        with self.lock:
            total_done = sum(self.window_progress.values())
            if total_done == 0:
                return "計算中..."

            elapsed = time.time() - self.start_time
            avg_time = elapsed / total_done
            remaining = self.total_work - total_done
            eta_seconds = avg_time * remaining

            if eta_seconds < 60:
                return f"{eta_seconds:.0f}秒"
            elif eta_seconds < 3600:
                return f"{eta_seconds/60:.1f}分"
            else:
                hours = int(eta_seconds // 3600)
                mins = int((eta_seconds % 3600) // 60)
                return f"{hours}時{mins}分"

    def get_summary(self) -> str:
        """獲取進度摘要"""
        with self.lock:
            lines = []
            for wid in range(1, self.num_windows + 1):
                stats = self.window_stats.get(wid, {})
                progress = self.window_progress[wid]
                pct = progress / self.generations * 100
                strategy = WINDOW_STRATEGIES.get(wid, {}).get('name', f'視窗{wid}')
                best = stats.get('best_fitness', 0)
                lines.append(f"   {strategy}: {progress}/{self.generations} ({pct:.0f}%) 最佳:{best:.3f}")
            return "\n".join(lines)

    def close(self) -> float:
        """關閉進度條"""
        if self.pbar:
            self.pbar.close()
        total_time = time.time() - self.start_time
        return total_time

# =============================================================================
# 第八部分：數據載入（全域共享）
# =============================================================================
print("\n📊 載入市場數據...")
data_load_start = time.time()

# 基礎數據
close = data.get('price:收盤價')
vol = data.get('price:成交股數')
open_ = data.get('price:開盤價')
high = data.get('price:最高價')
low = data.get('price:最低價')
adj_close = data.get("etl:adj_close")

pe = data.get('price_earning_ratio:本益比')
股價淨值比 = data.get("price_earning_ratio:股價淨值比")

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
limit_up_all_day = (close_high_low_open == limit_up).fillna(False)

rsi = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
atr = data.indicator('ATR', adjust_price=True, timeperiod=10)
entry_volatility = atr/adj_close

print(f"✅ 數據載入完成 ({time.time()-data_load_start:.1f}秒)")

# =============================================================================
# 第九部分：gene_to_params（與主版本相同）
# =============================================================================
def gene_to_params(gene):
    """基因解碼"""
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 160
    if len(gene) < required_length:
        gene = gene + [0.5] * (required_length - len(gene))
    gene = gene[:required_length]

    alloc_sum = sum(gene[0:6])
    allocation = [gene[i]/alloc_sum for i in range(6)] if alloc_sum > 0 else [1/6]*6

    low_vol_pe_params = {
        'rev_ma3_ma12_ratio': gene[6], 'rev_consistency': gene[7],
        'volatility_threshold': gene[8]/1000, 'margin_usage_limit': gene[9],
        'non_op_income_limit': gene[10], 'min_volume': abs(gene[11]*1000),
        'pe_min': gene[12], 'pe_max': gene[13], 'top_n': max(1, int(gene[14])),
        'min_rev_yoy_threshold': -gene[120], 'rev_decline_period': max(1, int(gene[121])),
        'max_rev_yoy_threshold': gene[122], 'old_trend_period': max(1, int(gene[123])),
        'old_trend_match': max(1, int(gene[124])), 'rev_bottom_window': max(1, int(gene[125])),
        'rev_bottom_ratio': gene[126], 'rev_bottom_sustain': max(1, int(gene[127])),
        'min_rev_mom_growth': -gene[128], 'rev_mom_sustain': max(1, int(gene[129])),
        'quarter_ma': max(1, int(gene[130])), 'half_year_ma': max(1, int(gene[131])),
        'long_ma': max(1, int(gene[132])), 'recent_rev_period': max(1, int(gene[133])),
        'annual_rev_period': max(1, int(gene[134])), 'pb_min': gene[135], 'pb_max': gene[136],
        'min_gpm': gene[137], 'gpm_sustain_period': max(1, int(gene[138])),
        'min_roe': gene[139], 'roe_sustain_period': max(1, int(gene[140]))
    }

    small_inv_params = {
        'market_value_limit': gene[15]*1e9, 'market_rev_ratio_limit': gene[16],
        'rev_yoy_growth_limit': -gene[17], 'rev_mom_growth_limit': -gene[18],
        'rsv_period': max(1, int(gene[19])), 'ma_period': max(1, int(gene[20])),
        'volume_threshold': abs(gene[21]*1000), 'top_n': max(1, int(gene[22])),
        'min_free_cash_flow': gene[150], 'min_roe': gene[151], 'min_op_profit_growth': gene[152]
    }

    turbo_params = {
        'rev_ma_period': max(1, int(gene[23])), 'rev_ma_lookback': max(1, int(gene[24])),
        'price_high_window': max(1, int(gene[25])), 'min_volume': abs(gene[26]*1000),
        'min_price': gene[27], 'rsi_threshold': gene[28], 'pe_limit': gene[29],
        'top_n': max(1, int(gene[30])), 'performance_ma_period': max(1, int(gene[100])),
        'performance_threshold': gene[101], 'rsi_trend_period': max(1, int(gene[102])),
        'min_gpm': gene[103], 'gpm_sustain_period': max(1, int(gene[104])),
        'min_btpm': gene[105], 'btpm_sustain_period': max(1, int(gene[106])),
        'min_atpm': gene[107], 'atpm_sustain_period': max(1, int(gene[108])),
        'rev_growth_percentile': gene[109], 'boss_min_level': max(1, int(gene[110])),
        'boss_max_level': max(1, int(gene[111])), 'min_boss_ratio': gene[112]
    }

    high_yield_turtle_params = {
        'min_yield_ratio': gene[31], 'min_op_earn_ratio': gene[32],
        'min_boss_hold': gene[33], 'min_volume': gene[34]*1000,
        'max_volume': gene[35]*1000, 'top_n': int(gene[36])
    }

    low_vol_index_params = {
        'min_volume': gene[37]*1000, 'std_window': int(gene[38]),
        'std_threshold': gene[39], 'top_n': int(gene[40])
    }

    market_indicator_params = {
        'new_high_window': int(gene[43]), 'min_year_growth': -gene[44],
        'max_year_growth': gene[45], 'rev_bottom_ratio': gene[46],
        'min_month_growth': -gene[47], 'min_volume': gene[48]*1000, 'top_n': int(gene[49])
    }

    overall_params = {
        'stop_loss': gene[51]/100, 'trail_stop': gene[52]/100,
        'take_profit': gene[53]/100, 'position_limit': gene[54]/100,
        'trade_at_price': ["open", "close", "high_low_avg", "open_close_avg"][int(gene[55])%4],
        'liquidity_threshold': gene[56]*1e6
    }

    return allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_turtle_params, low_vol_index_params, market_indicator_params, overall_params

# =============================================================================
# 第十部分：策略函數（簡化版，避免重複）
# =============================================================================
def strategy_low_volatility_pe(params):
    peg = pe/營業利益成長率
    cond1 = rev_ma3/rev_ma12 > params['rev_ma3_ma12_ratio']
    cond2 = rev/rev.shift(1) > params['rev_consistency']
    tree_select = ((融資使用率 <= params['margin_usage_limit']) &
                   (entry_volatility <= params['volatility_threshold']) &
                   (業外收支營收率 < params['non_op_income_limit']))
    vol_cond = vol.average(1) > params['min_volume']
    pe_range = (params['pe_min'] <= pe) & (pe <= params['pe_max'])

    cond_all = cond1 & cond2 & tree_select & vol_cond & pe_range & ~limit_up_all_day
    position = peg[cond_all & (peg > 0)].is_smallest(params['top_n'])
    return position.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_small_investor(params):
    當月營收 = data.get('monthly_revenue:當月營收') * 1000
    當季營收 = 當月營收.rolling(4).sum()
    市值營收比 = 市值 / 當季營收
    cond1 = 市值 < params['market_value_limit']
    cond5 = 市值營收比 < params['market_rev_ratio_limit']
    cond6 = vol > params['volume_threshold']
    ma_period = params['ma_period']
    cond_ma = (close > close.average(ma_period)) & (close > close.average(ma_period*2))
    rsv_period = params['rsv_period']
    rsv = (close - close.rolling(rsv_period).min()) / (close.rolling(rsv_period).max() - close.rolling(rsv_period).min())
    position = ((cond1 & cond5 & cond6 & cond_ma) * rsv).is_largest(params['top_n'])
    return position.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_revenue_price_turbo(params):
    rev_ma = rev.average(max(1, int(params['rev_ma_period'])))
    rev_ma_lookback = max(int(params['rev_ma_period'])+1, int(params['rev_ma_lookback']))
    cond_rev = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=1).max()
    cond_price = (close == close.rolling(260).max()).sustain(max(1, int(params['price_high_window'])), 1)
    cond_vol = vol.average(1) > params['min_volume']
    long_ma = (close > close.average(5)) & (close > close.average(20)) & (close > close.average(60))
    conditions = cond_rev & cond_price & cond_vol & long_ma & (close > params['min_price']) & ~limit_up_all_day
    position = (rev_yoy_growth * conditions)
    position = position[position > 0].is_largest(params['top_n'])
    return position.reindex(rev.index_str_to_date().index, method="ffill")

def strategy_high_yield_turtle(params):
    yield_ratio = data.get('price_earning_ratio:殖利率(%)')
    sma20 = close.average(20)
    sma60 = close.average(60)
    cond1 = yield_ratio >= params['min_yield_ratio']
    cond2 = (close > sma20) & (close > sma60)
    cond3 = rev.average(3) > rev.average(12)
    cond6 = (vol.average(5) >= params['min_volume']) & (vol.average(5) <= params['max_volume'])
    cond_all = (cond1 & cond2 & cond3 & cond6) * rev_yoy_growth
    position = cond_all[cond_all > 0].is_largest(params['top_n'])
    return position.reindex(rev.index_str_to_date().index, method='ffill')

def strategy_low_volatility_index(params):
    cap = data.get('etl:market_value')
    std = close.pct_change().rolling(params['std_window']).std().rank(axis=1, pct=True)
    position = cap[(vol.average(20) > params['min_volume']) &
                   (close > close.average(60)) & (close > close.average(120)) &
                   (std < params['std_threshold'])].is_smallest(params['top_n'])
    return position.reindex(close.index, method='ffill')

def strategy_market_indicator(params):
    vol_ma = vol.average(10)
    cond1 = close == close.rolling(params['new_high_window']).max()
    cond2 = ~(rev_yoy_growth < params['min_year_growth']).sustain(3)
    cond6 = vol_ma > params['min_volume']
    buy = (cond1 & cond2 & cond6) * vol_ma
    buy = buy[buy > 0].is_smallest(params['top_n'])
    return buy.reindex(rev.index_str_to_date().index, method='ffill')

# =============================================================================
# 第十一部分：持股配比正規化
# =============================================================================
def normalize_position_weights(position_df):
    """正規化持股配比（3% 倍數）"""
    result = pd.DataFrame(0.0, index=position_df.index, columns=position_df.columns)

    for date in position_df.index:
        row = position_df.loc[date]
        total = row.sum()
        if total == 0:
            continue

        normalized = row / total
        mask = normalized >= MIN_POSITION_WEIGHT
        filtered = normalized[mask]

        if filtered.empty:
            max_stocks = int(1.0 / MIN_POSITION_WEIGHT)
            top_stocks = normalized.nlargest(min(len(normalized), max_stocks))
            filtered = top_stocks / top_stocks.sum() if top_stocks.sum() > 0 else top_stocks

        if filtered.sum() == 0:
            continue

        filtered = filtered / filtered.sum()
        quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

        total_q = quantized.sum()
        if total_q > 0 and abs(total_q - 1.0) > 0.001:
            max_idx = quantized.idxmax()
            quantized[max_idx] += (1.0 - total_q)

        quantized = quantized[quantized >= MIN_POSITION_WEIGHT - 0.001]
        result.loc[date, quantized.index] = quantized

    return result

# =============================================================================
# 第十二部分：合併策略
# =============================================================================
def combined_strategy(gene, apply_normalization=True):
    """合併策略"""
    try:
        allocation, low_vol_pe_params, small_inv_params, turbo_params, high_yield_turtle_params, low_vol_index_params, market_indicator_params, overall_params = gene_to_params(gene)

        pos1 = strategy_low_volatility_pe(low_vol_pe_params)
        pos2 = strategy_small_investor(small_inv_params)
        pos3 = strategy_revenue_price_turbo(turbo_params)
        pos4 = strategy_high_yield_turtle(high_yield_turtle_params)
        pos5 = strategy_low_volatility_index(low_vol_index_params)
        pos6 = strategy_market_indicator(market_indicator_params)

        position_combined = (pos1*allocation[0] + pos2*allocation[1] + pos3*allocation[2] +
                            pos4*allocation[3] + pos5*allocation[4] + pos6*allocation[5])

        liquid_stocks = 平均成交金額 > overall_params['liquidity_threshold']
        position_combined = position_combined * liquid_stocks

        if position_combined.empty:
            return pd.DataFrame(), overall_params

        if not isinstance(position_combined.index, pd.DatetimeIndex):
            position_combined.index = pd.to_datetime(position_combined.index, errors='coerce')

        start_ts = pd.Timestamp(BACKTEST_START)
        position_combined = position_combined[position_combined.index >= start_ts]

        if apply_normalization:
            position_combined = normalize_position_weights(position_combined)

        return position_combined, overall_params

    except Exception as e:
        return pd.DataFrame(), {}

# =============================================================================
# 第十三部分：回測函數
# =============================================================================
def run_backtest(gene, upload=False, name="Strategy"):
    """執行回測"""
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
    except:
        return None

# =============================================================================
# 第十四部分：適應度函數（支持視窗策略）
# =============================================================================
def create_fitness_function(window_strategy: Dict):
    """創建視窗專屬適應度函數"""
    sharpe_weight = window_strategy.get('sharpe_weight', 1.0)
    capacity_weight = window_strategy.get('capacity_weight', 1.0)

    def evaluate(gene):
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

            # 計算適應度（根據視窗策略加權）
            sharpe_score = min(1.0, sharpe / TARGET_SHARPE) * 4.0 * sharpe_weight
            capacity_score = min(1.0, capacity / MIN_CAPACITY) * 2.5 * capacity_weight

            if max_dd <= MAX_DRAWDOWN:
                drawdown_score = 2.0
            else:
                drawdown_score = max(0, 2.0 - (max_dd - MAX_DRAWDOWN) * 10)

            return_score = min(1.0, annual_return / TARGET_ANNUAL_RETURN) * 1.5

            fitness = sharpe_score + capacity_score + drawdown_score + return_score
            return (fitness,)

        except:
            return (0.0,)

    return evaluate

# =============================================================================
# 第十五部分：單視窗演化引擎
# =============================================================================
class SingleWindowEvolver:
    """單視窗演化器"""

    def __init__(self, window_id: int, shared_pool: SharedElitePool,
                 progress_tracker: MultiWindowProgressTracker):
        self.window_id = window_id
        self.shared_pool = shared_pool
        self.progress = progress_tracker
        self.strategy = WINDOW_STRATEGIES.get(window_id, {})

        # 設定 DEAP
        self._setup_deap()

    def _setup_deap(self):
        """設定 DEAP"""
        # 為避免衝突，每個視窗使用不同的類名
        fitness_name = f"FitnessMax_W{self.window_id}"
        ind_name = f"Individual_W{self.window_id}"

        if fitness_name in dir(creator):
            delattr(creator, fitness_name)
        if ind_name in dir(creator):
            delattr(creator, ind_name)

        creator.create(fitness_name, base.Fitness, weights=(1.0,))
        creator.create(ind_name, list, fitness=getattr(creator, fitness_name))

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, getattr(creator, ind_name),
                             self.toolbox.attr_float, n=160)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", create_fitness_function(self.strategy))
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)

        mut_rate = self.strategy.get('mutation_rate', MUTATION_RATE)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                             low=0.0, up=1.0, eta=20.0, indpb=mut_rate/5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self, n_generations: int):
        """執行演化"""
        strategy_name = self.strategy.get('name', f'視窗{self.window_id}')
        print(f"\n🖥️  視窗 {self.window_id} ({strategy_name}) 開始演化...")

        # 初始化族群
        population = self.toolbox.population(n=POPULATION_PER_WINDOW)

        # 注入共享精英
        migrants = self.shared_pool.get_migrants(5, exclude_window=self.window_id)
        for i, migrant in enumerate(migrants[:min(5, len(population))]):
            population[i][:] = migrant['gene']

        # 初始評估
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        best_local = None
        best_local_fitness = 0

        # 演化循環
        for gen in range(n_generations):
            # 選擇、交叉、變異
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mutant in offspring:
                if random.random() < self.strategy.get('mutation_rate', MUTATION_RATE):
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 評估
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # 精英保留
            n_elite = max(2, int(POPULATION_PER_WINDOW * ELITE_RATIO))
            elites = tools.selBest(population, n_elite)
            population = elites + tools.selBest(offspring, POPULATION_PER_WINDOW - n_elite)

            # 統計
            fitnesses = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]

            # 更新最佳
            if best_fitness > best_local_fitness:
                best_local = list(best_ind)
                best_local_fitness = best_fitness

                # 添加到共享池
                self.shared_pool.add_elite(best_local, best_fitness, self.window_id)

            # 更新進度
            stats = {
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_sharpe': best_fitness * TARGET_SHARPE / 10.0,  # 估算
            }
            self.progress.update_window(self.window_id, gen, stats)

            # 精英遷移
            if (gen + 1) % MIGRATION_INTERVAL == 0:
                migrants = self.shared_pool.get_migrants(MIGRATION_SIZE, exclude_window=self.window_id)
                for i, migrant in enumerate(migrants[:min(MIGRATION_SIZE, len(population)-n_elite)]):
                    population[n_elite + i][:] = migrant['gene']
                    del population[n_elite + i].fitness.values

            # 每 5 代完整回測最佳
            if (gen + 1) % FULL_BACKTEST_INTERVAL == 0 and best_local:
                result = run_backtest(best_local, upload=False,
                                     name=f"W{self.window_id}_Gen{gen+1}")
                if result:
                    self.shared_pool.add_elite(
                        best_local, best_fitness, self.window_id,
                        metrics=result
                    )

                    # 如果是全域最佳，通知
                    if best_fitness >= self.shared_pool.best_ever_fitness:
                        notifier.send_new_global_best(result, self.window_id)

        return best_local, best_local_fitness

# =============================================================================
# 第十六部分：多視窗並行控制器
# =============================================================================
class MultiWindowController:
    """多視窗並行控制器"""

    def __init__(self):
        self.shared_pool = shared_pool
        self.progress = MultiWindowProgressTracker(NUM_WINDOWS, N_GENERATIONS)
        self.results = {}

    def run_all_windows(self):
        """運行所有視窗"""
        print(f"\n{'='*80}")
        print(f"🚀 啟動 {NUM_WINDOWS} 個並行視窗")
        print(f"{'='*80}")

        # 載入歷史精英
        self.shared_pool.load(f"{BASE_DIR}/shared_elite/elite_pool.pkl")

        # 開始追蹤
        self.progress.start()

        # 發送 Discord 通知
        notifier.send_embed(
            title="🚀 多視窗演化開始",
            description=f"並行視窗: {NUM_WINDOWS}\n"
                       f"每視窗族群: {POPULATION_PER_WINDOW}\n"
                       f"總世代數: {N_GENERATIONS}\n"
                       f"精英遷移: 每 {MIGRATION_INTERVAL} 代",
            color='info',
            fields=[
                {"name": f"視窗 {i}", "value": WINDOW_STRATEGIES.get(i, {}).get('name', ''), "inline": True}
                for i in range(1, NUM_WINDOWS + 1)
            ]
        )

        # 使用 ThreadPoolExecutor 並行（避免 Colab 多進程問題）
        with ThreadPoolExecutor(max_workers=NUM_WINDOWS) as executor:
            futures = {}

            for wid in range(1, NUM_WINDOWS + 1):
                evolver = SingleWindowEvolver(wid, self.shared_pool, self.progress)
                future = executor.submit(evolver.run, N_GENERATIONS)
                futures[future] = wid

            # 等待完成
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    best_gene, best_fitness = future.result()
                    self.results[wid] = {
                        'gene': best_gene,
                        'fitness': best_fitness
                    }
                    print(f"\n✅ 視窗 {wid} 完成, 最佳 Fitness: {best_fitness:.4f}")
                except Exception as e:
                    print(f"\n❌ 視窗 {wid} 錯誤: {e}")

        # 關閉進度追蹤
        total_time = self.progress.close()

        # 保存共享池
        self.shared_pool.save(f"{BASE_DIR}/shared_elite/elite_pool.pkl")

        # 最終回測全域最佳
        self._final_evaluation(total_time)

        return self.results

    def _final_evaluation(self, total_time: float):
        """最終評估"""
        print(f"\n{'='*80}")
        print("📊 最終評估全域最佳...")
        print(f"{'='*80}")

        best = self.shared_pool.get_best()

        if best:
            result = run_backtest(best['gene'], upload=True, name="MultiWindow_GlobalBest")

            if result:
                print(f"\n🏆 全域最佳結果:")
                print(f"   來源視窗: {best['window_id']}")
                print(f"   夏普值: {result['sharpe']:.4f}")
                print(f"   年化報酬: {result['annual_return']*100:.1f}%")
                print(f"   最大回檔: {result['max_drawdown']*100:.1f}%")
                print(f"   胃納量: {result['capacity']/1e4:.1f}萬")

                # 驗證持股配比
                print(f"\n🔍 持股配比驗證:")
                last_pos = result['position'].iloc[-1]
                non_zero = last_pos[last_pos > 0].sort_values(ascending=False)
                print(f"   持股數: {len(non_zero)}")
                for stock, weight in non_zero.head(10).items():
                    print(f"   {stock}: {weight*100:.1f}%")

                # 保存最佳基因
                with open(f"{BASE_DIR}/results/global_best.pkl", 'wb') as f:
                    pickle.dump({
                        'gene': best['gene'],
                        'metrics': result,
                        'window_id': best['window_id'],
                        'timestamp': datetime.now().isoformat()
                    }, f)

                # Discord 通知
                pool_stats = self.shared_pool.get_stats()
                notifier.send_completion(result, total_time, pool_stats['by_window'])

        print(f"\n💾 結果已保存至: {BASE_DIR}")

# =============================================================================
# 第十七部分：主程式
# =============================================================================
def main():
    """主程式"""
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║         六組合快快龍 基因演算法 v11.0 - 多視窗並行版                         ║
╠════════════════════════════════════════════════════════════════════════════╣
║  🖥️  並行視窗: {NUM_WINDOWS} 個                                                       ║
║  👥 每視窗族群: {POPULATION_PER_WINDOW}                                                      ║
║  🔄 精英遷移: 每 {MIGRATION_INTERVAL} 代                                                    ║
║  🎯 目標: 夏普 {TARGET_SHARPE}+, 胃納 {MIN_CAPACITY/1e4:.0f}萬+, 回檔 {MAX_DRAWDOWN*100:.0f}%以內                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║  視窗策略:                                                                  ║""")

    for wid, strategy in WINDOW_STRATEGIES.items():
        print(f"║    視窗 {wid}: {strategy['name']:<12} (夏普權重:{strategy['sharpe_weight']}, 胃納權重:{strategy['capacity_weight']})  ║")

    print(f"""╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # 創建控制器並運行
    controller = MultiWindowController()
    results = controller.run_all_windows()

    print(f"\n✅ 多視窗演化完成！")
    print(f"📁 結果保存於: {BASE_DIR}")

    return results

if __name__ == "__main__":
    results = main()
