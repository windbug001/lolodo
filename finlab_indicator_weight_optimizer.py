#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 FinLab 台股指標權重優化系統 v1.0
FinLab Taiwan Stock Indicator Weight Optimizer
================================================================================

【核心功能】
✅ 基因演算法優化指標權重組合
✅ 預載所有指標數據，記憶體快取回測
✅ 並行回測加速（joblib）
✅ 中斷恢復機制（Checkpoint）
✅ In-Sample / Out-of-Sample 驗證
✅ 每 10 代完整回測與視覺化

【優化目標】
- 夏普值 (Sharpe Ratio) 最大化
- 資金胃納量 >= 500 萬
- 持倉數量：3-30 檔

【指標類別】
1. 基本面：ROE、營收成長率、毛利率、本益比等
2. 技術面：RSI、均線乖離、成交量等
3. 籌碼面：法人買賣超、融資餘額等

版本：v1.0 (2025-12-27)
環境：Google Colab Pro+ (High-RAM, 多核心 CPU)
================================================================================
"""

from __future__ import annotations

import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 第一部分：核心設定
# =============================================================================
import sys
import json
import pickle
import time
import random
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# === 核心設定 ===
WINDOW_ID = int(os.environ.get('WINDOW_ID', 1))
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# 優化目標
TARGET_SHARPE = 3.0
MIN_CAPACITY = 5_000_000  # 500 萬
MIN_HOLDINGS = 3          # 最少持股數
MAX_HOLDINGS = 30         # 最多持股數
MIN_WEIGHT_PER_STOCK = 0.03  # 單一標的最小權重 3%

# GA 演化參數
POPULATION_SIZE = 40      # 種群大小
N_GENERATIONS = 50        # 演化代數
MUTATION_RATE = 0.25      # 突變率
CROSSOVER_RATE = 0.7      # 交叉率
ELITE_RATIO = 0.1         # 精英保留比例
TOURNAMENT_SIZE = 3       # 錦標賽選擇大小

# 並行設定
N_JOBS = -1               # 使用所有 CPU 核心 (-1)
CHECKPOINT_INTERVAL = 10  # 每 N 代存檔

# 回測時間設定
IN_SAMPLE_START = '2017-01-01'
IN_SAMPLE_END = '2022-12-31'
OUT_SAMPLE_START = '2023-01-01'
OUT_SAMPLE_END = None  # 到最新

# 選股數量
TOP_N_STOCKS = 15

print(f"=" * 80)
print(f"🧬 FinLab 台股指標權重優化系統 v1.0 - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e4:.0f}萬")
print(f"   🔄 每 {CHECKPOINT_INTERVAL} 代完整回測")
print(f"=" * 80)

# =============================================================================
# 第二部分：套件載入
# =============================================================================
def clear_finlab_cache():
    """清除 FinLab 快取"""
    import glob
    import shutil
    import subprocess

    print("🔧 清除快取...")
    try:
        subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-type', 'f', '-delete'],
                      stderr=subprocess.DEVNULL, timeout=5)
    except:
        pass

    cache_patterns = ['/root/.finlab*', '/tmp/.finlab*', '/tmp/finlab*']
    for pattern in cache_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except:
            pass

clear_finlab_cache()

# 安裝套件
def install_packages():
    required = ['finlab', 'deap', 'joblib', 'tqdm']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            os.system(f'pip install {pkg} -q')

install_packages()

import finlab
from finlab import data
from finlab.backtest import sim

from deap import base, creator, tools
from joblib import Parallel, delayed
from tqdm import tqdm

print(f"✅ 套件載入完成 - FinLab {finlab.__version__}")

# =============================================================================
# 第三部分：環境設定
# =============================================================================
def setup_environment():
    """設定環境"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_dir = '/content/drive/MyDrive/FinLab_Indicator_Optimizer'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except:
        base_dir = './finlab_indicator_output'
        in_colab = False
        print("⚠️ 本地環境")

    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab 登入成功")

    return base_dir, in_colab

BASE_DIR, IN_COLAB = setup_environment()

@dataclass
class PathManager:
    """路徑管理器"""
    base_dir: str = BASE_DIR
    window_id: int = WINDOW_ID

    def __post_init__(self):
        self.output_dir = f"{self.base_dir}/window_{self.window_id}"
        self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        self.results_dir = f"{self.output_dir}/results"

        for d in [self.output_dir, self.checkpoint_dir, self.results_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> str:
        return f"{self.checkpoint_dir}/ga_checkpoint.pkl"

    @property
    def best_weights_file(self) -> str:
        return f"{self.results_dir}/best_weights.json"

paths = PathManager()
print(f"📁 工作目錄: {paths.output_dir}")

# =============================================================================
# 第四部分：DataCenter - 資料中心
# =============================================================================
class DataCenter:
    """
    資料中心 - 預載所有指標數據

    設計原則：
    1. 所有數據在初始化時一次性載入
    2. 回測循環中只從記憶體存取，不再調用 API
    3. 數據經過標準化處理（Z-score 或分位數）
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DataCenter._initialized:
            self._cache = {}
            self._indicator_names = []
            self._load_all_data()
            DataCenter._initialized = True

    def _safe_get(self, key: str, retry: bool = True) -> pd.DataFrame:
        """安全載入數據"""
        try:
            return data.get(key)
        except Exception as e:
            if retry:
                print(f"   ⚠️ 重試載入: {key}")
                time.sleep(1)
                return self._safe_get(key, retry=False)
            else:
                print(f"   ❌ 載入失敗: {key} - {e}")
                return pd.DataFrame()

    def _load_all_data(self):
        """載入所有數據"""
        print("\n📊 載入指標數據...")
        start_time = time.time()

        # === 基礎價量數據 ===
        self._cache['close'] = self._safe_get('price:收盤價')
        self._cache['open'] = self._safe_get('price:開盤價')
        self._cache['high'] = self._safe_get('price:最高價')
        self._cache['low'] = self._safe_get('price:最低價')
        self._cache['volume'] = self._safe_get('price:成交股數')
        self._cache['adj_close'] = self._safe_get('etl:adj_close')
        self._cache['market_value'] = self._safe_get('etl:market_value')

        # === 基本面指標 ===
        print("   載入基本面指標...")
        fundamental_indicators = {
            # 獲利能力
            'roe': 'fundamental_features:ROE綜合損益',
            'roa': 'fundamental_features:ROA綜合損益',
            'gross_margin': 'fundamental_features:營業毛利率',
            'operating_margin': 'fundamental_features:營業利益率',
            'net_margin': 'fundamental_features:稅後淨利率',

            # 成長性
            'rev_yoy': 'monthly_revenue:去年同月增減(%)',
            'rev_mom': 'monthly_revenue:上月比較增減(%)',
            'op_income_growth': 'fundamental_features:營業利益成長率',

            # 估值
            'pe': 'price_earning_ratio:本益比',
            'pb': 'price_earning_ratio:股價淨值比',
            'dividend_yield': 'price_earning_ratio:殖利率(%)',

            # 財務健全
            'debt_ratio': 'fundamental_features:負債比率',
            'current_ratio': 'fundamental_features:流動比率',
            'quick_ratio': 'fundamental_features:速動比率',

            # 其他
            'non_op_income_ratio': 'fundamental_features:業外收支營收率',
        }

        for name, key in fundamental_indicators.items():
            try:
                self._cache[name] = self._safe_get(key)
                self._indicator_names.append(name)
            except Exception as e:
                print(f"      ⚠️ 跳過 {name}: {e}")

        # === 技術面指標 ===
        print("   載入技術面指標...")
        close = self._cache['close']
        volume = self._cache['volume']

        # RSI
        try:
            self._cache['rsi_5'] = data.indicator('RSI', timeperiod=5)
            self._cache['rsi_14'] = data.indicator('RSI', timeperiod=14)
            self._indicator_names.extend(['rsi_5', 'rsi_14'])
        except:
            pass

        # 均線乖離
        for period in [5, 20, 60, 120]:
            name = f'ma_bias_{period}'
            try:
                ma = close.average(period)
                self._cache[name] = (close - ma) / ma * 100
                self._indicator_names.append(name)
            except:
                pass

        # 成交量變化
        try:
            self._cache['vol_ratio_5'] = volume / volume.average(5)
            self._cache['vol_ratio_20'] = volume / volume.average(20)
            self._indicator_names.extend(['vol_ratio_5', 'vol_ratio_20'])
        except:
            pass

        # 價格動量
        for period in [5, 20, 60]:
            name = f'momentum_{period}'
            try:
                self._cache[name] = close.pct_change(period) * 100
                self._indicator_names.append(name)
            except:
                pass

        # 波動率 (ATR)
        try:
            self._cache['atr'] = data.indicator('ATR', timeperiod=14)
            self._cache['atr_ratio'] = self._cache['atr'] / close * 100
            self._indicator_names.extend(['atr_ratio'])
        except:
            pass

        # === 籌碼面指標 ===
        print("   載入籌碼面指標...")
        chip_indicators = {
            'margin_balance': 'margin_transactions:融資餘額',
            'margin_ratio': 'margin_transactions:融資使用率',
            'short_balance': 'margin_transactions:融券餘額',
        }

        for name, key in chip_indicators.items():
            try:
                self._cache[name] = self._safe_get(key)
                self._indicator_names.append(name)
            except:
                pass

        # 法人買賣超
        try:
            foreign = self._safe_get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
            investment_trust = self._safe_get('institutional_investors_trading_summary:投信買賣超股數')

            if not foreign.empty:
                self._cache['foreign_net'] = foreign
                self._indicator_names.append('foreign_net')
            if not investment_trust.empty:
                self._cache['trust_net'] = investment_trust
                self._indicator_names.append('trust_net')
        except:
            pass

        # === 數據標準化 ===
        print("   數據標準化處理...")
        self._standardize_indicators()

        elapsed = time.time() - start_time
        print(f"\n✅ 數據載入完成 ({elapsed:.1f}s)")
        print(f"   📈 共 {len(self._indicator_names)} 個指標")
        print(f"   📋 指標列表: {', '.join(self._indicator_names[:10])}...")

    def _standardize_indicators(self):
        """
        指標標準化處理

        方法：橫截面 Z-score 標準化
        - 每個時間點，對所有股票計算 Z-score
        - Z = (X - mean) / std
        """
        for name in self._indicator_names:
            if name in self._cache and not self._cache[name].empty:
                df = self._cache[name]
                # 橫截面標準化
                mean = df.mean(axis=1)
                std = df.std(axis=1)
                self._cache[f'{name}_zscore'] = df.sub(mean, axis=0).div(std + 1e-10, axis=0)

    def get(self, key: str) -> pd.DataFrame:
        """獲取數據"""
        return self._cache.get(key, pd.DataFrame())

    def get_indicator(self, name: str, standardized: bool = True) -> pd.DataFrame:
        """獲取指標（可選標準化版本）"""
        if standardized:
            zscore_key = f'{name}_zscore'
            if zscore_key in self._cache:
                return self._cache[zscore_key]
        return self._cache.get(name, pd.DataFrame())

    @property
    def indicator_names(self) -> List[str]:
        """獲取所有指標名稱"""
        return self._indicator_names.copy()

    @property
    def n_indicators(self) -> int:
        """指標數量"""
        return len(self._indicator_names)

# 初始化資料中心
data_center = DataCenter()

# =============================================================================
# 第五部分：StrategyGA - 遺傳演算法策略
# =============================================================================
class StrategyGA:
    """
    遺傳演算法策略類別

    基因編碼：
    - 長度 = 指標數量
    - 每個元素為 [-1, 1] 的浮點數，代表該指標的係數
    - 正數表示「越大越好」，負數表示「越小越好」

    適應度函數：
    - 主目標：In-Sample Sharpe Ratio
    - 懲罰項：持倉數量、流動性、集中度
    """

    def __init__(self, data_center: DataCenter):
        self.dc = data_center
        self.n_genes = data_center.n_indicators
        self.indicator_names = data_center.indicator_names

        # 回測結果快取
        self._fitness_cache = {}

    def decode_weights(self, genes: List[float]) -> Dict[str, float]:
        """
        解碼基因為指標權重

        Args:
            genes: 基因序列 [-1, 1]

        Returns:
            指標權重字典
        """
        weights = {}
        for i, name in enumerate(self.indicator_names):
            if i < len(genes):
                weights[name] = genes[i]
            else:
                weights[name] = 0.0
        return weights

    def calculate_composite_score(self, weights: Dict[str, float],
                                   date_range: Tuple[str, str] = None) -> pd.DataFrame:
        """
        計算加權綜合分數

        Args:
            weights: 指標權重字典
            date_range: 日期範圍 (start, end)

        Returns:
            綜合分數 DataFrame
        """
        composite_score = None
        total_weight = 0

        for name, weight in weights.items():
            if abs(weight) < 0.01:  # 忽略接近零的權重
                continue

            indicator = self.dc.get_indicator(name, standardized=True)
            if indicator.empty:
                continue

            # 過濾日期範圍
            if date_range:
                start, end = date_range
                if start:
                    indicator = indicator.loc[start:]
                if end:
                    indicator = indicator.loc[:end]

            if composite_score is None:
                composite_score = indicator * weight
            else:
                # 對齊索引
                composite_score, indicator = composite_score.align(indicator, fill_value=0)
                composite_score = composite_score + indicator * weight

            total_weight += abs(weight)

        if composite_score is None:
            return pd.DataFrame()

        # 正規化
        if total_weight > 0:
            composite_score = composite_score / total_weight

        return composite_score

    def generate_position(self, weights: Dict[str, float],
                          date_range: Tuple[str, str] = None,
                          top_n: int = TOP_N_STOCKS) -> pd.DataFrame:
        """
        生成持倉信號

        Args:
            weights: 指標權重
            date_range: 日期範圍
            top_n: 選股數量

        Returns:
            持倉信號 DataFrame
        """
        # 計算綜合分數
        score = self.calculate_composite_score(weights, date_range)
        if score.empty:
            return pd.DataFrame()

        # 獲取收盤價用於過濾
        close = self.dc.get('close')
        volume = self.dc.get('volume')
        market_value = self.dc.get('market_value')

        # 過濾條件
        # 1. 股價 > 10 元
        price_filter = close > 10

        # 2. 成交量 > 10 萬股
        vol_filter = volume > 100000

        # 3. 市值 > 10 億
        mv_filter = market_value > 1e9

        # 合併過濾
        valid_mask = price_filter & vol_filter & mv_filter

        # 對齊並應用過濾
        score, valid_mask = score.align(valid_mask, fill_value=False)
        filtered_score = score.where(valid_mask, np.nan)

        # 選出分數最高的 top_n 檔
        position = filtered_score.is_largest(top_n)

        return position

    def evaluate_fitness(self, genes: List[float],
                         use_cache: bool = True) -> Tuple[float, Dict]:
        """
        評估個體適應度

        Args:
            genes: 基因序列
            use_cache: 是否使用快取

        Returns:
            (適應度分數, 詳細指標字典)
        """
        # 快取檢查
        gene_key = tuple(round(g, 4) for g in genes)
        if use_cache and gene_key in self._fitness_cache:
            return self._fitness_cache[gene_key]

        try:
            # 解碼權重
            weights = self.decode_weights(genes)

            # 生成持倉（In-Sample 期間）
            position = self.generate_position(
                weights,
                date_range=(IN_SAMPLE_START, IN_SAMPLE_END),
                top_n=TOP_N_STOCKS
            )

            if position.empty or position.sum().sum() == 0:
                result = (-100.0, {'error': 'empty_position'})
                self._fitness_cache[gene_key] = result
                return result

            # 執行回測
            report = sim(
                position=position,
                fee_ratio=1.425 / 1000,
                tax_ratio=3 / 1000,
                trade_at_price='close',
                upload=False,
                name='GA_Eval'
            )

            metrics = report.get_metrics()

            # 提取指標
            sharpe = metrics['ratio'].get('sharpeRatio', 0) or 0
            annual_return = metrics['profitability'].get('annualReturn', 0) or 0
            max_drawdown = abs(metrics['risk'].get('maxDrawdown', 0) or 0)
            capacity = metrics['liquidity'].get('capacity', 0) or 0
            win_rate = metrics['profitability'].get('winRate', 0) or 0

            # === 計算適應度 ===
            fitness = sharpe  # 基礎適應度

            # 懲罰項
            penalties = 0

            # 1. 持倉數量懲罰
            avg_holdings = position.sum(axis=1).mean()
            if avg_holdings < MIN_HOLDINGS:
                penalties += (MIN_HOLDINGS - avg_holdings) * 10
            elif avg_holdings > MAX_HOLDINGS:
                penalties += (avg_holdings - MAX_HOLDINGS) * 5

            # 2. 流動性懲罰
            if capacity < MIN_CAPACITY:
                penalties += (1 - capacity / MIN_CAPACITY) * 20

            # 3. 最大回撤懲罰
            if max_drawdown > 0.3:
                penalties += (max_drawdown - 0.3) * 30

            # 4. 年化報酬加分
            if annual_return > 0.2:
                fitness += annual_return * 2

            # 最終適應度
            fitness = fitness - penalties

            details = {
                'sharpe': sharpe,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'capacity': capacity,
                'win_rate': win_rate,
                'avg_holdings': avg_holdings,
                'penalties': penalties,
            }

            result = (fitness, details)
            self._fitness_cache[gene_key] = result
            return result

        except Exception as e:
            result = (-100.0, {'error': str(e)})
            self._fitness_cache[gene_key] = result
            return result

    def clear_cache(self):
        """清除快取"""
        self._fitness_cache.clear()
        gc.collect()

strategy_ga = StrategyGA(data_center)

# =============================================================================
# 第六部分：DEAP 設定
# =============================================================================
# 清除舊定義
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

# 創建適應度（單目標最大化）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 基因初始化：[-1, 1] 的隨機浮點數
def random_gene():
    return random.uniform(-1, 1)

toolbox.register("attr_float", random_gene)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=data_center.n_indicators)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 評估函數
def eval_individual(individual):
    fitness, _ = strategy_ga.evaluate_fitness(individual)
    return (fitness,)

toolbox.register("evaluate", eval_individual)

# 交叉：模擬二進制交叉
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=-1.0, up=1.0, eta=20.0)

# 變異：多項式有界變異
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=-1.0, up=1.0, eta=20.0, indpb=0.1)

# 選擇：錦標賽選擇
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

print(f"✅ DEAP 配置完成 (基因長度: {data_center.n_indicators})")

# =============================================================================
# 第七部分：GA Runner - 演化執行器
# =============================================================================
class GARunner:
    """
    遺傳演算法執行器

    功能：
    1. 並行評估加速
    2. 精英保留策略
    3. 中斷恢復機制
    4. 定期回測報告
    """

    def __init__(self, toolbox, strategy: StrategyGA, paths: PathManager):
        self.toolbox = toolbox
        self.strategy = strategy
        self.paths = paths
        self.history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.full_backtest_results = []

    def save_checkpoint(self, population: List, generation: int):
        """保存檢查點"""
        checkpoint = {
            'population': population,
            'generation': generation,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'full_backtest_results': self.full_backtest_results,
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.paths.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"   💾 檢查點已保存: 第 {generation} 代")

    def load_checkpoint(self) -> Tuple[Optional[List], int]:
        """載入檢查點"""
        if not os.path.exists(self.paths.checkpoint_file):
            return None, 0

        try:
            with open(self.paths.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            self.best_individual = checkpoint.get('best_individual')
            self.best_fitness = checkpoint.get('best_fitness', float('-inf'))
            self.history = checkpoint.get('history', [])
            self.full_backtest_results = checkpoint.get('full_backtest_results', [])

            print(f"✅ 載入檢查點: 第 {checkpoint['generation']} 代")
            print(f"   最佳適應度: {self.best_fitness:.4f}")

            return checkpoint['population'], checkpoint['generation']

        except Exception as e:
            print(f"⚠️ 檢查點載入失敗: {e}")
            return None, 0

    def evaluate_population_parallel(self, population: List) -> List:
        """並行評估種群"""
        invalid = [ind for ind in population if not ind.fitness.valid]

        if not invalid:
            return population

        # 使用 joblib 並行評估
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(eval_individual)(ind) for ind in tqdm(invalid, desc="評估中", leave=False)
        )

        for ind, fit in zip(invalid, results):
            ind.fitness.values = fit

        return population

    def run_full_backtest(self, individual: List, generation: int):
        """執行完整回測（In-Sample + Out-of-Sample）"""
        print(f"\n{'='*70}")
        print(f"📊 第 {generation} 代 - 完整回測")
        print(f"{'='*70}")

        weights = self.strategy.decode_weights(individual)

        results = {}

        # In-Sample 回測
        try:
            position_is = self.strategy.generate_position(
                weights,
                date_range=(IN_SAMPLE_START, IN_SAMPLE_END),
                top_n=TOP_N_STOCKS
            )

            if not position_is.empty:
                report_is = sim(
                    position=position_is,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    trade_at_price='close',
                    upload=False,
                    name=f'Gen{generation}_InSample'
                )
                metrics_is = report_is.get_metrics()

                results['in_sample'] = {
                    'sharpe': metrics_is['ratio'].get('sharpeRatio', 0) or 0,
                    'annual_return': metrics_is['profitability'].get('annualReturn', 0) or 0,
                    'max_drawdown': abs(metrics_is['risk'].get('maxDrawdown', 0) or 0),
                    'capacity': metrics_is['liquidity'].get('capacity', 0) or 0,
                }
        except Exception as e:
            results['in_sample'] = {'error': str(e)}

        # Out-of-Sample 回測
        try:
            position_oos = self.strategy.generate_position(
                weights,
                date_range=(OUT_SAMPLE_START, OUT_SAMPLE_END),
                top_n=TOP_N_STOCKS
            )

            if not position_oos.empty:
                report_oos = sim(
                    position=position_oos,
                    fee_ratio=1.425/1000,
                    tax_ratio=3/1000,
                    trade_at_price='close',
                    upload=False,
                    name=f'Gen{generation}_OutSample'
                )
                metrics_oos = report_oos.get_metrics()

                results['out_sample'] = {
                    'sharpe': metrics_oos['ratio'].get('sharpeRatio', 0) or 0,
                    'annual_return': metrics_oos['profitability'].get('annualReturn', 0) or 0,
                    'max_drawdown': abs(metrics_oos['risk'].get('maxDrawdown', 0) or 0),
                    'capacity': metrics_oos['liquidity'].get('capacity', 0) or 0,
                }
        except Exception as e:
            results['out_sample'] = {'error': str(e)}

        # 顯示結果
        print(f"\n┌─────────────────────────────────────────────────────────────────┐")
        print(f"│  📊 第 {generation} 代完整回測結果                                │")
        print(f"├─────────────────────────────────────────────────────────────────┤")

        if 'in_sample' in results and 'error' not in results['in_sample']:
            is_r = results['in_sample']
            print(f"│  【In-Sample {IN_SAMPLE_START} ~ {IN_SAMPLE_END}】                           │")
            print(f"│    夏普值:   {is_r['sharpe']:>8.2f}  年化報酬: {is_r['annual_return']*100:>6.1f}%          │")
            print(f"│    最大回撤: {is_r['max_drawdown']*100:>8.1f}%  胃納量:   {is_r['capacity']/1e4:>6.0f}萬          │")

        if 'out_sample' in results and 'error' not in results['out_sample']:
            oos_r = results['out_sample']
            print(f"│  【Out-of-Sample {OUT_SAMPLE_START} ~ 最新】                              │")
            print(f"│    夏普值:   {oos_r['sharpe']:>8.2f}  年化報酬: {oos_r['annual_return']*100:>6.1f}%          │")
            print(f"│    最大回撤: {oos_r['max_drawdown']*100:>8.1f}%  胃納量:   {oos_r['capacity']/1e4:>6.0f}萬          │")

        print(f"├─────────────────────────────────────────────────────────────────┤")

        # 顯示權重
        top_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"│  🔝 前五大權重指標:                                              │")
        for name, w in top_weights:
            direction = "📈" if w > 0 else "📉"
            print(f"│    {direction} {name:<20}: {w:>+.3f}                           │")

        print(f"└─────────────────────────────────────────────────────────────────┘")

        # 保存結果
        results['generation'] = generation
        results['weights'] = weights
        results['timestamp'] = datetime.now().isoformat()
        self.full_backtest_results.append(results)

        # 保存到檔案
        result_file = f"{self.paths.results_dir}/gen{generation}_backtest.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"{'='*70}\n")

    def run(self, n_generations: int = N_GENERATIONS,
            resume: bool = True) -> Tuple[List, creator.Individual]:
        """
        執行遺傳演算法

        Args:
            n_generations: 演化代數
            resume: 是否從檢查點恢復

        Returns:
            (最終種群, 最佳個體)
        """
        print(f"\n{'='*70}")
        print(f"🚀 開始演化")
        print(f"   種群: {POPULATION_SIZE}, 代數: {n_generations}")
        print(f"   突變率: {MUTATION_RATE}, 交叉率: {CROSSOVER_RATE}")
        print(f"   並行核心: {N_JOBS if N_JOBS > 0 else 'ALL'}")
        print(f"{'='*70}\n")

        # 嘗試恢復
        start_gen = 0
        if resume:
            population, start_gen = self.load_checkpoint()
            if population is None:
                population = self.toolbox.population(n=POPULATION_SIZE)
        else:
            population = self.toolbox.population(n=POPULATION_SIZE)

        # 初始評估
        if start_gen == 0:
            print("📊 初始評估...")
            population = self.evaluate_population_parallel(population)

            # 第 0 代完整回測
            best_ind = max(population, key=lambda x: x.fitness.values[0])
            self.run_full_backtest(best_ind, 0)

        # 精英數量
        n_elite = max(1, int(POPULATION_SIZE * ELITE_RATIO))

        # 演化循環
        for gen in range(start_gen, n_generations):
            start_time = time.time()

            # 選擇
            offspring = self.toolbox.select(population, len(population) - n_elite)
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

            # 精英保留
            elite = tools.selBest(population, n_elite)

            # 評估
            offspring = self.evaluate_population_parallel(offspring)

            # 合併
            population = elite + offspring

            # 統計
            fits = [ind.fitness.values[0] for ind in population]
            best_fit = max(fits)
            avg_fit = sum(fits) / len(fits)

            # 更新最佳
            current_best = max(population, key=lambda x: x.fitness.values[0])
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_individual = list(current_best)

            elapsed = time.time() - start_time

            # 記錄歷史
            self.history.append({
                'generation': gen + 1,
                'best_fitness': best_fit,
                'avg_fitness': avg_fit,
                'elapsed': elapsed,
            })

            # 輸出
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"=== 第 {gen+1}/{n_generations} 代 ===")
                print(f"   最佳適應度: {best_fit:.4f}")
                print(f"   平均適應度: {avg_fit:.4f}")
                print(f"   歷史最佳:   {self.best_fitness:.4f}")
                print(f"   耗時: {elapsed:.1f}s")

            # 每 N 代完整回測
            if (gen + 1) % CHECKPOINT_INTERVAL == 0:
                self.run_full_backtest(current_best, gen + 1)
                self.save_checkpoint(population, gen + 1)

        # 最終結果
        print(f"\n{'='*70}")
        print(f"🏆 演化完成！")
        print(f"   最佳適應度: {self.best_fitness:.4f}")
        print(f"{'='*70}")

        # 保存最佳權重
        if self.best_individual:
            best_weights = self.strategy.decode_weights(self.best_individual)
            with open(self.paths.best_weights_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'weights': best_weights,
                    'fitness': self.best_fitness,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ 最佳權重已保存: {self.paths.best_weights_file}")

        return population, self.best_individual

# =============================================================================
# 第八部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║     FinLab 台股指標權重優化系統 v1.0                               ║
║     Genetic Algorithm + Multi-Indicator Weight Optimization       ║
╠════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 {TARGET_SHARPE}+, 胃納量 {MIN_CAPACITY/1e4:.0f}萬+                              ║
║  📊 指標：{data_center.n_indicators} 個                                               ║
║  🧬 基因：{data_center.n_indicators} 個權重參數                                        ║
║  🔄 每 {CHECKPOINT_INTERVAL} 代完整回測                                            ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 建立執行器
    runner = GARunner(toolbox, strategy_ga, paths)

    # 執行演化
    population, best_individual = runner.run(N_GENERATIONS, resume=True)

    # 最終完整回測
    if best_individual:
        print(f"\n{'='*70}")
        print("📊 最終最佳個體完整回測")
        print(f"{'='*70}")
        runner.run_full_backtest(best_individual, N_GENERATIONS)

    return population, best_individual

# =============================================================================
# 第九部分：參數調整指南
# =============================================================================
"""
================================================================================
📖 參數調整指南
================================================================================

【GA 核心參數】

1. POPULATION_SIZE (種群大小)
   - 預設值: 40
   - 建議範圍: 20-100
   - 增大: 搜索更全面，但計算時間增加
   - 減小: 計算更快，但可能陷入局部最優

2. N_GENERATIONS (演化代數)
   - 預設值: 50
   - 建議範圍: 30-200
   - 增大: 更多優化機會，收斂更完全
   - 減小: 節省時間，可能未充分收斂

3. MUTATION_RATE (突變率)
   - 預設值: 0.25
   - 建議範圍: 0.1-0.4
   - 增大: 增加多樣性，避免早熟收斂
   - 減小: 保持優良基因，收斂更快

4. CROSSOVER_RATE (交叉率)
   - 預設值: 0.7
   - 建議範圍: 0.5-0.9
   - 增大: 更多基因交換，加速探索
   - 減小: 保持個體穩定性

5. ELITE_RATIO (精英比例)
   - 預設值: 0.1
   - 建議範圍: 0.05-0.2
   - 增大: 保留更多優秀個體
   - 減小: 增加種群多樣性

【選股參數】

1. TOP_N_STOCKS (選股數量)
   - 預設值: 15
   - 建議範圍: 5-30
   - 增大: 風險分散，但可能稀釋報酬
   - 減小: 集中投資，風險較高

2. MIN_WEIGHT_PER_STOCK (最小權重)
   - 預設值: 3%
   - 建議範圍: 2%-5%
   - 避免持股過於分散

【快速入門】

# 快速測試（5分鐘）
POPULATION_SIZE = 20
N_GENERATIONS = 10
CHECKPOINT_INTERVAL = 5

# 標準優化（30分鐘）
POPULATION_SIZE = 40
N_GENERATIONS = 50
CHECKPOINT_INTERVAL = 10

# 深度優化（2小時）
POPULATION_SIZE = 80
N_GENERATIONS = 100
CHECKPOINT_INTERVAL = 20

================================================================================
"""

if __name__ == "__main__":
    population, best = main()
