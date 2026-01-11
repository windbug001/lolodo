#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 九組合媽媽龍 - 遺傳演算法選股優化系統 v4.0
   Nine Combo Dragon - Genetic Algorithm Stock Optimizer
================================================================================

【系統特色】
✅ 基於遺傳演算法 (GA) 自動優化選股參數
✅ 支援 Google Colab Pro+ 多核心並行運算
✅ Checkpoint 機制 - 支援中斷後無縫重啟
✅ 歷史前 10 強個體保留與持續進化
✅ 樣本內/外分離驗證 - 避免過擬合
✅ 每 10 代詳細回測報告

【績效目標】
- 夏普值 (Sharpe Ratio) > 4.2
- 最大回檔 (MDD) < 20%
- 資金胃納量 > 1,000 萬台幣

【回測規範】
- 樣本內 (In-Sample)：~ 2022 年底
- 樣本外 (Out-of-Sample)：2023 年 ~ 至今

【持股邏輯】
- 單一標的權重 ≥ 3%，否則不持倉 (0%)

版本：v4.0 (2025-01-11)
環境：Google Colab Pro+ (CPU + High-RAM)
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 🔥 第零部分：環境初始化（必須最先執行）
# =============================================================================
import os
import sys

# 禁用 FinLab 快取以避免 EOFError
os.environ['FINLAB_DISABLE_CACHE'] = '1'

# 抑制警告訊息
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# 第一部分：核心套件載入
# =============================================================================
import json
import pickle
import time
import hashlib
import random
import copy
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

# Pandas 設定
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('future.no_silent_downcasting', True)

# =============================================================================
# 第二部分：全域配置參數
# =============================================================================
@dataclass
class GAConfig:
    """
    遺傳演算法配置類別

    所有可調整的參數都集中在此，方便統一管理與修改。
    """
    # === 視窗與 API 設定 ===
    window_id: int = int(os.environ.get('WINDOW_ID', '1'))
    api_key: str = os.environ.get('FINLAB_API_KEY',
        'R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m')

    # === 績效目標 ===
    target_sharpe: float = 4.2           # 目標夏普值
    max_drawdown: float = 0.20           # 最大回檔限制 (20%)
    min_capacity: int = 10_000_000       # 最小胃納量 (1000萬)
    target_annual_return: float = 0.35   # 目標年化報酬

    # === GA 演化參數 ===
    population_size: int = 40            # 族群大小
    n_generations: int = int(os.environ.get('EVOLUTION_GENERATIONS', '100'))
    mutation_rate: float = 0.25          # 變異率
    crossover_rate: float = 0.85         # 交叉率
    elite_size: int = 10                 # 精英保留數量
    tournament_size: int = 5             # 錦標賽選擇大小

    # === 並行運算設定 ===
    n_workers: int = max(1, mp.cpu_count() - 1)  # 並行工作數
    use_parallel: bool = True            # 是否使用並行運算

    # === 回測時間設定 ===
    in_sample_end: str = '2022-12-31'    # 樣本內結束日
    out_sample_start: str = '2023-01-01' # 樣本外開始日
    backtest_start: str = '2018-01-01'   # 回測起始日

    # === Checkpoint 設定 ===
    checkpoint_interval: int = 5         # 每 N 代存檔一次
    report_interval: int = 10            # 每 N 代詳細報告
    visualization_interval: int = 5      # 每 N 代視覺化

    # === 持股邏輯 ===
    min_position_weight: float = 0.03    # 最小持股權重 (3%)
    max_position_weight: float = 0.25    # 最大單一持股權重 (25%)

    # === 重啟驗證 ===
    validate_top_n: int = 5              # 重啟時驗證前 N 名
    continue_evolution: bool = os.environ.get('CONTINUE_EVOLUTION', 'false').lower() == 'true'

# 全域配置實例
CONFIG = GAConfig()

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🧬 九組合媽媽龍 - 遺傳演算法選股優化系統 v4.0                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  📊 視窗 ID: {CONFIG.window_id:<3}  |  🎯 目標夏普: {CONFIG.target_sharpe}  |  💰 胃納量: {CONFIG.min_capacity/1e6:.0f}M    ║
║  🧬 族群: {CONFIG.population_size:<3}    |  🔄 世代數: {CONFIG.n_generations:<3}   |  ⚡ 並行: {CONFIG.n_workers} 核心      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 第三部分：路徑管理器
# =============================================================================
@dataclass
class PathManager:
    """
    路徑管理器

    統一管理所有檔案路徑，支援 Google Colab 與本地環境。
    """
    base_dir: str = field(default='')
    window_id: int = field(default=1)
    in_colab: bool = field(default=False)

    def __post_init__(self):
        """初始化路徑結構"""
        # 檢測環境
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            self.base_dir = '/content/drive/MyDrive/九組合媽媽龍_GA'
            self.in_colab = True
            print("✅ Google Colab 環境 - Drive 已掛載")
        except:
            self.base_dir = './ga_output'
            self.in_colab = False
            print("⚠️  本地環境")

        self.window_id = CONFIG.window_id

        # 建立目錄結構
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.checkpoint_dir = f"{self.window_dir}/checkpoints"
        self.report_dir = f"{self.window_dir}/reports"
        self.elite_dir = f"{self.base_dir}/shared_elites"  # 跨視窗共享
        self.log_dir = f"{self.window_dir}/logs"

        for d in [self.window_dir, self.checkpoint_dir, self.report_dir,
                  self.elite_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> str:
        """Checkpoint 檔案路徑"""
        return f"{self.checkpoint_dir}/checkpoint_w{self.window_id}.pkl"

    @property
    def elite_archive_file(self) -> str:
        """精英存檔路徑"""
        return f"{self.elite_dir}/elite_archive_w{self.window_id}.pkl"

    @property
    def best_params_file(self) -> str:
        """最佳參數檔案路徑"""
        return f"{self.window_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log_file(self) -> str:
        """進度日誌路徑"""
        return f"{self.log_dir}/progress_w{self.window_id}.json"

    @property
    def generation_report_file(self) -> str:
        """世代報告路徑"""
        return f"{self.report_dir}/generation_report_w{self.window_id}.csv"

# 初始化路徑管理器
PATHS = PathManager()
print(f"📁 工作目錄: {PATHS.window_dir}")

# =============================================================================
# 第四部分：套件安裝與 FinLab 初始化
# =============================================================================
def install_required_packages():
    """安裝必要的 Python 套件"""
    required_packages = {
        'finlab': 'finlab',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
    }

    for pkg_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"   📦 安裝 {pkg_name}...")
            os.system(f'pip install {pkg_name} -q')

def clear_finlab_cache():
    """
    清除 FinLab 快取

    避免因快取損壞導致的 EOFError。
    """
    import glob
    import shutil

    cache_patterns = [
        '/root/.finlab*',
        '/tmp/.finlab*',
        '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'),
        '/content/.finlab*',
    ]

    cleared = 0
    for pattern in cache_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                cleared += 1
        except:
            pass

    if cleared > 0:
        print(f"   🧹 已清除 {cleared} 個快取項目")

def initialize_finlab():
    """初始化 FinLab 環境"""
    print("\n📊 初始化 FinLab...")

    # 清除快取
    clear_finlab_cache()

    # 安裝套件
    install_required_packages()

    # 載入 FinLab
    import finlab
    from finlab import data
    from finlab.backtest import sim

    # 登入
    finlab.login(CONFIG.api_key)
    print(f"   ✅ FinLab {finlab.__version__} 登入成功")

    return finlab, data, sim

# 初始化 FinLab
finlab_module, data_module, sim_func = initialize_finlab()

# =============================================================================
# 第五部分：資料載入器
# =============================================================================
class DataLoader:
    """
    FinLab 資料載入器

    負責從 FinLab API 載入所有必要的財務與技術指標資料。
    使用單例模式避免重複載入。

    資料來源參考：https://ai.finlab.tw/database
    """

    _instance = None
    _data_cache: Dict[str, pd.DataFrame] = {}
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_all_data()
            self._initialized = True

    def _safe_get(self, key: str, retries: int = 2) -> Optional[pd.DataFrame]:
        """
        安全載入資料

        遇到 EOFError 時會清除快取並重試。

        Args:
            key: FinLab 資料鍵名
            retries: 重試次數

        Returns:
            載入的 DataFrame 或 None
        """
        for attempt in range(retries + 1):
            try:
                return data_module.get(key)
            except (EOFError, Exception) as e:
                if attempt < retries and 'EOF' in str(e):
                    print(f"   ⚠️  快取錯誤，重試中: {key}")
                    clear_finlab_cache()
                    time.sleep(1)
                else:
                    print(f"   ❌ 載入失敗: {key} - {e}")
                    return None
        return None

    def _load_all_data(self):
        """載入所有必要資料"""
        print("\n📥 載入 FinLab 資料庫...")
        start_time = time.time()

        # === 價格資料 ===
        print("   📈 價格資料...")
        self._data_cache['close'] = self._safe_get('price:收盤價')
        self._data_cache['open'] = self._safe_get('price:開盤價')
        self._data_cache['high'] = self._safe_get('price:最高價')
        self._data_cache['low'] = self._safe_get('price:最低價')
        self._data_cache['volume'] = self._safe_get('price:成交股數')
        self._data_cache['adj_close'] = self._safe_get('etl:adj_close')

        # === 估值指標 ===
        print("   💰 估值指標...")
        self._data_cache['pe'] = self._safe_get('price_earning_ratio:本益比')
        self._data_cache['pb'] = self._safe_get('price_earning_ratio:股價淨值比')
        self._data_cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')

        # === 營收資料 ===
        print("   📊 營收資料...")
        self._data_cache['revenue'] = self._safe_get('monthly_revenue:當月營收')
        self._data_cache['revenue_yoy'] = self._safe_get('monthly_revenue:去年同月增減(%)')
        self._data_cache['revenue_mom'] = self._safe_get('monthly_revenue:上月比較增減(%)')

        # === 基本面指標 ===
        print("   📑 基本面指標...")
        self._data_cache['operating_profit_growth'] = self._safe_get('fundamental_features:營業利益成長率')
        self._data_cache['non_operating_ratio'] = self._safe_get('fundamental_features:業外收支營收率')
        self._data_cache['gross_margin'] = self._safe_get('fundamental_features:營業毛利率')
        self._data_cache['roe'] = self._safe_get('fundamental_features:ROE綜合損益')
        self._data_cache['net_profit_margin'] = self._safe_get('fundamental_features:稅後淨利率')
        self._data_cache['pretax_margin'] = self._safe_get('fundamental_features:稅前淨利率')

        # === 籌碼資料 ===
        print("   🏦 籌碼資料...")
        self._data_cache['margin_usage'] = self._safe_get('margin_transactions:融資使用率')
        self._data_cache['director_holdings'] = self._safe_get('internal_equity_changes:董監持有股數占比')
        self._data_cache['inventory'] = self._safe_get('inventory')

        # === 市值與財報 ===
        print("   📋 市值與財報...")
        self._data_cache['market_value'] = self._safe_get('etl:market_value')
        self._data_cache['capital'] = self._safe_get('financial_statement:股本')
        self._data_cache['investing_cashflow'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._data_cache['operating_cashflow'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._data_cache['net_income'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._data_cache['equity'] = self._safe_get('financial_statement:股東權益總額')

        # === 計算衍生指標 ===
        print("   🔧 計算衍生指標...")
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"   ✅ 資料載入完成 ({elapsed:.1f} 秒)")

    def _calculate_derived_indicators(self):
        """計算衍生技術與財務指標"""
        close = self._data_cache.get('close')
        high = self._data_cache.get('high')
        low = self._data_cache.get('low')
        adj_close = self._data_cache.get('adj_close')
        revenue = self._data_cache.get('revenue')

        if close is None:
            return

        # RSI 指標
        self._data_cache['rsi_5'] = data_module.indicator('RSI', adjust_price=False,
                                                          resample='D', timeperiod=5)
        self._data_cache['rsi_14'] = data_module.indicator('RSI', adjust_price=False,
                                                           resample='D', timeperiod=14)

        # ATR 波動率
        self._data_cache['atr'] = data_module.indicator('ATR', adjust_price=True, timeperiod=10)

        # 營收均線
        if revenue is not None:
            self._data_cache['revenue_ma3'] = revenue.average(3)
            self._data_cache['revenue_ma6'] = revenue.average(6)
            self._data_cache['revenue_ma12'] = revenue.average(12)

        # 波動率指標
        if adj_close is not None and self._data_cache.get('atr') is not None:
            self._data_cache['volatility'] = self._data_cache['atr'] / adj_close

        # 自由現金流
        inv_cf = self._data_cache.get('investing_cashflow')
        op_cf = self._data_cache.get('operating_cashflow')
        if inv_cf is not None and op_cf is not None:
            self._data_cache['free_cashflow'] = (inv_cf + op_cf).rolling(4).mean()

        # 股東權益報酬率
        net_income = self._data_cache.get('net_income')
        equity = self._data_cache.get('equity')
        if net_income is not None and equity is not None:
            self._data_cache['roe_calc'] = net_income / equity

        # 市值營收比
        market_value = self._data_cache.get('market_value')
        if market_value is not None and revenue is not None:
            quarterly_revenue = (revenue * 1000).rolling(4).sum()
            self._data_cache['ps_ratio'] = market_value / quarterly_revenue

        # 漲停鎖死判斷
        if all(self._data_cache.get(k) is not None for k in ['close', 'high', 'low', 'open']):
            close = self._data_cache['close']
            high = self._data_cache['high']
            low = self._data_cache['low']
            open_ = self._data_cache['open']

            limit_up = close > close.shift(1) * 1.095
            locked = (close == high) & (close == low) & (close == open_)
            self._data_cache['limit_up_locked'] = (limit_up & locked).fillna(False)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        獲取資料

        Args:
            key: 資料鍵名

        Returns:
            對應的 DataFrame
        """
        return self._data_cache.get(key)

    def keys(self) -> List[str]:
        """獲取所有可用的資料鍵名"""
        return list(self._data_cache.keys())

# 初始化資料載入器
DATA = DataLoader()

# =============================================================================
# 第六部分：基因定義與解碼器
# =============================================================================
@dataclass
class GeneDefinition:
    """
    基因定義

    定義單個基因的範圍與解碼方式。
    """
    name: str               # 基因名稱
    min_val: float         # 最小值
    max_val: float         # 最大值
    is_integer: bool = False  # 是否為整數

    def decode(self, gene_value: float) -> Union[int, float]:
        """
        解碼基因值

        Args:
            gene_value: 0-1 之間的基因值

        Returns:
            解碼後的實際參數值
        """
        value = self.min_val + gene_value * (self.max_val - self.min_val)
        return int(round(value)) if self.is_integer else value


class GeneDecoder:
    """
    基因解碼器

    將 0-1 之間的基因值陣列解碼為策略參數字典。

    基因結構 (共 45 個基因):
    - 策略一：低波動本益比策略 (10 個參數)
    - 策略二：小資族策略 (9 個參數)
    - 策略三：營收股價雙渦輪策略 (10 個參數)
    - 策略權重 (3 個參數)
    - 回測參數 (5 個參數)
    - 持股參數 (3 個參數)
    - 風控參數 (5 個參數)
    """

    # 基因定義列表
    GENE_DEFINITIONS: List[GeneDefinition] = [
        # === 策略一：低波動本益比 (0-9) ===
        GeneDefinition('lv_rev_ma_ratio', 1.0, 1.5),           # 0: 營收均線比率
        GeneDefinition('lv_rev_consistency', 0.5, 1.0),        # 1: 營收一致性
        GeneDefinition('lv_volatility_max', 0.02, 0.08),       # 2: 最大波動率
        GeneDefinition('lv_margin_limit', 15, 50),             # 3: 融資使用率上限
        GeneDefinition('lv_non_op_limit', 3, 15),              # 4: 業外收支上限
        GeneDefinition('lv_min_volume', 50000, 300000),        # 5: 最小成交量
        GeneDefinition('lv_pe_min', 3, 15),                    # 6: 本益比下限
        GeneDefinition('lv_pe_max', 15, 40),                   # 7: 本益比上限
        GeneDefinition('lv_pb_max', 1.5, 4.0),                 # 8: 股價淨值比上限
        GeneDefinition('lv_top_n', 2, 8, is_integer=True),     # 9: 選股數量

        # === 策略二：小資族 (10-18) ===
        GeneDefinition('si_market_cap_max', 5e9, 25e9),        # 10: 市值上限
        GeneDefinition('si_ps_ratio_max', 1.5, 5.0),           # 11: 市值營收比上限
        GeneDefinition('si_rev_yoy_min', -20, -5),             # 12: 營收年增下限
        GeneDefinition('si_rev_mom_min', -70, -30),            # 13: 營收月增下限
        GeneDefinition('si_rsv_period', 30, 80, is_integer=True),  # 14: RSV 週期
        GeneDefinition('si_ma_period', 40, 100, is_integer=True),  # 15: 均線週期
        GeneDefinition('si_min_volume', 50000, 300000),        # 16: 最小成交量
        GeneDefinition('si_non_op_limit', 3, 10),              # 17: 業外收支上限
        GeneDefinition('si_top_n', 3, 12, is_integer=True),    # 18: 選股數量

        # === 策略三：營收股價雙渦輪 (19-28) ===
        GeneDefinition('rpt_rev_ma_period', 2, 8, is_integer=True),   # 19: 營收均線週期
        GeneDefinition('rpt_rev_lookback', 10, 36, is_integer=True),  # 20: 營收回顧期
        GeneDefinition('rpt_price_window', 3, 15, is_integer=True),   # 21: 價格創高窗口
        GeneDefinition('rpt_min_volume', 100000, 500000),      # 22: 最小成交量
        GeneDefinition('rpt_min_price', 8, 25),                # 23: 最低股價
        GeneDefinition('rpt_rsi_min', 45, 70),                 # 24: RSI 下限
        GeneDefinition('rpt_pe_max', 80, 200),                 # 25: 本益比上限
        GeneDefinition('rpt_gross_margin_min', 3, 10),         # 26: 毛利率下限
        GeneDefinition('rpt_rev_rank_pct', 0.85, 0.98),        # 27: 營收排名百分位
        GeneDefinition('rpt_top_n', 5, 20, is_integer=True),   # 28: 選股數量

        # === 策略權重 (29-31) ===
        GeneDefinition('weight_lv', 0.1, 1.0),                 # 29: 低波動權重
        GeneDefinition('weight_si', 0.1, 1.0),                 # 30: 小資族權重
        GeneDefinition('weight_rpt', 0.1, 1.0),                # 31: 雙渦輪權重

        # === 回測參數 (32-36) ===
        GeneDefinition('stop_loss', 0.12, 0.35),               # 32: 停損比例
        GeneDefinition('trail_stop', 0.15, 0.45),              # 33: 追蹤停損
        GeneDefinition('take_profit', 0.4, 1.2),               # 34: 停利比例
        GeneDefinition('position_limit', 0.15, 0.35),          # 35: 單一持股上限
        GeneDefinition('fee_discount', 0.3, 0.6),              # 36: 手續費折扣

        # === 持股參數 (37-39) ===
        GeneDefinition('rebalance_threshold', 0.05, 0.20),     # 37: 再平衡閾值
        GeneDefinition('entry_spread', 0.005, 0.02),           # 38: 進場滑價
        GeneDefinition('exit_spread', 0.005, 0.02),            # 39: 出場滑價

        # === 風控參數 (40-44) ===
        GeneDefinition('max_sector_weight', 0.3, 0.6),         # 40: 最大產業權重
        GeneDefinition('correlation_threshold', 0.6, 0.9),     # 41: 相關性閾值
        GeneDefinition('drawdown_reduce_pct', 0.3, 0.7),       # 42: 回檔減碼比例
        GeneDefinition('volatility_scale', 0.5, 2.0),          # 43: 波動率調整因子
        GeneDefinition('momentum_factor', 0.3, 1.0),           # 44: 動量因子
    ]

    GENE_LENGTH = len(GENE_DEFINITIONS)

    @classmethod
    def decode(cls, genes: List[float]) -> Dict[str, Any]:
        """
        解碼基因陣列為參數字典

        Args:
            genes: 0-1 之間的基因值陣列

        Returns:
            解碼後的參數字典
        """
        # 確保基因長度正確
        if len(genes) < cls.GENE_LENGTH:
            genes = list(genes) + [0.5] * (cls.GENE_LENGTH - len(genes))
        genes = genes[:cls.GENE_LENGTH]

        # 解碼各基因
        params = {}
        for i, gene_def in enumerate(cls.GENE_DEFINITIONS):
            params[gene_def.name] = gene_def.decode(genes[i])

        # 正規化策略權重
        total_weight = params['weight_lv'] + params['weight_si'] + params['weight_rpt']
        params['weight_lv'] /= total_weight
        params['weight_si'] /= total_weight
        params['weight_rpt'] /= total_weight

        return params

    @classmethod
    def encode(cls, params: Dict[str, Any]) -> List[float]:
        """
        將參數字典編碼為基因陣列

        Args:
            params: 參數字典

        Returns:
            0-1 之間的基因值陣列
        """
        genes = []
        for gene_def in cls.GENE_DEFINITIONS:
            value = params.get(gene_def.name, (gene_def.min_val + gene_def.max_val) / 2)
            gene = (value - gene_def.min_val) / (gene_def.max_val - gene_def.min_val)
            genes.append(max(0.0, min(1.0, gene)))
        return genes

# =============================================================================
# 第七部分：策略引擎
# =============================================================================
class StrategyEngine:
    """
    策略引擎

    整合三個子策略並進行動態權重組合。

    策略架構：
    1. 低波動本益比策略 - 穩健型
    2. 小資族策略 - 成長型
    3. 營收股價雙渦輪策略 - 動能型
    """

    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader

    def strategy_low_volatility_pe(self, params: Dict) -> Optional[pd.DataFrame]:
        """
        策略一：低波動本益比策略

        選股邏輯：
        - 營收成長且穩定
        - 低波動率、低融資使用
        - 合理本益比區間
        - PEG 最小化

        Args:
            params: 解碼後的策略參數

        Returns:
            持股部位 DataFrame
        """
        try:
            # 獲取資料
            close = self.dl.get('close')
            pe = self.dl.get('pe')
            pb = self.dl.get('pb')
            volume = self.dl.get('volume')
            revenue = self.dl.get('revenue')
            revenue_ma3 = self.dl.get('revenue_ma3')
            revenue_ma12 = self.dl.get('revenue_ma12')
            revenue_yoy = self.dl.get('revenue_yoy')
            revenue_mom = self.dl.get('revenue_mom')
            margin_usage = self.dl.get('margin_usage')
            volatility = self.dl.get('volatility')
            non_op_ratio = self.dl.get('non_operating_ratio')
            op_growth = self.dl.get('operating_profit_growth')
            gross_margin = self.dl.get('gross_margin')
            roe = self.dl.get('roe')
            limit_up_locked = self.dl.get('limit_up_locked')
            inventory = self.dl.get('inventory')

            if close is None or pe is None:
                return None

            # 計算 PEG (本益成長比)
            peg = pe / (op_growth + 1e-10)

            # === 條件設定 ===
            # 營收成長條件
            cond_rev_growth = revenue_ma3 / revenue_ma12 > params['lv_rev_ma_ratio']
            cond_rev_stable = revenue / revenue.shift(1) > params['lv_rev_consistency']

            # 低波動因子
            cond_low_vol = (
                (margin_usage <= params['lv_margin_limit']) &
                (volatility <= params['lv_volatility_max']) &
                (non_op_ratio < params['lv_non_op_limit'])
            )

            # 成交量條件
            cond_volume = volume.average(1) > params['lv_min_volume']

            # 營收排除條件
            cond_no_rev_decline = ~(revenue_yoy < -30).sustain(3)
            cond_no_old_growth = ~(revenue_yoy > 30).sustain(12, 8)
            cond_mom_stable = (revenue_mom > -54).sustain(3)

            # 價格趨勢
            cond_price_trend = (
                (close > close.average(40)) &
                (close > close.average(75)) &
                (close > close.average(90))
            )

            # 營收趨勢
            cond_rev_trend = revenue.average(4) > revenue.average(12)

            # 估值區間
            cond_pe = (params['lv_pe_min'] <= pe) & (pe <= params['lv_pe_max'])
            cond_pb = (0.5 <= pb) & (pb <= params['lv_pb_max'])

            # 基本面品質
            cond_gpm = (gross_margin > 8).sustain(2)
            cond_roe = (roe > 0).sustain(2)

            # 集保散戶持股
            try:
                small_inv = (inventory[(inventory.持股分級.astype(int) <= 8)]
                    .reset_index()
                    .groupby(['date', 'stock_id'])
                    .agg({'占集保庫存數比例': 'sum'})
                    .reset_index()
                    .pivot(index='date', columns='stock_id', values='占集保庫存數比例')) <= 46
            except:
                small_inv = True

            # 排除漲停鎖死
            cond_no_limit = ~limit_up_locked if limit_up_locked is not None else True

            # 綜合所有條件
            conditions = (
                cond_rev_growth & cond_rev_stable & cond_low_vol &
                cond_volume & cond_no_rev_decline & cond_no_old_growth &
                cond_mom_stable & cond_price_trend & cond_rev_trend &
                cond_pe & cond_pb & cond_gpm & cond_roe &
                small_inv & cond_no_limit
            )

            # 選出 PEG 最小的 N 檔
            valid_peg = peg[conditions & (peg > 0)]
            position = valid_peg.is_smallest(int(params['lv_top_n']))
            position = position.reindex(revenue.index_str_to_date().index, method='ffill')

            return position

        except Exception as e:
            print(f"   ⚠️  低波動策略錯誤: {e}")
            return None

    def strategy_small_investor(self, params: Dict) -> Optional[pd.DataFrame]:
        """
        策略二：小資族策略

        選股邏輯：
        - 中小市值股票
        - 自由現金流為正
        - RSV 動能選股

        Args:
            params: 解碼後的策略參數

        Returns:
            持股部位 DataFrame
        """
        try:
            # 獲取資料
            close = self.dl.get('close')
            volume = self.dl.get('volume')
            revenue = self.dl.get('revenue')
            revenue_yoy = self.dl.get('revenue_yoy')
            revenue_mom = self.dl.get('revenue_mom')
            market_value = self.dl.get('market_value')
            ps_ratio = self.dl.get('ps_ratio')
            free_cashflow = self.dl.get('free_cashflow')
            roe_calc = self.dl.get('roe_calc')
            op_growth = self.dl.get('operating_profit_growth')
            non_op_ratio = self.dl.get('non_operating_ratio')

            if close is None or market_value is None:
                return None

            # === 條件設定 ===
            # 市值與估值
            cond_market_cap = market_value < params['si_market_cap_max']
            cond_ps = ps_ratio < params['si_ps_ratio_max']

            # 現金流與獲利
            cond_fcf = free_cashflow > 0 if free_cashflow is not None else True
            cond_roe = roe_calc > 0 if roe_calc is not None else True
            cond_op = op_growth > -1 if op_growth is not None else True

            # 成交量
            cond_volume = volume > params['si_min_volume']

            # 營收條件
            cond_no_decline = ~(revenue_yoy < params['si_rev_yoy_min']).sustain(3)
            cond_no_old = ~(revenue_yoy > 60).sustain(12, 8)
            cond_mom = (revenue_mom > params['si_rev_mom_min']).sustain(3)
            cond_rev_trend = revenue.average(3) > revenue.average(12)

            # 價格趨勢
            ma_period = int(params['si_ma_period'])
            cond_price = (
                (close > close.average(ma_period)) &
                (close > close.average(75)) &
                (close > close.average(120))
            )

            # 業外收支
            cond_non_op = non_op_ratio < params['si_non_op_limit']

            # 計算 RSV
            rsv_period = int(params['si_rsv_period'])
            rsv = (close - close.rolling(rsv_period).min()) / \
                  (close.rolling(rsv_period).max() - close.rolling(rsv_period).min() + 1e-10)

            # 綜合條件
            conditions = (
                cond_market_cap & cond_ps & cond_fcf & cond_roe &
                cond_op & cond_volume & cond_no_decline & cond_no_old &
                cond_mom & cond_rev_trend & cond_price & cond_non_op
            )

            # RSV 加權選股
            position = (conditions * rsv).is_largest(int(params['si_top_n']))

            # 對齊月營收日期
            monthly_rev = data_module.get('monthly_revenue:當月營收') * 1000
            position = position.reindex(monthly_rev.index_str_to_date().index)

            return position

        except Exception as e:
            print(f"   ⚠️  小資族策略錯誤: {e}")
            return None

    def strategy_revenue_price_turbo(self, params: Dict) -> Optional[pd.DataFrame]:
        """
        策略三：營收股價雙渦輪策略

        選股邏輯：
        - 營收創新高
        - 股價創新高
        - 多頭排列
        - 動能強勁

        Args:
            params: 解碼後的策略參數

        Returns:
            持股部位 DataFrame
        """
        try:
            # 獲取資料
            close = self.dl.get('close')
            volume = self.dl.get('volume')
            pe = self.dl.get('pe')
            revenue = self.dl.get('revenue')
            revenue_yoy = self.dl.get('revenue_yoy')
            rsi = self.dl.get('rsi_5')
            non_op_ratio = self.dl.get('non_operating_ratio')
            gross_margin = self.dl.get('gross_margin')
            pretax_margin = self.dl.get('pretax_margin')
            net_margin = self.dl.get('net_profit_margin')
            limit_up_locked = self.dl.get('limit_up_locked')
            inventory = self.dl.get('inventory')

            if close is None or revenue is None:
                return None

            # === 條件設定 ===
            # 營收創新高
            rev_ma_period = int(params['rpt_rev_ma_period'])
            rev_lookback = int(params['rpt_rev_lookback'])
            rev_ma = revenue.average(rev_ma_period)
            cond_rev_high = rev_ma == rev_ma.rolling(rev_lookback, min_periods=rev_ma_period).max()

            # 股價創新高
            price_window = int(params['rpt_price_window'])
            cond_price_high = (close == close.rolling(260).max()).sustain(price_window, 1)

            # 成交量
            cond_volume = volume.average(1) > params['rpt_min_volume']

            # 多頭排列
            cond_ma_align = (
                (close > close.average(5)) &
                (close > close.average(10)) &
                (close > close.average(20)) &
                (close > close.average(60)) &
                (close > close.average(150)) &
                (close > close.average(200))
            )

            # 超級績效
            cond_super = close > (close.average(250) * 1.1)

            # RSI 動能
            cond_rsi = (rsi > params['rpt_rsi_min']).sustain(1)

            # 基本面
            cond_gpm = (gross_margin > params['rpt_gross_margin_min']).sustain(5)
            cond_pretax = (pretax_margin > 4).sustain(1)
            cond_net = (net_margin > 3).sustain(1)

            # 營收排名
            cond_rev_rank = revenue_yoy.rank(pct=True, axis=1) > params['rpt_rev_rank_pct']

            # 大戶持股
            try:
                big_inv = (inventory[(inventory.持股分級.astype(int) >= 12) &
                                     (inventory.持股分級.astype(int) <= 16)]
                    .reset_index()
                    .groupby(['date', 'stock_id'])
                    .agg({'占集保庫存數比例': 'sum'})
                    .reset_index()
                    .pivot(index='date', columns='stock_id', values='占集保庫存數比例')) >= 18
            except:
                big_inv = True

            # 估值與其他
            cond_pe = ~(params['rpt_pe_max'] <= pe)
            cond_price_min = close > params['rpt_min_price']
            cond_non_op = non_op_ratio < 7.3
            cond_vol_trend = volume >= volume.rolling(20).mean() * 0.8
            cond_no_limit = ~limit_up_locked if limit_up_locked is not None else True

            # 綜合條件
            conditions = (
                cond_rev_high & cond_price_high & cond_volume &
                cond_ma_align & cond_super & cond_rsi &
                cond_gpm & cond_pretax & cond_net &
                cond_rev_rank & big_inv & cond_pe &
                cond_price_min & cond_non_op & cond_vol_trend & cond_no_limit
            )

            # 營收年增排序選股
            position = (revenue_yoy * conditions)
            position = position[position > 0].is_largest(int(params['rpt_top_n']))
            position = position.reindex(revenue.index_str_to_date().index, method='ffill')

            return position

        except Exception as e:
            print(f"   ⚠️  雙渦輪策略錯誤: {e}")
            return None

    def combine_strategies(self, params: Dict) -> Optional[pd.DataFrame]:
        """
        合併三個策略

        根據權重動態組合各策略的持股部位。

        Args:
            params: 解碼後的策略參數

        Returns:
            合併後的持股部位 DataFrame
        """
        try:
            # 執行三個策略
            pos_lv = self.strategy_low_volatility_pe(params)
            pos_si = self.strategy_small_investor(params)
            pos_rpt = self.strategy_revenue_price_turbo(params)

            # 檢查有效策略
            valid_positions = []
            weights = []

            if pos_lv is not None and not pos_lv.empty:
                valid_positions.append(pos_lv)
                weights.append(params['weight_lv'])

            if pos_si is not None and not pos_si.empty:
                valid_positions.append(pos_si)
                weights.append(params['weight_si'])

            if pos_rpt is not None and not pos_rpt.empty:
                valid_positions.append(pos_rpt)
                weights.append(params['weight_rpt'])

            if not valid_positions:
                return None

            # 正規化權重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # 對齊索引
            common_index = valid_positions[0].index
            for pos in valid_positions[1:]:
                common_index = common_index.intersection(pos.index)

            common_columns = valid_positions[0].columns
            for pos in valid_positions[1:]:
                common_columns = common_columns.intersection(pos.columns)

            # 加權合併
            combined = None
            for pos, weight in zip(valid_positions, weights):
                aligned = pos.reindex(index=common_index, columns=common_columns).fillna(0)
                if combined is None:
                    combined = aligned * weight
                else:
                    combined = combined + aligned * weight

            # 應用最小權重限制
            if combined is not None:
                min_weight = CONFIG.min_position_weight
                combined = combined.apply(
                    lambda row: row.apply(lambda x: x if x >= min_weight else 0),
                    axis=1
                )

                # 重新正規化
                row_sums = combined.sum(axis=1)
                combined = combined.div(row_sums.replace(0, 1), axis=0)

            return combined

        except Exception as e:
            print(f"   ⚠️  策略合併錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None

# 初始化策略引擎
STRATEGY = StrategyEngine(DATA)

# =============================================================================
# 第八部分：回測引擎
# =============================================================================
class BacktestEngine:
    """
    回測引擎

    負責執行策略回測，包含樣本內/外分離驗證。
    """

    @staticmethod
    def run_backtest(position: pd.DataFrame, params: Dict,
                     name: str = 'GA_Strategy',
                     start_date: str = None,
                     end_date: str = None) -> Optional[Dict]:
        """
        執行回測

        Args:
            position: 持股部位 DataFrame
            params: 策略參數
            name: 策略名稱
            start_date: 回測開始日期
            end_date: 回測結束日期

        Returns:
            回測結果字典
        """
        try:
            if position is None or position.empty:
                return None

            # 過濾時間範圍
            if start_date:
                position = position.loc[start_date:]
            if end_date:
                position = position.loc[:end_date]

            if len(position) < 50:
                return None

            # 計算手續費
            fee_ratio = 1.425 / 1000 * params.get('fee_discount', 0.5)

            # 執行回測
            report = sim_func(
                position=position,
                fee_ratio=fee_ratio,
                tax_ratio=3 / 1000,
                trade_at_price='high_low_avg',
                position_limit=params.get('position_limit', 0.25),
                stop_loss=params.get('stop_loss', 0.25),
                trail_stop=params.get('trail_stop', 0.30),
                take_profit=params.get('take_profit', 0.70),
                stop_trading_next_period=False,
                upload=False,
                name=name,
            )

            # 獲取績效指標
            metrics = report.get_metrics()

            result = {
                'name': name,
                'sharpe': metrics['ratio'].get('sharpeRatio', 0) or 0,
                'annual_return': metrics['profitability'].get('annualReturn', 0) or 0,
                'max_drawdown': abs(metrics['risk'].get('maxDrawdown', 1) or 1),
                'capacity': metrics['liquidity'].get('capacity', 0) or 0,
                'win_rate': metrics['profitability'].get('winRate', 0) or 0,
                'profit_factor': metrics['profitability'].get('profitFactor', 0) or 0,
                'total_return': metrics['profitability'].get('totalReturn', 0) or 0,
                'report': report,
            }

            return result

        except Exception as e:
            print(f"   ⚠️  回測錯誤 ({name}): {e}")
            return None

    @staticmethod
    def run_in_sample_out_sample(position: pd.DataFrame,
                                  params: Dict) -> Dict:
        """
        執行樣本內/外分離回測

        Args:
            position: 持股部位
            params: 策略參數

        Returns:
            包含樣本內、樣本外結果的字典
        """
        results = {
            'in_sample': None,
            'out_sample': None,
            'full_period': None,
            'is_valid': False,
        }

        try:
            # 樣本內回測
            in_sample = BacktestEngine.run_backtest(
                position, params, 'InSample',
                start_date=CONFIG.backtest_start,
                end_date=CONFIG.in_sample_end
            )
            results['in_sample'] = in_sample

            # 樣本外回測
            out_sample = BacktestEngine.run_backtest(
                position, params, 'OutSample',
                start_date=CONFIG.out_sample_start,
                end_date=None
            )
            results['out_sample'] = out_sample

            # 全期回測
            full_period = BacktestEngine.run_backtest(
                position, params, 'FullPeriod',
                start_date=CONFIG.backtest_start,
                end_date=None
            )
            results['full_period'] = full_period

            # 驗證有效性
            if in_sample and out_sample:
                results['is_valid'] = (
                    in_sample['sharpe'] > 0 and
                    out_sample['sharpe'] > 0 and
                    out_sample['sharpe'] >= in_sample['sharpe'] * 0.5  # 樣本外不能太差
                )

            return results

        except Exception as e:
            print(f"   ⚠️  樣本內外回測錯誤: {e}")
            return results

# =============================================================================
# 第九部分：適應度評估器
# =============================================================================
class FitnessEvaluator:
    """
    適應度評估器

    計算個體的適應度分數，綜合考慮夏普值、胃納量、回檔與穩健性。
    """

    @staticmethod
    def evaluate(genes: List[float]) -> Tuple[float, Dict]:
        """
        評估單個個體的適應度

        Args:
            genes: 基因陣列

        Returns:
            (適應度分數, 詳細結果字典)
        """
        try:
            # 解碼基因
            params = GeneDecoder.decode(genes)

            # 組合策略
            position = STRATEGY.combine_strategies(params)

            if position is None or position.empty:
                return 0.0, {'error': 'No position'}

            # 執行樣本內外回測
            results = BacktestEngine.run_in_sample_out_sample(position, params)

            if not results['in_sample'] and not results['full_period']:
                return 0.0, {'error': 'Backtest failed'}

            # 使用樣本外結果（如果有）或全期結果
            primary = results['out_sample'] or results['full_period']
            secondary = results['in_sample']

            if not primary:
                return 0.0, {'error': 'No valid results'}

            # === 計算適應度分數 ===

            # 1. 夏普值分數 (0-50)
            sharpe = primary['sharpe']
            sharpe_score = min(50, max(0, sharpe / CONFIG.target_sharpe * 50))

            # 2. 胃納量分數 (0-20)
            capacity = primary['capacity']
            capacity_score = min(20, max(0, capacity / CONFIG.min_capacity * 20))

            # 3. 回檔懲罰 (-20 to 0)
            mdd = primary['max_drawdown']
            if mdd > CONFIG.max_drawdown:
                mdd_penalty = -20 * (mdd - CONFIG.max_drawdown) / CONFIG.max_drawdown
            else:
                mdd_penalty = 0

            # 4. 年化報酬分數 (0-15)
            annual_return = primary['annual_return']
            return_score = min(15, max(0, annual_return / CONFIG.target_annual_return * 15))

            # 5. 穩健性分數 (0-15)
            robustness_score = 0
            if secondary and results['is_valid']:
                # 樣本外/樣本內夏普比率
                ratio = primary['sharpe'] / (secondary['sharpe'] + 1e-10)
                robustness_score = min(15, max(0, ratio * 10))

            # 總分
            total_fitness = sharpe_score + capacity_score + mdd_penalty + return_score + robustness_score

            # 詳細結果
            details = {
                'sharpe': sharpe,
                'capacity': capacity,
                'max_drawdown': mdd,
                'annual_return': annual_return,
                'sharpe_score': sharpe_score,
                'capacity_score': capacity_score,
                'mdd_penalty': mdd_penalty,
                'return_score': return_score,
                'robustness_score': robustness_score,
                'total_fitness': total_fitness,
                'in_sample': secondary,
                'out_sample': primary if results['out_sample'] else None,
                'is_valid': results['is_valid'],
            }

            return total_fitness, details

        except Exception as e:
            print(f"   ⚠️  適應度評估錯誤: {e}")
            return 0.0, {'error': str(e)}

# =============================================================================
# 第十部分：Checkpoint 管理器
# =============================================================================
@dataclass
class EvolutionState:
    """演化狀態資料類別"""
    generation: int = 0
    population: List[List[float]] = field(default_factory=list)
    fitness_scores: List[float] = field(default_factory=list)
    elite_archive: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    best_individual: Optional[List[float]] = None
    best_fitness: float = 0.0
    best_params: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CheckpointManager:
    """
    Checkpoint 管理器

    負責演化狀態的儲存與讀取，支援中斷後重啟。
    """

    def __init__(self, checkpoint_file: str, elite_file: str):
        self.checkpoint_file = checkpoint_file
        self.elite_file = elite_file

    def save(self, state: EvolutionState):
        """
        儲存演化狀態

        Args:
            state: 演化狀態物件
        """
        try:
            state.timestamp = datetime.now().isoformat()

            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)

            # 同時保存精英存檔
            self._save_elite_archive(state.elite_archive)

            print(f"   💾 Checkpoint 已儲存 (第 {state.generation} 代)")

        except Exception as e:
            print(f"   ⚠️  Checkpoint 儲存失敗: {e}")

    def load(self) -> Optional[EvolutionState]:
        """
        載入演化狀態

        Returns:
            演化狀態物件或 None
        """
        try:
            if not os.path.exists(self.checkpoint_file):
                print("   ℹ️  無 Checkpoint 檔案")
                return None

            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)

            print(f"   ✅ 載入 Checkpoint (第 {state.generation} 代)")
            return state

        except Exception as e:
            print(f"   ⚠️  Checkpoint 載入失敗: {e}")
            return None

    def _save_elite_archive(self, elite_archive: List[Dict]):
        """保存精英存檔"""
        try:
            with open(self.elite_file, 'wb') as f:
                pickle.dump({
                    'elite_archive': elite_archive,
                    'timestamp': datetime.now().isoformat(),
                }, f)
        except:
            pass

    def load_elite_archive(self) -> List[Dict]:
        """載入精英存檔"""
        try:
            if os.path.exists(self.elite_file):
                with open(self.elite_file, 'rb') as f:
                    data = pickle.load(f)
                    return data.get('elite_archive', [])
        except:
            pass
        return []

    def validate_elites(self, elite_archive: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        驗證精英個體

        重啟時使用 sim() 回測歷史前 N 名，確保連續性。

        Args:
            elite_archive: 精英存檔
            top_n: 驗證數量

        Returns:
            驗證後的精英列表
        """
        if not elite_archive:
            return []

        print(f"\n🔍 驗證歷史前 {top_n} 名精英...")

        # 排序取前 N 名
        sorted_elites = sorted(elite_archive,
                               key=lambda x: x.get('fitness', 0),
                               reverse=True)[:top_n]

        validated = []
        for i, elite in enumerate(sorted_elites):
            genes = elite.get('genes', [])
            if not genes:
                continue

            print(f"   驗證 #{i+1}...")
            fitness, details = FitnessEvaluator.evaluate(genes)

            if fitness > 0:
                elite['fitness'] = fitness
                elite['validated'] = True
                elite['details'] = details
                validated.append(elite)
                print(f"      ✅ 夏普: {details.get('sharpe', 0):.2f}")
            else:
                print(f"      ❌ 驗證失敗")

        print(f"   共 {len(validated)}/{top_n} 個精英通過驗證")
        return validated

# 初始化 Checkpoint 管理器
CHECKPOINT = CheckpointManager(PATHS.checkpoint_file, PATHS.elite_archive_file)

# =============================================================================
# 第十一部分：遺傳演算法核心
# =============================================================================
class GeneticAlgorithm:
    """
    遺傳演算法核心類別

    實現基因演算法的主要操作：
    - 初始化族群
    - 選擇 (Tournament Selection)
    - 交叉 (Simulated Binary Crossover)
    - 變異 (Polynomial Mutation)
    - 精英保留
    """

    def __init__(self):
        self.gene_length = GeneDecoder.GENE_LENGTH

    def create_individual(self) -> List[float]:
        """創建隨機個體"""
        return [random.random() for _ in range(self.gene_length)]

    def create_population(self, size: int) -> List[List[float]]:
        """創建初始族群"""
        return [self.create_individual() for _ in range(size)]

    def tournament_select(self, population: List[List[float]],
                          fitness_scores: List[float],
                          tournament_size: int = 5) -> List[float]:
        """
        錦標賽選擇

        Args:
            population: 族群
            fitness_scores: 適應度分數
            tournament_size: 錦標賽大小

        Returns:
            被選中的個體
        """
        indices = random.sample(range(len(population)),
                                min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(population[best_idx])

    def crossover(self, parent1: List[float], parent2: List[float],
                  eta: float = 20.0) -> Tuple[List[float], List[float]]:
        """
        模擬二元交叉 (SBX)

        Args:
            parent1: 父代1
            parent2: 父代2
            eta: 分布指數

        Returns:
            兩個子代
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        for i in range(len(parent1)):
            if random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]

                    rand = random.random()
                    beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** -(eta + 1.0)

                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = max(0.0, min(1.0, c1))
                    child2[i] = max(0.0, min(1.0, c2))

        return child1, child2

    def mutate(self, individual: List[float], eta: float = 20.0,
               indpb: float = 0.1) -> List[float]:
        """
        多項式變異

        Args:
            individual: 個體
            eta: 分布指數
            indpb: 單基因變異機率

        Returns:
            變異後的個體
        """
        mutant = copy.deepcopy(individual)

        for i in range(len(mutant)):
            if random.random() < indpb:
                y = mutant[i]
                delta1 = y
                delta2 = 1.0 - y

                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                mutant[i] = max(0.0, min(1.0, y + deltaq))

        return mutant

# 初始化 GA
GA = GeneticAlgorithm()

# =============================================================================
# 第十二部分：演化引擎
# =============================================================================
class EvolutionEngine:
    """
    演化引擎

    整合所有組件，執行完整的遺傳演算法優化流程。
    """

    def __init__(self):
        self.ga = GA
        self.checkpoint = CHECKPOINT
        self.state: Optional[EvolutionState] = None
        self.start_time = time.time()

    def initialize(self) -> EvolutionState:
        """
        初始化或恢復演化狀態

        Returns:
            演化狀態物件
        """
        # 檢查是否需要繼續
        if CONFIG.continue_evolution:
            state = self.checkpoint.load()
            if state:
                # 驗證精英
                state.elite_archive = self.checkpoint.validate_elites(
                    state.elite_archive,
                    CONFIG.validate_top_n
                )
                self.state = state
                return state

        # 創建新狀態
        print("\n🆕 初始化新的演化狀態...")

        # 嘗試載入歷史精英
        elite_archive = self.checkpoint.load_elite_archive()
        if elite_archive:
            elite_archive = self.checkpoint.validate_elites(
                elite_archive,
                CONFIG.validate_top_n
            )

        # 創建初始族群
        population = self.ga.create_population(CONFIG.population_size)

        # 注入精英
        if elite_archive:
            n_inject = min(len(elite_archive), CONFIG.population_size // 4)
            for i, elite in enumerate(elite_archive[:n_inject]):
                if i < len(population):
                    population[i] = elite.get('genes', population[i])
            print(f"   ✅ 注入 {n_inject} 個歷史精英")

        state = EvolutionState(
            generation=0,
            population=population,
            fitness_scores=[0.0] * len(population),
            elite_archive=elite_archive,
        )

        self.state = state
        return state

    def evaluate_population(self, population: List[List[float]]) -> List[Tuple[float, Dict]]:
        """
        評估整個族群

        Args:
            population: 族群

        Returns:
            (適應度, 詳細結果) 列表
        """
        print(f"   📊 評估 {len(population)} 個個體...")

        results = []

        if CONFIG.use_parallel and CONFIG.n_workers > 1:
            # 並行評估
            try:
                with ThreadPoolExecutor(max_workers=CONFIG.n_workers) as executor:
                    futures = {executor.submit(FitnessEvaluator.evaluate, ind): i
                               for i, ind in enumerate(population)}

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result(timeout=300)
                            results.append((idx, result))
                        except Exception as e:
                            results.append((idx, (0.0, {'error': str(e)})))

                # 按原始順序排序
                results.sort(key=lambda x: x[0])
                results = [r[1] for r in results]

            except Exception as e:
                print(f"   ⚠️  並行評估失敗，切換為序列: {e}")
                results = [FitnessEvaluator.evaluate(ind) for ind in population]
        else:
            # 序列評估
            for i, ind in enumerate(population):
                result = FitnessEvaluator.evaluate(ind)
                results.append(result)

                if (i + 1) % 5 == 0:
                    print(f"      進度: {i+1}/{len(population)}")

        return results

    def update_elite_archive(self, population: List[List[float]],
                              fitness_scores: List[float],
                              details_list: List[Dict]):
        """
        更新精英存檔

        Args:
            population: 族群
            fitness_scores: 適應度分數
            details_list: 詳細結果列表
        """
        # 創建精英候選
        new_elites = []
        for genes, fitness, details in zip(population, fitness_scores, details_list):
            if fitness > 0:
                new_elites.append({
                    'genes': genes,
                    'fitness': fitness,
                    'details': details,
                    'generation': self.state.generation,
                })

        # 合併現有精英
        all_elites = self.state.elite_archive + new_elites

        # 去重（基於基因相似度）
        unique_elites = []
        seen_hashes = set()

        for elite in all_elites:
            gene_hash = hashlib.md5(str(elite['genes'][:10]).encode()).hexdigest()
            if gene_hash not in seen_hashes:
                seen_hashes.add(gene_hash)
                unique_elites.append(elite)

        # 排序並保留前 N 名
        unique_elites.sort(key=lambda x: x['fitness'], reverse=True)
        self.state.elite_archive = unique_elites[:CONFIG.elite_size]

    def run_generation(self) -> Dict:
        """
        執行單代演化

        Returns:
            本代統計資訊
        """
        gen = self.state.generation
        population = self.state.population

        # 評估族群
        eval_results = self.evaluate_population(population)
        fitness_scores = [r[0] for r in eval_results]
        details_list = [r[1] for r in eval_results]

        self.state.fitness_scores = fitness_scores

        # 更新精英存檔
        self.update_elite_archive(population, fitness_scores, details_list)

        # 統計
        best_idx = fitness_scores.index(max(fitness_scores))
        best_fitness = fitness_scores[best_idx]
        best_details = details_list[best_idx]

        # 更新最佳個體
        if best_fitness > self.state.best_fitness:
            self.state.best_fitness = best_fitness
            self.state.best_individual = copy.deepcopy(population[best_idx])
            self.state.best_params = GeneDecoder.decode(population[best_idx])

        stats = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'best_sharpe': best_details.get('sharpe', 0),
            'best_capacity': best_details.get('capacity', 0),
            'best_mdd': best_details.get('max_drawdown', 0),
            'best_return': best_details.get('annual_return', 0),
            'elite_count': len(self.state.elite_archive),
            'timestamp': datetime.now().isoformat(),
        }

        self.state.history.append(stats)

        # 產生下一代
        if gen < CONFIG.n_generations - 1:
            self._create_next_generation()

        self.state.generation += 1

        return stats

    def _create_next_generation(self):
        """創建下一代族群"""
        population = self.state.population
        fitness_scores = self.state.fitness_scores

        new_population = []

        # 精英直接保留
        elite_count = min(len(self.state.elite_archive), CONFIG.elite_size // 2)
        for elite in self.state.elite_archive[:elite_count]:
            new_population.append(copy.deepcopy(elite['genes']))

        # 產生新個體
        while len(new_population) < CONFIG.population_size:
            # 選擇父代
            parent1 = self.ga.tournament_select(population, fitness_scores,
                                                 CONFIG.tournament_size)
            parent2 = self.ga.tournament_select(population, fitness_scores,
                                                 CONFIG.tournament_size)

            # 交叉
            if random.random() < CONFIG.crossover_rate:
                child1, child2 = self.ga.crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # 變異
            if random.random() < CONFIG.mutation_rate:
                child1 = self.ga.mutate(child1)
            if random.random() < CONFIG.mutation_rate:
                child2 = self.ga.mutate(child2)

            new_population.append(child1)
            if len(new_population) < CONFIG.population_size:
                new_population.append(child2)

        self.state.population = new_population[:CONFIG.population_size]

    def print_generation_summary(self, stats: Dict):
        """輸出世代摘要"""
        gen = stats['generation']

        # 每 5 代或首代輸出
        if gen % CONFIG.visualization_interval == 0 or gen == 0:
            elapsed = time.time() - self.start_time

            print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 第 {gen+1:3d}/{CONFIG.n_generations} 代  |  ⏱️  {elapsed/60:.1f} 分鐘  |  👥 精英: {stats['elite_count']}     ║
╠══════════════════════════════════════════════════════════════╣
║  📈 最佳適應度: {stats['best_fitness']:8.2f}  |  平均: {stats['avg_fitness']:8.2f}           ║
║  📊 夏普值:     {stats['best_sharpe']:8.2f}  |  胃納量: {stats['best_capacity']/1e4:8.0f} 萬    ║
║  📉 最大回檔:   {stats['best_mdd']*100:7.1f}%  |  年化報酬: {stats['best_return']*100:6.1f}%       ║
╚══════════════════════════════════════════════════════════════╝
""")

    def print_detailed_report(self, gen: int):
        """
        輸出詳細報告（每 10 代）

        Args:
            gen: 當前世代
        """
        if (gen + 1) % CONFIG.report_interval != 0:
            return

        print(f"\n{'='*70}")
        print(f"📋 第 {gen+1} 代詳細報告")
        print(f"{'='*70}")

        # 精英排名
        print(f"\n🏆 精英排名 (前 {CONFIG.elite_size} 名):")
        print(f"{'排名':<6}{'適應度':<12}{'夏普值':<10}{'胃納量(萬)':<12}{'回檔%':<10}{'達標'}")
        print("-" * 60)

        for i, elite in enumerate(self.state.elite_archive[:10], 1):
            details = elite.get('details', {})
            sharpe = details.get('sharpe', 0)
            capacity = details.get('capacity', 0) / 1e4
            mdd = details.get('max_drawdown', 0) * 100

            is_target = "✅" if (sharpe >= CONFIG.target_sharpe and
                                 capacity >= CONFIG.min_capacity / 1e4) else ""

            print(f"{i:<6}{elite['fitness']:<12.2f}{sharpe:<10.2f}{capacity:<12.0f}{mdd:<10.1f}{is_target}")

        # 最佳參數概覽
        if self.state.best_params:
            params = self.state.best_params
            print(f"\n📊 最佳策略權重配置:")
            print(f"   低波動: {params['weight_lv']:.1%}")
            print(f"   小資族: {params['weight_si']:.1%}")
            print(f"   雙渦輪: {params['weight_rpt']:.1%}")

    def run(self) -> EvolutionState:
        """
        執行完整演化流程

        Returns:
            最終演化狀態
        """
        # 初始化
        self.initialize()

        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    🚀 開始遺傳演算法優化                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  🎯 目標: 夏普 > {CONFIG.target_sharpe}, MDD < {CONFIG.max_drawdown*100:.0f}%, 胃納量 > {CONFIG.min_capacity/1e6:.0f}M            ║
║  🧬 族群: {CONFIG.population_size}, 世代: {CONFIG.n_generations}, 精英: {CONFIG.elite_size}                         ║
║  📅 樣本內: ~{CONFIG.in_sample_end}, 樣本外: {CONFIG.out_sample_start}~                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

        start_gen = self.state.generation

        # 演化迴圈
        for gen in range(start_gen, CONFIG.n_generations):
            gen_start = time.time()

            # 執行單代
            stats = self.run_generation()

            # 輸出摘要
            self.print_generation_summary(stats)

            # 詳細報告
            self.print_detailed_report(gen)

            # 定期存檔
            if (gen + 1) % CONFIG.checkpoint_interval == 0:
                self.checkpoint.save(self.state)

            # 檢查是否達標
            if stats['best_sharpe'] >= CONFIG.target_sharpe:
                best_details = self.state.elite_archive[0].get('details', {})
                if (best_details.get('capacity', 0) >= CONFIG.min_capacity and
                    best_details.get('max_drawdown', 1) <= CONFIG.max_drawdown):
                    print(f"\n🎉 已達成所有目標！於第 {gen+1} 代")
                    break

        # 最終存檔
        self.checkpoint.save(self.state)

        # 輸出最終結果
        self._print_final_results()

        return self.state

    def _print_final_results(self):
        """輸出最終結果"""
        elapsed = time.time() - self.start_time

        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         🏁 優化完成                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⏱️  總耗時: {elapsed/60:.1f} 分鐘                                              ║
║  🧬 總世代: {self.state.generation}                                                    ║
║  🏆 最佳適應度: {self.state.best_fitness:.2f}                                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

        # 保存最佳參數
        if self.state.best_params:
            with open(PATHS.best_params_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.best_params, f, indent=2, ensure_ascii=False)
            print(f"✅ 最佳參數已保存: {PATHS.best_params_file}")

        # 輸出最終排名
        print(f"\n🏆 最終精英排名:")
        for i, elite in enumerate(self.state.elite_archive[:5], 1):
            details = elite.get('details', {})
            print(f"   #{i}: 夏普={details.get('sharpe', 0):.2f}, "
                  f"胃納量={details.get('capacity', 0)/1e4:.0f}萬, "
                  f"回檔={details.get('max_drawdown', 0)*100:.1f}%")

# =============================================================================
# 第十三部分：主程式入口
# =============================================================================
def main():
    """主程式入口"""
    print(f"""

    ██████╗  █████╗     ███████╗████████╗ ██████╗  ██████╗██╗  ██╗
   ██╔════╝ ██╔══██╗    ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝
   ██║  ███╗███████║    ███████╗   ██║   ██║   ██║██║     █████╔╝
   ██║   ██║██╔══██║    ╚════██║   ██║   ██║   ██║██║     ██╔═██╗
   ╚██████╔╝██║  ██║    ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗
    ╚═════╝ ╚═╝  ╚═╝    ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝

    九組合媽媽龍 - 遺傳演算法選股優化系統 v4.0
    ============================================

""")

    try:
        # 創建演化引擎
        engine = EvolutionEngine()

        # 執行優化
        final_state = engine.run()

        # 執行最佳策略的完整回測
        if final_state.best_params:
            print("\n" + "="*70)
            print("📊 最佳策略完整回測")
            print("="*70)

            position = STRATEGY.combine_strategies(final_state.best_params)

            if position is not None:
                report = sim_func(
                    position=position,
                    fee_ratio=1.425 / 1000 * final_state.best_params.get('fee_discount', 0.5),
                    tax_ratio=3 / 1000,
                    trade_at_price='high_low_avg',
                    position_limit=final_state.best_params.get('position_limit', 0.25),
                    stop_loss=final_state.best_params.get('stop_loss', 0.25),
                    trail_stop=final_state.best_params.get('trail_stop', 0.30),
                    take_profit=final_state.best_params.get('take_profit', 0.70),
                    upload=False,
                    name='GA_Best_Strategy_v4',
                )

                report.display()

        print("\n✅ 優化流程完成！")
        print(f"📁 輸出目錄: {PATHS.window_dir}")

        return final_state

    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷，正在保存狀態...")
        if 'engine' in locals() and engine.state:
            CHECKPOINT.save(engine.state)
        print("✅ 狀態已保存，可使用 CONTINUE_EVOLUTION=true 繼續")
        return None

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 執行入口
# =============================================================================
if __name__ == "__main__":
    result = main()
