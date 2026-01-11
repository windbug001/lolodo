#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 九組合天空龍 - 遺傳演算法選股優化系統 v5.0
   Nine Combo Sky Dragon - Genetic Algorithm Stock Optimizer
================================================================================

【系統特色】
✅ 9 個獨立評分策略的動態權重組合
✅ 100 個可優化基因參數
✅ 基於遺傳演算法 (GA) 自動優化
✅ 支援 Google Colab Pro+ 多核心並行運算
✅ Checkpoint 機制 - 支援中斷後無縫重啟
✅ 歷史前 10 強個體保留與持續進化
✅ 樣本內/外分離驗證 - 避免過擬合

【九大策略】
S1: 低波動價值股    S2: 小型成長股      S3: 營收雙渦輪
S4: 高殖利率價值股  S5: 低波動穩健股    S6: 創高突破股
S7: 財務品質股      S8: 技術動能股      S9: 綜合品質動能股

【績效目標】
- 夏普值 (Sharpe Ratio) > 4.2
- 最大回檔 (MDD) < 20%
- 資金胃納量 > 1,000 萬台幣

【回測規範】
- 樣本內 (In-Sample)：~ 2022 年底
- 樣本外 (Out-of-Sample)：2023 年 ~ 至今
- 每 10 代進行一次詳細回測報告

【持股邏輯】
- 單一標的權重 ≥ 3%，否則不持倉 (0%)

版本：v5.0 (2025-01-11)
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
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

# Pandas 設定
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
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
    api_key: str = os.environ.get('FINLAB_API_KEY', '')

    # === 績效目標 ===
    target_sharpe: float = 4.2           # 目標夏普值
    max_drawdown: float = 0.20           # 最大回檔限制 (20%)
    min_capacity: int = 10_000_000       # 最小胃納量 (1000萬)
    target_annual_return: float = 0.40   # 目標年化報酬

    # === GA 演化參數 ===
    population_size: int = int(os.environ.get('POPULATION_SIZE', '30'))
    n_generations: int = int(os.environ.get('EVOLUTION_GENERATIONS', '100'))
    mutation_rate: float = 0.25          # 變異率
    crossover_rate: float = 0.85         # 交叉率
    elite_size: int = 10                 # 精英保留數量
    tournament_size: int = 5             # 錦標賽選擇大小

    # === 並行運算設定 ===
    n_workers: int = max(1, mp.cpu_count() - 1)
    use_parallel: bool = os.environ.get('USE_PARALLEL', 'true').lower() == 'true'

    # === 回測時間設定 ===
    train_start: str = '2017-01-01'      # 訓練起始日
    train_end: str = '2022-12-31'        # 訓練結束日（樣本內）
    test_start: str = '2023-01-01'       # 測試起始日（樣本外）
    test_end: str = datetime.now().strftime('%Y-%m-%d')

    # === Checkpoint 設定 ===
    checkpoint_interval: int = 5         # 每 N 代存檔一次
    report_interval: int = 10            # 每 N 代詳細報告
    visualization_interval: int = 5      # 每 N 代視覺化

    # === 持股邏輯 ===
    min_position_weight: float = 0.03    # 最小持股權重 (3%)
    target_stocks: int = 15              # 目標持股數
    min_volume_filter: int = 100000      # 最小成交量過濾

    # === 重啟驗證 ===
    validate_top_n: int = 5              # 重啟時驗證前 N 名
    continue_evolution: bool = os.environ.get('CONTINUE_EVOLUTION', 'false').lower() == 'true'

# 全域配置實例
CONFIG = GAConfig()

# 基因長度
GENE_LENGTH = 100

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🐉 九組合天空龍 - 遺傳演算法選股優化系統 v5.0                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 視窗: {CONFIG.window_id:<2} | 🎯 目標夏普: {CONFIG.target_sharpe} | 💰 胃納量: {CONFIG.min_capacity/1e6:.0f}M | 🧬 基因: {GENE_LENGTH}     ║
║  👥 族群: {CONFIG.population_size:<2} | 🔄 世代: {CONFIG.n_generations:<3} | ⚡ 並行: {CONFIG.n_workers} 核心 | 📈 策略: 9 個          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 第三部分：路徑管理器
# =============================================================================
@dataclass
class PathManager:
    """路徑管理器 - 統一管理所有檔案路徑"""
    base_dir: str = field(default='')
    window_id: int = field(default=1)
    in_colab: bool = field(default=False)

    def __post_init__(self):
        """初始化路徑結構"""
        # 檢測環境
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            self.base_dir = os.environ.get('BASE_DIR',
                '/content/drive/MyDrive/九組合天空龍_GA_v5')
            self.in_colab = True
            print("✅ Google Colab 環境 - Drive 已掛載")
        except:
            self.base_dir = os.environ.get('BASE_DIR', './dragon_ga_output')
            self.in_colab = False
            print("⚠️  本地環境模式")

        self.window_id = CONFIG.window_id

        # 建立目錄結構
        self.checkpoint_dir = f"{self.base_dir}/checkpoints"
        self.report_dir = f"{self.base_dir}/reports"
        self.elite_dir = f"{self.base_dir}/elites"
        self.log_dir = f"{self.base_dir}/logs"

        for d in [self.base_dir, self.checkpoint_dir, self.report_dir,
                  self.elite_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> str:
        return f"{self.checkpoint_dir}/checkpoint_w{self.window_id}.pkl"

    @property
    def pareto_file(self) -> str:
        return f"{self.elite_dir}/pareto_v5_w{self.window_id}.pkl"

    @property
    def best_params_file(self) -> str:
        return f"{self.base_dir}/best_params_v5_w{self.window_id}.json"

    @property
    def progress_file(self) -> str:
        return f"{self.log_dir}/progress_w{self.window_id}.json"

# 初始化路徑管理器
PATHS = PathManager()
print(f"📁 工作目錄: {PATHS.base_dir}")

# =============================================================================
# 第四部分：FinLab 初始化與資料載入
# =============================================================================
def get_finlab_api_key() -> str:
    """獲取 FinLab API Key"""
    # 優先從環境變數
    api_key = os.environ.get('FINLAB_API_KEY', '')
    if api_key:
        return api_key

    # 嘗試從 Colab userdata
    try:
        from google.colab import userdata
        api_key = userdata.get('FINLAB_API_KEY')
        if api_key:
            return api_key
    except:
        pass

    # 預設 API Key
    return "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"


def clear_finlab_cache():
    """清除 FinLab 快取"""
    import glob
    import shutil

    cache_patterns = [
        '/root/.finlab*', '/tmp/.finlab*', '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'), '/content/.finlab*',
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

    # 安裝必要套件
    try:
        import finlab
    except ImportError:
        print("   📦 安裝 finlab...")
        os.system('pip install finlab -q')
        import finlab

    # 清除快取
    clear_finlab_cache()

    # 登入
    api_key = get_finlab_api_key()
    if api_key:
        finlab.login(api_key)
        print(f"   ✅ FinLab {finlab.__version__} 登入成功")
    else:
        print("   ⚠️  未提供 API Key")

    from finlab import data
    from finlab.backtest import sim

    return finlab, data, sim


# 初始化 FinLab
finlab_module, data_api, sim_func = initialize_finlab()


class DataLoader:
    """
    FinLab 資料載入器

    負責從 FinLab API 載入所有必要的財務與技術指標資料。
    資料來源參考：https://ai.finlab.tw/database
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_all_data()
            DataLoader._initialized = True

    def _safe_get(self, key: str) -> Optional[pd.DataFrame]:
        """安全載入資料"""
        try:
            return data_api.get(key)
        except Exception as e:
            print(f"   ⚠️  載入失敗: {key} - {e}")
            return None

    def _load_all_data(self):
        """載入所有必要資料"""
        print("\n📥 載入 FinLab 資料庫...")
        start_time = time.time()

        # 設定股票宇宙
        data_api.set_universe('TSE_OTC')

        # === 價格資料 ===
        print("   📈 價格資料...")
        self.close = self._safe_get('price:收盤價')
        self.open = self._safe_get('price:開盤價')
        self.high = self._safe_get('price:最高價')
        self.low = self._safe_get('price:最低價')
        self.volume = self._safe_get('price:成交股數')
        self.adj_close = self._safe_get('etl:adj_close')

        # === 估值指標 ===
        print("   💰 估值指標...")
        self.pe = self._safe_get('price_earning_ratio:本益比')
        self.pb = self._safe_get('price_earning_ratio:股價淨值比')
        self.dividend_yield = self._safe_get('price_earning_ratio:殖利率(%)')

        # === 營收資料 ===
        print("   📊 營收資料...")
        self.revenue = self._safe_get('monthly_revenue:當月營收')
        self.revenue_yoy = self._safe_get('monthly_revenue:去年同月增減(%)')
        self.revenue_mom = self._safe_get('monthly_revenue:上月比較增減(%)')

        # === 基本面指標 ===
        print("   📑 基本面指標...")
        self.op_profit_growth = self._safe_get('fundamental_features:營業利益成長率')
        self.non_op_ratio = self._safe_get('fundamental_features:業外收支營收率')
        self.gross_margin = self._safe_get('fundamental_features:營業毛利率')
        self.roe = self._safe_get('fundamental_features:ROE綜合損益')
        self.net_profit_margin = self._safe_get('fundamental_features:稅後淨利率')
        self.op_margin = self._safe_get('fundamental_features:營業利益率')

        # === 籌碼資料 ===
        print("   🏦 籌碼資料...")
        self.margin_usage = self._safe_get('margin_transactions:融資使用率')
        self.director_holdings = self._safe_get('internal_equity_changes:董監持有股數占比')

        # === 市值與財報 ===
        print("   📋 市值與財報...")
        self.market_value = self._safe_get('etl:market_value')
        self.inv_cashflow = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self.op_cashflow = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self.net_income = self._safe_get('fundamental_features:經常稅後淨利')
        self.equity = self._safe_get('financial_statement:股東權益總額')

        # === 計算衍生指標 ===
        print("   🔧 計算衍生指標...")
        self._calculate_derived()

        elapsed = time.time() - start_time
        print(f"   ✅ 資料載入完成 ({elapsed:.1f} 秒)")

    def _calculate_derived(self):
        """計算衍生指標"""
        # 技術指標
        self.rsi = data_api.indicator('RSI', adjust_price=False, resample='D', timeperiod=5)
        self.atr = data_api.indicator('ATR', adjust_price=True, timeperiod=10)

        # 營收均線
        self.rev_ma3 = self.revenue.average(3)
        self.rev_ma12 = self.revenue.average(12)

        # 波動率
        if self.atr is not None and self.adj_close is not None:
            self.volatility = self.atr / self.adj_close
        else:
            self.volatility = None

        # 自由現金流
        if self.inv_cashflow is not None and self.op_cashflow is not None:
            self.free_cashflow = (self.inv_cashflow + self.op_cashflow).rolling(4).mean()
        else:
            self.free_cashflow = None

        # ROE 計算
        if self.net_income is not None and self.equity is not None:
            self.roe_calc = self.net_income / self.equity
        else:
            self.roe_calc = self.roe

        # 當月營收（千元轉元）
        self.monthly_rev = self.revenue * 1000 if self.revenue is not None else None

        # 市值營收比
        if self.market_value is not None and self.monthly_rev is not None:
            self.ps_ratio = self.market_value / self.monthly_rev.rolling(4).sum()
        else:
            self.ps_ratio = None

        # 漲停鎖死判斷
        if all(x is not None for x in [self.close, self.high, self.low, self.open]):
            limit_up = self.close > self.close.shift(1) * 1.095
            locked = ((self.close == self.high) & (self.close == self.low) &
                      (self.close == self.open))
            self.limit_up_locked = (limit_up & locked).fillna(False)
        else:
            self.limit_up_locked = None

        # 流動性遮罩
        if self.volume is not None:
            self.liquid_mask = self.volume.average(20) > CONFIG.min_volume_filter
        else:
            self.liquid_mask = None


# 初始化資料載入器
DATA = DataLoader()

# =============================================================================
# 第五部分：基因解碼器 (100 個基因)
# =============================================================================
class GeneDecoder:
    """
    基因解碼器

    將 100 個 0-1 之間的基因值解碼為策略參數。

    基因結構：
    [0-8]   : 9 個策略權重 (w1-w9)
    [9-17]  : 9 個策略嚴格度 (s1-s9_strictness)
    [18-26] : 9 個策略評分權重 (s1-s9_score_weight)
    [27-35] : 技術面參數
    [36-44] : 基本面參數
    [45-53] : 籌碼面參數
    [54-62] : 流動性參數
    [63-71] : 動能參數
    [72-80] : 價值參數
    [81-89] : 風控參數
    [90-99] : 全局參數
    """

    @staticmethod
    def decode(genes: List[float]) -> Dict[str, Any]:
        """解碼基因為參數字典"""
        # 確保基因長度
        genes = list(genes) + [0.5] * (GENE_LENGTH - len(genes))
        genes = genes[:GENE_LENGTH]

        p = {}

        # === 策略權重 [0-8] ===
        raw_weights = [max(0.05, genes[i]) for i in range(9)]
        total_weight = sum(raw_weights)
        for i in range(9):
            p[f'w{i+1}'] = raw_weights[i] / total_weight

        # === 策略嚴格度與評分權重 [9-26] ===
        for i in range(9):
            p[f's{i+1}_strictness'] = genes[9 + i] * 0.8 + 0.1      # 0.1 ~ 0.9
            p[f's{i+1}_score_weight'] = genes[18 + i] * 2.0 + 0.5   # 0.5 ~ 2.5

        # === 技術面參數 [27-35] ===
        p['ma_short'] = int(genes[27] * 40 + 5)        # 5 ~ 45
        p['ma_mid'] = int(genes[28] * 100 + 20)        # 20 ~ 120
        p['ma_long'] = int(genes[29] * 200 + 60)       # 60 ~ 260
        p['rsi_low'] = genes[30] * 30 + 20             # 20 ~ 50
        p['rsi_high'] = genes[31] * 30 + 60            # 60 ~ 90
        p['atr_mult'] = genes[32] * 2.0 + 0.5          # 0.5 ~ 2.5
        p['price_momentum'] = int(genes[33] * 60 + 20) # 20 ~ 80
        p['vol_ma'] = int(genes[34] * 20 + 5)          # 5 ~ 25
        p['breakout_period'] = int(genes[35] * 200 + 60)  # 60 ~ 260

        # === 基本面參數 [36-44] ===
        p['pe_min'] = genes[36] * 10 + 3               # 3 ~ 13
        p['pe_max'] = genes[37] * 30 + 15              # 15 ~ 45
        p['pb_max'] = genes[38] * 5 + 1                # 1 ~ 6
        p['roe_min'] = genes[39] * 20                  # 0 ~ 20
        p['gpm_min'] = genes[40] * 30                  # 0 ~ 30
        p['rev_growth_min'] = genes[41] * 50 - 20     # -20 ~ 30
        p['opm_min'] = genes[42] * 20                  # 0 ~ 20
        p['fcf_positive'] = genes[43] > 0.5           # True/False
        p['dividend_min'] = genes[44] * 5              # 0 ~ 5

        # === 籌碼面參數 [45-53] ===
        p['margin_max'] = genes[45] * 40 + 10         # 10 ~ 50
        p['director_min'] = genes[46] * 30            # 0 ~ 30
        p['foreign_trend'] = int(genes[47] * 20 + 5)  # 5 ~ 25
        p['trust_trend'] = int(genes[48] * 20 + 5)    # 5 ~ 25
        p['chip_score_weight'] = genes[49] * 1.5 + 0.5  # 0.5 ~ 2.0
        p['big_player_ratio'] = genes[50] * 30 + 10   # 10 ~ 40
        p['retail_max'] = genes[51] * 30 + 30         # 30 ~ 60
        p['institutional_min'] = genes[52] * 20       # 0 ~ 20
        p['chip_momentum'] = int(genes[53] * 10 + 5)  # 5 ~ 15

        # === 流動性參數 [54-62] ===
        p['min_vol'] = genes[54] * 300000 + 50000     # 50K ~ 350K
        p['min_cap'] = genes[55] * 5e9 + 1e9          # 1B ~ 6B
        p['max_cap'] = genes[56] * 200e9 + 10e9       # 10B ~ 210B
        p['vol_spike'] = genes[57] * 2.0 + 1.0        # 1.0 ~ 3.0
        p['liquidity_score_weight'] = genes[58] * 1.5 + 0.5  # 0.5 ~ 2.0
        p['turnover_min'] = genes[59] * 0.5           # 0 ~ 0.5
        p['spread_max'] = genes[60] * 2.0 + 0.5       # 0.5 ~ 2.5
        p['vol_consistency'] = genes[61] * 0.5 + 0.3  # 0.3 ~ 0.8
        p['price_min'] = genes[62] * 20 + 10          # 10 ~ 30

        # === 動能參數 [63-71] ===
        p['momentum_window'] = int(genes[63] * 60 + 20)  # 20 ~ 80
        p['momentum_threshold'] = genes[64] * 0.3     # 0 ~ 0.3
        p['trend_strength'] = genes[65] * 0.5 + 0.3   # 0.3 ~ 0.8
        p['volatility_max'] = genes[66] * 0.1 + 0.02  # 0.02 ~ 0.12
        p['momentum_score_weight'] = genes[67] * 1.5 + 0.5  # 0.5 ~ 2.0
        p['acceleration'] = genes[68] * 0.3           # 0 ~ 0.3
        p['breakout_strength'] = genes[69] * 0.5 + 0.3  # 0.3 ~ 0.8
        p['trend_days'] = int(genes[70] * 30 + 10)    # 10 ~ 40
        p['reversal_threshold'] = genes[71] * 0.2 + 0.1  # 0.1 ~ 0.3

        # === 價值參數 [72-80] ===
        p['value_pe_weight'] = genes[72] * 1.5 + 0.5  # 0.5 ~ 2.0
        p['value_pb_weight'] = genes[73] * 1.5 + 0.5
        p['value_div_weight'] = genes[74] * 1.5 + 0.5
        p['value_fcf_weight'] = genes[75] * 1.5 + 0.5
        p['value_growth_weight'] = genes[76] * 1.5 + 0.5
        p['value_score_weight'] = genes[77] * 1.5 + 0.5
        p['margin_of_safety'] = genes[78] * 0.3 + 0.1  # 0.1 ~ 0.4
        p['quality_premium'] = genes[79] * 0.5        # 0 ~ 0.5
        p['growth_discount'] = genes[80] * 0.3        # 0 ~ 0.3

        # === 風控參數 [81-89] ===
        p['stop_loss'] = genes[81] * 0.15 + 0.10      # 10% ~ 25%
        p['trail_stop'] = genes[82] * 0.25 + 0.15     # 15% ~ 40%
        p['take_profit'] = genes[83] * 0.5 + 0.3      # 30% ~ 80%
        p['position_limit'] = genes[84] * 0.20 + 0.15 # 15% ~ 35%
        p['drawdown_exit'] = genes[85] * 0.15 + 0.15  # 15% ~ 30%
        p['correlation_limit'] = genes[86] * 0.3 + 0.5  # 0.5 ~ 0.8
        p['sector_limit'] = genes[87] * 0.3 + 0.2     # 20% ~ 50%
        p['beta_max'] = genes[88] * 1.0 + 1.0         # 1.0 ~ 2.0
        p['var_limit'] = genes[89] * 0.1 + 0.05       # 5% ~ 15%

        # === 全局參數 [90-99] ===
        p['n_stocks'] = int(genes[90] * 10 + 10)      # 10 ~ 20
        p['rebalance_threshold'] = genes[91] * 0.15 + 0.05  # 5% ~ 20%
        p['score_decay'] = genes[92] * 0.3 + 0.7      # 0.7 ~ 1.0
        p['overlap_bonus'] = genes[93] * 1.5 + 1.0    # 1.0 ~ 2.5
        p['diversity_weight'] = genes[94] * 0.5       # 0 ~ 0.5
        p['momentum_blend'] = genes[95] * 0.5 + 0.25  # 0.25 ~ 0.75
        p['value_blend'] = genes[96] * 0.5 + 0.25
        p['quality_blend'] = genes[97] * 0.5 + 0.25
        p['entry_timing'] = genes[98] * 0.3           # 0 ~ 0.3
        p['exit_timing'] = genes[99] * 0.3            # 0 ~ 0.3

        return p

    @staticmethod
    def encode(params: Dict[str, Any]) -> List[float]:
        """將參數字典編碼回基因（用於精英注入）"""
        genes = [0.5] * GENE_LENGTH

        # 策略權重
        for i in range(9):
            w = params.get(f'w{i+1}', 1/9)
            genes[i] = min(1.0, max(0.0, w * 9 / 1.05))  # 近似還原

        # 其他參數的逆向編碼（簡化版本）
        if 'ma_short' in params:
            genes[27] = (params['ma_short'] - 5) / 40
        if 'pe_min' in params:
            genes[36] = (params['pe_min'] - 3) / 10
        if 'stop_loss' in params:
            genes[81] = (params['stop_loss'] - 0.10) / 0.15
        if 'n_stocks' in params:
            genes[90] = (params['n_stocks'] - 10) / 10

        return [max(0.0, min(1.0, g)) for g in genes]

# =============================================================================
# 第六部分：九大評分策略
# =============================================================================
class StrategyScorer:
    """
    九大策略評分器

    每個策略獨立計算評分，最終加權組合。
    """

    def __init__(self, data_loader: DataLoader):
        self.d = data_loader

    def score_strategy_1(self, p: Dict) -> pd.DataFrame:
        """
        策略一：低波動價值股

        選股邏輯：
        - 本益比合理
        - 低波動率
        - 營收成長穩定
        - 高毛利率與 ROE
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']) &
                        ~self.d.limit_up_locked)

            # PE 評分
            pe_valid = (self.d.pe > p['pe_min']) & (self.d.pe < p['pe_max'])
            pe_score = (1 - (self.d.pe - p['pe_min']) / (p['pe_max'] - p['pe_min'] + 1e-10))
            pe_score = pe_score.clip(0, 1).where(pe_valid, 0)

            # 波動率評分
            vol_score = (1 - self.d.volatility / p['volatility_max']).clip(0, 1)

            # 營收成長評分
            rev_ratio = self.d.rev_ma3 / (self.d.rev_ma12 + 1e-10)
            rev_score = ((rev_ratio - 1) / 0.5 + 0.5).clip(0, 1)

            # 毛利率評分
            gpm_score = (self.d.gross_margin / 50).clip(0, 1)

            # ROE 評分
            roe_score = (self.d.roe / 30).clip(0, 1)

            # 綜合評分
            total = (pe_score * 0.25 + vol_score * 0.25 + rev_score * 0.20 +
                    gpm_score * 0.15 + roe_score * 0.15) * p['s1_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略1錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_2(self, p: Dict) -> pd.DataFrame:
        """
        策略二：小型成長股

        選股邏輯：
        - 中小市值
        - 正自由現金流
        - 低市值營收比
        - RSV 動能
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.market_value < p['max_cap']) &
                        (self.d.volume.average(20) > p['min_vol']))

            # 市值評分
            cap_score = (1 - self.d.market_value / p['max_cap']).clip(0, 1)
            cap_score = cap_score.where(self.d.market_value > p['min_cap'], 0)

            # 自由現金流評分
            fcf_score = (self.d.free_cashflow > 0).astype(float)
            if p['fcf_positive']:
                fcf_score = fcf_score * 1.5

            # 市值營收比評分
            ps_score = (1 - self.d.ps_ratio / 5).clip(0, 1)

            # ROE 評分
            roe_score = (self.d.roe_calc * 10).clip(0, 1)

            # RSV 動能
            window = p['momentum_window']
            rsv = ((self.d.close - self.d.close.rolling(window).min()) /
                   (self.d.close.rolling(window).max() - self.d.close.rolling(window).min() + 1e-10))

            # 綜合評分
            total = (cap_score * 0.20 + fcf_score * 0.25 + ps_score * 0.20 +
                    roe_score * 0.20 + rsv.clip(0, 1) * 0.15) * p['s2_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略2錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_3(self, p: Dict) -> pd.DataFrame:
        """
        策略三：營收雙渦輪

        選股邏輯：
        - 營收創新高
        - 股價創新高
        - 營收年增強勁
        - RSI 動能
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']) &
                        ~self.d.limit_up_locked)

            # 營收創新高評分
            rev_ma = self.d.revenue.average(3)
            rev_high = (rev_ma == rev_ma.rolling(12, min_periods=3).max()).astype(float)

            # 股價創新高評分
            price_high = (self.d.close == self.d.close.rolling(p['breakout_period']).max()).astype(float) * 0.8
            price_high += (self.d.close > self.d.close.average(p['ma_long'])).astype(float) * 0.2

            # 營收年增評分
            rev_yoy_score = (self.d.revenue_yoy / 100 + 0.5).clip(0, 1)

            # 毛利率穩定性評分
            gpm_std = self.d.gross_margin.rolling(4).std()
            gpm_score = (gpm_std < 5).astype(float) * 0.5 + (self.d.gross_margin / 40).clip(0, 0.5)

            # RSI 評分
            rsi_score = ((self.d.rsi - 30) / 40).clip(0, 1)

            # 綜合評分
            total = (rev_high * 0.30 + price_high * 0.25 + rev_yoy_score * 0.20 +
                    gpm_score * 0.15 + rsi_score * 0.10) * p['s3_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略3錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_4(self, p: Dict) -> pd.DataFrame:
        """
        策略四：高殖利率價值股

        選股邏輯：
        - 高殖利率
        - 董監持股穩定
        - 營業利益率佳
        - 均線多頭
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.dividend_yield > 0) &
                        (self.d.volume.average(20) > p['min_vol']))

            # 殖利率評分
            div_score = (self.d.dividend_yield / 10).clip(0, 1)

            # 董監持股評分
            director_score = (self.d.director_holdings / 50).clip(0, 1)

            # 營業利益率評分
            opm_score = (self.d.op_margin / 30).clip(0, 1)

            # 營收趨勢評分
            rev_trend = (self.d.revenue.average(3) > self.d.revenue.average(12)).astype(float)

            # 均線多頭評分
            ma_trend = ((self.d.close > self.d.close.average(p['ma_short'])) &
                       (self.d.close > self.d.close.average(p['ma_mid']))).astype(float)

            # 綜合評分
            total = (div_score * 0.30 + director_score * 0.20 + opm_score * 0.20 +
                    rev_trend * 0.15 + ma_trend * 0.15) * p['s4_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略4錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_5(self, p: Dict) -> pd.DataFrame:
        """
        策略五：低波動穩健股

        選股邏輯：
        - 波動率最低
        - 均線多頭排列
        - 低融資使用率
        - 成交量穩定
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']))

            # 波動率排名評分（越低越好）
            std_60 = self.d.close.pct_change().rolling(60).std()
            std_rank = std_60.rank(axis=1, pct=True)
            vol_score = (1 - std_rank).clip(0, 1)

            # 均線多頭評分
            ma_bull = ((self.d.close > self.d.close.average(p['ma_short'])).astype(float) * 0.3 +
                      (self.d.close > self.d.close.average(p['ma_mid'])).astype(float) * 0.3 +
                      (self.d.close > self.d.close.average(p['ma_long'])).astype(float) * 0.4)

            # 市值評分
            cap_rank = self.d.market_value.rank(axis=1, pct=True)
            cap_score = (1 - cap_rank).clip(0, 1)
            cap_score = cap_score.where(self.d.market_value > p['min_cap'], 0)

            # 融資使用率評分
            margin_score = (1 - self.d.margin_usage / p['margin_max']).clip(0, 1)

            # 成交量穩定性評分
            vol_std = self.d.volume.rolling(20).std() / (self.d.volume.rolling(20).mean() + 1e-10)
            vol_stable = (1 - vol_std / 2).clip(0, 1)

            # 綜合評分
            total = (vol_score * 0.30 + ma_bull * 0.25 + cap_score * 0.20 +
                    margin_score * 0.15 + vol_stable * 0.10) * p['s5_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略5錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_6(self, p: Dict) -> pd.DataFrame:
        """
        策略六：創高突破股

        選股邏輯：
        - 股價接近/創260日新高
        - 成交量放大
        - 營收月增、年增正向
        - 均線支撐
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']) &
                        ~self.d.limit_up_locked)

            # 股價創高評分
            high_260 = self.d.close.rolling(260).max()
            high_score = ((self.d.close / high_260).clip(0.8, 1) - 0.8) / 0.2

            # 成交量評分
            vol_ma = self.d.volume.average(p['vol_ma'])
            vol_score = ((self.d.volume / vol_ma - 0.5) / 2).clip(0, 1)

            # 營收月增評分
            rev_mom_score = ((self.d.revenue_mom + 20) / 40).clip(0, 1)

            # 營收年增評分
            rev_yoy_score = ((self.d.revenue_yoy + 30) / 60).clip(0, 1)

            # 均線支撐評分
            ma_support = (self.d.close > self.d.close.average(60)).astype(float)

            # 綜合評分
            total = (high_score * 0.35 + vol_score * 0.20 + rev_mom_score * 0.15 +
                    rev_yoy_score * 0.15 + ma_support * 0.15) * p['s6_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略6錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_7(self, p: Dict) -> pd.DataFrame:
        """
        策略七：財務品質股

        選股邏輯：
        - 營業利益成長率高
        - ROE 高
        - 毛利率高
        - 淨利率高
        - 低波動調整
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']))

            # 財務指標排名
            scores = []
            for f in [self.d.op_profit_growth, self.d.roe, self.d.gross_margin,
                     self.d.net_profit_margin, self.d.op_margin]:
                if f is not None:
                    scores.append(f.rank(axis=1, pct=True).fillna(0))

            if not scores:
                return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

            finance_score = sum(scores) / len(scores)

            # 波動調整
            std_60 = self.d.close.pct_change().rolling(60).std()
            vol_adj = (1 - std_60.rank(axis=1, pct=True)).fillna(0.5) * 0.3

            # 均線調整
            ma_adj = (self.d.close > self.d.close.average(p['ma_mid'])).astype(float) * 0.2

            # 綜合評分
            total = (finance_score * 0.7 + vol_adj + ma_adj) * p['s7_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略7錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_8(self, p: Dict) -> pd.DataFrame:
        """
        策略八：技術動能股

        選股邏輯：
        - SMA 動能強
        - 動能排名高
        - 均線斜率向上
        - 成交量趨勢向上
        - 中等波動
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume > p['min_vol']))

            # SMA 動能評分
            sma_ratio = self.d.close / self.d.close.rolling(p['ma_long']).mean() - 1
            sma_score = (sma_ratio / 0.3 + 0.5).clip(0, 1)

            # 動能排名評分
            mom_return = self.d.close / self.d.close.shift(p['momentum_window']) - 1
            mom_rank = mom_return.rank(axis=1, pct=True).fillna(0.5)

            # 均線斜率評分
            ma60 = self.d.close.rolling(60).mean()
            slope = ma60 / ma60.shift(60) - 1
            slope_score = (slope / 0.3 + 0.5).clip(0, 1)

            # 成交量趨勢評分
            vol_trend = self.d.volume.rolling(20).mean() / self.d.volume.rolling(60).mean() - 0.5
            vol_trend_score = vol_trend.clip(0, 1)

            # 中等波動評分（不要太高也不要太低）
            std_rank = self.d.close.pct_change().rolling(60).std().rank(axis=1, pct=True)
            vol_mid = 1 - abs(std_rank - 0.5) * 2

            # 綜合評分
            total = (sma_score * 0.25 + mom_rank * 0.25 + slope_score * 0.20 +
                    vol_trend_score * 0.15 + vol_mid * 0.15) * p['s8_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略8錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def score_strategy_9(self, p: Dict) -> pd.DataFrame:
        """
        策略九：綜合品質動能股

        選股邏輯：
        - 財務品質綜合排名
        - 均線多頭
        - 低波動
        - 成交量排名
        - 低融資使用率
        """
        try:
            # 基本條件
            base_cond = (self.d.liquid_mask &
                        (self.d.volume.average(20) > p['min_vol']))

            # 財務品質綜合評分
            fs = []
            for f in [self.d.op_profit_growth, self.d.roe, self.d.gross_margin,
                     self.d.net_profit_margin, self.d.op_margin]:
                if f is not None:
                    fs.append(f.average(4).rank(axis=1, pct=True).fillna(0))

            if fs:
                finance = sum(fs) / len(fs)
            else:
                finance = pd.DataFrame(0.5, index=self.d.close.index, columns=self.d.close.columns)

            # 均線多頭評分
            ma_bull = ((self.d.close > self.d.close.average(20)).astype(float) * 0.2 +
                      (self.d.close > self.d.close.average(60)).astype(float) * 0.3 +
                      (self.d.close > self.d.close.average(120)).astype(float) * 0.5)

            # 低波動評分
            std_rank = self.d.close.pct_change().rolling(60).std().rank(axis=1, pct=True)
            vol_low = (1 - std_rank).fillna(0.5)

            # 成交量排名評分
            vol_rank = self.d.volume.rolling(20).mean().rank(axis=1, pct=True).fillna(0.5)

            # 融資安全評分
            margin_safe = (self.d.margin_usage < p['margin_max']).astype(float)

            # 綜合評分
            total = (finance * 0.30 + ma_bull * 0.25 + vol_low * 0.20 +
                    vol_rank * 0.15 + margin_safe * 0.10) * p['s9_score_weight']

            return total.where(base_cond, 0).fillna(0)

        except Exception as e:
            print(f"   ⚠️  策略9錯誤: {e}")
            return pd.DataFrame(0, index=self.d.close.index, columns=self.d.close.columns)

    def get_all_strategies(self) -> List[Callable]:
        """獲取所有策略函數"""
        return [
            self.score_strategy_1,
            self.score_strategy_2,
            self.score_strategy_3,
            self.score_strategy_4,
            self.score_strategy_5,
            self.score_strategy_6,
            self.score_strategy_7,
            self.score_strategy_8,
            self.score_strategy_9,
        ]


# 初始化策略評分器
SCORER = StrategyScorer(DATA)

# =============================================================================
# 第七部分：策略組合器
# =============================================================================
class StrategyCombiner:
    """
    策略組合器

    將 9 個策略的評分加權組合，並進行持股權重正規化。
    """

    def __init__(self, scorer: StrategyScorer):
        self.scorer = scorer
        self.strategies = scorer.get_all_strategies()
        self.strategy_names = [
            '低波動價值', '小型成長', '營收雙渦輪',
            '高殖利率', '低波動穩健', '創高突破',
            '財務品質', '技術動能', '綜合品質動能'
        ]

    def combine(self, params: Dict) -> pd.DataFrame:
        """
        組合所有策略

        Args:
            params: 解碼後的參數字典

        Returns:
            持股權重 DataFrame
        """
        scores = []

        # 計算各策略評分
        for i, strategy_func in enumerate(self.strategies):
            try:
                score = strategy_func(params)
                if score is not None and not score.empty:
                    weight = params.get(f'w{i+1}', 1/9)
                    scores.append(score * weight)
            except Exception as e:
                print(f"   ⚠️  策略{i+1}執行錯誤: {e}")

        if not scores:
            return pd.DataFrame()

        # 加權合併
        combined = reduce(lambda a, b: a.add(b, fill_value=0), scores)
        combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 持股數量限制
        n_stocks = params.get('n_stocks', CONFIG.target_stocks)
        max_stocks = int(1 / CONFIG.min_position_weight)
        n_stocks = min(n_stocks, max_stocks)

        # 正規化權重
        def normalize_row(row):
            """正規化單列權重，確保最小權重限制"""
            try:
                row_numeric = pd.to_numeric(row, errors='coerce').fillna(0)
                valid = row_numeric[row_numeric > 0]

                if len(valid) == 0:
                    return row * 0.0  # 保持原索引，全部設為 0

                # 選出 top N
                top_n = valid.nlargest(min(n_stocks, len(valid)))

                # 迭代移除不滿足最小權重的標的
                for _ in range(10):
                    total = top_n.sum()
                    if total <= 0:
                        return row * 0.0

                    normalized = top_n / total
                    below_min = normalized < CONFIG.min_position_weight

                    if not below_min.any():
                        # 使用 reindex 確保索引對齊
                        return normalized.reindex(row.index, fill_value=0.0)

                    top_n = top_n[~below_min]
                    if len(top_n) == 0:
                        return row * 0.0

                # 最終正規化
                total = top_n.sum()
                if total > 0:
                    final = top_n / total
                    return final.reindex(row.index, fill_value=0.0)

                return row * 0.0

            except Exception:
                return row * 0.0  # 發生錯誤時返回全零

        return combined.apply(normalize_row, axis=1)


# 初始化策略組合器
COMBINER = StrategyCombiner(SCORER)

# =============================================================================
# 第八部分：回測引擎
# =============================================================================
class BacktestEngine:
    """回測引擎"""

    @staticmethod
    def run(position: pd.DataFrame, params: Dict,
            start_date: str = None, end_date: str = None,
            name: str = 'Strategy') -> Optional[Dict]:
        """
        執行回測

        Args:
            position: 持股權重 DataFrame
            params: 策略參數
            start_date: 開始日期
            end_date: 結束日期
            name: 策略名稱

        Returns:
            回測結果字典
        """
        try:
            if position is None or position.empty:
                return None

            # 過濾時間
            if start_date:
                position = position.loc[start_date:]
            if end_date:
                position = position.loc[:end_date]

            if len(position) < 50:
                return None

            # 執行回測
            report = sim_func(
                position=position,
                fee_ratio=1.425 / 1000 * 0.5,  # 5折手續費
                tax_ratio=3 / 1000,
                trade_at_price='high_low_avg',
                position_limit=params.get('position_limit', 0.25),
                stop_loss=params.get('stop_loss', 0.20),
                trail_stop=params.get('trail_stop', 0.25),
                take_profit=params.get('take_profit', 0.50),
                stop_trading_next_period=False,
                upload=False,
                name=name,
            )

            # 獲取績效指標
            metrics = report.get_metrics()

            return {
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

        except Exception as e:
            print(f"   ⚠️  回測錯誤 ({name}): {e}")
            return None

    @staticmethod
    def evaluate_fitness(genes: List[float]) -> Tuple[float, Dict]:
        """
        評估個體適應度

        Args:
            genes: 基因陣列

        Returns:
            (適應度分數, 詳細結果)
        """
        try:
            # 解碼基因
            params = GeneDecoder.decode(genes)

            # 組合策略
            position = COMBINER.combine(params)

            if position is None or position.empty:
                return 0.0, {'error': 'No position'}

            # 樣本內回測
            in_sample = BacktestEngine.run(
                position, params,
                start_date=CONFIG.train_start,
                end_date=CONFIG.train_end,
                name='InSample'
            )

            # 樣本外回測
            out_sample = BacktestEngine.run(
                position, params,
                start_date=CONFIG.test_start,
                end_date=CONFIG.test_end,
                name='OutSample'
            )

            # 主要使用樣本外結果
            primary = out_sample or in_sample

            if not primary:
                return 0.0, {'error': 'Backtest failed'}

            # === 計算適應度 ===
            sharpe = primary['sharpe']
            capacity = primary['capacity']
            mdd = primary['max_drawdown']
            annual_return = primary['annual_return']

            # 夏普值分數 (0-50)
            sharpe_score = min(50, max(0, sharpe / CONFIG.target_sharpe * 50))

            # 胃納量分數 (0-20)
            capacity_score = min(20, max(0, capacity / CONFIG.min_capacity * 20))

            # MDD 懲罰 (-20 to 0)
            if mdd > CONFIG.max_drawdown:
                mdd_penalty = -20 * (mdd - CONFIG.max_drawdown) / CONFIG.max_drawdown
            else:
                mdd_penalty = 0

            # 年化報酬分數 (0-15)
            return_score = min(15, max(0, annual_return / CONFIG.target_annual_return * 15))

            # 穩健性分數 (0-15)
            robustness = 0
            if in_sample and out_sample and in_sample['sharpe'] > 0:
                ratio = out_sample['sharpe'] / (in_sample['sharpe'] + 1e-10)
                robustness = min(15, max(0, ratio * 10))

            # 總適應度
            fitness = sharpe_score + capacity_score + mdd_penalty + return_score + robustness

            details = {
                'sharpe': sharpe,
                'capacity': capacity,
                'max_drawdown': mdd,
                'annual_return': annual_return,
                'sharpe_score': sharpe_score,
                'capacity_score': capacity_score,
                'mdd_penalty': mdd_penalty,
                'return_score': return_score,
                'robustness': robustness,
                'fitness': fitness,
                'in_sample': in_sample,
                'out_sample': out_sample,
            }

            return fitness, details

        except Exception as e:
            print(f"   ⚠️  適應度評估錯誤: {e}")
            return 0.0, {'error': str(e)}

# =============================================================================
# 第九部分：Checkpoint 管理器
# =============================================================================
@dataclass
class EvolutionState:
    """演化狀態"""
    generation: int = 0
    population: List[List[float]] = field(default_factory=list)
    fitness_scores: List[float] = field(default_factory=list)
    elite_archive: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    best_genes: Optional[List[float]] = None
    best_fitness: float = 0.0
    best_params: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CheckpointManager:
    """Checkpoint 管理器"""

    def __init__(self):
        self.checkpoint_file = PATHS.checkpoint_file
        self.pareto_file = PATHS.pareto_file

    def save(self, state: EvolutionState):
        """儲存狀態"""
        try:
            state.timestamp = datetime.now().isoformat()

            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)

            # 儲存 Pareto 存檔
            with open(self.pareto_file, 'wb') as f:
                pickle.dump({
                    'elite_archive': state.elite_archive,
                    'best_genes': state.best_genes,
                    'best_fitness': state.best_fitness,
                    'timestamp': state.timestamp,
                }, f)

            print(f"   💾 Checkpoint 已儲存 (第 {state.generation} 代)")

        except Exception as e:
            print(f"   ⚠️  儲存失敗: {e}")

    def load(self) -> Optional[EvolutionState]:
        """載入狀態"""
        try:
            if not os.path.exists(self.checkpoint_file):
                return None

            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)

            print(f"   ✅ 載入 Checkpoint (第 {state.generation} 代)")
            return state

        except Exception as e:
            print(f"   ⚠️  載入失敗: {e}")
            return None

    def load_pareto(self) -> List[Dict]:
        """載入 Pareto 存檔"""
        try:
            if os.path.exists(self.pareto_file):
                with open(self.pareto_file, 'rb') as f:
                    data = pickle.load(f)
                    return data.get('elite_archive', [])
        except:
            pass
        return []

    def validate_elites(self, elites: List[Dict], top_n: int = 5) -> List[Dict]:
        """驗證精英個體"""
        if not elites:
            return []

        print(f"\n🔍 驗證歷史前 {top_n} 名精英...")

        sorted_elites = sorted(elites, key=lambda x: x.get('fitness', 0), reverse=True)[:top_n]
        validated = []

        for i, elite in enumerate(sorted_elites):
            genes = elite.get('genes', [])
            if not genes:
                continue

            print(f"   驗證 #{i+1}...")
            fitness, details = BacktestEngine.evaluate_fitness(genes)

            if fitness > 0:
                elite['fitness'] = fitness
                elite['validated'] = True
                validated.append(elite)
                print(f"      ✅ 夏普: {details.get('sharpe', 0):.2f}")
            else:
                print(f"      ❌ 驗證失敗")

        return validated


# 初始化
CHECKPOINT = CheckpointManager()

# =============================================================================
# 第十部分：遺傳演算法核心
# =============================================================================
class GeneticAlgorithm:
    """遺傳演算法核心"""

    def create_individual(self) -> List[float]:
        """創建隨機個體"""
        return [random.random() for _ in range(GENE_LENGTH)]

    def create_population(self, size: int) -> List[List[float]]:
        """創建族群"""
        return [self.create_individual() for _ in range(size)]

    def tournament_select(self, population: List[List[float]],
                          fitness: List[float], k: int = 5) -> List[float]:
        """錦標賽選擇"""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return copy.deepcopy(population[best_idx])

    def crossover(self, p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """模擬二元交叉 (SBX)"""
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        eta = 20.0

        for i in range(len(p1)):
            if random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-10:
                y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])

                rand = random.random()
                beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-10)
                alpha = 2.0 - beta ** -(eta + 1.0)

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                c1[i] = max(0, min(1, 0.5 * ((y1 + y2) - betaq * (y2 - y1))))
                c2[i] = max(0, min(1, 0.5 * ((y1 + y2) + betaq * (y2 - y1))))

        return c1, c2

    def mutate(self, ind: List[float], indpb: float = 0.1) -> List[float]:
        """多項式變異"""
        mutant = copy.deepcopy(ind)
        eta = 20.0

        for i in range(len(mutant)):
            if random.random() < indpb:
                y = mutant[i]
                delta1, delta2 = y, 1.0 - y

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

                mutant[i] = max(0, min(1, y + deltaq))

        return mutant


GA = GeneticAlgorithm()

# =============================================================================
# 第十一部分：演化引擎
# =============================================================================
class EvolutionEngine:
    """演化引擎"""

    def __init__(self):
        self.state: Optional[EvolutionState] = None
        self.start_time = time.time()

    def initialize(self) -> EvolutionState:
        """初始化或恢復狀態"""
        # 嘗試繼續
        if CONFIG.continue_evolution:
            state = CHECKPOINT.load()
            if state:
                state.elite_archive = CHECKPOINT.validate_elites(
                    state.elite_archive, CONFIG.validate_top_n)
                self.state = state
                return state

        print("\n🆕 初始化新的演化...")

        # 載入歷史精英
        elites = CHECKPOINT.load_pareto()
        if elites:
            elites = CHECKPOINT.validate_elites(elites, CONFIG.validate_top_n)

        # 創建族群
        population = GA.create_population(CONFIG.population_size)

        # 注入精英
        if elites:
            n_inject = min(len(elites), CONFIG.population_size // 4)
            for i, elite in enumerate(elites[:n_inject]):
                if i < len(population):
                    population[i] = elite.get('genes', population[i])
            print(f"   ✅ 注入 {n_inject} 個歷史精英")

        state = EvolutionState(
            population=population,
            fitness_scores=[0.0] * len(population),
            elite_archive=elites,
        )

        self.state = state
        return state

    def evaluate_population(self) -> List[Tuple[float, Dict]]:
        """評估族群"""
        population = self.state.population
        print(f"   📊 評估 {len(population)} 個個體...")

        results = []

        if CONFIG.use_parallel and CONFIG.n_workers > 1:
            try:
                with ThreadPoolExecutor(max_workers=CONFIG.n_workers) as executor:
                    futures = {executor.submit(BacktestEngine.evaluate_fitness, ind): i
                               for i, ind in enumerate(population)}

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result(timeout=300)
                            results.append((idx, result))
                        except:
                            results.append((idx, (0.0, {'error': 'timeout'})))

                results.sort(key=lambda x: x[0])
                results = [r[1] for r in results]

            except Exception as e:
                print(f"   ⚠️  並行失敗，改用序列: {e}")
                results = [BacktestEngine.evaluate_fitness(ind) for ind in population]
        else:
            for i, ind in enumerate(population):
                result = BacktestEngine.evaluate_fitness(ind)
                results.append(result)
                if (i + 1) % 5 == 0:
                    print(f"      進度: {i+1}/{len(population)}")

        return results

    def update_elites(self, details_list: List[Dict]):
        """更新精英存檔"""
        new_elites = []
        for genes, (fitness, details) in zip(self.state.population,
                                              zip(self.state.fitness_scores, details_list)):
            if fitness > 0:
                new_elites.append({
                    'genes': genes,
                    'fitness': fitness,
                    'details': details,
                    'generation': self.state.generation,
                })

        # 合併去重
        all_elites = self.state.elite_archive + new_elites
        seen = set()
        unique = []

        for e in all_elites:
            h = hashlib.md5(str(e['genes'][:10]).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(e)

        unique.sort(key=lambda x: x['fitness'], reverse=True)
        self.state.elite_archive = unique[:CONFIG.elite_size]

    def create_next_generation(self):
        """產生下一代"""
        population = self.state.population
        fitness = self.state.fitness_scores
        new_pop = []

        # 精英保留
        for elite in self.state.elite_archive[:CONFIG.elite_size // 2]:
            new_pop.append(copy.deepcopy(elite['genes']))

        # 產生子代
        while len(new_pop) < CONFIG.population_size:
            p1 = GA.tournament_select(population, fitness, CONFIG.tournament_size)
            p2 = GA.tournament_select(population, fitness, CONFIG.tournament_size)

            if random.random() < CONFIG.crossover_rate:
                c1, c2 = GA.crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            if random.random() < CONFIG.mutation_rate:
                c1 = GA.mutate(c1)
            if random.random() < CONFIG.mutation_rate:
                c2 = GA.mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < CONFIG.population_size:
                new_pop.append(c2)

        self.state.population = new_pop[:CONFIG.population_size]

    def run_generation(self) -> Dict:
        """執行單代"""
        gen = self.state.generation

        # 評估
        results = self.evaluate_population()
        fitness_scores = [r[0] for r in results]
        details_list = [r[1] for r in results]

        self.state.fitness_scores = fitness_scores

        # 更新精英
        self.update_elites(details_list)

        # 找最佳
        best_idx = fitness_scores.index(max(fitness_scores))
        best_fitness = fitness_scores[best_idx]
        best_details = details_list[best_idx]

        if best_fitness > self.state.best_fitness:
            self.state.best_fitness = best_fitness
            self.state.best_genes = copy.deepcopy(self.state.population[best_idx])
            self.state.best_params = GeneDecoder.decode(self.state.best_genes)

        # 統計
        stats = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(fitness_scores),
            'best_sharpe': best_details.get('sharpe', 0),
            'best_capacity': best_details.get('capacity', 0),
            'best_mdd': best_details.get('max_drawdown', 0),
            'best_return': best_details.get('annual_return', 0),
            'elite_count': len(self.state.elite_archive),
        }

        self.state.history.append(stats)

        # 產生下一代
        if gen < CONFIG.n_generations - 1:
            self.create_next_generation()

        self.state.generation += 1

        return stats

    def print_summary(self, stats: Dict):
        """輸出摘要"""
        gen = stats['generation']

        if gen % CONFIG.visualization_interval == 0 or gen == 0:
            elapsed = (time.time() - self.start_time) / 60

            # 策略權重
            weights_str = ""
            if self.state.best_params:
                weights = [self.state.best_params.get(f'w{i+1}', 0) * 100 for i in range(9)]
                weights_str = " ".join([f"S{i+1}:{w:.0f}%" for i, w in enumerate(weights) if w > 5])

            print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🐉 第 {gen+1:3d}/{CONFIG.n_generations} 代  |  ⏱️  {elapsed:.1f}分  |  👥 精英: {stats['elite_count']}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📈 最佳適應度: {stats['best_fitness']:7.2f}  |  平均: {stats['avg_fitness']:7.2f}                            ║
║  📊 夏普: {stats['best_sharpe']:5.2f}  |  胃納量: {stats['best_capacity']/1e4:6.0f}萬  |  回檔: {stats['best_mdd']*100:4.1f}%             ║
║  📋 {weights_str:<70} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def print_detailed_report(self, gen: int):
        """詳細報告（每 10 代）"""
        if (gen + 1) % CONFIG.report_interval != 0:
            return

        print(f"\n{'='*80}")
        print(f"📋 第 {gen+1} 代詳細報告")
        print(f"{'='*80}")

        print(f"\n🏆 精英排名:")
        print(f"{'排名':<6}{'適應度':<10}{'夏普':<8}{'胃納量(萬)':<12}{'回檔%':<10}{'達標'}")
        print("-" * 60)

        for i, elite in enumerate(self.state.elite_archive[:10], 1):
            d = elite.get('details', {})
            sharpe = d.get('sharpe', 0)
            cap = d.get('capacity', 0) / 1e4
            mdd = d.get('max_drawdown', 0) * 100
            target = "✅" if sharpe >= CONFIG.target_sharpe and cap >= CONFIG.min_capacity/1e4 else ""
            print(f"{i:<6}{elite['fitness']:<10.2f}{sharpe:<8.2f}{cap:<12.0f}{mdd:<10.1f}{target}")

        # 最佳策略權重
        if self.state.best_params:
            print(f"\n📊 最佳策略權重配置:")
            names = ['低波動價值', '小型成長', '營收雙渦輪', '高殖利率',
                    '低波動穩健', '創高突破', '財務品質', '技術動能', '綜合品質']
            for i, name in enumerate(names):
                w = self.state.best_params.get(f'w{i+1}', 0) * 100
                if w > 3:
                    print(f"   S{i+1} {name}: {w:.1f}%")

    def run(self) -> EvolutionState:
        """執行完整演化"""
        self.initialize()

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 開始九組合天空龍 GA 優化                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 目標: 夏普 > {CONFIG.target_sharpe}, MDD < {CONFIG.max_drawdown*100:.0f}%, 胃納量 > {CONFIG.min_capacity/1e6:.0f}M                        ║
║  🧬 基因: {GENE_LENGTH}, 族群: {CONFIG.population_size}, 世代: {CONFIG.n_generations}                                       ║
║  📅 樣本內: {CONFIG.train_start}~{CONFIG.train_end}  樣本外: {CONFIG.test_start}~                   ║
║  📊 策略: 9 個獨立評分策略動態加權組合                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        start_gen = self.state.generation

        for gen in range(start_gen, CONFIG.n_generations):
            stats = self.run_generation()
            self.print_summary(stats)
            self.print_detailed_report(gen)

            # 存檔
            if (gen + 1) % CONFIG.checkpoint_interval == 0:
                CHECKPOINT.save(self.state)

            # 達標檢查
            if stats['best_sharpe'] >= CONFIG.target_sharpe:
                d = self.state.elite_archive[0].get('details', {})
                if (d.get('capacity', 0) >= CONFIG.min_capacity and
                    d.get('max_drawdown', 1) <= CONFIG.max_drawdown):
                    print(f"\n🎉 達成目標！於第 {gen+1} 代")
                    break

        # 最終存檔
        CHECKPOINT.save(self.state)
        self._print_final()

        return self.state

    def _print_final(self):
        """輸出最終結果"""
        elapsed = (time.time() - self.start_time) / 60

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🏁 優化完成                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ⏱️  總耗時: {elapsed:.1f} 分鐘                                                       ║
║  🧬 總世代: {self.state.generation}                                                           ║
║  🏆 最佳適應度: {self.state.best_fitness:.2f}                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        # 儲存最佳參數
        if self.state.best_params:
            with open(PATHS.best_params_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.best_params, f, indent=2, ensure_ascii=False)
            print(f"✅ 最佳參數: {PATHS.best_params_file}")

        # 輸出排名
        print(f"\n🏆 最終精英排名:")
        for i, e in enumerate(self.state.elite_archive[:5], 1):
            d = e.get('details', {})
            print(f"   #{i}: 夏普={d.get('sharpe', 0):.2f}, "
                  f"胃納={d.get('capacity', 0)/1e4:.0f}萬, "
                  f"回檔={d.get('max_drawdown', 0)*100:.1f}%")

# =============================================================================
# 第十二部分：主程式
# =============================================================================
def main():
    """主程式入口"""
    print(f"""

    ██████╗ ██████╗  █████╗  ██████╗  ██████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗████╗  ██║
    ██║  ██║██████╔╝███████║██║  ███╗██║   ██║██╔██╗ ██║
    ██║  ██║██╔══██╗██╔══██║██║   ██║██║   ██║██║╚██╗██║
    ██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║
    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝

    🐉 九組合天空龍 GA v5.0 - 9策略 x 100基因
    ================================================

""")

    try:
        engine = EvolutionEngine()
        final_state = engine.run()

        # 最佳策略完整回測
        if final_state.best_params:
            print("\n" + "="*80)
            print("📊 最佳策略完整回測")
            print("="*80)

            position = COMBINER.combine(final_state.best_params)

            if position is not None and not position.empty:
                report = sim_func(
                    position=position,
                    fee_ratio=1.425 / 1000 * 0.5,
                    tax_ratio=3 / 1000,
                    trade_at_price='high_low_avg',
                    position_limit=final_state.best_params.get('position_limit', 0.25),
                    stop_loss=final_state.best_params.get('stop_loss', 0.20),
                    trail_stop=final_state.best_params.get('trail_stop', 0.25),
                    take_profit=final_state.best_params.get('take_profit', 0.50),
                    upload=False,
                    name='DragonGA_v5_Best',
                )
                report.display()

        print("\n✅ 優化完成！")
        print(f"📁 輸出: {PATHS.base_dir}")

        return final_state

    except KeyboardInterrupt:
        print("\n\n⚠️  中斷，保存中...")
        if 'engine' in locals() and engine.state:
            CHECKPOINT.save(engine.state)
        print("✅ 已保存，可用 CONTINUE_EVOLUTION=true 繼續")
        return None

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
