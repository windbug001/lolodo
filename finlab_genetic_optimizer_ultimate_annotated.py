#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🧬 FinLab 台股基因演算法優化系統 - 終極強化版 v2.0 (帶詳細註解)
================================================================================

這個程式使用「遺傳演算法 (Genetic Algorithm)」來自動優化台股選股策略的參數。

【什麼是遺傳演算法？】
遺傳演算法模仿自然界的「物競天擇、適者生存」原理：
1. 創建一群「個體」（每個個體代表一組策略參數）
2. 評估每個個體的「適應度」（用回測績效來衡量）
3. 讓優秀的個體「交配」產生後代（參數混合）
4. 對後代進行「變異」（隨機微調參數）
5. 淘汰表現差的個體，保留優秀的
6. 重複以上步驟，直到找到最佳參數

【程式架構】
┌─────────────────────────────────────────────────────────────┐
│  main()  主程式入口                                          │
│    │                                                         │
│    ├── FinLabDataLoader    載入 FinLab 股票數據              │
│    │                                                         │
│    ├── GeneDecoder         將基因解碼為策略參數              │
│    │                                                         │
│    ├── StrategyEngine      執行三個子策略並合併              │
│    │   ├── 策略一：低波動本益比                              │
│    │   ├── 策略二：小資族策略                                │
│    │   └── 策略三：營收股價雙渦輪                            │
│    │                                                         │
│    ├── BacktestEngine      執行回測（樣本內/外分離）         │
│    │                                                         │
│    ├── EvolutionEngine     NSGA-II 演化引擎                  │
│    │   ├── 選擇 (Selection)                                  │
│    │   ├── 交叉 (Crossover)                                  │
│    │   └── 變異 (Mutation)                                   │
│    │                                                         │
│    ├── ParetoArchiveManager 保存歷史最佳個體                 │
│    │                                                         │
│    └── CheckpointManager    支援中斷後重啟                   │
└─────────────────────────────────────────────────────────────┘
================================================================================
"""

from __future__ import annotations  # 允許在類型提示中使用尚未定義的類別名稱

# ============================================================================
# 🔥【最重要】在任何 import 之前禁用 FinLab 快取
#
# 為什麼要這樣做？
# FinLab 會將數據快取到本地磁碟，有時候快取檔案會損壞（尤其是 Colab 環境）
# 這會導致程式啟動時出現 EOFError 或其他錯誤
# 設定 FINLAB_DISABLE_CACHE = '1' 可以強制 FinLab 每次都從伺服器載入最新數據
# ============================================================================
import os
os.environ['FINLAB_DISABLE_CACHE'] = '1'  # 禁用 FinLab 的本地快取功能

import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告訊息（讓輸出更乾淨）
warnings.filterwarnings("ignore", category=FutureWarning)  # 特別忽略 FutureWarning

# =============================================================================
# 第一部分：核心設定 - 匯入必要的 Python 套件
# =============================================================================
import sys              # 系統相關功能
import json             # JSON 格式處理（用於保存設定檔）
import pickle           # Python 物件序列化（用於保存檢查點）
import time             # 時間相關功能（計時用）
import hashlib          # 雜湊函數（用於去重複）
import random           # 隨機數生成（遺傳演算法需要隨機性）
import multiprocessing as mp  # 多進程並行處理
from datetime import datetime, timedelta  # 日期時間處理
from pathlib import Path  # 路徑處理（比 os.path 更現代化）
from typing import Dict, List, Tuple, Optional, Any, Callable  # 類型提示
from dataclasses import dataclass, field  # 資料類別（簡化類別定義）
from functools import partial  # 函數部分應用
from concurrent.futures import ProcessPoolExecutor, as_completed  # 並行執行器
import traceback        # 錯誤追蹤（用於除錯）

import numpy as np      # 數值計算庫
import pandas as pd     # 資料分析庫（處理股票數據的核心）

pd.set_option('display.max_columns', None)  # 顯示所有欄位（不省略）
pd.set_option('future.no_silent_downcasting', True)  # 避免 pandas 警告

# ============================================================================
# 🔥 核心設定區（請根據需求修改）
#
# 這些是程式的「超參數」，控制整個優化過程的行為
# ============================================================================

# ----- 視窗 ID -----
# 用於多視窗並行優化，每個視窗有獨立的輸出目錄
# 可以同時開 4 個 Colab 視窗，分別設定 WINDOW_ID = 1, 2, 3, 4
# 這樣可以同時探索不同的參數空間，加速找到最佳解
WINDOW_ID = int(os.environ.get('WINDOW_ID', 1))  # 從環境變數讀取，預設為 1

# ----- FinLab API Key -----
# 這是您的 FinLab VIP 會員金鑰，用於存取 FinLab 的股票數據 API
FINLAB_API_KEY = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

# ----- 績效目標 -----
# 這些是我們希望策略達成的目標
TARGET_SHARPE = 4.2           # 目標夏普值（風險調整後報酬，越高越好，4.2 是非常優秀的水準）
MIN_CAPACITY = 10_000_000     # 最小資金胃納量（1000萬台幣，表示策略可以承載多少資金）
TARGET_ANNUAL_RETURN = 0.30   # 目標年化報酬率（30%）
MAX_DRAWDOWN = 0.20           # 最大回檔限制（20%，表示最大虧損不超過 20%）

# ----- 持股配比要求 -----
# 這確保每檔股票的持股比例不會太小（太小的話交易成本會吃掉獲利）
MIN_POSITION_WEIGHT = 0.03    # 最小持股比例 3%（低於這個比例就不買）
POSITION_WEIGHT_STEP = 0.03   # 持股步長 3%（持股比例必須是 3% 的倍數：3%, 6%, 9%...）

# ----- GA 演化參數 -----
# 這些參數控制遺傳演算法的行為
POPULATION_SIZE = 50          # 族群大小（每一代有多少個「個體」，也就是多少組參數要測試）
N_GENERATIONS = 100           # 總世代數（演化要進行多少代）
MUTATION_RATE = 0.20          # 變異率（20% 的機率會對基因進行隨機變異）
CROSSOVER_RATE = 0.80         # 交叉率（80% 的機率會讓兩個個體交配產生後代）
ELITE_RATIO = 0.30            # 精英注入比例（30% 的新族群會從歷史最佳個體注入）

# ----- 並行設定 -----
# 使用多核心 CPU 加速評估
N_WORKERS = max(1, mp.cpu_count() - 1)  # 使用 CPU 核心數 - 1 個工作進程

# ----- 回測時間設定（樣本內/外分離）-----
# 樣本內：用於訓練（找最佳參數）
# 樣本外：用於驗證（確認參數不是過度擬合）
IN_SAMPLE_START = '2017-01-01'    # 樣本內開始日期
IN_SAMPLE_END = '2022-12-31'       # 樣本內結束日期
OUT_OF_SAMPLE_START = '2023-01-01' # 樣本外開始日期
OUT_OF_SAMPLE_END = None           # 樣本外結束日期（None 表示至今）

# ----- Walk-Forward 設定 -----
# Walk-Forward 是一種交叉驗證方法，用於檢驗策略的穩定性
WALK_FORWARD_WINDOWS = 3  # 分割成 3 個時間窗口
TRAIN_MONTHS = 24         # 每個窗口的訓練期 24 個月
TEST_MONTHS = 6           # 每個窗口的測試期 6 個月

# ----- 報告頻率 -----
DETAILED_REPORT_INTERVAL = 10  # 每 10 代生成詳細報告
PROGRESS_LOG_INTERVAL = 1      # 每代記錄進度
CHECKPOINT_INTERVAL = 10       # 每 10 代保存檢查點（可中斷重啟）
TOP_N_VALIDATION = 5           # 重啟時驗證前 5 名

# 顯示系統資訊
print(f"{'=' * 80}")
print(f"🚀 FinLab 台股基因演算法優化系統 v2.0 - 視窗 {WINDOW_ID}")
print(f"   🎯 目標：夏普 >= {TARGET_SHARPE}, 胃納量 >= {MIN_CAPACITY/1e7:.0f}00萬, MDD < {MAX_DRAWDOWN*100:.0f}%")
print(f"   📊 樣本內: {IN_SAMPLE_START} ~ {IN_SAMPLE_END}")
print(f"   📊 樣本外: {OUT_OF_SAMPLE_START} ~ {'至今' if OUT_OF_SAMPLE_END is None else OUT_OF_SAMPLE_END}")
print(f"   🖥️  並行工作數: {N_WORKERS}")
print(f"{'=' * 80}")


# =============================================================================
# 第二部分：快取清理與套件安裝
# =============================================================================
def clear_finlab_cache():
    """
    清除可能損壞的 FinLab 快取

    為什麼需要這個函數？
    FinLab 會將從伺服器下載的數據快取到本地 .pkl 檔案
    如果 Colab 執行階段被中斷，這些快取檔案可能會損壞
    損壞的快取會導致 EOFError 或資料讀取錯誤

    這個函數會在程式啟動時清除所有可能損壞的快取
    """
    import glob       # 用於檔案路徑匹配（支援萬用字元 *）
    import shutil     # 用於刪除目錄
    import subprocess # 用於執行系統命令

    print("🔧 清除 FinLab 快取...")

    # 方法 1: 使用 find 命令刪除所有 .pkl 檔案
    # .pkl 是 Python pickle 格式，用於序列化物件
    try:
        subprocess.run(
            ['find', '/root', '/tmp', '-name', '*.pkl', '-type', 'f', '-delete'],
            stderr=subprocess.DEVNULL,  # 忽略錯誤輸出
            timeout=10                   # 最多等待 10 秒
        )
    except Exception:
        pass  # 如果失敗就跳過

    # 方法 2: 清除 finlab 相關目錄
    # 這些是 FinLab 可能使用的快取目錄
    cache_patterns = [
        '/root/.finlab*',              # Colab 環境的 FinLab 快取
        '/tmp/.finlab*',               # 暫存目錄的 FinLab 快取
        '/tmp/finlab*',
        os.path.expanduser('~/.finlab*'),  # 使用者目錄的 FinLab 快取
        '/content/.finlab*',           # Colab 工作目錄
        '/root/*finlab*',
        '/tmp/*finlab*',
    ]

    cleared = 0  # 計數器：記錄清除了多少檔案/目錄
    for pattern in cache_patterns:
        try:
            matches = glob.glob(pattern)  # 找出所有符合模式的檔案/目錄
            for path in matches:
                if os.path.isdir(path):
                    shutil.rmtree(path)   # 刪除整個目錄
                else:
                    os.remove(path)       # 刪除單一檔案
                cleared += 1
        except Exception:
            pass

    # 方法 3: 清除 pandas 快取
    pandas_cache = os.path.expanduser('~/.cache/pandas')
    if os.path.exists(pandas_cache):
        try:
            shutil.rmtree(pandas_cache)
            cleared += 1
        except Exception:
            pass

    if cleared > 0:
        print(f"   ✅ 已清除 {cleared} 個快取檔案/目錄")
    else:
        print("   ℹ️  無快取需要清除")


# 🔥 在載入 FinLab 之前先清除快取（這很重要！）
clear_finlab_cache()


def install_packages():
    """
    安裝必要套件

    這個函數會檢查必要的 Python 套件是否已安裝
    如果沒有安裝，就自動安裝

    必要套件：
    - finlab: FinLab 量化交易框架，提供股票數據和回測功能
    - deap: 遺傳演算法框架，提供演化計算的工具
    - tqdm: 進度條顯示，讓使用者知道程式執行進度
    """
    required = {
        'finlab': 'finlab',  # 套件名稱: 匯入名稱
        'deap': 'deap',
        'tqdm': 'tqdm',
    }

    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)  # 嘗試匯入套件
        except ImportError:
            # 如果匯入失敗，表示套件未安裝
            print(f"   安裝 {pkg_name}...")
            os.system(f'pip install {pkg_name} -q')  # 使用 pip 安裝（-q 表示安靜模式）


install_packages()

# 載入 FinLab 相關模組
import finlab                    # FinLab 主模組
from finlab import data          # 數據存取模組
from finlab.backtest import sim  # 回測模擬函數

# 載入 DEAP（Distributed Evolutionary Algorithms in Python）
from deap import base, creator, tools, algorithms

# 載入 tqdm（進度條）
from tqdm import tqdm

print(f"✅ 套件載入完成")
print(f"   - FinLab: {finlab.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Pandas: {pd.__version__}")


# =============================================================================
# 第三部分：環境設定與登入
# =============================================================================
def setup_environment():
    """
    設定執行環境

    這個函數做兩件事：
    1. 掛載 Google Drive（如果在 Colab 環境）
       - Google Drive 用於保存結果和檢查點
       - 這樣即使 Colab 斷線，結果也不會遺失

    2. 登入 FinLab
       - 使用 API Key 進行身份驗證
       - 登入後才能存取 FinLab 的股票數據
    """
    # 嘗試掛載 Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        # 設定基礎目錄為 Google Drive 中的資料夾
        base_dir = '/content/drive/MyDrive/FinLab_GA_優化_終極版'
        in_colab = True
        print("✅ Google Drive 已掛載")
    except Exception:
        # 如果不在 Colab 環境（例如本地執行），使用當前目錄
        base_dir = './finlab_ga_output'
        in_colab = False
        print("⚠️ 本地環境模式")

    # 登入 FinLab
    finlab.login(FINLAB_API_KEY)
    print("✅ FinLab VIP 登入成功")

    return base_dir, in_colab


BASE_DIR, IN_COLAB = setup_environment()


@dataclass
class PathManager:
    """
    路徑管理器

    這個類別負責管理所有輸出檔案的路徑
    使用 @dataclass 裝飾器可以自動生成 __init__ 方法

    目錄結構：
    BASE_DIR/
    ├── window_1/              # 視窗 1 的輸出目錄
    │   ├── output/            # 最佳參數等輸出
    │   ├── history/           # 演化歷史
    │   ├── reports/           # 詳細報告
    │   └── checkpoint.pkl     # 檢查點（可中斷重啟）
    ├── window_2/              # 視窗 2 的輸出目錄
    │   └── ...
    ├── shared_pareto/         # 所有視窗共享的 Pareto 前緣
    │   ├── pareto_archive_w1.pkl
    │   ├── pareto_archive_w2.pkl
    │   └── global_pareto_archive.pkl
    └── window_1_logs/         # 日誌（供監控系統讀取）
        └── progress_history.json
    """
    base_dir: str = BASE_DIR     # 基礎目錄
    window_id: int = WINDOW_ID   # 視窗 ID

    def __post_init__(self):
        """dataclass 初始化完成後自動呼叫"""
        # 設定各種目錄路徑
        self.window_dir = f"{self.base_dir}/window_{self.window_id}"
        self.output_dir = f"{self.window_dir}/output"
        self.pareto_dir = f"{self.base_dir}/shared_pareto"   # Pareto Archive 共享
        self.history_dir = f"{self.window_dir}/history"
        self.log_dir = f"{self.window_dir}_logs"             # 日誌目錄
        self.reports_dir = f"{self.window_dir}/reports"      # 詳細報告

        # 建立所有必要的目錄
        for d in [self.window_dir, self.output_dir, self.pareto_dir,
                  self.history_dir, self.log_dir, self.reports_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def pareto_archive(self) -> str:
        """本視窗的 Pareto 存檔路徑"""
        return f"{self.pareto_dir}/pareto_archive_w{self.window_id}.pkl"

    @property
    def global_pareto_archive(self) -> str:
        """全局 Pareto 存檔（所有視窗共享）"""
        return f"{self.pareto_dir}/global_pareto_archive.pkl"

    @property
    def checkpoint_file(self) -> str:
        """檢查點檔案路徑"""
        return f"{self.window_dir}/checkpoint.pkl"

    @property
    def best_params_file(self) -> str:
        """最佳參數檔案路徑"""
        return f"{self.output_dir}/best_params_w{self.window_id}.json"

    @property
    def progress_log(self) -> str:
        """進度日誌路徑"""
        return f"{self.log_dir}/progress_history.json"

    @property
    def validation_log(self) -> str:
        """驗證結果日誌路徑"""
        return f"{self.log_dir}/validation_results.json"


paths = PathManager()
print(f"📁 工作目錄: {paths.window_dir}")


# =============================================================================
# 第四部分：數據載入（使用 FinLab API）
# =============================================================================
class FinLabDataLoader:
    """
    FinLab 數據載入器

    這是一個 Singleton 模式的類別，確保整個程式只有一個數據載入器實例
    這樣可以避免重複載入數據，節省時間和記憶體

    主要功能：
    1. 從 FinLab API 載入各種股票數據
    2. 計算衍生指標（如營收均線、波動率等）
    3. 將數據快取在記憶體中，供策略引擎使用

    載入的數據類型：
    - 價格數據：開盤、收盤、最高、最低、成交量
    - 估值數據：本益比、股價淨值比、殖利率
    - 營收數據：當月營收、年增率、月增率
    - 基本面：營業利益成長率、毛利率、ROE 等
    - 籌碼：融資使用率、董監持股比例
    - 技術指標：RSI、ATR
    """

    _instance = None   # Singleton 實例
    _cache = {}        # 數據快取
    _loaded = False    # 是否已載入

    def __new__(cls):
        """
        實作 Singleton 模式
        確保無論呼叫幾次，都只會創建一個實例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化：只在第一次創建實例時載入數據"""
        if not self._loaded:
            self._load_all_data()
            FinLabDataLoader._loaded = True

    def _safe_get(self, key: str, retry: bool = True):
        """
        安全地載入數據

        如果遇到 EOFError（快取損壞），會自動清除快取並重試

        Parameters:
            key: FinLab 數據的 key，例如 'price:收盤價'
            retry: 是否在失敗時重試

        Returns:
            pandas DataFrame 格式的數據
        """
        try:
            return data.get(key)
        except (EOFError, Exception) as e:
            if 'EOF' in str(e) and retry:
                print(f"   ⚠️ 快取損壞，清除後重試: {key}")
                import subprocess
                subprocess.run(['find', '/root', '/tmp', '-name', '*.pkl', '-delete'],
                               stderr=subprocess.DEVNULL)
                return self._safe_get(key, retry=False)
            else:
                raise

    def _load_all_data(self):
        """
        載入所有需要的數據

        這是數據載入的核心函數
        會從 FinLab API 載入各種股票數據，並計算衍生指標
        """
        print("📊 載入 FinLab 數據...")
        start_time = time.time()

        # ===== 價格相關數據 =====
        # 這些是最基本的股價數據，每日更新
        self._cache['close'] = self._safe_get('price:收盤價')      # 收盤價
        self._cache['vol'] = self._safe_get('price:成交股數')      # 成交量
        self._cache['open'] = self._safe_get('price:開盤價')       # 開盤價
        self._cache['high'] = self._safe_get('price:最高價')       # 最高價
        self._cache['low'] = self._safe_get('price:最低價')        # 最低價
        self._cache['adj_close'] = self._safe_get("etl:adj_close") # 調整後收盤價（已考慮除權息）

        # ===== 估值數據 =====
        # 用於判斷股票是否便宜或昂貴
        self._cache['pe'] = self._safe_get('price_earning_ratio:本益比')        # 本益比 = 股價 / 每股盈餘
        self._cache['pb'] = self._safe_get("price_earning_ratio:股價淨值比")     # 股價淨值比 = 股價 / 每股淨值
        self._cache['dividend_yield'] = self._safe_get('price_earning_ratio:殖利率(%)')  # 殖利率

        # ===== 營收數據 =====
        # 營收是判斷公司成長性的重要指標
        self._cache['rev'] = self._safe_get('monthly_revenue:當月營收')           # 當月營收
        self._cache['rev_yoy_growth'] = self._safe_get('monthly_revenue:去年同月增減(%)')  # 年增率
        self._cache['rev_month_growth'] = self._safe_get('monthly_revenue:上月比較增減(%)')  # 月增率

        # ===== 基本面指標 =====
        # 這些指標反映公司的獲利能力和財務狀況
        self._cache['營業利益成長率'] = self._safe_get('fundamental_features:營業利益成長率')
        self._cache['業外收支營收率'] = self._safe_get('fundamental_features:業外收支營收率')
        self._cache['營業毛利率'] = self._safe_get("fundamental_features:營業毛利率")
        self._cache['ROE綜合損益'] = self._safe_get("fundamental_features:ROE綜合損益")
        self._cache['稅後淨利率'] = self._safe_get("fundamental_features:稅後淨利率")
        self._cache['稅前淨利率'] = self._safe_get("fundamental_features:稅前淨利率")

        # ===== 籌碼資料 =====
        # 籌碼反映市場參與者的行為
        self._cache['融資使用率'] = self._safe_get('margin_transactions:融資使用率')
        self._cache['董監持有股數占比'] = self._safe_get("internal_equity_changes:董監持有股數占比")
        self._cache['inventory'] = self._safe_get("inventory")  # 集保庫存（股東結構）

        # ===== 市值資料 =====
        self._cache['市值'] = self._safe_get('etl:market_value')

        # ===== 財務報表 =====
        self._cache['股本'] = self._safe_get('financial_statement:股本')
        self._cache['投資活動現金流'] = self._safe_get('financial_statement:投資活動之淨現金流入_流出')
        self._cache['營業活動現金流'] = self._safe_get('financial_statement:營業活動之淨現金流入_流出')
        self._cache['稅後淨利'] = self._safe_get('fundamental_features:經常稅後淨利')
        self._cache['權益總計'] = self._safe_get('financial_statement:股東權益總額')

        # ===== 技術指標 =====
        # RSI: 相對強弱指標，用於判斷超買超賣
        self._cache['rsi'] = data.indicator("RSI", adjust_price=False, resample="D", timeperiod=5)
        # ATR: 平均真實範圍，用於衡量波動性
        self._cache['atr'] = data.indicator('ATR', adjust_price=True, timeperiod=10)

        # 計算衍生指標
        self._calculate_derived_indicators()

        elapsed = time.time() - start_time
        print(f"✅ 數據載入完成 ({elapsed:.1f}s)")
        print(f"   股票數: {len(self._cache['close'].columns)}")
        print(f"   期間: {self._cache['close'].index[0]} ~ {self._cache['close'].index[-1]}")

    def _calculate_derived_indicators(self):
        """
        計算衍生指標

        這些指標是從原始數據計算出來的
        用於策略的選股條件判斷
        """
        close = self._cache['close']
        high = self._cache['high']
        low = self._cache['low']
        open_ = self._cache['open']
        adj_close = self._cache['adj_close']
        atr = self._cache['atr']
        rev = self._cache['rev']

        # 營收均線：用於判斷營收趨勢
        # rev_ma3: 近 3 個月營收平均
        # rev_ma12: 近 12 個月營收平均
        # 如果 rev_ma3 > rev_ma12，表示營收呈上升趨勢
        self._cache['rev_ma3'] = rev.average(3)
        self._cache['rev_ma12'] = rev.average(12)

        # 波動率：ATR / 收盤價
        # 用於衡量股票的波動程度
        # 波動率越低，表示股票越穩定
        self._cache['entry_volatility'] = atr / adj_close

        # 漲停鎖死判斷
        # 漲停鎖死的股票無法買入，需要排除
        # 判斷條件：收盤價 = 最高價 = 最低價 = 開盤價，且漲幅 > 9.5%
        limit_up = (close > close.shift(1) * 1.095)  # 漲幅超過 9.5%
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

        self._cache['limit_up_all_day'] = limit_up_all_day.fillna(False)

        # 自由現金流 = 營業活動現金流 + 投資活動現金流
        # 自由現金流為正表示公司能產生現金，是財務健康的指標
        df1 = self._cache['投資活動現金流']
        df2 = self._cache['營業活動現金流']
        self._cache['自由現金流'] = (df1 + df2).rolling(4).mean()  # 取最近 4 季平均

        # 股東權益報酬率 (ROE) = 稅後淨利 / 股東權益
        # ROE 越高表示公司越會賺錢
        self._cache['股東權益報酬率'] = self._cache['稅後淨利'] / self._cache['權益總計']

        # 當季營收（最近 4 個月營收加總）
        當月營收 = self._cache['rev'] * 1000
        self._cache['當季營收'] = 當月營收.rolling(4).sum()
        # 市值營收比 = 市值 / 當季營收
        # 市值營收比越低表示越便宜
        self._cache['市值營收比'] = self._cache['市值'] / self._cache['當季營收']

    def get(self, key: str):
        """
        獲取數據

        Parameters:
            key: 數據名稱

        Returns:
            對應的 DataFrame，如果 key 不存在則返回 None
        """
        return self._cache.get(key)


# 全局數據載入器（使用延遲初始化）
_data_loader = None


def get_data_loader() -> FinLabDataLoader:
    """
    獲取數據載入器（延遲初始化）

    第一次呼叫時會創建實例並載入數據
    之後的呼叫會返回同一個實例
    """
    global _data_loader
    if _data_loader is None:
        _data_loader = FinLabDataLoader()
    return _data_loader


# =============================================================================
# 第五部分：基因解碼器
# =============================================================================
class GeneDecoder:
    """
    基因解碼器

    【什麼是基因？】
    在遺傳演算法中，每個「個體」都有一組「基因」
    基因是一串數字（0.0 ~ 1.0 之間的浮點數）

    這個類別負責將這串數字「解碼」成實際的策略參數

    例如：
    - 基因 [0.3] -> 本益比下限 = 0.3 * 10 + 3 = 6
    - 基因 [0.7] -> 持股數量 = int(0.7 * 5 + 2) = 5

    【基因結構】（共 32 個基因）
    - 基因 0-8：策略一（低波動本益比）的 9 個參數
    - 基因 9-16：策略二（小資族）的 8 個參數
    - 基因 17-24：策略三（營收股價雙渦輪）的 8 個參數
    - 基因 25-27：三個策略的權重
    - 基因 28-31：回測參數（止損、停利等）
    """

    GENE_LENGTH = 32  # 基因長度

    def decode(self, genes: List[float]) -> Dict[str, Any]:
        """
        解碼基因為策略參數

        Parameters:
            genes: 基因列表，長度為 32，每個值在 0.0 ~ 1.0 之間

        Returns:
            策略參數字典
        """
        # 確保基因長度正確
        if len(genes) < self.GENE_LENGTH:
            genes = list(genes) + [0.5] * (self.GENE_LENGTH - len(genes))
        genes = genes[:self.GENE_LENGTH]

        params = {}

        # ===== 策略一：低波動本益比（9 個參數）=====
        # 這些參數控制策略一的選股條件

        # 營收趨勢：近 3 月營收 / 近 12 月營收 > 這個值
        # 範圍：1.0 ~ 1.5，值越大要求營收成長越快
        params['lv_rev_ma3_ma12_ratio'] = genes[0] * 0.5 + 1.0

        # 營收連續性：本月營收 / 上月營收 > 這個值
        # 範圍：0.5 ~ 0.9
        params['lv_rev_consistency'] = genes[1] * 0.4 + 0.5

        # 波動率門檻：只選波動率低於這個值的股票
        # 範圍：0.02 ~ 0.07（2% ~ 7%）
        params['lv_volatility_threshold'] = genes[2] * 0.05 + 0.02

        # 融資使用率上限：融資使用率不超過這個值
        # 範圍：20 ~ 50（%），融資使用率高表示散戶多
        params['lv_margin_usage_limit'] = genes[3] * 30 + 20

        # 業外收入上限：業外收入佔營收比例不超過這個值
        # 範圍：5 ~ 15（%），業外收入太高表示本業不穩定
        params['lv_non_op_income_limit'] = genes[4] * 10 + 5

        # 最小成交量：只選成交量大於這個值的股票
        # 範圍：10萬 ~ 30萬股
        params['lv_min_volume'] = genes[5] * 200000 + 100000

        # 本益比範圍
        params['lv_pe_min'] = genes[6] * 10 + 3     # 本益比下限：3 ~ 13
        params['lv_pe_max'] = genes[7] * 20 + 15    # 本益比上限：15 ~ 35

        # 選股數量
        params['lv_top_n'] = int(genes[8] * 5 + 2)  # 選出 2 ~ 7 檔

        # ===== 策略二：小資族（8 個參數）=====
        # 專注於中小型股的策略

        # 市值上限：只選市值低於這個值的股票
        params['si_market_value_limit'] = genes[9] * 10e9 + 10e9  # 100億 ~ 200億

        # 市值營收比上限
        params['si_market_rev_ratio_limit'] = genes[10] * 3 + 2   # 2 ~ 5

        # 營收年增率下限（用於排除衰退股）
        params['si_rev_yoy_growth_limit'] = genes[11] * 10 - 15   # -15% ~ -5%

        # 營收月增率下限
        params['si_rev_mom_growth_limit'] = genes[12] * 30 - 70   # -70% ~ -40%

        # RSV 計算週期
        params['si_rsv_period'] = int(genes[13] * 30 + 40)        # 40 ~ 70 天

        # 均線週期
        params['si_ma_period'] = int(genes[14] * 40 + 50)         # 50 ~ 90 天

        # 最小成交量
        params['si_volume_threshold'] = genes[15] * 200000 + 100000  # 10萬 ~ 30萬

        # 選股數量
        params['si_top_n'] = int(genes[16] * 6 + 4)               # 4 ~ 10 檔

        # ===== 策略三：營收股價雙渦輪（8 個參數）=====
        # 動能策略：尋找營收創新高、股價創新高的股票

        # 營收均線週期
        params['rpt_rev_ma_period'] = int(genes[17] * 5 + 3)      # 3 ~ 8 月

        # 營收創新高回看期間
        params['rpt_rev_ma_lookback'] = int(genes[18] * 15 + 15)  # 15 ~ 30 月

        # 股價創新高窗口
        params['rpt_price_high_window'] = int(genes[19] * 8 + 4)  # 4 ~ 12 天

        # 最小成交量
        params['rpt_min_volume'] = genes[20] * 300000 + 200000    # 20萬 ~ 50萬

        # 最低股價
        params['rpt_min_price'] = genes[21] * 10 + 10             # 10 ~ 20 元

        # RSI 門檻
        params['rpt_rsi_threshold'] = genes[22] * 20 + 50         # 50 ~ 70

        # 本益比上限
        params['rpt_pe_limit'] = genes[23] * 100 + 100            # 100 ~ 200

        # 選股數量
        params['rpt_top_n'] = int(genes[24] * 10 + 8)             # 8 ~ 18 檔

        # ===== 策略權重（3 個）=====
        # 控制三個策略的權重分配
        raw_weights = genes[25:28]
        total = sum(raw_weights) + 1e-10  # 加一個小數避免除以零
        params['weight_lv'] = raw_weights[0] / total    # 策略一權重
        params['weight_si'] = raw_weights[1] / total    # 策略二權重
        params['weight_rpt'] = raw_weights[2] / total   # 策略三權重

        # ===== 回測參數（4 個）=====
        params['stop_loss'] = genes[28] * 0.2 + 0.15       # 止損：15% ~ 35%
        params['trail_stop'] = genes[29] * 0.3 + 0.2       # 移動停損：20% ~ 50%
        params['take_profit'] = genes[30] * 0.5 + 0.5      # 停利：50% ~ 100%
        params['position_limit'] = genes[31] * 0.2 + 0.25  # 單檔持股上限：25% ~ 45%

        return params


# 創建全局基因解碼器實例
gene_decoder = GeneDecoder()


# =============================================================================
# 後續部分省略（策略引擎、回測引擎、演化引擎等）
# 完整註解版請參考原始碼
# =============================================================================

print("""
================================================================================
📚 程式碼註解說明完成！

【主要模組說明】

1. FinLabDataLoader（第四部分）
   - 從 FinLab API 載入股票數據
   - 使用 Singleton 模式避免重複載入
   - 自動計算衍生指標

2. GeneDecoder（第五部分）
   - 將 32 個基因解碼為策略參數
   - 每個基因值在 0.0 ~ 1.0 之間
   - 透過線性轉換映射到實際參數範圍

3. StrategyEngine（第五部分）
   - 策略一：低波動本益比 - 穩健型
   - 策略二：小資族策略 - 中小型股
   - 策略三：營收股價雙渦輪 - 動能型

4. BacktestEngine（第七部分）
   - 樣本內回測（2017-2022）
   - 樣本外回測（2023-至今）
   - Walk-Forward 驗證

5. EvolutionEngine（第十五部分）
   - NSGA-II 多目標優化
   - 選擇、交叉、變異操作
   - 檢查點保存與恢復

6. ParetoArchiveManager（第十部分）
   - 保存歷史最佳個體
   - 多視窗共享精英池
   - 持久化到 Google Drive

================================================================================
""")
