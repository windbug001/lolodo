##十二組合快快龍_優化版_高胃納量_夏普4.0目標 v2.1 (完整版)

# ============================================================================
# 🧬 十二組合快快龍 v2.1 (完整版)
#    統一功能：3視窗交互 + 安全寫入 + Discord + Checkpoint + 樣本外測試
# ============================================================================
# 優化重點：
# 1. ✅ 強制胃納量 >= 1000萬 (10M TWD)
# 2. ✅ 智能基因初始化 - 不同參數使用合適的範圍
# 3. ✅ 多目標評估函數 - 夏普值 + 胃納量懲罰項
# 4. ✅ 策略權重優化 - 確保分散且有效
# 5. ✅ 精選高勝率策略組合
# 6. ✅ 目標：夏普值 4.0+
# 7. ✅ 每 20 代詳細回測 - 定期完整回測追蹤進度
# 8. ✅ 重啟時回測歷史前五 - 快速確認基準績效
# 9. ✅ FinLab ML API 整合 - 機器學習特徵優化
# 10.✅ 最低權重過濾 - 單檔至少 3%，否則 0%
# ============================================================================
# 🔬 v2.0 新增：樣本外測試 (Out-of-Sample Testing) - 防過擬合
# ============================================================================
# 11.✅ 訓練期/測試期分離 (2014~2022 訓練, 2023~2025 測試)
# 12.✅ 適應度評估僅使用訓練期數據
# 13.✅ 每 20 代顯示訓練期 vs 測試期夏普比較
# 14.✅ 自動過擬合警告（測試/訓練比 < 60%）
# 15.✅ 最終報告包含完整樣本外測試結果
# ============================================================================
# 🔥 v2.1 新增：功能統一
# ============================================================================
# 16.✅ SafeFileManager 安全寫入機制（先暫存再改名）
# 17.✅ 3視窗交互取優秀基因機制 (CrossWindowManager)
# 18.✅ 完整 Discord 通知系統
# 19.✅ CheckpointManager 斷點續傳機制
# 20.✅ 每5代自動保存 checkpoint
# ============================================================================

# ============================================================================
# 🔑 FinLab VIP 登入
# ============================================================================
from finlab import login
login('R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m')

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
import time
import hashlib
import argparse
import json
import threading
from pathlib import Path
import shutil
from contextlib import contextmanager
import copy
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# 嘗試導入 tqdm 進度條
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm 未安裝，將使用基本進度顯示")
    # 簡易 tqdm 替代
    class tqdm:
        def __init__(self, iterable=None, desc="", total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total or (len(iterable) if iterable else 0)
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if self.n % 10 == 0 or self.n == self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total} ({100*self.n/self.total:.1f}%)", end="", flush=True)
            print()
        def update(self, n=1):
            self.n += n

# 嘗試導入 psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil 未安裝，使用備用進程檢測方案")

# 抑制警告
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('future.no_silent_downcasting', True)

# ============================================================================
# 🎯 2026 統一優化目標
# ============================================================================
TARGET_SHARPE = 4.2          # 🔥 目標夏普值 4.2
MIN_CAPACITY = 10_000_000    # 🔥 胃納量 1000萬
MAX_DRAWDOWN = 0.15          # 🔥 最大回檔 15%
TARGET_ANNUAL_RETURN = 0.4   # 年化報酬 40%
CAPACITY_PENALTY_WEIGHT = 0.5  # 胃納量懲罰權重

# 🔥 持股配比約束
MIN_POSITION_WEIGHT = 0.03   # 最小持股 3%
POSITION_WEIGHT_STEP = 0.03  # 持股必須是 3% 倍數

# 🔄 自適應持股設定
ADAPTIVE_POSITION = True     # 啟用自適應持股
BULL_MARKET_STOCKS = 12      # 多頭市場持股數
BEAR_MARKET_STOCKS = 6       # 空頭市場持股數
MARKET_THRESHOLD = 0.5       # 市場判斷閾值（>50% 股票在均線上=多頭）

# GA 演化參數
POPULATION_SIZE = 100        # 🔥 增加族群到 100
N_GENERATIONS = 300          # 🔥 增加至 300 代
MUTATION_RATE = 0.15         # 降低突變率以穩定

# ============================================================================
# 🔄 定期回測設定
# ============================================================================
BACKTEST_EVERY_N_GEN = 20    # 每 N 代執行一次詳細回測
TOP_N_HISTORICAL = 5         # 重啟時回測歷史前 N 名

# ============================================================================
# 🔬 樣本外測試設定 (Out-of-Sample Testing) - 防過擬合
# ============================================================================
TRAIN_START = '2014-01-01'   # 訓練期開始
TRAIN_END = '2022-12-31'     # 訓練期結束
TEST_START = '2023-01-01'    # 測試期開始
TEST_END = '2025-12-31'      # 測試期結束

# 過擬合警告閾值
OVERFIT_SHARPE_RATIO = 0.6   # 測試期夏普 / 訓練期夏普 < 0.6 則警告
OVERFIT_RETURN_RATIO = 0.5   # 測試期報酬 / 訓練期報酬 < 0.5 則警告

OOS_TEST_EVERY_N_GEN = 20    # 每 N 代執行一次樣本外測試

# ============================================================================
# 📊 持倉權重設定
# ============================================================================
MIN_STOCK_WEIGHT = 0.03    # 單檔股票最低權重 3%，低於此值設為 0%

# ============================================================================
# 🤖 FinLab ML API 設定
# ============================================================================
USE_ML_FEATURES = True     # 是否使用機器學習特徵
ML_FEATURE_WEIGHT = 0.1    # ML 特徵在評估中的權重

# ============================================================================
# 第一部分：改良的智能檔案鎖定機制
# ============================================================================
class ImprovedFileLock:
    """改良的檔案鎖定類，專為 Colab 多 notebook 環境優化"""

    def __init__(self, filename, timeout=10, check_interval=0.1):
        self.filename = filename
        self.timeout = timeout
        self.check_interval = check_interval
        self.lock_file = f"{filename}.lock"
        self.pid = os.getpid()
        self.acquired = False

    def _is_process_alive(self, pid):
        """檢查進程是否存活"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.pid_exists(pid)
            except:
                pass
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _read_lock_info(self):
        """讀取鎖檔案資訊"""
        if not os.path.exists(self.lock_file):
            return None
        try:
            with open(self.lock_file, 'r') as f:
                data = json.load(f)
                return data
        except:
            try:
                os.remove(self.lock_file)
            except:
                pass
            return None

    def _write_lock_info(self):
        """寫入鎖檔案資訊"""
        lock_info = {
            'pid': self.pid,
            'timestamp': time.time(),
            'window_id': WINDOW_ID if 'WINDOW_ID' in globals() else 'unknown'
        }
        with open(self.lock_file, 'w') as f:
            json.dump(lock_info, f)

    def _clean_stale_lock(self):
        """清理過期鎖"""
        lock_info = self._read_lock_info()
        if lock_info is None:
            return True
        lock_pid = lock_info.get('pid')
        if lock_pid and not self._is_process_alive(lock_pid):
            try:
                os.remove(self.lock_file)
                return True
            except:
                pass
        lock_time = lock_info.get('timestamp', 0)
        if time.time() - lock_time > 120:
            try:
                os.remove(self.lock_file)
                return True
            except:
                pass
        return False

    def acquire(self, wait=True):
        """獲取鎖"""
        start_time = time.time()
        while True:
            self._clean_stale_lock()
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._write_lock_info()
                self.acquired = True
                return True
            except (FileExistsError, OSError):
                pass
            if not wait:
                return False
            if time.time() - start_time > self.timeout:
                return False
            time.sleep(self.check_interval)

    def release(self):
        """釋放鎖"""
        if self.acquired and os.path.exists(self.lock_file):
            lock_info = self._read_lock_info()
            if lock_info and lock_info.get('pid') == self.pid:
                try:
                    os.remove(self.lock_file)
                    self.acquired = False
                except:
                    pass

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"無法獲取檔案鎖: {self.filename}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def safe_file_operation(operation, max_retries=3, delay=0.5):
    """安全的檔案操作"""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e


# ============================================================================
# 🔒 安全檔案管理器 (SafeFileManager)
# ============================================================================
class SafeFileManager:
    """
    安全的檔案寫入管理器
    - 使用暫存檔 + 改名的原子性寫入
    - 自動備份舊檔案
    - 寫入後驗證完整性
    """

    @staticmethod
    def safe_pickle_save(data, filepath, min_size=100):
        """安全的 pickle 寫入"""
        temp_path = filepath + '.tmp'
        backup_path = filepath + '.backup'

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)

            if not SafeFileManager.verify_pickle_file(temp_path, min_size):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

            if os.path.exists(filepath):
                if SafeFileManager.verify_pickle_file(filepath, min_size):
                    shutil.copy2(filepath, backup_path)

            shutil.move(temp_path, filepath)

            if SafeFileManager.verify_pickle_file(filepath, min_size):
                return True
            else:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, filepath)
                return False

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if os.path.exists(backup_path) and not os.path.exists(filepath):
                try:
                    shutil.copy2(backup_path, filepath)
                except:
                    pass
            return False

    @staticmethod
    def verify_pickle_file(filepath, min_size=100):
        """驗證 pkl 檔案是否正常"""
        try:
            if not os.path.exists(filepath):
                return False
            size = os.path.getsize(filepath)
            if size < min_size:
                return False
            with open(filepath, 'rb') as f:
                pickle.load(f)
            return True
        except:
            return False

    @staticmethod
    def safe_json_save(data, filepath):
        """安全的 JSON 寫入"""
        temp_path = filepath + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            shutil.move(temp_path, filepath)
            return True
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

print("✅ 安全檔案管理器 (SafeFileManager) 已載入")


# ============================================================================
# 持股配比管理器（3% 倍數約束）
# ============================================================================
class PositionWeightManager:
    """
    持股配比管理器

    功能：
    1. 確保每隻股票至少 3%
    2. 確保持股是 3% 倍數（3%, 6%, 9%, 12%...）
    3. 低於 3% 的直接設為 0%
    """

    @staticmethod
    def normalize_position(position_df) -> 'pd.DataFrame':
        """
        正規化持股比例

        規則：
        1. 每行獨立處理
        2. 低於 3% 的設為 0
        3. 剩餘的正規化並量化為 3% 倍數
        """
        if position_df is None or position_df.empty:
            return position_df

        try:
            # 確保是數值型
            result = position_df.astype(float).copy()

            for idx in result.index:
                row = result.loc[idx].copy()
                row_sum = row.sum()

                if row_sum == 0 or pd.isna(row_sum):
                    continue

                # 正規化
                normalized = row / row_sum

                # 過濾低於 3% 的
                filtered = normalized.where(normalized >= MIN_POSITION_WEIGHT, 0.0)

                # 重新正規化剩餘的
                filtered_sum = filtered.sum()
                if filtered_sum > 0:
                    filtered = filtered / filtered_sum

                    # 量化為 3% 倍數
                    quantized = (filtered / POSITION_WEIGHT_STEP).round() * POSITION_WEIGHT_STEP

                    # 調整總和為 1
                    total = quantized.sum()
                    if total > 0 and abs(total - 1.0) > 0.01:
                        max_col = quantized.idxmax()
                        quantized.loc[max_col] += (1.0 - total)

                    result.loc[idx] = quantized.values
                else:
                    result.loc[idx] = 0.0

            return result.astype(float)

        except Exception as e:
            # 如果處理失敗，返回原始 DataFrame
            print(f"   ⚠️ 持股正規化失敗: {e}")
            return position_df

position_weight_mgr = PositionWeightManager()


# ============================================================================
# 🔔 Discord 通知系統
# ============================================================================
# 優先使用環境變數，否則使用預設值
DISCORD_WEBHOOK_URL = os.environ.get(
    'DISCORD_WEBHOOK_URL',
    "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk"
)

class DiscordNotifier:
    """Discord 通知管理器"""

    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.enabled = True
        self.start_time = None

        if self.enabled:
            try:
                payload = {"content": f"✅ 十二組合快快龍 v2.1 (完整版) - 視窗 {WINDOW_ID} 啟動\n   訓練期: {TRAIN_START}~{TRAIN_END}\n   測試期: {TEST_START}~{TEST_END}", "username": "天空龍"}
                requests.post(self.webhook_url, json=payload, timeout=5)
                print("✅ Discord 通知系統已連接")
            except:
                self.enabled = False
                print("⚠️ Discord 通知系統未連接")

    def send(self, message):
        """發送簡單訊息"""
        if not self.enabled:
            return
        try:
            payload = {"content": message, "username": "天空龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
        except:
            pass

    def send_embed(self, title, description, color_type='info'):
        """發送嵌入式訊息"""
        if not self.enabled:
            return
        colors = {'success': 0x00ff00, 'warning': 0xffff00, 'error': 0xff0000, 'info': 0x0099ff}
        color = colors.get(color_type, colors['info'])
        try:
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            payload = {"embeds": [embed], "username": "天空龍"}
            requests.post(self.webhook_url, json=payload, timeout=5)
        except:
            pass

# 初始化 Discord 通知器（稍後在 WINDOW_ID 確定後初始化）
notifier = None


# ============================================================================
# 第二部分：獨立的資料管理器
# ============================================================================
class IndependentDataManager:
    """完全獨立的資料管理器"""

    def __init__(self, window_id, cache_dir):
        self.window_id = window_id
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f'data_cache_twelve_optimized_w{window_id}.pkl')
        self.data_dict = None

    def is_cache_valid(self):
        """檢查緩存是否有效"""
        if not os.path.exists(self.cache_file):
            return False
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.cache_file))
        current_time = datetime.datetime.now()
        return file_time.date() == current_time.date()

    def load_from_cache(self):
        """從緩存載入"""
        lock_file = f"{self.cache_file}_meta"
        try:
            with ImprovedFileLock(lock_file, timeout=5):
                def load_op():
                    with open(self.cache_file, 'rb') as f:
                        return pickle.load(f)
                self.data_dict = safe_file_operation(load_op)
                print(f"視窗 {self.window_id} 從緩存載入資料")
                return True
        except Exception as e:
            print(f"視窗 {self.window_id} 載入緩存失敗: {e}")
            return False

    def save_to_cache(self, data_dict):
        """儲存到緩存"""
        lock_file = f"{self.cache_file}_meta"
        try:
            with ImprovedFileLock(lock_file, timeout=5):
                def save_op():
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(data_dict, f)
                safe_file_operation(save_op)
                self.data_dict = copy.deepcopy(data_dict)
                print(f"視窗 {self.window_id} 資料已儲存到緩存")
        except Exception as e:
            print(f"視窗 {self.window_id} 儲存緩存失敗: {e}")

    def load_fresh_data(self):
        """載入新資料"""
        print(f"視窗 {self.window_id} 獨立載入資料...")
        time.sleep(random.uniform(0, 3))

        try:
            data_dict = {}

            # 價格數據
            data_dict['close'] = self._clean_data(data.get('price:收盤價'))
            data_dict['vol'] = self._clean_data(data.get('price:成交股數'))
            data_dict['open_'] = self._clean_data(data.get('price:開盤價'))
            data_dict['high'] = self._clean_data(data.get('price:最高價'))
            data_dict['low'] = self._clean_data(data.get('price:最低價'))
            data_dict['adj_close'] = self._clean_data(data.get("etl:adj_close"))

            # 財務數據
            data_dict['pe'] = data.get('price_earning_ratio:本益比')
            data_dict['rev'] = data.get('monthly_revenue:當月營收')
            data_dict['rev_yoy_growth'] = data.get('monthly_revenue:去年同月增減(%)')
            data_dict['rev_month_growth'] = data.get('monthly_revenue:上月比較增減(%)')
            data_dict['股價淨值比'] = data.get("price_earning_ratio:股價淨值比")
            data_dict['殖利率'] = data.get('price_earning_ratio:殖利率(%)')

            # 基本面指標
            data_dict['營業利益成長率'] = data.get('fundamental_features:營業利益成長率')
            data_dict['業外收支營收率'] = data.get('fundamental_features:業外收支營收率')
            data_dict['營業毛利率'] = data.get("fundamental_features:營業毛利率")
            data_dict['ROE綜合損益'] = data.get("fundamental_features:ROE綜合損益")
            data_dict['稅後淨利率'] = data.get("fundamental_features:稅後淨利率")
            data_dict['稅前淨利率'] = data.get("fundamental_features:稅前淨利率")
            data_dict['營業利益率'] = data.get('fundamental_features:營業利益率')

            # 籌碼資料
            data_dict['融資使用率'] = data.get('margin_transactions:融資使用率')
            data_dict['董監持有股數占比'] = data.get("internal_equity_changes:董監持有股數占比")

            # inventory 資料較大，記憶體不足或載入失敗時跳過
            try:
                data_dict['inventory'] = data.get("inventory")
            except Exception as e:
                error_msg = str(e).lower()
                error_type = type(e).__name__.lower()
                # 捕捉常見的資料載入錯誤：記憶體、EOF、pickle 損壞等
                skip_keywords = ['memory', 'allocate', 'eof', 'input', 'pickle', 'corrupt']
                if any(kw in error_msg or kw in error_type for kw in skip_keywords):
                    print(f"⚠️ 視窗 {self.window_id}: inventory 資料載入失敗，跳過（{type(e).__name__}）")
                    data_dict['inventory'] = None
                else:
                    raise e

            # 市值資料
            data_dict['市值'] = self._clean_data(data.get('etl:market_value'))

            # 十二策略需要的額外數據
            data_dict['股本'] = data.get('financial_statement:股本')
            data_dict['合約負債'] = data.get('financial_statement:合約負債_流動')
            data_dict['研究發展費'] = data.get("financial_statement:研究發展費")
            data_dict['營業收入淨額'] = data.get("financial_statement:營業收入淨額")

            # 現金流資料
            df1 = data.get('financial_statement:投資活動之淨現金流入_流出')
            df2 = data.get('financial_statement:營業活動之淨現金流入_流出')
            data_dict['自由現金流'] = (df1 + df2).rolling(4).mean()

            # 稅後淨利與股東權益
            稅後淨利 = data.get('fundamental_features:經常稅後淨利')
            權益總計 = data.get('financial_statement:股東權益總額')
            權益總計_safe = 權益總計.where(權益總計 > 0, np.nan)
            data_dict['股東權益報酬率'] = 稅後淨利 / 權益總計_safe
            data_dict['權益總計'] = 權益總計
            data_dict['稅後淨利'] = 稅後淨利

            self.data_dict = data_dict
            self.save_to_cache(data_dict)
            return True

        except Exception as e:
            print(f"視窗 {self.window_id} 載入資料失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _clean_data(self, df, min_value=1e-10):
        """清理資料"""
        if df is None or df.empty:
            return df
        if isinstance(df, pd.DataFrame):
            df = df.replace(0, np.nan)
            df = df.where(df > 0, np.nan)
        elif isinstance(df, pd.Series):
            df = df.replace(0, np.nan)
            df = df.where(df > 0, np.nan)
        return df

    def get_data(self):
        """獲取資料字典"""
        return self.data_dict


# ============================================================================
# 第三部分：環境設定
# ============================================================================
def get_window_id():
    """獲取視窗ID"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=1, choices=[1, 2, 3])
    args, _ = parser.parse_known_args()
    return args.window

WINDOW_ID = get_window_id()
print(f"\n{'='*60}")
print(f"🚀 啟動視窗 {WINDOW_ID} - 十二組合快快龍（高胃納量優化版）")
print(f"🎯 目標：夏普值 {TARGET_SHARPE}+ | 胃納量 {MIN_CAPACITY/1e6:.0f}M+")
print(f"PID: {os.getpid()}")
print(f"時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# =============================================================================
# 🔥 環境檢測與路徑設定
# =============================================================================
def detect_google_drive_path():
    """
    自動偵測本地 Google Drive 路徑
    支援 Windows Google Drive for Desktop
    """
    import platform

    if platform.system() == 'Windows':
        # Windows 常見 Google Drive 路徑
        possible_paths = [
            os.path.expanduser('~/Google Drive'),
            os.path.expanduser('~/Google 雲端硬碟'),
            'G:/My Drive',
            'G:/我的雲端硬碟',
            'D:/Google Drive',
            'D:/My Drive',
            os.path.expandvars('%USERPROFILE%/Google Drive'),
        ]

        # 也檢查所有磁碟機
        for drive_letter in 'DEFGHIJ':
            possible_paths.append(f'{drive_letter}:/My Drive')
            possible_paths.append(f'{drive_letter}:/我的雲端硬碟')
            possible_paths.append(f'{drive_letter}:/Google Drive')

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 偵測到 Google Drive: {path}")
                return path
    else:
        # Linux/Mac
        possible_paths = [
            os.path.expanduser('~/Google Drive'),
            os.path.expanduser('~/google-drive'),
            '/mnt/google-drive',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path

    return None


# ============================================================================
# Colab 環境檢測（改良版）
# ============================================================================
def is_colab_environment():
    """檢測是否在 Colab 環境中運行"""
    # 方法1：檢查 sys.modules
    if 'google.colab' in sys.modules:
        return True
    # 方法2：檢查 Colab 特有路徑
    if os.path.exists('/content/drive/MyDrive'):
        return True
    # 方法3：檢查環境變數
    if os.environ.get('COLAB_GPU') is not None:
        return True
    # 方法4：嘗試導入 google.colab
    try:
        import google.colab
        return True
    except ImportError:
        pass
    # 方法5：檢查 /content 目錄（Colab 特有）
    if os.path.exists('/content') and os.path.isdir('/content'):
        return True
    return False


IN_COLAB = is_colab_environment()

# Colab 環境設定
if IN_COLAB:
    # 只有在 notebook 直接執行時才掛載（subprocess 已經有掛載好的 Drive）
    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except:
            pass  # 已經掛載了

    print(f"✅ 檢測到 Colab 環境")
    BASE_PATH = '/content/drive/MyDrive/投資策略優化_十二策略_高胃納量版'
    WINDOW_PATH = f'{BASE_PATH}/window_{WINDOW_ID}'
    DRIVE_PATH = f'{WINDOW_PATH}/working'
    CACHE_PATH = f'{WINDOW_PATH}/cache'
    CHECKPOINT_PATH = f'{WINDOW_PATH}/checkpoints'
    LOG_PATH = f'{WINDOW_PATH}/logs'

    for path in [DRIVE_PATH, CACHE_PATH, CHECKPOINT_PATH, LOG_PATH]:
        os.makedirs(path, exist_ok=True)

    os.chdir(DRIVE_PATH)
    print(f"工作目錄: {os.getcwd()}")

    SHARED_PATH = f'{BASE_PATH}/shared_best'
    os.makedirs(SHARED_PATH, exist_ok=True)

    ADDITIONAL_SEARCH_PATHS = [
        '/content/drive/MyDrive/投資策略優化_十二策略_獨立版',
        '/content/drive/MyDrive/投資策略優化_十二策略_統一版'
    ]
else:
    # 🔥 本地執行模式（VS Code / 命令列）
    IN_COLAB = False

    # 優先使用環境變數指定的路徑
    GOOGLE_DRIVE_PATH = os.environ.get('GOOGLE_DRIVE_PATH', None)

    if GOOGLE_DRIVE_PATH is None:
        # 自動偵測 Google Drive
        GOOGLE_DRIVE_PATH = detect_google_drive_path()

    if GOOGLE_DRIVE_PATH and os.path.exists(GOOGLE_DRIVE_PATH):
        # 🔥 使用 Google Drive 共享（與 Colab 同步）
        print(f"📁 使用 Google Drive: {GOOGLE_DRIVE_PATH}")
        BASE_PATH = os.path.join(GOOGLE_DRIVE_PATH, '投資策略優化_十二策略_高胃納量版')
        SHARED_PATH = os.path.join(BASE_PATH, 'shared_best')
        ADDITIONAL_SEARCH_PATHS = [
            os.path.join(GOOGLE_DRIVE_PATH, '投資策略優化_十二策略_獨立版'),
            os.path.join(GOOGLE_DRIVE_PATH, '投資策略優化_十二策略_統一版')
        ]
    else:
        # 本地模式（無 Google Drive）
        print("⚠️ 未偵測到 Google Drive，使用本地目錄")
        print("   提示：設定環境變數 GOOGLE_DRIVE_PATH 指定 Google Drive 路徑")
        BASE_PATH = '.'
        SHARED_PATH = './shared_best'
        ADDITIONAL_SEARCH_PATHS = []

    WINDOW_PATH = os.path.join(BASE_PATH, f'window_{WINDOW_ID}')
    DRIVE_PATH = os.path.join(WINDOW_PATH, 'working')
    CACHE_PATH = os.path.join(WINDOW_PATH, 'cache')
    CHECKPOINT_PATH = os.path.join(WINDOW_PATH, 'checkpoints')
    LOG_PATH = os.path.join(WINDOW_PATH, 'logs')

    for path in [DRIVE_PATH, CACHE_PATH, CHECKPOINT_PATH, LOG_PATH, SHARED_PATH]:
        os.makedirs(path, exist_ok=True)

    os.chdir(DRIVE_PATH)
    print(f"📂 工作目錄: {os.getcwd()}")
    print(f"📂 Checkpoint: {CHECKPOINT_PATH}")
    print(f"📂 共享目錄: {SHARED_PATH}")

# 設定常數
FEE_RATIO = 1.425/1000
TAX_RATIO = 3/1000
STAGNATION_THRESHOLD = 3

# 🔥 3視窗交互設定
CROSS_WINDOW_INTERVAL = 10  # 每10代交換一次
CROSS_WINDOW_TOP_N = 5      # 每次交換最佳5個基因
ALL_WINDOW_IDS = [1, 2, 3]  # 所有視窗ID

# ============================================================================
# 🔥 3視窗交互機制 - CrossWindowManager
# ============================================================================
class CrossWindowManager:
    """3視窗交互管理器"""

    def __init__(self, base_dir, window_id, shared_dir):
        self.base_dir = base_dir
        self.window_id = window_id
        self.shared_dir = shared_dir
        Path(self.shared_dir).mkdir(parents=True, exist_ok=True)

    def save_elite_to_shared(self, population, generation):
        """保存精英到共享資料夾"""
        try:
            best_individuals = sorted(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else 0, reverse=True)[:CROSS_WINDOW_TOP_N]

            elite_data = {
                'window_id': self.window_id,
                'generation': generation,
                'timestamp': datetime.datetime.now().isoformat(),
                'elites': [{'gene': list(ind), 'fitness': ind.fitness.values[0] if ind.fitness.valid else 0}
                          for ind in best_individuals]
            }

            filename = f"{self.shared_dir}/elite_w{self.window_id}_g{generation}.pkl"
            success = SafeFileManager.safe_pickle_save(elite_data, filename)
            if success:
                print(f"   💾 精英已保存到共享資料夾 (視窗{self.window_id})")
            return success
        except Exception as e:
            print(f"   ⚠️ 保存精英失敗: {e}")
            return False

    def load_other_windows_elites(self):
        """載入其他視窗的精英"""
        all_elites = []
        try:
            for window_id in ALL_WINDOW_IDS:
                if window_id == self.window_id:
                    continue

                pattern = f"{self.shared_dir}/elite_w{window_id}_g*.pkl"
                files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

                if files:
                    latest_file = files[0]
                    try:
                        with open(latest_file, 'rb') as f:
                            data = pickle.load(f)
                        for elite in data.get('elites', []):
                            elite['source_window'] = window_id
                            all_elites.append(elite)
                    except:
                        continue
        except:
            pass
        return all_elites

    def perform_cross_window_exchange(self, population, generation, creator_class):
        """執行跨視窗基因交換"""
        print(f"\n   🔄 第 {generation} 代 - 3視窗交互...")

        self.save_elite_to_shared(population, generation)
        other_elites = self.load_other_windows_elites()

        if other_elites:
            n_inject = min(len(other_elites), CROSS_WINDOW_TOP_N)
            population.sort(key=lambda x: x.fitness.values[0] if x.fitness.valid else 0)

            for i, elite in enumerate(other_elites[:n_inject]):
                new_ind = creator_class(elite['gene'])
                new_ind.fitness.values = (elite['fitness'],)
                population[i] = new_ind

            print(f"   ✅ 注入 {n_inject} 個其他視窗精英")
            if notifier:
                notifier.send(f"🔄 視窗{self.window_id} 第{generation}代\n注入 {n_inject} 個外部精英")
        else:
            print(f"   ℹ️ 無其他視窗精英可注入")

        return population

    def get_all_windows_best(self):
        """取得所有視窗的最佳結果"""
        results = {}
        for window_id in ALL_WINDOW_IDS:
            pattern = f"{self.shared_dir}/elite_w{window_id}_g*.pkl"
            files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if files:
                try:
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    if data.get('elites'):
                        results[window_id] = {
                            'best_fitness': data['elites'][0]['fitness'],
                            'generation': data['generation']
                        }
                except:
                    continue
        return results

# 初始化 CrossWindowManager
cross_window_mgr = CrossWindowManager(BASE_PATH, WINDOW_ID, SHARED_PATH)
print("✅ 3視窗交互管理器已初始化")

# 初始化 Discord 通知器
notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)

# 診斷模式
DIAGNOSTIC_MODE = True

def diagnostic_print(message):
    """診斷訊息輸出"""
    if DIAGNOSTIC_MODE:
        print(f"[診斷] {message}")


# ============================================================================
# 第四部分：載入資料與計算衍生指標
# ============================================================================
print(f"\n視窗 {WINDOW_ID} 開始載入資料...")

data_manager = IndependentDataManager(WINDOW_ID, CACHE_PATH)

if data_manager.is_cache_valid():
    if data_manager.load_from_cache():
        data_dict = data_manager.get_data()
        for key, value in data_dict.items():
            globals()[key] = value
    else:
        if not data_manager.load_fresh_data():
            raise Exception(f"視窗 {WINDOW_ID} 無法載入資料")
        data_dict = data_manager.get_data()
        for key, value in data_dict.items():
            globals()[key] = value
else:
    if not data_manager.load_fresh_data():
        raise Exception(f"視窗 {WINDOW_ID} 無法載入資料")
    data_dict = data_manager.get_data()
    for key, value in data_dict.items():
        globals()[key] = value

# 計算衍生指標
print(f"視窗 {WINDOW_ID} 計算衍生指標...")

rev_ma3 = rev.average(3)
rev_ma12 = rev.average(12)

成交金額 = (close * vol).replace(0.0, np.nan)
成交金額 = 成交金額.where(成交金額 > 0, np.nan)
平均成交金額 = 成交金額.average(20)
capacity = 平均成交金額

# 🎯 高胃納量篩選器 - 強制最低 1000 萬
high_capacity_filter = capacity >= MIN_CAPACITY

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

print(f"✅ 視窗 {WINDOW_ID} 資料載入與指標計算完成")
print(f"📊 高胃納量股票數量: {high_capacity_filter.iloc[-1].sum()} 檔 (>= {MIN_CAPACITY/1e6:.0f}M)")

# ============================================================================
# 🤖 FinLab ML API 整合 - 機器學習特徵
# ============================================================================
ml_features = {}

if USE_ML_FEATURES:
    print(f"\n🤖 載入 FinLab ML 特徵...")
    try:
        # 嘗試載入 FinLab 機器學習模組
        from finlab.ml import feature_factory

        # 動量特徵
        ml_features['momentum_score'] = close.pct_change(20).rank(axis=1, pct=True)

        # 波動率特徵
        ml_features['volatility_rank'] = close.pct_change().rolling(20).std().rank(axis=1, pct=True)

        # 成交量異常特徵
        ml_features['volume_anomaly'] = (vol / vol.rolling(60).mean()).rank(axis=1, pct=True)

        # 價格位置特徵 (相對於52週高低點)
        high_52w = close.rolling(252).max()
        low_52w = close.rolling(252).min()
        ml_features['price_position'] = (close - low_52w) / (high_52w - low_52w + 1e-10)

        # 營收動能特徵
        ml_features['rev_momentum'] = rev_yoy_growth.rolling(3).mean().rank(axis=1, pct=True)

        # 綜合 ML 評分
        ml_features['ml_composite_score'] = (
            ml_features['momentum_score'] * 0.25 +
            (1 - ml_features['volatility_rank']) * 0.20 +  # 低波動加分
            ml_features['volume_anomaly'] * 0.15 +
            ml_features['price_position'] * 0.20 +
            ml_features['rev_momentum'] * 0.20
        )

        print(f"✅ ML 特徵載入完成 ({len(ml_features)} 個特徵)")

    except ImportError:
        print("⚠️ FinLab ML 模組未安裝，使用基礎特徵")
        # 基礎替代特徵
        ml_features['ml_composite_score'] = close.pct_change(20).rank(axis=1, pct=True)

    except Exception as e:
        print(f"⚠️ ML 特徵載入失敗: {e}")
        ml_features['ml_composite_score'] = pd.DataFrame(0.5, index=close.index, columns=close.columns)


# ============================================================================
# 🚀 預先計算昂貴指標 - 避免重複計算
# ============================================================================
print("\n🔄 預先計算昂貴指標...")
precompute_start = time.time()

# 預先計算 small_inv_under50 - 這個操作非常昂貴，只需計算一次
PRECOMPUTED_SMALL_INV = None
PRECOMPUTED_INVENTORY_BY_LEVEL = {}

if inventory is not None:
    try:
        PRECOMPUTED_SMALL_INV = (inventory[(inventory.持股分級.astype(int) <= 8)]
                                  .reset_index()
                                  .groupby(["date", "stock_id"])
                                  .agg({"占集保庫存數比例": "sum"})
                                  .reset_index()
                                  .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) <= 46
        print(f"✅ small_inv_under50 預計算完成")
    except Exception as e:
        print(f"⚠️ small_inv_under50 預計算失敗: {e}")
        PRECOMPUTED_SMALL_INV = None

    # 預先計算不同持股分級的 inventory 數據（用於 boss_inventory）
    try:
        for min_level in range(9, 16):
            for max_level in range(min_level, 16):
                key = (min_level, max_level)
                filtered = inventory[(inventory.持股分級.astype(int) >= min_level) &
                                    (inventory.持股分級.astype(int) <= max_level)]
                if not filtered.empty:
                    PRECOMPUTED_INVENTORY_BY_LEVEL[key] = (filtered
                        .reset_index()
                        .groupby(["date", "stock_id"])
                        .agg({"占集保庫存數比例": "sum"})
                        .reset_index()
                        .pivot(index="date", columns="stock_id", values="占集保庫存數比例"))
        print(f"✅ boss_inventory 預計算完成 ({len(PRECOMPUTED_INVENTORY_BY_LEVEL)} 個組合)")
    except Exception as e:
        print(f"⚠️ boss_inventory 預計算失敗: {e}")
else:
    print("⚠️ inventory 資料未載入，跳過大股東持股預計算（策略 4 將使用替代條件）")

precompute_time = time.time() - precompute_start
print(f"✅ 預計算完成，耗時 {precompute_time:.1f} 秒")


# ============================================================================
# 第五部分：🎯 智能基因初始化 - 不同參數使用合適的範圍
# ============================================================================
GENE_RANGES = {
    # 策略權重 (0-11): 0.5 ~ 3.0
    'allocation': (0.5, 3.0),

    # 本益比相關: 5 ~ 50
    'pe': (5, 50),

    # 股價淨值比: 0.5 ~ 5
    'pb': (0.5, 5),

    # 成交量 (張): 100 ~ 5000
    'volume': (100, 5000),

    # 比率類 (%): 0 ~ 100
    'ratio': (0, 100),

    # 成長率 (%): -50 ~ 200
    'growth': (-50, 200),

    # 天數/期數: 1 ~ 60
    'period': (1, 60),

    # 選股數量: 3 ~ 30
    'top_n': (3, 30),

    # 市值 (億): 10 ~ 1000
    'market_cap': (10, 1000),

    # 殖利率 (%): 1 ~ 15
    'yield': (1, 15),

    # 停損停利 (%): 5 ~ 50
    'stop': (5, 50),
}

def smart_gene_init(gene_type='default'):
    """智能基因初始化 - 根據參數類型使用合適的範圍"""
    ranges = GENE_RANGES.get(gene_type, (0.1, 10))
    return random.uniform(ranges[0], ranges[1])


# ============================================================================
# 🔄 自適應持股機制 - 根據市場狀況動態調整持股數量
# ============================================================================
def get_market_sentiment(close_df, ma_period=60):
    """
    計算市場情緒指標
    返回：每日市場情緒分數 (0~1)，越高代表越多股票在均線之上
    """
    ma = close_df.rolling(ma_period).mean()
    above_ma = (close_df > ma).astype(float)
    sentiment = above_ma.mean(axis=1)  # 每日有多少比例的股票在均線上
    return sentiment


def get_adaptive_top_n(close_df, base_top_n, sentiment=None):
    """
    根據市場情緒動態調整持股數量
    - 多頭市場（>50% 股票在均線上）：增加持股
    - 空頭市場（<50% 股票在均線上）：減少持股
    """
    if not ADAPTIVE_POSITION:
        return base_top_n

    if sentiment is None:
        sentiment = get_market_sentiment(close_df)

    # 根據情緒調整持股數量
    bull_ratio = BULL_MARKET_STOCKS / base_top_n if base_top_n > 0 else 1
    bear_ratio = BEAR_MARKET_STOCKS / base_top_n if base_top_n > 0 else 0.5

    # 線性插值：空頭(0) -> 多頭(1)
    adaptive_ratio = bear_ratio + (bull_ratio - bear_ratio) * sentiment.clip(0, 1)
    adaptive_top_n = (base_top_n * adaptive_ratio).round().astype(int).clip(lower=3)

    return adaptive_top_n


# ============================================================================
# 第六部分：十二個策略函數定義（高胃納量優化版）
# ============================================================================

def strategy_low_volatility_pe(params):
    """策略1: 低波動本益比策略 - 高胃納量版"""
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

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    # 🚀 使用預計算的 small_inv_under50（避免重複計算）
    small_inv_under50 = PRECOMPUTED_SMALL_INV if PRECOMPUTED_SMALL_INV is not None else pd.DataFrame(True, index=close.index, columns=close.columns)

    cond_all = (cond1 & cond2 & tree_select_factor & cond排除月營收成長趨勢過老
                & cond排除月營收連3月衰退 & cond收盤價大於季線及半年線
                & cond近三個月營收大於年營收 & cond單月營收月增率
                & ~(limit_up_all_day) & (condition_近1日成交均量大於100張)
                & (gpm_trend_1) & (roe_trend_1) & (small_inv_under50)
                & pb_range_1 & pe_range_1 & cond_capacity)

    position = peg[cond_all & (peg > 0)].is_smallest(params['top_n']).reindex(rev.index_str_to_date().index, method='ffill')
    return position


def strategy_small_investor(params):
    """策略2: 小資族策略 - 高胃納量版"""
    當月營收 = data.get('monthly_revenue:當月營收') * 1000
    當季營收 = 當月營收.rolling(4).sum()
    市值營收比 = 市值 / 當季營收

    # 放寬市值限制以適應高胃納量
    cond1 = (市值 < params['market_value_limit'])
    cond2 = 自由現金流 > params['min_free_cash_flow']
    cond3 = 股東權益報酬率 > params['min_roe']
    cond4 = 營業利益成長率 > params['min_op_profit_growth']
    cond5 = 市值營收比 < params['market_rev_ratio_limit']
    cond6 = vol > params['volume_threshold']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

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

    position = ((cond1 & cond2 & cond3 & cond4 & cond5 & cond6
                 & cond排除月營收成長趨勢過老 & cond單月營收月增率連續3月大於閾值
                 & cond排除月營收連3月衰退 & cond近三個月營收大於年營收
                 & cond收盤價大於均線 & 業外收支營收率占比低
                 & cond確認營收底部 & cond_capacity) * rsv).is_largest(params['top_n'])

    position = position.reindex(當月營收.index_str_to_date().index, method='ffill')
    return position


def strategy_revenue_price_turbo(params):
    """策略3: 營收股價雙渦輪策略 - 高胃納量版"""
    rev_ma_period = max(1, int(params['rev_ma_period']))
    rev_ma = rev.average(rev_ma_period)
    rev_ma_lookback = max(rev_ma_period + 1, int(params['rev_ma_lookback']))
    condition_近N月平均營收創M個月來新高 = rev_ma == rev_ma.rolling(rev_ma_lookback, min_periods=1).max()

    price_high_window = max(1, int(params['price_high_window']))
    condition_近N日內有1日股價創新高 = (close == close.rolling(260).max()).sustain(price_high_window, 1)
    condition_成交均量大於閾值 = vol.average(1) > params['min_volume']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    long_ma_pattern = ((close > close.average(5)) & (close > close.average(10))
                     & (close > close.average(20)) & (close > close.average(60))
                     & (close > close.average(120)))

    收盤價_超級績效 = close > (close.average(params['performance_ma_period'])*params['performance_threshold'])
    rsi_higt_trend = (rsi > params['rsi_threshold']).sustain(params['rsi_trend_period'])
    gpm_trend_1 = (營業毛利率 > params['min_gpm']).sustain(params['gpm_sustain_period'])
    btpm_trend_1 = (稅前淨利率 > params['min_btpm']).sustain(params['btpm_sustain_period'])
    atpm_trend_1 = (稅後淨利率 > params['min_atpm']).sustain(params['atpm_sustain_period'])
    rev_rise_nsatisfy_2 = rev_yoy_growth.rank(pct=True, axis=1) > params['rev_growth_percentile']

    # 🚀 使用預計算的 boss_inventory（避免重複計算）
    boss_key = (int(params['boss_min_level']), int(params['boss_max_level']))
    if boss_key in PRECOMPUTED_INVENTORY_BY_LEVEL:
        boss_inventory_over400 = PRECOMPUTED_INVENTORY_BY_LEVEL[boss_key] >= params['min_boss_ratio']
    elif inventory is not None:
        # 如果沒有預計算但有 inventory，使用原始計算
        boss_inventory_over400 = (inventory[(inventory.持股分級.astype(int) >= params['boss_min_level']) &
                                            (inventory.持股分級.astype(int) <= params['boss_max_level'])]
                                .reset_index()
                                .groupby(["date", "stock_id"])
                                .agg({"占集保庫存數比例": "sum"})
                                .reset_index()
                                .pivot(index="date", columns="stock_id", values="占集保庫存數比例")) >= params['min_boss_ratio']
    else:
        # inventory 未載入時，使用替代條件（董監持股）
        boss_inventory_over400 = 董監持有股數占比 >= 10  # 董監持股 >= 10% 作為替代

    pe_limit = params['pe_limit']
    pe_range_1 = (pe_limit <= pe)
    min_price = params['min_price']

    conditions = (condition_近N月平均營收創M個月來新高 & condition_近N日內有1日股價創新高
                 & condition_成交均量大於閾值 & long_ma_pattern & gpm_trend_1
                 & btpm_trend_1 & atpm_trend_1 & rev_rise_nsatisfy_2
                 & (close > min_price) & (業外收支營收率 < 7.3) & ~(pe_range_1)
                 & (收盤價_超級績效) & ((vol >= vol.rolling(20).mean()*0.8))
                 & (rsi_higt_trend) & (boss_inventory_over400) & ~(limit_up_all_day)
                 & cond_capacity)

    position = rev_yoy_growth * conditions
    position = position[position > 0].is_largest(params['top_n']).reindex(rev.index_str_to_date().index, method="ffill")
    return position


def strategy_high_yield_turtle(params):
    """策略4: 高殖利率烏龜策略 - 高胃納量版"""
    sma20 = close.average(20)
    sma60 = close.average(60)

    cond1 = 殖利率 >= params['min_yield_ratio']
    cond2 = (close > sma20) & (close > sma60)
    cond3 = rev.average(3) > rev.average(12)
    cond4 = 營業利益率 >= params['min_op_earn_ratio']
    cond5 = 董監持有股數占比 >= params['min_boss_hold']
    cond6 = (vol.average(5) >= params['min_volume']) & (vol.average(5) <= params['max_volume'])

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond_capacity
    cond_all = cond_all * rev_yoy_growth

    position = cond_all[cond_all > 0].is_largest(params['top_n'])
    position = position.reindex(rev.index_str_to_date().index, method='ffill')
    return position


def strategy_low_volatility_index(params):
    """策略5: 低波動性指標策略 - 高胃納量版"""
    std = close.pct_change().rolling(params['std_window']).std().rank(axis=1, pct=True)

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    position = 市值[(vol.average(20) > params['min_volume'])
        & (close > close.average(60)) & (close > close.average(120))
        & (close > close.average(250)) & (std < params['std_threshold'])
        & cond_capacity
    ].is_smallest(params['top_n'])

    position = position.reindex(close.index_str_to_date().index, method='ffill')
    return position


def strategy_market_indicator(params):
    """策略6: 藏獒外掛大盤指針策略 - 高胃納量版"""
    vol_ma = vol.average(10)

    cond1 = (close == close.rolling(params['new_high_window']).max())
    cond2 = ~(rev_yoy_growth < params['min_year_growth']).sustain(3)
    cond3 = ~(rev_yoy_growth > params['max_year_growth']).sustain(12, 8)
    cond4 = ((rev.rolling(12).min())/(rev) < params['rev_bottom_ratio']).sustain(3)
    cond5 = (rev_month_growth > params['min_month_growth']).sustain(3)
    cond6 = vol_ma > params['min_volume']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond_capacity
    buy = vol_ma * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(params['top_n'])

    position = buy.reindex(rev.index_str_to_date().index, method='ffill')
    return position


def strategy_prison_rabbit(params):
    """策略7: 監獄兔策略 - 高胃納量版"""
    rev_growth_ma = rev_yoy_growth.average(params['growth_ma_period'])

    cond1 = rev_growth_ma > params['min_growth_rate']
    cond2 = pe < params['max_pe']
    cond3 = pe > params['min_pe']
    cond4 = ROE綜合損益 > params['min_roe']
    cond5 = 營業毛利率 > params['min_gpm']
    cond6 = vol.average(20) > params['min_volume']

    momentum = close / close.shift(params['momentum_period'])
    cond7 = momentum > params['min_momentum']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond_capacity
    position = (cond_all * momentum).is_largest(params['top_n'])

    position = position.reindex(rev.index_str_to_date().index, method='ffill')
    return position


def strategy_elite_momentum(params):
    """策略8: 精選強勢股策略 - 高胃納量版"""
    rs = close / close.average(params['rs_period'])
    vol_ratio = vol / vol.average(params['vol_ma_period'])

    cond1 = rs > params['min_rs']
    cond2 = vol_ratio > params['min_vol_ratio']
    cond3 = close > close.average(params['ma_period'])
    cond4 = 營業利益成長率 > params['min_op_growth']
    cond5 = rev_yoy_growth > params['min_rev_growth']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond_capacity
    position = (cond_all * rs).is_largest(params['top_n'])

    position = position.reindex(close.index_str_to_date().index, method='ffill')
    return position


def strategy_pure_technical(params):
    """策略9: 純技術分析策略 - 高胃納量版"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal

    ma20 = close.average(20)
    std20 = close.rolling(20).std()
    upper_band = ma20 + (std20 * params['bb_multiplier'])

    rsi_overbought = rsi > params['rsi_overbought']

    cond1 = macd_histogram > 0
    cond2 = close > ma20
    cond3 = close < upper_band
    cond4 = ~rsi_overbought
    cond5 = vol > vol.average(params['vol_period'])

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond_capacity
    score = (100 - rsi) * cond_all

    position = score[score > 0].is_largest(params['top_n'])
    position = position.reindex(close.index_str_to_date().index, method='ffill')
    return position


def strategy_scaredy_cat(params):
    """策略10: 膽小貓策略 - 高胃納量版"""
    volatility = close.rolling(params['vol_window']).std() / close.rolling(params['vol_window']).mean()

    cond1 = volatility < params['max_volatility']
    cond2 = 市值 > params['min_market_cap']
    cond3 = pe > params['min_pe']
    cond4 = pe < params['max_pe']
    cond5 = 殖利率 > params['min_yield']
    cond6 = 營業毛利率 > params['min_gpm']
    cond7 = ROE綜合損益 > params['min_roe']

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    price_stability = 1 / (volatility + 0.001)

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond_capacity
    position = (cond_all * price_stability).is_largest(params['top_n'])

    position = position.reindex(close.index_str_to_date().index, method='ffill')
    return position


def strategy_contract_debt_construction(params):
    """策略11: 合約負債建築工策略 - 高胃納量版"""
    contract_debt_growth = 合約負債 / 合約負債.shift(4)

    cond1 = contract_debt_growth > params['min_debt_growth']
    cond2 = 合約負債 > params['min_debt_amount']
    cond3 = rev_yoy_growth > params['min_rev_growth']
    cond4 = 營業毛利率 > params['min_gpm']
    cond5 = close > close.average(params['ma_period'])

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond_capacity
    score = contract_debt_growth * cond_all

    position = score[score > 0].is_largest(params['top_n'])
    position = position.reindex(合約負債.index_str_to_date().index, method='ffill')
    return position


def strategy_rd_maniac(params):
    """策略12: 研發魔人策略 - 高胃納量版"""
    rd_ratio = 研究發展費 / 營業收入淨額
    rd_growth = 研究發展費 / 研究發展費.shift(4)

    cond1 = rd_ratio > params['min_rd_ratio']
    cond2 = rd_growth > params['min_rd_growth']
    cond3 = 營業利益成長率 > params['min_op_growth']
    cond4 = rev_yoy_growth > params['min_rev_growth']
    cond5 = 市值 > params['min_market_cap']
    cond6 = close > close.average(params['ma_period'])

    # 🎯 強制高胃納量篩選
    cond_capacity = high_capacity_filter

    cond_all = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond_capacity
    score = rd_ratio * rd_growth * cond_all

    position = score[score > 0].is_largest(params['top_n'])
    position = position.reindex(研究發展費.index_str_to_date().index, method='ffill')
    return position


# ============================================================================
# 第七部分：🎯 優化版 gene_to_params_twelve - 智能參數映射
# ============================================================================
def gene_to_params_twelve(gene):
    """將基因轉換為十二個策略的參數 - 優化版"""
    if not isinstance(gene, list):
        gene = list(gene)

    required_length = 251
    current_length = len(gene)

    if current_length < required_length:
        extension = [random.uniform(0.1, 10)] * (required_length - current_length)
        gene = gene + extension
    elif current_length > required_length:
        gene = gene[:required_length]

    # 🎯 優化策略權重分配 - 確保最低權重
    alloc_raw = [max(0.05, abs(gene[i])) for i in range(12)]  # 最低 5% 權重
    alloc_sum = sum(alloc_raw)
    allocation = [val/alloc_sum for val in alloc_raw]

    # 策略1: 低波動本益比策略參數
    low_vol_pe_params = {
        'rev_ma3_ma12_ratio': max(0.8, min(1.5, abs(gene[12]))),
        'rev_consistency': max(0.9, min(1.2, abs(gene[13]))),
        'volatility_threshold': max(0.01, min(0.1, abs(gene[14])/100)),
        'margin_usage_limit': max(10, min(80, abs(gene[15]) * 10)),
        'non_op_income_limit': max(5, min(20, abs(gene[16]) * 2)),
        'min_volume': max(500, abs(gene[17] * 500)),  # 提高最低成交量
        'pe_min': max(5, min(15, abs(gene[18]))),
        'pe_max': max(20, min(50, abs(gene[19]) * 5)),
        'top_n': max(5, min(20, int(abs(gene[20])))),
        'min_rev_yoy_threshold': -max(5, min(30, abs(gene[120]))),
        'rev_decline_period': max(2, min(6, int(abs(gene[121])))),
        'max_rev_yoy_threshold': max(30, min(100, abs(gene[122]) * 10)),
        'old_trend_period': max(6, min(18, int(abs(gene[123])))),
        'old_trend_match': max(4, min(12, int(abs(gene[124])))),
        'rev_bottom_window': max(6, min(18, int(abs(gene[125])))),
        'rev_bottom_ratio': max(1.0, min(1.5, abs(gene[126]))),
        'rev_bottom_sustain': max(2, min(6, int(abs(gene[127])))),
        'min_rev_mom_growth': -max(5, min(20, abs(gene[128]))),
        'rev_mom_sustain': max(2, min(5, int(abs(gene[129])))),
        'quarter_ma': max(40, min(80, int(abs(gene[130]) * 8))),
        'half_year_ma': max(100, min(150, int(abs(gene[131]) * 15))),
        'long_ma': max(180, min(260, int(abs(gene[132]) * 26))),
        'recent_rev_period': max(2, min(4, int(abs(gene[133])))),
        'annual_rev_period': max(10, min(14, int(abs(gene[134])))),
        'pb_min': max(0.5, min(2, abs(gene[135]))),
        'pb_max': max(3, min(8, abs(gene[136]))),
        'min_gpm': max(10, min(40, abs(gene[137]) * 4)),
        'gpm_sustain_period': max(2, min(6, int(abs(gene[138])))),
        'min_roe': max(5, min(20, abs(gene[139]) * 2)),
        'roe_sustain_period': max(2, min(6, int(abs(gene[140])))),
        'min_capacity': MIN_CAPACITY  # 🎯 強制最低胃納量
    }

    # 策略2: 小資族策略參數
    small_inv_params = {
        'market_value_limit': max(50e9, abs(gene[21]) * 10e9),  # 放寬市值上限
        'market_rev_ratio_limit': max(1, min(5, abs(gene[22]))),
        'rev_yoy_growth_limit': -max(5, min(20, abs(gene[23]))),
        'rev_mom_growth_limit': -max(3, min(15, abs(gene[24]))),
        'rsv_period': max(5, min(20, int(abs(gene[25])))),
        'ma_period': max(20, min(60, int(abs(gene[26]) * 6))),
        'volume_threshold': max(500, abs(gene[27] * 500)),
        'top_n': max(5, min(20, int(abs(gene[28])))),
        'min_free_cash_flow': gene[150] * 1e6,
        'min_roe': max(5, min(25, abs(gene[151]) * 2.5)),
        'min_op_profit_growth': max(5, min(50, abs(gene[152]) * 5)),
        'min_capacity': MIN_CAPACITY
    }

    # 策略3: 營收股價雙渦輪策略參數
    turbo_params = {
        'rev_ma_period': max(2, min(6, int(abs(gene[29])))),
        'rev_ma_lookback': max(6, min(18, int(abs(gene[30])))),
        'price_high_window': max(5, min(30, int(abs(gene[31])))),
        'min_volume': max(500, abs(gene[32] * 500)),
        'min_price': max(20, min(100, abs(gene[33]) * 10)),
        'rsi_threshold': max(40, min(70, abs(gene[34]) * 7)),
        'pe_limit': max(30, min(80, abs(gene[35]) * 8)),
        'top_n': max(5, min(20, int(abs(gene[36])))),
        'performance_ma_period': max(120, min(260, int(abs(gene[100]) * 26))),
        'performance_threshold': max(1.0, min(1.5, abs(gene[101]))),
        'rsi_trend_period': max(3, min(10, int(abs(gene[102])))),
        'min_gpm': max(10, min(40, abs(gene[103]) * 4)),
        'gpm_sustain_period': max(2, min(6, int(abs(gene[104])))),
        'min_btpm': max(5, min(30, abs(gene[105]) * 3)),
        'btpm_sustain_period': max(2, min(6, int(abs(gene[106])))),
        'min_atpm': max(3, min(25, abs(gene[107]) * 2.5)),
        'atpm_sustain_period': max(2, min(6, int(abs(gene[108])))),
        'rev_growth_percentile': max(0.5, min(0.9, abs(gene[109]) / 10)),
        'boss_min_level': max(10, min(14, int(abs(gene[110])))),
        'boss_max_level': max(14, min(16, int(abs(gene[111])))),
        'min_boss_ratio': max(30, min(60, abs(gene[112]) * 6)),
        'min_capacity': MIN_CAPACITY
    }

    # 策略4: 高殖利率烏龜策略參數
    high_yield_turtle_params = {
        'min_yield_ratio': max(3, min(8, abs(gene[37]))),
        'min_op_earn_ratio': max(5, min(20, gene[38] * 2)),
        'min_boss_hold': max(10, min(50, abs(gene[39]) * 5)),
        'min_volume': max(300, abs(gene[40] * 300)),
        'max_volume': max(5000, abs(gene[41] * 1000)),
        'top_n': max(5, min(20, int(abs(gene[42])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略5: 低波動性指標策略參數
    low_vol_index_params = {
        'min_volume': max(500, abs(gene[43] * 500)),
        'std_window': max(10, min(60, int(abs(gene[44]) * 6))),
        'std_threshold': max(0.1, min(0.5, abs(gene[45]) / 20)),
        'top_n': max(5, min(20, int(abs(gene[46])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略6: 藏獒外掛大盤指針策略參數
    market_indicator_params = {
        'new_high_window': max(60, min(260, int(abs(gene[47]) * 26))),
        'min_year_growth': -max(5, min(20, abs(gene[48]))),
        'max_year_growth': max(50, min(150, abs(gene[49]) * 15)),
        'rev_bottom_ratio': max(1.0, min(1.5, abs(gene[50]))),
        'min_month_growth': -max(5, min(20, abs(gene[51]))),
        'min_volume': max(500, abs(gene[52] * 500)),
        'top_n': max(5, min(20, int(abs(gene[53])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略7: 監獄兔策略參數
    prison_rabbit_params = {
        'growth_ma_period': max(2, min(6, int(abs(gene[54])))),
        'min_growth_rate': max(10, min(50, abs(gene[55]) * 5)),
        'max_pe': max(30, min(80, abs(gene[56]) * 8)),
        'min_pe': max(5, min(15, abs(gene[57]))),
        'min_roe': max(8, min(25, abs(gene[58]) * 2.5)),
        'min_gpm': max(15, min(45, abs(gene[59]) * 4.5)),
        'min_volume': max(500, abs(gene[60] * 500)),
        'momentum_period': max(10, min(60, int(abs(gene[61]) * 6))),
        'min_momentum': max(1.0, min(1.5, abs(gene[62]))),
        'top_n': max(5, min(20, int(abs(gene[63])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略8: 精選強勢股策略參數
    elite_momentum_params = {
        'rs_period': max(10, min(60, int(abs(gene[64]) * 6))),
        'vol_ma_period': max(10, min(30, int(abs(gene[65]) * 3))),
        'min_rs': max(1.0, min(1.5, abs(gene[66]))),
        'min_vol_ratio': max(0.8, min(2.0, abs(gene[67]))),
        'ma_period': max(20, min(60, int(abs(gene[68]) * 6))),
        'min_op_growth': max(10, min(50, abs(gene[69]) * 5)),
        'min_rev_growth': max(5, min(40, abs(gene[70]) * 4)),
        'top_n': max(5, min(20, int(abs(gene[71])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略9: 純技術分析策略參數
    pure_technical_params = {
        'bb_multiplier': max(1.5, min(3.0, abs(gene[72]))),
        'rsi_oversold': max(20, min(40, abs(gene[73]) * 4)),
        'rsi_overbought': max(60, min(80, abs(gene[74]) * 8)),
        'vol_period': max(10, min(30, int(abs(gene[75]) * 3))),
        'top_n': max(5, min(20, int(abs(gene[76])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略10: 膽小貓策略參數
    scaredy_cat_params = {
        'vol_window': max(10, min(60, int(abs(gene[77]) * 6))),
        'max_volatility': max(0.02, min(0.1, abs(gene[78]) / 100)),
        'min_market_cap': max(30e9, abs(gene[79]) * 10e9),  # 提高最低市值
        'min_pe': max(5, min(15, abs(gene[80]))),
        'max_pe': max(25, min(50, abs(gene[81]) * 5)),
        'min_yield': max(2, min(8, abs(gene[82]))),
        'min_gpm': max(15, min(40, abs(gene[83]) * 4)),
        'min_roe': max(8, min(25, abs(gene[84]) * 2.5)),
        'top_n': max(5, min(20, int(abs(gene[85])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略11: 合約負債建築工策略參數
    contract_debt_params = {
        'min_debt_growth': max(1.0, min(2.0, abs(gene[86]))),
        'min_debt_amount': max(1e8, abs(gene[87]) * 1e8),
        'min_rev_growth': max(5, min(40, abs(gene[88]) * 4)),
        'min_gpm': max(10, min(35, abs(gene[89]) * 3.5)),
        'ma_period': max(20, min(60, int(abs(gene[90]) * 6))),
        'top_n': max(5, min(20, int(abs(gene[91])))),
        'min_capacity': MIN_CAPACITY
    }

    # 策略12: 研發魔人策略參數
    rd_maniac_params = {
        'min_rd_ratio': max(0.02, min(0.15, abs(gene[92]) / 100)),
        'min_rd_growth': max(1.0, min(2.0, abs(gene[93]))),
        'min_op_growth': max(10, min(50, abs(gene[94]) * 5)),
        'min_rev_growth': max(5, min(40, abs(gene[95]) * 4)),
        'min_market_cap': max(20e9, abs(gene[96]) * 10e9),
        'ma_period': max(20, min(60, int(abs(gene[97]) * 6))),
        'top_n': max(5, min(20, int(abs(gene[98])))),
        'min_capacity': MIN_CAPACITY
    }

    # 整體參數
    overall_params = {
        'stop_loss': max(0.05, min(0.25, abs(gene[200])/100)),
        'trail_stop': max(0.05, min(0.30, abs(gene[201])/100)),
        'take_profit': max(0.20, min(0.80, abs(gene[202])/100)),
        'position_limit': max(0.10, min(0.30, abs(gene[203])/100)),
        'trade_at_price': ["open", "close", "high_low_avg", "open_close_avg"][int(abs(gene[204])) % 4],
        'liquidity_threshold': MIN_CAPACITY,  # 🎯 強制最低流動性
        'capacity_threshold': MIN_CAPACITY,   # 🎯 強制最低胃納量
    }

    return (allocation, low_vol_pe_params, small_inv_params, turbo_params,
            high_yield_turtle_params, low_vol_index_params, market_indicator_params,
            prison_rabbit_params, elite_momentum_params, pure_technical_params,
            scaredy_cat_params, contract_debt_params, rd_maniac_params, overall_params)


# ============================================================================
# 第八部分：combined_strategy_twelve - 高胃納量版
# ============================================================================
def combined_strategy_twelve(gene, start_date_str="2014-01-01", end_date_str=None):
    """
    根據基因參數生成合併策略 - 高胃納量版

    Args:
        gene: 基因向量
        start_date_str: 開始日期 (用於樣本外測試)
        end_date_str: 結束日期 (用於樣本外測試)
    """
    try:
        results = gene_to_params_twelve(gene)
        allocation = results[0]
        strategy_params = results[1:13]
        overall_params = results[13]

        strategies = [
            (strategy_low_volatility_pe, strategy_params[0]),
            (strategy_small_investor, strategy_params[1]),
            (strategy_revenue_price_turbo, strategy_params[2]),
            (strategy_high_yield_turtle, strategy_params[3]),
            (strategy_low_volatility_index, strategy_params[4]),
            (strategy_market_indicator, strategy_params[5]),
            (strategy_prison_rabbit, strategy_params[6]),
            (strategy_elite_momentum, strategy_params[7]),
            (strategy_pure_technical, strategy_params[8]),
            (strategy_scaredy_cat, strategy_params[9]),
            (strategy_contract_debt_construction, strategy_params[10]),
            (strategy_rd_maniac, strategy_params[11])
        ]

        position_list = []
        for i, (strategy_func, params) in enumerate(strategies):
            try:
                position = strategy_func(params)
                if position.empty:
                    position = pd.DataFrame(0, index=close.index, columns=close.columns)
                position = position.astype(float)
                position_weighted = position * allocation[i]
                position_list.append(position_weighted)
            except Exception as e:
                # 策略執行失敗時使用空倉位
                position = pd.DataFrame(0, index=close.index, columns=close.columns)
                position_list.append(position)

        position_combined = reduce(lambda x, y: x.add(y, fill_value=0), position_list)

        # 標準化權重
        row_sums = position_combined.sum(axis=1)
        for idx in position_combined.index:
            if row_sums[idx] > 1.0:
                position_combined.loc[idx] = position_combined.loc[idx] / row_sums[idx]

        # 🎯 強制高胃納量篩選 - 雙重保險
        position_combined = position_combined * high_capacity_filter

        # 🎯 最低權重過濾：每檔股票至少 3%，否則設為 0%
        position_combined = position_combined.where(position_combined >= MIN_STOCK_WEIGHT, 0)

        # 再次標準化（確保總權重不超過 100%）
        row_sums_after = position_combined.sum(axis=1)
        for idx in position_combined.index:
            if row_sums_after[idx] > 1.0:
                position_combined.loc[idx] = position_combined.loc[idx] / row_sums_after[idx]

        # 時間範圍處理
        if position_combined.empty:
            return pd.DataFrame(), overall_params

        if not isinstance(position_combined.index, pd.DatetimeIndex):
            position_combined.index = pd.to_datetime(position_combined.index, errors='coerce')
            position_combined = position_combined.loc[~position_combined.index.isna()]

        # 開始日期過濾
        start_date_ts = pd.Timestamp(start_date_str)
        position_combined = position_combined[position_combined.index >= start_date_ts]

        # 🔬 結束日期過濾（用於樣本外測試）
        if end_date_str:
            end_date_ts = pd.Timestamp(end_date_str)
            position_combined = position_combined[position_combined.index <= end_date_ts]
        elif 'safe_end_date' in globals() and globals()['safe_end_date'] is not None:
            safe_end_date = globals()['safe_end_date']
            if not isinstance(safe_end_date, pd.Timestamp):
                safe_end_date = pd.Timestamp(safe_end_date)
            position_combined = position_combined[position_combined.index <= safe_end_date]

        return position_combined, overall_params
    except Exception as e:
        print(f"合併策略函數發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}


# ============================================================================
# 第九部分：🎯 優化版評估函數 - 加入胃納量懲罰 + 樣本外測試
# ============================================================================
def evaluate_strategy(individual):
    """
    評估策略的適應度函數 - 高胃納量優化版
    🔬 只使用訓練期數據進行評估，防止過擬合
    """
    try:
        # 🔬 只使用訓練期數據進行評估
        position_combined, overall_params = combined_strategy_twelve(
            individual,
            start_date_str=TRAIN_START,
            end_date_str=TRAIN_END
        )

        if position_combined.empty:
            return -999.0,

        有效持倉數 = (position_combined.sum(axis=1) > 0).sum()
        if 有效持倉數 == 0:
            return -999.0,

        # 🎯 計算實際胃納量
        position_capacity = (position_combined * capacity).sum(axis=1)
        avg_capacity = position_capacity[position_capacity > 0].mean()

        # 如果平均胃納量低於目標，給予懲罰
        capacity_penalty = 0
        if pd.notna(avg_capacity) and avg_capacity < MIN_CAPACITY:
            capacity_penalty = (MIN_CAPACITY - avg_capacity) / MIN_CAPACITY * CAPACITY_PENALTY_WEIGHT

        # 🔥 應用 3% 持股約束
        position_combined = position_weight_mgr.normalize_position(position_combined)

        report = sim(
            position=position_combined,
            stop_loss=overall_params['stop_loss'],
            trail_stop=overall_params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=overall_params['trade_at_price'],
            position_limit=overall_params['position_limit'],
            take_profit=overall_params['take_profit'],
            upload=False
        )

        if report is None:
            return -999.0,

        metrics = report.get_metrics()
        sharpe_ratio = metrics['ratio']['sharpeRatio']

        # 🎯 綜合評分 = 夏普值 - 胃納量懲罰
        final_score = sharpe_ratio - capacity_penalty

        if DIAGNOSTIC_MODE and random.random() < 0.05:
            annual_return = metrics['profitability']['annualReturn']
            max_drawdown = metrics['risk']['maxDrawdown']
            diagnostic_print(f"評估: 夏普={sharpe_ratio:.4f}, 胃納量={avg_capacity/1e6:.1f}M, "
                           f"年化={annual_return*100:.1f}%, 回撤={max_drawdown*100:.1f}%")

        return final_score,

    except Exception as e:
        diagnostic_print(f"評估錯誤: {e}")
        return -999.0,


# ============================================================================
# 🔬 樣本外測試函數 (Out-of-Sample Testing)
# ============================================================================
def run_backtest_period(individual, start_date, end_date, name="Strategy"):
    """
    🔬 指定期間回測（用於樣本外測試）

    Args:
        individual: 基因向量
        start_date: 開始日期
        end_date: 結束日期
        name: 策略名稱
    """
    try:
        position_combined, overall_params = combined_strategy_twelve(
            individual,
            start_date_str=start_date,
            end_date_str=end_date
        )

        if position_combined.empty:
            return None

        # 計算胃納量
        position_capacity = (position_combined * capacity).sum(axis=1)
        avg_capacity = position_capacity[position_capacity > 0].mean()

        # 🔥 應用 3% 持股約束
        position_combined = position_weight_mgr.normalize_position(position_combined)

        report = sim(
            position=position_combined,
            stop_loss=overall_params['stop_loss'],
            trail_stop=overall_params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=overall_params['trade_at_price'],
            position_limit=overall_params['position_limit'],
            take_profit=overall_params['take_profit'],
            upload=False,
            name=name
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
            'capacity': avg_capacity if pd.notna(avg_capacity) else 0,
            'period': f"{start_date} ~ {end_date}",
            'position': position_combined
        }
    except Exception as e:
        return None


def run_oos_test(individual, name="OOS_Test"):
    """
    🔬 執行樣本外測試（同時回測訓練期和測試期）

    Returns:
        dict: 包含 train_result, test_result, overfit_warning
    """
    # 訓練期回測
    train_result = run_backtest_period(individual, TRAIN_START, TRAIN_END, f"{name}_Train")

    # 測試期回測
    test_result = run_backtest_period(individual, TEST_START, TEST_END, f"{name}_Test")

    if train_result is None or test_result is None:
        return None

    # 計算過擬合指標
    sharpe_ratio = test_result['sharpe'] / train_result['sharpe'] if train_result['sharpe'] > 0 else 0
    return_ratio = test_result['annual_return'] / train_result['annual_return'] if train_result['annual_return'] > 0 else 0

    # 過擬合警告
    overfit_warnings = []
    if sharpe_ratio < OVERFIT_SHARPE_RATIO and train_result['sharpe'] > 1.0:
        overfit_warnings.append(f"⚠️ 夏普衰減嚴重: {sharpe_ratio:.1%} (測試/訓練)")
    if return_ratio < OVERFIT_RETURN_RATIO and train_result['annual_return'] > 0.2:
        overfit_warnings.append(f"⚠️ 報酬衰減嚴重: {return_ratio:.1%} (測試/訓練)")

    return {
        'train': train_result,
        'test': test_result,
        'sharpe_ratio': sharpe_ratio,
        'return_ratio': return_ratio,
        'overfit_warnings': overfit_warnings,
        'is_overfit': len(overfit_warnings) > 0
    }


def run_detailed_oos_test(individual, gen):
    """
    🔬 執行詳細樣本外測試（顯示完整回測報告）

    每 N 代執行一次，顯示測試期的詳細回測結果
    """
    try:
        # === 訓練期回測（簡要） ===
        train_position, params = combined_strategy_twelve(
            individual,
            start_date_str=TRAIN_START,
            end_date_str=TRAIN_END
        )
        if train_position.empty or train_position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        train_position = position_weight_mgr.normalize_position(train_position)

        train_report = sim(
            position=train_position,
            stop_loss=params['stop_loss'],
            trail_stop=params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params['trade_at_price'],
            position_limit=params['position_limit'],
            take_profit=params['take_profit'],
            upload=False,
            name=f"天空龍_W{WINDOW_ID}_Gen{gen}_訓練期"
        )

        if train_report is None:
            return None
        train_metrics = train_report.get_metrics()
        train_sharpe = train_metrics['ratio'].get('sharpeRatio', 0) or 0

        # === 測試期回測（詳細顯示） ===
        test_position, params = combined_strategy_twelve(
            individual,
            start_date_str=TEST_START,
            end_date_str=TEST_END
        )
        if test_position.empty or test_position.sum().sum() == 0:
            return None

        # 🔥 應用 3% 持股約束
        test_position = position_weight_mgr.normalize_position(test_position)

        print(f"\n{'='*60}")
        print(f"📊 第 {gen} 代 - 測試期詳細回測 ({TEST_START[:4]}~{TEST_END[:4]})")
        print(f"{'='*60}")

        test_report = sim(
            position=test_position,
            stop_loss=params['stop_loss'],
            trail_stop=params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=params['trade_at_price'],
            position_limit=params['position_limit'],
            take_profit=params['take_profit'],
            upload=False,
            name=f"天空龍_W{WINDOW_ID}_Gen{gen}_測試期"
        )

        if test_report is None:
            return None

        # 顯示詳細報告
        test_report.display()

        test_metrics = test_report.get_metrics()
        test_sharpe = test_metrics['ratio'].get('sharpeRatio', 0) or 0

        # 計算過擬合指標
        sharpe_ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0
        is_overfit = sharpe_ratio < OVERFIT_SHARPE_RATIO and train_sharpe > 1.0

        print(f"\n📈 訓練期夏普: {train_sharpe:.3f}")
        print(f"📈 測試期夏普: {test_sharpe:.3f}")
        print(f"📊 測試/訓練比: {sharpe_ratio:.1%} {'🔴 過擬合警告' if is_overfit else '🟢 通過'}")

        return {
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'sharpe_ratio': sharpe_ratio,
            'is_overfit': is_overfit
        }

    except Exception as e:
        print(f"   ⚠️ 詳細OOS測試失敗: {e}")
        return None


def print_oos_comparison(oos_result):
    """打印樣本外測試比較結果"""
    if oos_result is None:
        print("   ⚠️ 樣本外測試失敗")
        return

    train = oos_result['train']
    test = oos_result['test']

    print(f"\n   {'─'*55}")
    print(f"   📊 樣本外測試結果比較")
    print(f"   {'─'*55}")
    print(f"   {'指標':<12} {'訓練期':>12} {'測試期':>12} {'比率':>10}")
    print(f"   {'':<12} {'('+TRAIN_START[:4]+'~'+TRAIN_END[:4]+')':>12} {'('+TEST_START[:4]+'~'+TEST_END[:4]+')':>12}")
    print(f"   {'─'*55}")
    print(f"   {'夏普值':<10} {train['sharpe']:>12.3f} {test['sharpe']:>12.3f} {oos_result['sharpe_ratio']:>9.1%}")
    print(f"   {'年化報酬':<10} {train['annual_return']*100:>11.1f}% {test['annual_return']*100:>11.1f}% {oos_result['return_ratio']:>9.1%}")
    print(f"   {'最大回檔':<10} {train['max_drawdown']*100:>11.1f}% {test['max_drawdown']*100:>11.1f}%")
    print(f"   {'胃納量(萬)':<10} {train['capacity']/1e4:>12.0f} {test['capacity']/1e4:>12.0f}")
    print(f"   {'─'*55}")

    if oos_result['is_overfit']:
        print(f"   🔴 過擬合警告:")
        for warning in oos_result['overfit_warnings']:
            print(f"      {warning}")
    else:
        print(f"   🟢 通過樣本外測試！測試期表現穩定")


def perform_oos_backtest(individual, label, generation):
    """執行詳細樣本外測試並顯示完整回測報告"""
    print("\n" + "🔬"*30)
    print(f"🔬 {label} - 詳細樣本外測試 (第 {generation} 代)")
    print("🔬"*30)

    # 使用新的詳細測試函數（顯示完整回測報告）
    oos_result = run_detailed_oos_test(individual, generation)

    if oos_result:
        # Discord 通知
        if 'send_discord_notification' in globals():
            overfit_status = "🔴 過擬合警告" if oos_result['is_overfit'] else "🟢 測試通過"
            send_discord_notification(
                f"🔬 天空龍 視窗{WINDOW_ID} 第{generation}代 OOS測試",
                f"訓練期夏普: {oos_result['train_sharpe']:.3f}\n"
                f"測試期夏普: {oos_result['test_sharpe']:.3f}\n"
                f"比率: {oos_result['sharpe_ratio']:.1%} {overfit_status}"
            )

        return oos_result
    else:
        print("   ⚠️ 詳細樣本外測試失敗")
        return None


# ============================================================================
# 第十部分：完整回測函數
# ============================================================================
def perform_full_backtest(individual, label, sharpe_value):
    """執行完整回測並顯示詳細報告"""
    print("\n" + "🏆"*30)
    print(f"🏆 {label} - 執行完整回測 🏆")
    print(f"   夏普值: {sharpe_value:.4f}")
    print("🏆"*30 + "\n")

    try:
        position_combined, overall_params = combined_strategy_twelve(individual)

        # 🎯 計算胃納量
        position_capacity = (position_combined * capacity).sum(axis=1)
        avg_capacity = position_capacity[position_capacity > 0].mean()
        print(f"📊 平均胃納量: {avg_capacity/1e6:.2f}M TWD")

        # 🔥 應用 3% 持股約束
        position_combined = position_weight_mgr.normalize_position(position_combined)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report = sim(
            position=position_combined,
            stop_loss=overall_params['stop_loss'],
            trail_stop=overall_params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=overall_params['trade_at_price'],
            position_limit=overall_params['position_limit'],
            take_profit=overall_params['take_profit'],
            name=f"{label}_視窗{WINDOW_ID}_夏普{sharpe_value:.4f}_胃納{avg_capacity/1e6:.0f}M",
            upload=True
        )

        if report:
            print("\n📊 完整回測報告：")
            report.display()

            result_data = {
                'individual': individual,
                'fitness': sharpe_value,
                'label': label,
                'window_id': WINDOW_ID,
                'avg_capacity': avg_capacity,
                'timestamp': datetime.datetime.now(),
                'report': report,
                'metrics': report.get_metrics()
            }

            safe_label = label.replace(" ", "_").replace("/", "-")
            result_file = f'{safe_label}_window_{WINDOW_ID}_sharpe{sharpe_value:.4f}_cap{avg_capacity/1e6:.0f}M_{timestamp_str}.pkl'
            with open(result_file, 'wb') as f:
                pickle.dump(result_data, f)

            print(f"\n✅ 回測結果已儲存: {result_file}")

            metrics = report.get_metrics()
            print(f"\n📈 關鍵績效指標：")
            print(f"   夏普比率: {metrics['ratio']['sharpeRatio']:.4f}")
            print(f"   年化報酬: {metrics['profitability']['annualReturn']*100:.2f}%")
            print(f"   最大回撤: {metrics['risk']['maxDrawdown']*100:.2f}%")
            print(f"   平均胃納量: {avg_capacity/1e6:.2f}M TWD ({'✅ 達標' if avg_capacity >= MIN_CAPACITY else '❌ 未達標'})")

            allocation, *_ = gene_to_params_twelve(individual)
            strategies = [
                "低波動本益比", "小資族", "營收股價雙渦輪", "高殖利率烏龜",
                "低波動性指標", "藏獒外掛大盤指針", "監獄兔", "精選強勢股",
                "純技術分析", "膽小貓", "合約負債建築工", "研發魔人"
            ]
            print(f"\n💼 策略資金配置：")
            for name, weight in zip(strategies, allocation):
                if weight > 0.01:
                    print(f"   {name}: {weight:.2%}")

            return True
        else:
            print("⚠️ 回測報告生成失敗")
            return False

    except Exception as e:
        print(f"⚠️ 完整回測執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 第十一部分：載入歷史最佳
# ============================================================================
def load_historical_best():
    """載入歷史最佳參數（前十名）"""
    print("\n=== 搜尋歷史最佳檢查點 ===")

    best_individuals = []

    # 使用 IN_COLAB 變數判斷（支援 subprocess）
    if not IN_COLAB or DRIVE_PATH is None:
        print("⚠️ 非 Colab 環境或無 Drive 路徑，跳過歷史最佳搜尋")
        return best_individuals

    def extract_from_halloffame(cp_temp):
        if "halloffame" not in cp_temp or not cp_temp["halloffame"]:
            return []
        results = []
        for ind in cp_temp["halloffame"][:5]:
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                continue
            try:
                ind_list = list(ind)
                if len(ind_list) < 160:
                    continue
            except:
                continue
            results.append({'individual': ind, 'fitness': ind.fitness.values[0]})
        return results

    def extract_from_population(cp_temp):
        if "population" not in cp_temp or not cp_temp["population"]:
            return []
        valid_pop = []
        for ind in cp_temp["population"]:
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                continue
            try:
                ind_list = list(ind)
                if len(ind_list) < 160:
                    continue
            except:
                continue
            valid_pop.append({'individual': ind, 'fitness': ind.fitness.values[0]})
        valid_pop.sort(key=lambda x: x['fitness'], reverse=True)
        return valid_pop[:5]

    search_paths = [BASE_PATH, DRIVE_PATH] + ADDITIONAL_SEARCH_PATHS
    search_paths = [path for path in search_paths if path and os.path.exists(path)]

    print(f"將搜尋以下 {len(search_paths)} 個路徑：")
    for i, path in enumerate(search_paths, 1):
        print(f"  {i}. {path}")

    all_checkpoints = []

    for search_path in search_paths:
        pattern = os.path.join(search_path, "**", "*.pkl")
        matching_files = glob.glob(pattern, recursive=True)

        path_name = os.path.basename(search_path)
        print(f"\n🔍 搜尋 {path_name}: 找到 {len(matching_files)} 個檔案")

        for file in matching_files:
            filename = os.path.basename(file)
            if any(skip in filename.lower() for skip in ['temp', '~', '.tmp']):
                continue

            try:
                with open(file, 'rb') as f:
                    cp_temp = pickle.load(f)

                extracted = []
                extracted.extend(extract_from_halloffame(cp_temp))
                if not extracted:
                    extracted.extend(extract_from_population(cp_temp))

                if extracted:
                    for ind_data in extracted:
                        all_checkpoints.append({
                            'individual': ind_data['individual'],
                            'fitness': ind_data['fitness'],
                            'source': filename,
                            'path': os.path.dirname(file)
                        })
            except:
                continue

    if all_checkpoints:
        all_checkpoints.sort(key=lambda x: x['fitness'], reverse=True)

        unique_checkpoints = []
        seen_fitness = set()
        for cp in all_checkpoints:
            fitness_key = round(cp['fitness'], 6)
            if fitness_key not in seen_fitness:
                seen_fitness.add(fitness_key)
                unique_checkpoints.append(cp)

        best_10 = unique_checkpoints[:10]

        print(f"\n✅ 找到 {len(all_checkpoints)} 個歷史個體（去重後 {len(unique_checkpoints)} 個），選擇前10名：")
        print(f"{'排名':<6}{'夏普值':<12}{'來源檔案':<50}")
        print("-" * 70)

        for i, item in enumerate(best_10, 1):
            source_short = item['source'][:47] + "..." if len(item['source']) > 50 else item['source']
            print(f"{i:<6}{item['fitness']:<12.4f}{source_short:<50}")
            best_individuals.append(item['individual'])

        print(f"\n📊 統計資訊：")
        print(f"  - 最佳夏普值: {best_10[0]['fitness']:.4f}")
        print(f"  - 平均夏普值: {np.mean([x['fitness'] for x in best_10]):.4f}")
    else:
        print("\n⚠️ 未找到任何可用的歷史個體")

    return best_individuals


# ============================================================================
# 🔄 重啟時回測歷史前五名
# ============================================================================
def backtest_top_n_historical(individuals, n=TOP_N_HISTORICAL):
    """重啟時回測歷史前 N 名個體"""
    if not individuals:
        print("⚠️ 沒有歷史個體可供回測")
        return []

    print("\n" + "🏆"*30)
    print(f"🏆 重啟時回測歷史前 {n} 名個體 🏆")
    print("🏆"*30)

    results = []
    top_n = individuals[:n]

    for i, ind in enumerate(top_n, 1):
        print(f"\n{'='*60}")
        print(f"📊 回測第 {i} 名歷史個體")
        print(f"{'='*60}")

        try:
            # 確保個體有正確長度
            if len(ind) < 251:
                ind = list(ind) + [random.uniform(0.1, 10)] * (251 - len(ind))

            position_combined, overall_params = combined_strategy_twelve(ind)

            if position_combined.empty:
                print(f"⚠️ 第 {i} 名：無有效持倉")
                continue

            # 計算胃納量
            position_capacity = (position_combined * capacity).sum(axis=1)
            avg_capacity = position_capacity[position_capacity > 0].mean()

            # 🔥 應用 3% 持股約束
            position_combined = position_weight_mgr.normalize_position(position_combined)

            report = sim(
                position=position_combined,
                stop_loss=overall_params['stop_loss'],
                trail_stop=overall_params['trail_stop'],
                fee_ratio=FEE_RATIO,
                tax_ratio=TAX_RATIO,
                trade_at_price=overall_params['trade_at_price'],
                position_limit=overall_params['position_limit'],
                take_profit=overall_params['take_profit'],
                name=f"歷史第{i}名_胃納{avg_capacity/1e6:.0f}M",
                upload=True
            )

            if report:
                metrics = report.get_metrics()
                sharpe = metrics['ratio']['sharpeRatio']
                annual_return = metrics['profitability']['annualReturn']
                max_drawdown = metrics['risk']['maxDrawdown']

                print(f"\n📈 第 {i} 名績效:")
                print(f"   夏普比率: {sharpe:.4f}")
                print(f"   年化報酬: {annual_return*100:.2f}%")
                print(f"   最大回撤: {max_drawdown*100:.2f}%")
                print(f"   平均胃納量: {avg_capacity/1e6:.2f}M TWD")

                results.append({
                    'rank': i,
                    'individual': ind,
                    'sharpe': sharpe,
                    'annual_return': annual_return,
                    'max_drawdown': max_drawdown,
                    'avg_capacity': avg_capacity,
                    'report': report
                })

        except Exception as e:
            print(f"⚠️ 第 {i} 名回測失敗: {e}")
            continue

    # 總結
    if results:
        print("\n" + "="*60)
        print("📊 歷史前五名回測總結")
        print("="*60)
        print(f"{'排名':<6}{'夏普值':<12}{'年化報酬':<12}{'最大回撤':<12}{'胃納量(M)':<12}")
        print("-" * 54)
        for r in results:
            print(f"{r['rank']:<6}{r['sharpe']:<12.4f}{r['annual_return']*100:<12.2f}{r['max_drawdown']*100:<12.2f}{r['avg_capacity']/1e6:<12.2f}")

    return results


# ============================================================================
# 🔄 定期回測函數 (每 N 代執行)
# ============================================================================
def periodic_backtest(halloffame, generation, label="定期回測"):
    """每 N 代執行一次詳細回測"""
    if not halloffame or len(halloffame) == 0:
        return None

    best_ind = halloffame[0]
    best_fitness = best_ind.fitness.values[0] if best_ind.fitness.valid else -999

    print("\n" + "📊"*20)
    print(f"📊 第 {generation} 代定期回測 (每 {BACKTEST_EVERY_N_GEN} 代)")
    print(f"   當前最佳夏普值: {best_fitness:.4f}")
    print("📊"*20)

    try:
        position_combined, overall_params = combined_strategy_twelve(best_ind)

        if position_combined.empty:
            print("⚠️ 無有效持倉，跳過回測")
            return None

        # 計算胃納量
        position_capacity = (position_combined * capacity).sum(axis=1)
        avg_capacity = position_capacity[position_capacity > 0].mean()

        # 🔥 應用 3% 持股約束
        position_combined = position_weight_mgr.normalize_position(position_combined)

        report = sim(
            position=position_combined,
            stop_loss=overall_params['stop_loss'],
            trail_stop=overall_params['trail_stop'],
            fee_ratio=FEE_RATIO,
            tax_ratio=TAX_RATIO,
            trade_at_price=overall_params['trade_at_price'],
            position_limit=overall_params['position_limit'],
            take_profit=overall_params['take_profit'],
            name=f"第{generation}代_{label}_夏普{best_fitness:.4f}",
            upload=True
        )

        if report:
            metrics = report.get_metrics()
            print(f"\n📈 第 {generation} 代績效報告:")
            print(f"   夏普比率: {metrics['ratio']['sharpeRatio']:.4f}")
            print(f"   年化報酬: {metrics['profitability']['annualReturn']*100:.2f}%")
            print(f"   最大回撤: {metrics['risk']['maxDrawdown']*100:.2f}%")
            print(f"   平均胃納量: {avg_capacity/1e6:.2f}M TWD ({'✅' if avg_capacity >= MIN_CAPACITY else '❌'})")

            # 顯示策略配置
            allocation, *_ = gene_to_params_twelve(best_ind)
            strategies = [
                "低波動本益比", "小資族", "營收股價雙渦輪", "高殖利率烏龜",
                "低波動性指標", "藏獒外掛大盤指針", "監獄兔", "精選強勢股",
                "純技術分析", "膽小貓", "合約負債建築工", "研發魔人"
            ]
            print(f"\n💼 策略配置 (權重 > 5%):")
            for name, weight in zip(strategies, allocation):
                if weight > 0.05:
                    print(f"   {name}: {weight:.1%}")

            return report

    except Exception as e:
        print(f"⚠️ 定期回測失敗: {e}")
        return None


# ============================================================================
# 第十二部分：🎯 智能初始化族群 - 使用合適的參數範圍
# ============================================================================
def create_smart_individual(toolbox):
    """創建智能初始化的個體"""
    gene = []

    # 策略權重 (0-11)
    for _ in range(12):
        gene.append(smart_gene_init('allocation'))

    # 策略1參數 (12-20)
    gene.append(smart_gene_init('ratio'))      # rev_ma3_ma12_ratio
    gene.append(smart_gene_init('ratio'))      # rev_consistency
    gene.append(smart_gene_init('ratio'))      # volatility_threshold
    gene.append(smart_gene_init('ratio'))      # margin_usage_limit
    gene.append(smart_gene_init('ratio'))      # non_op_income_limit
    gene.append(smart_gene_init('volume'))     # min_volume
    gene.append(smart_gene_init('pe'))         # pe_min
    gene.append(smart_gene_init('pe'))         # pe_max
    gene.append(smart_gene_init('top_n'))      # top_n

    # 策略2參數 (21-28)
    gene.append(smart_gene_init('market_cap')) # market_value_limit
    gene.append(smart_gene_init('ratio'))      # market_rev_ratio_limit
    gene.append(smart_gene_init('growth'))     # rev_yoy_growth_limit
    gene.append(smart_gene_init('growth'))     # rev_mom_growth_limit
    gene.append(smart_gene_init('period'))     # rsv_period
    gene.append(smart_gene_init('period'))     # ma_period
    gene.append(smart_gene_init('volume'))     # volume_threshold
    gene.append(smart_gene_init('top_n'))      # top_n

    # 策略3參數 (29-36)
    gene.append(smart_gene_init('period'))     # rev_ma_period
    gene.append(smart_gene_init('period'))     # rev_ma_lookback
    gene.append(smart_gene_init('period'))     # price_high_window
    gene.append(smart_gene_init('volume'))     # min_volume
    gene.append(smart_gene_init('ratio'))      # min_price
    gene.append(smart_gene_init('ratio'))      # rsi_threshold
    gene.append(smart_gene_init('pe'))         # pe_limit
    gene.append(smart_gene_init('top_n'))      # top_n

    # 繼續填充剩餘基因至251個
    while len(gene) < 251:
        gene.append(random.uniform(0.1, 10))

    return creator.Individual(gene)


def initialize_population_with_best(toolbox, pop_size=100):
    """初始化族群，包含歷史最佳個體"""
    historical_best = load_historical_best()

    if historical_best and len(historical_best) > 0:
        print(f"\n✅ 成功載入 {len(historical_best)} 個歷史最佳個體")

        population = []
        for ind in historical_best:
            new_ind = toolbox.clone(ind)
            if len(new_ind) < 251:
                new_ind.extend([random.uniform(0.1, 10)] * (251 - len(new_ind)))
            elif len(new_ind) > 251:
                new_ind = creator.Individual(new_ind[:251])
            population.append(new_ind)

        num_variations = min(40, pop_size // 2)
        for _ in range(num_variations):
            base_ind = random.choice(historical_best)
            mutated_ind = toolbox.clone(base_ind)
            if len(mutated_ind) < 251:
                mutated_ind.extend([random.uniform(0.1, 10)] * (251 - len(mutated_ind)))
            elif len(mutated_ind) > 251:
                mutated_ind = creator.Individual(mutated_ind[:251])
            toolbox.mutate(mutated_ind)
            del mutated_ind.fitness.values
            population.append(mutated_ind)

        remaining = pop_size - len(population)
        if remaining > 0:
            # 🎯 使用智能初始化
            for _ in range(remaining):
                population.append(create_smart_individual(toolbox))

        print(f"\n📦 族群組成（總計 {len(population)} 個）：")
        print(f"  - 歷史最佳: {len(historical_best)} 個")
        print(f"  - 變異版本: {num_variations} 個")
        print(f"  - 智能隨機: {remaining} 個")

    else:
        print("⚠️ 未找到歷史最佳個體，使用智能隨機初始化")
        population = [create_smart_individual(toolbox) for _ in range(pop_size)]

    return population


# ============================================================================
# 第十三部分：共享機制
# ============================================================================
def share_best_individuals(population, window_id):
    """將最佳個體分享到共享資料夾"""
    if SHARED_PATH is None:
        return
    try:
        valid_individuals = [ind for ind in population if hasattr(ind, 'fitness') and ind.fitness.valid]
        if not valid_individuals:
            return
        valid_individuals.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        best_5 = valid_individuals[:5]
        shared_file = os.path.join(SHARED_PATH, f'best_window_{window_id}.pkl')
        with ImprovedFileLock(shared_file, timeout=5):
            with open(shared_file, 'wb') as f:
                pickle.dump(best_5, f)
    except:
        pass


def load_shared_best_individuals():
    """載入其他視窗分享的最佳個體"""
    if SHARED_PATH is None:
        return []
    all_shared = []
    for window in [1, 2, 3]:
        if window == WINDOW_ID:
            continue
        shared_file = os.path.join(SHARED_PATH, f'best_window_{window}.pkl')
        if os.path.exists(shared_file):
            try:
                with ImprovedFileLock(shared_file, timeout=5):
                    with open(shared_file, 'rb') as f:
                        shared_individuals = pickle.load(f)
                        all_shared.extend(shared_individuals)
            except:
                continue
    return all_shared


# ============================================================================
# 第十四部分：檢查點管理（升級版 - 使用 SafeFileManager）
# ============================================================================
class CheckpointManager:
    """Checkpoint 管理器 - 支援斷點續傳"""

    def __init__(self, window_id, checkpoint_path):
        self.window_id = window_id
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_window_{window_id}_latest.pkl')
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    def save(self, population, generation, halloffame=None, history=None):
        """保存 checkpoint（使用 SafeFileManager）"""
        try:
            checkpoint_data = {
                'population': [{'genes': list(ind), 'fitness': ind.fitness.values[0] if ind.fitness.valid else None}
                              for ind in population],
                'generation': generation,
                'halloffame': [{'genes': list(ind), 'fitness': ind.fitness.values[0]}
                              for ind in halloffame] if halloffame else [],
                'history': history or [],
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state(),
                'timestamp': datetime.datetime.now().isoformat(),
                'window_id': self.window_id
            }

            success = SafeFileManager.safe_pickle_save(checkpoint_data, self.checkpoint_file)
            if success:
                print(f"   💾 Checkpoint 已保存 (第 {generation} 代)")

                # 同時保存帶世代號的備份
                gen_file = os.path.join(self.checkpoint_path, f'checkpoint_window_{self.window_id}_gen_{generation}.pkl')
                SafeFileManager.safe_pickle_save(checkpoint_data, gen_file)

            return success

        except Exception as e:
            print(f"   ⚠️ Checkpoint 保存失敗: {e}")
            return False

    def load(self, creator_class):
        """載入 checkpoint"""
        if not os.path.exists(self.checkpoint_file):
            print("ℹ️ 無 checkpoint，從頭開始")
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            if not data.get('population') or not data.get('generation'):
                print("⚠️ Checkpoint 無效，從頭開始")
                return None

            # 恢復隨機狀態
            if data.get('random_state'):
                random.setstate(data['random_state'])
            if data.get('numpy_state'):
                np.random.set_state(data['numpy_state'])

            # 重建族群
            population = []
            for ind_data in data['population']:
                new_ind = creator_class(ind_data['genes'])
                if ind_data['fitness'] is not None:
                    new_ind.fitness.values = (ind_data['fitness'],)
                population.append(new_ind)

            print(f"✅ 從 checkpoint 恢復: 第 {data['generation']} 代, 族群 {len(population)}")
            print(f"   📅 保存時間: {data.get('timestamp', 'N/A')}")

            return {
                'population': population,
                'generation': data['generation'],
                'history': data.get('history', [])
            }

        except Exception as e:
            print(f"⚠️ Checkpoint 載入失敗: {e}")
            return None

    def exists(self):
        """檢查是否有 checkpoint"""
        return os.path.exists(self.checkpoint_file)

# 初始化 CheckpointManager
checkpoint_mgr = CheckpointManager(WINDOW_ID, CHECKPOINT_PATH)
print("✅ Checkpoint 管理器已初始化")


# 保留舊函數以相容
def save_checkpoint(checkpoint_data, window_id, generation):
    """儲存檢查點（舊版相容）"""
    checkpoint_file = os.path.join(CHECKPOINT_PATH, f'checkpoint_window_{window_id}_gen_{generation}.pkl')
    latest_file = os.path.join(CHECKPOINT_PATH, f'checkpoint_window_{window_id}_latest.pkl')

    try:
        success = SafeFileManager.safe_pickle_save(checkpoint_data, checkpoint_file)
        if success:
            SafeFileManager.safe_pickle_save(checkpoint_data, latest_file)
            print(f"   💾 Checkpoint 已保存 (第 {generation} 代)")
    except:
        pass


def load_checkpoint(window_id):
    """載入檢查點（舊版相容）"""
    latest_file = os.path.join(CHECKPOINT_PATH, f'checkpoint_window_{window_id}_latest.pkl')
    if not os.path.exists(latest_file):
        return None
    try:
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    except:
        return None


# ============================================================================
# 第十五部分：主要基因演算法執行函數
# ============================================================================
def run_genetic_algorithm():
    """執行基因演算法優化 - 高胃納量版"""

    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_gene():
        return random.uniform(0.1, 10)

    toolbox.register("attr_float", init_gene)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=251)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", lambda x: creator.Individual(x[:]))

    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 追蹤歷史最佳夏普值
    historical_best_sharpe = -999.0
    historical_best_individual = None

    cp = load_checkpoint(WINDOW_ID)

    if cp:
        print(f"📂 載入進行中的檢查點")
        population = cp['population']
        start_gen = cp['generation']
        halloffame = cp['halloffame']
        logbook = cp['logbook']
        best_fitness_history = cp.get('best_fitness', [])
        stagnation_count = cp.get('stagnation_count', 0)
        historical_best_sharpe = cp.get('historical_best_sharpe', -999.0)
        historical_best_individual = cp.get('historical_best_individual', None)

        print(f"從第 {start_gen} 代繼續")
        print(f"歷史最佳夏普值: {historical_best_sharpe:.4f}")

    else:
        print(f"🔄 初始化新族群（高胃納量優化版）")

        population = initialize_population_with_best(toolbox, pop_size=100)

        start_gen = 0
        halloffame = tools.HallOfFame(10)
        logbook = tools.Logbook()
        best_fitness_history = []
        stagnation_count = 0

        # 載入歷史最佳時立即回測
        print("\n" + "="*60)
        print("🔍 搜尋並回測歷史最佳個體")
        print("="*60)

        historical_best_loaded = load_historical_best()
        if historical_best_loaded:
            best_loaded_ind = None
            best_loaded_sharpe = -999.0

            for ind in historical_best_loaded:
                if hasattr(ind, 'fitness') and ind.fitness.valid:
                    halloffame.update([ind])
                    if ind.fitness.values[0] > best_loaded_sharpe:
                        best_loaded_sharpe = ind.fitness.values[0]
                        best_loaded_ind = ind

            if best_loaded_ind is not None and best_loaded_sharpe > -999.0:
                historical_best_sharpe = best_loaded_sharpe
                historical_best_individual = best_loaded_ind

                print(f"\n📊 歷史最佳夏普值: {historical_best_sharpe:.4f}")

                # 🔄 重啟時回測歷史前五名
                print(f"\n🔄 執行歷史前 {TOP_N_HISTORICAL} 名完整回測...")
                backtest_top_n_historical(historical_best_loaded, TOP_N_HISTORICAL)

                print(f"\n✅ 歷史最佳基準已建立")
                print(f"   目標：超越 {historical_best_sharpe:.4f}")

    if start_gen == 0:
        print("\n" + "="*60)
        print("📊 評估初始族群...")
        print("="*60)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        total_count = len(invalid_ind)
        print(f"🔢 待評估個體數量: {total_count}")

        # 評估第一個個體來估計時間
        eval_start = time.time()
        first_fitness = toolbox.evaluate(invalid_ind[0])
        invalid_ind[0].fitness.values = first_fitness
        first_eval_time = time.time() - eval_start
        estimated_total = first_eval_time * (total_count - 1)
        print(f"⏱️ 單個個體評估時間: {first_eval_time:.1f}秒")
        print(f"⏱️ 預估總時間: {estimated_total/60:.1f}分鐘")
        print("-"*60)

        # 使用 tqdm 顯示進度評估剩餘個體
        remaining_ind = invalid_ind[1:]
        fitnesses = []
        for ind in tqdm(remaining_ind, desc="🧬 評估進度", unit="個體"):
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
            fitnesses.append(fit)

        eval_total_time = time.time() - eval_start
        print(f"\n✅ 初始族群評估完成！耗時: {eval_total_time/60:.1f}分鐘")

        valid_population = [ind for ind in population if ind.fitness.values[0] > -999]
        if not valid_population:
            print("⚠️ 警告：所有初始個體都無效！")
            return None, -999

        best_initial = max(valid_population, key=lambda ind: ind.fitness.values[0])
        print(f"🏆 初始族群最佳夏普值: {best_initial.fitness.values[0]:.4f}")

        if best_initial.fitness.values[0] > historical_best_sharpe:
            print(f"\n🎉 初始族群中發現超越歷史的個體！")
            perform_full_backtest(best_initial, "初始族群新高", best_initial.fitness.values[0])
            historical_best_sharpe = best_initial.fitness.values[0]
            historical_best_individual = best_initial

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    NGEN = 1000

    for gen in range(start_gen, NGEN):
        print(f"\n{'='*60}")
        print(f"🧬 視窗 {WINDOW_ID} - 第 {gen+1} 代 | 目標: 夏普{TARGET_SHARPE}+ 胃納{MIN_CAPACITY/1e6:.0f}M+")
        print(f"{'='*60}")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        elite_size = len(population) // 10
        elite = tools.selBest(population, elite_size)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        mutation_rate = 0.3 if stagnation_count < 3 else 0.5
        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # 使用 tqdm 顯示評估進度
        gen_eval_start = time.time()
        for ind in tqdm(invalid_ind, desc=f"🧬 第{gen+1}代評估", unit="個體", leave=False):
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
        gen_eval_time = time.time() - gen_eval_start
        print(f"   評估 {len(invalid_ind)} 個體，耗時 {gen_eval_time:.1f}秒")

        population[:] = elite + offspring[:-elite_size]

        halloffame.update(population)
        record = stats.compile(population)
        logbook.append(record)

        best_fitness = record['max']
        best_fitness_history.append(best_fitness)

        # 創造歷史新高時自動回測
        if best_fitness > historical_best_sharpe:
            improvement = best_fitness - historical_best_sharpe
            print(f"\n🎉🎉🎉 創造歷史新高！🎉🎉🎉")
            print(f"   歷史最佳: {historical_best_sharpe:.4f}")
            print(f"   新紀錄: {best_fitness:.4f}")
            print(f"   提升: +{improvement:.4f}")

            best_ind = halloffame[0]
            perform_full_backtest(best_ind, f"第{gen+1}代歷史新高", best_fitness)

            historical_best_sharpe = best_fitness
            historical_best_individual = best_ind

        elif len(best_fitness_history) > 1 and best_fitness > best_fitness_history[-2]:
            print(f"🎉 本次運行新紀錄！夏普值: {best_fitness:.4f}")
            print(f"   （距離歷史最佳還差 {historical_best_sharpe - best_fitness:.4f}）")

        print(f"📊 統計: 平均={record['avg']:.4f}, 最大={best_fitness:.4f}")
        print(f"   距離目標: {TARGET_SHARPE - best_fitness:.4f}")
        print(f"   距離歷史最佳: {historical_best_sharpe - best_fitness:.4f}")

        if best_fitness >= TARGET_SHARPE:
            print(f"\n🎯 達到目標夏普值 {TARGET_SHARPE}！")
            if best_fitness > historical_best_sharpe:
                best_ind = halloffame[0]
                perform_full_backtest(best_ind, f"第{gen+1}代達標", best_fitness)
            break

        # 停滯處理
        if len(best_fitness_history) > STAGNATION_THRESHOLD:
            recent_best = best_fitness_history[-STAGNATION_THRESHOLD:]
            if max(recent_best) - min(recent_best) < 0.01:
                stagnation_count += 1
                print(f"⚠️ 檢測到停滯 (連續 {stagnation_count} 代)")

                if stagnation_count >= STAGNATION_THRESHOLD:
                    print(f"🔄 執行基因重組...")

                    historical_best_individuals = load_historical_best()
                    if historical_best_individuals:
                        new_individuals = []
                        for hist_ind in historical_best_individuals:
                            for _ in range(3):
                                mutated = toolbox.clone(hist_ind)
                                if len(mutated) < 251:
                                    mutated.extend([random.uniform(0.1, 10)] * (251 - len(mutated)))
                                elif len(mutated) > 251:
                                    mutated = creator.Individual(mutated[:251])
                                tools.mutGaussian(mutated, mu=0, sigma=2, indpb=0.3)
                                del mutated.fitness.values
                                new_individuals.append(mutated)

                        population.sort(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -1000)
                        num_to_replace = min(len(new_individuals), len(population)//3)
                        population[:num_to_replace] = new_individuals[:num_to_replace]
                        print(f"💉 注入 {num_to_replace} 個變異個體")

                    shared_best = load_shared_best_individuals()
                    if shared_best:
                        num_to_add = min(len(shared_best), 10)
                        population[-num_to_add:] = shared_best[:num_to_add]
                        print(f"💉 注入 {num_to_add} 個來自其他視窗的個體")

                    stagnation_count = 0
            else:
                stagnation_count = 0

        if (gen + 1) % 10 == 0:
            share_best_individuals(population, WINDOW_ID)

        # 🔥 每 CROSS_WINDOW_INTERVAL 代執行3視窗交互
        if (gen + 1) % CROSS_WINDOW_INTERVAL == 0:
            population = cross_window_mgr.perform_cross_window_exchange(population, gen + 1, creator.Individual)
            # 重新評估新注入的個體
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

        # 🔄 每 N 代執行定期回測
        if (gen + 1) % BACKTEST_EVERY_N_GEN == 0:
            periodic_backtest(halloffame, gen + 1, "定期檢查")

            # 🔬 每 N 代執行樣本外測試
            if len(halloffame) > 0:
                best_individual = halloffame[0]
                perform_oos_backtest(best_individual, "最佳個體", gen + 1)

        # 🔥 每 5 代保存 checkpoint（使用新 CheckpointManager）
        if (gen + 1) % 5 == 0:
            checkpoint_mgr.save(
                population=population,
                generation=gen + 1,
                halloffame=list(halloffame),
                history=best_fitness_history
            )
            # 同時保存舊格式（相容性）
            cp = {
                'population': population,
                'generation': gen + 1,
                'halloffame': halloffame,
                'logbook': logbook,
                'best_fitness': best_fitness_history,
                'stagnation_count': stagnation_count,
                'historical_best_sharpe': historical_best_sharpe,
                'historical_best_individual': historical_best_individual,
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state()
            }
            save_checkpoint(cp, WINDOW_ID, gen + 1)

    # 最終回測
    print("\n" + "="*60)
    print("最終回測最佳個體")
    print("="*60)

    best_ind = halloffame[0]
    position_combined, overall_params = combined_strategy_twelve(best_ind)

    # 🔥 應用 3% 持股約束
    position_combined = position_weight_mgr.normalize_position(position_combined)

    # 🔥 上傳到 FinLab（覆蓋舊版本）
    report = sim(
        position=position_combined,
        stop_loss=overall_params['stop_loss'],
        trail_stop=overall_params['trail_stop'],
        fee_ratio=FEE_RATIO,
        tax_ratio=TAX_RATIO,
        trade_at_price=overall_params['trade_at_price'],
        position_limit=overall_params['position_limit'],
        take_profit=overall_params['take_profit'],
        name=f"天空龍_視窗{WINDOW_ID}_最佳策略",
        upload=True
    )

    if report:
        report.display()
        print(f"\n✅ 最佳策略已上傳到 FinLab: 天空龍_視窗{WINDOW_ID}_最佳策略")

    # 🔬 最終樣本外測試
    print("\n" + "="*60)
    print("🔬 最終樣本外測試 - 檢測過擬合")
    print("="*60)

    final_oos = run_oos_test(best_ind, name="天空龍_Final")
    if final_oos:
        print_oos_comparison(final_oos)

        # Discord 通知
        if 'send_discord_notification' in globals():
            overfit_status = "🔴 過擬合警告" if final_oos['is_overfit'] else "🟢 測試通過"
            send_discord_notification(
                f"🎉 視窗{WINDOW_ID} 演化完成 - 樣本外測試",
                f"📊 訓練期({TRAIN_START[:4]}~{TRAIN_END[:4]}):\n"
                f"  夏普: {final_oos['train']['sharpe']:.3f}\n"
                f"  年化: {final_oos['train']['annual_return']*100:.1f}%\n"
                f"🔬 測試期({TEST_START[:4]}~{TEST_END[:4]}):\n"
                f"  夏普: {final_oos['test']['sharpe']:.3f}\n"
                f"  年化: {final_oos['test']['annual_return']*100:.1f}%\n"
                f"比率: {final_oos['sharpe_ratio']:.1%}\n"
                f"{overfit_status}"
            )

    final_result = {
        'best_individual': best_ind,
        'best_fitness': best_ind.fitness.values[0],
        'halloffame': halloffame,
        'logbook': logbook,
        'window_id': WINDOW_ID,
        'historical_best_sharpe': historical_best_sharpe,
        'min_capacity': MIN_CAPACITY,
        'timestamp': datetime.datetime.now(),
        'oos_result': final_oos  # 🔬 保存樣本外測試結果
    }

    result_file = f'final_result_twelve_high_capacity_window_{WINDOW_ID}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(final_result, f)

    print(f"\n✅ 優化完成！最終結果已儲存至: {result_file}")
    print(f"📊 歷史最佳夏普值: {historical_best_sharpe:.4f}")
    print(f"📊 最低胃納量要求: {MIN_CAPACITY/1e6:.0f}M TWD")
    if final_oos:
        print(f"🔬 樣本外測試: {'🟢 通過' if not final_oos['is_overfit'] else '🔴 過擬合警告'}")

    return best_ind, best_ind.fitness.values[0]


# ============================================================================
# 第十六部分：主程式
# ============================================================================
if __name__ == "__main__":
    try:
        print(f"\n{'='*80}")
        print(f"🚀 十二組合快快龍 - 高胃納量優化版")
        print(f"🎯 目標夏普值: {TARGET_SHARPE}")
        print(f"💰 最低胃納量: {MIN_CAPACITY/1e6:.0f}M TWD")
        print(f"🖥️ 視窗編號: {WINDOW_ID}")
        print(f"{'='*80}\n")

        # 設定回測期間
        safe_end_date_data = pd.Timestamp(datetime.datetime.now().strftime('%Y-%m-%d'))
        effective_start_date = pd.Timestamp("2014-01-01")
        effective_end_date = safe_end_date_data

        globals()['safe_end_date'] = effective_end_date
        globals()['safe_start_date'] = effective_start_date

        print(f"回測期間: {effective_start_date.strftime('%Y-%m-%d')} 至 {effective_end_date.strftime('%Y-%m-%d')}\n")

        best_individual, best_fitness = run_genetic_algorithm()

        print(f"\n{'='*80}")
        print(f"🏁 視窗 {WINDOW_ID} 優化完成總結")
        print(f"{'='*80}")
        print(f"最佳夏普值: {best_fitness:.4f}")
        print(f"最低胃納量: {MIN_CAPACITY/1e6:.0f}M TWD")

        if best_fitness >= TARGET_SHARPE:
            print(f"✅ 成功達到目標夏普值 {TARGET_SHARPE}！")
        else:
            print(f"⚠️ 未達到目標夏普值，當前最佳: {best_fitness:.4f}")

        allocation, *_ = gene_to_params_twelve(best_individual)
        strategies = [
            "低波動本益比", "小資族", "營收股價雙渦輪", "高殖利率烏龜",
            "低波動性指標", "藏獒外掛大盤指針", "監獄兔", "精選強勢股",
            "純技術分析", "膽小貓", "合約負債建築工", "研發魔人"
        ]

        print("\n💼 最佳策略資金配置:")
        for name, weight in zip(strategies, allocation):
            if weight > 0.01:
                print(f"   {name}: {weight:.2%}")

        print(f"\n✅ 高胃納量優化版執行完成！\n")

    except KeyboardInterrupt:
        print(f"\n視窗 {WINDOW_ID} 收到中斷信號")
        sys.exit(0)
    except Exception as e:
        print(f"\n視窗 {WINDOW_ID} 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
