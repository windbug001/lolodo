#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 九組合媽媽龍 Transformer v1.0 - 完整可執行版本
Nine Combo Mama Dragon Transformer - Full Implementation
================================================================================

【執行環境】Google Colab Pro+ (GPU T4/V100/A100)

【執行方式】
1. 在 Colab 左側面板點選 🔑 (Secrets)
2. 新增 FINLAB_API_KEY，貼上你的 API Key
3. 將此程式碼全部複製貼上到 Colab cell 中執行

================================================================================
"""

from __future__ import annotations

import os
import math
import time
import json
import pickle
import warnings
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# 第一部分：環境設定
# =============================================================================
print("🔧 設定環境...")

# 安裝套件
def install_packages():
    import subprocess
    packages = ['torch', 'finlab']
    for pkg in packages:
        try:
            __import__(pkg.split('[')[0])
        except ImportError:
            print(f"   安裝 {pkg}...")
            subprocess.run(['pip', 'install', pkg, '-q'], check=True)

install_packages()

# 載入 PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 設定 device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU 模式: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("⚠️ CPU 模式（建議使用 GPU）")

# 載入 FinLab
import finlab
from finlab import data
from finlab.backtest import sim

# FinLab 登入 - 從 Colab Secrets 取得 API Key
from google.colab import userdata
FINLAB_API_KEY = userdata.get('FINLAB_API_KEY')
finlab.login(FINLAB_API_KEY)
print(f"✅ FinLab 登入成功")

# =============================================================================
# 第二部分：配置
# =============================================================================
class Config:
    """模型與訓練配置"""

    # 模型架構
    d_model = 128           # 嵌入維度（減小以加速訓練）
    n_heads = 4             # 注意力頭數
    n_layers = 4            # Transformer 層數
    d_ff = 512              # FFN 維度
    dropout = 0.1
    max_stocks = 1500

    # 特徵數量
    n_features = 50         # 總特徵數

    # 選股設定
    top_k = 15              # 選取 Top-K 股票
    position_limit = 0.15   # 單股最大持倉

    # 訓練設定
    batch_size = 16
    learning_rate = 5e-4
    weight_decay = 1e-4
    epochs = 50
    patience = 8

    # 回測設定
    train_start = '2017-01-01'
    train_end = '2022-12-31'
    test_start = '2023-01-01'

    # 市場狀態
    market_states = 3       # BEAR=0, RANGE=1, BULL=2

config = Config()

# =============================================================================
# 第三部分：數據載入
# =============================================================================
class FinLabDataLoader:
    """FinLab 數據載入器"""

    _cache = {}

    @classmethod
    def load_all(cls):
        """載入所有需要的數據"""
        if cls._cache:
            return

        print("📦 載入 FinLab 數據...")
        start = time.time()

        # 價格數據
        cls._cache['close'] = data.get('price:收盤價')
        cls._cache['open'] = data.get('price:開盤價')
        cls._cache['high'] = data.get('price:最高價')
        cls._cache['low'] = data.get('price:最低價')
        cls._cache['volume'] = data.get('price:成交股數')

        # 估值數據
        cls._cache['pe'] = data.get('price_earning_ratio:本益比')
        cls._cache['pb'] = data.get('price_earning_ratio:股價淨值比')
        cls._cache['dividend'] = data.get('price_earning_ratio:殖利率(%)')

        # 營收數據
        cls._cache['rev'] = data.get('monthly_revenue:當月營收')
        cls._cache['rev_yoy'] = data.get('monthly_revenue:去年同月增減(%)')

        # 基本面
        cls._cache['roe'] = data.get('fundamental_features:ROE綜合損益')
        cls._cache['gpm'] = data.get('fundamental_features:營業毛利率')
        cls._cache['npm'] = data.get('fundamental_features:稅後淨利率')
        cls._cache['oig'] = data.get('fundamental_features:營業利益成長率')

        # 籌碼
        cls._cache['margin'] = data.get('margin_transactions:融資使用率')

        # 市值
        cls._cache['market_cap'] = data.get('etl:market_value')

        print(f"✅ 數據載入完成 ({time.time()-start:.1f}s)")

    @classmethod
    def get(cls, key: str) -> pd.DataFrame:
        if not cls._cache:
            cls.load_all()
        return cls._cache.get(key)


# =============================================================================
# 第四部分：特徵工程
# =============================================================================
class FeatureExtractor:
    """股票特徵提取器"""

    def __init__(self):
        FinLabDataLoader.load_all()
        self.close = FinLabDataLoader.get('close')
        self.feature_names = []

    def extract_all_features(self, lookback: int = 60) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
        """
        提取所有日期的特徵

        Returns:
            features: (n_dates, n_stocks, n_features)
            stock_ids: 股票列表
            dates: 日期索引
        """
        print("🔧 提取特徵...")
        start = time.time()

        # 輔助函數：確保 DataFrame 的 index 是 DatetimeIndex
        def ensure_datetime_index(df):
            if df is None:
                return None
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df

        # 載入並轉換所有數據的 index
        close = ensure_datetime_index(FinLabDataLoader.get('close'))
        high = ensure_datetime_index(FinLabDataLoader.get('high'))
        low = ensure_datetime_index(FinLabDataLoader.get('low'))
        volume = ensure_datetime_index(FinLabDataLoader.get('volume'))
        pe = ensure_datetime_index(FinLabDataLoader.get('pe'))
        pb = ensure_datetime_index(FinLabDataLoader.get('pb'))
        dividend = ensure_datetime_index(FinLabDataLoader.get('dividend'))
        rev = ensure_datetime_index(FinLabDataLoader.get('rev'))
        rev_yoy = ensure_datetime_index(FinLabDataLoader.get('rev_yoy'))
        roe = ensure_datetime_index(FinLabDataLoader.get('roe'))
        gpm = ensure_datetime_index(FinLabDataLoader.get('gpm'))
        npm = ensure_datetime_index(FinLabDataLoader.get('npm'))
        margin = ensure_datetime_index(FinLabDataLoader.get('margin'))
        market_cap = ensure_datetime_index(FinLabDataLoader.get('market_cap'))

        # 找共同股票
        common_stocks = close.columns
        n_stocks = len(common_stocks)

        # 計算衍生特徵
        returns_1d = close.pct_change(1)
        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)
        returns_60d = close.pct_change(60)

        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()

        volatility_20d = returns_1d.rolling(20).std()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # 價格位置
        high_60 = close.rolling(60).max()
        low_60 = close.rolling(60).min()
        price_position = (close - low_60) / (high_60 - low_60 + 1e-10)

        # 對齊月營收數據到日頻率（使用 ffill 前向填充）
        rev_daily = rev.reindex(close.index, method='ffill')
        rev_yoy_daily = rev_yoy.reindex(close.index, method='ffill')
        rev_ma3 = rev_daily.rolling(3, min_periods=1).mean()
        rev_ma12 = rev_daily.rolling(12, min_periods=1).mean()

        # 對齊其他低頻數據到日頻率
        pe_daily = pe.reindex(close.index, method='ffill') if pe is not None else None
        pb_daily = pb.reindex(close.index, method='ffill') if pb is not None else None
        dividend_daily = dividend.reindex(close.index, method='ffill') if dividend is not None else None
        roe_daily = roe.reindex(close.index, method='ffill') if roe is not None else None
        gpm_daily = gpm.reindex(close.index, method='ffill') if gpm is not None else None
        npm_daily = npm.reindex(close.index, method='ffill') if npm is not None else None
        margin_daily = margin.reindex(close.index, method='ffill') if margin is not None else None
        market_cap_daily = market_cap.reindex(close.index, method='ffill') if market_cap is not None else None

        # 建立特徵字典
        feature_dict = {
            # 報酬率 (4)
            'ret_1d': returns_1d,
            'ret_5d': returns_5d,
            'ret_20d': returns_20d,
            'ret_60d': returns_60d,

            # 均線偏離 (3)
            'ma5_dev': close / ma5 - 1,
            'ma20_dev': close / ma20 - 1,
            'ma60_dev': close / ma60 - 1,

            # 均線趨勢 (2)
            'ma5_gt_ma20': (ma5 > ma20).astype(float),
            'ma20_gt_ma60': (ma20 > ma60).astype(float),

            # 波動率 (2)
            'volatility': volatility_20d,
            'volatility_rank': volatility_20d.rank(axis=1, pct=True),

            # 成交量 (3)
            'vol_ma_ratio': volume / vol_ma20,
            'vol_trend': vol_ma5 / vol_ma20,
            'vol_rank': volume.rank(axis=1, pct=True),

            # RSI (3)
            'rsi': rsi / 100,
            'rsi_oversold': (rsi < 30).astype(float),
            'rsi_overbought': (rsi > 70).astype(float),

            # 價格位置 (3)
            'price_position': price_position,
            'new_high_20': (close >= close.rolling(20).max()).astype(float),
            'new_high_60': (close >= close.rolling(60).max()).astype(float),

            # 估值 (6)
            'pe_rank': pe_daily.rank(axis=1, pct=True) if pe_daily is not None else None,
            'pb_rank': pb_daily.rank(axis=1, pct=True) if pb_daily is not None else None,
            'dividend_rank': dividend_daily.rank(axis=1, pct=True) if dividend_daily is not None else None,
            'pe_valid': ((pe_daily > 0) & (pe_daily < 100)).astype(float) if pe_daily is not None else None,
            'pb_valid': ((pb_daily > 0) & (pb_daily < 10)).astype(float) if pb_daily is not None else None,
            'high_dividend': (dividend_daily > 3).astype(float) if dividend_daily is not None else None,

            # 營收 (4)
            'rev_yoy_rank': rev_yoy_daily.rank(axis=1, pct=True),
            'rev_positive': (rev_yoy_daily > 0).astype(float),
            'rev_strong': (rev_yoy_daily > 20).astype(float),
            'rev_ma3_ma12': rev_ma3 / (rev_ma12 + 1e-10),

            # 獲利能力 (4)
            'roe_rank': roe_daily.rank(axis=1, pct=True) if roe_daily is not None else None,
            'gpm_rank': gpm_daily.rank(axis=1, pct=True) if gpm_daily is not None else None,
            'npm_rank': npm_daily.rank(axis=1, pct=True) if npm_daily is not None else None,
            'roe_positive': (roe_daily > 0).astype(float) if roe_daily is not None else None,

            # 籌碼 (2)
            'margin_rank': margin_daily.rank(axis=1, pct=True) if margin_daily is not None else None,
            'margin_low': (margin_daily < 30).astype(float) if margin_daily is not None else None,

            # 市值 (2)
            'market_cap_rank': market_cap_daily.rank(axis=1, pct=True) if market_cap_daily is not None else None,
            'small_cap': (market_cap_daily.rank(axis=1, pct=True) < 0.3).astype(float) if market_cap_daily is not None else None,
        }

        # 過濾掉 None 值的特徵
        feature_dict = {k: v for k, v in feature_dict.items() if v is not None}

        self.feature_names = list(feature_dict.keys())
        n_features = len(self.feature_names)

        # 對齊所有特徵到共同日期
        common_dates = close.index[lookback:]
        n_dates = len(common_dates)

        print(f"   特徵數: {n_features}, 日期數: {n_dates}, 股票數: {n_stocks}")

        # 構建特徵張量
        features = np.zeros((n_dates, n_stocks, n_features), dtype=np.float32)

        for i, (name, df) in enumerate(feature_dict.items()):
            if df is not None:
                # 確保 df 的 index 是 DatetimeIndex
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                # 對齊
                aligned = df.reindex(index=common_dates, columns=common_stocks)
                # 標準化
                aligned = (aligned - aligned.mean()) / (aligned.std() + 1e-10)
                # 裁剪
                aligned = aligned.clip(-5, 5)
                # 填充
                aligned = aligned.fillna(0)
                features[:, :, i] = aligned.values

        print(f"✅ 特徵提取完成 ({time.time()-start:.1f}s)")
        print(f"   特徵張量形狀: {features.shape}")

        return features, list(common_stocks), common_dates


# =============================================================================
# 第五部分：計算標籤（未來報酬）
# =============================================================================
def compute_labels(close: pd.DataFrame, dates: pd.DatetimeIndex,
                   horizon: int = 20, top_pct: float = 0.1) -> np.ndarray:
    """
    計算標籤：未來 horizon 天的報酬率是否為前 top_pct%

    Returns:
        labels: (n_dates, n_stocks) 二元標籤
    """
    print(f"📊 計算標籤（預測 {horizon} 天後報酬）...")

    # 確保 index 是 DatetimeIndex
    close = close.copy()
    close.index = pd.to_datetime(close.index)

    # 計算未來報酬
    future_returns = close.pct_change(horizon).shift(-horizon)

    # 對齊
    aligned = future_returns.reindex(index=dates)

    # 計算每日的報酬排名，取前 top_pct% 為正例
    labels = (aligned.rank(axis=1, pct=True) > (1 - top_pct)).astype(float)
    labels = labels.fillna(0).values

    print(f"   標籤形狀: {labels.shape}")
    print(f"   正例比例: {labels.mean()*100:.1f}%")

    return labels


def compute_future_returns(close: pd.DataFrame, dates: pd.DatetimeIndex,
                           horizon: int = 20) -> np.ndarray:
    """計算未來報酬（用於 Sharpe 損失）"""
    # 確保 index 是 DatetimeIndex
    close = close.copy()
    close.index = pd.to_datetime(close.index)

    future_returns = close.pct_change(1).shift(-1)
    aligned = future_returns.reindex(index=dates)

    # 構建 (n_dates, n_stocks, horizon) 張量
    n_dates = len(dates)
    n_stocks = len(close.columns)

    returns_tensor = np.zeros((n_dates, n_stocks, horizon), dtype=np.float32)

    for i in range(n_dates):
        if i + horizon < len(close):
            end_idx = dates[i]
            # 獲取未來 horizon 天的報酬
            future_slice = future_returns.loc[dates[i]:].iloc[:horizon]
            if len(future_slice) == horizon:
                returns_tensor[i] = future_slice.values.T

    return returns_tensor


def compute_market_state(close: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    計算市場狀態

    Returns:
        market_state: (n_dates,) 0=BEAR, 1=RANGE, 2=BULL
    """
    print("📊 計算市場狀態...")

    # 確保 index 是 DatetimeIndex
    close = close.copy()
    close.index = pd.to_datetime(close.index)

    # 使用市場平均價格
    market_avg = close.mean(axis=1)

    ma40 = market_avg.rolling(40).mean()
    ma120 = market_avg.rolling(120).mean()
    trend = ma40.pct_change(20)

    # 判斷狀態
    bull = (market_avg > ma40) & (ma40 > ma120) & (trend > 0.02)
    bear = (market_avg < ma40) & (ma40 < ma120) & (trend < -0.02)

    market_state = np.ones(len(close), dtype=np.int64)  # 預設 RANGE
    market_state[bull] = 2  # BULL
    market_state[bear] = 0  # BEAR

    # 對齊到 dates
    state_series = pd.Series(market_state, index=close.index)
    aligned = state_series.reindex(index=dates).fillna(1).astype(np.int64)

    print(f"   BEAR: {(aligned.values==0).sum()}, RANGE: {(aligned.values==1).sum()}, BULL: {(aligned.values==2).sum()}")

    return aligned.values


# =============================================================================
# 第六部分：Transformer 模型
# =============================================================================
class StockTransformer(nn.Module):
    """股票選擇 Transformer"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 特徵嵌入
        self.feature_embed = nn.Sequential(
            nn.Linear(config.n_features, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )

        # 市場狀態嵌入
        self.market_embed = nn.Embedding(config.market_states, config.d_model)

        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # 輸出頭
        self.score_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, features: torch.Tensor, market_state: torch.Tensor):
        """
        Args:
            features: (batch, n_stocks, n_features)
            market_state: (batch,)

        Returns:
            scores: (batch, n_stocks)
        """
        batch_size, n_stocks, _ = features.size()

        # 特徵嵌入
        x = self.feature_embed(features)  # (batch, n_stocks, d_model)

        # 添加市場狀態
        market = self.market_embed(market_state)  # (batch, d_model)
        x = x + market.unsqueeze(1) * 0.1

        # Transformer 編碼
        x = self.transformer(x)  # (batch, n_stocks, d_model)

        # 評分
        scores = self.score_head(x).squeeze(-1)  # (batch, n_stocks)

        return scores

    def get_portfolio(self, features: torch.Tensor, market_state: torch.Tensor,
                      stock_ids: List[str], top_k: int = None) -> Dict[str, float]:
        """獲取投資組合"""
        top_k = top_k or self.config.top_k

        self.eval()
        with torch.no_grad():
            scores = self.forward(features.unsqueeze(0), market_state.unsqueeze(0))
            scores = scores.squeeze(0)

            # Softmax
            weights = F.softmax(scores, dim=-1)

            # Top-K
            top_weights, top_indices = torch.topk(weights, top_k)
            top_weights = top_weights / top_weights.sum()

            portfolio = {}
            for w, idx in zip(top_weights.cpu().numpy(), top_indices.cpu().numpy()):
                stock_id = stock_ids[idx]
                portfolio[stock_id] = min(float(w), self.config.position_limit)

            # 正規化
            total = sum(portfolio.values())
            portfolio = {k: v / total for k, v in portfolio.items()}

            return portfolio


# =============================================================================
# 第七部分：訓練
# =============================================================================
class Trainer:
    """模型訓練器"""

    def __init__(self, model: StockTransformer, config: Config):
        self.model = model.to(device)
        self.config = config

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            features = batch[0].to(device)
            market_state = batch[1].to(device)
            labels = batch[2].to(device)

            scores = self.model(features, market_state)

            # BCE 損失
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch[0].to(device)
                market_state = batch[1].to(device)
                labels = batch[2].to(device)

                scores = self.model(features, market_state)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*60}")
        print(f"🚀 開始訓練 Transformer")
        print(f"{'='*60}")

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f"   ✅ 最佳模型!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\n⚠️ Early stopping")
                    break

        # 載入最佳模型
        self.model.load_state_dict(torch.load('best_model.pt'))
        print(f"\n✅ 訓練完成!")


# =============================================================================
# 第七部分 B：Walk-Forward 驗證器
# =============================================================================
class WalkForwardValidator:
    """
    Walk-Forward 滾動驗證器

    每次訓練使用過去的數據，測試使用未來一年的數據
    這樣可以避免過擬合，得到更真實的回測結果
    """

    def __init__(self, config: Config):
        self.config = config
        self.results = []

    def run(self, features: np.ndarray, labels: np.ndarray,
            market_states: np.ndarray, stock_ids: List[str],
            dates: pd.DatetimeIndex) -> Tuple[List[pd.DataFrame], Dict]:
        """
        執行 Walk-Forward 驗證

        Args:
            features: 特徵張量 (n_dates, n_stocks, n_features)
            labels: 標籤 (n_dates, n_stocks)
            market_states: 市場狀態 (n_dates,)
            stock_ids: 股票列表
            dates: 日期索引

        Returns:
            all_positions: 每個 fold 的持倉 DataFrame 列表
            summary: 驗證摘要
        """
        print(f"\n{'='*60}")
        print(f"🔄 開始 Walk-Forward 驗證")
        print(f"{'='*60}")

        # 定義時間窗口（每年一個 fold）
        years = sorted(set(d.year for d in dates))

        # 至少需要 2 年訓練數據，所以從第 3 年開始測試
        min_train_years = 2
        test_years = years[min_train_years:]

        print(f"📅 可用年份: {years}")
        print(f"📅 測試年份: {test_years}")

        all_positions = []
        fold_results = []

        for fold_idx, test_year in enumerate(test_years):
            print(f"\n{'─'*60}")
            print(f"📊 Fold {fold_idx + 1}/{len(test_years)}: 測試 {test_year} 年")
            print(f"{'─'*60}")

            # 找出訓練和測試的索引
            train_mask = np.array([d.year < test_year for d in dates])
            test_mask = np.array([d.year == test_year for d in dates])

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) < 100 or len(test_indices) < 20:
                print(f"   ⚠️ 數據不足，跳過此 fold")
                continue

            # 分割訓練集為 train/val (80/20)
            val_split = int(len(train_indices) * 0.8)
            train_idx = train_indices[:val_split]
            val_idx = train_indices[val_split:]

            print(f"   訓練: {len(train_idx)} 樣本 ({dates[train_idx[0]].year}-{dates[train_idx[-1]].year})")
            print(f"   驗證: {len(val_idx)} 樣本")
            print(f"   測試: {len(test_indices)} 樣本 ({test_year})")

            # 準備數據
            train_features = features[train_idx]
            train_labels = labels[train_idx]
            train_states = market_states[train_idx]

            val_features = features[val_idx]
            val_labels = labels[val_idx]
            val_states = market_states[val_idx]

            # 創建 DataLoader
            train_dataset = TensorDataset(
                torch.FloatTensor(train_features),
                torch.LongTensor(train_states),
                torch.FloatTensor(train_labels)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(val_features),
                torch.LongTensor(val_states),
                torch.FloatTensor(val_labels)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

            # 創建新模型（每個 fold 從頭訓練）
            model = StockTransformer(self.config)

            # 訓練（減少 epochs 以加速）
            fold_config = Config()
            fold_config.n_features = self.config.n_features
            fold_config.epochs = 30  # 減少訓練輪數
            fold_config.patience = 5

            trainer = Trainer(model, fold_config)
            trainer.fit(train_loader, val_loader)

            # 在測試期間生成持倉
            model.eval()
            positions = []
            valid_dates_fold = []

            with torch.no_grad():
                for i in test_indices[::20]:  # 每月換股
                    if i >= len(features):
                        continue
                    feat = torch.FloatTensor(features[i]).to(device)
                    state = torch.LongTensor([market_states[i]]).to(device)

                    portfolio = model.get_portfolio(feat, state.squeeze(0), stock_ids)

                    pos_series = pd.Series(portfolio, name=dates[i])
                    positions.append(pos_series)
                    valid_dates_fold.append(dates[i])

            if positions:
                # 取得 FinLab 原始收盤價數據（用於確保索引格式一致）
                close = data.get('price:收盤價')
                close_dates = pd.to_datetime(close.index)

                # 將日期對齊到 FinLab 數據的索引格式
                aligned_positions = []
                for pos_series, target_date in zip(positions, valid_dates_fold):
                    date_mask = close_dates <= target_date
                    if date_mask.any():
                        actual_date = close.index[date_mask][-1]
                        pos_series.name = actual_date
                        aligned_positions.append(pos_series)

                if aligned_positions:
                    position_df = pd.concat(aligned_positions, axis=1).T
                    position_df = position_df.fillna(0)
                    all_positions.append(position_df)

                    fold_results.append({
                        'fold': fold_idx + 1,
                        'test_year': test_year,
                        'n_positions': len(aligned_positions),
                        'train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None,
                        'val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else None,
                    })

                    print(f"   ✅ 生成 {len(aligned_positions)} 個持倉決策")

        # 合併所有持倉
        if all_positions:
            combined_positions = pd.concat(all_positions, axis=0)
            combined_positions = combined_positions.sort_index()
        else:
            combined_positions = pd.DataFrame()

        summary = {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'total_positions': len(combined_positions) if not combined_positions.empty else 0,
        }

        print(f"\n{'='*60}")
        print(f"✅ Walk-Forward 驗證完成")
        print(f"   完成 {summary['n_folds']} 個 folds")
        print(f"   總計 {summary['total_positions']} 個持倉決策")
        print(f"{'='*60}")

        return combined_positions, summary


# =============================================================================
# 第八部分：回測（使用 FinLab 官方建議方式）
# =============================================================================
def create_position_from_portfolio(portfolio_dict: Dict[str, float],
                                    close: pd.DataFrame,
                                    date_idx) -> pd.Series:
    """
    從投資組合字典建立符合 FinLab 格式的持倉 Series

    Args:
        portfolio_dict: 股票代碼 -> 權重 的字典
        close: FinLab 的收盤價 DataFrame（用於取得正確的索引格式）
        date_idx: 日期索引（必須是 close.index 中的值）
    """
    # 建立一個與 close 相同 columns 的 Series，值為 0
    pos = pd.Series(0.0, index=close.columns, name=date_idx)

    # 填入投資組合權重
    for stock_id, weight in portfolio_dict.items():
        if stock_id in pos.index:
            pos[stock_id] = weight

    return pos


def run_backtest(model: StockTransformer, features: np.ndarray,
                 market_states: np.ndarray, stock_ids: List[str],
                 dates: pd.DatetimeIndex, start_date: str,
                 name: str = "Transformer_Strategy",
                 upload: bool = True) -> Any:
    """執行回測 - 使用 FinLab 官方建議的方式"""
    print(f"\n📊 執行回測: {name}")

    # 取得 FinLab 原始收盤價數據（保持原始索引格式）
    close = data.get('price:收盤價')

    model.eval()

    # 轉換 start_date 為 Timestamp 以便比較
    start_ts = pd.Timestamp(start_date)

    # 找到開始日期的索引
    start_idx = 0
    for i, d in enumerate(dates):
        if d >= start_ts:
            start_idx = i
            break

    positions = []

    with torch.no_grad():
        for i in range(start_idx, len(dates) - 20, 20):  # 每月換股
            feat = torch.FloatTensor(features[i]).unsqueeze(0).to(device)
            state = torch.LongTensor([market_states[i]]).to(device)

            portfolio = model.get_portfolio(feat.squeeze(0), state.squeeze(0), stock_ids)

            # 找到 close 中對應的日期（確保格式一致）
            target_date = dates[i]
            # 在 close.index 中找最接近的日期
            close_dates = pd.to_datetime(close.index)
            date_mask = close_dates <= target_date
            if date_mask.any():
                actual_date = close.index[date_mask][-1]
                pos = create_position_from_portfolio(portfolio, close, actual_date)
                positions.append(pos)

    if not positions:
        print("   ⚠️ 無有效持倉")
        return None

    # 合併成 DataFrame（索引格式與 FinLab 數據一致）
    position_df = pd.concat(positions, axis=1).T
    position_df = position_df.fillna(0)

    print(f"   持倉 DataFrame: {position_df.shape}")
    print(f"   期間: {position_df.index[0]} ~ {position_df.index[-1]}")

    # 執行 FinLab 回測
    report = sim(
        position=position_df,
        fee_ratio=1.425 / 1000,
        tax_ratio=3 / 1000,
        trade_at_price="open",
        position_limit=config.position_limit,
        stop_loss=0.15,
        trail_stop=0.25,
        take_profit=0.5,
        upload=upload,
        name=name
    )

    return report


# =============================================================================
# 第九部分：主程式
# =============================================================================
def run_backtest_from_positions(position_df: pd.DataFrame, name: str, upload: bool = True) -> Any:
    """從持倉 DataFrame 執行回測"""
    if position_df.empty:
        print("   ⚠️ 無有效持倉")
        return None

    # 取得 FinLab 原始收盤價數據（用於確保索引格式一致）
    close = data.get('price:收盤價')

    # 重新對齊 position_df 的索引到 close 的索引格式
    position_df = position_df.copy()

    # 將 position_df 的日期索引對齊到 close 的索引
    new_index = []
    close_dates = pd.to_datetime(close.index)

    for idx in position_df.index:
        target_date = pd.to_datetime(idx)
        date_mask = close_dates <= target_date
        if date_mask.any():
            actual_date = close.index[date_mask][-1]
            new_index.append(actual_date)
        else:
            new_index.append(close.index[0])

    position_df.index = new_index

    print(f"\n📊 執行回測: {name}")
    print(f"   持倉 DataFrame: {position_df.shape}")
    print(f"   期間: {position_df.index[0]} ~ {position_df.index[-1]}")

    report = sim(
        position=position_df,
        fee_ratio=1.425 / 1000,
        tax_ratio=3 / 1000,
        trade_at_price="open",
        position_limit=config.position_limit,
        stop_loss=0.15,
        trail_stop=0.25,
        take_profit=0.5,
        upload=upload,
        name=name
    )

    return report


def main(use_walk_forward: bool = True):
    """
    主程式

    Args:
        use_walk_forward: 是否使用 Walk-Forward 驗證（預設 True）
                         設為 False 則使用傳統單次訓練方式
    """
    print("""
╔════════════════════════════════════════════════════════════════╗
║     🐉 九組合媽媽龍 Transformer v2.0                           ║
║     Stock Selection with Transformer Architecture              ║
╠════════════════════════════════════════════════════════════════╣
║  📊 模型: Transformer Encoder (4層, 128維)                     ║
║  🎯 目標: 學習選出未來高報酬股票                                ║
║  🔄 驗證: Walk-Forward 滾動驗證（避免過擬合）                   ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # 1. 提取特徵
    extractor = FeatureExtractor()
    features, stock_ids, dates = extractor.extract_all_features()

    # 2. 計算標籤和市場狀態
    close = FinLabDataLoader.get('close')
    labels = compute_labels(close, dates)
    market_states = compute_market_state(close, dates)

    # 更新 config
    config.n_features = features.shape[2]

    if use_walk_forward:
        # =====================================================================
        # Walk-Forward 驗證模式（推薦）
        # =====================================================================
        print("\n🔄 使用 Walk-Forward 驗證模式")

        validator = WalkForwardValidator(config)
        combined_positions, summary = validator.run(
            features, labels, market_states, stock_ids, dates
        )

        # 執行整體回測
        if not combined_positions.empty:
            print("\n" + "="*60)
            print("📊 Walk-Forward 整體回測結果")
            print("="*60)

            report = run_backtest_from_positions(
                combined_positions,
                name="Transformer_MamaDragon_WalkForward",
                upload=True
            )

            if report:
                report.display()

            return None, summary

        else:
            print("⚠️ 無有效的持倉數據")
            return None, summary

    else:
        # =====================================================================
        # 傳統單次訓練模式
        # =====================================================================
        print("\n📊 使用傳統單次訓練模式")

        # 分割訓練/驗證/測試
        train_end_ts = pd.Timestamp(config.train_end)
        train_end_idx = 0
        for i, d in enumerate(dates):
            if d >= train_end_ts:
                train_end_idx = i
                break

        val_split = int(train_end_idx * 0.8)

        train_features = features[:val_split]
        train_labels = labels[:val_split]
        train_states = market_states[:val_split]

        val_features = features[val_split:train_end_idx]
        val_labels = labels[val_split:train_end_idx]
        val_states = market_states[val_split:train_end_idx]

        print(f"\n📊 數據分割:")
        print(f"   訓練: {len(train_features)} 樣本")
        print(f"   驗證: {len(val_features)} 樣本")
        print(f"   測試: {len(features) - train_end_idx} 樣本")

        # 創建 DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_states),
            torch.FloatTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_features),
            torch.LongTensor(val_states),
            torch.FloatTensor(val_labels)
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # 創建模型
        model = StockTransformer(config)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 模型參數: {total_params:,}")

        # 訓練
        trainer = Trainer(model, config)
        trainer.fit(train_loader, val_loader)

        # 樣本內回測
        print("\n" + "="*60)
        print("📊 樣本內回測 (2017-2022) - ⚠️ 僅供參考，可能過擬合")
        print("="*60)

        in_sample_report = run_backtest(
            model, features, market_states, stock_ids, dates,
            start_date=config.train_start,
            name="Transformer_MamaDragon_InSample",
            upload=True
        )

        if in_sample_report:
            in_sample_report.display()

        # 樣本外回測
        print("\n" + "="*60)
        print("📊 樣本外回測 (2023-至今) - 真實表現")
        print("="*60)

        out_sample_report = run_backtest(
            model, features, market_states, stock_ids, dates,
            start_date=config.test_start,
            name="Transformer_MamaDragon_OutSample",
            upload=True
        )

        if out_sample_report:
            out_sample_report.display()

        print("\n✅ 完成!")

        return model, trainer.history


if __name__ == "__main__":
    # 使用 Walk-Forward 驗證（推薦，結果更真實）
    model, result = main(use_walk_forward=True)

    # 如果想用傳統方式，改成：
    # model, history = main(use_walk_forward=False)
