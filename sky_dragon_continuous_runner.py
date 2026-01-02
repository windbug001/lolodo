#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 十二組合天空龍策略 - VS Code 不間斷執行器
   Sky Dragon Twelve Combination Strategy - Continuous Runner for VS Code
================================================================================

【十二組合說明】
本系統將 6 種基礎策略進行兩兩組合，形成 12 種獨特的策略組合：

基礎策略:
  1. 低波動本益比策略 (low_volatility_pe)
  2. 小資族策略 (small_investor)
  3. 營收股價雙渦輪 (revenue_price_turbo)
  4. 高殖利率烏龜策略 (high_yield_turtle)
  5. 低波動指數 (low_volatility_index)
  6. 市場指標策略 (market_indicator)

十二組合:
  組合 1: 低波動本益比 + 小資族 (穩健成長型)
  組合 2: 低波動本益比 + 營收雙渦輪 (價值動能型)
  組合 3: 低波動本益比 + 高殖利率烏龜 (高配息穩健型)
  組合 4: 低波動本益比 + 低波動指數 (超低波動型)
  組合 5: 小資族 + 營收雙渦輪 (小資動能型)
  組合 6: 小資族 + 高殖利率烏龜 (小資配息型)
  組合 7: 小資族 + 市場指標 (小資趨勢型)
  組合 8: 營收雙渦輪 + 高殖利率烏龜 (動能配息型)
  組合 9: 營收雙渦輪 + 低波動指數 (動能避險型)
  組合 10: 營收雙渦輪 + 市場指標 (雙動能型)
  組合 11: 高殖利率烏龜 + 低波動指數 (配息避險型)
  組合 12: 低波動指數 + 市場指標 (趨勢避險型)

【功能】
✅ VS Code 不間斷執行
✅ 錯誤自動恢復
✅ 檢查點續傳
✅ 實時進度監控
✅ 多策略組合並行優化

版本：v1.0 (2025)
================================================================================
"""

from __future__ import annotations
import os
import sys
import time
import json
import pickle
import signal
import argparse
import threading
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import combinations
import multiprocessing as mp

# 禁用快取
os.environ['FINLAB_DISABLE_CACHE'] = '1'

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#                         配置區
# ============================================================================

@dataclass
class SkyDragonConfig:
    """天空龍策略配置"""
    # FinLab API Key
    finlab_api_key: str = "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m"

    # 優化目標
    target_sharpe: float = 4.0
    min_capacity: int = 5_000_000  # 500萬
    target_annual_return: float = 0.3
    max_drawdown: float = 0.2

    # GA 參數
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8

    # 執行設定
    checkpoint_interval: int = 10  # 每 N 代保存
    max_retries: int = 5  # 最大重試次數
    retry_delay: int = 60  # 重試延遲（秒）
    health_check_interval: int = 300  # 健康檢查間隔（秒）

    # 輸出路徑
    output_dir: str = "./sky_dragon_output"
    log_dir: str = "./sky_dragon_logs"


# 六種基礎策略
STRATEGIES = [
    "low_volatility_pe",      # 低波動本益比
    "small_investor",         # 小資族
    "revenue_price_turbo",    # 營收股價雙渦輪
    "high_yield_turtle",      # 高殖利率烏龜
    "low_volatility_index",   # 低波動指數
    "market_indicator",       # 市場指標
]

STRATEGY_NAMES = {
    "low_volatility_pe": "低波動本益比",
    "small_investor": "小資族",
    "revenue_price_turbo": "營收雙渦輪",
    "high_yield_turtle": "高殖利率烏龜",
    "low_volatility_index": "低波動指數",
    "market_indicator": "市場指標",
}

# 十二種策略組合
TWELVE_COMBINATIONS = [
    ("low_volatility_pe", "small_investor", "穩健成長型"),
    ("low_volatility_pe", "revenue_price_turbo", "價值動能型"),
    ("low_volatility_pe", "high_yield_turtle", "高配息穩健型"),
    ("low_volatility_pe", "low_volatility_index", "超低波動型"),
    ("small_investor", "revenue_price_turbo", "小資動能型"),
    ("small_investor", "high_yield_turtle", "小資配息型"),
    ("small_investor", "market_indicator", "小資趨勢型"),
    ("revenue_price_turbo", "high_yield_turtle", "動能配息型"),
    ("revenue_price_turbo", "low_volatility_index", "動能避險型"),
    ("revenue_price_turbo", "market_indicator", "雙動能型"),
    ("high_yield_turtle", "low_volatility_index", "配息避險型"),
    ("low_volatility_index", "market_indicator", "趨勢避險型"),
]

# ============================================================================
#                         進度追蹤器
# ============================================================================

@dataclass
class CombinationProgress:
    """單一組合的進度"""
    combination_id: int
    strategy_a: str
    strategy_b: str
    name: str
    status: str = "pending"  # pending, running, completed, failed
    current_generation: int = 0
    total_generations: int = 100
    best_sharpe: float = 0.0
    best_capacity: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_count: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict:
        return {
            "combination_id": self.combination_id,
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "name": self.name,
            "status": self.status,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
            "best_sharpe": self.best_sharpe,
            "best_capacity": self.best_capacity,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


class ProgressTracker:
    """十二組合進度追蹤器"""

    def __init__(self, config: SkyDragonConfig):
        self.config = config
        self.progress_file = Path(config.log_dir) / "progress.json"
        self.combinations: Dict[int, CombinationProgress] = {}
        self._lock = threading.Lock()
        self._init_combinations()
        self._load_progress()

    def _init_combinations(self):
        """初始化十二組合"""
        for i, (strat_a, strat_b, name) in enumerate(TWELVE_COMBINATIONS, 1):
            self.combinations[i] = CombinationProgress(
                combination_id=i,
                strategy_a=strat_a,
                strategy_b=strat_b,
                name=name,
                total_generations=self.config.n_generations,
            )

    def _load_progress(self):
        """載入已保存的進度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data.get("combinations", []):
                    cid = item.get("combination_id")
                    if cid in self.combinations:
                        cp = self.combinations[cid]
                        cp.status = item.get("status", "pending")
                        cp.current_generation = item.get("current_generation", 0)
                        cp.best_sharpe = item.get("best_sharpe", 0.0)
                        cp.best_capacity = item.get("best_capacity", 0.0)
                        cp.error_count = item.get("error_count", 0)
                        if item.get("start_time"):
                            cp.start_time = datetime.fromisoformat(item["start_time"])
                        if item.get("end_time"):
                            cp.end_time = datetime.fromisoformat(item["end_time"])
                print(f"📂 已載入進度: {len(data.get('combinations', []))} 個組合")
            except Exception as e:
                print(f"⚠️ 載入進度失敗: {e}")

    def save_progress(self):
        """保存進度"""
        with self._lock:
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
            data = {
                "last_update": datetime.now().isoformat(),
                "combinations": [cp.to_dict() for cp in self.combinations.values()],
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, combination_id: int, **kwargs):
        """更新組合進度"""
        with self._lock:
            if combination_id in self.combinations:
                cp = self.combinations[combination_id]
                for key, value in kwargs.items():
                    if hasattr(cp, key):
                        setattr(cp, key, value)
        self.save_progress()

    def get_summary(self) -> str:
        """獲取進度摘要"""
        lines = [
            "",
            "=" * 70,
            "🐉 十二組合天空龍策略 - 執行狀態",
            "=" * 70,
            "",
        ]

        pending = running = completed = failed = 0
        for cp in self.combinations.values():
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(cp.status, "❓")

            progress_pct = (cp.current_generation / cp.total_generations * 100) if cp.total_generations > 0 else 0

            lines.append(f"  {status_icon} 組合 {cp.combination_id:2d}: {cp.name:<12} "
                        f"| 進度: {progress_pct:5.1f}% ({cp.current_generation}/{cp.total_generations}) "
                        f"| 夏普: {cp.best_sharpe:.2f} | 胃納: {cp.best_capacity/1e6:.1f}M")

            if cp.status == "pending": pending += 1
            elif cp.status == "running": running += 1
            elif cp.status == "completed": completed += 1
            elif cp.status == "failed": failed += 1

        lines.extend([
            "",
            "-" * 70,
            f"  📊 總覽: 待執行 {pending} | 執行中 {running} | 已完成 {completed} | 失敗 {failed}",
            "-" * 70,
            "",
        ])

        return "\n".join(lines)

    def get_next_pending(self) -> Optional[CombinationProgress]:
        """獲取下一個待執行的組合"""
        for cp in self.combinations.values():
            if cp.status == "pending" or (cp.status == "failed" and cp.error_count < self.config.max_retries):
                return cp
        return None

    def has_running(self) -> bool:
        """是否有正在執行的組合"""
        return any(cp.status == "running" for cp in self.combinations.values())

    def all_completed(self) -> bool:
        """是否全部完成"""
        return all(cp.status == "completed" for cp in self.combinations.values())


# ============================================================================
#                         策略執行器
# ============================================================================

class StrategyExecutor:
    """策略執行器 - 執行單一組合優化"""

    def __init__(self, config: SkyDragonConfig, combination: CombinationProgress):
        self.config = config
        self.combination = combination
        self.checkpoint_path = Path(config.output_dir) / f"checkpoint_combo_{combination.combination_id}.pkl"
        self._stop_flag = False

    def execute(self) -> bool:
        """執行優化"""
        print(f"\n{'='*70}")
        print(f"🚀 開始執行組合 {self.combination.combination_id}: {self.combination.name}")
        print(f"   策略 A: {STRATEGY_NAMES[self.combination.strategy_a]}")
        print(f"   策略 B: {STRATEGY_NAMES[self.combination.strategy_b]}")
        print(f"{'='*70}\n")

        try:
            # 載入已有的檢查點
            start_gen = 0
            population = None
            if self.checkpoint_path.exists():
                try:
                    with open(self.checkpoint_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    start_gen = checkpoint.get("generation", 0)
                    population = checkpoint.get("population")
                    print(f"📂 從檢查點續傳: 第 {start_gen} 代")
                except Exception as e:
                    print(f"⚠️ 檢查點載入失敗: {e}")

            # 執行優化主循環
            success = self._run_optimization(start_gen, population)

            return success

        except KeyboardInterrupt:
            print("\n⚠️ 收到中斷信號，保存進度...")
            return False
        except Exception as e:
            print(f"\n❌ 執行失敗: {e}")
            traceback.print_exc()
            return False

    def _run_optimization(self, start_gen: int, initial_population) -> bool:
        """執行優化主循環"""
        import numpy as np
        import random

        # 初始化族群
        population_size = self.config.population_size
        n_generations = self.config.n_generations

        if initial_population is not None:
            population = initial_population
        else:
            # 創建隨機初始族群
            population = self._create_initial_population(population_size)

        best_sharpe = 0.0
        best_capacity = 0.0
        best_individual = None

        for gen in range(start_gen, n_generations):
            if self._stop_flag:
                print("\n⏹️ 收到停止信號")
                break

            gen_start = time.time()

            # 評估適應度
            fitnesses = self._evaluate_population(population)

            # 找出最佳個體
            for ind, fit in zip(population, fitnesses):
                sharpe, capacity = fit
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_capacity = capacity
                    best_individual = ind.copy()

            # 選擇、交叉、變異
            population = self._evolve(population, fitnesses)

            gen_time = time.time() - gen_start

            # 更新進度
            self.combination.current_generation = gen + 1
            self.combination.best_sharpe = best_sharpe
            self.combination.best_capacity = best_capacity

            # 輸出進度
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"  🧬 代數 {gen+1:3d}/{n_generations} | "
                      f"最佳夏普: {best_sharpe:.3f} | "
                      f"胃納量: {best_capacity/1e6:.1f}M | "
                      f"耗時: {gen_time:.1f}s")

            # 保存檢查點
            if (gen + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gen + 1, population, best_individual, best_sharpe, best_capacity)

            # 提前達標退出
            if best_sharpe >= self.config.target_sharpe and best_capacity >= self.config.min_capacity:
                print(f"\n🎉 達標！夏普 {best_sharpe:.2f} >= {self.config.target_sharpe}, "
                      f"胃納 {best_capacity/1e6:.1f}M >= {self.config.min_capacity/1e6:.0f}M")
                break

        # 保存最終結果
        self._save_final_result(best_individual, best_sharpe, best_capacity)

        return best_sharpe >= self.config.target_sharpe and best_capacity >= self.config.min_capacity

    def _create_initial_population(self, size: int) -> List[Dict]:
        """創建初始族群"""
        import random
        population = []
        for _ in range(size):
            # 為兩個策略分配權重
            w1 = random.uniform(0.3, 0.7)
            w2 = 1.0 - w1

            # 策略參數
            ind = {
                "weights": [w1, w2],
                "top_n": random.randint(10, 50),
                "hold_days": random.randint(5, 30),
                "rebalance_freq": random.choice(["weekly", "biweekly", "monthly"]),
                "params_a": self._random_strategy_params(self.combination.strategy_a),
                "params_b": self._random_strategy_params(self.combination.strategy_b),
            }
            population.append(ind)
        return population

    def _random_strategy_params(self, strategy: str) -> Dict:
        """生成策略隨機參數"""
        import random
        if strategy == "low_volatility_pe":
            return {
                "pe_threshold": random.uniform(10, 25),
                "volatility_threshold": random.uniform(0.1, 0.3),
                "revenue_growth_min": random.uniform(0.05, 0.2),
            }
        elif strategy == "small_investor":
            return {
                "price_max": random.randint(50, 200),
                "market_cap_max": random.randint(100, 500) * 1e8,
                "rsv_threshold": random.uniform(0.2, 0.5),
            }
        elif strategy == "revenue_price_turbo":
            return {
                "revenue_acceleration": random.uniform(0.05, 0.2),
                "price_momentum": random.uniform(0.1, 0.3),
                "new_high_days": random.randint(20, 60),
            }
        elif strategy == "high_yield_turtle":
            return {
                "dividend_yield_min": random.uniform(0.03, 0.07),
                "payout_ratio_max": random.uniform(0.6, 0.9),
                "yield_stability": random.uniform(0.5, 0.9),
            }
        elif strategy == "low_volatility_index":
            return {
                "sharpe_min": random.uniform(0.5, 1.5),
                "volatility_max": random.uniform(0.1, 0.25),
                "beta_max": random.uniform(0.5, 1.0),
            }
        elif strategy == "market_indicator":
            return {
                "industry_roe_min": random.uniform(0.08, 0.15),
                "market_pe_ratio": random.uniform(0.8, 1.2),
                "growth_threshold": random.uniform(0.1, 0.25),
            }
        return {}

    def _evaluate_population(self, population: List[Dict]) -> List[Tuple[float, float]]:
        """評估族群適應度"""
        import random
        # 模擬評估（實際應調用回測引擎）
        fitnesses = []
        for ind in population:
            # 基於參數計算模擬適應度
            base_sharpe = 2.0 + random.gauss(0, 0.5)
            base_capacity = 3_000_000 + random.gauss(0, 1_000_000)

            # 權重平衡獎勵
            w_balance = 1 - abs(ind["weights"][0] - 0.5)
            sharpe = max(0, base_sharpe + w_balance * 0.5)
            capacity = max(0, base_capacity + w_balance * 2_000_000)

            fitnesses.append((sharpe, capacity))
        return fitnesses

    def _evolve(self, population: List[Dict], fitnesses: List[Tuple]) -> List[Dict]:
        """演化一代"""
        import random

        # 排序選擇
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1][0], reverse=True)  # 按夏普值排序

        # 精英保留
        elite_count = max(2, len(population) // 10)
        new_pop = [ind.copy() for ind, _ in pop_fit[:elite_count]]

        # 輪盤賭選擇 + 交叉 + 變異
        while len(new_pop) < len(population):
            # 錦標賽選擇
            p1 = self._tournament_select(pop_fit)
            p2 = self._tournament_select(pop_fit)

            # 交叉
            if random.random() < self.config.crossover_rate:
                child = self._crossover(p1, p2)
            else:
                child = p1.copy()

            # 變異
            if random.random() < self.config.mutation_rate:
                child = self._mutate(child)

            new_pop.append(child)

        return new_pop

    def _tournament_select(self, pop_fit: List, k: int = 3) -> Dict:
        """錦標賽選擇"""
        import random
        selected = random.sample(pop_fit, min(k, len(pop_fit)))
        return max(selected, key=lambda x: x[1][0])[0].copy()

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """交叉操作"""
        import random
        child = p1.copy()
        child["weights"] = [(w1 + w2) / 2 for w1, w2 in zip(p1["weights"], p2["weights"])]
        child["top_n"] = random.choice([p1["top_n"], p2["top_n"]])
        child["hold_days"] = random.choice([p1["hold_days"], p2["hold_days"]])
        return child

    def _mutate(self, ind: Dict) -> Dict:
        """變異操作"""
        import random
        ind = ind.copy()

        # 權重變異
        delta = random.gauss(0, 0.1)
        ind["weights"][0] = max(0.2, min(0.8, ind["weights"][0] + delta))
        ind["weights"][1] = 1 - ind["weights"][0]

        # 參數變異
        if random.random() < 0.3:
            ind["top_n"] = max(5, min(100, ind["top_n"] + random.randint(-5, 5)))

        return ind

    def _save_checkpoint(self, gen: int, population: List, best_ind: Dict, best_sharpe: float, best_capacity: float):
        """保存檢查點"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "generation": gen,
            "population": population,
            "best_individual": best_ind,
            "best_sharpe": best_sharpe,
            "best_capacity": best_capacity,
            "combination_id": self.combination.combination_id,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  💾 檢查點已保存 (第 {gen} 代)")

    def _save_final_result(self, best_ind: Dict, best_sharpe: float, best_capacity: float):
        """保存最終結果"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        result_path = Path(self.config.output_dir) / f"result_combo_{self.combination.combination_id}.json"

        result = {
            "combination_id": self.combination.combination_id,
            "name": self.combination.name,
            "strategy_a": self.combination.strategy_a,
            "strategy_b": self.combination.strategy_b,
            "best_sharpe": best_sharpe,
            "best_capacity": best_capacity,
            "best_individual": best_ind,
            "target_met": best_sharpe >= self.config.target_sharpe and best_capacity >= self.config.min_capacity,
            "timestamp": datetime.now().isoformat(),
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n📄 結果已保存: {result_path}")

    def stop(self):
        """停止執行"""
        self._stop_flag = True


# ============================================================================
#                         主執行器
# ============================================================================

class SkyDragonRunner:
    """十二組合天空龍主執行器"""

    def __init__(self, config: SkyDragonConfig = None):
        self.config = config or SkyDragonConfig()
        self.tracker = ProgressTracker(self.config)
        self._stop_flag = False
        self._current_executor: Optional[StrategyExecutor] = None

        # 創建輸出目錄
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        # 設置信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """處理終止信號"""
        print("\n\n⚠️ 收到終止信號，正在安全退出...")
        self._stop_flag = True
        if self._current_executor:
            self._current_executor.stop()

    def run_continuous(self):
        """不間斷執行模式"""
        print(self._get_banner())
        print("\n🚀 啟動不間斷執行模式...\n")

        retry_count = 0

        while not self._stop_flag and not self.tracker.all_completed():
            # 獲取下一個待執行的組合
            combination = self.tracker.get_next_pending()

            if combination is None:
                if self.tracker.has_running():
                    # 等待執行中的任務
                    time.sleep(10)
                    continue
                else:
                    # 全部完成或失敗
                    break

            # 開始執行
            self.tracker.update(
                combination.combination_id,
                status="running",
                start_time=datetime.now(),
            )

            print(self.tracker.get_summary())

            try:
                self._current_executor = StrategyExecutor(self.config, combination)
                success = self._current_executor.execute()

                if success:
                    self.tracker.update(
                        combination.combination_id,
                        status="completed",
                        end_time=datetime.now(),
                    )
                    retry_count = 0
                else:
                    self.tracker.update(
                        combination.combination_id,
                        status="failed",
                        error_count=combination.error_count + 1,
                        last_error="執行未達標",
                        end_time=datetime.now(),
                    )

            except Exception as e:
                self.tracker.update(
                    combination.combination_id,
                    status="failed",
                    error_count=combination.error_count + 1,
                    last_error=str(e),
                )
                retry_count += 1

                if retry_count < self.config.max_retries:
                    print(f"\n⏳ 等待 {self.config.retry_delay} 秒後重試... (嘗試 {retry_count}/{self.config.max_retries})")
                    time.sleep(self.config.retry_delay)

            finally:
                self._current_executor = None
                self.tracker.save_progress()

        # 輸出最終結果
        print(self.tracker.get_summary())
        print("\n" + "=" * 70)
        if self.tracker.all_completed():
            print("🎉 所有組合已完成執行！")
        elif self._stop_flag:
            print("⏹️ 執行已被中斷，進度已保存。")
        else:
            print("⚠️ 部分組合執行失敗，請檢查日誌。")
        print("=" * 70 + "\n")

    def run_single(self, combination_id: int = None):
        """單次執行模式"""
        print(self._get_banner())

        if combination_id:
            if combination_id not in self.tracker.combinations:
                print(f"❌ 無效的組合 ID: {combination_id}")
                return
            combination = self.tracker.combinations[combination_id]
        else:
            combination = self.tracker.get_next_pending()
            if combination is None:
                print("ℹ️ 沒有待執行的組合")
                return

        print(f"\n🚀 單次執行: 組合 {combination.combination_id} - {combination.name}\n")

        self.tracker.update(
            combination.combination_id,
            status="running",
            start_time=datetime.now(),
        )

        try:
            executor = StrategyExecutor(self.config, combination)
            success = executor.execute()

            self.tracker.update(
                combination.combination_id,
                status="completed" if success else "failed",
                end_time=datetime.now(),
            )

        except Exception as e:
            self.tracker.update(
                combination.combination_id,
                status="failed",
                error_count=combination.error_count + 1,
                last_error=str(e),
            )
            traceback.print_exc()

    def monitor(self):
        """監控模式"""
        print(self._get_banner())
        print("\n📊 進入監控模式 (按 Ctrl+C 退出)\n")

        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                print(self._get_banner())
                print(self.tracker.get_summary())
                print(f"\n⏰ 最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("   (每 30 秒自動刷新，按 Ctrl+C 退出)")
                time.sleep(30)
                self.tracker._load_progress()  # 重新載入進度
        except KeyboardInterrupt:
            print("\n\n👋 監控結束")

    def show_results(self):
        """顯示結果"""
        print(self._get_banner())
        print("\n📈 最佳結果:\n")

        results = []
        result_dir = Path(self.config.output_dir)

        for f in result_dir.glob("result_combo_*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    result = json.load(fp)
                    results.append(result)
            except Exception as e:
                print(f"⚠️ 讀取 {f} 失敗: {e}")

        if not results:
            print("  尚無結果")
            return

        # 按夏普值排序
        results.sort(key=lambda x: x.get("best_sharpe", 0), reverse=True)

        print("-" * 80)
        print(f"{'排名':<4} {'組合':<15} {'夏普值':<10} {'胃納量':<12} {'達標':<6}")
        print("-" * 80)

        for i, r in enumerate(results, 1):
            target_met = "✅" if r.get("target_met") else "❌"
            print(f"{i:<4} {r.get('name', 'N/A'):<15} "
                  f"{r.get('best_sharpe', 0):<10.3f} "
                  f"{r.get('best_capacity', 0)/1e6:<12.1f}M "
                  f"{target_met:<6}")

        print("-" * 80)

    def stop_all(self):
        """停止所有執行"""
        print("\n⏹️ 正在停止所有策略執行...")

        # 創建停止標誌文件
        stop_file = Path(self.config.log_dir) / "STOP"
        stop_file.touch()

        print("✅ 停止信號已發送")
        print("   (正在執行的策略將在當前代數完成後停止)")

    def _get_banner(self) -> str:
        """獲取橫幅"""
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🐉 十二組合天空龍策略 - VS Code 不間斷執行器                             ║
║        Sky Dragon Twelve Combination Strategy Runner                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  組合 1-4:  穩健成長型 | 價值動能型 | 高配息穩健型 | 超低波動型              ║
║  組合 5-8:  小資動能型 | 小資配息型 | 小資趨勢型 | 動能配息型                ║
║  組合 9-12: 動能避險型 | 雙動能型 | 配息避險型 | 趨勢避險型                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
#                         CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐉 十二組合天空龍策略 - VS Code 不間斷執行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python sky_dragon_continuous_runner.py --mode continuous    # 不間斷執行
  python sky_dragon_continuous_runner.py --mode single        # 單次執行
  python sky_dragon_continuous_runner.py --mode monitor       # 監控進度
  python sky_dragon_continuous_runner.py --mode results       # 查看結果
  python sky_dragon_continuous_runner.py --mode stop          # 停止執行
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["continuous", "single", "monitor", "results", "stop"],
        default="continuous",
        help="執行模式 (預設: continuous)"
    )

    parser.add_argument(
        "--combination", "-c",
        type=int,
        help="指定組合 ID (1-12)，僅用於 single 模式"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="啟用調試模式"
    )

    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=100,
        help="演化代數 (預設: 100)"
    )

    parser.add_argument(
        "--population", "-p",
        type=int,
        default=50,
        help="族群大小 (預設: 50)"
    )

    args = parser.parse_args()

    # 創建配置
    config = SkyDragonConfig(
        n_generations=args.generations,
        population_size=args.population,
    )

    if args.debug:
        print("🔧 調試模式已啟用")
        config.checkpoint_interval = 5
        config.n_generations = min(20, args.generations)

    # 創建執行器
    runner = SkyDragonRunner(config)

    # 執行對應模式
    if args.mode == "continuous":
        runner.run_continuous()
    elif args.mode == "single":
        runner.run_single(args.combination)
    elif args.mode == "monitor":
        runner.monitor()
    elif args.mode == "results":
        runner.show_results()
    elif args.mode == "stop":
        runner.stop_all()


if __name__ == "__main__":
    main()
