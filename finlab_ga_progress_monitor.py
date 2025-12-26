#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
📊 FinLab 基因演算法進度監控系統
FinLab GA Progress Monitor
================================================================================

功能：
✅ 實時監控 1-4 個視窗的優化進度
✅ 自動解析日誌檔案
✅ 視覺化進度條
✅ 預測達標時間
✅ 比較多視窗效能

使用方式：
在獨立的 Colab Cell 執行此腳本，可同時監控多個優化視窗

版本：v1.0 (2025-12-26)
================================================================================
"""

import os
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# =============================================================================
# 設定區
# =============================================================================
# Google Drive 路徑（與主程式一致）
try:
    BASE_DIR = '/content/drive/MyDrive/FinLab_GA_優化_終極版'
except:
    BASE_DIR = './finlab_ga_output'

# 監控設定
REFRESH_INTERVAL = 60  # 每60秒刷新一次
TARGET_SHARPE = 4.0
MIN_CAPACITY = 10_000_000

# =============================================================================
# 進度解析器
# =============================================================================
class ProgressParser:
    """解析進度日誌"""

    def __init__(self, window_id: int):
        self.window_id = window_id
        self.log_dir = f"{BASE_DIR}/window_{window_id}_logs"
        self.history_file = f"{self.log_dir}/progress_history.json"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def parse_latest_progress(self) -> Optional[Dict]:
        """解析最新進度"""
        # 嘗試從歷史檔案讀取
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    if history:
                        return history[-1]
            except:
                pass

        return None

    def save_progress(self, progress: Dict):
        """保存進度"""
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except:
                pass

        history.append(progress)

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

# =============================================================================
# 進度顯示器
# =============================================================================
class ProgressDisplay:
    """進度視覺化顯示"""

    @staticmethod
    def print_progress_bar(current: float, total: float, width: int = 40) -> str:
        """生成進度條"""
        filled = int((current / total) * width)
        bar = '█' * filled + '░' * (width - filled)
        percent = (current / total) * 100
        return f"[{bar}] {percent:.1f}%"

    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化時間"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.0f}分鐘"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小時"

    @staticmethod
    def estimate_remaining_time(progress: Dict) -> Optional[str]:
        """預估剩餘時間"""
        if not progress or progress['current_gen'] == 0:
            return None

        # 計算平均每代時間
        avg_time_per_gen = progress['elapsed_total'] / progress['current_gen']

        # 預估達到目標需要的代數
        sharpe_gap = TARGET_SHARPE - progress['best_sharpe']
        if sharpe_gap <= 0:
            return "✅ 已達標"

        # 計算進步速度
        if progress['current_gen'] >= 5:
            improvement_rate = progress['best_sharpe'] / progress['current_gen']
            estimated_gens = sharpe_gap / improvement_rate if improvement_rate > 0 else 999
        else:
            estimated_gens = 100 - progress['current_gen']

        remaining_time = estimated_gens * avg_time_per_gen
        return ProgressDisplay.format_time(remaining_time)

# =============================================================================
# 多視窗監控器
# =============================================================================
class MultiWindowMonitor:
    """多視窗監控器"""

    def __init__(self, window_ids: List[int] = [1, 2, 3, 4]):
        self.window_ids = window_ids
        self.parsers = {wid: ProgressParser(wid) for wid in window_ids}
        self.display = ProgressDisplay()

    def get_all_progress(self) -> Dict[int, Optional[Dict]]:
        """獲取所有視窗進度"""
        return {wid: parser.parse_latest_progress()
                for wid, parser in self.parsers.items()}

    def print_dashboard(self, all_progress: Dict[int, Optional[Dict]]):
        """打印儀表板"""
        os.system('clear' if os.name != 'nt' else 'cls')

        print("=" * 80)
        print("📊 FinLab 基因演算法優化進度監控")
        print(f"⏰ 更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 統計
        active_windows = sum(1 for p in all_progress.values() if p is not None)
        completed_windows = sum(1 for p in all_progress.values()
                               if p and p.get('best_sharpe', 0) >= TARGET_SHARPE)

        print(f"\n📈 總覽:")
        print(f"   活躍視窗: {active_windows}/{len(self.window_ids)}")
        print(f"   已達標: {completed_windows}/{len(self.window_ids)}")

        # 各視窗詳情
        for wid in sorted(all_progress.keys()):
            progress = all_progress[wid]
            print(f"\n{'='*80}")
            print(f"🖥️  視窗 {wid}")
            print(f"{'='*80}")

            if progress is None:
                print("   ⚠️  無進度數據（未啟動或日誌缺失）")
                continue

            # 基本資訊
            current_gen = progress.get('current_gen', 0)
            total_gen = progress.get('total_gen', 100)
            best_sharpe = progress.get('best_sharpe', 0)
            best_capacity = progress.get('best_capacity', 0)
            best_composite = progress.get('best_composite', 0)
            robustness = progress.get('robustness', 0)

            print(f"\n   代數進度: {current_gen}/{total_gen}")
            print(f"   {self.display.print_progress_bar(current_gen, total_gen)}")

            print(f"\n   🎯 夏普值: {best_sharpe:.2f} / {TARGET_SHARPE:.1f}")
            sharpe_percent = (best_sharpe / TARGET_SHARPE) * 100
            print(f"   {self.display.print_progress_bar(best_sharpe, TARGET_SHARPE)}")

            print(f"\n   💰 胃納量: {best_capacity:.0f} 萬 / {MIN_CAPACITY/1e4:.0f} 萬")
            capacity_status = "✅ 已達標" if best_capacity >= MIN_CAPACITY/1e4 else "⏳ 未達標"
            print(f"   {capacity_status}")

            print(f"\n   📊 綜合分數: {best_composite:.4f}")
            print(f"   🛡️  穩健性: {robustness:.2f}")

            # 時間資訊
            if 'elapsed_total' in progress:
                elapsed = progress['elapsed_total']
                print(f"\n   ⏱️  已用時間: {self.display.format_time(elapsed)}")

                remaining = self.display.estimate_remaining_time(progress)
                if remaining:
                    print(f"   ⏳ 預估剩餘: {remaining}")

            # 達標狀態
            if best_sharpe >= TARGET_SHARPE:
                print(f"\n   🎉 視窗 {wid} 已達到夏普值目標！")

        print(f"\n{'='*80}")
        print(f"下次更新: {REFRESH_INTERVAL} 秒後")
        print(f"{'='*80}")

    def monitor_loop(self):
        """監控循環"""
        print("🚀 啟動進度監控...")
        print(f"📡 監控視窗: {self.window_ids}")
        print(f"🔄 刷新間隔: {REFRESH_INTERVAL} 秒")
        print(f"\n按 Ctrl+C 停止監控\n")

        try:
            while True:
                all_progress = self.get_all_progress()
                self.print_dashboard(all_progress)
                time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n✋ 監控已停止")

# =============================================================================
# 簡易日誌寫入器（供主程式使用）
# =============================================================================
class ProgressLogger:
    """進度日誌記錄器（主程式調用）"""

    def __init__(self, window_id: int, base_dir: str = BASE_DIR):
        self.window_id = window_id
        self.log_dir = f"{base_dir}/window_{window_id}_logs"
        self.history_file = f"{self.log_dir}/progress_history.json"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log_generation(self, gen: int, total_gen: int, stats: Dict):
        """記錄單代進度"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'window_id': self.window_id,
            'current_gen': gen,
            'total_gen': total_gen,
            'best_sharpe': stats.get('best_sharpe', 0),
            'best_capacity': stats.get('best_capacity', 0),
            'best_composite': stats.get('best_composite', 0),
            'best_return': stats.get('best_return', 0),
            'robustness': stats.get('best_robustness', 0),
            'elapsed_total': time.time() - self.start_time,
        }

        # 讀取歷史
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except:
                pass

        # 添加新記錄
        history.append(progress)

        # 保存
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

# =============================================================================
# 主程式
# =============================================================================
def main():
    """主程式入口"""
    print("=" * 80)
    print("📊 FinLab 基因演算法進度監控系統")
    print("=" * 80)
    print()

    # 檢測哪些視窗正在運行
    active_windows = []
    for wid in [1, 2, 3, 4]:
        log_dir = f"{BASE_DIR}/window_{wid}_logs"
        if os.path.exists(log_dir):
            active_windows.append(wid)

    if not active_windows:
        print("⚠️  未檢測到任何活躍視窗")
        print(f"請確認路徑: {BASE_DIR}")
        print("\n提示: 主程式執行後會自動創建日誌目錄")
        return

    print(f"✅ 檢測到活躍視窗: {active_windows}")
    print()

    # 啟動監控
    monitor = MultiWindowMonitor(window_ids=active_windows)
    monitor.monitor_loop()

if __name__ == "__main__":
    main()
