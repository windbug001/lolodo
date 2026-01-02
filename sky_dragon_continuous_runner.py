#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 十二組合天空龍策略 - VS Code 不間斷執行器
   Sky Dragon Twelve Strategy Runner for VS Code
================================================================================

【十二策略說明】
本系統使用 12 種專業台股量化策略進行基因演算法優化：

📊 基礎策略:
  1. 低波動本益比策略 (low_volatility_pe) - 穩健價值型
  2. 小資族策略 (small_investor) - 低價優質股
  3. 營收股價雙渦輪 (revenue_price_turbo) - 營收動能型
  4. 高殖利率烏龜策略 (high_yield_turtle) - 穩定配息型
  5. 低波動指數 (low_volatility_index) - 防守避險型
  6. 市場指標策略 (market_indicator) - 行業趨勢型

🚀 進階策略:
  7. 監獄兔策略 (prison_rabbit) - 成長動能型
  8. 精選強勢股 (elite_momentum) - 相對強度型
  9. 純技術分析 (pure_technical) - MACD/RSI/布林
  10. 膽小貓策略 (scaredy_cat) - 超低波動型
  11. 合約負債建築工 (contract_debt_construction) - 營建業特化
  12. 研發魔人策略 (rd_maniac) - 研發導向型

【核心功能】
✅ 3 視窗並行執行 (Colab Pro+ / VS Code)
✅ 基因演算法自動優化權重
✅ 目標：夏普值 4.2+ / 胃納量 1000萬+
✅ 斷點續傳 + Discord 通知
✅ 樣本外測試防過擬合

版本：v2.1 (2025)
================================================================================
"""

from __future__ import annotations
import os
import sys
import time
import json
import signal
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil

# ============================================================================
#                         配置區
# ============================================================================

# 原始策略腳本名稱
STRATEGY_SCRIPT = "十二組合快快龍_優化版_高胃納量.py"

# 預設設定
DEFAULT_CONFIG = {
    "finlab_api_key": "R5XcZHGBZgEO5zz+6e1iYAe3wcFiimTUNaCMKsnZEiM42Wp49xW46MySUZT1W/Ee#vip_m",
    "discord_webhook": "https://discord.com/api/webhooks/1429310065877323796/U8lefLn9F1FhHaRXt8a024gHP5alrnM_mXF8QXfhLiddhpV5AqUpkPEaYNEDLbzuuNdk",
    "google_drive_path": "",  # 自動偵測
    "max_retries": 10,
    "retry_delay": 120,  # 秒
    "health_check_interval": 300,  # 秒
}

# 十二策略定義
TWELVE_STRATEGIES = [
    ("low_volatility_pe", "低波動本益比", "穩健價值型"),
    ("small_investor", "小資族", "低價優質股"),
    ("revenue_price_turbo", "營收雙渦輪", "營收動能型"),
    ("high_yield_turtle", "高殖利率烏龜", "穩定配息型"),
    ("low_volatility_index", "低波動指數", "防守避險型"),
    ("market_indicator", "市場指標", "行業趨勢型"),
    ("prison_rabbit", "監獄兔", "成長動能型"),
    ("elite_momentum", "精選強勢股", "相對強度型"),
    ("pure_technical", "純技術分析", "MACD/RSI/布林"),
    ("scaredy_cat", "膽小貓", "超低波動型"),
    ("contract_debt_construction", "合約負債建築工", "營建業特化"),
    ("rd_maniac", "研發魔人", "研發導向型"),
]


# ============================================================================
#                         進度追蹤
# ============================================================================

class WindowProgress:
    """視窗進度追蹤"""

    def __init__(self, log_dir: str = "./sky_dragon_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.log_dir / "runner_progress.json"
        self.windows = {1: {}, 2: {}, 3: {}}
        self._load()

    def _load(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.windows = data.get("windows", self.windows)
            except:
                pass

    def save(self):
        data = {
            "last_update": datetime.now().isoformat(),
            "windows": self.windows,
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update(self, window_id: int, **kwargs):
        self.windows[window_id].update(kwargs)
        self.windows[window_id]["last_update"] = datetime.now().isoformat()
        self.save()

    def get_summary(self) -> str:
        lines = [
            "",
            "=" * 70,
            "🐉 十二組合天空龍策略 - 視窗狀態",
            "=" * 70,
        ]

        for wid in [1, 2, 3]:
            w = self.windows.get(wid, {})
            status = w.get("status", "未啟動")
            gen = w.get("generation", 0)
            sharpe = w.get("best_sharpe", 0)
            capacity = w.get("best_capacity", 0)

            icon = {"running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "⏳")
            lines.append(f"  {icon} 視窗 {wid}: {status:<8} | 代數: {gen:3d} | "
                        f"夏普: {sharpe:.2f} | 胃納: {capacity/1e6:.1f}M")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
#                         執行器
# ============================================================================

class SkyDragonRunner:
    """十二組合天空龍執行器"""

    def __init__(self, window_id: int = 1, config: dict = None):
        self.window_id = window_id
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.progress = WindowProgress()
        self._process: Optional[subprocess.Popen] = None
        self._stop_flag = False

        # 設置信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\n⚠️ 收到終止信號，正在安全退出...")
        self._stop_flag = True
        if self._process:
            self._process.terminate()

    def _get_script_path(self) -> Path:
        """獲取策略腳本路徑"""
        script_path = Path(__file__).parent / STRATEGY_SCRIPT
        if not script_path.exists():
            # 嘗試在當前目錄找
            script_path = Path(STRATEGY_SCRIPT)
        return script_path

    def run_continuous(self):
        """不間斷執行模式"""
        print(self._get_banner())
        print(f"\n🚀 啟動視窗 {self.window_id} - 不間斷執行模式\n")

        script_path = self._get_script_path()
        if not script_path.exists():
            print(f"❌ 找不到策略腳本: {STRATEGY_SCRIPT}")
            print("   請確保腳本在同一目錄下")
            return

        retry_count = 0

        while not self._stop_flag and retry_count < self.config["max_retries"]:
            self.progress.update(self.window_id, status="running", retry_count=retry_count)
            print(f"\n{'='*60}")
            print(f"🔄 執行 #{retry_count + 1} | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")

            try:
                success = self._run_strategy(script_path)

                if success:
                    self.progress.update(self.window_id, status="completed")
                    print("\n✅ 策略執行完成！")
                    break
                else:
                    retry_count += 1
                    if retry_count < self.config["max_retries"]:
                        delay = self.config["retry_delay"]
                        print(f"\n⏳ {delay} 秒後重試... ({retry_count}/{self.config['max_retries']})")
                        time.sleep(delay)

            except KeyboardInterrupt:
                print("\n⚠️ 使用者中斷")
                break
            except Exception as e:
                retry_count += 1
                self.progress.update(self.window_id, status="failed", last_error=str(e))
                print(f"\n❌ 執行錯誤: {e}")

                if retry_count < self.config["max_retries"]:
                    delay = self.config["retry_delay"]
                    print(f"⏳ {delay} 秒後重試...")
                    time.sleep(delay)

        if self._stop_flag:
            self.progress.update(self.window_id, status="stopped")
            print("\n⏹️ 執行已停止")
        elif retry_count >= self.config["max_retries"]:
            self.progress.update(self.window_id, status="failed")
            print(f"\n❌ 已達最大重試次數 ({self.config['max_retries']})")

    def _run_strategy(self, script_path: Path) -> bool:
        """執行策略腳本"""
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "WINDOW_ID": str(self.window_id),
            "FINLAB_API_KEY": self.config["finlab_api_key"],
            "DISCORD_WEBHOOK_URL": self.config["discord_webhook"],
        })

        if self.config.get("google_drive_path"):
            env["GOOGLE_DRIVE_PATH"] = self.config["google_drive_path"]

        cmd = [
            sys.executable,
            str(script_path),
            "--window", str(self.window_id),
        ]

        print(f"執行: python {script_path.name} --window {self.window_id}")
        print("-" * 60)

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 實時輸出
        while True:
            line = self._process.stdout.readline()
            if not line and self._process.poll() is not None:
                break
            if line:
                print(line.rstrip())

                # 解析進度
                if "代數" in line or "Generation" in line:
                    self._parse_progress(line)

        return_code = self._process.wait()
        self._process = None

        return return_code == 0

    def _parse_progress(self, line: str):
        """解析進度資訊"""
        import re
        # 嘗試解析代數
        gen_match = re.search(r'(?:代數|Gen(?:eration)?)\s*[:#]?\s*(\d+)', line)
        if gen_match:
            gen = int(gen_match.group(1))
            self.progress.update(self.window_id, generation=gen)

        # 嘗試解析夏普值
        sharpe_match = re.search(r'夏普[值]?\s*[:#]?\s*([\d.]+)', line)
        if sharpe_match:
            sharpe = float(sharpe_match.group(1))
            self.progress.update(self.window_id, best_sharpe=sharpe)

        # 嘗試解析胃納量
        cap_match = re.search(r'胃納[量]?\s*[:#]?\s*([\d.]+)', line)
        if cap_match:
            cap = float(cap_match.group(1)) * 1e6
            self.progress.update(self.window_id, best_capacity=cap)

    def run_single(self):
        """單次執行"""
        print(self._get_banner())
        print(f"\n🚀 視窗 {self.window_id} - 單次執行\n")

        script_path = self._get_script_path()
        if not script_path.exists():
            print(f"❌ 找不到策略腳本: {STRATEGY_SCRIPT}")
            return

        self.progress.update(self.window_id, status="running")
        success = self._run_strategy(script_path)
        self.progress.update(self.window_id, status="completed" if success else "failed")

    def monitor(self):
        """監控模式"""
        print(self._get_banner())
        print("\n📊 進入監控模式 (按 Ctrl+C 退出)\n")

        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                print(self._get_banner())
                self.progress._load()  # 重新載入
                print(self.progress.get_summary())
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (每 30 秒刷新)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 監控結束")

    def show_strategies(self):
        """顯示策略說明"""
        print(self._get_banner())
        print("\n📊 十二策略詳細說明:\n")
        print("-" * 70)

        for i, (code, name, desc) in enumerate(TWELVE_STRATEGIES, 1):
            print(f"  {i:2d}. {name:<12} ({code:<28}) - {desc}")

        print("-" * 70)
        print("""
🎯 優化目標:
   - 夏普值: >= 4.2
   - 胃納量: >= 1000萬 TWD
   - 最大回撤: <= 15%
   - 年化報酬: >= 40%

📈 GA 參數:
   - 族群大小: 30
   - 演化代數: 300
   - 變異率: 15%
   - 交叉率: 80%
""")

    def _get_banner(self) -> str:
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🐉 十二組合天空龍策略 - VS Code 不間斷執行器 v2.1                        ║
║        Sky Dragon Twelve Strategy Continuous Runner                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 12 種專業量化策略 | 🎯 夏普 4.2+ | 💰 胃納量 1000萬+                      ║
║  🔄 斷點續傳 | 📱 Discord 通知 | 🔬 樣本外測試防過擬合                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
#                         多視窗執行器
# ============================================================================

class MultiWindowRunner:
    """多視窗並行執行器"""

    def __init__(self, windows: list = None):
        self.windows = windows or [1, 2, 3]
        self.processes = {}

    def run_all(self):
        """啟動所有視窗"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     🐉 十二組合天空龍 - 多視窗並行執行                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        script_path = Path(__file__).parent / STRATEGY_SCRIPT
        if not script_path.exists():
            script_path = Path(STRATEGY_SCRIPT)

        if not script_path.exists():
            print(f"❌ 找不到策略腳本: {STRATEGY_SCRIPT}")
            return

        for window_id in self.windows:
            print(f"🚀 啟動視窗 {window_id}...")
            self._start_window(window_id, script_path)
            time.sleep(5)  # 間隔啟動

        print(f"\n✅ 已啟動 {len(self.windows)} 個視窗")
        print("   使用 --mode monitor 查看進度")

    def _start_window(self, window_id: int, script_path: Path):
        """啟動單一視窗"""
        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--mode", "continuous",
            "--window", str(window_id),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.processes[window_id] = process


# ============================================================================
#                         CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐉 十二組合天空龍策略 - VS Code 不間斷執行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單視窗不間斷執行
  python sky_dragon_continuous_runner.py --mode continuous --window 1

  # 單次執行
  python sky_dragon_continuous_runner.py --mode single --window 2

  # 監控所有視窗進度
  python sky_dragon_continuous_runner.py --mode monitor

  # 啟動所有 3 個視窗
  python sky_dragon_continuous_runner.py --mode all

  # 查看策略說明
  python sky_dragon_continuous_runner.py --mode strategies

環境變數:
  WINDOW_ID          視窗 ID (1-3)
  FINLAB_API_KEY     FinLab API Key
  DISCORD_WEBHOOK_URL Discord Webhook URL
  GOOGLE_DRIVE_PATH  Google Drive 路徑
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["continuous", "single", "monitor", "all", "strategies"],
        default="continuous",
        help="執行模式"
    )

    parser.add_argument(
        "--window", "-w",
        type=int,
        choices=[1, 2, 3],
        default=int(os.environ.get("WINDOW_ID", 1)),
        help="視窗 ID (1-3)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="調試模式"
    )

    args = parser.parse_args()

    if args.mode == "all":
        runner = MultiWindowRunner()
        runner.run_all()
    elif args.mode == "monitor":
        runner = SkyDragonRunner(args.window)
        runner.monitor()
    elif args.mode == "strategies":
        runner = SkyDragonRunner(args.window)
        runner.show_strategies()
    elif args.mode == "single":
        runner = SkyDragonRunner(args.window)
        runner.run_single()
    else:  # continuous
        runner = SkyDragonRunner(args.window)
        runner.run_continuous()


if __name__ == "__main__":
    main()
