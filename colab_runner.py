#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 六組合快快龍 Colab Pro+ 執行器 v1.0
================================================================================

使用方式 (在 Colab 中執行):

# 方法一：直接執行
!git clone -b claude/strategy-six-dragon-jy0Et https://github.com/windbug001/lolodo.git /content/lolodo
%run /content/lolodo/colab_runner.py --window 1 --api-key YOUR_FINLAB_API_KEY

# 方法二：指定 checkpoint 繼續
%run /content/lolodo/colab_runner.py --window 1 --api-key YOUR_KEY --checkpoint checkpoint_window_1_latest.pkl

================================================================================
"""

import os
import sys
import argparse
import time
import threading

# =============================================================================
# 🔄 防斷線腳本 v2 - 更穩定
# =============================================================================
def setup_anti_disconnect():
    """設定雙重防斷線機制"""
    try:
        from IPython.display import display, Javascript

        # JavaScript 防斷線
        display(Javascript('''
        function KeepAlive() {
            console.log("🔄 " + new Date().toLocaleTimeString());
            // 多重保活機制
            document.querySelector("colab-connect-button")?.click();
            document.querySelector("#connect")?.click();
            // 模擬滾動
            window.scrollBy(0, 1);
            window.scrollBy(0, -1);
        }
        setInterval(KeepAlive, 45000);  // 45秒一次更安全
        console.log("✅ 防斷線 v2 已啟動");
        '''))

        # Python 背景心跳（雙保險）
        def python_heartbeat():
            while True:
                time.sleep(120)
                print(f"💓 {time.strftime('%H:%M:%S')}", end=' ', flush=True)

        heartbeat_thread = threading.Thread(target=python_heartbeat, daemon=True)
        heartbeat_thread.start()

        print("✅ 雙重防斷線已啟動")
        return True
    except Exception as e:
        print(f"⚠️ 防斷線設定失敗 (非 Colab 環境?): {e}")
        return False


# =============================================================================
# 📁 Google Drive 掛載
# =============================================================================
def mount_drive():
    """掛載 Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive 已掛載")
        return True
    except Exception as e:
        print(f"⚠️ Drive 掛載失敗: {e}")
        return False


# =============================================================================
# 📥 下載/更新最新版本
# =============================================================================
def update_from_github(target_dir: str, branch: str = "claude/strategy-six-dragon-jy0Et"):
    """從 GitHub 下載最新版本"""
    import subprocess

    os.makedirs(target_dir, exist_ok=True)

    temp_repo = "/content/temp_repo_update"

    # 清理舊的 temp repo
    subprocess.run(f"rm -rf {temp_repo}", shell=True)

    # Clone 最新版本
    print(f"📥 正在從 GitHub 下載最新版本 (branch: {branch})...")
    result = subprocess.run(
        f"git clone -b {branch} https://github.com/windbug001/lolodo.git {temp_repo}",
        shell=True, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"❌ Git clone 失敗: {result.stderr}")
        return False

    # 複製主要檔案
    files_to_copy = [
        "genetic_algo_v12_cross_window.py",
        "colab_runner.py",
    ]

    for f in files_to_copy:
        src = os.path.join(temp_repo, f)
        if os.path.exists(src):
            subprocess.run(f'cp "{src}" "{target_dir}/"', shell=True)
            print(f"   ✅ 已複製 {f}")

    # 清理
    subprocess.run(f"rm -rf {temp_repo}", shell=True)

    print(f"✅ 已更新到最新版本於 {target_dir}")
    return True


# =============================================================================
# 🚀 主程式
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='六組合快快龍 Colab 執行器')
    parser.add_argument('--window', '-w', type=int, default=1, choices=[1, 2, 3],
                        help='視窗 ID (1, 2, 或 3)')
    parser.add_argument('--api-key', '-k', type=str, required=True,
                        help='FinLab API Key')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='從指定 checkpoint 繼續 (例如: checkpoint_window_1_latest.pkl)')
    parser.add_argument('--no-update', action='store_true',
                        help='不從 GitHub 更新')
    parser.add_argument('--branch', '-b', type=str,
                        default='claude/strategy-six-dragon-jy0Et',
                        help='GitHub branch')

    args = parser.parse_args()

    print("=" * 70)
    print("🐉 六組合快快龍 v12.5 - Colab Pro+ 執行器")
    print("=" * 70)

    # 1. 防斷線
    setup_anti_disconnect()

    # 2. 掛載 Drive
    mount_drive()

    # 3. 設定目標目錄
    TARGET_DIR = "/content/drive/MyDrive/投資策略優化_v12_六組合快快龍"
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 4. 更新程式碼
    if not args.no_update:
        update_from_github(TARGET_DIR, args.branch)

    # 5. 設定環境變數
    os.environ['FINLAB_API_KEY'] = args.api_key
    os.environ['WINDOW_ID'] = str(args.window)

    if args.checkpoint:
        os.environ['CHECKPOINT_PATH'] = args.checkpoint
        print(f"📂 從 checkpoint 繼續: {args.checkpoint}")

    print(f"\n🔧 環境設定:")
    print(f"   WINDOW_ID = {args.window}")
    print(f"   TARGET_DIR = {TARGET_DIR}")
    print(f"   CHECKPOINT = {args.checkpoint or '(從頭開始)'}")

    # 6. 切換目錄並執行
    os.chdir(TARGET_DIR)
    print(f"\n🚀 開始執行基因演算法...\n")
    print("=" * 70)

    # 執行主程式
    ga_script = os.path.join(TARGET_DIR, "genetic_algo_v12_cross_window.py")
    if os.path.exists(ga_script):
        exec(open(ga_script).read(), globals())
    else:
        print(f"❌ 找不到 {ga_script}")
        sys.exit(1)


if __name__ == "__main__":
    main()
