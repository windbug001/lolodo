#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ GA 優化監控中心 v1.0
監控所有視窗的優化進度（小小龍、快快龍、天空龍）
"""

import os
import pickle
import time
from datetime import datetime
from pathlib import Path

# ===== 設定區 =====
# Google Drive 基礎路徑
BASE_DIR = "/content/drive/MyDrive/FinLab_GA_優化_終極版"

# 監控的程式配置
PROGRAMS = {
    "小小龍": {
        "prefix": "xiaoxiaolong",
        "strategies": 3,
        "windows": [1, 2, 3],
        "target_capacity": 5_000_000,
    },
    "快快龍": {
        "prefix": "kuaikuailong",
        "strategies": 6,
        "windows": [1, 2, 3],
        "target_capacity": 10_000_000,
    },
    "天空龍": {
        "prefix": "tiankonglong",
        "strategies": 12,
        "windows": [1, 2, 3],
        "target_capacity": 50_000_000,
    },
}

# 刷新間隔（秒）
REFRESH_INTERVAL = 1800  # 30 分鐘

def clear_screen():
    """清除螢幕"""
    os.system('clear' if os.name != 'nt' else 'cls')
    # Colab 用 print 多行空白
    print("\n" * 50)

def load_checkpoint(program_name, window_id):
    """載入 checkpoint 檔案"""
    config = PROGRAMS.get(program_name)
    if not config:
        return None

    # 嘗試多種可能的 checkpoint 路徑
    possible_paths = [
        # 標準路徑
        f"{BASE_DIR}/window_{window_id}/checkpoint_{config['prefix']}_w{window_id}.pkl",
        f"{BASE_DIR}/window_{window_id}/checkpoint_w{window_id}.pkl",
        # 直接在 window 資料夾
        f"{BASE_DIR}/window{window_id}/checkpoint.pkl",
        f"{BASE_DIR}/window_{window_id}/checkpoint.pkl",
        # 舊格式
        f"{BASE_DIR}/checkpoint_window{window_id}.pkl",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    data['_checkpoint_path'] = path
                    return data
            except Exception as e:
                continue

    return None

def load_hall_of_fame(program_name, window_id):
    """載入名人堂檔案"""
    config = PROGRAMS.get(program_name)
    if not config:
        return None

    possible_paths = [
        f"{BASE_DIR}/window_{window_id}/hall_of_fame_{config['prefix']}_w{window_id}.pkl",
        f"{BASE_DIR}/window_{window_id}/hall_of_fame_w{window_id}.pkl",
        f"{BASE_DIR}/window_{window_id}/halloffame.pkl",
        f"{BASE_DIR}/hall_of_fame_window{window_id}.pkl",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                continue

    return None

def format_number(num):
    """格式化數字顯示"""
    if num is None:
        return "N/A"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.2f}"

def get_best_fitness(checkpoint_data, hof_data):
    """取得最佳適應度"""
    best = None

    # 從 checkpoint 取得
    if checkpoint_data:
        if 'best_fitness' in checkpoint_data:
            best = checkpoint_data['best_fitness']
        elif 'halloffame' in checkpoint_data and checkpoint_data['halloffame']:
            try:
                ind = checkpoint_data['halloffame'][0]
                if hasattr(ind, 'fitness') and ind.fitness.valid:
                    best = ind.fitness.values
            except:
                pass

    # 從 hall of fame 取得
    if hof_data and not best:
        try:
            if hasattr(hof_data, '__iter__') and len(hof_data) > 0:
                ind = hof_data[0]
                if hasattr(ind, 'fitness') and ind.fitness.valid:
                    best = ind.fitness.values
        except:
            pass

    return best

def scan_windows():
    """掃描所有視窗狀態"""
    results = []

    for program_name, config in PROGRAMS.items():
        for window_id in config['windows']:
            checkpoint = load_checkpoint(program_name, window_id)
            hof = load_hall_of_fame(program_name, window_id)

            status = {
                'program': program_name,
                'window': window_id,
                'strategies': config['strategies'],
                'generation': None,
                'max_gen': None,
                'population': None,
                'best_sharpe': None,
                'best_capacity': None,
                'best_return': None,
                'best_drawdown': None,
                'last_update': None,
                'status': '❓ 未偵測',
                'checkpoint_path': None,
            }

            if checkpoint:
                status['checkpoint_path'] = checkpoint.get('_checkpoint_path', '')
                status['generation'] = checkpoint.get('generation', checkpoint.get('gen', None))
                status['max_gen'] = checkpoint.get('max_generations', checkpoint.get('ngen', 100))
                status['population'] = len(checkpoint.get('population', [])) if 'population' in checkpoint else None

                # 取得最佳適應度
                best_fitness = get_best_fitness(checkpoint, hof)
                if best_fitness:
                    if len(best_fitness) >= 1:
                        status['best_sharpe'] = best_fitness[0]
                    if len(best_fitness) >= 2:
                        status['best_capacity'] = best_fitness[1]
                    if len(best_fitness) >= 3:
                        status['best_return'] = best_fitness[2]
                    if len(best_fitness) >= 4:
                        status['best_drawdown'] = best_fitness[3]

                # 計算狀態
                if status['generation'] is not None:
                    if status['max_gen'] and status['generation'] >= status['max_gen']:
                        status['status'] = '✅ 完成'
                    else:
                        status['status'] = '🔄 運行中'

                # 最後更新時間
                if status['checkpoint_path']:
                    try:
                        mtime = os.path.getmtime(status['checkpoint_path'])
                        status['last_update'] = datetime.fromtimestamp(mtime)
                    except:
                        pass

            results.append(status)

    return results

def print_dashboard(results):
    """打印監控儀表板"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 100)
    print(f"{'🖥️  GA 優化監控中心':^90}")
    print(f"{'更新時間: ' + now:^100}")
    print("=" * 100)
    print()

    # 按程式分組顯示
    current_program = None

    for r in results:
        if r['program'] != current_program:
            current_program = r['program']
            config = PROGRAMS[current_program]
            print(f"\n{'─' * 100}")
            print(f"📊 {current_program} ({config['strategies']}策略) | 目標胃納量: {format_number(config['target_capacity'])}")
            print(f"{'─' * 100}")
            print(f"{'視窗':<6} {'狀態':<12} {'世代':<12} {'夏普':<10} {'胃納量':<12} {'年化報酬':<10} {'回撤':<10} {'最後更新':<20}")
            print(f"{'-' * 100}")

        gen_str = f"{r['generation']}/{r['max_gen']}" if r['generation'] is not None else "N/A"
        sharpe_str = f"{r['best_sharpe']:.3f}" if r['best_sharpe'] else "N/A"
        cap_str = format_number(r['best_capacity'])
        ret_str = f"{r['best_return']:.1%}" if r['best_return'] else "N/A"
        dd_str = f"{r['best_drawdown']:.1%}" if r['best_drawdown'] else "N/A"
        update_str = r['last_update'].strftime("%H:%M:%S") if r['last_update'] else "N/A"

        print(f"W{r['window']:<5} {r['status']:<12} {gen_str:<12} {sharpe_str:<10} {cap_str:<12} {ret_str:<10} {dd_str:<10} {update_str:<20}")

    print()
    print("=" * 100)

    # 統計摘要
    running = sum(1 for r in results if '運行中' in r['status'])
    completed = sum(1 for r in results if '完成' in r['status'])
    unknown = sum(1 for r in results if '未偵測' in r['status'])

    print(f"\n📈 統計: 🔄 運行中: {running} | ✅ 完成: {completed} | ❓ 未偵測: {unknown}")
    print(f"\n💡 提示: 每 {REFRESH_INTERVAL} 秒自動刷新 | 按 Ctrl+C 停止監控")
    print("=" * 100)

def monitor_loop():
    """監控主迴圈"""
    print("🚀 啟動 GA 優化監控中心...")
    print(f"📁 監控路徑: {BASE_DIR}")
    print()

    try:
        while True:
            clear_screen()
            results = scan_windows()
            print_dashboard(results)

            # 等待下次刷新
            for i in range(REFRESH_INTERVAL, 0, -1):
                print(f"\r⏳ 下次刷新: {i} 秒  ", end='', flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n👋 監控已停止")

def run_once():
    """單次執行（不循環）"""
    results = scan_windows()
    print_dashboard(results)
    return results

# ===== 主程式 =====
if __name__ == "__main__":
    # 檢查是否在 Colab 環境
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        # Colab 環境：提供選擇
        print("🖥️ GA 優化監控中心")
        print("1. 單次查看")
        print("2. 持續監控")
        print()
        # 預設持續監控
        monitor_loop()
    else:
        monitor_loop()
