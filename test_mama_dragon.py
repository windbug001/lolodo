#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🐉 九組合媽媽龍策略 - 快速測試腳本
================================================================================

使用方式：
    python test_mama_dragon.py [模式]

模式選項：
    1 或 analyze   : 策略分析（快速回測各策略）
    2 或 evolve    : 基因演算法優化（完整演化）
    3 或 best      : 回測最佳個體
    4 或 live      : Live Performance 回測
    5 或 single    : 測試單一策略

範例：
    python test_mama_dragon.py 1          # 快速分析
    python test_mama_dragon.py analyze    # 同上
    python test_mama_dragon.py 5          # 測試單一策略
"""

import sys
import os

# 設置環境
os.environ['FINLAB_DISABLE_CACHE'] = '1'

def print_menu():
    """顯示選單"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           🐉 九組合媽媽龍策略 - 測試系統                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   請選擇測試模式：                                                        ║
║                                                                          ║
║   [1] 策略分析     - 快速回測各策略表現                                   ║
║   [2] 基因演算法   - 執行 NSGA-II 多目標優化                              ║
║   [3] 回測最佳     - 使用已保存的最佳參數回測                             ║
║   [4] Live 回測    - 執行 Live Performance 回測                          ║
║   [5] 單策略測試   - 測試單一策略                                        ║
║   [6] 參數檢視     - 顯示 180 個基因參數結構                              ║
║                                                                          ║
║   [q] 離開                                                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


def test_single_strategy(strategy_num: int = 1):
    """測試單一策略"""
    from nine_combo_mama_dragon_strategy import (
        strategy_engine, gene_decoder, GeneDecoder, run_backtest
    )

    strategy_map = {
        1: ('低波動本益比', strategy_engine.strategy_1_low_volatility_pe),
        2: ('小資族', strategy_engine.strategy_2_small_investor),
        3: ('營收股價雙渦輪', strategy_engine.strategy_3_revenue_price_turbo),
        4: ('高殖利率烏龜', strategy_engine.strategy_4_high_yield_turtle),
        5: ('低波動性指標', strategy_engine.strategy_5_low_volatility_index),
        6: ('藏獒外掛大盤指針', strategy_engine.strategy_6_market_indicator),
        7: ('小蝦米跟大鯨魚', strategy_engine.strategy_7_shrimp_whale),
        8: ('純技術趨勢', strategy_engine.strategy_8_tech_trend),
        9: ('財報指標20大', strategy_engine.strategy_9_fundamental_20),
    }

    if strategy_num not in strategy_map:
        print(f"❌ 無效的策略編號: {strategy_num}")
        return

    name, func = strategy_map[strategy_num]
    print(f"\n🔍 測試策略 {strategy_num}: {name}")
    print("=" * 60)

    # 使用預設參數
    params = gene_decoder.decode([0.5] * GeneDecoder.GENE_LENGTH)

    print("📊 執行策略...")
    position = func(params)

    if position is not None and not position.empty:
        print(f"   持股天數: {len(position)}")
        print(f"   平均持股數: {position.sum(axis=1).mean():.1f}")

        print("\n📈 執行回測...")
        result = run_backtest(position, params, name, full_report=True)
    else:
        print("   ⚠️ 無持股訊號")


def show_gene_structure():
    """顯示基因參數結構"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🧬 180 個基因參數結構                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   索引範圍      策略/用途                              參數數量           ║
║   ─────────────────────────────────────────────────────────────────────  ║
║   0-8          九策略權重 (weight_1 ~ weight_9)          9 個            ║
║   ─────────────────────────────────────────────────────────────────────  ║
║   9-28         策略1: 低波動本益比 (s1_*)               20 個            ║
║   29-48        策略2: 小資族 (s2_*)                     20 個            ║
║   49-68        策略3: 營收股價雙渦輪 (s3_*)             20 個            ║
║   69-84        策略4: 高殖利率烏龜 (s4_*)               16 個            ║
║   85-100       策略5: 低波動性指標 (s5_*)               16 個            ║
║   101-116      策略6: 藏獒外掛大盤指針 (s6_*)           16 個            ║
║   117-136      策略7: 小蝦米跟大鯨魚 (s7_*)             20 個            ║
║   137-156      策略8: 純技術趨勢 (s8_*)                 20 個            ║
║   157-173      策略9: 財報指標20大 (s9_*)               17 個            ║
║   ─────────────────────────────────────────────────────────────────────  ║
║   174-179      回測與風控參數                            6 個            ║
║                                                                          ║
║   總計: 180 個可演化參數                                                  ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   🎯 優化目標：                                                          ║
║      • 夏普值 >= 4.2                                                     ║
║      • 胃納量 >= 1,000 萬台幣                                            ║
║      • 最大回檔 <= 20%                                                   ║
║      • 每股占比 >= 3%                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """主程式"""
    # 處理命令列參數
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        mode_map = {
            '1': 1, 'analyze': 1, 'analysis': 1,
            '2': 2, 'evolve': 2, 'ga': 2, 'evolution': 2,
            '3': 3, 'best': 3,
            '4': 4, 'live': 4,
            '5': 5, 'single': 5, 'test': 5,
            '6': 6, 'params': 6, 'gene': 6,
        }
        mode = mode_map.get(arg, 0)

        if mode == 0:
            print(f"❌ 無效的模式: {arg}")
            print(__doc__)
            return
    else:
        # 互動模式
        print_menu()
        try:
            choice = input("請輸入選項 [1-6, q]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            return

        if choice in ('q', 'quit', 'exit'):
            print("👋 Bye!")
            return

        mode_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        mode = mode_map.get(choice, 0)

        if mode == 0:
            print("❌ 無效的選項")
            return

    # 執行選擇的模式
    if mode == 6:
        show_gene_structure()
        return

    # 載入主模組
    print("\n📦 載入九組合媽媽龍策略模組...")

    try:
        from nine_combo_mama_dragon_strategy import (
            analyze_strategies,
            EvolutionEngine, ParetoArchiveManager,
            backtest_best_individual, run_live_backtest,
            N_GENERATIONS
        )
        print("✅ 模組載入成功")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 執行對應模式
    if mode == 1:
        print("\n🔍 執行策略分析...")
        analyze_strategies()

    elif mode == 2:
        print("\n🧬 執行基因演算法優化...")
        pareto_mgr = ParetoArchiveManager()
        engine = EvolutionEngine(pareto_mgr)
        population, pareto_front = engine.run(n_generations=N_GENERATIONS, resume=True)
        backtest_best_individual(pareto_mgr)

    elif mode == 3:
        print("\n📊 回測最佳個體...")
        pareto_mgr = ParetoArchiveManager()
        backtest_best_individual(pareto_mgr)

    elif mode == 4:
        print("\n📈 執行 Live Performance 回測...")
        run_live_backtest()

    elif mode == 5:
        print("\n請選擇要測試的策略 (1-9)：")
        print("  1. 低波動本益比    2. 小資族        3. 營收股價雙渦輪")
        print("  4. 高殖利率烏龜    5. 低波動性指標  6. 藏獒外掛大盤指針")
        print("  7. 小蝦米跟大鯨魚  8. 純技術趨勢    9. 財報指標20大")
        try:
            strategy_num = int(input("輸入策略編號 [1-9]: "))
            test_single_strategy(strategy_num)
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\n⚠️ 取消")

    print("\n✅ 測試完成！")


if __name__ == "__main__":
    main()
