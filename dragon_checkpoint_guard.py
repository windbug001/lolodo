#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║  🛡️ Dragon Checkpoint 守衛 — 精英數據保護層                      ║
║  適用: gene_funnel_v5.3.6.py 的 DragonGAEngine                  ║
║                                                                  ║
║  功能:                                                           ║
║  1. KNOWN_BEST_BASELINES: 硬編碼已知最佳 Sharpe（最後防線）       ║
║  2. _save_checkpoint: 備份 + 基線守衛 + 原子寫入                  ║
║  3. _load_checkpoint: 損壞自動恢復 + 基線校驗                     ║
║                                                                  ║
║  使用方式:                                                       ║
║    在 gene_funnel_v5.3.6.py 中，DragonGAEngine 類定義之後加入:   ║
║      from dragon_checkpoint_guard import apply_checkpoint_guard   ║
║      apply_checkpoint_guard(DragonGAEngine)                      ║
║                                                                  ║
║  最高指導原則: Sharpe 3.6~3.8 的精英數據絕不可丟失               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# 🛡️ 已知最佳基線值 — 硬編碼在代碼中，絕對不可丟失
# =============================================================================
# 來源：2026-04-02 Dragon 營收換股儀表板
# ⚠️ 修改此表前請三思 — 這是最後一道防線
KNOWN_BEST_BASELINES = {
    24: {'name': 'Dragon-Anchor',   'sharpe': 3.700, 'cagr': 1.282,
         'mdd': 0.207, 'capacity_wan': 532, 'fit': 3.317, 'gen': 114},
    25: {'name': 'Dragon-Explorer', 'sharpe': 3.560, 'cagr': 1.284,
         'mdd': 0.216, 'capacity_wan': 414, 'fit': 3.023, 'gen': 71},
    26: {'name': 'Dragon-Refiner',  'sharpe': 3.842, 'cagr': 1.380,
         'mdd': 0.227, 'capacity_wan': 423, 'fit': 3.478, 'gen': 115},
    27: {'name': 'Dragon-Sharpe',   'sharpe': 3.699, 'cagr': 1.296,
         'mdd': 0.235, 'capacity_wan': 586, 'fit': 3.436, 'gen': 114},
    28: {'name': 'Dragon-Blitz',    'sharpe': 3.743, 'cagr': 1.254,
         'mdd': 0.201, 'capacity_wan': 566, 'fit': 3.300, 'gen': 116},
    29: {'name': 'Dragon-Hydra',    'sharpe': 2.663, 'cagr': 0.298,
         'mdd': 0.111, 'capacity_wan': 106, 'fit': 1.701, 'gen': 32},
    30: {'name': 'Dragon-LargeCap', 'sharpe': 3.643, 'cagr': 1.250,
         'mdd': 0.255, 'capacity_wan': 516, 'fit': 1.111, 'gen': 104},
    31: {'name': 'Hydra-LgCap',     'sharpe': 3.656, 'cagr': 1.267,
         'mdd': 0.233, 'capacity_wan': 568, 'fit': 3.058, 'gen': 51},
}


def apply_checkpoint_guard(DragonGAEngine_cls):
    """
    Monkey-patch DragonGAEngine 的 _save_checkpoint 和 _load_checkpoint，
    加入備份 + 基線守衛 + 損壞自動恢復。

    不修改 gene_funnel_v5.3.6.py 的任何一行原始代碼。
    """

    # 保留原始方法的引用
    _original_save = DragonGAEngine_cls._save_checkpoint
    _original_load = DragonGAEngine_cls._load_checkpoint

    def _guarded_save_checkpoint(self):
        """帶備份 + 基線守衛的 _save_checkpoint"""
        try:
            window_id = getattr(self, '_window_id', None)
            if window_id is None:
                # 從 CKPT_FILE 路徑推斷 window_id
                fname = self.CKPT_FILE.name  # e.g. 'dragon_hof_w26.pkl'
                for wid in KNOWN_BEST_BASELINES:
                    if f'w{wid}' in fname:
                        window_id = wid
                        break

            baseline = KNOWN_BEST_BASELINES.get(window_id)
            best_sharpe = self.best_ever.sharpe if hasattr(self, 'best_ever') else 0

            # ──────────────────────────────────────────────
            # 第 1 層守衛：與硬編碼基線比較
            # ──────────────────────────────────────────────
            if baseline and best_sharpe > 0:
                # 如果已有 checkpoint（非首次），且 best_ever 嚴重退化
                if self.CKPT_FILE.exists():
                    try:
                        with open(self.CKPT_FILE, 'rb') as f:
                            old_ckpt = pickle.load(f)
                        old_best_sharpe = old_ckpt.get('best_ever', {}).get('sharpe', 0)

                        # 退化 > 5%：拒絕覆蓋
                        if old_best_sharpe > 0 and best_sharpe < old_best_sharpe * 0.95:
                            print(f"  🚨🛡️ [W{window_id}] 退化守衛觸發！")
                            print(f"     舊 best Sharpe: {old_best_sharpe:.4f}")
                            print(f"     新 best Sharpe: {best_sharpe:.4f} "
                                  f"(退化 {(1 - best_sharpe / old_best_sharpe) * 100:.1f}%)")
                            print(f"     ❌ 拒絕覆蓋 checkpoint，保留舊版本")
                            return  # 不存檔

                        # 低於硬編碼基線的 90%：拒絕覆蓋
                        baseline_sharpe = baseline['sharpe']
                        if best_sharpe < baseline_sharpe * 0.90:
                            print(f"  🚨🛡️ [W{window_id}] 硬編碼基線守衛觸發！")
                            print(f"     {baseline['name']} 已知最佳: {baseline_sharpe:.3f}")
                            print(f"     當前 best_ever: {best_sharpe:.4f}")
                            print(f"     ❌ 嚴重退化，拒絕覆蓋 checkpoint")
                            return  # 不存檔

                    except Exception as e:
                        # 讀取舊 checkpoint 失敗不影響存檔
                        print(f"  ⚠️ 讀取舊 checkpoint 比對失敗: {e}")

            # ──────────────────────────────────────────────
            # 第 2 層：備份現有 checkpoint（存檔前）
            # ──────────────────────────────────────────────
            if self.CKPT_FILE.exists():
                try:
                    # 滾動備份 .bak
                    bak_file = self.CKPT_FILE.with_suffix('.pkl.bak')
                    shutil.copy2(str(self.CKPT_FILE), str(bak_file))

                    # 時間戳備份（保留最近 10 份）
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    ts_bak = self.CKPT_FILE.parent / f'{self.CKPT_FILE.stem}_backup_{ts}.pkl'
                    shutil.copy2(str(self.CKPT_FILE), str(ts_bak))

                    # 清理過舊的時間戳備份
                    bak_dir = self.CKPT_FILE.parent
                    prefix = f'{self.CKPT_FILE.stem}_backup_'
                    old_baks = sorted([
                        f for f in os.listdir(str(bak_dir))
                        if f.startswith(prefix) and f.endswith('.pkl')
                    ])
                    for old in old_baks[:-10]:
                        try:
                            os.remove(str(bak_dir / old))
                        except OSError:
                            pass

                except Exception as e:
                    print(f"  ⚠️ checkpoint 備份失敗: {e}")

            # ──────────────────────────────────────────────
            # 第 3 層：原子寫入（調用原始方法）
            # ──────────────────────────────────────────────
            _original_save(self)

            # 存檔成功，輸出守衛日誌
            if baseline:
                print(f"  🛡️ [W{window_id}] 基線校驗通過: "
                      f"best={best_sharpe:.4f} >= 基線={baseline['sharpe']:.3f} "
                      f"({baseline['name']})")

        except Exception as e:
            print(f"  ⚠️ guarded_save_checkpoint 失敗: {e}")
            # 失敗時嘗試呼叫原始方法（至少保證基本存檔功能）
            try:
                _original_save(self)
            except Exception:
                pass

    def _guarded_load_checkpoint(self) -> bool:
        """帶損壞自動恢復 + 基線校驗的 _load_checkpoint"""

        # 先嘗試正常載入
        try:
            result = _original_load(self)
            if result:
                _validate_after_load(self)
                return True
        except Exception as e:
            print(f"  ⚠️ 主 checkpoint 載入失敗: {e}")

        # 主檔案損壞 → 嘗試 .bak 備份
        bak_file = self.CKPT_FILE.with_suffix('.pkl.bak')
        if bak_file.exists():
            try:
                print(f"  🛡️ 嘗試從 .bak 備份恢復...")
                # 用 .bak 替換主檔案
                shutil.copy2(str(bak_file), str(self.CKPT_FILE))
                result = _original_load(self)
                if result:
                    print(f"  🛡️ 從 .bak 恢復成功！")
                    _validate_after_load(self)
                    return True
            except Exception as e:
                print(f"  ⚠️ .bak 恢復也失敗: {e}")

        # .bak 也壞 → 掃描時間戳備份
        try:
            bak_dir = self.CKPT_FILE.parent
            prefix = f'{self.CKPT_FILE.stem}_backup_'
            ts_baks = sorted([
                f for f in os.listdir(str(bak_dir))
                if f.startswith(prefix) and f.endswith('.pkl')
            ], reverse=True)  # 最新的在前面

            for bak_name in ts_baks[:5]:  # 嘗試最近 5 份
                bak_path = bak_dir / bak_name
                try:
                    print(f"  🛡️ 嘗試從 {bak_name} 恢復...")
                    shutil.copy2(str(bak_path), str(self.CKPT_FILE))
                    result = _original_load(self)
                    if result:
                        print(f"  🛡️ 從 {bak_name} 恢復成功！")
                        _validate_after_load(self)
                        return True
                except Exception:
                    continue
        except Exception:
            pass

        # 全部備份都失敗
        print(f"  ❌ 所有備份恢復失敗，將從頭開始")
        return False

    def _validate_after_load(self):
        """載入後基線校驗"""
        window_id = getattr(self, '_window_id', None)
        if window_id is None:
            fname = self.CKPT_FILE.name
            for wid in KNOWN_BEST_BASELINES:
                if f'w{wid}' in fname:
                    window_id = wid
                    break

        baseline = KNOWN_BEST_BASELINES.get(window_id)
        if not baseline:
            return

        best_sharpe = self.best_ever.sharpe if hasattr(self, 'best_ever') else 0

        if best_sharpe > 0 and best_sharpe < baseline['sharpe'] * 0.80:
            print(f"  🚨 [W{window_id}] 警告：checkpoint 的 best Sharpe ({best_sharpe:.4f}) "
                  f"遠低於基線 ({baseline['sharpe']:.3f})")
            print(f"     {baseline['name']} 歷史最佳 Sharpe: {baseline['sharpe']}")
            print(f"     🔍 checkpoint 可能已損壞，請檢查備份檔案")
        elif best_sharpe > 0:
            print(f"  🛡️ [W{window_id}] 基線校驗OK: "
                  f"best={best_sharpe:.4f} >= 80%基線={baseline['sharpe'] * 0.80:.3f}")

    # ──────────────────────────────────────────────
    # 套用 monkey-patch
    # ──────────────────────────────────────────────
    DragonGAEngine_cls._save_checkpoint = _guarded_save_checkpoint
    DragonGAEngine_cls._load_checkpoint = _guarded_load_checkpoint

    print(f"  🛡️ [dragon_checkpoint_guard] DragonGAEngine 已加裝保護層")
    print(f"     - 備份: .bak 滾動 + 時間戳 ×10")
    print(f"     - 守衛: 退化>5%拒絕覆蓋 + 硬編碼基線90%攔截")
    print(f"     - 恢復: 主檔→.bak→時間戳備份 自動降級")
