# 🐉 三策略小小龍 v2.0 - 遺傳演算法優化系統

## 📋 系統概述

基於遺傳演算法 (GA) 的台股選股優化系統，整合三大策略並支援自動迭代、Check-point 存取與並行運算。

### 🎯 優化目標

| 指標 | 目標值 | 說明 |
|------|--------|------|
| 夏普值 | > **4.2** | 核心目標 |
| 最大回撤 | < **20%** | 懲罰項 |
| 資金胃納量 | > **1,000萬** | 確保策略可執行性 |
| 年化報酬 | > **30%** | 次要目標 |

### 📊 回測規範

| 期間 | 時間範圍 | 用途 |
|------|----------|------|
| In-Sample | 2017 ~ **2022 年底** | 訓練期（參數優化）|
| Out-of-Sample | **2023 ~ 至今** | 測試期（驗證泛化能力）|

---

## 🧬 核心功能

### 1️⃣ 三策略合併

```
┌─────────────────────────────────────────────────────────────┐
│                    三策略小小龍                              │
├─────────────────────────────────────────────────────────────┤
│  📈 策略一：低波動本益比策略                                  │
│     - 低融資使用率 + 合理 PE 區間                           │
│     - 營收穩定成長 + ROE 正向                               │
│                                                             │
│  💼 策略二：小資族策略                                       │
│     - 小市值高成長 + 自由現金流為正                          │
│     - RSV 動能選股                                          │
│                                                             │
│  🚀 策略三：營收股價雙渦輪策略                               │
│     - 營收創新高 + 股價創新高                               │
│     - 多頭排列 + 大戶持股高                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2️⃣ 持股權重限制

```python
# 🔥 核心規則：單一標的權重 >= 3%，不滿則不持倉
MIN_POSITION_WEIGHT = 0.03  # 最低持股權重
POSITION_WEIGHT_STEP = 0.03  # 權重必須是 3% 的倍數

# 調整後的權重：0%, 3%, 6%, 9%, 12%, ...
```

### 3️⃣ 適應度函數（含 MDD 懲罰）

```python
def evaluate_fitness(individual):
    """
    多目標適應度評估

    Returns:
        (sharpe_score,      # 夏普值分數 (0-5)
         capacity_score,    # 胃納量分數 (0-5)
         return_score,      # 年化報酬分數 (0-2)
         penalty_score)     # 懲罰分數（MDD + 穩健性）
    """
    # MDD 懲罰
    if max_drawdown > 0.20:
        mdd_penalty = (max_drawdown - 0.20) * 20

    # 穩健性懲罰（Out-of-Sample vs In-Sample）
    if not is_robust:
        penalty -= (1 - robustness) * 3
```

### 4️⃣ 精英保留策略

```python
ELITE_SIZE = 10  # 保留前 10 強精英

# 每代演化後：
# 1. 合併父代 + 子代
# 2. NSGA-II 非支配排序
# 3. 保留前 POPULATION_SIZE 個體
# 4. 更新 Pareto Archive（前 10 強）
```

### 5️⃣ Check-point 存取

```python
# 自動保存
CHECKPOINT_INTERVAL = 5  # 每 5 代保存一次

# 支援斷點續傳
checkpoint_mgr.save(generation, population, pareto_front, history)

# 重啟後自動恢復
checkpoint = checkpoint_mgr.load()
population, start_gen = checkpoint_mgr.restore_population(checkpoint)
```

### 6️⃣ 重啟後自動驗證

```python
AUTO_VERIFY_TOP_N = 5  # 驗證前 5 名

# 重啟時自動執行：
# 1. 載入歷史精英
# 2. 使用最新數據重新 sim() 回測
# 3. 確認策略仍然有效
# 4. 輸出驗證結果
```

---

## 🚀 快速開始

### Google Colab 使用

```python
# ===== 小小龍 v2.0 視窗 1 =====
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['WINDOW_ID'] = "1"  # 視窗 1-4

!pip install -q finlab deap joblib tqdm

!curl -sL "https://raw.githubusercontent.com/windbug001/lolodo/claude/genetic-algorithm-stock-optimizer-hoaRM/finlab_genetic_optimizer_v2.py" \
  -o "/content/drive/MyDrive/FinLab_GA_小小龍_v2/finlab_genetic_optimizer_v2.py"

%cd /content/drive/MyDrive/FinLab_GA_小小龍_v2
%run finlab_genetic_optimizer_v2.py
```

### 多視窗執行

開啟 4 個 Colab 視窗，分別設定：

| 視窗 | 設定 |
|------|------|
| 視窗 1 | `os.environ['WINDOW_ID'] = "1"` |
| 視窗 2 | `os.environ['WINDOW_ID'] = "2"` |
| 視窗 3 | `os.environ['WINDOW_ID'] = "3"` |
| 視窗 4 | `os.environ['WINDOW_ID'] = "4"` |

---

## 📁 輸出檔案結構

```
/content/drive/MyDrive/FinLab_GA_小小龍_v2/
├── shared_pareto/                    # 🔥 共享 Pareto Archive
│   ├── pareto_archive_dragon_w1.pkl
│   ├── pareto_archive_dragon_w2.pkl
│   └── ...
│
├── window_1/                         # 視窗 1 輸出
│   ├── output/
│   │   └── best_params_w1.json       # 最佳參數
│   ├── checkpoints/
│   │   └── checkpoint_w1.pkl         # Check-point
│   └── history/
│
├── window_1_logs/                    # 視窗 1 日誌（供監控）
│   ├── progress_history.json         # 進度歷史
│   └── validation_results.json       # 驗證結果
│
└── ...
```

---

## ⚙️ 參數配置

### GA 演化參數

```python
POPULATION_SIZE = 60      # 族群大小
N_GENERATIONS = 100       # 演化代數
MUTATION_RATE = 0.20      # 變異率
CROSSOVER_RATE = 0.85     # 交叉率
ELITE_SIZE = 10           # 精英保留數量
```

### 優化目標

```python
TARGET_SHARPE = 4.2           # 夏普值目標
MIN_CAPACITY = 10_000_000     # 胃納量目標（1000萬）
TARGET_ANNUAL_RETURN = 0.30   # 年化報酬目標
MAX_DRAWDOWN = 0.20           # 最大回撤限制
```

### 回測時間

```python
IN_SAMPLE_START = '2017-01-01'
IN_SAMPLE_END = '2022-12-31'      # 訓練期結束
OUT_SAMPLE_START = '2023-01-01'   # 測試期開始
OUT_SAMPLE_END = None             # 至今
```

---

## 🧬 基因結構（32 個參數）

| 區段 | 參數數量 | 說明 |
|------|----------|------|
| 策略一（低波動本益比）| 9 | 營收比率、波動閾值、PE 區間等 |
| 策略二（小資族）| 8 | 市值限制、RSV 週期等 |
| 策略三（雙渦輪）| 8 | 營收均線、價格高點窗口等 |
| 策略權重 | 3 | 三策略配置比例 |
| 回測參數 | 4 | 停損、停利、持股上限 |

---

## 📊 輸出範例

```
╔════════════════════════════════════════════════════════════════════╗
║     🐉 三策略小小龍 - 遺傳演算法優化系統 v2.0                        ║
╠════════════════════════════════════════════════════════════════════╣
║  🎯 目標：夏普 4.2+, 胃納量 1000萬+, MDD < 20%                     ║
║  📊 訓練期：2017-01-01 ~ 2022-12-31                                ║
║  📊 測試期：2023-01-01 ~ 至今                                      ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
📋 步驟 1: 驗證歷史精英
======================================================================
🔍 自動驗證歷史前 5 名精英

   驗證第 1 名...
   ✅ 有效 | 夏普: 4.35 | 胃納量: 12.5百萬 | MDD: 18.2%

   驗證第 2 名...
   ✅ 有效 | 夏普: 4.28 | 胃納量: 11.8百萬 | MDD: 17.5%
   ...

======================================================================
📋 步驟 2: 執行遺傳演算法優化
======================================================================

=== 第 5/100 代 🎯 ===
   最佳綜合: 8.4521
   最佳夏普: 4.32 (目標: 4.2)
   最佳胃納量: 11.5 百萬
   穩健性: 0.85
   耗時: 45.2s

======================================================================
🏆 Pareto 最優解集（前 10 名）
======================================================================
排名  綜合      夏普      胃納量(百萬)  達標
----------------------------------------------------------------------
1     8.4521    4.32      11.5          ✅
2     8.2103    4.25      10.8          ✅
3     8.0845    4.18      12.2
...
```

---

## 🔧 常見問題

### Q1: 如何調整目標夏普值？

```python
# 修改 finlab_genetic_optimizer_v2.py 中的設定
TARGET_SHARPE = 4.5  # 調整為 4.5
```

### Q2: 如何增加演化代數？

```python
N_GENERATIONS = 200  # 增加到 200 代
```

### Q3: 如何修改持股權重限制？

```python
MIN_POSITION_WEIGHT = 0.05   # 最低 5%
POSITION_WEIGHT_STEP = 0.05  # 5% 的倍數
```

### Q4: 記憶體不足怎麼辦？

1. 使用 Colab Pro+ 的 High-RAM 模式
2. 減少 `POPULATION_SIZE`
3. 關閉並行運算：`USE_PARALLEL = False`

---

## 📈 版本更新記錄

### v2.0 Dragon Edition (2025-01-12)

- ✅ 目標夏普值提升至 4.2
- ✅ 新增 In-Sample / Out-of-Sample 嚴格分離
- ✅ 新增 3% 最低持股權重限制
- ✅ 新增重啟後自動驗證歷史前 5 名
- ✅ 新增並行運算加速
- ✅ 強化 MDD 懲罰項
- ✅ 精英保留策略（前 10 強）
- ✅ Pickle/Joblib Check-point 存取

---

## 📜 授權

MIT License - 僅供學習研究使用，投資有風險。
