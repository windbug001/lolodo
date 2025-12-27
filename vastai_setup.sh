#!/bin/bash
# ==============================================================================
# Vast.ai 一鍵設置腳本 - FinLab GA 遺傳演算法優化系統
# ==============================================================================

echo "========================================"
echo "🚀 Vast.ai 環境設置開始"
echo "========================================"

# 1. 更新系統
echo "📦 更新系統套件..."
apt update -qq

# 2. 安裝必要工具
echo "📦 安裝必要工具..."
apt install -y -qq tmux wget curl git

# 3. 安裝 Python 套件
echo "📦 安裝 Python 套件..."
pip install -q finlab deap joblib scikit-learn pandas numpy requests

# 4. 創建工作目錄
echo "📁 創建工作目錄..."
mkdir -p /root/ga_optimizer
cd /root/ga_optimizer

# 5. 下載程式
echo "📥 下載優化程式..."
BRANCH="claude/add-backtesting-monitoring-eC5lt"
BASE_URL="https://raw.githubusercontent.com/windbug001/lolodo/${BRANCH}"

wget -q "${BASE_URL}/finlab_genetic_optimizer_ultimate.py" -O ga_3strategy.py
wget -q "${BASE_URL}/genetic_algo_v10_ultimate.py" -O ga_6strategy.py
wget -q "${BASE_URL}/finlab_ga_progress_monitor.py" -O monitor.py

# 6. 創建多視窗啟動腳本
echo "📝 創建啟動腳本..."

cat > /root/ga_optimizer/start_windows.sh << 'SCRIPT'
#!/bin/bash
# 啟動多視窗優化

STRATEGY=$1  # "3" or "6"
START_WINDOW=$2  # 起始視窗編號
COUNT=$3  # 視窗數量

if [ -z "$STRATEGY" ] || [ -z "$START_WINDOW" ] || [ -z "$COUNT" ]; then
    echo "用法: ./start_windows.sh <策略:3或6> <起始視窗> <視窗數量>"
    echo "範例: ./start_windows.sh 3 6 5  # 啟動 3策略的視窗 6-10"
    exit 1
fi

if [ "$STRATEGY" == "3" ]; then
    SCRIPT_FILE="ga_3strategy.py"
elif [ "$STRATEGY" == "6" ]; then
    SCRIPT_FILE="ga_6strategy.py"
else
    echo "策略必須是 3 或 6"
    exit 1
fi

cd /root/ga_optimizer

for i in $(seq 0 $((COUNT-1))); do
    WINDOW_ID=$((START_WINDOW + i))
    SESSION_NAME="w${WINDOW_ID}"
    SCRIPT_COPY="ga_w${WINDOW_ID}.py"

    # 複製並修改視窗 ID
    cp $SCRIPT_FILE $SCRIPT_COPY
    sed -i "s/WINDOW_ID = 1/WINDOW_ID = ${WINDOW_ID}/" $SCRIPT_COPY

    # 在 tmux 中啟動
    tmux new-session -d -s $SESSION_NAME "python $SCRIPT_COPY 2>&1 | tee log_w${WINDOW_ID}.txt"

    echo "✅ 視窗 ${WINDOW_ID} 已啟動 (tmux session: ${SESSION_NAME})"
    sleep 2
done

echo ""
echo "========================================"
echo "🎉 所有視窗已啟動!"
echo "========================================"
echo ""
echo "查看所有視窗: tmux ls"
echo "進入視窗 N:   tmux attach -t wN"
echo "離開視窗:     Ctrl+B 然後按 D"
echo "查看日誌:     tail -f log_wN.txt"
SCRIPT

chmod +x /root/ga_optimizer/start_windows.sh

# 7. 創建監控腳本
cat > /root/ga_optimizer/check_status.sh << 'SCRIPT'
#!/bin/bash
echo "========================================"
echo "📊 執行狀態檢查"
echo "========================================"
echo ""
echo "🖥️ 運行中的 tmux sessions:"
tmux ls 2>/dev/null || echo "   (無)"
echo ""
echo "📈 最新進度:"
for f in /root/ga_optimizer/log_w*.txt; do
    if [ -f "$f" ]; then
        WINDOW=$(basename $f .txt | sed 's/log_//')
        LAST_LINE=$(tail -1 "$f" 2>/dev/null)
        echo "   $WINDOW: $LAST_LINE"
    fi
done
echo ""
echo "💾 磁碟使用:"
df -h /root | tail -1
echo ""
echo "🧠 記憶體使用:"
free -h | grep Mem
SCRIPT

chmod +x /root/ga_optimizer/check_status.sh

# 8. 創建 FinLab API Key 設定提示
cat > /root/ga_optimizer/README.txt << 'README'
==============================================================================
🚀 FinLab GA 優化系統 - Vast.ai 版
==============================================================================

【重要】首次使用請設定 FinLab API Key:
   編輯 ga_3strategy.py 或 ga_6strategy.py
   找到 FINLAB_API_KEY = "..." 這行，填入你的 API Key

【啟動視窗】
   # 啟動 3策略 視窗 6-10 (共5個)
   ./start_windows.sh 3 6 5

   # 啟動 6策略 視窗 6-10 (共5個)
   ./start_windows.sh 6 6 5

【管理視窗】
   tmux ls                  # 查看所有視窗
   tmux attach -t w6        # 進入視窗 6
   Ctrl+B 然後按 D          # 離開視窗(不中斷)
   tmux kill-session -t w6  # 關閉視窗 6

【查看狀態】
   ./check_status.sh        # 查看所有視窗狀態
   tail -f log_w6.txt       # 即時查看視窗 6 日誌

【檔案說明】
   ga_3strategy.py  - 3策略優化程式
   ga_6strategy.py  - 6策略優化程式
   monitor.py       - 進度監控程式
   log_wN.txt       - 視窗 N 的執行日誌

==============================================================================
README

# 完成
echo ""
echo "========================================"
echo "🎉 設置完成!"
echo "========================================"
echo ""
echo "📁 工作目錄: /root/ga_optimizer"
echo ""
echo "📖 使用說明:"
echo "   cd /root/ga_optimizer"
echo "   cat README.txt"
echo ""
echo "🚀 快速啟動 (視窗 6-10):"
echo "   ./start_windows.sh 3 6 5   # 3策略"
echo "   ./start_windows.sh 6 6 5   # 6策略"
echo ""
