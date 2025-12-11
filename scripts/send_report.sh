#!/bin/bash
# Send Performance Report to Telegram
# ====================================
# Use with cron: 0 */2 * * * /path/to/send_report.sh
#
# This script:
# 1. Loads environment from .env
# 2. Runs the performance report with --send-telegram
# 3. Logs output to a file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/logs/report_cron.log"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/logs"

# Log start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 2-hour performance report..." >> "$LOG_FILE"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment if it exists
if [ -f "venv_ml/bin/activate" ]; then
    source venv_ml/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Run report
python -m reports.performance_report --hours 2 --send-telegram >> "$LOG_FILE" 2>&1

# Log completion
if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report sent successfully" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report failed with exit code $?" >> "$LOG_FILE"
fi

# Keep log file from growing too large (keep last 1000 lines)
tail -n 1000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
