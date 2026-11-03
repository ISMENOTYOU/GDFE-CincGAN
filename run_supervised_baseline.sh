#!/usr/bin/env bash
# Run from the Linux server. Keep this file next to train.py.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

srgan_python="${SRGAN_PYTHON:-/home/dell/anaconda3/envs/yfr/bin/python}"
if [[ ! -x "$srgan_python" ]]; then
    printf 'Python executable not found: %s\n' "$srgan_python" >&2
    exit 1
fi
if pgrep -u "$(id -u)" -af '[t]rain.py'; then
    printf 'An existing train.py process was found. Check and stop that task before starting this experiment.\n' >&2
    exit 1
fi

srgan_baseline_tag="$(date +%Y%m%d_%H%M%S)"
srgan_baseline_dir="logs_supervised_${srgan_baseline_tag}"
srgan_baseline_log="train_supervised_${srgan_baseline_tag}.log"

# Requires the patched train.py / utils files. No old checkpoint is loaded.
# G/G1 have only direct MSE objectives for all 20 epochs. G2/F learn detached
# inverse mappings independently; discriminator updates and feedback are off.
nohup env CUDA_VISIBLE_DEVICES=0 SRGAN_BATCH_SIZE=7 \
    SRGAN_EPOCHS=20 SRGAN_WARMUP_EPOCHS=20 \
    SRGAN_DATASET_DIR="${SRGAN_DATASET_DIR:-./Unpaired henan}" \
    SRGAN_INIT_LR=5e-5 SRGAN_D_INIT_LR=1e-5 SRGAN_GRAD_CLIP=1 \
    SRGAN_RESUME= SRGAN_SAVE_DIR="$srgan_baseline_dir" \
    "$srgan_python" -u ./train.py > "$srgan_baseline_log" 2>&1 < /dev/null &
srgan_baseline_pid=$!
printf '%s\n' "$srgan_baseline_pid" > train_baseline.pid
printf '%s\n' "$srgan_baseline_log" > train_baseline_log.txt
printf '%s\n' "$srgan_baseline_dir" > train_baseline_dir.txt
printf 'Started PID: %s\nResults: %s/%s\nLog: %s/%s\n' \
    "$srgan_baseline_pid" "$PWD" "$srgan_baseline_dir" "$PWD" "$srgan_baseline_log"
printf 'Monitor: tail -n 20 -f "$(cat train_baseline_log.txt)"\n'
