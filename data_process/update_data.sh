#!/bin/bash

# 打包 data/Surge_Frames 下的数据集到 data/CAIR_SDS 中
# 自动为更新创建新版本 (如 dataset_v2.tar, dataset_v3.tar)
# 支持断点续传与日志记录

DATA_LIST=(
    # "SurgAction160"
    "GynSurg_Action"
    "GynSurg_Smoke"
    "GynSurg_Bleeding"
)
SOURCE_ROOT="data/Surge_Frames"
TARGET_ROOT="data/CAIR_SDS/Surgery_frames"
LOG_FILE="packaging_Surge_Frames.log"

mkdir -p "$TARGET_ROOT"
touch "$LOG_FILE"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

is_completed() {
    local dataset="$1"
    grep -q "COMPLETED: $dataset" "$LOG_FILE"
    return $?
}

# 自动获取新的版本号
get_next_version_file() {
    local base_name="$1"
    local ext="$2"
    local i=1
    local new_file="$TARGET_ROOT/${base_name}.${ext}"
    while [ -f "$new_file" ]; do
        ((i++))
        new_file="$TARGET_ROOT/${base_name}_v${i}.${ext}"
    done
    echo "$new_file"
}

total=${#DATA_LIST[@]}
completed=0
skipped=0
failed=0

log_message "开始打包任务，共 $total 个数据集"
log_message "目标目录: $TARGET_ROOT"

for data_name in "${DATA_LIST[@]}"; do
    echo "------------------------------------------"
    src_dir="$SOURCE_ROOT/$data_name"

    if [ ! -d "$src_dir" ]; then
        log_message "ERROR: $data_name (源目录不存在: $src_dir)"
        ((failed++))
        continue
    fi

    # 找一个新的 tar 文件名（支持版本号）
    tar_file=$(get_next_version_file "$data_name" "tar")

    log_message "START: $data_name → $(basename "$tar_file")"

    if tar -cf "$tar_file" -C "$SOURCE_ROOT" "$data_name" 2>&1 | tee -a "$LOG_FILE"; then
        if [ -f "$tar_file" ] && [ -s "$tar_file" ]; then
            log_message "COMPLETED: $data_name (文件: $(basename "$tar_file"), 大小: $(du -h "$tar_file" | cut -f1))"
            ((completed++))
        else
            log_message "ERROR: $data_name (tar文件为空或创建失败)"
            rm -f "$tar_file"
            ((failed++))
        fi
    else
        log_message "ERROR: $data_name (打包过程出错)"
        rm -f "$tar_file"
        ((failed++))
    fi
done

echo "=========================================="
log_message "打包任务完成!"
log_message "总数: $total, 完成: $completed, 失败: $failed"

if [ $failed -gt 0 ]; then
    log_message "失败的数据集可以重新运行脚本继续打包"
fi

