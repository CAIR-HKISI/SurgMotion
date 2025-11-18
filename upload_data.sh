#!/bin/bash

# 用于打包 draft_data_clean.ipynb (36-42) 定义的四个数据集，并支持断点续传。
# 目标：将每个数据集主目录(data/Surge_Frames/下的级目录)分别打成tar包放到data/CAIR_SDS下，重复运行会跳过已成功的。

DATA_LIST=(
    "Atlas_labeled"
    "Private_pumch_labeled"
    "Private_pwh_labeled"
    "Private_TSS_labeled"
)
SOURCE_ROOT="data/Surge_Frames"
TARGET_ROOT="data/CAIR_SDS"
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

total=${#DATA_LIST[@]}
completed=0
skipped=0
failed=0

log_message "开始打包任务，共 $total 个数据集"
log_message "目标目录: $TARGET_ROOT"

for data_name in "${DATA_LIST[@]}"; do
    echo "------------------------------------------"
    src_dir="$SOURCE_ROOT/$data_name"
    tar_file="$TARGET_ROOT/${data_name}.tar"

    if is_completed "$data_name"; then
        log_message "SKIPPED: $data_name (已完成)"
        ((skipped++))
        continue
    fi

    if [ ! -d "$src_dir" ]; then
        log_message "ERROR: $data_name (源目录不存在: $src_dir)"
        ((failed++))
        continue
    fi

    log_message "START: $data_name"
    # 支持断点续传：如果目标tar已存在且明细已记录为完成，将跳过。
    # 如果tar已存在但没完成（或小于100KB、受损），则删除并重新打包
    if [ -f "$tar_file" ] && ! is_completed "$data_name"; then
        log_message "WARN: $data_name (检测到不完整tar，重新打包)"
        rm -f "$tar_file"
    fi

    # 执行打包（加C保证tar包内层次干净），并实时记录日志
    if tar -cf "$tar_file" -C "$SOURCE_ROOT" "$data_name" 2>&1 | tee -a "$LOG_FILE"; then
        if [ -f "$tar_file" ] && [ -s "$tar_file" ]; then
            log_message "COMPLETED: $data_name (文件大小: $(du -h "$tar_file" | cut -f1))"
            ((completed++))
        else
            log_message "ERROR: $data_name (tar文件创建失败或为空)"
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
log_message "总数: $total, 完成: $completed, 跳过: $skipped, 失败: $failed"

if [ $failed -gt 0 ]; then
    log_message "失败的数据集可以重新运行脚本继续打包"
fi

