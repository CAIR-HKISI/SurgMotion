#!/bin/bash

# # 配置参数
# SOURCE_DIR="data/Landscopy"
# TARGET_DIR="data/CAIR_SDS/Landscopy"
# LOG_FILE="packaging.log"



# 配置参数
# data_name="Ophthalmology"
# SOURCE_DIR="data/${data_name}"
# TARGET_DIR="data/CAIR_SDS/${data_name}"
# LOG_FILE="packaging_${data_name}.log"


data_name="Open_surgery"
SOURCE_DIR="data/${data_name}"
TARGET_DIR="data/CAIR_SDS/${data_name}"
LOG_FILE="packaging_${data_name}.log"

# 创建目标目录和日志文件
mkdir -p "$TARGET_DIR"
touch "$LOG_FILE"

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查是否已完成打包
is_completed() {
    local dataset="$1"
    grep -q "COMPLETED: $dataset" "$LOG_FILE"
    return $?
}

# 获取所有数据集目录
datasets=($(ls -d $SOURCE_DIR/*/ 2>/dev/null | xargs -n 1 basename))

log_message "开始打包任务，总共 ${#datasets[@]} 个数据集"
log_message "目标目录: $TARGET_DIR"

# 统计信息
total=${#datasets[@]}
completed=0
skipped=0
failed=0

# 循环处理每个数据集
for dataset in "${datasets[@]}"; do
    echo "----------------------------------------"
    
    # 检查是否已经完成
    if is_completed "$dataset"; then
        log_message "SKIPPED: $dataset (已完成)"
        ((skipped++))
        continue
    fi
    
    # 检查源目录是否存在
    if [ ! -d "$SOURCE_DIR/$dataset" ]; then
        log_message "ERROR: $dataset (源目录不存在)"
        ((failed++))
        continue
    fi
    
    # 开始打包
    log_message "START: $dataset"
    tar_file="$TARGET_DIR/${dataset}.tar"
    
    # 执行打包
    if tar -cf "$tar_file" -C "$SOURCE_DIR" "$dataset" 2>&1 | tee -a "$LOG_FILE"; then
        # 验证tar文件是否创建成功且大小大于0
        if [ -f "$tar_file" ] && [ -s "$tar_file" ]; then
            log_message "COMPLETED: $dataset (文件大小: $(du -h "$tar_file" | cut -f1))"
            ((completed++))
        else
            log_message "ERROR: $dataset (tar文件创建失败或为空)"
            rm -f "$tar_file"  # 删除失败的文件
            ((failed++))
        fi
    else
        log_message "ERROR: $dataset (打包过程出错)"
        rm -f "$tar_file"  # 删除失败的文件
        ((failed++))
    fi
done

# 输出最终统计
echo "========================================"
log_message "打包任务完成!"
log_message "总数: $total, 完成: $completed, 跳过: $skipped, 失败: $failed"

# 显示未完成的数据集
if [ $failed -gt 0 ]; then
    log_message "失败的数据集可以重新运行脚本继续打包"
fi