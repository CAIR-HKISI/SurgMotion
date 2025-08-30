# M2CAI2016 数据集处理结果

## 概述
本目录包含了处理后的M2CAI2016数据集CSV文件，用于手术阶段识别任务。

## 生成的文件
- `train_metadata.csv`: 训练集元数据 (67,569 条记录)
- `test_metadata.csv`: 测试集元数据 (26,957 条记录)  
- `val_metadata.csv`: 验证集元数据 (26,957 条记录)

## 数据格式
每个CSV文件包含以下列：
- `index`: 帧索引 (1fps)
- `Hospital`: 医院名称 (固定为 "m2cai2016")
- `Year`: 年份 (固定为 2016)
- `Case_Name`: 视频名称
- `Case_ID`: 视频ID
- `Frame_Path`: 帧文件路径
- `Phase_GT`: 阶段标签 (数字编码 0-7)
- `Phase_Name`: 阶段名称
- `Split`: 数据集分割 (train/test/val)

## 阶段映射
| Phase_GT | Phase_Name |
|----------|------------|
| 0 | TrocarPlacement |
| 1 | Preparation |
| 2 | CalotTriangleDissection |
| 3 | ClippingCutting |
| 4 | GallbladderDissection |
| 5 | GallbladderPackaging |
| 6 | CleaningCoagulation |
| 7 | GallbladderRetraction |

## 数据集统计

### 训练集
- 总记录数: 67,569
- 视频数量: 27
- 阶段分布:
  - CalotTriangleDissection: 17,062
  - GallbladderDissection: 16,850
  - CleaningCoagulation: 8,529
  - GallbladderRetraction: 7,998
  - ClippingCutting: 7,607
  - TrocarPlacement: 4,913
  - Preparation: 2,763
  - GallbladderPackaging: 1,847

### 测试集
- 总记录数: 26,957
- 视频数量: 14
- 阶段分布:
  - CalotTriangleDissection: 8,765
  - GallbladderDissection: 6,267
  - CleaningCoagulation: 3,749
  - ClippingCutting: 2,345
  - TrocarPlacement: 1,866
  - GallbladderRetraction: 1,561
  - GallbladderPackaging: 1,529
  - Preparation: 875

## 处理说明
1. 原始标注文件为25fps，转换为1fps
2. 检查帧文件存在性，只保留存在的帧
3. 去除了重复的帧记录
4. 验证集使用测试集数据

## 使用方法
```python
import pandas as pd

# 读取训练集
train_df = pd.read_csv('train_metadata.csv')

# 读取测试集
test_df = pd.read_csv('test_metadata.csv')

# 读取验证集
val_df = pd.read_csv('val_metadata.csv')
```
