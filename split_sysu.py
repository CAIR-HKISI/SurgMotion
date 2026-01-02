import pandas as pd

step_csv_path="data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/step_all_metadata.csv"
skill_csv_path="data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/skill_all_metadata.csv"

## 统计标签分布
step_df = pd.read_csv(step_csv_path)
skill_df = pd.read_csv(skill_csv_path)

step_df['label'].value_counts()
skill_df['label'].value_counts()

## 用表格打印结果
print(step_df['label'].value_counts().to_frame())
print(skill_df['label'].value_counts().to_frame())

## 对于step 过滤掉label为10的样本
## 对于skill，过滤掉label为0的样本
step_df = step_df[step_df['label'] != 10]
skill_df = skill_df[skill_df['label'] != 0]

## 重新映射标签，使其从0开始连续
def remap_labels(df, name):
    unique_labels = sorted(df['label'].unique())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    print(f"Remapping {name} labels: {label_map}")
    df['label'] = df['label'].map(label_map)
    return df

step_df = remap_labels(step_df, "step")
skill_df = remap_labels(skill_df, "skill")

## 用表格打印结果
print(step_df['label'].value_counts().to_frame())
print(skill_df['label'].value_counts().to_frame())

# 按照case id 进行划分，70%作为train，30%作为test
all_case_ids = pd.Series(step_df['case_id'].unique())
train_case_ids = all_case_ids.sample(frac=0.7, random_state=42)
test_case_ids = all_case_ids[~all_case_ids.isin(train_case_ids)]

# 用表格打印结果
print("Train Cases:")
print(train_case_ids.to_frame(name='case_id'))
print("\nTest Cases:")
print(test_case_ids.to_frame(name='case_id'))

## 统计训练和测试的标签分布，防止出现不均衡
train_df = step_df[step_df['case_id'].isin(train_case_ids)].groupby('label').size().to_frame()
test_df = step_df[step_df['case_id'].isin(test_case_ids)].groupby('label').size().to_frame()

print(train_df)
print(test_df)

### 写入train和test的csv
# 生成文件路径
step_train_path = step_csv_path.replace('all', 'train')
step_test_path = step_csv_path.replace('all', 'test')
skill_train_path = skill_csv_path.replace('all', 'train')
skill_test_path = skill_csv_path.replace('all', 'test')

# 过滤数据
step_train_df = step_df[step_df['case_id'].isin(train_case_ids)]
step_test_df = step_df[step_df['case_id'].isin(test_case_ids)]

skill_train_df = skill_df[skill_df['case_id'].isin(train_case_ids)]
skill_test_df = skill_df[skill_df['case_id'].isin(test_case_ids)]

# 写入CSV
step_train_df.to_csv(step_train_path, index=False)
step_test_df.to_csv(step_test_path, index=False)
skill_train_df.to_csv(skill_train_path, index=False)
skill_test_df.to_csv(skill_test_path, index=False)

print(f"Step train data saved to {step_train_path}")
print(f"Step test data saved to {step_test_path}")
print(f"Skill train data saved to {skill_train_path}")
print(f"Skill test data saved to {skill_test_path}")

