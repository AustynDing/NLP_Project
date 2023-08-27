from sklearn.utils import shuffle
from loadData import *
import json

# 合并训练集和测试集的数据和标签
all_text = train_text + test_text
all_label = train_label + test_label

# 随机洗牌，确保数据集的样本分布随机且均匀
all_text, all_label = shuffle(all_text, all_label, random_state=42)

# 计算总样本数量
total_samples = len(all_text)

# 划分比例
train_size = int(0.7 * total_samples)
val_size = int(0.1 * total_samples)

# 划分训练集、验证集和测试集
train_text_split = all_text[:train_size]
train_label_split = all_label[:train_size]
val_text = all_text[train_size:train_size + val_size]
val_label = all_label[train_size:train_size + val_size]
test_text_split = all_text[train_size + val_size:]
test_label_split = all_label[train_size + val_size:]


# 计算评论的平均长度、最大长度和最小长度
def calculate_lengths(text_data):
    lengths = [len(sentence.split()) for sentence in text_data]
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    return avg_length, max_length, min_length


avg_length_train, max_length_train, min_length_train = calculate_lengths(train_text_split)
avg_length_val, max_length_val, min_length_val = calculate_lengths(val_text)
avg_length_test, max_length_test, min_length_test = calculate_lengths(test_text_split)

# 保存数据到文件
data = {
    "train": {"text": train_text_split, "label": train_label_split},
    "validation": {"text": val_text, "label": val_label},
    "test": {"text": test_text_split, "label": test_label_split}
}

with open("data.json", "w") as outfile:
    json.dump(data, outfile)