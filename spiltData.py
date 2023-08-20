from sklearn.utils import shuffle
from loadData import *

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

# 打印统计结果
print("Training Set:")
print("Positive samples:", sum(train_label_split), "Negative samples:", len(train_label_split) - sum(train_label_split))
print("Average length:", avg_length_train, "Max length:", max_length_train, "Min length:", min_length_train)
print()

print("Validation Set:")
print("Positive samples:", sum(val_label), "Negative samples:", len(val_label) - sum(val_label))
print("Average length:", avg_length_val, "Max length:", max_length_val, "Min length:", min_length_val)
print()

print("Test Set:")
print("Positive samples:", sum(test_label_split), "Negative samples:", len(test_label_split) - sum(test_label_split))
print("Average length:", avg_length_test, "Max length:", max_length_test, "Min length:", min_length_test)
