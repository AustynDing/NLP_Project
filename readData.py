import json

# 读取数据
with open("data.json", "r") as infile:
    loaded_data = json.load(infile)

train_data = loaded_data["train"]
val_data = loaded_data["validation"]
test_data = loaded_data["test"]

# 获取训练集、验证集和测试集的text和label
train_text = loaded_data["train"]["text"]
train_label = loaded_data["train"]["label"]

val_text = loaded_data["validation"]["text"]
val_label = loaded_data["validation"]["label"]

test_text = loaded_data["test"]["text"]
test_label = loaded_data["test"]["label"]

# # 打印训练集中的第一条数据
# print("Train Text:", train_text[0])
# print("Train Label:", train_label[0])
#
# # 打印验证集中的第一条数据
# print("Validation Text:", val_text[0])
# print("Validation Label:", val_label[0])
#
# # 打印测试集中的第一条数据
# print("Test Text:", test_text[0])
# print("Test Label:", test_label[0])