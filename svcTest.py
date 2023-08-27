import pandas as pd
from readData import train_text, train_label
from gensim.models import Word2Vec

# 构建Word2Vec模型
model = Word2Vec.load("word2vec.model")
# 创建一个空的DataFrame来存储评论向量和标签
vectors = []
labels = []

# 遍历每条评论，并将其转换为向量
for comment, label in zip(train_text, train_label):
    words = comment.split()
    sentence_vector = [model.wv[word] for word in words if word in model.wv]  # 将评论拆分为单词并获取每个单词的向量
    if sentence_vector:
        sentence_vector = sum(sentence_vector) / len(sentence_vector)  # 计算平均向量作为整个评论的向量
        vectors.append(sentence_vector)
        labels.append(label)

# 创建包含向量和标签的DataFrame
df = pd.DataFrame({"vector": vectors, "label": labels})

# 保存DataFrame为CSV文件
df.to_csv("comment_vectors.csv", index=False, header=False)
