import pandas as pd
from readData import *
from gensim.models import Word2Vec


def convert_comments_to_vectors(model, text, labels, output_file):
    # 创建一个空的列表来存储评论向量和标签
    vectors = []
    labelList = []

    # 遍历每条评论，并将其转换为向量
    for comment, label in zip(text, labels):
        words = comment.split()
        sentence_vector = [model.wv[word] for word in words if word in model.wv]
        if sentence_vector:
            sentence_vector = sum(sentence_vector) / len(sentence_vector)
            vectors.append(sentence_vector)
            labelList.append(label)

    # 创建包含向量和标签的DataFrame
    df = pd.DataFrame({"vector": vectors, "label": labelList})

    # 保存DataFrame为CSV文件
    df.to_csv(output_file, index=False)


# 构建Word2Vec模型
model = Word2Vec.load("word2vec.model")
convert_comments_to_vectors(model, train_text, train_label, 'train.csv')
convert_comments_to_vectors(model, test_text, test_label, 'test.csv')
convert_comments_to_vectors(model, val_text, val_label, 'validate.csv')
