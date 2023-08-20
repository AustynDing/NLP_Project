from gensim.models import Word2Vec
from loadData import train_text
train_text = [sentence.split() for sentence in train_text]
print(train_text[:2])


# 调用Word2Vec训练 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(train_text, vector_size=20, window=2, min_count=3, epochs=5, negative=10, sg=1)
print("has的词向量：\n", model.wv.get_vector('has'))
print("\n和has相关性最高的前20个词语：")
print(model.wv.most_similar('has', topn=20))  # 与Nation最相关的前20个词语
