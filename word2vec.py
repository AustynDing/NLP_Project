from gensim.models import Word2Vec
from loadData import train_text
train_text = [sentence.split() for sentence in train_text]
# train_text: 预处理后的训练文本数据。
# vector_size: 词向量的维度，这里设置为20。
# window: 上下文窗口大小，表示在预测当前单词时考虑前后的上下文词汇数，这里设置为2。
# min_count: 考虑计算的单词的最低词频阈值，出现次数低于此阈值的单词将被忽略，这里设置为3。
# epochs: 训练的迭代次数，这里设置为5。
# negative: 负采样的数量，用于优化模型训练，这里设置为10。
# sg: 训练算法的选择，sg=1表示使用Skip-gram算法，sg=0表示使用CBOW算法
# 调用Word2Vec训练 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(train_text, vector_size=20, window=2, min_count=3, epochs=5, negative=10, sg=1)
# print("has的词向量：\n", model.wv.get_vector('has'))
# print("\n和has相关性最高的前20个词语：")
# print(model.wv.most_similar('has', topn=20))  # 与Nation最相关的前20个词语
# 保存模型
model.save("word2vec.model")

# vector = model.wv['computer']
# print(vector)
# [ 0.31186014 -0.17845365  1.0966598  -0.10880593 -0.05730803  0.02422507
#   0.41112718  0.9443151  -0.36718005  0.55546355  0.77747893  0.32190266
#  -0.3869654  -0.13872573  0.19698325 -0.3372114   0.87689173 -0.55207354
#   0.18718082 -0.71431994]
