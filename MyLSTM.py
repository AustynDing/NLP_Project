import torch
from torch import nn
from torch.autograd import Variable


# 按照阈值进行分类
def getBinaryTensor(tensor, boundary=0.5):
    one = torch.ones_like(tensor)
    zero = torch.zeros_like(tensor)
    binary_tensor = torch.where(tensor > boundary, one, zero)
    return binary_tensor


class MyLSTM(nn.Module):
    # input_size:这是输入数据的特征维度。每个句子的词向量表示是输入数据的特征
    # hidden_dim:这是 LSTM 中隐藏状态的维度，也称为 LSTM 单元中的单元数。较大的隐藏状态维度可以捕捉更复杂的模式，但也可能导致更多的计算开销。
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(MyLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        # 此处 input_size 是我们 word2vec 的词向量的维度；
        # 这里设置了输入的第一个维度为 batchsize，那么在后面构造输入的时候，需要保证第一个维度是 batch size 数量
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # 创建一个线性层对象，将LSTM的输出映射到二分类的结果上.hidden_dim表示输入维度,1表示输出维度（二分类）
        self.softmax = nn.Softmax(dim=output_size)  # 创建一个softmax层对象，用于将输出进行归一化，得到概率分布.dim=1表示对第一维进行softmax操作（即每个样本的输出）。

    def init_hidden(self, batch_size):  # 初始化两个隐藏向量 h0 和 c0
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

    def forward(self, x):  # 不可以自己显式调用，pytorch 内部自带调用机制
        # input 是传递给 lstm 的输入，它的 shape 应该是（每一个文本的词语数量，batch size，词向量维度）
        # 输入的时候需要将 input 构造成
        self.hidden = self.init_hidden(x.size(0))  # input.size(0)得到 batch_size
        # x = x.view(len(x), 1, -1).to(torch.float32)  # 通过使用view直接修改维度，并修改精度
        # https://blog.csdn.net/weixin_35757704/article/details/118384899
        # https://blog.csdn.net/qq_19841133/article/details/127824863
        lstm_out, _ = self.lstm(x, self.hidden)
        lstm_out = self.fc(lstm_out[:, -1, :])
        lstm_out = torch.sigmoid(lstm_out)  # Sigmoid激活函数处理后的输出值（0到1之间的概率）代表了模型对样本属于正类的置信度。
        lstm_out = getBinaryTensor(lstm_out, torch.mean(lstm_out).item())  # 使用平均值作为阈值，将概率映射到0或者1上
        # lstm_out = self.softmax(lstm_out)
        lstm_out = lstm_out.squeeze()  # 使用squeeze对维数进行压缩
        # print(lstm_out, lstm_out.shape)  #
        return lstm_out  # 查看文档，了解 lstm_out 到底是什么

# "batch size"（批次大小）是深度学习中一个重要的超参数，它定义了在训练过程中一次迭代所使用的样本数量。
# 具体来说，每个迭代步骤中，模型会根据给定的批次大小从训练数据中随机选择一组样本进行前向传播、计算损失和反向传播。
# 批次大小的选择会影响训练过程的速度和内存消耗。
# 较大的批次大小可以加快训练速度，因为可以并行处理更多的数据，但也可能导致内存占用较大。
# 较小的批次大小可能会让训练过程更稳定，但可能会增加每个迭代步骤之间的计算负担。
