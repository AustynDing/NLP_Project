import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset


# 注意这里是继承了一个 Dataset 基础类，很多功能是这个基础类就有的，
# 我们只需要按照这个示例实现__init__和__getitem__就行了
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        # 这里使用 pandas 读取 csv 文件，参数应该是一个文件路径，这个文件里面放的是每一张图片的类型，图片是另外存放的
        # 我们的数据不同，我们是评论的向量和标签放在一起的
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    # 这个方法是关键，需要我们根据 idx 返回每一条对应序号的数据，包括 x 和 y，这个例子中 x 是 image,y 是 label
    # 我们的评论数据中，x 是转换之后的 word vector 数组，y 是极性标签
    # 我们的数据集比这个示例要简单，可以直接在__init__中将所有数据以数组读出来，然后在__getitem__中用 idx 获取就行了
    def __getitem__(self, idx):
        vector_str = self.data.loc[idx, 'vector']
        # 解析 CSV 中的向量字符串为 NumPy 数组
        vector = np.array(eval(vector_str))
        label = int(self.data.loc[idx, 'label'])

        if self.transform:
            vector = self.transform(vector)
        if self.target_transform:
            label = self.target_transform(label)

        return vector, label