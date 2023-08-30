import torch
from torch import nn
from torch.utils.data import DataLoader
from customDataset import CustomDataset
from MyLSTM import MyLSTM


def train(dataloader, model, loss_fn, optimizer):  # 模型训练过程的定义
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)  # 获取模型的预测结果
        # print('pred: ', pred, 'y: ', y)
        loss = loss_fn(pred, y)  # 计算预测误差

        # Backpropagation
        loss.requires_grad = True
        # 解决RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn 报错问题
        loss.retain_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清零梯度

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, funcName):  # 模型测试过程的定义
    size = len(dataloader.dataset)  # 数据集大小
    num_batches = len(dataloader)  # 批次数
    model.eval()  # 设置模型为评估模式
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # 将数据移到设备上
            pred = model(X)  # 获取模型的预测结果
            loss += loss_fn(pred, y).item()  # pred 是模型的预测输出，y 是真实标签。loss 是计算得到的交叉熵损失值，表示模型预测与真实标签之间的差异程度。
            correct += (pred == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"{funcName} Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg {funcName} loss: {loss:>8f} \n")


if __name__ == '__main__':
    training_data = CustomDataset('train.csv')
    test_data = CustomDataset('test.csv')
    vali_data = CustomDataset('validate.csv')

    batch_size = 64
    input_size = 20  # 词向量维度
    hidden_size = 128  # 隐藏层维度
    num_layers = 2  # LSTM 层数
    output_size = 1  # 输出维度，二分类问题

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    vali_dataloader = DataLoader(vali_data, batch_size=batch_size)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = MyLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    loss_fn = nn.BCELoss()  # 损失函数用于衡量模型的预测结果与真实标签之间的差距，或者说是预测的准确程度
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(train_dataloader, model, loss_fn, optimizer)
    model.eval()
    test(test_dataloader, model, loss_fn, 'Test')
    test(vali_dataloader, model, loss_fn, 'Validate')
    print("Done!")

    torch.save(model.state_dict(), "model/haha")
    print("Saved PyTorch Model State to the project root folder!")
