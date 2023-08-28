import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from customDataset import CustomDataset
from MyLSTM import MyLSTM


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),  # 输入512个神经元，输出512个神经元
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):  # 不可以自己显式调用，pytorch内部自带调用机制
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):  # 模型训练过程的定义
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)  # 获取模型的预测结果
        loss = loss_fn(pred, y)  # 计算预测误差

        # Backpropagation
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清零梯度

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):  # 模型测试过程的定义
    size = len(dataloader.dataset)  # 数据集大小
    num_batches = len(dataloader)  # 批次数
    model.eval()  # 设置模型为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # 将数据移到设备上
            pred = model(X)  # 获取模型的预测结果
            test_loss += loss_fn(pred, y).item()  # pred 是模型的预测输出，y 是真实标签。loss 是计算得到的交叉熵损失值，表示模型预测与真实标签之间的差异程度。
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # Download training data from open datasets.
    training_data = CustomDataset('train.csv')
    # Download test data from open datasets.
    test_data = CustomDataset('test.csv')

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = MyLSTM(20, 2).to(device)

    loss_fn = nn.CrossEntropyLoss()  # 损失函数用于衡量模型的预测结果与真实标签之间的差距，或者说是预测的准确程度
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model/haha")
    print("Saved PyTorch Model State to the project root folder!")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
