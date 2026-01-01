import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------- 1. 定义模型（修正卷积层输出尺寸，匹配64*16*16） ----------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 特征提取器：确保输出特征图尺寸为16x16（以便全连接层输入为64*16*16）
        self.feature_extractor = nn.Sequential(
            # 假设输入图像尺寸为64x64（单通道），经过以下卷积+池化后得到16x16
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 64x64 → 64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 → 32x32（第一次池化）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32 → 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16（第二次池化，得到目标尺寸）
        )
        self.flatten = nn.Flatten()
        # 分类器：输入维度修正为64*16*16（与你提供的参数一致）
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),  # 64通道×16×16特征图，展平后作为输入
            nn.ReLU(),
            nn.Linear(256, 10)  # 10分类任务（根据实际类别数调整）
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ---------------------- 2. 设备配置与初始化 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()  # 多分类损失
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)  # 增加momentum加速收敛


# ---------------------- 3. 训练/测试函数（逻辑不变，确保数据尺寸匹配） ----------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 100 == 0 or batch == num_batches - 1:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.shape[0]  # 累积总损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


# ---------------------- 4. 训练循环（适配64x64输入图像） ----------------------
if __name__ == "__main__":
    # 模拟64x64单通道图像的数据集（匹配模型输入尺寸）
    class DummyDataset(Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            # 输入尺寸为(1, 64, 64)，与模型要求一致
            return torch.randn(1, 64, 64), torch.randint(0, 10, (1,)).item()


    train_dataloader = DataLoader(DummyDataset(), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(DummyDataset(), batch_size=64, shuffle=False)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")