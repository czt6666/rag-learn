# data
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view((x.size(0), -1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(5):
#     total_loss = 0
#     for images, labels in train_loader:
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#
#         # loss and optimizer
#         optimizer.zero_grad()  # 清梯度
#         loss.backward()  # 算梯度
#         optimizer.step()  # 更新参数
#
#     print(f"epoch {epoch}, loss = {total_loss / len(train_loader)}")
#
# torch.save(model.state_dict(), "mnist_model.pth")


model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# eval
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print("accuracy:", accuracy)

# 可视化
# images, labels = next(iter(test_loader))
# outputs = model(images)
# _, predicted = torch.max(outputs, 1)
#
# index = 30
# plt.imshow(images[index].squeeze(), cmap="gray")
# plt.title(f"true={labels[index].item()}, pred={predicted[index].item()}")
# plt.show()


# 可视化 2
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#
#         wrong_idx = (predicted != labels)
#         print(wrong_idx)
#
#         if wrong_idx.sum() > 0:
#             wrong_images = images[wrong_idx]
#             wrong_labels = labels[wrong_idx]
#             wrong_preds = predicted[wrong_idx]
#
#             # 只展示前5个错误
#             for i in range(min(15, wrong_images.size(0))):
#                 plt.imshow(wrong_images[i].squeeze(), cmap="gray")
#                 plt.title(f"true={wrong_labels[i].item()}, pred={wrong_preds[i].item()}")
#                 plt.show()
