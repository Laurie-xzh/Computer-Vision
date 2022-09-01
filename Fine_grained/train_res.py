# 导入库

import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 设置超参数

BATCH_SIZE = 32

EPOCHS = 15

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda')
# 数据预处理

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomVerticalFlip(),

    transforms.RandomCrop(50),

    transforms.RandomResizedCrop(150),

    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

])

# 读取数据

dataset_train = datasets.ImageFolder('E:\\code\\csdn\\data\\train', transform)

print(dataset_train.imgs)

# 对应文件夹的label

print(dataset_train.class_to_idx)

dataset_test = datasets.ImageFolder('E:\\code\\csdn\\data\\val', transform)

# 对应文件夹的label

print(dataset_test.class_to_idx)

# 导入数据

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


# 定义网络

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7,stride=2,padding= (3,3) , bias= False)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)

        self.conv2_1 = nn.Conv2d(64, 64, 3,padding=1)

        self.conv2_2 = nn.Conv2d(64, 64, 3,padding=1)

        self.conv2_3 = nn.Conv2d(64,128,1,stride=2)

        self.conv3_1 = nn.Conv2d(64, 128, 3,padding=1,stride=2)

        self.conv3_2 = nn.Conv2d(128, 128, 3,padding=1)

        self.conv3_3 = nn.Conv2d(128,128,3,padding=1)

        self.conv3_4 = nn.Conv2d(128, 128, 3,padding=1)

        self.conv3_5 = nn.Conv2d(128, 256, 1,stride=2)

        self.conv4_1 = nn.Conv2d(128, 256, 3,stride=2,padding=(1,1))

        self.conv4_2 = nn.Conv2d(256, 256, 3,padding=1)

        self.conv4_3 = nn.Conv2d(256, 256, 3,padding=1)

        self.conv4_4 = nn.Conv2d(256, 256, 3,padding=1)

        self.conv4_5 = nn.Conv2d(256, 512, 1,stride=2)

        self.conv5_1 = nn.Conv2d(256, 512, 3,stride=2,padding=1)

        self.conv5_2 = nn.Conv2d(512, 512, 3,padding=1)

        self.conv5_3 = nn.Conv2d(512, 512, 3)

        self.conv5_4 = nn.Conv2d(512, 512, 3,padding=2)

        self.max_pool2 = nn.AdaptiveAvgPool2d((1,1))

        self.flt = nn.Flatten()

        self.Linear=nn.Linear(512,1)


    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)

        x = F.relu(x)

        x = self.max_pool1(x)

        out = self.conv2_1(x)

        out = F.relu(out)

        out = self.conv2_2(out)

        x = x + out

        x = F.relu(x)
        #conv2_1
        out = self.conv2_1(x)

        out = F.relu(out)

        out = self.conv2_2(out)

        x = x + out

        x = F.relu(x)
        #conv2_2
        out = self.conv2_1(x)

        out = F.relu(out)

        out = self.conv2_2(out)

        x = x + out

        x = F.relu(x)
        #conv2_3
        out = self.conv3_1(x)

        out = F.relu(out)

        out = self.conv3_2(out)

        x = out+self.conv2_3(x)

        x = F.relu(x)
        #conv3_1
        out = self.conv3_3(x)

        out = F.relu(out)

        out = self.conv3_4(out)

        x = x + out

        x = F.relu(x)
        #conv3_2
        out = self.conv3_3(x)

        out = F.relu(out)

        out = self.conv3_4(out)

        x = x + out

        x = F.relu(x)
        #conv3_3
        out = self.conv3_3(x)

        out = F.relu(out)

        out = self.conv3_4(out)

        x = x + out

        x = F.relu(x)
        #conv3_4
        out = self.conv4_1(x)

        out = F.relu(out)

        out = self.conv4_2(out)

        x = out+self.conv3_5(x)

        x = F.relu(x)
        #conv4_1
        out = self.conv4_3(x)

        out = F.relu(out)

        out = self.conv4_4(out)

        x = out + x

        x = F.relu(x)
        #conv4_2
        out = self.conv4_3(x)

        out = F.relu(out)

        out = self.conv4_4(out)

        x = out + x

        x = F.relu(x)
        # conv4_3
        out = self.conv4_3(x)

        out = F.relu(out)

        out = self.conv4_4(out)

        x = out + x

        x = F.relu(x)
        # conv4_4
        out = self.conv4_3(x)

        out = F.relu(out)

        out = self.conv4_4(out)

        x = out + x

        x = F.relu(x)
        # conv4_5
        out = self.conv4_3(x)

        out = F.relu(out)

        out = self.conv4_4(out)

        x = out + x

        x = F.relu(x)
        # conv4_6
        out = self.conv5_1(x)

        out = F.relu(out)

        out = self.conv5_2(out)

        x = out + self.conv4_5(x)

        x = F.relu(x)
        #conv5_1
        out = self.conv5_3(x)

        out = F.relu(out)

        out = self.conv5_4(out)

        x = out + x

        x = F.relu(x)
        #conv5_2

        x = self.max_pool2(x)

        x = self.flt(x)
        x = F.sigmoid(x)
        x = self.Linear(x)



        return x


modellr = 5e-5

# 实例化模型并且移动到GPU

model = ConvNet().to(DEVICE)

# 选择简单暴力的Adam优化器，学习率调低

optimizer = optim.Adam(model.parameters(), lr=modellr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    modellrnew = modellr * (0.1 ** (epoch // 5))

    print("lr:", modellrnew)

    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        output = model(data)

        # print(output)

        loss = F.binary_cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


# 定义测试过程

def val(model, device, test_loader):
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)

            output = model(data)

            # print(output)

            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()  # 将一批的损失相加

            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)

            correct += pred.eq(target.long()).sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

            test_loss, correct, len(test_loader.dataset),

            100. * correct / len(test_loader.dataset)))


# 训练

for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)

    train(model, DEVICE, train_loader, optimizer, epoch)

    val(model, DEVICE, test_loader)

torch.save(model, 'model.pth')