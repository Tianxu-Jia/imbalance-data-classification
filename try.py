

import torch.nn as nn
import torch
import torch.optim as optim


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

size = 10000
X1 = torch.randn(size, 2)
X2 = torch.randn(size, 2) + 1.5
xx = torch.cat([X1, X2], dim=0)
Y1 = torch.zeros(size, 1)
Y2 = torch.ones(size, 1)
y_train = torch.cat([Y1, Y2], dim=0)

r = torch.randperm(size)
xx = xx[r, :]
y_train = y_train[r, :]
#xx = torch.empty(10000, 2).uniform_(0, 1)
#bernoulli_prob = torch.empty(10000,1).uniform_(0, 1)
#y_train = torch.bernoulli(bernoulli_prob)

class dataset(torch.utils.data.Dataset):
    def __init__(self, xx, yy):
        self.len = len(yy)
        self.xx = xx
        self.yy = yy

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.xx[idx], self.yy[idx]

feature_dataset = dataset(xx, y_train)
fleature_dataloader = torch.utils.data.DataLoader(dataset=feature_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

from torch.autograd import Variable

class fusion_nn(nn.Module):
    def __init__(self):
        super(fusion_nn, self).__init__()
        self.linear1 = nn.Linear(2, 50)
        self.linear2 = nn.Linear(50, 1)
        self.softmax = nn.Softmax()

    def forward(self, xx):
        f = self.softmax(self.linear1(xx))
        out = self.softmax(self.linear2(f))
        return out


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


model = Net()
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print_loss_step = 10
total_loss = 0

for epoch in range(100):
    for i, data in enumerate(fleature_dataloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, labels.to(torch.float32))
        loss.backward()
        optimizer.step()

        if i%print_loss_step == 0:
            print('epoch: ', epoch, '  i: ', i, '  loss: ', total_loss/print_loss_step)
            total_loss = 0
        else:
            total_loss += loss.data




