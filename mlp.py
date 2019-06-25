import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    
    def __init__(self, input_size):
        super(Net, self).__init__()
        hlayer1 = int(input_size*10)
        hlayer2 = int((input_size*10)/2)
        self.fc1 = nn.Linear(input_size, hlayer1)
        self.relu1 = nn.ReLU()
        #self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hlayer1, hlayer2)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(hlayer2, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        #dout = self.dout(h1)
        a2 = self.fc2(h1)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

def train_epoch(model, opt, criterion, X, Y, batch_size=50):
    model.train()
    losses = []
    X = torch.from_numpy(X.values).type(torch.FloatTensor)
    Y = torch.from_numpy(Y.values).type(torch.FloatTensor)
    Y = Y.view(-1,1)
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    return losses

def test(test_x, net):
    preds = []
    for index, sample in test_x.iterrows():
        preds.append(net(torch.from_numpy(sample.values).type(torch.FloatTensor)).data.numpy())
    return preds

def dbc(train_x, train_y, test_x, test_y):
    net = Net(train_x.shape[1])
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    e_losses = []
    num_epochs = 50
    for e in range(num_epochs):
        e_losses += train_epoch(net, opt, criterion, train_x, train_y)
    
    preds = test(test_x, net)
    preds_train = test(train_x, net)
    return preds, preds_train
