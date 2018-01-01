import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, n_B, n_C):
        super(Discriminator, self).__init__()
        self.n_B = n_B
        self.n_C = n_C
        self.num_channels = 20

        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 20)

        T_init = torch.randn(self.num_channels, n_B * n_C) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)
        # self.bn = nn.BatchNorm1d(20 + n_B)
        self.fc3 = nn.Linear(20 + n_B, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # #### Minibatch Discrimination ###
        inp = x.view(-1, self.num_channels)
        M = inp.mm(self.T_tensor)
        M = M.view(-1, self.n_B, self.n_C)

        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)

        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1)
        output = output - output.mean()
        x = torch.cat((inp, output), 1)
        # #### Minibatch Discrimination ###
        # x = self.bn(x)
        x = self.fc3(x)

        x = F.sigmoid(x)
        return x
