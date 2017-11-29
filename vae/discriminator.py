import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, n_B, n_C, use_gpu=False):
        super(Discriminator, self).__init__()
        self.n_B = n_B
        self.n_C = n_C
        self.use_gpu = use_gpu
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)

        T_ten_init = torch.randn(100, n_B * n_C) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)

        self.fc3 = nn.Linear(100 + n_B, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # #### Minibatch Discrimination ###
        T_tensor = self.T_tensor
        if self.use_gpu:
            T_tensor = T_tensor.cuda()

        Ms = x.mm(T_tensor)
        Ms = Ms.view(-1, self.n_B, self.n_C)

        out_tensor = []
        for i in range(Ms.size()[0]):

            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i

            out_tensor.append(out_i)

        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.n_B)
        x = torch.cat((x, out_T), 1)
        # #### Minibatch Discrimination ###

        x = F.sigmoid(self.fc3(x))
        return x
