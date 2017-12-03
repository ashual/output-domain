import torch
from vae.minibatch_std_concat_layer import minibatch_std_concat_layer
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, n_B, n_C, use_gpu=False):
        super(Discriminator, self).__init__()
        self.use_gpu = use_gpu
        self.fc1 = nn.Linear(100, 200)
        # self.minibatch = minibatch_std_concat_layer()
        self.fc2 = nn.Linear(20, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.minibatch(x)
        x = F.sigmoid(self.fc2(x))
        return x
