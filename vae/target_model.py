from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 500)
        self.fc31 = nn.Linear(500, 100)
        self.fc32 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 500)
        self.fc5 = nn.Linear(500, 512)
        self.fc6 = nn.Linear(512, 512)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2, 2, return_indices=True)
        self.max_unpool2d = nn.MaxUnpool2d(2, 2)
        self.indices1 = None
        self.indices2 = None
        self.indices3 = None
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = x.view(-1, 3, 28, 28)
        x, self.indices1 = self.max_pool2d(self.relu(self.conv1(x)))
        x, self.indices2 = self.max_pool2d(self.relu(self.conv2(x)))
        x, self.indices3 = self.max_pool2d(self.relu(self.conv3(x)))
        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc4(z))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = x.view(-1, 128, 2, 2)
        x = self.max_unpool2d(x, self.indices3, output_size=torch.Size([-1, 128, 5, 5]))
        x = self.conv4(x)
        x = self.max_unpool2d(x, self.indices2)
        x = self.conv5(x)
        x = self.max_unpool2d(x, self.indices1)
        x = self.conv6(x)
        return self.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def encoder_only(self, x):
        mu, _ = self.encode(x.view(-1, 2352))
        return mu