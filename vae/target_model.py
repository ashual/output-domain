from torch import nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2352, 784)
        self.fc2 = nn.Linear(784, 400)
        self.fc31 = nn.Linear(400, 100)
        self.fc32 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, 400)
        self.fc5 = nn.Linear(400, 784)
        self.fc6 = nn.Linear(784, 2352)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
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
        return self.sigmoid(self.fc6(x))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2352))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def encoder_only(self, x):
        mu, _ = self.encode(x.view(-1, 2352))
        return mu
