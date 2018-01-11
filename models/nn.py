import torch
from torch import nn
from torch.autograd import Variable


class LeakyReLUBNNSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNNSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.BatchNorm2d(n_out, affine=False)]
        model += [Bias2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUBNNSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNNSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.BatchNorm2d(n_out, affine=False)]
        model += [Bias2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class Bias2d(nn.Module):
    def __init__(self, channels):
        super(Bias2d, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.normal_(0, 0.002)

    def forward(self, x):
        n, c, h, w = x.size()
        return x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n, c, h, w)


class GaussianVAE2D(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(GaussianVAE2D, self).__init__()
        self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.en_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        self.en_mu.weight.data.normal_(0, 0.002)
        self.en_mu.bias.data.normal_(0, 0.002)
        self.en_sigma.weight.data.normal_(0, 0.002)
        self.en_sigma.bias.data.normal_(0, 0.002)

    def forward(self, x):
        mu = self.en_mu(x)
        sd = self.softplus(self.en_sigma(x))
        return mu, sd

    def sample(self, x):
        mu = self.en_mu(x)
        sd = self.softplus(self.en_sigma(x))
        # TODO: add .cuda() to Variable
        noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu.size(3)))
        return mu + sd.mul(noise), mu, sd


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)
