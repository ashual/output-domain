import torch
from torch.nn import functional as F


def generator_loss_function(recon_x, x, mu, logvar, args):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 784

    return BCE + KLD


def generator_loss_function2(recon_x, x, args):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    return BCE


def discriminator_loss_function(x):
    return -torch.mean(x)
