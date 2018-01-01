from torchvision import datasets, transforms
import torch

mnist = datasets.MNIST('../data_MNIST', train=True, download=True, transform=transforms.ToTensor())
data = mnist.train_data.view(-1, 784).float()
train_labels = mnist.train_labels


def nn(s):
    _, d = torch.topk(-torch.sum((data - s)**2, dim=1), 2, dim=0)
    return d[1]


counter = 0.
for i, sample in enumerate(data):
    nn_index = nn(sample)
    if i == nn_index:
        raise Exception('same')
    if train_labels[nn_index] == train_labels[i]:
        counter += 1
    if i % 100 == 0:
        print('\r {}\{} {}'.format(i, len(data), counter/(i+1)))
print(counter/len(data))
