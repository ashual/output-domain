from __future__ import print_function
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from mnist_classifier.model import Net

GPU_FILE = '../mnist_classifier/model_gpu_1.pt'
CPU_FILE = '../mnist_classifier/model_cpu_1.pt'


class ClassifyMNIST:
    def __init__(self, args):
        self.args = args
        self.args.lr = 0.01
        self.args.momentum = 0.5
        if args.cuda:
            if os.path.isfile(GPU_FILE):
                self.model = torch.load(GPU_FILE)
            else:
                self.model = Net().cuda()
                self.train()
        else:
            if os.path.isfile(CPU_FILE):
                self.model = torch.load(CPU_FILE)
            else:
                self.model = Net()
                self.train()

    def train(self):
        args = self.args
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                                                transforms.Normalize(
                                                                                                    (0.1307,),
                                                                                                    (0.3081,))])),
                                                   batch_size=args.batch_size, shuffle=True, **kwargs)
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        for epoch in range(1, 20 + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))
        if args.cuda:
            torch.save(self.model, GPU_FILE)
        else:
            torch.save(self.model, CPU_FILE)

    def test(self, data):
        data = data.view(-1, 1, 28, 28)
        output = self.model(data)
        results = output.data.max(1, keepdim=True)[1]
        return results
