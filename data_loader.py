from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(args, is_train, dataset_name):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if dataset_name == 'mnist':
        dataset = datasets.MNIST('./data/MNIST', train=is_train, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'fashionMnist':
        dataset = datasets.FashionMNIST('./data/FashionMNIST', train=is_train, download=True,
                                        transform=transforms.ToTensor())
    else:
        raise Exception('Dataset does not exist')
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
