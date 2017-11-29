from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def get_loader(args, train):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    kwargs = {'pin_memory': True} if args.cuda else {}
    split = 'train' if train else 'test'

    transform = transforms.Compose([transforms.Resize(size=(args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    target = datasets.SVHN(root=args.svhn_path, download=True, transform=transform, split=split)
    source = datasets.MNIST(root=args.mnist_path, download=True, transform=transform, train=train)

    target_loader = DataLoader(dataset=target, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               **kwargs)
    source_loader = DataLoader(dataset=source, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               **kwargs)

    return target_loader, source_loader
