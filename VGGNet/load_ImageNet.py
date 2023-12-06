import os
import torchvision.transforms as transforms
# from torchvision.datasets import DatasetFolder
from torchvision import datasets
import torch
def load_ImageNet(ImageNet_PATH, batch_size=64,size = 224): 
    
    traindir = os.path.join(ImageNet_PATH, 'mini_train')
    valdir   = os.path.join(ImageNet_PATH, 'mini_val')
    # print('traindir = ',traindir)
    # print('valdir = ',valdir)
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalizer
        ])
    )
    # print('train_dataset = ',len(train_dataset))
    # print('val_dataset   = ',len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, train_dataset, val_dataset

