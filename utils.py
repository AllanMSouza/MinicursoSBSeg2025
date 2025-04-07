import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

# torch.manual_seed(50)

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando o dispositivo:", device)




def cria_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(28), 
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'cifar100':  
        dataset = datasets.CIFAR100(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    if dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(28), 
                                 transforms.CenterCrop(28), 
                                 transforms.ToTensor()
                    ])
    
    if dataset_name == 'celeba':
        dataset = datasets.CelebA(".torch/", download=True)
        tp      = transforms.Compose([transforms.Resize(32), 
                                 transforms.CenterCrop(32), 
                                 transforms.ToTensor()
                    ])
    
    return dataset, tp

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        
def weights_init_mnist(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(0, 1)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(0, 1)

def cria_modelo(dataset_name):
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        modelo = MNISTCNN().to(device)
        
    if dataset_name == 'cifar10':
        modelo = LeNet(num_classes=10).to(device)
        modelo.apply(weights_init)
        
    if dataset_name == 'cifar100':
        modelo = LeNet(num_classes=100).to(device)
        modelo.apply(weights_init)
        
    if dataset_name == 'fashionmnist':
        modelo = MNISTCNN().to(device)
        
    
    return modelo
    
def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def get_imagem_e_label(img_index, dataset, tp, num_classes=10):
    gt_data         = tp(dataset[img_index][0]).to(device)
    gt_data         = gt_data.view(1, *gt_data.size())
    gt_label        = torch.Tensor([dataset[img_index][1]]).long().to(device)
    gt_label        = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)
    
    return gt_data, gt_onehot_label

def cria_ruido(tamanho_imagem, tamanho_label):
    fake_data  = torch.randn(tamanho_imagem).to(device).requires_grad_(True)
    fake_label = torch.randn(tamanho_label).to(device).requires_grad_(True)
    
    return fake_data, fake_label

def mostra_imagem(imagem):
    plt.figure(figsize=(2, 2))
    tt = transforms.ToPILImage()
    plt.imshow(tt(imagem[0].cpu()))
    # plt.title("Imagem Real")
    # print("O label verdadeiro é: %d." % imagem.item(), "\n O Onehot label é: %d." % torch.argmax(label, dim=-1).item())
    
    
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def plot_historico_ataque(historico, label_fake):
    tt = transforms.ToPILImage()
    plt.figure(figsize=(15, 3.5))
    for i in range(14):
        plt.subplot(2, 7, i + 1)
        plt.imshow(historico[i * 10])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    print("Label predito é %d." % torch.argmax(label_fake, dim=-1).item())
    
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out