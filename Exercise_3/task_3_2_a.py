import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pdb

class NoisyFashionMNIST(Dataset):

    def __init__(self, incoming_df):

        super(NoisyFashionMNIST, self).__init__()
        self.incoming_df = incoming_df


    def __len__(self):
        return len(self.incoming_df)

    def __getitem__(self, idx):

        # TODO: add noise

        #image =

        #noise =

        #noisy_image = noise + image

        # return a pair of data [original_image, noisy_image]
        # see Tutorial for hints


        pass



if __name__ == '__main__':


    fmnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor())
    fmnist_noisy = NoisyFashionMNIST(fmnist_train)



    for i in range(80):


        #TODO: extract images
        index = random.randint(100,200)
        dp = fmnist_noisy[index]


    #TODO: visualize (noisy_image, clean_image) in a grid