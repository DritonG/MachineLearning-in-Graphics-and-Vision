import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torchvision


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

        image = self.incoming_df.data[idx] / 255.0

        noise = torch.rand([28, 28], dtype=torch.float)

        noisy_image = noise + image

        return {'image': image, 'noisy_image': noisy_image}



if __name__ == '__main__':


    fmnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor())
    fmnist_noisy = NoisyFashionMNIST(fmnist_train)

    images = []
    for i in range(80):


        #TODO: extract images
        index = random.randint(100,200)
        dp = fmnist_noisy[index]

        images.append(dp['noisy_image'].expand(1, 28, 28))
        images.append(dp['image'].expand(1, 28, 28).to(dtype=torch.float))
        

    #TODO: visualize (noisy_image, clean_image) in a grid
    images_grid = make_grid(images, nrow=10)

    plt.imshow(np.transpose(images_grid.numpy(), (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.show()

    torchvision.utils.save_image(images_grid, '3-2-a.jpg')
