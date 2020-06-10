import torch
import torch.nn as nn
import torch.utils as utils
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import pdb

# default `log_dir` is "runs" - we'll be more specific here
saver_path = ('./noskip_log/')
writer = SummaryWriter(saver_path)

epoch = 20
batch_size = 300
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyFashionMNIST(Dataset):

    # TODO: fill

    pass


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        #TODO : model

    def forward(self, x):

        #TODO: fill forward model

        pass

model = Model().to(device=device)

# Data

fmnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None,
                                 download=True)
fmnist_test = dset.FashionMNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None,
                                download=True)
# noise
fmnist_noisy_train = NoisyFashionMNIST(fmnist_train)
fmnist_noisy_test = NoisyFashionMNIST(fmnist_test)


# TODO : Batch and shuffle
# Batch and prepare
train_loader = []
test_loader = []

# Check output of autoencoder

# Set some more stuff
parameters = list(model.parameters())

#TODO: set optimizer
optimizer = []

for i in range(epoch):


    loss = 0.
    loss_test = 0.

    # TRAIN

    for idx, batch_info in enumerate(tqdm(train_loader)):

        model.train()

        #TODO: fill data

        noisy_image = []
        clean_image = []

        image = clean_image.to(device=device)
        image_n = noisy_image.to(device=device)

        output = model(image_n)

        loss = torch.abs(output - image).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss',
                          loss.item(),
                          i * len(train_loader) + idx)

        images = torch.cat([image_n.cpu().repeat(1, 3, 1, 1), output.cpu().repeat(1, 3, 1, 1), image.cpu().repeat(1, 3, 1, 1)], axis=3)
        img_grid = torchvision.utils.make_grid(images.cpu())
        # pdb.set_trace()
        writer.add_image('Input VS Output VS Ground Truth TRAINING',
                         img_grid,
                         global_step=i * len(train_loader) + idx)

    # TEST

    for idx_test, batch_info_test in enumerate(tqdm(test_loader)):

        optimizer.zero_grad()

        with torch.no_grad():

            # TODO: fill test data

            noisy_image_t = []
            clean_image_t = []

            image_t = clean_image_t.to(device=device)
            image_n_t = noisy_image_t.to(device=device)

            output_t = model(image_n_t)

            loss_test = torch.abs(output_t - image_t).mean()

            writer.add_scalar('Validation loss',
                              loss_test.item(),
                              i * len(test_loader) + idx_test)

            images_test = torch.cat([image_n_t.cpu().repeat(1, 3, 1, 1), output_t.cpu().repeat(1, 3, 1, 1),image_t.cpu().repeat(1, 3, 1, 1)], axis=3)
            img_grid_t = torchvision.utils.make_grid(images_test.cpu())
            # pdb.set_trace()
            writer.add_image('Input VS Output VS Ground Truth VALIDATION',
                             img_grid_t,
                             global_step=i * len(test_loader) + idx_test)

    print('epoch done: ', i)
    # torch.save([model], PATH)
    print('Train Loss :', loss.item())
    print('Validation loss : ', loss_test.item())
