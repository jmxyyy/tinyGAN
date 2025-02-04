import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import zipfile
import imageio
import h5py

def data2h5():
    hdf5_file = '../data/celeba_aligned_small.h5py'
    total_images = 20000

    with h5py.File(hdf5_file, 'w') as hf:
        count = 0
        with zipfile.ZipFile('../data/img_align_celeba.zip', 'r') as zf:
            for i in zf.namelist():
                if (i[-4:] == '.jpg'):
                    # extract image
                    ofile = zf.extract(i)
                    img = imageio.imread(ofile)
                    os.remove(ofile)

                    # add image data to HDF5 file with new name
                    hf.create_dataset('img_align_celeba/' + str(count) + '.jpg', data=img, compression="gzip",
                                      compression_opts=9)

                    count = count + 1
                    if count % 1000 == 0:
                        print("images done .. ", count)

                    # stop when total_images reached
                    if count == total_images:
                        break


def crop_centre(img, new_width, new_height):
    height, width, _ = img.shape
    startx = width // 2 - new_width // 2
    starty = height // 2 - new_height // 2
    return img[starty: starty + new_height, startx: startx + new_width, :]


class CelebADataset(Dataset):
    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        img = np.array(self.dataset[str(index) + '.jpg'])
        # crop to 128x128 square
        img = crop_centre(img, 128, 128)
        return torch.cuda.FloatTensor(img).permute(2, 0, 1).view(1, 3, 128, 128) / 255.0

    def plot_image(self, index):
        img = np.array(self.dataset[str(index) + '.jpg'])
        # crop to 128x128 square
        img = crop_centre(img, 128, 128)
        plt.imshow(img, interpolation='nearest')



def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 3, kernel_size=8, stride=2),
            nn.GELU(),
            
            
            View(3 * 10 * 10),
            nn.Linear(3 * 10 * 10, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        # if self.counter % 10000 == 0:
        #     print("counter = ", self.counter)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plt_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(figsize=(16, 8), ylim=(0, 1.0), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 3 * 11 * 11),
            nn.GELU(),
            
            View((1, 3, 11, 11)),
            nn.ConvTranspose2d(3, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 3, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)

        loss = D.loss_func(d_output, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plt_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(figsize=(16, 8), ylim=(0, 1.0), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        plt.show()


if __name__ == '__main__':
    # dataset = torchvision.datasets.CelebA(root='../data', download=True)
    # data2h5()
    celebaData = CelebADataset('../data/celeba_aligned_small.h5py')

    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
        print("using cuda:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D = Discriminator().to(device)
    G = Generator().to(device)
    epochs = 8

    for epoch in range(epochs):
        print("epoch = ", epoch + 1)
        for image_data_tensor in celebaData:
            D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
            D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))
            G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))

        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('epoch:' + str(epoch + 1))
        for i in range(2):
            for j in range(3):
                output = G.forward(generate_random_seed(100))
                img = output.detach().permute(0, 2, 3, 1).view(128, 128, 3).cpu().numpy()
                ax[i, j].imshow(img, interpolation='none', cmap='Blues')
        plt.show()

    D.plt_progress()
    G.plt_progress()

