import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def crop_centre(img, new_width, new_height):
    height, width, _ = img.shape
    startx = width // 2 - new_width // 2
    starty = height // 2 - new_height // 2
    return img[starty: starty + new_height, startx: startx + new_width, :]


class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0

        image_values = torch.cuda.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0
        return label, image_values, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()


def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0,size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor


def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784 + 10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, image_tensor, label_tensor):
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)

    def train(self, inputs, label_tensor, targets):
        outputs = self.forward(inputs, label_tensor)
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
            nn.Linear(100 + 10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)

    def train(self, D, inputs, label_tensor, targets):
        g_output = self.forward(inputs, label_tensor)
        d_output = D.forward(g_output, label_tensor)

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

    def plt_image(self, label):
        label_tensor = torch.zeros(10)
        label_tensor[label] = 1.0
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('number:' + str(label + 1))
        for i in range(2):
            for j in range(3):
                ax[i, j].imshow(
                    G.forward(generate_random_seed(100), label_tensor).detach().cpu().numpy().reshape(28, 28),
                    interpolation='none', cmap='Blues')
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(device=device)
        print("using cuda:", torch.cuda.get_device_name(0))

    D = Discriminator().to(device=device)
    G = Generator().to(device=device)

    mnist_dataset = MnistDataset('../data/mnist_train.csv')

    epoch = 12

    for i in range(epoch):
        print("epoch = ", i + 1)
        for label, image_data_tensor, label_tensor in mnist_dataset:
            D.train(image_data_tensor, label_tensor, torch.cuda.FloatTensor([1.0]))
            random_label = generate_random_one_hot(10)
            D.train(G.forward(generate_random_seed(100), random_label).detach(),
                    random_label, torch.cuda.FloatTensor([0.0]))
            random_label = generate_random_one_hot(10)
            G.train(D, generate_random_seed(100), random_label, torch.cuda.FloatTensor([1.0]))

    D.plt_progress()
    G.plt_progress()

    for label in range(10):
        G.plt_image(label)

