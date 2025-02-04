import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks

def generate_real():
    real_data = torch.FloatTensor(
        [np.random.uniform(0.8, 1.0),
        np.random.uniform(0.0, 0.2),
        np.random.uniform(0.8, 1.0),
        np.random.uniform(0.0, 0.2)]
    )
    return real_data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

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
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plt_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(figsize=(16, 8), ylim=(0, 1.0), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

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


if __name__ == '__main__':
    D = Discriminator()
    G = Generator()
    train_epoch = 10000
    image_list = []

    for i in range(train_epoch):
        D.train(generate_real(), torch.FloatTensor([1.0]))

        D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))

        G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))

        if i % 1000 == 0:
            if i % 10000 == 0 and i != 0:
                D.plt_progress()
            image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())

    plt.figure(figsize=(16, 8))
    plt.imshow(np.array(image_list).T, interpolation='none', cmap='Blues')
    plt.show()