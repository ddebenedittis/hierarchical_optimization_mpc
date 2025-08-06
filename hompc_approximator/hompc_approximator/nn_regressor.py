#!/usr/bin/env python3

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from torch.utils.data import DataLoader, Dataset

print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = torch.device('cuda')


class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.layers(x)


class ProcessedData(Dataset):
    def __init__(self, x, y):
        x_temp = np.zeros((x.shape[0], x.shape[1] + 1))
        x_temp[:, 0:2] = x[:, 0:2]
        x_temp[:, 2] = np.cos(x[:, 2])
        x_temp[:, 3] = np.sin(x[:, 2])

        self.x = torch.reshape(torch.from_numpy(x_temp).to(torch.float32), x_temp.shape).cuda()
        self.y = torch.reshape(torch.from_numpy(y).to(torch.float32), y.shape).cuda()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index, :]


def main():
    # ============================= Load The Data ============================ #

    pkg_share_dir = get_package_share_directory('hompc_approximator')

    states = np.loadtxt(pkg_share_dir + '/data/' + 'states.csv', delimiter=',')
    inputs_star = np.loadtxt(pkg_share_dir + '/data/' + 'inputs_star.csv', delimiter=',')

    n_input = 4
    n_hidden = 50
    n_output = 2

    # =================== Create The Model And The Dataset =================== #

    model = Model(n_input, n_hidden, n_output).cuda()

    # Instantiate Dataset object for current training data
    d = ProcessedData(states, inputs_star)

    # Instantiate DataLoader
    #    we use the 4 batches of 25 observations each (full data has 100 observations)
    #    we also shuffle the data
    train_dataloader = DataLoader(d, batch_size=1000, shuffle=True)

    # ============================ Train The Model =========================== #

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_train_steps = 10000

    # Training loop
    for epoch in range(n_train_steps):
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_dataloader):
            inputs, targets = data

            optimizer.zero_grad()

            model(inputs)

            outputs = model(torch.reshape(inputs, (train_dataloader.batch_size, n_input))).squeeze()

            loss = loss_function(outputs, targets)

            loss.backward()

            optimizer.step()

            current_loss += loss.item()

        print('Loss after epoch %5d: %.3f' % (epoch + 1, current_loss))
        if (epoch + 1) % 10 == 0:
            current_loss = 0.0

    torch.save(model.state_dict(), pkg_share_dir + 'dio_zanzara.trc')

    print('Training process has finished.')


if __name__ == '__main__':
    main()
