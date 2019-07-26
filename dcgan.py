#! /usr/bin/env python

"""
Run a DCGAN on MNIST.

Usage:
  ./viz-blur [--device=<device>] [--epochs=<epochs>]
  ./viz-blur (-h | --help)

Options
  --device=<device>  PyTorch device to use [default: cuda:0].
  --epochs=<epochs>  Number of epochs to run for [default: 10].
  -h --help          Show this screen.
"""

import docopt
import torch
import torchvision
import torchvision.transforms as T


class G(torch.nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.input = torch.nn.Linear(100, 2048)
        self.deconv = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1,
                                     output_padding=1),  # 128 x 8 x 8
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                                     output_padding=1),  # 64 x 16 x 16
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1,
                                     output_padding=1),  # 3 x 32 x 32
            torch.nn.Tanh())

    def forward(self, noise):
        input_out = self.input(noise)
        seq_in = input_out.reshape(input_out.shape[0], 128, 4, 4)
        return self.deconv(seq_in)


class D(torch.nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=2, padding=1),  # 64 x 16 x 16
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 x 8 x 8
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 128 x 4 x 4
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm2d(128))
        self.output = torch.nn.Sequential(torch.nn.Linear(2048, 1),
                                          torch.nn.Sigmoid())

    def forward(self, image):
        seq_out = self.conv(image)
        output_in = seq_out.reshape(seq_out.shape[0], 2048)
        return self.output(output_in)


def viz(arr):
    """
    Visualize an MNIST image in the terminal.

    :param arr: a tensor of shape (1, 32, 32) with values between 0 and 1
    """
    arr = arr.reshape(32, 32)
    for i, _ in enumerate(arr):
        out = ''
        for j, _ in enumerate(arr[i]):
            out += _to_unicode(arr[i][j].item())
        print(out)


def _to_unicode(value):
    if -1. <= value < -0.6:
        return u' ' * 2
    if -0.6 <= value < -0.2:
        return u'\u2598' * 2
    if -0.2 <= value < 0.2:
        return u'\u259A' * 2
    if 0.2 <= value < 0.6:
        return u'\u259B' * 2
    if 0.5 <= value <= 1.:
        return u'\u2588' * 2
    raise ValueError('Invalid value {}'.format(value))


def main(args):
    device = torch.device(args['--device'])
    generator = G()
    discriminator = D()
    loss = torch.nn.BCELoss()
    generator.to(device)
    discriminator.to(device)
    loss.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4,
                                   betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4,
                                   betas=(0.5, 0.999))
    transform = T.Compose([T.Resize(32),
                           T.ToTensor(),
                           T.Normalize([0.5], [0.5])])
    dataset = torchvision.datasets.MNIST('.', download=True,
                                         transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True)
    for epoch in range(int(args['--epochs'])):
        for i, (image, labels) in enumerate(dataloader):
            image = image.to(device)
            labels = labels.to(device)
            batch_size = len(image)
            noise = torch.randn(batch_size, 100).to(device)
            g_sample = generator(noise)
            ones = torch.ones(batch_size, 1).to(device)
            zeros = torch.zeros(batch_size, 1).to(device)
            d_optimizer.zero_grad()
            d_loss_real = loss(discriminator(image), ones)
            d_loss_fake = loss(discriminator(g_sample), zeros)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
            g_optimizer.zero_grad()
            g_loss = loss(discriminator(g_sample), ones)
            g_loss.backward()
            g_optimizer.step()
            if not i % 20:
                viz(g_sample[0])


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args)
