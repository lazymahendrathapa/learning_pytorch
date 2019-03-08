import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

os.makedirs('../data/gan_mnist_images', exist_ok=True)

# Parameters

# Number of epochs of training
n_epochs = 200

# Size of the batches
batch_size = 64

# Learning Rate
lr = 0.0002

# Adam: decay of first order momentum of gradient
b1 = 0.5

# Adam: decay of second order momentum of gradient
b2 = 0.999

# Number of cpu threads to use during batch generation
n_cpu = 4

# Dimensionality of the latent space
latent_dim = 100

# Size of each image dimension
img_size = 28

# Number of image channels
channels = 1

# Interval between image samples
sample_interval = 400

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=True),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)

        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
data_path = '../data/mnist'
os.makedirs(data_path, exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=True, download=True,
                   transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                        ])), batch_size=batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr,
                               betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Training

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0),
                         requires_grad=False)

        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1,
                                             (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated
        # samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(
            epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i

        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25],
                       '../data/gan_mnist_images/{}.png'.format(batches_done),
                       nrow=5, normalize=True)
