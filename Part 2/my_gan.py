import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    filelist = [f for f in os.listdir('images') if f.endswith('.png')]
    d_losses = []
    g_losses = []
    for f in filelist:
        os.remove(os.path.join('images', f))
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)

            optimizer_G.zero_grad()
            z = torch.randn(imgs.shape[0], args.latent_dim).to(device)
            gen_imgs = generator(z)
            g_loss = -torch.mean(torch.log(discriminator(gen_imgs)))
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = -torch.mean(torch.log(discriminator(imgs)))
            fake_loss = -torch.mean(torch.log(1 - discriminator(gen_imgs.detach())))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.n_epochs}], Step [{i + 1}/{len(dataloader)}], "
                      f"Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

            if batches_done == 0:
                sample_images(generator, epoch, "Start")
            elif batches_done == args.n_epochs // 2 * len(dataloader):
                sample_images(generator, epoch, "Halfway")
            elif batches_done == args.n_epochs * len(dataloader) - 1:
                sample_images(generator, epoch, "End")
    if os.path.exists('loss_curves.png'):
        os.remove('loss_curves.png')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')


def sample_images(generator, epoch, stage):
    z = torch.randn(25, args.latent_dim).to(device)
    gen_imgs = generator(z)
    save_image(gen_imgs,
               'sample_images/{}_{}.png'.format(stage, epoch),
               nrow=5, normalize=True)



def interpolate(generator):
    z1 = torch.randn(1, args.latent_dim).to(device)
    z2 = torch.randn(1, args.latent_dim).to(device)
    z_interp = torch.zeros(1, args.latent_dim).to(device)

    for alpha in torch.linspace(0, 1, 9):
        z = alpha * z1 + (1 - alpha) * z2
        z_interp = torch.cat((z_interp, z), dim=0)

    z_interp = z_interp[1:]  # Remove the initial zeros

    gen_imgs = generator(z_interp)
    save_image(gen_imgs,
               'interpolated_images/interpolation.png',
               nrow=9, normalize=True)

def main():
    os.makedirs('images', exist_ok=True)
    os.makedirs('sample_images', exist_ok=True)
    os.makedirs('interpolated_images', exist_ok=True)



    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    torch.save(generator.state_dict(), "mnist_generator.pt")

    interpolate(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500, help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
