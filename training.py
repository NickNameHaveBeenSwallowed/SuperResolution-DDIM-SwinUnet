import torch
import numpy as np

from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from load_data import CustomDataset
from scheduler import Scheduler
from SwinUnet import SwinUnet

if __name__ == '__main__':

    device = torch.device("mps")
    batch_size = 16
    lr = 1e-4
    epochs = 200
    denoise_steps = 1000
    sr_ratio = 8

    train_dataset = CustomDataset(
        "./DIV2K_train_HR", img_size=(512, 512), sr_ratio=sr_ratio,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SwinUnet(channels=3, dim=96, mlp_ratio=4, patch_size=4, window_size=8,
                    depth=[2, 2, 6, 2], nheads=[3, 6, 12, 24]).to(device)

    model.load_state_dict(torch.load("SwinUNet-SR8.pth", map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = Scheduler(model, denoise_steps)

    model.train()
    for epoch in range(epochs):

        print('*' * 40)

        train_loss = []

        for i, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):

            x, y = data
            x = Variable(x).to(torch.float32).to(device)
            y = Variable(y).to(torch.float32).to(device)

            t = torch.randint(low=0, high=denoise_steps, size=(x.shape[0],)).to(device)
            training_loss = scheduler.training_losses(x, y, t)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            train_loss.append(training_loss.item())

        torch.save(model.state_dict(), f"unet-sr{sr_ratio}.pth")
        print('Finish  {}  Loss: {:.6f}'.format(epoch + 1, np.mean(train_loss)))

