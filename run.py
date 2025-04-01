import numpy as np
import torch

from SwinUnet import SwinUnet
from scheduler import Scheduler
from PIL import Image

import argparse
import datetime
import os

def main(args):

    device = torch.device(args.device)

    model = SwinUnet(channels=3, dim=96, mlp_ratio=4, patch_size=4, window_size=8,
                     depth=[2, 2, 6, 2], nheads=[3, 6, 12, 24]).to(device)

    sr_ratio = args.sr_ratio

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    scheduler = Scheduler(model, args.denoise_steps)

    image_path = args.image_path
    img = Image.open(image_path)

    img_size = img.size

    assert img_size[0] >= 256 and img_size[1] >= 256, "图片的最小尺寸为256"

    img_size = (
        (img_size[0] // 256) * 256 * sr_ratio,
        (img_size[1] // 256) * 256 * sr_ratio
    )

    img = img.resize(img_size)

    img_arr = np.array(img)

    if img_arr.shape[-1] == 4: img_arr = img_arr[..., :3]

    img_arr = img_arr.transpose(2, 0, 1) / 255.

    img_arr = 2 * (img_arr - 0.5)

    img_arr = torch.from_numpy(img_arr).float().to(device)
    img_arr = img_arr.unsqueeze(0)

    if args.use_ddim:
        y = scheduler.ddim(img_arr, device, sub_sequence_step=args.ddim_sub_sequence_steps)[-1]
    else:
        y = scheduler.ddpm(img_arr, device)[-1]

    y = y.transpose(1, 2, 0)
    y = (y + 1.) / 2
    y *= 255.0

    new_img = Image.fromarray(y.astype(np.uint8))

    new_img.save(os.path.join(args.results_dir, str(datetime.datetime.now()) + ".png"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--sr_ratio", type=int, default=8)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--denoise_steps", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="SwinUNet-SR8.pth")
    parser.add_argument("--use_ddim", type=int, default=1)
    parser.add_argument("--ddim_sub_sequence_steps", type=int, default=25)
    args = parser.parse_args()
    main(args)

