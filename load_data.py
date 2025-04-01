import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def is_image_file(file_path):
    # 定义常见的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png'}
    # 获取文件的扩展名并判断是否在图片扩展名集合中
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions


class CustomDataset(Dataset):

    def __init__(self, path, img_size=None, sr_ratio=8):
        super().__init__()

        files = os.listdir(path)

        self.img_size = img_size

        self.files = []

        for file in files:
            self.files.append(os.path.join(path, file))

        self.ratio = sr_ratio

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        hr_img = Image.open(self.files[idx])

        if self.img_size is not None:
            hr_img = hr_img.resize(self.img_size)
            hr_size = hr_img.size

        else:
            hr_size = hr_img.size

            hr_size = ((hr_size[0] // 256 + 1) * 256, (hr_size[1] // 256 + 1) * 256)
            hr_img = hr_img.resize(hr_size)

        hr_arr = np.array(hr_img).transpose(2, 0, 1) / 255.

        lr_img = hr_img.resize((hr_size[0] // self.ratio, hr_size[1] // self.ratio))
        lr_img = lr_img.resize(hr_size)
        lr_arr = np.array(lr_img).transpose(2, 0, 1) / 255.

        lr_arr = 2 * (lr_arr - 0.5)
        hr_arr = 2 * (hr_arr - 0.5)

        return lr_arr, hr_arr


class ImageNetDataset(Dataset):

    def __init__(self, path, img_size=(256, 256), sr_ratio=8):
        super().__init__()

        self.img_size = img_size

        class_dirs = os.listdir(path)

        self.files = []

        for class_dir in class_dirs:

            files = os.listdir(os.path.join(path, class_dir))

            for file in files:

                if is_image_file(os.path.join(path, class_dir, file)):
                    self.files.append(os.path.join(path, class_dir, file))

        self.ratio = sr_ratio

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        hr_img = Image.open(self.files[idx])
        hr_img = hr_img.convert("RGB")

        if self.img_size is not None:
            hr_img = hr_img.resize(self.img_size)

        hr_size = hr_img.size
        hr_arr = np.array(hr_img).transpose(2, 0, 1) / 255.

        lr_img = hr_img.resize((hr_size[0] // self.ratio, hr_size[1] // self.ratio))
        lr_img = lr_img.resize(hr_size)
        lr_arr = np.array(lr_img).transpose(2, 0, 1) / 255.

        lr_arr = 2 * (lr_arr - 0.5)
        hr_arr = 2 * (hr_arr - 0.5)

        return lr_arr, hr_arr

