import numpy as np

import torch

import torch.nn.functional as F

from tqdm import tqdm

def extract_into_tensor(arr, timesteps, broadcast_shape):

    res = torch.from_numpy(arr).to(torch.float32).to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

class Scheduler:

    def __init__(self, denoise_model, denoise_steps, beta_start=1e-4, beta_end=0.005):

        self.model = denoise_model

        betas = np.array(
            np.linspace(beta_start, beta_end, denoise_steps),
            dtype=np.float64
        )

        self.denoise_steps = denoise_steps

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas

        self.sqrt_alphas = np.sqrt(alphas)
        self.one_minus_alphas = 1.0 - alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    def q_sample(self, y0, t, noise):

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, y0.shape) * y0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y0.shape) * noise
        )

    def training_losses(self, x, y, t):

        noise = torch.randn_like(y)
        y_t = self.q_sample(y, t, noise)

        predict_noise = self.model(torch.cat([x, y_t], dim=1), t)

        return F.mse_loss(predict_noise, noise)


    @torch.no_grad()
    def ddpm(self, x, device):

        y = torch.randn(*x.shape, device=device)

        for t in tqdm(reversed(range(0, self.denoise_steps)), total=self.denoise_steps):

            t = torch.tensor([t], device=device).repeat(x.shape[0])
            t_mask = (t != 0).float().view(-1, *([1] * (len(y.shape) - 1)))

            eps = self.model(torch.cat([x, y], dim=1), t)

            y = y - (
                    extract_into_tensor(self.one_minus_alphas, t, y.shape) * eps
                    / extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y.shape)
            )

            y = y / extract_into_tensor(self.sqrt_alphas, t, y.shape)

            sigma = torch.sqrt(
                extract_into_tensor(self.one_minus_alphas, t, y.shape)
                * (1.0 - extract_into_tensor(self.alphas_cumprod_prev, t, y.shape))
                / (1.0 - extract_into_tensor(self.alphas_cumprod, t, y.shape))
            )

            y = y + sigma * torch.randn_like(y) * t_mask

            y = y.clip(-1, 1)

        return y.detach().cpu().numpy()


    @torch.no_grad()
    def ddim(self, x, device, eta=0.0, sub_sequence_step=25):
        # 初始化 y 为高斯噪声
        y = torch.randn(*x.shape, device=device)
        # 构造跳步采样的时间序列，从 denoise_steps-1 到 0，每隔 jump 取一个时间步
        t_seq = list(range(self.denoise_steps - 1, -1, -sub_sequence_step))
        for i in tqdm(range(len(t_seq)), total=len(t_seq)):
            # 当前时间步 t 和下一个采样时间步 s（若为最后一步，则 s 设为 0）
            t = t_seq[i]
            s = 0 if i == len(t_seq) - 1 else t_seq[i + 1]
            # 构造与 batch 数量相同的时间步张量
            t_tensor = torch.tensor([t], device=device).repeat(x.shape[0])
            s_tensor = torch.tensor([s], device=device).repeat(x.shape[0])

            # 用模型预测噪声
            eps = self.model(torch.cat([x, y], dim=1), t_tensor)
            # 提取当前和下一个时间步对应的 α 累积乘积
            alpha_bar_t = extract_into_tensor(self.alphas_cumprod, t_tensor, y.shape)
            alpha_bar_s = extract_into_tensor(self.alphas_cumprod, s_tensor, y.shape)

            # 根据 DDIM 公式预测原始样本 x0 的估计
            y0_pred = (y - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # 计算控制随机性的 sigma
            sigma = 0.0
            if eta > 0.0 and s > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_s) / (1 - alpha_bar_t) *
                    (1 - alpha_bar_t / alpha_bar_s)
                )
            # 利用预测的 x0 和当前噪声方向更新至下一个时间步的样本
            y = torch.sqrt(alpha_bar_s) * y0_pred + torch.sqrt(1 - alpha_bar_s - sigma ** 2) * eps
            # 若 eta > 0 则在更新后加入噪声（最后一步不添加）
            if eta > 0.0 and s > 0:
                y = y + sigma * torch.randn_like(y)

            y = y.clip(-1, 1)

        return y.detach().cpu().numpy()
