import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from hydra.utils import instantiate


class EDM(pl.LightningModule):
    def __init__(
        self,
        denoiser_network_cfg: dict,
        P_mean: float,
        P_std: float,
        sigma_data: float,
        p_uncond: float,
        optim_cfg: dict,
        lr_scheduler_cfg: dict
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.denoiser_network = instantiate(denoiser_network_cfg)

        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.p_uncond = p_uncond

        self.optim_cfg = optim_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg

    def forward(self, x, sigma, class_labels, **model_kwargs):
        # compute scatter shape [B, 1, 1, 1]
        shape = [x.shape[0]] + [1] * (x.ndim - 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in   = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = sigma.log() / 4

        # apply network on scaled input
        F_x = self.denoiser_network(
            (c_in.view(shape) * x),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs
        )

        # final denoised output
        D_x = c_skip.view(shape) * x + c_out.view(shape) * F_x
        return D_x

    def training_step(self, batch, batch_idx):
        x0, y = batch  # unpack data and labels
        batch_size = x0.size(0)
        device = x0.device

        # sample sigma
        sigma_vals = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))
        sigma = sigma_vals.view(batch_size, *([1] * (x0.ndim - 1)))

        noise = torch.randn_like(x0) * sigma
        x_noisy = x0 + noise

        # classifier-free guidance
        mask = torch.rand(batch_size, device=self.device) > self.p_uncond
        y_input = y.clone()
        y_input[~mask] = 0

        D_x = self.forward(x_noisy, sigma_vals, y_input)
        loss = F.mse_loss(D_x, x0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, y = batch  # unpack data and labels
        batch_size = x0.size(0)
        device = x0.device

        # sample sigma
        sigma_vals = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))
        sigma = sigma_vals.view(batch_size, *([1] * (x0.ndim - 1)))

        noise = torch.randn_like(x0) * sigma
        x_noisy = x0 + noise

        # conditional output and loss
        D_x_cond = self.forward(x_noisy, sigma_vals, y)
        loss_cond = F.mse_loss(D_x_cond, x0)
        self.log("val_loss", loss_cond, prog_bar=True)

        # unconditional output and loss
        y_uncond = torch.zeros_like(y)
        D_x_uncond = self.forward(x_noisy, sigma_vals, y_uncond)
        loss_uncond = F.mse_loss(D_x_uncond, x0)
        self.log("unconditional_val_loss", loss_uncond, prog_bar=True)

        return loss_cond

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())

        schedulers_list = []
        for sched_cfg in self.lr_scheduler_cfg.schedulers:
            schedulers_list.append(instantiate(sched_cfg, optimizer=optimizer))

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers_list,
            milestones=self.lr_scheduler_cfg.milestones,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def sampling(
        self,
        latents: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        *,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float('inf'),
        S_noise: float = 1.0,
        randn_like = torch.randn_like,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        EDM sampler (Algorithm 2) with Heun correction.
        """
        device = latents.device

        # sigma-schedule
        i = torch.arange(num_steps, device=device)
        t_steps = (sigma_max**(1/rho)
                   + i / (num_steps - 1) * (sigma_min**(1/rho) - sigma_max**(1/rho))
                  ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # x_0
        x_next = latents * t_steps[0]

        # Main loop
        for j, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Churn
            gamma = (min(S_churn / num_steps, (2**0.5) - 1.0)
                     if (S_min <= t_cur <= S_max) else 0.0)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Heun predictor
            denoised = self.forward(x_hat, t_hat, class_labels, **model_kwargs)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # RK2 correction
            if j < num_steps - 1:
                denoised_next = self.forward(
                    x_next.to(latents.dtype),
                    t_next.to(latents.dtype),
                    class_labels,
                    **model_kwargs
                )
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * 0.5 * (d_cur + d_prime)

        return x_next
