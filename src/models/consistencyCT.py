import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from hydra.utils import instantiate


class ConsistencyCT(pl.LightningModule):
    """
    Consistency Training (CT) implementation
    for class-conditional data.

    References:
    Song et al., 2023 – Consistency Models
    Song & Dhariwal, 2023 – Improved Techniques for Training CMs
    """
    def __init__(
        self,
        denoiser_network_cfg: dict,
        P_mean: float,
        P_std: float,
        sigma_data: float,
        optim_cfg: dict,
        lr_scheduler_cfg: dict
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # denoiser network
        self.denoiser_network = instantiate(denoiser_network_cfg)

        # log-normal schedule params
        self.P_mean = P_mean
        self.P_std = P_std

        # data noise level (for c_skip / c_out)
        self.sigma_data = sigma_data

        # optimizer & scheduler configs
        self.optim_cfg = optim_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg

    def forward(self, x, sigma, class_labels, **model_kwargs):
        """
        The consistency mapping maps a noisy x at noise level sigma back toward clean data.
        Implements the same skip/out/in parameterization as EDM:
            D_x = c_skip * x + c_out * F_theta(c_in * x, emb(sigma), class_labels)
        """
        # broadcasting shapes
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in   = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = sigma.log() / 4

        # apply network, passing class_labels
        F_theta = self.denoiser_network(
            (c_in.view(shape) * x),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs
        )

        # final consistency output
        D_x = c_skip.view(shape) * x + c_out.view(shape) * F_theta
        return D_x

    def training_step(self, batch, batch_idx):
        """
        Consistency training (no teacher): sample two noise levels sigma1, sigma2 ~ LogNormal(P_mean,P_std),
        share one eps noise, form x1,x2, and minimize || f(x1,sigma1, y) - f(x2,sigma2, y) ||^2.
        """
        x0, y = batch
        batch_size = x0.size(0)
        device = x0.device

        # sample two noise levels
        sigma1 = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))
        sigma2 = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))

        # shared noise eps
        eps = torch.randn_like(x0)

        # noisy inputs at two levels
        x1 = x0 + eps * sigma1.view(batch_size, *([1] * (x0.ndim - 1)))
        x2 = x0 + eps * sigma2.view(batch_size, *([1] * (x0.ndim - 1)))

        # model outputs (pass class labels)
        D1 = self.forward(x1, sigma1, class_labels=y)
        D2 = self.forward(x2, sigma2, class_labels=y)

        # consistency loss
        loss = F.mse_loss(D1, D2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Same as training but logs a validation consistency loss.
        """
        x0, y = batch
        batch_size = x0.size(0)
        device = x0.device

        sigma1 = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))
        sigma2 = torch.exp(self.P_mean + self.P_std * torch.randn(batch_size, device=device))
        eps = torch.randn_like(x0)

        x1 = x0 + eps * sigma1.view(batch_size, *([1] * (x0.ndim - 1)))
        x2 = x0 + eps * sigma2.view(batch_size, *([1] * (x0.ndim - 1)))

        D1 = self.forward(x1, sigma1, class_labels=y)
        D2 = self.forward(x2, sigma2, class_labels=y)

        loss = F.mse_loss(D1, D2)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optim_cfg, params=self.parameters())

        schedulers = [
            instantiate(s_cfg, optimizer=optimizer)
            for s_cfg in self.lr_scheduler_cfg.schedulers
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
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
        class_labels: torch.Tensor,
        *,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        randn_like = torch.randn_like,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Multistep consistency sampler (Algorithm 2 from Song et al. 2023).
        Falls back to one-step when num_steps=1.
        """
        device = latents.device

        # build noise schedule: t_0...t_{N-1}, plus final eps=0
        i = torch.arange(num_steps, device=device, dtype=latents.dtype)
        t_steps = (sigma_max**(1/rho)
                   + i/(num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])], dim=0)

        # initial sample: apply mapping once
        x = self.forward(latents * t_steps[0], t_steps[0], class_labels=class_labels, **model_kwargs)

        # for each intermediate noise level
        for t in t_steps[1:-1]:
            z = randn_like(x)
            sigma_noise = torch.sqrt(torch.clamp(t**2 - sigma_min**2, min=0.0))
            x_b = x + sigma_noise * z

            # apply consistency mapping with labels
            x = self.forward(x_b, t, class_labels=class_labels, **model_kwargs)

        return x
