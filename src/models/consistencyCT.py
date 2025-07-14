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
        sigma_data: float,
        train_num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        optim_cfg: dict,
        lr_scheduler_cfg: dict
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # denoiser network
        self.denoiser_network = instantiate(denoiser_network_cfg)

        # data noise level (for c_skip / c_out)
        self.sigma_data = sigma_data

        # CT-specific schedule
        self.train_num_steps = train_num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        # Pre-compute & register the t_steps
        # t_i = (sigma_max^(1/ρ) + i/(N-1)*(sigma_min^(1/ρ)-sigma_max^(1/rho)))^rho
        i = torch.arange(self.train_num_steps, dtype=torch.float32)
        t_i = (sigma_max**(1/rho)
               + i/(self.train_num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
        t_steps = torch.cat([t_i, torch.zeros(1)], dim=0)
        self.register_buffer("t_steps", t_steps)

        # optimizer & scheduler configs
        self.optim_cfg = optim_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg

    def forward(self, x, sigma, class_labels, **model_kwargs):
        """
        The consistency mapping f_θ(x,sigma) using the same skip/out/in
        parameterization as EDM.
        """
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = sigma.log() / 4

        F_theta = self.denoiser_network(
            (c_in.view(shape) * x),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs
        )
        D_x = c_skip.view(shape) * x + c_out.view(shape) * F_theta
        return D_x

    def training_step(self, batch, batch_idx):
        """
        Algorithm 3 (CT) step:
          1. Sample i ~ Uniform{0,...,N-1}.
          2. Sample z ~ N(0,I); form x_{t_{i+1}}, x_{t_i} from same z.
          3. Compute student f_θ(x_{t_{i+1}}, t_{i+1}) (grad-on).
          4. Compute "teacher" f_θ(x_{t_i}, t_i) under no_grad (stopgrad θ).
          5. L = ||student - teacher||^2.
        """
        x0, y = batch
        B, device = x0.size(0), x0.device

        idx = torch.randint(0, self.train_num_steps, (B,), device=device)
        t_i   = self.t_steps[idx].view(B, *([1] * (x0.ndim - 1)))
        t_ip1 = self.t_steps[idx + 1].view(B, *([1] * (x0.ndim - 1)))
        z = torch.randn_like(x0)
        x_ti = x0 + z * t_i
        x_tip1 = x0 + z * t_ip1

        with torch.no_grad():
            D_ti = self.forward(x_ti, t_i.flatten(), class_labels=y)
        D_tip1 = self.forward(x_tip1, t_ip1.flatten(), class_labels=y)

        # consistency loss
        loss = F.mse_loss(D_ti, D_tip1)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, y = batch
        B, device = x0.size(0), x0.device

        idx = torch.randint(0, self.train_num_steps, (B,), device=device)
        t_i   = self.t_steps[idx].view(B, *([1] * (x0.ndim - 1)))
        t_ip1 = self.t_steps[idx + 1].view(B, *([1] * (x0.ndim - 1)))
        z = torch.randn_like(x0)
        x_ti   = x0 + z * t_i
        x_tip1 = x0 + z * t_ip1

        with torch.no_grad():
            D_ti = self.forward(x_ti, t_i.flatten(), class_labels=y)
        D_tip1 = self.forward(x_tip1, t_ip1.flatten(), class_labels=y)

        loss = F.mse_loss(D_ti, D_tip1)
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
        randn_like=torch.randn_like,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Unchanged from before: multistep consistency sampler 
        (falls back to one-step if num_steps=1).
        """
        device = latents.device
        i = torch.arange(num_steps, device=device, dtype=latents.dtype)
        t_steps = (
            sigma_max ** (1 / rho)
            + i / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros(1, device=device)], dim=0)

        x = self.forward(latents * t_steps[0], t_steps[0].view(1), class_labels=class_labels, **model_kwargs)
        for t in t_steps[1:-1]:
            z = randn_like(x)
            sigma_noise = torch.sqrt(torch.clamp(t**2 - sigma_min**2, min=0.0))
            x_b = x + sigma_noise * z
            x = self.forward(x_b, t.view(1), class_labels=class_labels, **model_kwargs)
        return x
