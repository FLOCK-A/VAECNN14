# augmentations.py
import torch
import torch.nn.functional as F


def _to_btf(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Normalize input to (B, T, F). Return (x_btf, meta) for restoring.
    Supported: (T,F), (B,T,F), (B,1,T,F)
    """
    meta = {"orig_dim": x.dim(), "had_channel": False}
    if x.dim() == 2:  # (T,F)
        x = x.unsqueeze(0)  # (1,T,F)
    elif x.dim() == 4:      # (B,1,T,F)
        meta["had_channel"] = True
        x = x.squeeze(1)    # (B,T,F)
    elif x.dim() != 3:
        raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")
    return x, meta


def _restore_from_btf(x_btf: torch.Tensor, meta: dict) -> torch.Tensor:
    if meta["orig_dim"] == 2:
        return x_btf.squeeze(0)
    if meta["orig_dim"] == 4 and meta["had_channel"]:
        return x_btf.unsqueeze(1)
    return x_btf


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


class ASCStrongAugment:
    """
    Spectrogram-level augmentations for ASC (no pipeline changes).
    Apply on (B,T,F) or (T,F) log-mel features.

    Includes:
      - time rolling
      - SpecAugment (time/freq masking)
      - FilterAugment (linear)
      - Freq-MixStyle (freq-wise stats mixing)
      - Mixup (source-only, returns soft labels)
    """
    def __init__(
        self,
        num_classes: int = 10,
        enable: bool = True,

        # time rolling
        p_time_roll: float = 0.5,

        # SpecAugment
        p_specaug: float = 0.8,
        freq_mask_ratio: float = 0.15,
        time_mask_ratio: float = 0.15,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0,

        # FilterAugment (linear)
        p_filter: float = 0.5,
        filter_n_band: tuple[int, int] = (4, 8),
        filter_db: tuple[float, float] = (-6.0, 6.0),
        filter_mode: str = "add",  # "add" (log domain) or "mul"

        # Freq-MixStyle
        p_fms: float = 0.5,
        fms_alpha: float = 0.6,
        fms_mix: str = "crossdomain",  # "random" or "crossdomain"

        # Mixup (source-only)
        p_mixup: float = 1.0,
        mixup_alpha: float = 0.4,
    ):
        self.num_classes = num_classes
        self.enable = enable

        self.p_time_roll = p_time_roll

        self.p_specaug = p_specaug
        self.freq_mask_ratio = freq_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value

        self.p_filter = p_filter
        self.filter_n_band = filter_n_band
        self.filter_db = filter_db
        self.filter_mode = filter_mode

        self.p_fms = p_fms
        self.fms_alpha = fms_alpha
        self.fms_mix = fms_mix

        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor | None, num_src_samples: int | None):
        """
        Returns:
          x_aug: same shape as input x
          y_soft_full: (B,num_classes) if mixup applied else None
        """
        if not self.enable:
            return x, None

        x_btf, meta = _to_btf(x)
        B, T, Freq = x_btf.shape
        device = x_btf.device
        # Apply SpecAugment/FilterAugment on source-only when num_src_samples is provided (adaptation).
        n_apply = B if (num_src_samples is None) else int(num_src_samples)
        n_apply = max(0, min(B, n_apply))
        y_soft_full = None

        # ---- time rolling (label-preserving) ----
        if self.p_time_roll > 0:
            shifts = torch.randint(low=0, high=T, size=(n_apply,), device=device)
            # per-sample roll (loop is fine; B is not huge in ASC)
            for i in range(n_apply):
                if torch.rand((), device=device) >= self.p_time_roll:
                    continue
                x_btf[i] = torch.roll(x_btf[i], shifts=int(shifts[i].item()), dims=0)

        # ---- SpecAugment (masking) ----
        if self.p_specaug > 0:
            max_f = max(1, int(self.freq_mask_ratio * Freq))
            max_t = max(1, int(self.time_mask_ratio * T))

            for i in range(n_apply):
                if torch.rand((), device=device) >= self.p_specaug:
                    continue
                # freq masks
                for _ in range(self.num_freq_masks):
                    fw = int(torch.randint(0, max_f + 1, (1,), device=device).item())
                    if fw <= 0 or fw >= Freq:
                        continue
                    f0 = int(torch.randint(0, Freq - fw + 1, (1,), device=device).item())
                    x_btf[i, :, f0:f0 + fw] = self.mask_value

                # time masks
                for _ in range(self.num_time_masks):
                    tw = int(torch.randint(0, max_t + 1, (1,), device=device).item())
                    if tw <= 0 or tw >= T:
                        continue
                    t0 = int(torch.randint(0, T - tw + 1, (1,), device=device).item())
                    x_btf[i, t0:t0 + tw, :] = self.mask_value

        # ---- FilterAugment (linear) ----
        if self.p_filter > 0:
            nmin, nmax = self.filter_n_band
            db_min, db_max = self.filter_db

            for i in range(n_apply):
                if torch.rand((), device=device) >= self.p_filter:
                    continue
                n_bound = int(torch.randint(nmin, nmax + 1, (1,), device=device).item())
                # boundaries on [0, Freq-1]
                # choose (n_bound-1) internal points, plus endpoints
                if n_bound <= 1:
                    idx = torch.tensor([0, Freq - 1], device=device, dtype=torch.long)
                else:
                    internal = torch.randint(1, Freq - 1, (n_bound - 1,), device=device)
                    idx = torch.cat([torch.tensor([0], device=device), internal, torch.tensor([Freq - 1], device=device)])
                    idx = torch.unique(idx)
                    idx, _ = torch.sort(idx)

                # gains at boundary points
                gains_db = (db_min + (db_max - db_min) * torch.rand(idx.numel(), device=device))

                # interpolate to full curve (Freq,)
                full = torch.empty(Freq, device=device)
                for j in range(idx.numel() - 1):
                    left, right = int(idx[j].item()), int(idx[j + 1].item())
                    if right == left:
                        full[left] = gains_db[j]
                        continue
                    seg = torch.linspace(gains_db[j], gains_db[j + 1], steps=(right - left + 1), device=device)
                    full[left:right + 1] = seg

                if self.filter_mode == "add":
                    x_btf[i] = x_btf[i] + full.unsqueeze(0)  # add along freq
                elif self.filter_mode == "mul":
                    # convert dB to amplitude multiplier (rough)
                    mul = torch.pow(10.0, full / 20.0)
                    x_btf[i] = x_btf[i] * mul.unsqueeze(0)
                else:
                    raise ValueError(f"Unknown filter_mode: {self.filter_mode}")

        # ---- Freq-MixStyle ----
        if self.p_fms > 0 and n_apply >= 2 and torch.rand((), device=device) < self.p_fms:
            # receiver: source
            xs = x_btf[:n_apply]  # (ns, T, F)
            xs_f_t = xs.transpose(1, 2)  # (ns, F, T)

            mu_s = xs_f_t.mean(dim=2, keepdim=True)
            sig_s = xs_f_t.std(dim=2, keepdim=True).clamp_min(1e-5)
            xs_norm = (xs_f_t - mu_s) / sig_s

            beta = torch.distributions.Beta(self.fms_alpha, self.fms_alpha)
            lam = beta.sample((n_apply,)).to(device=device).view(n_apply, 1, 1)

            # donor stats
            use_target_donor = (self.fms_mix == "target_donor") and ((B - n_apply) >= 1)
            if use_target_donor:
                xt = x_btf[n_apply:]  # donor: target (NOT modified)
                xt_f_t = xt.transpose(1, 2)  # (nt, F, T)
                mu_t = xt_f_t.mean(dim=2, keepdim=True)
                sig_t = xt_f_t.std(dim=2, keepdim=True).clamp_min(1e-5)

                donor_idx = torch.randint(low=0, high=xt_f_t.size(0), size=(n_apply,), device=device)
                mu2 = mu_t[donor_idx]
                sig2 = sig_t[donor_idx]
            else:
                # fallback: source-only donor inside source
                perm = torch.randperm(n_apply, device=device)
                mu2 = mu_s[perm]
                sig2 = sig_s[perm]

            mu_mix = mu_s * lam + mu2 * (1 - lam)
            sig_mix = sig_s * lam + sig2 * (1 - lam)

            xs_f_t_new = xs_norm * sig_mix + mu_mix

            # only write back to source; target remains unchanged
            x_btf[:n_apply] = xs_f_t_new.transpose(1, 2)

        # ---- Mixup (source-only) ----
        if (
            self.p_mixup > 0
            and y is not None
            and num_src_samples is not None
            and int(num_src_samples) >= 2
            and torch.rand((), device=device) < self.p_mixup
        ):
            ns = int(num_src_samples)
            xs = x_btf[:ns]  # (ns,T,F)
            ys = y[:ns]      # (ns,)

            perm = torch.randperm(ns, device=device)
            beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
            lam = beta.sample((ns,)).to(device=device).view(ns, 1, 1)

            xs_mix = xs * lam + xs[perm] * (1 - lam)
            x_btf[:ns] = xs_mix

            # soft labels
            y_onehot = F.one_hot(ys, num_classes=self.num_classes).float()
            y_mix = y_onehot * lam.view(ns, 1) + y_onehot[perm] * (1 - lam.view(ns, 1))

            y_soft_full = torch.zeros((B, self.num_classes), device=device, dtype=torch.float32)
            y_soft_full[:ns] = y_mix

        x_out = _restore_from_btf(x_btf, meta)
        return x_out, y_soft_full
