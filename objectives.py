import torch
import torch.nn.functional as F

from tools import mapto_SPD_cone


jitter=1e-7


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def orthogonality_loss(z_s: torch.Tensor, z_d: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    非对抗 L_orth：惩罚 z_s 与 z_d 的相关性（用标准化后的交叉协方差）
    输出为 mean-squared correlation，尺度对维度不敏感，更好调参。
    """
    n = z_s.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=z_s.device)

    zs = z_s - z_s.mean(dim=0, keepdim=True)
    zd = z_d - z_d.mean(dim=0, keepdim=True)

    zs = zs / (zs.std(dim=0, keepdim=True) + eps)
    zd = zd / (zd.std(dim=0, keepdim=True) + eps)

    C = (zs.t() @ zd) / (n - 1)               # (Ds, Dd)
    return (C ** 2).mean()


def device_independence_loss(z_s: torch.Tensor, domains: torch.Tensor, num_devices: int | None = None, eps: float = 1e-5) -> torch.Tensor:
    """
    非对抗 L_indep：最小化 z_s 与设备标签 d 的统计依赖（线性HSIC/相关性惩罚）
    做法：z_s 与 one-hot(d) 的标准化交叉协方差的均方。
    """
    n = z_s.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=z_s.device)

    if num_devices is None:
        num_devices = int(domains.max().item()) + 1

    D = F.one_hot(domains, num_classes=num_devices).float()  # (n, K)

    zs = z_s - z_s.mean(dim=0, keepdim=True)
    zs = zs / (zs.std(dim=0, keepdim=True) + eps)

    D = D - D.mean(dim=0, keepdim=True)
    D = D / (D.std(dim=0, keepdim=True) + eps)

    C = (zs.t() @ D) / (n - 1)                # (Ds, K)
    return (C ** 2).mean()


def ddc_mmd(data, nb_src_samples):
    """
     This implements Equation 1 from the "Deep Domain Confusion: Maximizing for Domain Invariance" paper.

     Args:
         data (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
         nb_src_samples (int): Number of source samples in the batch.

     Returns:
         torch.Tensor: Scalar tensor representing the MMD distance.
     """
    phi_source = data[:nb_src_samples]
    phi_target = data[nb_src_samples:]

    mean_source = phi_source.mean(dim=0)
    mean_target = phi_target.mean(dim=0)

    diff = mean_source - mean_target

    mmd_squared = torch.sum(diff ** 2)

    return mmd_squared

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ||x_i - y_j||^2, shape: (Nx, Ny)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (Nx,1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (Ny,1)
    return (x_norm + y_norm.t() - 2.0 * (x @ y.t())).clamp_min(0.0)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Multi-kernel RBF: sum_k exp(-d2 / (2*sigma_k^2))
    d2 = _pairwise_sq_dists(x, y)  # (Nx,Ny)
    sig2 = (sigmas ** 2).clamp_min(eps).view(1, -1)  # (1,K)
    k = torch.exp(-d2.unsqueeze(-1) / (2.0 * sig2))   # (Nx,Ny,K)
    return k.sum(dim=-1)                              # (Nx,Ny)


def mmd_rbf(data: torch.Tensor, nb_src_samples: int, num_kernels: int = 5, kernel_mul: float = 2.0) -> torch.Tensor:
    """
    RBF 多核 MMD（biased 版本，数值稳定，适合 baseline）
    data: (N,D)=[src;tgt]
    """
    h_src = data[:nb_src_samples]
    h_tgt = data[nb_src_samples:]

    if h_src.size(0) == 0 or h_tgt.size(0) == 0:
        return torch.tensor(0.0, device=data.device)

    # median heuristic: 用 src-tgt 距离的中位数估计带宽基准
    with torch.no_grad():
        d2 = _pairwise_sq_dists(h_src, h_tgt)
        valid = d2[d2 > 0]
        median = torch.median(valid) if valid.numel() > 0 else torch.tensor(1.0, device=data.device)
        sigma_base = torch.sqrt(median).clamp_min(1e-3)

        offsets = torch.arange(num_kernels, device=data.device) - (num_kernels // 2)
        sigmas = sigma_base * (kernel_mul ** offsets.float())

    Kxx = _rbf_kernel(h_src, h_src, sigmas)
    Kyy = _rbf_kernel(h_tgt, h_tgt, sigmas)
    Kxy = _rbf_kernel(h_src, h_tgt, sigmas)

    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()



def coral_loss(batch, nb_src_samples):
    """
    This implements Equation 1 from the "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" paper.
    Args:
         batch (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
         nb_src_samples (int): Number of source samples in the batch.

    Returns:
        torch.Tensor: Scalar tensor representing the CORAL loss value.

    """

    H_s = batch[:nb_src_samples]
    H_t = batch[nb_src_samples:]

    cov_s = torch.cov(H_s.T)
    cov_t = torch.cov(H_t.T)

    loss = torch.sum((cov_s - cov_t) ** 2)

    return loss


def minimal_entropy_correlation_alignment(data, nb_src_samples):
    """
    Compute the Minimal-Entropy Correlation Alignment (log-CORAL) loss between source and target domains.
    This function is adapted from the official implementation at
    https://github.com/pmorerio/minimal-entropy-correlation-alignment/tree/master.

    Args:
        data (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
        nb_src_samples (int): Number of source samples in the batch.

    Returns:
        torch.Tensor: Log-CORAL loss value.
    """

    h_src = data[:nb_src_samples]
    h_trg = data[nb_src_samples:]

    batch_size = float(h_src.size(0))

    # Subtract the mean from the data matrix (center the data)
    h_src = h_src - torch.mean(h_src, dim=0)
    h_trg = h_trg - torch.mean(h_trg, dim=0)

    # Compute covariance matrices
    cov_source = (1.0 / (batch_size - 1)) * torch.mm(h_src.t(), h_src)
    cov_target = (1.0 / (batch_size - 1)) * torch.mm(h_trg.t(), h_trg)

    # Eigenvalue decomposition
    eig_vals_source, eig_vecs_source = torch.linalg.eigh(cov_source)
    eig_vals_target, eig_vecs_target = torch.linalg.eigh(cov_target)

    # Add small epsilon to avoid log(0)
    eig_vals_source = torch.clamp(eig_vals_source, min=1e-12)
    eig_vals_target = torch.clamp(eig_vals_target, min=1e-12)

    log_cov_source = torch.mm(eig_vecs_source, torch.mm(torch.diag(torch.log(eig_vals_source)), eig_vecs_source.t()))

    log_cov_target = torch.mm(eig_vecs_target, torch.mm(torch.diag(torch.log(eig_vals_target)), eig_vecs_target.t()))

    return torch.mean(torch.square(log_cov_source - log_cov_target))


def central_moment_discrepancy(data, nb_src_samples, n_moments=2):
    """
    Compute Central Moment Discrepancy (CMD) between source and target domains.
    This function is adapted from the officially released code at
    https://github.com/wzell/cmd/tree/master.

    Args:
        data (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
        nb_src_samples (int): Number of source samples in the batch.
        n_moments (int): Number of moments to match (default: 2).

    Returns:
        torch.Tensor: CMD loss value.
    """

    def matchnorm(x1, x2):
        """Compute L2 norm of difference between two tensors."""

        return torch.sqrt(torch.clamp(torch.sum((x1 - x2) ** 2), min=jitter))

    def scm(sx1, sx2, k):
        """Compute k-th central moment matching."""
        ss1 = torch.mean(sx1 ** k, dim=0)
        ss2 = torch.mean(sx2 ** k, dim=0)
        return matchnorm(ss1, ss2)

    def mmatch(x1, x2, n_moments):
        """Main moment matching function."""
        mx1 = torch.mean(x1, dim=0)
        mx2 = torch.mean(x2, dim=0)

        sx1 = x1 - mx1
        sx2 = x2 - mx2

        # Match first moment (means)
        dm = matchnorm(mx1, mx2)
        scms = dm

        # Match higher order central moments
        for i in range(n_moments - 1):
            scms = scms + scm(sx1, sx2, i + 2)

        return scms

    # Split data into source and target
    source_data = data[:nb_src_samples]
    target_data = data[nb_src_samples:]

    # Compute CMD loss
    cmd_loss = mmatch(source_data, target_data, n_moments)

    return cmd_loss


def higher_order_moment_matching(data, nb_src_samples, order=2):
    """
    Compute Higher-order Moment Matching (HoMM) loss using an adapted version of the exact implementation style
    from the official paper code for orders 2 and 3 at
    https://github.com/chenchao666/HoMM-Master/tree/master.

    Args:
        data (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
        nb_src_samples (int): Number of source samples in the batch.
        order (int): Order of moments to match (2 or 3).

    Returns:
        torch.Tensor: HoMM loss value.
    """
    h_src = data[:nb_src_samples]
    h_trg = data[nb_src_samples:]

    h_src = h_src - torch.mean(h_src, dim=0)
    h_trg = h_trg - torch.mean(h_trg, dim=0)

    if order == 2:
        xs = h_src.unsqueeze(-1)
        xt = h_trg.unsqueeze(-1)

        xs_1 = xs.transpose(1, 2)
        xt_1 = xt.transpose(1, 2)

        HR_Xs = xs * xs_1
        HR_Xs = torch.mean(HR_Xs, dim=0)

        HR_Xt = xt * xt_1
        HR_Xt = torch.mean(HR_Xt, dim=0)

    elif order == 3:
        xs = h_src.unsqueeze(-1).unsqueeze(-1)
        xt = h_trg.unsqueeze(-1).unsqueeze(-1)

        xs_1 = xs.transpose(1, 2)
        xs_2 = xs.transpose(1, 3)

        xt_1 = xt.transpose(1, 2)
        xt_2 = xt.transpose(1, 3)

        HR_Xs = xs * xs_1 * xs_2
        HR_Xs = torch.mean(HR_Xs, dim=0)

        HR_Xt = xt * xt_1 * xt_2
        HR_Xt = torch.mean(HR_Xt, dim=0)

    else:
        raise ValueError(f"Order {order} not supported. Use order 2 or 3.")

    return torch.mean((HR_Xs - HR_Xt) ** 2)


def geo_adapt(feat_batch, nb_src_samples, metric_type='hilbert'):
    """
    GeoAdapt: Geometric Moment Alignment
    Args:
        feat_batch (torch.Tensor): Combined batch of shape (N, D), where N is total samples and D is the feature dimension.
        nb_src_samples (int): Number of source samples in the batch.
        metric_type (str): The type of the distance metric to use.

    Returns:

    """

    Z_s = feat_batch[:nb_src_samples]
    Z_t = feat_batch[nb_src_samples:]

    mu_s = Z_s.mean(dim=0)
    mu_t = Z_t.mean(dim=0)

    cov_s = torch.cov(Z_s.T)
    cov_s = (cov_s + cov_s.T) / 2.0
    cov_s = cov_s + torch.eye(cov_s.size(0)).to(cov_s.device) * jitter

    cov_t = torch.cov(Z_t.T)
    cov_t = (cov_t + cov_t.T) / 2.0
    cov_t = cov_t + torch.eye(cov_t.size(0)).to(cov_t.device) * jitter

    Ps = mapto_SPD_cone(cov_s, mu_s, beta=1.0)
    Pt= mapto_SPD_cone(cov_t, mu_t, beta=1.0)

    Ps_eigvals = torch.linalg.eigvalsh(Ps).real
    Ps_det = torch.prod(Ps_eigvals)

    # print('Ps -- det: {:.2e}, max_eig: {:.4f}, min_eig: {:.4f}, k: {:.2e}'.format(Ps_det,
    #                                                                               Ps_eigvals.max(), Ps_eigvals.min(),
    #                                                                               Ps_eigvals.max()/Ps_eigvals.min())
    #       )
    # print('-------------------------------------------------------------------------------')

    if metric_type == 'hilbert':
        L0 = torch.linalg.cholesky(Ps)
        eigenvalues = torch.linalg.eigvals(torch.cholesky_solve(Pt, L0)).real

        max_eigenvalue = torch.max(eigenvalues)
        min_eigenvalue = torch.min(eigenvalues)

        loss = torch.log(max_eigenvalue / min_eigenvalue)

    elif metric_type == 'airm':
        L0 = torch.linalg.cholesky(Ps)
        eigenvalues = torch.linalg.eigvals(torch.cholesky_solve(Pt, L0)).real

        loss = torch.sqrt(torch.sum(torch.log(eigenvalues)**2))

    else:
        ValueError('The requested distance metric type is not implemented!')

    return loss, Ps_det


adapt_loss_functions = {
    'ddc': ddc_mmd,
    'mmd': mmd_rbf,
    'coral': coral_loss,
    'log_coral': minimal_entropy_correlation_alignment,
    'cmd': central_moment_discrepancy,
    'homm': higher_order_moment_matching,
    'geo_adapt': geo_adapt
}

'''
top-level wrapper
'''

def compute_batch_loss(batch, P):
    """
    使用P参数控制的批次损失计算函数
    P参数说明：
    - exp_mode: 实验模式，'classification' 表示仅分类，'adaptation' 表示域适应
    - adapt_method: 域适应方法，如 'ddc', 'coral', 'log_coral', 'cmd', 'homm', 'geo_adapt'
    - user_lamda: 用户定义的域适应损失权重
    - highest_moment: 中心矩匹配的最高阶数（用于CMD和HoMM方法）
    - dist_metric_type: 几何适应的距离度量类型（用于geo_adapt方法）
    - det_thr: 行列式阈值（用于geo_adapt方法）
    - current_epoch: 当前训练轮数（用于warmup）
    - lambda_dev: 设备分类损失权重
    """
    assert batch['logits'].dim() == 2

    if P.get('exp_mode') == 'adaptation' and 'adapt_method' in P:
        # 计算分类损失，只对源域样本计算
        if 'num_src_samples' in batch:
            ns = batch['num_src_samples']
            if 'labels_onehot' in batch and batch['labels_onehot'] is not None:
                class_loss = soft_cross_entropy(batch['logits'][:ns], batch['labels_onehot'][:ns])
            else:
                class_loss = torch.nn.functional.cross_entropy(batch['logits'][:ns], batch['labels'][:ns])
        else:
            if 'labels_onehot' in batch and batch['labels_onehot'] is not None:
                class_loss = soft_cross_entropy(batch['logits'], batch['labels_onehot'])
            else:
                class_loss = torch.nn.functional.cross_entropy(batch['logits'], batch['labels'])
        
        # 设备预测损失：对源+目标都计算（前提是给了 domains 与 dev_logits）
        dev_loss = None
        lambda_dev = P.get('lambda_dev', getattr(__import__('config.config', fromlist=['config']).config, 'LAMBDA_DEV', 0.1))

        if ('dev_logits' in batch) and (batch['dev_logits'] is not None) and ('domains' in batch):
            dev_loss = F.cross_entropy(batch['dev_logits'], batch['domains'])
        else:
            dev_loss = torch.tensor(0.0, device=batch['logits'].device)
        
        # 正交损失和独立性损失
        lambda_orth = P.get('lambda_orth', 0.0)
        lambda_indep = P.get('lambda_indep', 0.0)
        num_devices = P.get('num_devices', None)

        orth_loss = torch.tensor(0.0, device=batch['logits'].device)
        indep_loss = torch.tensor(0.0, device=batch['logits'].device)

        if lambda_orth > 0 and ('z_s' in batch) and ('z_d' in batch):
            orth_loss = orthogonality_loss(batch['z_s'], batch['z_d'])

        if lambda_indep > 0 and ('z_s' in batch) and ('domains' in batch):
            indep_loss = device_independence_loss(batch['z_s'], batch['domains'], num_devices=num_devices)

        # 计算域适应损失
        if 'latent_feat' in batch and 'num_src_samples' in batch:
            adapt_method = P['adapt_method']
            adapt_loss_fn = adapt_loss_functions.get(adapt_method)
            
            if adapt_loss_fn:
                if adapt_method == 'geo_adapt':
                    da_loss, batch['Ps_det'] = adapt_loss_fn(
                        batch['latent_feat'], 
                        batch['num_src_samples'],
                        metric_type=P.get('dist_metric_type', 'hilbert')
                    )

                    if batch.get('Ps_det', 1.0) >= P.get('det_thr', 1e-6):
                        lambda_weight = P.get('user_lamda', 0.1)
                    else:
                        lambda_weight = 0.0

                elif adapt_method in ['cmd', 'homm']:
                    da_loss = adapt_loss_fn(
                        batch['latent_feat'], 
                        batch['num_src_samples'], 
                        P.get('highest_moment', 2)
                    )
                    lambda_weight = P.get('user_lamda', 0.1)

                else:
                    da_loss = adapt_loss_fn(batch['latent_feat'], batch['num_src_samples'])
                    lambda_weight = P.get('user_lamda', 0.1)

                # 总损失 = 分类损失 + 权重 * 域适应损失 + 权重 * 设备损失 + 权重 * 正交损失 + 权重 * 独立性损失
                total_loss = class_loss + (lambda_weight * da_loss) + (lambda_dev * dev_loss) \
                             + (lambda_orth * orth_loss) + (lambda_indep * indep_loss)
                
                batch['da_loss_np'] = da_loss.clone().detach().cpu().numpy()
                batch['lamda'] = lambda_weight
            else:
                # 如果没有指定的适应方法，只使用分类损失、设备损失、正交损失和独立性损失
                total_loss = class_loss + (lambda_dev * dev_loss) \
                             + (lambda_orth * orth_loss) + (lambda_indep * indep_loss)
                batch['da_loss_np'] = None
                batch['lamda'] = 0.0
        else:
            # 如果没有域适应相关数据，只使用分类损失、设备损失、正交损失和独立性损失
            total_loss = class_loss + (lambda_dev * dev_loss) \
                         + (lambda_orth * orth_loss) + (lambda_indep * indep_loss)
            batch['da_loss_np'] = None
            batch['lamda'] = 0.0
            
        batch['loss_tensor'] = total_loss
        batch['dev_loss_np'] = dev_loss.clone().detach().cpu().numpy()
        batch['lambda_dev'] = lambda_dev
        batch['orth_loss_np'] = orth_loss.detach().cpu().numpy()
        batch['indep_loss_np'] = indep_loss.detach().cpu().numpy()
    else:
        # 只计算分类损失
        if 'labels_onehot' in batch and batch['labels_onehot'] is not None:
            class_loss = soft_cross_entropy(batch['logits'], batch['labels_onehot'])
        else:
            class_loss = torch.nn.functional.cross_entropy(batch['logits'], batch['labels'])
        
        # 设备预测损失：对源+目标都计算（前提是给了 domains 与 dev_logits）
        dev_loss = None
        lambda_dev = P.get('lambda_dev', getattr(__import__('config.config', fromlist=['config']).config, 'LAMBDA_DEV', 0.1))

        if ('dev_logits' in batch) and (batch['dev_logits'] is not None) and ('domains' in batch):
            dev_loss = F.cross_entropy(batch['dev_logits'], batch['domains'])
        else:
            dev_loss = torch.tensor(0.0, device=batch['logits'].device)
        
        # 正交损失和独立性损失
        lambda_orth = P.get('lambda_orth', 0.0)
        lambda_indep = P.get('lambda_indep', 0.0)
        num_devices = P.get('num_devices', None)

        orth_loss = torch.tensor(0.0, device=batch['logits'].device)
        indep_loss = torch.tensor(0.0, device=batch['logits'].device)

        if lambda_orth > 0 and ('z_s' in batch) and ('z_d' in batch):
            orth_loss = orthogonality_loss(batch['z_s'], batch['z_d'])

        if lambda_indep > 0 and ('z_s' in batch) and ('domains' in batch):
            indep_loss = device_independence_loss(batch['z_s'], batch['domains'], num_devices=num_devices)
        
        # 总损失 = 分类损失 + 权重 * 设备损失 + 权重 * 正交损失 + 权重 * 独立性损失
        total_loss = class_loss + (lambda_dev * dev_loss) \
                     + (lambda_orth * orth_loss) + (lambda_indep * indep_loss)
        batch['loss_tensor'] = total_loss
        batch['da_loss_np'] = None
        batch['lamda'] = 0.0
        batch['dev_loss_np'] = dev_loss.clone().detach().cpu().numpy()
        batch['lambda_dev'] = lambda_dev
        batch['orth_loss_np'] = orth_loss.detach().cpu().numpy()
        batch['indep_loss_np'] = indep_loss.detach().cpu().numpy()

    # 保存分类损失值
    batch['cs_loss_np'] = class_loss.clone().detach().cpu().numpy() if 'class_loss' in locals() else batch['loss_tensor'].clone().detach().cpu().numpy()

    return batch
