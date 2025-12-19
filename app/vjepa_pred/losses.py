import torch
import torch.nn.functional as F

class JepaLoss(torch.nn.Module):
    def __init__(self, metric_type="lp", loss_params=None, motion_weight_factor=0.0):
        super().__init__()
        self.name = f"JEPA_{metric_type.upper()}"
        self.metric_type = metric_type
        self.loss_params = loss_params or {}
        self.motion_weight_factor = motion_weight_factor
        
        # -- Metric Config
        self.p = float(self.loss_params.get("p", 1.0))
        self.normalize_by_p = bool(self.loss_params.get("normalize_by_p", True))

    def compute_metric(self, pred, target):
        """
        预测和GT之间计算metric方式
        """
        diff = torch.abs(pred - target)
        
        if self.metric_type in ("lp", "l_p", "power", "p"):
            # Metric: |x-y|^p
            metric = torch.pow(diff, self.p)
            # Mean over feature dim -> [N]
            metric = torch.mean(metric, dim=-1)
            return metric
            
        # Default L1
        return diff.mean(dim=-1)

    def compute_weights(self, motion_map=None):
        """
        Metric加权方式
        """
        if motion_map is None or self.motion_weight_factor == 0:
            return 1.0
        # motion_map is normalized [0,1]
        # boost motion regions by factor alpha
        return 1.0 + self.motion_weight_factor * motion_map

    def aggregate_loss(self, metric, weights):
        """
        Loss方式
        """
        # 1. Apply weights
        if torch.is_tensor(weights):
            loss_per_token = metric * weights
        elif weights != 1.0:
            loss_per_token = metric * weights
        else:
            loss_per_token = metric
            
        # 2. Aggregation (Mean)
        loss_val = torch.mean(loss_per_token)
        
        # 3. Normalization (Specific to Lp)
        if self.metric_type in ("lp", "l_p") and self.normalize_by_p and self.p != 0:
            loss_val = loss_val / self.p
            
        return loss_val

    def forward(self, pred, target, motion_map=None):
        metric = self.compute_metric(pred, target)
        weights = self.compute_weights(motion_map)
        return self.aggregate_loss(metric, weights)


class MotionLoss(torch.nn.Module):
    def __init__(self, metric_type="smooth_l1", loss_params=None, heatmap_cfg=None):
        super().__init__()
        self.name = f"Motion_{metric_type.upper()}"
        self.metric_type = metric_type
        self.loss_params = loss_params or {}
        
        # Loss Specific Params
        self.beta = float(self.loss_params.get("beta", 1.0))
        # Positive Weighting for Sparse Motion (MSE/L1)
        self.pos_weight = float(self.loss_params.get("pos_weight", 1.0))

    def compute_metric(self, pred, target):
        """
        预测和GT之间计算metric方式
        """
        # Metric Calculation
        if self.metric_type in ("l1", "mae"):
            return F.l1_loss(pred, target, reduction='none')
            
        if self.metric_type in ("l2", "mse"):
            return F.mse_loss(pred, target, reduction='none')
            
        if self.metric_type in ("smooth_l1", "huber"):
            return F.smooth_l1_loss(pred, target, beta=self.beta, reduction='none')
            
        raise ValueError(f"Unsupported motion loss type: {self.metric_type}")

    def compute_weights(self, target=None):
        """
        Metric加权方式
        针对手术场景：Global Motion (背景) 占据大面积但强度较低，Local Motion (器械) 强度高且稀疏。
        我们希望模型重点关注 Local Motion。
        
        策略：Soft Threshold Weighting
        - 当 Target 较小 (背景运动) 时，权重保持为 1.0 (或更低)。
        - 当 Target 超过阈值 (显著运动) 时，权重随着强度显著增加。
        """
        if target is None or self.pos_weight == 1.0:
            return 1.0
            
        # 确保 target 不需要梯度
        if target.requires_grad:
            target = target.detach()
            
        # 改进策略：使用非线性加权，抑制低幅值的 Global Motion 影响
        # 假设 pos_weight=5.0。
        # 如果 target=0.2 (背景): weight = 1 + 4 * 0.2^2 = 1.16 (增加不多)
        # 如果 target=0.9 (器械): weight = 1 + 4 * 0.9^2 = 4.24 (显著增加)
        # 使用平方项 (target^2) 可以自然地抑制小值，放大峰值。
        
        return 1.0 + (self.pos_weight - 1.0) * (target ** 2)

    def aggregate_loss(self, metric, weights):
        """
        Loss方式
        """
        # Focal Loss, Wasserstein returns scalar directly
        if metric.numel() == 1:
            return metric
            
        # Others return tensor
        if torch.is_tensor(weights):
            metric = metric * weights
        elif weights != 1.0:
            metric = metric * weights
            
        return metric.mean()

    def forward(self, pred, target):
        metric = self.compute_metric(pred, target)
        weights = self.compute_weights(target)
        return self.aggregate_loss(metric, weights)


class VarianceLoss(torch.nn.Module):
    """
    方差正则化 Loss，防止模型崩塌 (Mode Collapse)。
    改进版：支持 Instance-wise Variance，强制每个样本内部（时空维度）保持多样性。
    """
    def __init__(self, target_std=1.0, eps=1e-4):
        super().__init__()
        self.name = "Variance_Reg"
        self.target_std = target_std
        self.eps = eps

    def forward(self, x):
        """
        x: List[Tensor] 
           每个 Tensor 代表一个样本（Clip）内的所有预测 Token，形状 [N_tokens, D]。
           我们对每个 Tensor 单独计算方差，然后取平均。
           
        或者 x: Tensor [B, N, D]
        """
        if isinstance(x, torch.Tensor):
            # 如果传入的是 Tensor [B, N, D]，则按 B 维度拆分计算
            if x.dim() == 3:
                # Calculate std over N dimension (dim=1) for each batch
                # [B, D]
                std = torch.sqrt(x.var(dim=1, unbiased=True) + self.eps) 
                loss = torch.nn.functional.relu(self.target_std - std).mean()
                return loss
            else:
                # Fallback to global
                x = [x]

        losses = []
        for feat in x:
            # feat: [N_tokens, D]
            # 必须至少有 2 个 Token 才能计算方差
            if feat.size(0) > 1:
                # 计算该样本内部（时空）的方差
                # var: [D]
                var = feat.var(dim=0, unbiased=True)
                std = torch.sqrt(var + self.eps)
                
                # Hinge Loss: std 必须 > target_std
                # Mean over D dimension
                loss_i = torch.nn.functional.relu(self.target_std - std).mean()
                losses.append(loss_i)
        
        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            # Handle empty or single-token cases
            dev = x[0].device if x and isinstance(x[0], torch.Tensor) else torch.device('cuda')
            return torch.tensor(0.0, device=dev)


class CovarianceLoss(torch.nn.Module):
    """
    协方差正则化 Loss (类似于 VICReg 中的 Covariance Loss)。
    
    目的：
    防止 Feature Collapse 的另一种形式——维度冗余。
    VarianceLoss 保证了特征不为常数，CovarianceLoss 保证了不同的特征通道 (Feature Dimension) 
    编码了不同的信息（去相关性 Decorrelation）。
    
    计算方式：
    计算特征矩阵的协方差矩阵，使得非对角线元素趋向于 0。
    """
    def __init__(self, num_features=None):
        super().__init__()
        self.name = "Covariance_Reg"
        self.num_features = num_features

    def forward(self, x):
        """
        x: List[Tensor] or Tensor
        """
        if isinstance(x, list):
            # 合并所有 Tokens 进行全局统计
            # 这比单 Clip 统计更稳健，因为协方差需要较多样本
            if not x:
                return torch.tensor(0.0).to(x[0].device if x else 'cuda')
            x = torch.cat(x, dim=0) # [N_total, D]

        if x.dim() == 3:
            # [B, N, D] -> [B*N, D]
            x = x.flatten(0, 1)
            
        # x: [N, D]
        N, D = x.shape
        if N < 2:
            return torch.tensor(0.0, device=x.device)

        # 1. Center the features
        x = x - x.mean(dim=0, keepdim=True)
        
        # 2. Covariance Matrix: [D, D]
        cov_matrix = (x.T @ x) / (N - 1)
        
        # 3. Off-diagonal elements
        # 也就是让不同维度之间的相关性为 0
        off_diag = cov_matrix.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
        
        # Loss is sum of squares of off-diagonal elements
        loss = off_diag.pow(2).sum() / D
        
        return loss

class RelationLoss(torch.nn.Module):
    """
    关系一致性 Loss (Relation Loss / Structural Loss)
    
    目的：
    约束 Prediction 和 Target 之间的“Token 间关系结构”保持一致。
    如果是 Target 认为 Token A 和 Token B 很像，那么 Prediction 也应该认为它们很像。
    这通过匹配两者的 Gram Matrix (Similarity Matrix) 来实现。
    
    数学：
    Sim_pred = Normalize(Pred) @ Normalize(Pred).T  (大小 N x N)
    Sim_target = Normalize(Target) @ Normalize(Target).T
    Loss = SmoothL1(Sim_pred, Sim_target)
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.name = "Relation_Loss"
        self.temperature = temperature

    def forward(self, pred, target):
        """
        pred: [N, D]  (N 个 Masked Tokens)
        target: [N, D]
        """
        N, D = pred.shape
        if N < 2:
             return torch.tensor(0.0, device=pred.device)

        # 1. Normalize features (Cosine Similarity)
        # [N, D]
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # 2. Compute Similarity Matrix (Gram Matrix)
        # [N, N]
        sim_pred = torch.mm(pred_norm, pred_norm.t()) / self.temperature
        sim_target = torch.mm(target_norm, target_norm.t()) / self.temperature
        
        # 3. Compute Distance between structures
        # 使用 Smooth L1 更稳健
        loss = F.smooth_l1_loss(sim_pred, sim_target, reduction='mean')
        
        return loss
