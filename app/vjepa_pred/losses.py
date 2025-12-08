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
        self.gm_power = float(self.loss_params.get("power", 1.0))
        self.gm_eps = float(self.loss_params.get("eps", 1e-6))

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
            
        if self.metric_type in ("gm", "generalized_mean", "power_mean"):
            # Generalized Mean
            power = max(self.gm_power, 1e-6)
            gm = torch.pow(torch.mean(torch.pow(diff + self.gm_eps, power), dim=-1), 1.0 / power)
            return gm
            
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
        if isinstance(weights, torch.Tensor) or weights != 1.0:
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
        if weights != 1.0:
            metric = metric * weights
            
        return metric.mean()

    def forward(self, pred, target):
        metric = self.compute_metric(pred, target)
        weights = self.compute_weights(target)
        return self.aggregate_loss(metric, weights)

