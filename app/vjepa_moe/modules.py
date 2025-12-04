import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils.modules import MLP

class SoftMoEMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Router: 输入 -> 专家权重
        self.router = nn.Linear(in_features, num_experts)
        
        # Experts: 并行的 MLP
        self.experts = nn.ModuleList([
            MLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x: [Batch, Tokens, Dim]
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1) # [B, N, num_experts]
        
        final_output = 0
        for i in range(self.num_experts):
            # 获取第 i 个专家的输出
            expert_out = self.experts[i](x)
            # 获取第 i 个专家的权重 (广播用)
            w = routing_weights[..., i].unsqueeze(-1)
            final_output += w * expert_out
            
        return final_output
