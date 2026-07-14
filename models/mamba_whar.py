import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class BodyNodeAttentionUnit(nn.Module):
    """Spatial Graph Module (Extracts skeleton posture instantly)"""
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        agg_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )
        self.gin_agg = GINConv(agg_nn, train_eps=True)
        
        att_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.Tanh()
        )
        self.gin_att = GINConv(att_nn, train_eps=True)

    def forward(self, x, edge_index):
        node_features = self.gin_agg(x, edge_index)
        attention_weights = self.gin_att(x, edge_index)
        return node_features * attention_weights

class PurePyTorchMambaBlock(nn.Module):
    """Upgraded Pure PyTorch Mamba with Residuals, LayerNorm, and Dropout"""
    def __init__(self, d_model, expand=2, d_conv=4, dropout=0.3):
        super().__init__()
        self.d_inner = d_model * expand
        
        # Pre-normalization (standard in modern architectures)
        self.norm = nn.LayerNorm(d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Prevent overfitting to specific subjects
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        
        # 1. Save original input for residual connection
        residual = x
        
        # 2. Normalize
        x = self.norm(x)
        
        # 3. Mamba Forward Pass
        x_proj = self.in_proj(x)
        x_proj, res = x_proj.chunk(2, dim=-1)
        
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_out = self.ssm_proj(x_conv) * F.silu(res)
        out = self.out_proj(ssm_out)
        out = self.dropout(out)
        
        # 4. Residual Addition!
        return out + residual

class Mamba_WHAR(nn.Module):
    def __init__(self, num_nodes=5, in_features=9, hidden_dim=64, num_classes=19):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Skeleton Adjacency
        edges = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0]]
        self.base_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Spatial Graph
        self.spatial_graph = BodyNodeAttentionUnit(in_features, hidden_dim)
        
        # Temporal Mamba with sequence pooling
        mamba_dim = num_nodes * hidden_dim
        self.mamba = PurePyTorchMambaBlock(d_model=mamba_dim, dropout=0.3)
        
        self.norm_final = nn.LayerNorm(mamba_dim)
        self.classifier = nn.Linear(mamba_dim, num_classes)

    def get_batched_edge_index(self, batch_size, device):
        edge_index = self.base_edge_index.to(device)
        num_edges = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, batch_size)
        offset = torch.arange(0, batch_size, device=device) * self.num_nodes
        offset = offset.view(-1, 1).repeat(1, num_edges).view(1, -1)
        return batched_edge_index + offset

    def forward(self, x):
        B, T, N, F = x.shape
        device = x.device
        
        x_flat = x.view(B * T * N, F)
        edge_index = self.get_batched_edge_index(B * T, device)
        
        graph_out = self.spatial_graph(x_flat, edge_index)
        seq_x = graph_out.view(B, T, N * self.hidden_dim)
        
        # Temporal Processing
        mamba_out = self.mamba(seq_x)
        
        # Classify based on the FINAL normalized state
        last_state = self.norm_final(mamba_out[:, -1, :])
        return self.classifier(last_state)