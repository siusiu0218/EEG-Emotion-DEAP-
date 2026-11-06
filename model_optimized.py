# model_optimized.py
"""
GTE-Net (Hybrid Graph-Transformer Network for Emotion Recognition) 的优化版模型脚本。

主要优化点:
1.  在GAT层后全面添加 Layer Normalization，稳定训练过程。
2.  使用注意力机制改进图读出 (Graph Readout)，替代简单的平均池化。
3.  将主要的激活函数统一为 GELU，与 Transformer 保持一致。
4.  代码结构微调，增强可读性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 辅助组件 (与之前版本基本相同) ---
class Conv1DBankShared(nn.Module):
    def __init__(self, in_chan=1, bank_out_channels=32, ksizes=(15, 31, 63), C_out=64): # 卷积核必须是奇数？如果是偶数的话维度会出问题。（3，7，15），（15，31，63）
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, bank_out_channels, k, stride=1, padding=(k // 2)),
                nn.BatchNorm1d(bank_out_channels),
                nn.GELU() ## [优化]: 统一激活函数
            ) for k in ksizes
        ])
        combined_channels = bank_out_channels * len(ksizes)
        # self.depthwise = nn.Conv1d(combined_channels, combined_channels, kernel_size=3, padding=1, groups=combined_channels)
        self.pointwise = nn.Conv1d(combined_channels, C_out, kernel_size=1)
        self.bn_pw = nn.BatchNorm1d(C_out)
        self.act = nn.GELU() ## [优化]: 统一激活函数

    def forward(self, x):
        B, M, T = x.shape
        x_r = x.view(B * M, 1, T)
        outs = [conv(x_r) for conv in self.convs]
        cat = torch.cat(outs, dim=1)
        # dw = self.depthwise(cat)
        # pw = self.pointwise(dw)
        pw = self.pointwise(cat)
        pw = self.bn_pw(pw)
        pw = self.act(pw)
        pw = pw.view(B, M, pw.shape[1], T)
        return pw

class GraphAttentionLayerDense(nn.Module):
    def __init__(self, in_dim, out_dim, leaky_alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
        self.leaky = nn.LeakyReLU(leaky_alpha)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        B, M, _ = h.shape
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, M, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, M, -1, -1)
        cat = torch.cat([Wh_i, Wh_j], dim=-1)
        e = torch.matmul(cat, self.a).squeeze(-1)
        e = self.leaky(e)
        if adj is not None:
            if adj.dim() == 2: adj = adj.unsqueeze(0).expand(B, -1, -1)
            e = e.masked_fill(adj == 0, -9e15)
        alpha = F.softmax(e, dim=-1)
        out = torch.matmul(alpha, Wh)
        return out

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, head_out, n_heads=4, residual=True, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttentionLayerDense(in_dim, head_out) for _ in range(n_heads)])
        self.residual = residual
        self.out_dim = n_heads * head_out
        self.lin_out = nn.Linear(self.out_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_lin = nn.Linear(in_dim, self.out_dim) if in_dim != self.out_dim else nn.Identity()

    def forward(self, h, adj):
        head_outs = [head(h, adj) for head in self.heads]
        cat = torch.cat(head_outs, dim=-1)
        cat = self.lin_out(cat)
        cat = self.dropout(cat)
        if self.residual:
            cat = cat + self.res_lin(h)
        return cat

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, H_seq):
        scores = torch.matmul(self.proj(H_seq), self.query)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        out = (H_seq * alpha).sum(dim=1)
        return out
        
## [优化]: 新增通道注意力读出模块
class ChannelAttentionReadout(nn.Module):
    """
    改进的图读出机制，使用注意力学习对哪些通道（节点）赋予更高权重。
    """
    def __init__(self, in_dim):
        super().__init__()
        self.attention_scorer = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
    
    def forward(self, node_features):
        # node_features: (B, M, in_dim)
        # 计算每个通道的重要性分数
        scores = self.attention_scorer(node_features).squeeze(-1) # -> (B, M)
        # 将分数转换为权重
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # -> (B, M, 1)
        # 加权求和
        readout = (node_features * weights).sum(dim=1) # -> (B, in_dim)
        return readout

# -------------------------
# 优化版主模型: GTE-Net
# -------------------------

class GTENet(nn.Module):
    def __init__(self, n_channels=32, n_freq_bands=5, time_length=1024,
                 conv_out_channels=64, gat_in_channels=128, gat_heads=4,
                 gat_head_out=32, n_slices=1, d_model=128,
                 transformer_heads=8, transformer_layers=3, ffn_dim=512,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.transformer_layers = transformer_layers
        self.n_slices = n_slices

        self.freq_proj = nn.Sequential(nn.Linear(n_freq_bands, 64), nn.GELU())
        self.conv_frontend = Conv1DBankShared(C_out=conv_out_channels)

        assert gat_in_channels == conv_out_channels + 64, "GAT input dimension mismatch!"
        
        self.gat1 = MultiHeadGATLayer(gat_in_channels, head_out=gat_head_out, n_heads=gat_heads)
        self.gat1_out_dim = gat_heads * gat_head_out
        self.ln1 = nn.LayerNorm(self.gat1_out_dim) ## [优化]: GAT1之后添加LayerNorm

        self.gat2 = MultiHeadGATLayer(self.gat1_out_dim, head_out=gat_head_out, n_heads=gat_heads)
        self.gat2_out_dim = gat_heads * gat_head_out
        self.ln2 = nn.LayerNorm(self.gat2_out_dim) ## [优化]: GAT2之后添加LayerNorm
        
        ## [优化]: 使用通道注意力读出替换简单的平均池化
        self.channel_readout = ChannelAttentionReadout(self.gat2_out_dim)

        self.readout_proj = nn.Linear(self.gat2_out_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(n_slices, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=transformer_heads,
                                                   dim_feedforward=ffn_dim, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        # TransformerEncoder 内部已经包含了 LayerNorm，非常稳定
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.temp_pool = TemporalAttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), ## [优化]: 统一激活函数
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def _slice_windows(self, x):
        B, M, T = x.shape
        w = T // self.n_slices
        return [x[:, :, i*w : (i+1)*w] for i in range(self.n_slices)]

    def _spatial_pipeline(self, x_slice, freq_features, adj):
        time_feat = self.conv_frontend(x_slice)
        time_feat_pooled = time_feat.mean(dim=-1)
        freq_feat_proj = self.freq_proj(freq_features)
        node_features = torch.cat([time_feat_pooled, freq_feat_proj], dim=-1)

        g1_out = self.gat1(node_features, adj)
        g1_out_norm = self.ln1(g1_out) ## [优化]: 应用LayerNorm
        
        g2_out = self.gat2(g1_out_norm, adj)
        g2_out_norm = self.ln2(g2_out) ## [优化]: 应用LayerNorm

        return g2_out_norm

    def forward(self, x_time, x_freq, adj=None):
        slices = self._slice_windows(x_time)
        
        z_list = []
        for s in slices:
            s_g = self._spatial_pipeline(s, x_freq, adj)
            
            # [优化]: 使用新的读出机制
            z = self.channel_readout(s_g) # -> (B, gat2_out_dim)
            z_list.append(z)
            
        Z = torch.stack(z_list, dim=1)
        
        Z_proj = self.readout_proj(Z) # Z_proj应该是一个d_model维度的向量
        if self.transformer_layers == 0:
            H = Z_proj
        else:
            Z_pos = Z_proj + self.pos_emb.unsqueeze(0)

            H = self.transformer(Z_pos)
        H_pool = self.temp_pool(H)
        logits = self.classifier(H_pool)
        
        return logits

# -------------------------
# 示例用法 (保持不变)
# -------------------------
if __name__ == "__main__":
    BATCH = 4; CHANNELS = 32; TIME_POINTS = 1024
    FREQ_BANDS = 5; N_SLICES = 8; NUM_CLASSES = 3

    dummy_x_time = torch.randn(BATCH, CHANNELS, TIME_POINTS)
    dummy_x_freq = torch.randn(BATCH, CHANNELS, FREQ_BANDS)
    dummy_adj = torch.ones(CHANNELS, CHANNELS) 
    
    model = GTENet(
        n_channels=CHANNELS, n_freq_bands=FREQ_BANDS, time_length=TIME_POINTS,
        conv_out_channels=64, gat_in_channels=128, gat_heads=4, gat_head_out=32,
        n_slices=N_SLICES, d_model=128, transformer_heads=8,
        transformer_layers=3, num_classes=NUM_CLASSES
    )
    
    print("Initializing optimized model...")
    with torch.no_grad():
        logits = model(dummy_x_time, dummy_x_freq, dummy_adj)

    print("\n--- Optimized Model Test ---")
    print(f"Model output (logits) shape: {logits.shape}")
    print(f"Expected output shape: {(BATCH, NUM_CLASSES)}")
    assert logits.shape == (BATCH, NUM_CLASSES), "Output shape is incorrect!"
    print("\nOptimized model runs successfully!")
