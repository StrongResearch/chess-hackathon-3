import torch 
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=11, patch_size=4, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.mha(x, x, x)[0]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerTwoHeads(nn.Module):
    def __init__(self, in_channels=11, patch_size=4, embed_dim=192, num_heads=8, num_layers=6, num_classes1=3, num_classes2=1858):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (8 // patch_size) ** 2 + 1, embed_dim))
        
        self.transformer = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head1 = nn.Linear(embed_dim, num_classes1)
        self.head2 = nn.Linear(embed_dim, num_classes2)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        
        out1 = self.head1(cls_token_final)
        out2 = self.head2(cls_token_final)
        
        return out1, out2