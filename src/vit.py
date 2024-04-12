import torch.nn as nn
from torch import Tensor, meshgrid, zeros, cat, bmm, matmul, sqrt, tril
import torch
import numpy as np
from src.diff_utils import ConvType

# Credit for the baseline code for most of the classes in this file goes to Professor Humphrey Shi

class Patching(nn.Module):
    def __init__(self, channels, patch_size, embedding_dim, conv_type):
        super(Patching, self).__init__()
        if conv_type == ConvType.Conv2d:
            # input shape needs to be (N, Channels=3 for images, H, W)
            self.convs = nn.Conv2d(channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
            self.flatten = nn.Flatten(2, 3)
        else:
            # input shape needs to be (N, Channels=3 for images, Time, H, W)
            self.convs = nn.Conv3d(channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
            self.flatten = nn.Flatten(2, 4)

    def forward(self, x):
        # outputs (N, num_patches, embedding_dim)
        return self.flatten(self.convs(x)).transpose(-2, -1)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, seq_len, embedding_dim, num_heads, proj_dim, masked):
        super(MultiheadSelfAttention, self).__init__()

        self.value_proj = nn.Parameter(zeros((num_heads, embedding_dim, proj_dim)), requires_grad=True)
        self.key_proj = nn.Parameter(zeros((num_heads, embedding_dim, proj_dim)), requires_grad=True)
        self.q_heads = nn.Parameter(zeros((num_heads, seq_len, seq_len)), requires_grad=True)
        self.lifting = nn.Parameter(zeros((1, num_heads * proj_dim, embedding_dim)), requires_grad=True)
        self.masked = masked

    def forward(self, x):
        # x is of shape (N, 1 + num_patches, embedding_dim)
        k_proj = matmul(x, self.key_proj) # (N, num_heads, 1 + num_patches, proj_dim)
        v_proj = matmul(x, self.value_proj) # (N, num_heads, 1 + num_patches, proj_dim)
        a_ts = matmul(matmul(self.q_heads, v_proj), k_proj.transpose(-2, -1)).transpose(-2, -1) / np.sqrt(x.shape[-1]) # (N, num_heads, 1 + num_patches, 1 + num_patches)
        a_t = a_ts.softmax(dim=-1)
        if self.masked:
            a_t = tril(a_t) # (N, num_heads, 1 + num_patches, 1 + num_patches)
            a_t = a_t / torch.sum(a_t, dim=-1, keepdim=True)
        attn_vals = matmul(a_t, v_proj).swapaxes(1, 2).flatten(2, 3) # (N, 1 + num_patches, num_heads * proj_dim)
        mha_out = attn_vals @ self.lifting # (N, 1 + num_patches, embedding_dim)
        return mha_out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, seq_len, embedding_dim, num_heads, proj_dim, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()

        self.mhsa = MultiheadSelfAttention(seq_len, embedding_dim, num_heads, proj_dim, False)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(0.2)
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x is of shape (N, 1 + num_patches, embedding_dim)
        attn_output = self.mhsa(x.unsqueeze(1))
        x = self.layer_norm_1(x + attn_output)
        x_ff = self.ff(x)
        x = self.layer_norm_2(x + x_ff)
        return x


class TransformerDecoderLayer(nn.Module):
    '''Note that this is modified to excluse the masked multi-head attention and output embeddings'''
    def __init__(self, seq_len, embedding_dim, num_heads, proj_dim, hidden_dim):
        super(TransformerDecoderLayer, self).__init__()

        self.mhsa_1 = MultiheadSelfAttention(seq_len, embedding_dim, num_heads, proj_dim, True)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.mhsa_2 = MultiheadSelfAttention(seq_len, embedding_dim, num_heads, proj_dim, False)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(0.2)
        )
        self.layer_norm_3 = nn.LayerNorm(embedding_dim)

    def forward(self, encoder_out):
        # x is of shape (N, 1 + num_patches, embedding_dim)
        # attn_output = self.mhsa_1(inputs.unsqueeze(1))
        # x = self.layer_norm_1(inputs + attn_output)
        attn_output = self.mhsa_2(encoder_out.unsqueeze(1))
        x = self.layer_norm_2(encoder_out + attn_output)
        x_ff = self.ff(x)
        x = self.layer_norm_3(x + x_ff)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, channels, in_H, in_W, patch_size, embedding_dim, num_layers, num_heads, proj_dim, hidden_dim, out_dim, cond_dim=3, conv_type=ConvType.Conv2d):
        super(VisionTransformer, self).__init__()
        assert in_H % patch_size == 0 and in_W % patch_size == 0, "Input spatial dimensions must be divisible by patch size"

        self.patcher = Patching(channels, patch_size, embedding_dim, conv_type)
        self.patches_h = (in_H // patch_size)
        self.patches_w = (in_W // patch_size)
        num_patches = self.patches_h * self.patches_w

        sequence_length = num_patches + 1
        
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2)
        )
        self.pos_embed = nn.Parameter(zeros(1, sequence_length, embedding_dim), requires_grad=True)

        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderLayer(sequence_length, embedding_dim, num_heads, proj_dim, hidden_dim)
        for _ in range(num_layers)])

        self.transformer_decoder = nn.Sequential(*[
            TransformerDecoderLayer(sequence_length, embedding_dim, num_heads, proj_dim, hidden_dim)
        for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(embedding_dim)
        # self.mlp = nn.Linear(embedding_dim, out_dim)
        self.mlp = nn.ConvTranspose2d(embedding_dim, channels, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x, y):
        # x is of shape (N, channels, in_H, in_W)
        x = self.patcher(x) # (N, num_patches, embedding_dim)
        cond_token = self.cond_embedding(y).unsqueeze(1) # (N, 1, embedding_dim)
        x = cat((cond_token, x), dim=1) # (N, 1 + num_patches, embedding_dim)
        x += self.pos_embed
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x)[:,1:]
        x = self.mlp(self.layer_norm(x).swapaxes(1, 2).unflatten(2, (self.patches_h, self.patches_w)))
        return x