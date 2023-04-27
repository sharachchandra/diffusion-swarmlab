import torch
import math
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(1, (half_dim - 1))
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ErrorNet(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(), 
            nn.Linear(time_dim, dim*2)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.SiLU(),
            nn.Linear(dim*2, dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )


    def forward(self, x, time):

        r = x.clone()

        x = self.state_mlp(x)
        x = torch.cat((x, r), dim=1)
        x = self.res_mlp(x)
        t = self.time_mlp(time)
        # t = rearrange(t, "b c -> b c 1")
        scale, shift = t.chunk(2, dim=1)
        x = x * (scale + 1) + shift

        x = self.final_mlp(x)

        return x