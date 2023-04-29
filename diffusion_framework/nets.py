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
        fake_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(fake_dim),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.SiLU(), 
            nn.Linear(fake_dim, dim*2)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(dim, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, dim)
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(dim*2, fake_dim),
            nn.SiLU(),
            nn.Linear(fake_dim, dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, dim)
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
    
class CondErrorNet(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim
    ):
        super().__init__()

        time_dim = dim * 4
        fake_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(fake_dim),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.SiLU(), 
            nn.Linear(fake_dim, dim*2)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(dim+cond_dim, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, fake_dim*2),
            nn.GELU(),
            nn.Linear(fake_dim*2, fake_dim),
            nn.GELU(),
            nn.Linear(fake_dim, dim)
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(cond_dim+dim*2, fake_dim),
            nn.SiLU(),
            nn.Linear(fake_dim, dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(dim+cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )


    def forward(self, x, time, cond):
        state_mlp_input = torch.cat((x, cond), 1)
        r = x.clone()
        x = self.state_mlp(state_mlp_input)
        x = torch.cat((x, r), dim=1)
        
        res_mlp_input = torch.cat((x, cond), 1)
        x = self.res_mlp(res_mlp_input)

        t = self.time_mlp(time)
        scale, shift = t.chunk(2, dim=1)
        x = x * (scale + 1) + shift

        final_mlp_input = torch.cat((x, cond), 1)
        x = self.final_mlp(final_mlp_input)

        return x