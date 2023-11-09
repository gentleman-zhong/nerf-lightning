import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = encode_transient
        self.in_channels_t = in_channels_t if encode_transient else 0
        self.beta_min = beta_min

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir + self.in_channels_a, W // 2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                nn.Linear(W + in_channels_t, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir + self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir + self.in_channels_a], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        static_sigma = self.static_sigma(xyz_)  # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding)  # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1)  # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)  # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding)  # (B, 3)
        transient_beta = self.transient_beta(transient_encoding)  # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1)  # (B, 5)

        return torch.cat([static, transient], 1)  # (B, 9)



# class NeRF(nn.Module):
#     def __init__(self,
#                  D=8, W=256,
#                  in_channels_xyz=63, in_channels_dir=27,
#                  skips=[4]):
#         """
#         D: number of layers for density (sigma) encoder
#         W: number of hidden units in each layer
#         in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
#         in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
#         skips: add skip connection in the Dth layer
#         """
#         super(NeRF, self).__init__()
#         self.D = D
#         self.W = W
#         self.in_channels_xyz = in_channels_xyz
#         self.in_channels_dir = in_channels_dir
#         self.skips = skips
#
#         # xyz encoding layers
#         for i in range(D):
#             if i == 0:
#                 layer = nn.Linear(in_channels_xyz, W)
#             elif i in skips:
#                 layer = nn.Linear(W+in_channels_xyz, W)
#             else:
#                 layer = nn.Linear(W, W)
#             layer = nn.Sequential(layer, nn.ReLU(True))
#             setattr(self, f"xyz_encoding_{i+1}", layer)
#         self.xyz_encoding_final = nn.Linear(W, W)
#
#         # direction encoding layers
#         self.dir_encoding = nn.Sequential(
#                                 nn.Linear(W+in_channels_dir, W//2),
#                                 nn.ReLU(True))
#
#         # output layers
#         self.sigma = nn.Linear(W, 1)
#         self.rgb = nn.Sequential(
#                         nn.Linear(W//2, 3),
#                         nn.Sigmoid())
#
#     def forward(self, x, sigma_only=False):
#         """
#         Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
#         For rendering this ray, please see rendering.py
#
#         Inputs:
#             x: (B, self.in_channels_xyz(+self.in_channels_dir))
#                the embedded vector of position and direction
#             sigma_only: whether to infer sigma only. If True,
#                         x is of shape (B, self.in_channels_xyz)
#
#         Outputs:
#             if sigma_ony:
#                 sigma: (B, 1) sigma
#             else:
#                 out: (B, 4), rgb and sigma
#         """
#         if not sigma_only:
#             input_xyz, input_dir = \
#                 torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
#         else:
#             input_xyz = x
#
#         xyz_ = input_xyz
#         for i in range(self.D):
#             if i in self.skips:
#                 xyz_ = torch.cat([input_xyz, xyz_], -1)
#             xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
#
#         sigma = self.sigma(xyz_)
#         if sigma_only:
#             return sigma
#
#         xyz_encoding_final = self.xyz_encoding_final(xyz_)
#
#         dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
#         dir_encoding = self.dir_encoding(dir_encoding_input)
#         rgb = self.rgb(dir_encoding)
#
#         out = torch.cat([rgb, sigma], -1)
#
#         return out