import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math


def k_nearest(x, k):
    x_np = x.cpu().numpy()
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


class GaussianContainer(nn.Module):
    def __init__(self, xyz, rgb, num_gs, sh_degree, random_scale=1.0):
        super().__init__()
        if xyz is not None:
            num_gs = len(xyz)
        else:
            xyz = (torch.rand((num_gs, 3)) - 0.5) * random_scale
        self.sh_degree = sh_degree
        means = torch.nn.Parameter(xyz.float().contiguous())
        distances, _ = k_nearest(means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log((avg_dist + 0.001).repeat(1, 3)))
        quats = torch.nn.Parameter(random_quat_tensor(num_gs).contiguous())
        opacities = torch.nn.Parameter(
            torch.logit(0.1 * torch.ones(num_gs, 1)).contiguous()
        )
        dim_sh = (sh_degree + 1) ** 2
        if rgb is not None:
            shs = torch.zeros((rgb.shape[0], dim_sh, 3)).float()
            if sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(rgb / 260)
                shs[:, 1:, 3:] = 0.0
            else:
                shs[:, 0, :3] = torch.logit((rgb.float() + 1) / 260)
            features_dc = torch.nn.Parameter(shs[:, 0, :]).contiguous()
            features_rest = torch.nn.Parameter(shs[:, 1:, :]).contiguous()
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_gs, 3)).contiguous()
            features_rest = torch.nn.Parameter(
                torch.zeros((num_gs, dim_sh - 1, 3))
            ).contiguous()

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "features_dc": features_dc,
                "features_rest": features_rest,
            }
        )

    def appearance(self, degree: int | None = None) -> tuple[torch.Tensor, int | None]:
        """Return direct RGB for SH0 mode, or SH coefficients and active degree."""
        params = self.gauss_params
        if self.sh_degree == 0:
            return params["features_dc"].sigmoid(), None
        coefficients = torch.cat(
            (params["features_dc"][:, None], params["features_rest"]), dim=1
        )
        return coefficients, self.sh_degree if degree is None else degree

    @torch.no_grad()
    def save_ply(
        self, path: str, *, extra_attributes: dict[str, torch.Tensor] | None = None
    ):
        """Save raw Gaussian parameters, optionally with scalar per-GS fields."""
        import os
        import numpy as np
        from plyfile import PlyData, PlyElement

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        params = self.gauss_params

        def to_numpy(x):
            return x.detach().cpu().contiguous().numpy().astype(np.float32)

        # (N, 3)
        xyz = to_numpy(params["means"])
        normals = np.zeros_like(xyz)

        # (N, 3)
        features_dc = to_numpy(params["features_dc"])

        # (N, K-1, 3) -> (N, 3*(K-1))
        features_rest = to_numpy(params["features_rest"])
        features_rest = features_rest.transpose(0, 2, 1).reshape(
            xyz.shape[0], 3 * params["features_rest"].shape[1]
        )

        opacities = to_numpy(params["opacities"])  # (N, 1)
        scales = to_numpy(params["scales"])  # (N, 3)
        quats = to_numpy(params["quats"])  # (N, 4)

        attributes = (
            ["x", "y", "z", "nx", "ny", "nz"]
            + [f"f_dc_{i}" for i in range(features_dc.shape[1])]
            + [f"f_rest_{i}" for i in range(features_rest.shape[1])]
            + ["opacity"]
            + [f"scale_{i}" for i in range(scales.shape[1])]
            + [f"rot_{i}" for i in range(quats.shape[1])]
        )

        data = np.concatenate(
            [
                xyz,
                normals,
                features_dc,
                features_rest,
                opacities,
                scales,
                quats,
            ],
            axis=1,
        )

        if extra_attributes:
            extras = []
            for name, value in extra_attributes.items():
                if name in attributes or value.shape not in (
                    (len(xyz),),
                    (len(xyz), 1),
                ):
                    raise ValueError(
                        f"Expected a unique scalar per-Gaussian attribute: {name}"
                    )
                attributes.append(name)
                extras.append(to_numpy(value).reshape(len(xyz), 1))
            data = np.concatenate([data, *extras], axis=1)

        dtype = [(name, "f4") for name in attributes]
        elements = np.empty(xyz.shape[0], dtype=dtype)

        for i, name in enumerate(attributes):
            elements[name] = data[:, i]

        vertex = PlyElement.describe(elements, "vertex")
        PlyData([vertex], text=False).write(path)
