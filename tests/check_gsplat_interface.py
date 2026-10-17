"""gsplat-style layouts, camera conversion, alpha gradients and API guards.

Run: python -m tests.check_gsplat_interface
No gsplat or reference CUDA extension is imported by this check.
"""

from dataclasses import replace
import math
import subprocess
import sys

import torch
import torch.nn.functional as F

from dsl_splat import (
    ExtendedSettings,
    GaussianRasterizationSettings,
    GaussianRasterizer,
    rasterization,
)
from dsl_splat.aaa.settings import (
    CullingSettings,
    SortSettings,
    SortMode,
    GlobalSortOrder,
)
from dsl_splat.aaa.cameras import prepare_cameras
from dsl_splat.inputs import normalize_inputs


def fixture(n=11, channels=7, cameras=2):
    torch.manual_seed(93)
    means = torch.randn(n, 3, device="cuda") * 0.2
    means[:, 2] = torch.linspace(2.5, 5, n, device="cuda")
    quats = torch.randn(n, 4, device="cuda") * 2  # deliberately unnormalized
    scales = torch.rand(n, 3, device="cuda") * 0.12 + 0.1
    opacities = torch.rand(n, device="cuda") * 0.4 + 0.2
    colors = torch.randn(n, channels * 2, device="cuda")[:, ::2]
    views = torch.eye(4, device="cuda").repeat(cameras, 1, 1)
    if cameras > 1:
        a = 0.16
        views[1, :3, :3] = torch.tensor(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]],
            device="cuda",
        )
        views[1, 0, 3] = -0.3
    Ks = torch.tensor(
        [[43.0, 0, 29.0], [0, 37.0, 21.0], [0, 0, 1.0]], device="cuda"
    ).repeat(cameras, 1, 1)
    return [means, quats, scales, opacities, colors, views, Ks, 64, 48]


def settings(antialiased=True, centered=False):
    return ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.HIER, sort_order=GlobalSortOrder.PTD_MAX
        ),
        culling_settings=CullingSettings(True, True, True, True),
        load_balancing=True,
        proper_ewa_scaling=antialiased,
        eval_3D=True,
        new_aabb=centered,
    )


def direct_config(view, k, width, height, background, near=0.01, far=1e10):
    fx, fy, cx, cy = [
        float(k[row, col]) for row, col in ((0, 0), (1, 1), (0, 2), (1, 2))
    ]
    projection = torch.tensor(
        [
            [2 * fx / width, 0, 2 * cx / width - 1, 0],
            [0, 2 * fy / height, 2 * cy / height - 1, 0],
            [
                0,
                0,
                1 if math.isinf(far) else far / (far - near),
                -near if math.isinf(far) else -far * near / (far - near),
            ],
            [0, 0, 1, 0],
        ],
        device="cuda",
    )
    full = (projection @ view).T.contiguous()
    return GaussianRasterizationSettings(
        height,
        width,
        width / (2 * fx),
        height / (2 * fy),
        background,
        1.0,
        view.T.contiguous(),
        full,
        torch.linalg.inv(full).contiguous(),
        0,
        torch.linalg.inv(view)[:3, 3].contiguous(),
        False,
        settings(True, cx == width / 2 and cy == height / 2),
        False,
        False,
    )


@torch.no_grad()
def projection_depth_bounds():
    args = fixture(cameras=2)
    # Non-centered principal points, including one outside the image, and a
    # rotated/translated second camera verify the entire world->clip mapping.
    args[6][0, 0, 2], args[6][0, 1, 2] = 0.0, 48.0
    args[6][1, 0, 2], args[6][1, 1, 2] = -19.0, 73.0
    layout = normalize_inputs(*args[:7], None, None, None, None)
    for near, far in ((0.5, 10.0), (0.01, 100.0), (1.0, math.inf)):
        cameras = prepare_cameras(layout.viewmats, layout.Ks, 64, 48, near, far)
        # Preserve AAA's frontend 0/1 mapping and the effective lower depth
        # bound of its backend (-1,1) test, rather than changing AAA semantics.
        lower = near / 2 if math.isinf(far) else near * far / (2 * far - near)
        for camera in range(2):
            view, k = args[5][camera].double(), args[6][camera].double()
            z = torch.tensor(
                [
                    lower * 0.9,
                    lower,
                    near,
                    1e5 if math.isinf(far) else far,
                    2e5 if math.isinf(far) else far * 1.1,
                ],
                device="cuda",
                dtype=torch.float64,
            )
            points = torch.stack((z * 0.13, z * -0.17, z, torch.ones_like(z)), -1)
            world = points @ torch.linalg.inv(view).T
            clip = world @ cameras.projection[0, camera].double()
            ndc = clip[:, :3] / clip[:, 3:]
            torch.testing.assert_close(
                ndc[1, 2],
                torch.tensor(-1.0, device="cuda", dtype=torch.float64),
                rtol=0,
                atol=4e-6,
            )
            assert ndc[0, 2] < -1 and ndc[2, 2] > -1
            torch.testing.assert_close(
                ndc[2, 2],
                torch.tensor(0.0, device="cuda", dtype=torch.float64),
                rtol=0,
                atol=4e-6,
            )
            if math.isinf(far):
                assert 0.999 < ndc[3, 2] < ndc[4, 2] < 1
            else:
                torch.testing.assert_close(
                    ndc[3, 2],
                    torch.tensor(1.0, device="cuda", dtype=torch.float64),
                    rtol=0,
                    atol=4e-6,
                )
                assert ndc[4, 2] > 1
            pixels = (ndc[:, :2] + 1) * torch.tensor([32.0, 24.0], device="cuda")
            expected = torch.stack(
                (
                    k[0, 0] * points[:, 0] / z + k[0, 2],
                    k[1, 1] * points[:, 1] / z + k[1, 2],
                ),
                -1,
            )
            # Float32 world->clip coefficients lose some precision extremely
            # close to a translated camera (the smallest lower bound is ~.005).
            torch.testing.assert_close(pixels, expected, rtol=2e-6, atol=1e-4)
            torch.testing.assert_close(
                cameras.projection[0, camera] @ cameras.inverse_projection[0, camera],
                torch.eye(4, device="cuda"),
                rtol=0,
                atol=2e-5,
            )
    print(
        "PASS AAA frontend near/far -> NDC 0/1 and backend lower bound, infinite far, full poses and off-center intrinsics",
        flush=True,
    )


@torch.no_grad()
def forward_and_metadata():
    args = fixture()
    backgrounds = torch.randn(2, 7, device="cuda")
    color, alpha, packed = rasterization(
        *args,
        backgrounds=backgrounds,
        render_mode="RGB",
        rasterize_mode="antialiased",
    )
    dense_color, dense_alpha, dense = rasterization(
        *args,
        backgrounds=backgrounds,
        render_mode="RGB",
        packed=False,
        rasterize_mode="antialiased",
    )
    torch.testing.assert_close(color, dense_color, rtol=0, atol=0)
    torch.testing.assert_close(alpha, dense_alpha, rtol=0, atol=0)
    b, c, g = packed["batch_ids"], packed["camera_ids"], packed["gaussian_ids"]
    assert bool((b == 0).all()) and dense["supports_means2d_grad"]
    assert dense["conics"] is None and not dense["means2d"].requires_grad
    for name in ("radii", "means2d", "opacities", "tiles_per_gauss"):
        torch.testing.assert_close(packed[name], dense[name][c, g], rtol=0, atol=0)
        assert bool(torch.isfinite(dense[name]).all())
    render = GaussianRasterizer()
    for camera in range(2):
        m, q, s, o, f, v, k, w, h = args
        config = direct_config(v[camera], k[camera], w, h, backgrounds[camera, :3])
        for first in range(0, 7, 3):
            count = min(3, 7 - first)
            rgb = torch.zeros(len(m), 3, device="cuda")
            bg = torch.zeros(3, device="cuda")
            rgb[:, :count] = f[:, first : first + count]
            bg[:count] = backgrounds[camera, first : first + count]
            expected = render(
                means3D=m,
                rotations=F.normalize(q, dim=-1),
                scales=s,
                opacities=o,
                colors_precomp=rgb,
                raster_settings=config._replace(bg=bg),
            )[0]
            torch.testing.assert_close(
                color[camera, ..., first : first + count]
                .permute(2, 0, 1)
                .contiguous()
                .view(torch.int32),
                expected[:count].contiguous().view(torch.int32),
                rtol=0,
                atol=0,
            )
        expected = render(
            means3D=m,
            rotations=F.normalize(q, dim=-1),
            scales=s,
            opacities=o,
            colors_precomp=torch.ones(len(m), 3, device="cuda"),
            raster_settings=config._replace(bg=torch.zeros(3, device="cuda")),
            return_aux=True,
        )
        torch.testing.assert_close(
            alpha[camera, ..., 0], expected["color"][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            alpha[camera, ..., 0], 1 - expected["final_T"], rtol=0, atol=3e-7
        )
        valid = (dense["radii"][camera] > 0).all(-1)
        projected = (m @ v[camera, :3, :3].T + v[camera, :3, 3]) @ k[camera].T
        projected = projected[:, :2] / projected[:, 2:]
        torch.testing.assert_close(
            dense["means2d"][camera, valid], projected[valid], rtol=1e-6, atol=1e-5
        )
    assert color.shape == (2, 48, 64, 7) and "depths" not in dense
    print(
        "PASS native RGB slice oracle, differentiable-alpha values, packed/dense metadata",
        flush=True,
    )


def batches_and_sh():
    base = fixture(n=5, channels=3)
    args = [
        v.repeat((2, 2) + (1,) * v.ndim) if isinstance(v, torch.Tensor) else v
        for v in base
    ]
    # Batch dimensions [2,2], plus independent colors for both cameras.
    args[4] = args[4].unsqueeze(2).repeat(1, 1, 2, 1, 1)
    args[4][:, :, 1].add_(0.2)
    model = [v.detach().requires_grad_() if i < 5 else v for i, v in enumerate(args)]
    image, alpha, meta = rasterization(*model, render_mode="RGB", packed=True)
    assert image.shape == (2, 2, 2, 48, 64, 3) and alpha.shape == (2, 2, 2, 48, 64, 1)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                single = [v[i, j] if isinstance(v, torch.Tensor) else v for v in args]
                expected, a, _ = rasterization(*single, render_mode="RGB", packed=False)
                torch.testing.assert_close(image[i, j], expected, rtol=0, atol=0)
                torch.testing.assert_close(alpha[i, j], a, rtol=0, atol=0)
    (image.mean() + alpha.mean()).backward()
    assert all(
        v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model[:5]
    )
    assert meta["batch_ids"].max() == 3
    print(
        "PASS arbitrary batch dimensions, per-camera features and combined gradients",
        flush=True,
    )
    for degree in range(4):
        for per_camera in (False, True):
            args = fixture(n=5, channels=3)
            coefficients = (
                torch.randn((2, 5, 20, 3) if per_camera else (5, 20, 3), device="cuda")
                * 0.1
            )
            args[4] = coefficients.requires_grad_()
            args[0].requires_grad_()
            image, alpha, _ = rasterization(*args, sh_degree=degree, packed=False)
            (image.mean() + alpha.mean()).backward()
            assert bool(torch.isfinite(args[0].grad).all()) and bool(
                torch.isfinite(coefficients.grad).all()
            )
            assert float(coefficients.grad.abs().sum()) > 0
            assert not bool(coefficients.grad[..., (degree + 1) ** 2 :, :].any())
    print("PASS shared/per-camera SH0–3 and padded coefficient gradients", flush=True)


def culling_empty_and_signals():
    args = fixture(n=7, channels=3, cameras=1)
    args[0].zero_()
    args[0][:, 2] = torch.tensor(
        [0.45, 0.49, 2.0, 4.0, 10.1, 10.2, -2.0], device="cuda"
    )
    args[1].zero_()
    args[1][:, 0] = 1
    args[2][:] = args[0][:, 2:].abs() * 0.02
    args[2][:, 2] = args[0][:, 2].abs() * 0.1
    # Check AAA's original frontend/backend combination, including visible
    # centers between the effective lower depth bound and nominal near.
    for near, far in ((0.01, 1e10), (0.5, 10), (1.0, math.inf)):
        image, alpha, meta = rasterization(
            *args,
            near_plane=near,
            far_plane=far,
            packed=False,
            rasterize_mode="antialiased",
        )
        m, q, s, o, f, v, k, w, h = args
        config = direct_config(
            v[0], k[0], w, h, torch.zeros(4, device="cuda"), near, far
        )
        expected = GaussianRasterizer(config)(
            means3D=m,
            rotations=F.normalize(q, dim=-1),
            scales=s,
            opacities=o,
            features=torch.cat((f, torch.ones(len(m), 1, device="cuda")), -1),
            return_aux=True,
        )
        torch.testing.assert_close(
            image[0].permute(2, 0, 1), expected["color"][:3], rtol=0, atol=0
        )
        torch.testing.assert_close(
            alpha[0, ..., 0], expected["color"][3], rtol=0, atol=0
        )
        assert torch.equal((meta["radii"][0] > 0).all(-1), expected["radii"] > 0)
        assert not bool(meta["radii"][0, 6].any())
        if near == 0.5:
            assert bool((meta["radii"][0, :2] > 0).all())
            assert not bool(meta["radii"][0, 4:6].any())
    print("PASS AAA culling parity, including centers outside near/far", flush=True)
    clipped, alpha, meta = rasterization(*args, radius_clip=10000, packed=False)
    assert (
        not bool(clipped.any())
        and not bool(alpha.any())
        and not bool(meta["radii"].any())
    )
    leaves = [
        value.detach().requires_grad_() if index < 5 else value
        for index, value in enumerate(args)
    ]
    clipped, alpha, _ = rasterization(*leaves, radius_clip=10000, render_mode="RGB")
    (clipped.sum() + alpha.sum()).backward()
    assert all(
        value.grad is not None and not bool(value.grad.any()) for value in leaves[:5]
    )
    for n in (0, 5):
        args = fixture(n=n, channels=3, cameras=1)
        args[3].zero_()
        backgrounds = torch.tensor([[0.2, 0.3, 0.4]], device="cuda")
        image, alpha, meta = rasterization(*args, backgrounds=backgrounds)
        assert not bool(alpha.any()) and meta["gaussian_ids"].numel() == 0
        torch.testing.assert_close(
            image,
            backgrounds[:, None, None, :].expand_as(image),
            rtol=0,
            atol=0,
        )
    args = fixture(n=7, channels=3)
    extra = torch.randn(2, 7, 4, device="cuda", requires_grad=True)
    image, alpha, meta = rasterization(*args, extra_signals=extra, render_mode="RGB")
    assert meta["render_extra_signals"].shape == (2, 48, 64, 4) and image.shape[-1] == 3
    meta["render_extra_signals"].mean().backward()
    assert extra.grad is not None and float(extra.grad.abs().sum()) > 0
    print(
        "PASS radius clipping, empty/background and extra signal gradients",
        flush=True,
    )


def finite_differences():
    args = fixture(n=1, channels=3, cameras=1)
    args[0][:] = torch.tensor([[0.05, -0.02, 3.0]], device="cuda")
    args[1][:] = torch.tensor([[2.0, 0.2, -0.1, 0.15]], device="cuda")
    args[2][:] = torch.tensor([[0.3, 0.2, 0.25]], device="cuda")
    args[3].fill_(0.5)
    args[6][0, 0, 2], args[6][0, 1, 2] = 32.0, 24.0
    args[:5] = [v.detach().requires_grad_() for v in args[:5]]
    custom = settings(False)
    custom.culling_settings = CullingSettings()
    custom.load_balancing = False
    custom.new_aabb = False
    image, alpha, _ = rasterization(
        *args, render_mode="RGB", aaa_settings=custom, packed=False
    )
    color_weight = torch.zeros_like(image)
    alpha_weight = torch.zeros_like(alpha)
    color_weight[:, 22:26, 30:34] = (
        torch.randn_like(color_weight[:, 22:26, 30:34]) * 0.01
    )
    alpha_weight[:, 22:26, 30:34] = (
        torch.randn_like(alpha_weight[:, 22:26, 30:34]) * 0.01
    )
    (image * color_weight).sum().add((alpha * alpha_weight).sum()).backward()
    with torch.no_grad():
        for field, index in (
            (0, (0, 0)),
            (1, (0, 1)),
            (2, (0, 0)),
            (3, (0,)),
            (4, (0, 2)),
        ):
            value = args[field]
            original = float(value[index])
            epsilon = 1e-3
            losses = []
            for delta in (epsilon, -epsilon):
                value[index] = original + delta
                image, alpha, _ = rasterization(
                    *args, render_mode="RGB", aaa_settings=custom, packed=False
                )
                losses.append(
                    float(
                        (image.double() * color_weight.double()).sum()
                        + (alpha.double() * alpha_weight.double()).sum()
                    )
                )
            value[index] = original
            numeric = (losses[0] - losses[1]) / (2 * epsilon)
            analytic = float(value.grad[index])
            assert abs(numeric - analytic) < 2e-5 + 1e-2 * abs(analytic), (
                field,
                numeric,
                analytic,
            )
    # Alpha-only loss must produce a real opacity gradient, not a detached map.
    for value in args[:5]:
        value.grad = None
    _, alpha, _ = rasterization(*args, packed=False, aaa_settings=custom)
    alpha.sum().backward()
    assert float(args[3].grad.abs().sum()) > 0
    print(
        "PASS finite differences for means/quats/scales/opacities/colors and alpha-only loss",
        flush=True,
    )


def guards_and_standalone():
    args = fixture()
    for options in (
        dict(sparse_grad=True),
        dict(distributed=True),
        dict(with_eval3d=False),
        dict(camera_model="ortho"),
        dict(eps2d=0.5),
        dict(tile_size=8),
        dict(with_ut=True),
        dict(segmented=True),
        dict(radial_coeffs=torch.zeros(2, 6, device="cuda")),
        dict(rolling_shutter="rolling"),
        dict(render_mode="RGB-Ed"),
        *(dict(render_mode=mode) for mode in ("D", "ED", "RGB+D", "RGB+ED")),
    ):
        try:
            rasterization(*args, **options)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(options)
    for index in (6,):
        changed = list(args)
        changed[index] = changed[index].detach().requires_grad_()
        try:
            rasterization(*changed)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("intrinsic gradients accepted")
    with torch.no_grad():
        rasterization(*changed)
    for options in (
        dict(width=0),
        dict(sh_degree=4),
        dict(near_plane=1, far_plane=0.1),
    ):
        values = dict(
            zip(
                (
                    "means",
                    "quats",
                    "scales",
                    "opacities",
                    "colors",
                    "viewmats",
                    "Ks",
                    "width",
                    "height",
                ),
                args,
            )
        )
        values.update(options)
        try:
            rasterization(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(options)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name == 'gsplat' or name.startswith('gsplat.') or name.startswith(('diff_gaussian_rasterization', 'aaa_gaussian_rasterization')):
            raise ImportError('compatibility layer must be standalone')
sys.meta_path.insert(0,Reject())
from dsl_splat.aaa.gsplat import rasterization
from dsl_splat import rasterization as exported
assert exported is rasterization
print('PASS standalone gsplat-compatible import')
""",
        ],
        check=True,
    )
    print("PASS unsupported/invalid parameter guards", flush=True)


def main():
    projection_depth_bounds()
    forward_and_metadata()
    batches_and_sh()
    culling_empty_and_signals()
    finite_differences()
    guards_and_standalone()
    with torch.cuda.stream(torch.cuda.Stream()):
        args = fixture(n=3, channels=4, cameras=1)
        args[:5] = [v.detach().requires_grad_() for v in args[:5]]
        color, alpha, _ = rasterization(*args, render_mode="RGB")
        (color.mean() + alpha.mean()).backward()
        assert all(bool(torch.isfinite(v.grad).all()) for v in args[:5])
    torch.cuda.synchronize()
    print("PASS gsplat interface checks including non-default stream", flush=True)


if __name__ == "__main__":
    main()
