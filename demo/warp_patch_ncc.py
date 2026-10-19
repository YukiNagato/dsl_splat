"""Warp port of RaDe-GS's plane-warped patch NCC, for learning and comparison.

One thread handles one patch and computes NCC plus its depth/normal derivatives
in the same traversal, as in the CUDA reference. Images are grayscale float32.
``R`` follows the reference's column-storage convention: pass ``rotation.T``
for the mathematical rotation from the reference camera to the neighbor.
"""

from functools import lru_cache
import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from dsl_splat.runtime.interop import current_stream


@wp.func_native("return __float2int_rd(x);")
def float2int_rd(x: wp.float32) -> wp.int32: ...


@wp.func_native("return __float2int_ru(x);")
def float2int_ru(x: wp.float32) -> wp.int32: ...


@wp.func_native("return nextafterf(x, INFINITY);")
def nextafter_pos_inf(x: wp.float32) -> wp.float32: ...


@lru_cache(maxsize=None, typed=True)
def build_warp_patch_ncc_kernel(RADIUS: int, HALF_STEP: bool) -> wp.Kernel:
    """Specialize patch size/step once; shape and point count remain dynamic."""
    if isinstance(RADIUS, bool) or not isinstance(RADIUS, int) or RADIUS <= 0:
        raise ValueError("RADIUS must be a positive integer")
    if not isinstance(HALF_STEP, bool):
        raise TypeError("HALF_STEP must be bool")

    # Compute specialization constants in Python, not from runtime kernel locals.
    TOTAL_SIZE_INV = 1.0 / ((2 * RADIUS + 1) ** 2)
    RADIUS_F = float(RADIUS) * (0.5 if HALF_STEP else 1.0)
    RADIUS_SCALED = (RADIUS + 1) // 2
    LINE_LENGTH_SCALED = 2 * RADIUS_SCALED + 1

    @wp.kernel(module="unique", enable_backward=False)
    def warp_patch_ncc(
        P: int,
        depths: wp.array(dtype=wp.float32),
        normals: wp.array(dtype=wp.vec3),
        uvs: wp.array(dtype=wp.vec2i),
        R: wp.array(dtype=wp.float32),
        T: wp.array(dtype=wp.float32),
        image_r: wp.array(dtype=wp.float32),
        image_n: wp.array(dtype=wp.float32),
        fx_r: float,
        fy_r: float,
        cx_r: float,
        cy_r: float,
        fx_n: float,
        fy_n: float,
        cx_n: float,
        cy_n: float,
        image_height_r: int,
        image_width_r: int,
        image_height_n: int,
        image_width_n: int,
        ncc: wp.array(dtype=wp.float32),
        grad_depths: wp.array(dtype=wp.float32),
        grad_normals: wp.array(dtype=wp.vec3),
        valid: wp.array(dtype=wp.bool),
    ):
        idx = wp.tid()
        if idx >= P:
            return

        # Every point owns all its outputs, including paths that return early.
        ncc[idx] = 0.0
        grad_depths[idx] = 0.0
        grad_normals[idx] = wp.vec3(0.0)
        valid[idx] = False

        uv_center_r = uvs[idx]
        uc, vc = float(uv_center_r[0]), float(uv_center_r[1])
        depth = depths[idx]
        normal = normals[idx]
        point_n_r = wp.vec3(
            (float(uv_center_r[0]) - cx_r) / fx_r,
            (float(uv_center_r[1]) - cy_r) / fy_r,
            1.0,
        )
        distance = -wp.dot(point_n_r, normal) * depth

        all_inside = (
            uc - RADIUS_F > 0.0
            and uc + RADIUS_F < float(image_width_r - 1)
            and vc - RADIUS_F > 0.0
            and vc + RADIUS_F < float(image_height_r - 1)
        )

        if not all_inside or not wp.isfinite(distance) or distance == 0.0:
            return

        # Match CUDA float33 (three columns). The caller supplies rotation.T.
        # With a static shape, wp.zeros inside a kernel is per-thread local
        # storage, not a host-side allocation or a new Torch GPU tensor.
        R_mat = wp.zeros(shape=3, dtype=wp.vec3)
        R_mat[0] = wp.vec3(R[0], R[1], R[2])
        R_mat[1] = wp.vec3(R[3], R[4], R[5])
        R_mat[2] = wp.vec3(R[6], R[7], R[8])

        T_mat = wp.vec3(T[0], T[1], T[2])
        Hn = wp.zeros(shape=3, dtype=wp.vec3)
        Hn[0] = R_mat[0] - normal[0] / distance * T_mat
        Hn[1] = R_mat[1] - normal[1] / distance * T_mat
        Hn[2] = R_mat[2] - normal[2] / distance * T_mat

        H = wp.zeros(shape=3, dtype=wp.vec3)
        for i in range(wp.static(3)):
            H[i] = wp.vec3(
                fx_n * Hn[i][0] + cx_n * Hn[i][2],
                fy_n * Hn[i][1] + cy_n * Hn[i][2],
                Hn[i][2],
            )
        H[2] = H[0] * (-cx_r / fx_r) + H[1] * (-cy_r / fy_r) + H[2]
        H[0] = H[0] / fx_r
        H[1] = H[1] / fy_r

        H_uc = H[0] * float(uv_center_r[0]) + H[1] * float(uv_center_r[1]) + H[2]
        cached_line = wp.zeros(shape=wp.static(LINE_LENGTH_SCALED), dtype=wp.float32)

        if wp.static(HALF_STEP):
            v = uv_center_r.y - RADIUS_SCALED
            for i in range(LINE_LENGTH_SCALED):
                cached_line[i] = image_r[
                    v * image_width_r + uv_center_r.x - RADIUS_SCALED + i
                ]

        # float(...) makes mutable Warp locals, rather than compile-time literals.
        sum_c_r = float(0.0)
        sum_c_n = float(0.0)
        sum_c_r2 = float(0.0)
        sum_c_n2 = float(0.0)
        sum_c_r_c_n = float(0.0)

        grad_n = wp.vec3(0.0, 0.0, 0.0)
        grad_n2 = wp.vec3(0.0, 0.0, 0.0)
        grad_rn = wp.vec3(0.0, 0.0, 0.0)

        aux = T_mat / distance

        for dv in range(-wp.static(RADIUS), wp.static(RADIUS) + 1):
            dv_f = float(dv)
            if wp.static(HALF_STEP):
                dv_f = 0.5 * float(dv)
            is_odd_v = dv % 2 != 0

            # The previous reference row lives in cached_line; only fetch v1r.
            v1r = uv_center_r[1] + (float2int_ru(dv_f) if is_odd_v else dv / 2)

            last_c = float(0.0)

            if wp.static(HALF_STEP):
                if is_odd_v:
                    last_c = image_r[
                        v1r * image_width_r + uv_center_r[0] - RADIUS_SCALED
                    ]

            w_v0 = 0.5 if is_odd_v else 1.0
            w_v1 = 0.5 if is_odd_v else 0.0

            H_uc_v = H_uc + dv_f * H[1]

            for du in range(-wp.static(RADIUS), wp.static(RADIUS) + 1):
                du_f = float(du)
                if wp.static(HALF_STEP):
                    du_f = 0.5 * float(du)
                uv_r = wp.vec2(uc + du_f, vc + dv_f)
                if wp.static(HALF_STEP):
                    is_odd_u = du % 2 != 0
                    w_u0 = 0.5 if is_odd_u else 1.0
                    w_u1 = 0.5 if is_odd_u else 0.0

                    u0r = float2int_rd(du_f) if is_odd_u else du / 2
                    u1r = float2int_ru(du_f) if is_odd_u else du / 2

                    c00 = cached_line[u0r + RADIUS_SCALED]
                    c01 = cached_line[u1r + RADIUS_SCALED]
                    c10 = last_c
                    c11 = float(0.0)
                    if is_odd_v and is_odd_u:
                        c11 = image_r[v1r * image_width_r + uv_center_r[0] + u1r]
                        cached_line[u0r + RADIUS_SCALED] = last_c
                        last_c = c11

                    c_r = (c00 * w_u0 + c01 * w_u1) * w_v0 + (
                        c10 * w_u0 + c11 * w_u1
                    ) * w_v1
                else:
                    c_r = image_r[
                        (dv + uv_center_r[1]) * image_width_r + du + uv_center_r[0]
                    ]

                H_uv = H_uc_v + du_f * H[0]
                if (
                    H_uv[2] == 0.0
                    or not wp.isfinite(H_uv[0])
                    or not wp.isfinite(H_uv[1])
                    or not wp.isfinite(H_uv[2])
                ):
                    return
                uv_n = wp.vec2(H_uv[0] / H_uv[2], H_uv[1] / H_uv[2])
                if not wp.isfinite(uv_n[0]) or not wp.isfinite(uv_n[1]):
                    return
                # This deliberately retains CUDA's per-sample patch margin.
                inside = (
                    uv_n[0] - RADIUS_F > 0.0
                    and uv_n[0] + RADIUS_F < float(image_width_n - 1)
                    and uv_n[1] - RADIUS_F > 0.0
                    and uv_n[1] + RADIUS_F < float(image_height_n - 1)
                )
                all_inside = inside and all_inside

                # At integer pixels, select the next corner too: floor==ceil
                # would otherwise collapse the interpolation and its derivative.
                eps_u = nextafter_pos_inf(uv_n[0])
                eps_v = nextafter_pos_inf(uv_n[1])

                u0 = float2int_rd(uv_n[0])
                v0 = float2int_rd(uv_n[1])
                u1 = float2int_ru(eps_u)
                v1 = float2int_ru(eps_v)

                u0 = wp.clamp(u0, 0, image_width_n - 1)
                u1 = wp.clamp(u1, 0, image_width_n - 1)
                v0 = wp.clamp(v0, 0, image_height_n - 1)
                v1 = wp.clamp(v1, 0, image_height_n - 1)

                c00n = image_n[v0 * image_width_n + u0]
                c01n = image_n[v0 * image_width_n + u1]
                c10n = image_n[v1 * image_width_n + u0]
                c11n = image_n[v1 * image_width_n + u1]

                wv0 = float(v1) - uv_n[1]
                wv1 = uv_n[1] - float(v0)
                wu0 = float(u1) - uv_n[0]
                wu1 = uv_n[0] - float(u0)
                c_n = wv0 * (wu0 * c00n + wu1 * c01n) + wv1 * (wu0 * c10n + wu1 * c11n)

                sum_c_r += c_r
                sum_c_n += c_n
                sum_c_r2 += c_r * c_r
                sum_c_n2 += c_n * c_n
                sum_c_r_c_n += c_r * c_n

                dc_n_duv = wp.vec2(
                    -c00n * wv0 + c01n * wv0 - c10n * wv1 + c11n * wv1,
                    -c00n * wu0 - c01n * wu1 + c10n * wu0 + c11n * wu1,
                )

                dc_n_dH_uv = wp.vec3(
                    dc_n_duv[0] / H_uv[2],
                    dc_n_duv[1] / H_uv[2],
                    (-dc_n_duv[0] * uv_n[0] - dc_n_duv[1] * uv_n[1]) / H_uv[2],
                )

                dc_n_dHn_left = wp.vec3(
                    dc_n_dH_uv[0] * fx_n,
                    dc_n_dH_uv[1] * fy_n,
                    dc_n_dH_uv[0] * cx_n + dc_n_dH_uv[1] * cy_n + dc_n_dH_uv[2],
                )

                dc_n_dHn_right = wp.vec3(
                    (uv_r[0] - cx_r) / fx_r, (uv_r[1] - cy_r) / fy_r, 1.0
                )

                grad_aux = dc_n_dHn_right * wp.dot(dc_n_dHn_left, aux)
                grad_n += grad_aux
                grad_n2 += 2.0 * c_n * grad_aux
                grad_rn += c_r * grad_aux

            if wp.static(HALF_STEP):
                if is_odd_v:
                    cached_line[2 * RADIUS_SCALED] = last_c

        cross = sum_c_r_c_n - sum_c_r * sum_c_n * TOTAL_SIZE_INV
        variance_r = sum_c_r2 - sum_c_r * sum_c_r * TOTAL_SIZE_INV
        variance_n = sum_c_n2 - sum_c_n * sum_c_n * TOTAL_SIZE_INV
        output_ncc = cross * cross / (variance_r * variance_n + 1e-8)

        # Retain CUDA's epsilon/derivative convention, including its approximate
        # stabilized variance derivative, for matching training behavior.
        grad_cross = 2.0 * cross / (variance_r * variance_n + 1e-8)
        grad_variance_n = -output_ncc / (variance_n + 1e-8)

        grad_sum_c_n = (
            -grad_cross * sum_c_r - grad_variance_n * 2.0 * sum_c_n
        ) * TOTAL_SIZE_INV
        grad_sum_c_n2 = grad_variance_n
        grad_sum_c_r_c_n = grad_cross

        grad_aux = (
            grad_sum_c_n * grad_n + grad_sum_c_n2 * grad_n2 + grad_sum_c_r_c_n * grad_rn
        )
        grad_normal = -grad_aux
        grad_distance = wp.dot(grad_aux, normal) / distance

        grad_normal += -depth * grad_distance * point_n_r
        grad_depth = -wp.dot(point_n_r, normal) * grad_distance

        valid_patch = all_inside and variance_r > 5e-6 and variance_n > 5e-6

        ncc[idx] = output_ncc if valid_patch else 0.0
        grad_depths[idx] = grad_depth if valid_patch else 0.0
        grad_normals[idx] = grad_normal if valid_patch else wp.vec3(0.0, 0.0, 0.0)
        valid[idx] = valid_patch

    return warp_patch_ncc


def _forward(
    depths: Tensor,
    normals: Tensor,
    uvs: Tensor,
    R: Tensor,
    T: Tensor,
    image_r: Tensor,
    image_n: Tensor,
    intrinsics: tuple[float, ...],
    radius: int,
    half_step: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Launch the forward-mode kernel; the derivatives are computed here too."""
    wp.init()
    kernel = build_warp_patch_ncc_kernel(radius, half_step)
    ncc = torch.empty_like(depths)
    grad_depths = torch.empty_like(depths)
    grad_normals = torch.empty_like(normals)
    valid = torch.empty(depths.shape, dtype=torch.bool, device=depths.device)
    count = depths.numel()
    if count:
        height_r, width_r = image_r.shape
        height_n, width_n = image_n.shape
        stream = current_stream(depths.device)
        # from_torch creates zero-copy views; Torch owns every allocation.
        wp.launch(
            kernel,
            dim=count,
            inputs=[
                count,
                wp.from_torch(depths, dtype=wp.float32, requires_grad=False),
                wp.from_torch(normals, dtype=wp.vec3, requires_grad=False),
                wp.from_torch(uvs, dtype=wp.vec2i, requires_grad=False),
                wp.from_torch(R.reshape(-1), requires_grad=False),
                wp.from_torch(T, requires_grad=False),
                wp.from_torch(image_r.reshape(-1), requires_grad=False),
                wp.from_torch(image_n.reshape(-1), requires_grad=False),
                *intrinsics,
                height_r,
                width_r,
                height_n,
                width_n,
            ],
            outputs=[
                wp.from_torch(ncc, requires_grad=False),
                wp.from_torch(grad_depths, requires_grad=False),
                wp.from_torch(grad_normals, dtype=wp.vec3, requires_grad=False),
                wp.from_torch(valid, requires_grad=False),
            ],
            device=wp.device_from_torch(depths.device),
            stream=stream,
        )
        if debug:
            wp.synchronize_stream(stream)
    return ncc, grad_depths, grad_normals, valid


class _WarpPatchNCC(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        depths,
        normals,
        uvs,
        R,
        T,
        image_r,
        image_n,
        intrinsics,
        radius,
        half_step,
        debug,
    ):
        ncc, grad_depths, grad_normals, valid = _forward(
            depths,
            normals,
            uvs,
            R,
            T,
            image_r,
            image_n,
            intrinsics,
            radius,
            half_step,
            debug,
        )
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(valid)
        ctx.save_for_backward(grad_depths, grad_normals)
        return ncc, valid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_ncc, grad_valid):
        if grad_ncc is None:
            return (None,) * 11
        grad_depths, grad_normals = ctx.saved_tensors
        depth_gradient = grad_ncc * grad_depths if ctx.needs_input_grad[0] else None
        normal_gradient = (
            grad_ncc[:, None] * grad_normals if ctx.needs_input_grad[1] else None
        )
        return depth_gradient, normal_gradient, *(None,) * 9


def warp_patch_ncc(
    depths: Tensor,
    normals: Tensor,
    uvs: Tensor,
    R: Tensor,
    T: Tensor,
    image_r: Tensor,
    image_n: Tensor,
    fx_r: float,
    fy_r: float,
    cx_r: float,
    cy_r: float,
    fx_n: float,
    fy_n: float,
    cx_n: float,
    cy_n: float,
    debug: bool = False,
    *,
    radius: int = 3,
    half_step: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return squared NCC (P,) and a valid mask (P,), matching RaDe-GS.

    Inputs are CUDA float32 on one device: depths (P,), normals (P,3),
    R (3,3), T (3,), and two grayscale images (H,W). Pixel centers uvs (P,2)
    are int32 in (u,v) order. Depth is camera Z, and normals are expressed in
    the reference camera frame. R stores columns: pass rotation.T for the
    reference-to-neighbor rotation; T is that transform's translation.

    Only depths and normals have first-order gradients, as in CUDA. Images,
    poses and intrinsics are treated as constants. Forward computes the local
    derivatives; backward scales them by the incoming loss gradient.
    Defaults select CUDA's 7x7 patch sampled at half-pixel intervals.
    """
    if not isinstance(depths, Tensor) or not depths.is_cuda:
        raise ValueError("depths must be a CUDA float32 tensor")
    if depths.ndim != 1:
        raise ValueError("depths must have shape (P,)")
    count, device = depths.numel(), depths.device
    tensors = (
        ("depths", depths, (count,), torch.float32),
        ("normals", normals, (count, 3), torch.float32),
        ("uvs", uvs, (count, 2), torch.int32),
        ("R", R, (3, 3), torch.float32),
        ("T", T, (3,), torch.float32),
    )
    for name, value, shape, dtype in tensors:
        if (
            not isinstance(value, Tensor)
            or value.device != device
            or value.dtype != dtype
            or tuple(value.shape) != shape
        ):
            raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    for name, image in (("image_r", image_r), ("image_n", image_n)):
        if (
            not isinstance(image, Tensor)
            or image.device != device
            or image.dtype != torch.float32
            or image.ndim != 2
            or min(image.shape) == 0
        ):
            raise ValueError(
                f"{name} must be a nonempty float32 (H,W) image on {device}"
            )
    intrinsics = tuple(
        float(v) for v in (fx_r, fy_r, cx_r, cy_r, fx_n, fy_n, cx_n, cy_n)
    )
    if (
        not all(math.isfinite(v) for v in intrinsics)
        or min(intrinsics[0], intrinsics[1], intrinsics[4], intrinsics[5]) <= 0
    ):
        raise ValueError("intrinsics must be finite, with positive focal lengths")
    return _WarpPatchNCC.apply(
        depths.contiguous(),
        normals.contiguous(),
        uvs.contiguous(),
        R.contiguous(),
        T.contiguous(),
        image_r.contiguous(),
        image_n.contiguous(),
        intrinsics,
        radius,
        half_step,
        debug,
    )
