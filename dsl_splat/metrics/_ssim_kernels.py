"""SSIM's separable Gaussian filter, fused within each shared-memory tile.

Adapted from Rahul Goel's fused-ssim (MIT, 2024); see FUSED_SSIM_LICENSE.txt.
All GPU code uses Warp, including shared storage and block synchronization.
"""

from functools import lru_cache

import warp as wp

_Moments = wp.types.vector(5, wp.float32)
_WEIGHTS = wp.constant(
    wp.types.vector(6, wp.float32)(
        0.26601171493530273,
        0.21300552785396576,
        0.10936068743467331,
        0.036000773310661316,
        0.0075987582094967365,
        0.001028380123898387,
    )
)


@wp.func
def _moments(pixel: wp.vec2) -> _Moments:
    x, y = pixel[0], pixel[1]
    return _Moments(x, x * x, y, y * y, x * y)


@lru_cache(maxsize=8)
def forward_kernel(need_x: bool, need_y: bool) -> wp.Kernel:
    @wp.kernel(module="unique", enable_backward=False)
    def forward(
        x: wp.array4d(dtype=float),
        y: wp.array4d(dtype=float),
        channels: int,
        crop: int,
        c1: float,
        c2: float,
        output: wp.array4d(dtype=float),
        partial_x: wp.array3d(dtype=wp.vec3),
        partial_y: wp.array3d(dtype=wp.vec3),
    ):
        plane, by, bx, lane = wp.tid()
        batch, channel = plane // channels, plane % channels
        row, col = lane // 16, lane % 16
        yy, xx = by * 16 + row, bx * 16 + col
        # Warp 1.17 tile_load checks upper bounds but accepts negative offsets.
        # Explicit lower/upper checks implement zero padding safely on all sides.
        source = wp.tile_empty(shape=(26, 26), dtype=wp.vec2, storage="shared")
        for load_step in range(3):
            index = lane + load_step * 256
            r, c = index // 26, index % 26
            gy, gx = by * 16 + r - 5, bx * 16 + c - 5
            pixel = wp.vec2(0.0)
            if r < 26 and gy >= 0 and gx >= 0 and gy < x.shape[2] and gx < x.shape[3]:
                pixel = wp.vec2(x[batch, channel, gy, gx], y[batch, channel, gy, gx])
            wp.tile_scatter_masked(source, r, c, pixel, r < 26)
        horizontal = wp.tile_empty(shape=(26, 16), dtype=_Moments, storage="shared")
        for step in range(2):
            r = row + step * 16
            accum = _Moments(0.0)
            if r < 26:
                for d in range(1, 6):
                    accum += (
                        _moments(source[r, col + 5 - d])
                        + _moments(source[r, col + 5 + d])
                    ) * _WEIGHTS[d]
                accum += _moments(source[r, col + 5]) * _WEIGHTS[0]
            # Every thread participates; the scatter includes a block barrier.
            wp.tile_scatter_masked(horizontal, r, col, accum, r < 26)

        filtered = _Moments(0.0)
        for d in range(1, 6):
            filtered += (
                horizontal[row + 5 - d, col] + horizontal[row + 5 + d, col]
            ) * _WEIGHTS[d]
        filtered += horizontal[row + 5, col] * _WEIGHTS[0]

        oy, ox = yy - crop, xx - crop
        if oy >= 0 and ox >= 0 and oy < output.shape[2] and ox < output.shape[3]:
            mx, my = filtered[0], filtered[2]
            vx, vy = filtered[1] - mx * mx, filtered[3] - my * my
            a = mx * mx + my * my + c1
            b = wp.max(vx, 0.0) + wp.max(vy, 0.0) + c2
            luminance = 2.0 * mx * my + c1
            covariance = 2.0 * (filtered[4] - mx * my) + c2
            value = luminance * covariance / (a * b)
            output[batch, channel, oy, ox] = wp.clamp(value, -1.0, 1.0)
            if wp.static(need_x or need_y):
                # Derivatives are with respect to E[X], E[X^2], E[XY].
                # Honor the variance and final SSIM clamps in the forward.
                dvx, dvy, dxy = float(0.0), float(0.0), float(0.0)
                dmx, dmy = float(0.0), float(0.0)
                if value >= -1.0 and value <= 1.0:
                    dxy = 2.0 * luminance / (a * b)
                    if vx >= 0.0:
                        dvx = -value / b
                    if vy >= 0.0:
                        dvy = -value / b
                    dmx = (
                        2.0 * my * covariance / (a * b)
                        - 2.0 * mx * value / a
                        - 2.0 * mx * dvx
                        - my * dxy
                    )
                    dmy = (
                        2.0 * mx * covariance / (a * b)
                        - 2.0 * my * value / a
                        - 2.0 * my * dvy
                        - mx * dxy
                    )
                if wp.static(need_x):
                    partial_x[plane, oy, ox] = wp.vec3(dmx, dvx, dxy)
                if wp.static(need_y):
                    partial_y[plane, oy, ox] = wp.vec3(dmy, dvy, dxy)

    return forward


@lru_cache(maxsize=8)
def backward_kernel(need_x: bool, need_y: bool) -> wp.Kernel:
    @wp.kernel(module="unique", enable_backward=False)
    def backward(
        x: wp.array4d(dtype=float),
        y: wp.array4d(dtype=float),
        upstream: wp.array4d(dtype=float),
        partial_x: wp.array3d(dtype=wp.vec3),
        partial_y: wp.array3d(dtype=wp.vec3),
        channels: int,
        crop: int,
        dx: wp.array4d(dtype=float),
        dy: wp.array4d(dtype=float),
    ):
        plane, by, bx, lane = wp.tid()
        batch, channel = plane // channels, plane % channels
        row, col = lane // 16, lane % 16
        yy, xx = by * 16 + row, bx * 16 + col
        for side in range(wp.static(int(need_x) + int(need_y))):
            is_x = wp.static(need_x) and side == 0
            partial = partial_y
            if is_x:
                partial = partial_x
            shared = wp.tile_empty(shape=(26, 26), dtype=wp.vec3, storage="shared")
            for load_step in range(3):
                index = lane + load_step * 256
                r, c = index // 26, index % 26
                gy, gx = by * 16 + r - 5 - crop, bx * 16 + c - 5 - crop
                value = wp.vec3(0.0)
                if (
                    r < 26
                    and gy >= 0
                    and gx >= 0
                    and gy < upstream.shape[2]
                    and gx < upstream.shape[3]
                ):
                    value = partial[plane, gy, gx] * upstream[batch, channel, gy, gx]
                wp.tile_scatter_masked(shared, r, c, value, r < 26)
            horizontal = wp.tile_empty(shape=(26, 16), dtype=wp.vec3, storage="shared")
            for step in range(2):
                r = row + step * 16
                accum = wp.vec3(0.0)
                if r < 26:
                    for d in range(1, 6):
                        accum += (
                            shared[r, col + 5 - d] + shared[r, col + 5 + d]
                        ) * _WEIGHTS[d]
                    accum += shared[r, col + 5] * _WEIGHTS[0]
                wp.tile_scatter_masked(horizontal, r, col, accum, r < 26)
            filtered = wp.vec3(0.0)
            for d in range(1, 6):
                filtered += (
                    horizontal[row + 5 - d, col] + horizontal[row + 5 + d, col]
                ) * _WEIGHTS[d]
            filtered += horizontal[row + 5, col] * _WEIGHTS[0]
            if yy < x.shape[2] and xx < x.shape[3]:
                # Materialize values: Warp unrolls the two gradient sides, and
                # rebinding an array reference across unrolled bodies is invalid.
                px = x[batch, channel, yy, xx] + 0.0
                py = y[batch, channel, yy, xx] + 0.0
                if is_x:
                    dx[batch, channel, yy, xx] = (
                        filtered[0] + 2.0 * px * filtered[1] + py * filtered[2]
                    )
                else:
                    dy[batch, channel, yy, xx] = (
                        filtered[0] + 2.0 * py * filtered[1] + px * filtered[2]
                    )

    return backward
