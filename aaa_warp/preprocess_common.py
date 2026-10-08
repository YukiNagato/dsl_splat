"""Shared preprocessing: tile culling, SH, and ray-depth helper data."""

import warp as wp
from .geometry import rotation_matrix, frustum_minimum, tile_minimum_2d
from .sh import evaluate_sh


@wp.kernel(enable_backward=False)
def finish_preprocess(
    means: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    camera: wp.array(dtype=wp.vec3),
    sh: wp.array2d(dtype=wp.vec3),
    use_sh: bool,
    degree: int,
    modifier: float,
    sort_order: int,
    need_inverse: bool,
    eval3d: bool,
    tile_culling: bool,
    width: int,
    height: int,
    centers: wp.array(dtype=wp.vec2),
    rects: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec4),
    transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float),
    cov3d: wp.array2d(dtype=float),
    depths: wp.array(dtype=float),
    rgb: wp.array(dtype=wp.vec3),
    clamped: wp.array2d(dtype=wp.bool),
    inverse: wp.array3d(dtype=float),
    radii: wp.array(dtype=int),
    tiles: wp.array(dtype=int),
    valid: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    if radii[i] != 0 and tile_culling:
        p, e = centers[i], rects[i]
        gx, gy = (width + 15) // 16, (height + 15) // 16
        xmin = wp.clamp(int(wp.floor((p[0] - e[0]) / 16.0)), 0, gx)
        ymin = wp.clamp(int(wp.floor((p[1] - e[1]) / 16.0)), 0, gy)
        xmax = wp.clamp(int(wp.ceil((p[0] + e[0]) / 16.0)), 0, gx)
        ymax = wp.clamp(int(wp.ceil((p[1] + e[1]) / 16.0)), 0, gy)
        opacity = float(0.0)
        if eval3d:
            opacity = opacity3d[i]
        else:
            opacity = conics[i][3]
        threshold = wp.log(opacity / (1.0 / 255.0))
        count = int(0)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                lo = wp.vec2(float(x * 16), float(y * 16))
                hi = lo + wp.vec2(15.0)
                power = float(0.0)
                if eval3d:
                    power = frustum_minimum(lo, hi, transforms[i])
                else:
                    power = tile_minimum_2d(conics[i], p, lo, hi)
                if power <= threshold:
                    count += 1
        tiles[i] = count
        if count == 0:
            radii[i] = 0
    if radii[i] == 0:
        valid[i] = False
        tiles[i] = 0
        return
    valid[i] = True
    if sort_order != 0:
        depths[i] = wp.length(camera[0] - means[i])
    if use_sh:
        color = evaluate_sh(i, degree, means[i], camera[0], sh)
        for c in range(3):
            clamped[i, c] = color[c] < 0.0
        rgb[i] = wp.vec3(
            wp.max(color[0], 0.0), wp.max(color[1], 0.0), wp.max(color[2], 0.0)
        )
    # Precomputed features alias their Torch input in the host wrapper.
    if need_inverse and not eval3d:
        scale = scales[i]
        s = wp.vec3(
            1.0 / (modifier * wp.max(1.0e-3, scale[0])),
            1.0 / (modifier * wp.max(1.0e-3, scale[1])),
            1.0 / (modifier * wp.max(1.0e-3, scale[2])),
        )
        m = wp.mat33(s[0], 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, s[2]) * rotation_matrix(
            rotations[i]
        )
        cov = wp.transpose(m) * m
        upper = -cov * (camera[0] - means[i])
        inverse[i, 0, 0] = cov[0, 0]
        inverse[i, 0, 1] = cov[0, 1]
        inverse[i, 0, 2] = cov[0, 2]
        inverse[i, 1, 0] = cov[1, 1]
        inverse[i, 1, 1] = cov[1, 2]
        inverse[i, 1, 2] = cov[2, 2]
        inverse[i, 2, 0] = upper[0]
        inverse[i, 2, 1] = upper[1]
        inverse[i, 2, 2] = upper[2]
        inverse[i, 0, 3] = 0.0
        inverse[i, 1, 3] = 0.0
        inverse[i, 2, 3] = 0.0
