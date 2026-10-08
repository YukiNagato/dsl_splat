"""Forward 3D hierarchical ray sorting from AAA / StopThePop.

The reference algorithm is Copyright (c) 2024 Graz University of Technology,
MIT-licensed; see AAA-Gaussians-Rasterization/cuda_rasterizer/stopthepop/LICENSE.txt.
The CUDA implementation shares 4x4 and 2x2 queues between lanes. This first
Warp implementation keeps the same queue capacities per pixel, allowing the
streaming 64/8/4 ordering and blending to be checked independently. It is a
forward diagnostic path and intentionally does not register a backward pass.
"""

from .settings import GaussianRasterizationSettings
from .types import PreprocessedGaussians, TileBins, RenderOutput
import torch
import warp as wp

from .geometry import frustum_minimum


@wp.func
def _sample(g2s: wp.mat44, x: float, y: float):
    plane_x = g2s[0] - g2s[3] * x
    plane_y = g2s[1] - g2s[3] * y
    # Preserve the arithmetic order of consistent_common.cuh::max_contrib_ray.
    d = wp.vec3(
        plane_x[1] * plane_y[2] - plane_x[2] * plane_y[1],
        plane_x[2] * plane_y[0] - plane_x[0] * plane_y[2],
        plane_x[0] * plane_y[1] - plane_x[1] * plane_y[0],
    )
    m = wp.vec3(
        plane_x[3] * plane_y[0] - plane_x[0] * plane_y[3],
        plane_x[3] * plane_y[1] - plane_x[1] * plane_y[3],
        plane_x[3] * plane_y[2] - plane_x[2] * plane_y[3],
    )
    m_div_dd = m / wp.dot(d, d)
    p = wp.cross(d, m_div_dd)
    pos = wp.vec4(p[0], p[1], p[2], 1.0)
    depth = wp.dot(g2s[2], pos) * (1.0 / wp.dot(g2s[3], pos))
    return -0.5 * wp.dot(m, m_div_dd), depth


@wp.func
def _blend_first(
    p: int,
    head_ids: wp.array2d(dtype=wp.int32),
    head_alphas: wp.array2d(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    state: wp.vec4,
):
    alpha = head_alphas[0, p]
    transmittance = state[3] * (1.0 - alpha)
    active = transmittance >= 0.0001
    if active:
        rgb = colors[head_ids[0, p]]
        weight = alpha * state[3]
        state = wp.vec4(
            state[0] + rgb[0] * weight,
            state[1] + rgb[1] * weight,
            state[2] + rgb[2] * weight,
            transmittance,
        )
    return state, active


@wp.func
def _front_mid(
    p: int,
    n: int,
    mid_ids: wp.array2d(dtype=wp.int32),
    head_ids: wp.array2d(dtype=wp.int32),
    head_depths: wp.array2d(dtype=wp.float32),
    head_alphas: wp.array2d(dtype=wp.float32),
    transforms: wp.array(dtype=wp.mat44),
    opacities: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    x: float,
    y: float,
    head_count: int,
    contributor: int,
    active: bool,
    state: wp.vec4,
):
    # A CUDA head group consumes four mid slots, including invalid slots when
    # draining. A full head is blended before inspecting the next slot.
    for j in range(4):
        if head_count >= 4:
            state, still_active = _blend_first(p, head_ids, head_alphas, colors, state)
            active = active and still_active
            for k in range(3):
                head_ids[k, p] = head_ids[k + 1, p]
                head_depths[k, p] = head_depths[k + 1, p]
                head_alphas[k, p] = head_alphas[k + 1, p]
            head_count -= 1
        if j >= n:
            continue
        gid = mid_ids[j, p]
        if not active:
            continue
        contributor += 1
        power, depth = _sample(transforms[gid], x, y)
        if depth < -1.0 or depth > 1.0:
            continue
        alpha = wp.min(0.99, opacities[gid] * wp.exp(power))
        if alpha < 1.0 / 255.0:
            continue
        loc = head_count
        while loc > 0 and depth < head_depths[loc - 1, p]:
            head_depths[loc, p] = head_depths[loc - 1, p]
            head_ids[loc, p] = head_ids[loc - 1, p]
            head_alphas[loc, p] = head_alphas[loc - 1, p]
            loc -= 1
        head_depths[loc, p] = depth
        head_ids[loc, p] = gid
        head_alphas[loc, p] = alpha
        head_count += 1
    return head_count, contributor, active, state


@wp.func
def _push_mid(
    p: int,
    gid: int,
    depth: float,
    mid_ids: wp.array2d(dtype=wp.int32),
    mid_depths: wp.array2d(dtype=wp.float32),
    mid_count: int,
):
    loc = mid_count
    while loc > 0 and depth < mid_depths[loc - 1, p]:
        mid_depths[loc, p] = mid_depths[loc - 1, p]
        mid_ids[loc, p] = mid_ids[loc - 1, p]
        loc -= 1
    mid_depths[loc, p] = depth
    mid_ids[loc, p] = gid
    return mid_count + 1


@wp.func
def _flush_tail(
    p: int,
    amount: int,
    tail_count: int,
    mid_count: int,
    head_count: int,
    contributor: int,
    active: bool,
    state: wp.vec4,
    tail_ids: wp.array2d(dtype=wp.int32),
    tail_depths: wp.array2d(dtype=wp.float32),
    mid_ids: wp.array2d(dtype=wp.int32),
    mid_depths: wp.array2d(dtype=wp.float32),
    head_ids: wp.array2d(dtype=wp.int32),
    head_depths: wp.array2d(dtype=wp.float32),
    head_alphas: wp.array2d(dtype=wp.float32),
    transforms: wp.array(dtype=wp.mat44),
    opacities: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    mid_x: float,
    mid_y: float,
    x: float,
    y: float,
):
    # The tail sends 16 ordered candidates to the four 2x2 queues in groups of
    # four. Each mid queue holds up to eight entries and sends four to the head.
    for start in range(0, amount, 4):
        end = wp.min(start + 4, amount)
        for j in range(start, end):
            gid = tail_ids[j, p]
            _, depth = _sample(transforms[gid], mid_x, mid_y)
            mid_count = _push_mid(p, gid, depth, mid_ids, mid_depths, mid_count)
        if mid_count > 4:
            head_count, contributor, active, state = _front_mid(
                p,
                4,
                mid_ids,
                head_ids,
                head_depths,
                head_alphas,
                transforms,
                opacities,
                colors,
                x,
                y,
                head_count,
                contributor,
                active,
                state,
            )
            for j in range(mid_count - 4):
                mid_ids[j, p] = mid_ids[j + 4, p]
                mid_depths[j, p] = mid_depths[j + 4, p]
            mid_count -= 4
    for j in range(tail_count - amount):
        tail_ids[j, p] = tail_ids[j + amount, p]
        tail_depths[j, p] = tail_depths[j + amount, p]
    return tail_count - amount, mid_count, head_count, contributor, active, state


@wp.kernel(enable_backward=False)
def _render_3d(
    width: int,
    height: int,
    cull_4x4: bool,
    ranges: wp.array2d(dtype=wp.int32),
    point_list: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44),
    opacities: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    bg: wp.array(dtype=wp.vec3),
    tail_ids: wp.array2d(dtype=wp.int32),
    tail_depths: wp.array2d(dtype=wp.float32),
    mid_ids: wp.array2d(dtype=wp.int32),
    mid_depths: wp.array2d(dtype=wp.float32),
    head_ids: wp.array2d(dtype=wp.int32),
    head_depths: wp.array2d(dtype=wp.float32),
    head_alphas: wp.array2d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
    final_t: wp.array2d(dtype=wp.float32),
    contributors: wp.array2d(dtype=wp.int32),
):
    p = wp.tid()
    ix = p % width
    iy = p // width
    x, y = float(ix), float(iy)
    tile_id = (iy // 16) * ((width + 15) // 16) + ix // 16
    first, last = ranges[tile_id, 0], ranges[tile_id, 1]
    tail_x, tail_y = float(ix // 4 * 4 + 2), float(iy // 4 * 4 + 2)
    mid_x, mid_y = float(ix // 2 * 2 + 1), float(iy // 2 * 2 + 1)
    tail_count, mid_count, head_count, contributor = int(0), int(0), int(0), int(0)
    active = bool(True)
    state = wp.vec4(0.0, 0.0, 0.0, 1.0)
    for batch_start in range(first, last, 32):
        if not active:
            break
        for index in range(batch_start, wp.min(batch_start + 32, last)):
            gid = point_list[index]
            if gid < 0:
                continue
            g2s = transforms[gid]
            if cull_4x4:
                power = frustum_minimum(
                    wp.vec2(float(ix // 4 * 4), float(iy // 4 * 4)),
                    wp.vec2(float(ix // 4 * 4 + 3), float(iy // 4 * 4 + 3)),
                    g2s,
                )
                if wp.min(0.99, opacities[gid] * wp.exp(-power)) < 1.0 / 255.0:
                    continue
            _, depth = _sample(g2s, tail_x, tail_y)
            loc = tail_count
            while loc > 0 and depth < tail_depths[loc - 1, p]:
                tail_depths[loc, p] = tail_depths[loc - 1, p]
                tail_ids[loc, p] = tail_ids[loc - 1, p]
                loc -= 1
            tail_depths[loc, p] = depth
            tail_ids[loc, p] = gid
            tail_count += 1
        while tail_count > 32:
            tail_count, mid_count, head_count, contributor, active, state = _flush_tail(
                p,
                16,
                tail_count,
                mid_count,
                head_count,
                contributor,
                active,
                state,
                tail_ids,
                tail_depths,
                mid_ids,
                mid_depths,
                head_ids,
                head_depths,
                head_alphas,
                transforms,
                opacities,
                colors,
                mid_x,
                mid_y,
                x,
                y,
            )
    while tail_count > 0 and active:
        amount = wp.min(16, tail_count)
        tail_count, mid_count, head_count, contributor, active, state = _flush_tail(
            p,
            amount,
            tail_count,
            mid_count,
            head_count,
            contributor,
            active,
            state,
            tail_ids,
            tail_depths,
            mid_ids,
            mid_depths,
            head_ids,
            head_depths,
            head_alphas,
            transforms,
            opacities,
            colors,
            mid_x,
            mid_y,
            x,
            y,
        )
    while mid_count > 0 and active:
        amount = wp.min(4, mid_count)
        head_count, contributor, active, state = _front_mid(
            p,
            amount,
            mid_ids,
            head_ids,
            head_depths,
            head_alphas,
            transforms,
            opacities,
            colors,
            x,
            y,
            head_count,
            contributor,
            active,
            state,
        )
        for j in range(mid_count - amount):
            mid_ids[j, p] = mid_ids[j + amount, p]
            mid_depths[j, p] = mid_depths[j + amount, p]
        mid_count -= amount
    while head_count > 0 and active:
        state, active = _blend_first(p, head_ids, head_alphas, colors, state)
        for j in range(head_count - 1):
            head_ids[j, p] = head_ids[j + 1, p]
            head_alphas[j, p] = head_alphas[j + 1, p]
        head_count -= 1
    background = bg[0]
    output[0, iy, ix] = state[0] + state[3] * background[0]
    output[1, iy, ix] = state[1] + state[3] * background[1]
    output[2, iy, ix] = state[2] + state[3] * background[2]
    final_t[iy, ix] = state[3]
    contributors[iy, ix] = contributor


@torch.no_grad()
def render_hierarchical_3d(
    preprocessed: PreprocessedGaussians,
    bins: TileBins,
    raster_settings: GaussianRasterizationSettings,
) -> RenderOutput:
    """Render a 3D AAA scene with StopThePop's 64/8/4 queue structure.

    Returns ``color`` (3,H,W), ``final_T`` (H,W) and a diagnostic per-pixel
    candidate count. This implementation stores queues per pixel and is aimed
    at parity work; the CUDA source shares them within 4x4/2x2 groups.
    """
    settings = raster_settings.settings
    if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
        raise ValueError("requires eval_3D=True and sort_mode=HIER")
    sizes = settings.sort_settings.queue_sizes
    if sizes.tile_4x4 != 64 or sizes.tile_2x2 != 8 or sizes.per_pixel != 4:
        raise ValueError("only queue sizes 64/8/4 are supported")
    if raster_settings.render_depth:
        raise ValueError("render_depth is not supported")
    if "gauss2screen" not in preprocessed or "opacity" not in preprocessed:
        raise ValueError("preprocessed data must contain 3D transforms and opacity")
    width, height = raster_settings.image_width, raster_settings.image_height
    device = preprocessed["gauss2screen"].device
    pixels = width * height
    # Queue slot first keeps neighboring pixels contiguous for Warp threads.
    ids = torch.empty((64 + 8 + 4, pixels), dtype=torch.int32, device=device)
    depths = torch.empty((64 + 8 + 4, pixels), dtype=torch.float32, device=device)
    alphas = torch.empty((4, pixels), dtype=torch.float32, device=device)
    color = torch.empty((3, height, width), dtype=torch.float32, device=device)
    final_t = torch.empty((height, width), dtype=torch.float32, device=device)
    contributors = torch.empty((height, width), dtype=torch.int32, device=device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    wp.launch(
        _render_3d,
        dim=pixels,
        inputs=[
            width,
            height,
            bool(settings.culling_settings.hierarchical_4x4_culling),
            wp.from_torch(bins["ranges"]),
            wp.from_torch(bins["point_list"]),
            wp.from_torch(preprocessed["gauss2screen"], dtype=wp.mat44),
            wp.from_torch(preprocessed["opacity"]),
            wp.from_torch(preprocessed["rgb"], dtype=wp.vec3),
            wp.from_torch(raster_settings.bg.reshape(1, 3), dtype=wp.vec3),
            wp.from_torch(ids[:64, :]),
            wp.from_torch(depths[:64, :]),
            wp.from_torch(ids[64:72, :]),
            wp.from_torch(depths[64:72, :]),
            wp.from_torch(ids[72:76, :]),
            wp.from_torch(depths[72:76, :]),
            wp.from_torch(alphas),
        ],
        outputs=[
            wp.from_torch(color),
            wp.from_torch(final_t),
            wp.from_torch(contributors),
        ],
        stream=stream,
    )
    return dict(color=color, final_T=final_t, contributors=contributors)
