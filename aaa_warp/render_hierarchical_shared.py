"""Shared 4x4 queues for AAA's 3D hierarchical renderer.

The queue algorithm follows StopThePop, Copyright (c) 2024 Graz University of
Technology, MIT-licensed; see the local stopthepop/LICENSE.txt. An optional
third Warp kernel also shares the 2x2 queues.
"""
import torch
import warp as wp

from .geometry import frustum_minimum
from .render_hierarchical import _blend_first, _front_mid, _push_mid, _sample


@wp.func_native('return __float_as_uint(value);')
def _float_bits(value: wp.float32) -> wp.uint32:
    ...


@wp.func
def _tail_sort_key(depth: wp.float32, pair: int):
    # The native helper only reinterprets float bits. All queueing and sorting
    # remain Warp code. Pair position makes equal-depth ordering deterministic.
    if pair < 0:
        return wp.uint64(-1)
    bits = _float_bits(depth)
    ascending = wp.uint32(0)
    if (bits & wp.uint32(0x80000000)) != wp.uint32(0):
        ascending = ~bits
    else:
        ascending = bits ^ wp.uint32(0x80000000)
    return (wp.uint64(ascending) << wp.uint64(32)) | wp.uint64(pair)


@wp.kernel(enable_backward=False)
def _sort_tail(
    width: int, cull_4x4: bool,
    ranges: wp.array2d(dtype=wp.int32), point_list: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44), opacities: wp.array(dtype=wp.float32),
    scratch_ids: wp.array2d(dtype=wp.int32),
    scratch_depths: wp.array2d(dtype=wp.float32),
    stream: wp.array2d(dtype=wp.int32), counts: wp.array2d(dtype=wp.int32),
):
    group_id = wp.tid()
    tile = group_id // 16
    local = group_id % 16
    grid_width = (width + 15) // 16
    x0 = float((tile % grid_width) * 16 + (local % 4) * 4)
    y0 = float((tile // grid_width) * 16 + (local // 4) * 4)
    center_x, center_y = x0 + 2.0, y0 + 2.0
    first, last = ranges[tile, 0], ranges[tile, 1]
    count, produced = int(0), int(0)
    for batch_start in range(first, last, 32):
        for index in range(batch_start, wp.min(batch_start + 32, last)):
            gid = point_list[index]
            if gid < 0:
                continue
            g2s = transforms[gid]
            if cull_4x4:
                minimum = frustum_minimum(wp.vec2(x0, y0),
                                          wp.vec2(x0 + 3.0, y0 + 3.0), g2s)
                if wp.min(0.99, opacities[gid] * wp.exp(-minimum)) < 1.0 / 255.0:
                    continue
            _, depth = _sample(g2s, center_x, center_y)
            loc = count
            while loc > 0 and depth < scratch_depths[loc - 1, group_id]:
                scratch_depths[loc, group_id] = scratch_depths[loc - 1, group_id]
                scratch_ids[loc, group_id] = scratch_ids[loc - 1, group_id]
                loc -= 1
            scratch_depths[loc, group_id] = depth
            scratch_ids[loc, group_id] = gid
            count += 1
        while count > 32:
            for j in range(16):
                stream[local, first + produced + j] = scratch_ids[j, group_id]
            produced += 16
            for j in range(count - 16):
                scratch_ids[j, group_id] = scratch_ids[j + 16, group_id]
                scratch_depths[j, group_id] = scratch_depths[j + 16, group_id]
            count -= 16
    for j in range(count):
        stream[local, first + produced + j] = scratch_ids[j, group_id]
    counts[local, tile] = produced + count


@wp.kernel(enable_backward=False)
def _sort_tail_tiled(
    width: int, cull_4x4: bool,
    ranges: wp.array2d(dtype=wp.int32), point_list: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44), opacities: wp.array(dtype=wp.float32),
    scratch_ids: wp.array2d(dtype=wp.int32),
    scratch_keys: wp.array2d(dtype=wp.uint64),
    stream: wp.array2d(dtype=wp.int32), counts: wp.array2d(dtype=wp.int32),
):
    group_id, lane = wp.tid()
    tile = group_id // 16
    local = group_id % 16
    grid_width = (width + 15) // 16
    x0 = float((tile % grid_width) * 16 + (local % 4) * 4)
    y0 = float((tile // grid_width) * 16 + (local // 4) * 4)
    first, last = ranges[tile, 0], ranges[tile, 1]
    count, produced = int(0), int(0)
    for batch_start in range(first, last, 32):
        old_pair = int(-1)
        old_key = wp.uint64(-1)
        if lane < count:
            old_pair = scratch_ids[lane, group_id]
            old_key = scratch_keys[lane, group_id]
        new_pair = int(-1)
        new_depth = float(3.4028234663852886e38)
        index = batch_start + lane
        if index < last:
            candidate = point_list[index]
            if candidate >= 0:
                g2s = transforms[candidate]
                accept = bool(True)
                if cull_4x4:
                    minimum = frustum_minimum(wp.vec2(x0, y0),
                                              wp.vec2(x0 + 3.0, y0 + 3.0), g2s)
                    if wp.min(0.99, opacities[candidate] * wp.exp(-minimum)) < 1.0 / 255.0:
                        accept = False
                if accept:
                    _, new_depth = _sample(g2s, x0 + 2.0, y0 + 2.0)
                    new_pair = index

        keys = wp.tile_full(shape=(64,), value=wp.uint64(-1),
                            dtype=wp.uint64, storage='shared')
        values = wp.tile_full(shape=(64,), value=-1,
                              dtype=wp.int32, storage='shared')
        wp.tile_assign(keys, wp.tile(old_key), offset=(0,))
        wp.tile_assign(values, wp.tile(old_pair), offset=(0,))
        wp.tile_assign(keys, wp.tile(_tail_sort_key(new_depth, new_pair)), offset=(32,))
        wp.tile_assign(values, wp.tile(new_pair), offset=(32,))
        wp.tile_sort(keys, values)
        added = wp.tile_extract(wp.tile_sum(wp.tile(int(new_pair >= 0))), 0)
        count += added
        shift = int(0)
        if count > 32:
            if lane < 16:
                stream[local, first + produced + lane] = point_list[wp.tile_extract(values, lane)]
            produced += 16
            count -= 16
            shift += 16
        if count > 32:
            if lane < 16:
                stream[local, first + produced + lane] = point_list[wp.tile_extract(values, shift + lane)]
            produced += 16
            count -= 16
            shift += 16
        if lane < count:
            scratch_ids[lane, group_id] = wp.tile_extract(values, shift + lane)
            scratch_keys[lane, group_id] = wp.tile_extract(keys, shift + lane)
    if lane < count:
        stream[local, first + produced + lane] = point_list[scratch_ids[lane, group_id]]
    if lane == 0:
        counts[local, tile] = produced + count


@wp.kernel(enable_backward=False)
def _sort_mid(
    width: int, ranges: wp.array2d(dtype=wp.int32),
    tail_stream: wp.array2d(dtype=wp.int32),
    tail_counts: wp.array2d(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44),
    scratch_ids: wp.array2d(dtype=wp.int32),
    scratch_depths: wp.array2d(dtype=wp.float32),
    stream: wp.array2d(dtype=wp.int32), counts: wp.array2d(dtype=wp.int32),
):
    group_id = wp.tid()
    tile = group_id // 64
    local = group_id % 64
    grid_width = (width + 15) // 16
    x2, y2 = local % 8, local // 8
    parent = (y2 // 2) * 4 + x2 // 2
    mid_x = float((tile % grid_width) * 16 + x2 * 2 + 1)
    mid_y = float((tile // grid_width) * 16 + y2 * 2 + 1)
    first = ranges[tile, 0]
    input_count = tail_counts[parent, tile]
    count, produced = int(0), int(0)
    for group_start in range(0, input_count, 4):
        for j in range(group_start, wp.min(group_start + 4, input_count)):
            gid = tail_stream[parent, first + j]
            _, depth = _sample(transforms[gid], mid_x, mid_y)
            loc = count
            while loc > 0 and depth < scratch_depths[loc - 1, group_id]:
                scratch_depths[loc, group_id] = scratch_depths[loc - 1, group_id]
                scratch_ids[loc, group_id] = scratch_ids[loc - 1, group_id]
                loc -= 1
            scratch_depths[loc, group_id] = depth
            scratch_ids[loc, group_id] = gid
            count += 1
        if count > 4:
            for j in range(4):
                stream[local, first + produced + j] = scratch_ids[j, group_id]
            produced += 4
            for j in range(count - 4):
                scratch_ids[j, group_id] = scratch_ids[j + 4, group_id]
                scratch_depths[j, group_id] = scratch_depths[j + 4, group_id]
            count -= 4
    for j in range(count):
        stream[local, first + produced + j] = scratch_ids[j, group_id]
    counts[local, tile] = produced + count


@wp.kernel(enable_backward=False)
def _blend_pixels(
    width: int, ranges: wp.array2d(dtype=wp.int32),
    mid_stream: wp.array2d(dtype=wp.int32),
    mid_counts: wp.array2d(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44), opacities: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=wp.vec3),
    head_ids: wp.array2d(dtype=wp.int32),
    head_depths: wp.array2d(dtype=wp.float32),
    head_alphas: wp.array2d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32), final_t: wp.array2d(dtype=wp.float32),
    contributors: wp.array2d(dtype=wp.int32),
):
    p = wp.tid()
    ix, iy = p % width, p // width
    grid_width = (width + 15) // 16
    tile = (iy // 16) * grid_width + ix // 16
    local = ((iy % 16) // 2) * 8 + (ix % 16) // 2
    first = ranges[tile, 0]
    count = mid_counts[local, tile]
    head_count, contributor = int(0), int(0)
    active = bool(True)
    state = wp.vec4(0.0, 0.0, 0.0, 1.0)
    for group_start in range(0, count, 4):
        if not active:
            break
        for j in range(4):
            if head_count >= 4:
                state, still_active = _blend_first(p, head_ids, head_alphas, colors, state)
                active = active and still_active
                for k in range(3):
                    head_ids[k, p] = head_ids[k + 1, p]
                    head_depths[k, p] = head_depths[k + 1, p]
                    head_alphas[k, p] = head_alphas[k + 1, p]
                head_count -= 1
            if group_start + j >= count or not active:
                continue
            gid = mid_stream[local, first + group_start + j]
            contributor += 1
            power, depth = _sample(transforms[gid], float(ix), float(iy))
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


@wp.kernel(enable_backward=False)
def _blend_from_tail(
    width: int, ranges: wp.array2d(dtype=wp.int32),
    tail_stream: wp.array2d(dtype=wp.int32),
    tail_counts: wp.array2d(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44), opacities: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=wp.vec3),
    mid_ids: wp.array2d(dtype=wp.int32),
    mid_depths: wp.array2d(dtype=wp.float32),
    head_ids: wp.array2d(dtype=wp.int32),
    head_depths: wp.array2d(dtype=wp.float32),
    head_alphas: wp.array2d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32), final_t: wp.array2d(dtype=wp.float32),
    contributors: wp.array2d(dtype=wp.int32),
):
    p = wp.tid()
    ix, iy = p % width, p // width
    grid_width = (width + 15) // 16
    tile = (iy // 16) * grid_width + ix // 16
    parent = ((iy % 16) // 4) * 4 + (ix % 16) // 4
    mid_x = float(ix // 2 * 2 + 1)
    mid_y = float(iy // 2 * 2 + 1)
    first = ranges[tile, 0]
    count = tail_counts[parent, tile]
    mid_count, head_count, contributor = int(0), int(0), int(0)
    active = bool(True)
    state = wp.vec4(0.0, 0.0, 0.0, 1.0)
    for group_start in range(0, count, 4):
        if not active:
            break
        for j in range(group_start, wp.min(group_start + 4, count)):
            gid = tail_stream[parent, first + j]
            _, depth = _sample(transforms[gid], mid_x, mid_y)
            mid_count = _push_mid(p, gid, depth, mid_ids, mid_depths, mid_count)
        if mid_count > 4:
            head_count, contributor, active, state = _front_mid(
                p, 4, mid_ids, head_ids, head_depths, head_alphas,
                transforms, opacities, colors, float(ix), float(iy),
                head_count, contributor, active, state)
            for j in range(mid_count - 4):
                mid_ids[j, p] = mid_ids[j + 4, p]
                mid_depths[j, p] = mid_depths[j + 4, p]
            mid_count -= 4
    while mid_count > 0 and active:
        amount = wp.min(4, mid_count)
        head_count, contributor, active, state = _front_mid(
            p, amount, mid_ids, head_ids, head_depths, head_alphas,
            transforms, opacities, colors, float(ix), float(iy),
            head_count, contributor, active, state)
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
def render_hierarchical_3d(preprocessed, bins, raster_settings, *,
                           share_mid=False, cooperative_tail=True):
    """Render 3D AAA with one 64-item queue per 4x4, then 8 per 2x2.

    Returns ``color``, ``final_T``, and a diagnostic candidate count. The
    default shares the 4x4 tail queue and keeps a short mid/head queue per
    pixel. ``share_mid=True`` also shares the 2x2 queue, using a third kernel
    and an additional 64-entry-per-pair candidate stream. The default 4x4
    stage sorts each batch cooperatively with 32 Warp threads; setting
    ``cooperative_tail=False`` selects the serial reference for diagnostics.
    """
    wp.init()
    settings = raster_settings.settings
    if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
        raise ValueError('requires eval_3D=True and sort_mode=HIER')
    sizes = settings.sort_settings.queue_sizes
    if sizes.tile_4x4 != 64 or sizes.tile_2x2 != 8 or sizes.per_pixel != 4:
        raise ValueError('only queue sizes 64/8/4 are supported')
    if raster_settings.render_depth:
        raise ValueError('render_depth is not supported')
    if 'gauss2screen' not in preprocessed or 'opacity' not in preprocessed:
        raise ValueError('preprocessed data must contain 3D transforms and opacity')

    width, height = raster_settings.image_width, raster_settings.image_height
    device = preprocessed['gauss2screen'].device
    grid_width, grid_height = (width + 15) // 16, (height + 15) // 16
    tiles, pairs, pixels = grid_width * grid_height, bins['point_list'].numel(), width * height
    tail_groups, mid_groups = 16 * tiles, 64 * tiles
    tail_ids = torch.empty((64, tail_groups), dtype=torch.int32, device=device)
    tail_depths = torch.empty((64, tail_groups),
                              dtype=torch.int64 if cooperative_tail else torch.float32,
                              device=device)
    mid_capacity = mid_groups if share_mid else pixels
    mid_ids = torch.empty((8, mid_capacity), dtype=torch.int32, device=device)
    mid_depths = torch.empty((8, mid_capacity), dtype=torch.float32, device=device)
    head_ids = torch.empty((4, pixels), dtype=torch.int32, device=device)
    head_depths = torch.empty((4, pixels), dtype=torch.float32, device=device)
    head_alphas = torch.empty((4, pixels), dtype=torch.float32, device=device)
    tail_stream = torch.empty((16, pairs), dtype=torch.int32, device=device)
    mid_stream = torch.empty((64, pairs), dtype=torch.int32, device=device) if share_mid else None
    tail_counts = torch.empty((16, tiles), dtype=torch.int32, device=device)
    mid_counts = torch.empty((64, tiles), dtype=torch.int32, device=device) if share_mid else None
    color = torch.empty((3, height, width), dtype=torch.float32, device=device)
    final_t = torch.empty((height, width), dtype=torch.float32, device=device)
    contributors = torch.empty((height, width), dtype=torch.int32, device=device)

    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    ranges = wp.from_torch(bins['ranges'])
    transforms = wp.from_torch(preprocessed['gauss2screen'], dtype=wp.mat44)
    opacity = wp.from_torch(preprocessed['opacity'])
    rgb = wp.from_torch(preprocessed['rgb'], dtype=wp.vec3)
    tail_inputs = [width, bool(settings.culling_settings.hierarchical_4x4_culling),
                   ranges, wp.from_torch(bins['point_list']), transforms, opacity,
                   wp.from_torch(tail_ids),
                   wp.from_torch(tail_depths, dtype=wp.uint64 if cooperative_tail else wp.float32)]
    tail_outputs = [wp.from_torch(tail_stream), wp.from_torch(tail_counts)]
    if cooperative_tail:
        wp.launch_tiled(_sort_tail_tiled, dim=tail_groups, inputs=tail_inputs,
                        outputs=tail_outputs, block_dim=32, stream=stream)
    else:
        wp.launch(_sort_tail, dim=tail_groups, inputs=tail_inputs,
                  outputs=tail_outputs, stream=stream)
    background = wp.from_torch(raster_settings.bg.reshape(1, 3), dtype=wp.vec3)
    outputs = [wp.from_torch(color), wp.from_torch(final_t), wp.from_torch(contributors)]
    if share_mid:
        wp.launch(_sort_mid, dim=mid_groups, inputs=[
            width, ranges, wp.from_torch(tail_stream), wp.from_torch(tail_counts),
            transforms, wp.from_torch(mid_ids), wp.from_torch(mid_depths)],
            outputs=[wp.from_torch(mid_stream), wp.from_torch(mid_counts)], stream=stream)
        wp.launch(_blend_pixels, dim=pixels, inputs=[
            width, ranges, wp.from_torch(mid_stream), wp.from_torch(mid_counts),
            transforms, opacity, rgb, background,
            wp.from_torch(head_ids), wp.from_torch(head_depths),
            wp.from_torch(head_alphas)], outputs=outputs, stream=stream)
    else:
        wp.launch(_blend_from_tail, dim=pixels, inputs=[
            width, ranges, wp.from_torch(tail_stream), wp.from_torch(tail_counts),
            transforms, opacity, rgb, background,
            wp.from_torch(mid_ids), wp.from_torch(mid_depths),
            wp.from_torch(head_ids), wp.from_torch(head_depths),
            wp.from_torch(head_alphas)], outputs=outputs, stream=stream)
    return dict(color=color, final_T=final_t, contributors=contributors)
