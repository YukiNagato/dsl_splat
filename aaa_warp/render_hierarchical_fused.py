"""One-block-per-4x4 StopThePop renderer with on-chip streaming queues.

The ordering follows the reference Warp 64/8/4 queues. Each 32-thread block
sorts its tail cooperatively; 16 lanes render pixels with local mid/head
queues. No Gaussian/tile-sized candidate stream is materialized.
StopThePop formulas are MIT-licensed; see the reference's LICENSE.txt.
"""
import torch
import warp as wp
from .interop import current_stream

from .geometry import frustum_minimum
from .render_hierarchical import _sample
from .render_hierarchical_shared import _tail_sort_key, render_hierarchical_3d as _render_staged


ivec8 = wp.types.vector(length=8, dtype=wp.int32)
vec8 = wp.types.vector(length=8, dtype=wp.float32)


@wp.struct
class _Pixel:
    mid_ids: ivec8
    mid_depths: vec8
    head_ids: wp.vec4i
    head_depths: wp.vec4
    head_alphas: wp.vec4
    mid_count: int
    head_count: int
    contributor: int
    active: bool
    state: wp.vec4


@wp.func
def _blend(q: _Pixel, colors: wp.array(dtype=wp.vec3)):
    alpha = q.head_alphas[0]
    transmittance = q.state[3] * (1.0 - alpha)
    if transmittance < 0.0001:
        q.active = False
    else:
        rgb = colors[q.head_ids[0]]
        weight = alpha * q.state[3]
        q.state = wp.vec4(q.state[0] + rgb[0] * weight,
                          q.state[1] + rgb[1] * weight,
                          q.state[2] + rgb[2] * weight, transmittance)
    for k in range(3):
        q.head_ids[k] = q.head_ids[k + 1]
        q.head_depths[k] = q.head_depths[k + 1]
        q.head_alphas[k] = q.head_alphas[k + 1]
    q.head_ids[3] = -1
    q.head_depths[3] = 3.4028234663852886e38
    q.head_alphas[3] = 0.0
    q.head_count -= 1
    return q


@wp.func
def _push(q: _Pixel, gid: int, depth: float):
    # Fixed component accesses let the compiler keep this short queue in
    # registers; dynamic vector indexing creates thread-local memory arrays.
    candidate_id, candidate_depth = gid, depth
    inserted = bool(False)
    for j in range(8):
        if j <= q.mid_count and (inserted or j == q.mid_count or depth < q.mid_depths[j]):
            previous_id, previous_depth = q.mid_ids[j], q.mid_depths[j]
            q.mid_ids[j] = candidate_id
            q.mid_depths[j] = candidate_depth
            candidate_id, candidate_depth = previous_id, previous_depth
            inserted = True
    q.mid_count += 1
    return q


@wp.func
def _front(q: _Pixel, amount: int, x: float, y: float,
           transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
           colors: wp.array(dtype=wp.vec3)):
    # Consume four slots, even on the final partial group: the reference
    # blends a full head before checking whether the next slot is valid.
    for j in range(4):
        if q.head_count >= 4:
            q = _blend(q, colors)
        if j < amount and q.active:
            gid = q.mid_ids[j]
            q.contributor += 1
            power, depth = _sample(transforms[gid], x, y)
            if depth >= -1.0 and depth <= 1.0:
                alpha = wp.min(0.99, opacity[gid] * wp.exp(power))
                if alpha >= 1.0 / 255.0:
                    candidate_id, candidate_depth, candidate_alpha = gid, depth, alpha
                    inserted = bool(False)
                    for k in range(4):
                        if k <= q.head_count and (inserted or k == q.head_count or depth < q.head_depths[k]):
                            previous_id = q.head_ids[k]
                            previous_depth, previous_alpha = q.head_depths[k], q.head_alphas[k]
                            q.head_ids[k] = candidate_id
                            q.head_depths[k] = candidate_depth
                            q.head_alphas[k] = candidate_alpha
                            candidate_id = previous_id
                            candidate_depth, candidate_alpha = previous_depth, previous_alpha
                            inserted = True
                    q.head_count += 1
    for j in range(4):
        q.mid_ids[j] = q.mid_ids[j + 4]
        q.mid_depths[j] = q.mid_depths[j + 4]
        q.mid_ids[j + 4] = -1
        q.mid_depths[j + 4] = 3.4028234663852886e38
    q.mid_count -= amount
    return q


@wp.kernel(enable_backward=False)
def _render_fused(
    width: int, height: int, cull_4x4: bool,
    ranges: wp.array2d(dtype=int), point_list: wp.array(dtype=int),
    transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
    colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=wp.vec3),
    output: wp.array3d(dtype=float), final_t: wp.array2d(dtype=float),
    contributors: wp.array2d(dtype=int),
):
    group, lane = wp.tid()
    tile, local = group // 16, group % 16
    grid_width = (width + 15) // 16
    x0 = (tile % grid_width) * 16 + (local % 4) * 4
    y0 = (tile // grid_width) * 16 + (local // 4) * 4
    ix, iy = x0 + lane % 4, y0 + lane // 4
    mid_x, mid_y = float(ix // 2 * 2 + 1), float(iy // 2 * 2 + 1)
    q = _Pixel()
    q.mid_ids = ivec8(-1)
    q.mid_depths = vec8(3.4028234663852886e38)
    q.head_ids = wp.vec4i(-1)
    q.head_depths = wp.vec4(3.4028234663852886e38)
    q.active = lane < 16 and ix < width and iy < height
    q.state = wp.vec4(0.0, 0.0, 0.0, 1.0)
    first, last = ranges[tile, 0], ranges[tile, 1]
    count = int(0)
    old_pair, old_key = int(-1), wp.uint64(-1)
    keys = wp.tile_full(shape=(64,), value=wp.uint64(-1),
                        dtype=wp.uint64, storage='shared')
    values = wp.tile_full(shape=(64,), value=-1,
                          dtype=wp.int32, storage='shared')
    previous_keys = wp.tile_full(shape=(32,), value=wp.uint64(-1),
                                 dtype=wp.uint64, storage='shared')
    new_keys = wp.tile_full(shape=(32,), value=wp.uint64(-1),
                            dtype=wp.uint64, storage='shared')
    new_values = wp.tile_full(shape=(32,), value=-1,
                              dtype=wp.int32, storage='shared')
    for batch_start in range(first, last, 32):
        alive = wp.tile_extract(wp.tile_sum(wp.tile(int(q.active))), 0)
        if alive == 0:
            break
        pair = int(-1)
        depth = float(3.4028234663852886e38)
        index = batch_start + lane
        if index < last:
            gid = point_list[index]
            if gid >= 0:
                g2s = transforms[gid]
                accept = bool(True)
                if cull_4x4:
                    minimum = frustum_minimum(wp.vec2(float(x0), float(y0)),
                                              wp.vec2(float(x0 + 3), float(y0 + 3)), g2s)
                    if wp.min(0.99, opacity[gid] * wp.exp(-minimum)) < 1.0 / 255.0:
                        accept = False
                if accept:
                    _, depth = _sample(g2s, float(x0 + 2), float(y0 + 2))
                    pair = index
        # Sort only the new batch. The retained tail is already ordered;
        # merge the two 32-item lists by parallel binary-search ranks.
        wp.tile_assign(previous_keys, wp.tile(old_key))
        wp.tile_assign(new_keys, wp.tile(_tail_sort_key(depth, pair)))
        wp.tile_assign(new_values, wp.tile(pair))
        wp.tile_sort(new_keys, new_values)
        new_key = wp.tile_extract(new_keys, lane)
        new_pair = wp.tile_extract(new_values, lane)
        new_rank, old_rank, stride = int(0), int(0), int(16)
        for step in range(5):
            new_index, old_index = new_rank + stride, old_rank + stride
            if wp.tile_extract(previous_keys, new_index) <= new_key:
                new_rank = new_index
            if wp.tile_extract(new_keys, old_index) < old_key:
                old_rank = old_index
            stride //= 2
        if wp.tile_extract(previous_keys, 0) <= new_key:
            new_rank += 1
        if wp.tile_extract(new_keys, 0) < old_key:
            old_rank += 1
        # Upper/lower-bound tie rules give padded sentinels unique positions.
        wp.tile_scatter_masked(keys, new_rank + lane, new_key, True)
        wp.tile_scatter_masked(values, new_rank + lane, new_pair, True)
        wp.tile_scatter_masked(keys, old_rank + lane, old_key, True)
        wp.tile_scatter_masked(values, old_rank + lane, old_pair, True)
        count += wp.tile_extract(wp.tile_sum(wp.tile(int(pair >= 0))), 0)
        shift = int(0)
        while count > 32:
            for start in range(0, wp.min(count, 16), 4):
                for j in range(4):
                    candidate_pair = wp.tile_extract(values, shift + start + j)
                    if q.active:
                        gid = point_list[candidate_pair]
                        _, mid_depth = _sample(transforms[gid], mid_x, mid_y)
                        q = _push(q, gid, mid_depth)
                if q.active and q.mid_count > 4:
                    q = _front(q, 4, float(ix), float(iy), transforms, opacity, colors)
            count -= 16
            shift += 16
        old_pair = int(-1)
        old_key = wp.uint64(-1)
        next_pair = wp.tile_extract(values, shift + lane)
        next_key = wp.tile_extract(keys, shift + lane)
        if lane < count:
            old_pair, old_key = next_pair, next_key

    # Reconstruct the final tail from register-held entries. Collective tile
    # operations are executed by all lanes, including out-of-image pixels.
    wp.tile_assign(values, wp.tile(old_pair), offset=(0,))
    for start in range(0, count, 4):
        amount = wp.min(4, count - start)
        for j in range(amount):
            candidate_pair = wp.tile_extract(values, start + j)
            if q.active:
                gid = point_list[candidate_pair]
                _, mid_depth = _sample(transforms[gid], mid_x, mid_y)
                q = _push(q, gid, mid_depth)
        if q.active and q.mid_count > 4:
            q = _front(q, 4, float(ix), float(iy), transforms, opacity, colors)
    while q.mid_count > 0 and q.active:
        q = _front(q, wp.min(4, q.mid_count), float(ix), float(iy),
                    transforms, opacity, colors)
    while q.head_count > 0 and q.active:
        q = _blend(q, colors)
    if lane < 16 and ix < width and iy < height:
        background = bg[0]
        output[0, iy, ix] = q.state[0] + q.state[3] * background[0]
        output[1, iy, ix] = q.state[1] + q.state[3] * background[1]
        output[2, iy, ix] = q.state[2] + q.state[3] * background[2]
        final_t[iy, ix] = q.state[3]
        contributors[iy, ix] = q.contributor


@torch.no_grad()
def render_hierarchical_3d(preprocessed, bins, raster_settings, *, output=None,
                           share_mid=False, cooperative_tail=True):
    """Render using fused 4x4 blocks; optional output buffers can be reused.

    An output mapping must contain contiguous CUDA tensors ``color`` (3,H,W),
    ``final_T`` (H,W), and ``contributors`` (H,W), of float32/float32/int32.
    Reused outputs must be consumed before the next invocation on those buffers.
    The earlier diagnostic options ``share_mid=True`` and
    ``cooperative_tail=False`` select their staged implementations.
    """
    if share_mid or not cooperative_tail:
        if output is not None:
            raise ValueError('reusable output buffers require the fused renderer')
        return _render_staged(preprocessed, bins, raster_settings,
                              share_mid=share_mid, cooperative_tail=cooperative_tail)
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
    layouts = {'color': ((3, height, width), torch.float32),
               'final_T': ((height, width), torch.float32),
               'contributors': ((height, width), torch.int32)}
    if output is None:
        output = {name: torch.empty(shape, dtype=dtype, device=device)
                  for name, (shape, dtype) in layouts.items()}
    else:
        for name, (shape, dtype) in layouts.items():
            value = output.get(name)
            if (not isinstance(value, torch.Tensor) or value.device != device or
                    value.dtype != dtype or tuple(value.shape) != shape or
                    not value.is_contiguous() or value.requires_grad):
                raise ValueError(f'output[{name!r}] must be contiguous {dtype} '
                                 f'with shape {shape} on {device}, without gradients')
    stream = current_stream(device)
    wp.launch_tiled(_render_fused, dim=16 * ((width + 15) // 16) * ((height + 15) // 16),
                    inputs=[width, height, bool(settings.culling_settings.hierarchical_4x4_culling),
                            wp.from_torch(bins['ranges']), wp.from_torch(bins['point_list']),
                            wp.from_torch(preprocessed['gauss2screen'], dtype=wp.mat44),
                            wp.from_torch(preprocessed['opacity']),
                            wp.from_torch(preprocessed['rgb'], dtype=wp.vec3),
                            wp.from_torch(raster_settings.bg.reshape(1, 3), dtype=wp.vec3)],
                    outputs=[wp.from_torch(output['color']), wp.from_torch(output['final_T']),
                             wp.from_torch(output['contributors'])],
                    block_dim=32, stream=stream)
    return output
