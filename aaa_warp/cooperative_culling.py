"""GPU work queue and 32-lane cooperative tile culling.

The reference CUDA implementation checks 32 tiles per Gaussian serially, then
shares remaining work among warp lanes. Warp's tiled blocks provide the same
32-lane ownership and a collective integer reduction. The queue keeps short
Gaussians out of the collective stage.
"""
import warp as wp

from .geometry import frustum_minimum, tile_minimum_2d


@wp.func
def _tile_contributes(tile_index: int, xmin: int, ymin: int, rect_width: int,
                      i: int, center: wp.vec2, conics: wp.array(dtype=wp.vec4),
                      transforms: wp.array(dtype=wp.mat44),
                      eval3d: bool, threshold: float):
    y = tile_index // rect_width + ymin
    x = tile_index % rect_width + xmin
    lo = wp.vec2(float(x * 16), float(y * 16))
    hi = lo + wp.vec2(15.0)
    power = float(0.0)
    if eval3d:
        power = frustum_minimum(lo, hi, transforms[i])
    else:
        power = tile_minimum_2d(conics[i], center, lo, hi)
    return power <= threshold


@wp.func
def _rect_lower(center: wp.vec2, extent: wp.vec2, width: int, height: int):
    gx, gy = (width + 15) // 16, (height + 15) // 16
    xmin = wp.clamp(int(wp.floor((center[0] - extent[0]) / 16.0)), 0, gx)
    ymin = wp.clamp(int(wp.floor((center[1] - extent[1]) / 16.0)), 0, gy)
    xmax = wp.clamp(int(wp.ceil((center[0] + extent[0]) / 16.0)), 0, gx)
    ymax = wp.clamp(int(wp.ceil((center[1] + extent[1]) / 16.0)), 0, gy)
    return xmin, ymin, xmax - xmin, (xmax - xmin) * (ymax - ymin)


@wp.kernel(enable_backward=False)
def cull_first_32(
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec4), transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float), eval3d: bool, width: int, height: int,
    radii: wp.array(dtype=int), tiles: wp.array(dtype=int),
    queued: wp.array(dtype=int), queue_size: wp.array(dtype=int),
):
    i = wp.tid()
    if radii[i] == 0:
        return
    center = centers[i]
    xmin, ymin, rect_width, rect_count = _rect_lower(center, extents[i], width, height)
    opacity = float(0.0)
    if eval3d:
        opacity = opacity3d[i]
    else:
        opacity = conics[i][3]
    threshold = wp.log(opacity / (1.0 / 255.0))
    count = int(0)
    for tile_index in range(wp.min(rect_count, 32)):
        if _tile_contributes(tile_index, xmin, ymin, rect_width,
                             i, center, conics, transforms, eval3d, threshold):
            count += 1
    tiles[i] = count
    if rect_count > 32:
        slot = wp.atomic_add(queue_size, 0, 1)
        queued[slot] = i
    elif count == 0:
        radii[i] = 0


@wp.kernel(enable_backward=False)
def cull_remainder(
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec4), transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float), eval3d: bool, width: int, height: int,
    block_count: int,
    radii: wp.array(dtype=int), tiles: wp.array(dtype=int),
    queued: wp.array(dtype=int), queue_size: wp.array(dtype=int),
):
    block, lane = wp.tid()
    # Use a bounded grid: most scenes queue far fewer Gaussians than the
    # input count. Every lane in a block visits the same queued Gaussian.
    for slot in range(block, queue_size[0], block_count):
        i = queued[slot]
        center = centers[i]
        xmin, ymin, rect_width, rect_count = _rect_lower(center, extents[i], width, height)
        opacity = float(0.0)
        if eval3d:
            opacity = opacity3d[i]
        else:
            opacity = conics[i][3]
        threshold = wp.log(opacity / (1.0 / 255.0))
        local_count = int(0)
        for tile_index in range(32 + lane, rect_count, 32):
            if _tile_contributes(tile_index, xmin, ymin, rect_width,
                                 i, center, conics, transforms, eval3d, threshold):
                local_count += 1
        total = wp.tile_sum(wp.tile(local_count))[0]
        if lane == 0:
            count = tiles[i] + total
            tiles[i] = count
            if count == 0:
                radii[i] = 0


@wp.kernel(enable_backward=False)
def cull_remainder_3d(
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
    width: int, height: int, block_count: int,
    radii: wp.array(dtype=int), tiles: wp.array(dtype=int), valid: wp.array(dtype=wp.bool),
    queued: wp.array(dtype=int), queue_size: wp.array(dtype=int),
):
    block,lane = wp.tid()
    for slot in range(block,queue_size[0],block_count):
        i = queued[slot]
        xmin,ymin,rect_width,rect_count = _rect_lower(centers[i],extents[i],width,height)
        threshold = wp.log(opacity[i]/(1.0/255.0))
        local_count = int(0)
        for tile_index in range(32+lane,rect_count,32):
            x,y = xmin+tile_index%rect_width,ymin+tile_index//rect_width
            if frustum_minimum(wp.vec2(float(x*16),float(y*16)),
                               wp.vec2(float(x*16+15),float(y*16+15)),transforms[i]) <= threshold:
                local_count += 1
        total = wp.tile_sum(wp.tile(local_count))[0]
        if lane == 0:
            count = tiles[i]+total
            tiles[i] = count
            if count == 0:
                radii[i] = 0
                valid[i] = False
