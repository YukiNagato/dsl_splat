"""Python/Warp version of AAA's 3D HIER 64/8/4 renderer.

One 256-thread block owns a 16x16 tile. Each half warp shares a 64-slot
4x4 tail; each four-lane group shares an eight-slot 2x2 mid. Pixels keep
four-entry heads in registers. Queue state and algorithm are Python wp.funcs.
Only shared allocation, subgroup intrinsics, aligned loads and a rounded
reciprocal remain native. Shared queue access and ray math use Python/Warp.

Feature vectors and loops specialize at compile time for each channel count.
render_hierarchical_native remains the unchanged RGB baseline and default
for RGB. StopThePop algorithm: MIT, Graz UT, 2024.
"""

from .dispatch import FrameBindings, kernel_arg, launch
from .settings import GaussianRasterizationSettings
from .types import PreprocessedGaussians, TileBins, RenderOutput

from functools import cache
from typing import cast

import torch
import warp as wp

from .interop import current_stream, contiguous_features
from .render_adjoint import RenderAdjoint
from .render_warp_geometry import sample, frustum
from .render_warp_intrinsics import (
    shared_depths,
    shared_ids,
    sync,
    any_active,
    count_active,
    shuffle_int,
    shuffle_float,
    load_matrix,
)

INF = wp.constant(3.4028234663852886e38)
FULL = wp.constant(wp.uint32(0xFFFFFFFF))


@cache
def _pixel_type(channels: int) -> type:
    feature_type = wp.types.vector(channels, wp.float32)

    @wp.struct
    class Pixel:
        # Feature accumulators have a compile-time size; queue sizes stay fixed.
        active: bool
        transmittance: float
        color: feature_type
        contributors: int
        tail_count: int
        mid_count: int
        head_count: int
        head_depths: wp.vec4
        head_alphas: wp.vec4
        head_ids: wp.vec4i
        head_gaussians: wp.vec4
        x: float
        y: float
        gradient: feature_type
        final_color: feature_type
        final_transmittance: float
        background_gradient: float
        adjoint: RenderAdjoint

    return Pixel


@wp.struct
class Group:
    lane: int
    half_lane: int
    tail: int
    mid: int
    mid_rank: int
    x0: int
    y0: int
    x: int
    y: int
    half_mask: wp.uint32
    head_mask: wp.uint32


@wp.struct
class Queue:
    # A view into block-local shared arrays, not an allocated tensor.
    # Tail: [0, 1024), sixteen 64-entry runs.
    # Mid: [1024, 1536), sixty-four 8-entry runs.
    depths: wp.array(dtype=float)
    ids: wp.array(dtype=int)
    base: int


@wp.func
def tail_queue(depths: wp.array(dtype=float), ids: wp.array(dtype=int), group: int):
    q = Queue()
    q.depths = depths
    q.ids = ids
    q.base = group * 64
    return q


@wp.func
def mid_queue(depths: wp.array(dtype=float), ids: wp.array(dtype=int), group: int):
    q = Queue()
    q.depths = depths
    q.ids = ids
    q.base = 1024 + group * 8
    return q


@wp.func
def depth_at(q: Queue, index: int):
    return q.depths[q.base + index]


@wp.func
def id_at(q: Queue, index: int):
    return q.ids[q.base + index]


@wp.func
def store(q: Queue, index: int, depth: float, id: int):
    q.depths[q.base + index] = depth
    q.ids[q.base + index] = id


@cache
def _make_blend(channels: int) -> wp.Function:
    Pixel = _pixel_type(channels)
    feature_type = wp.types.vector(channels, wp.float32)

    @wp.func
    def blend_step(pixel: Pixel, colors: wp.array(dtype=feature_type)):
        p = pixel
        p.head_count -= 1
        if p.active:
            alpha = p.head_alphas[0]
            next_t = p.transmittance * (1.0 - alpha)
            if next_t < 0.0001:
                # CUDA does not blend the sample that triggers early termination.
                p.active = False
            else:
                rgb = colors[p.head_ids[0]]
                for ch in range(wp.static(channels)):
                    c = wp.static(ch)
                    p.color[c] = p.color[c] + rgb[c] * alpha * p.transmittance
                p.transmittance = next_t
                for index in range(1, 4):
                    j = wp.static(index)
                    p.head_depths[j - 1] = p.head_depths[j]
                    p.head_alphas[j - 1] = p.head_alphas[j]
                    p.head_ids[j - 1] = p.head_ids[j]
                p.head_depths[3] = INF
        return p

    return blend_step


@wp.func
def merge(
    q: Queue, n: int, mask: wp.uint32, rank: int, old_offset: int, depth: float, id: int
):
    """Merge two sorted n-entry runs; equal old entries precede new ones.

    Temporarily put the incoming run in old_offset, then scatter both runs
    by their binary-search ranks. Synchronization separates reads/writes.
    """
    new_pos = int(0)
    stride = n // 2
    while stride > 0:
        if depth_at(q, old_offset + new_pos + stride) <= depth:
            new_pos += stride
        stride //= 2
    new_pos += 1
    if depth_at(q, old_offset) > depth:
        new_pos = 0
    new_pos += rank
    old_depth = depth_at(q, old_offset + rank)
    q.depths[q.base + old_offset + rank] = depth
    sync(mask)
    old_pos = int(0)
    stride = n // 2
    while stride > 0:
        if depth_at(q, old_offset + old_pos + stride) < old_depth:
            old_pos += stride
        stride //= 2
    old_pos += 1
    if depth_at(q, old_offset) >= old_depth:
        old_pos = 0
    old_pos += rank
    old_id = id_at(q, old_offset + rank)
    sync(mask)
    store(q, new_pos, depth, id)
    store(q, old_pos, old_depth, old_id)
    sync(mask)


@cache
def _make_insert_head(channels: int) -> wp.Function:
    Pixel = _pixel_type(channels)

    @wp.func
    def insert_head(pixel: Pixel, depth: float, alpha: float, id: int, gaussian: float):
        p = pixel
        for slot in range(4):
            s = wp.static(slot)
            if depth < p.head_depths[s]:
                old_depth = float(p.head_depths[s])
                old_alpha = float(p.head_alphas[s])
                old_id = int(p.head_ids[s])
                old_gaussian = float(p.head_gaussians[s])
                p.head_depths[s] = depth
                p.head_alphas[s] = alpha
                p.head_ids[s] = id
                p.head_gaussians[s] = gaussian
                depth, alpha, id = old_depth, old_alpha, old_id
                gaussian = old_gaussian
        p.head_count += 1
        return p

    return insert_head


@wp.func
def rank_mid(group: Group, depth: float):
    rank = int(0)
    for index in range(1, 4):
        j = wp.static(index)
        other_rank = (group.mid_rank + j) % 4
        other = shuffle_float(group.head_mask, depth, other_rank, 4)
        if other < depth or (other == depth and other_rank < group.mid_rank):
            rank += 1
    return rank


def _make_stream_functions(
    blend_step: wp.Function, channels: int
) -> tuple[wp.Function, wp.Function]:
    Pixel = _pixel_type(channels)
    feature_type = wp.types.vector(channels, wp.float32)
    insert_head = _make_insert_head(channels)

    @wp.func
    def front_mid(
        pixel: Pixel,
        group: Group,
        mid: Queue,
        check_valid: bool,
        transforms: wp.array(dtype=wp.mat44),
        opacity: wp.array(dtype=float),
        colors: wp.array(dtype=feature_type),
    ):
        """Consume the front four mid entries, inserting samples into each head."""
        p = pixel
        if any_active(group.head_mask, p.active):
            load_id = id_at(mid, group.mid_rank)
            amplitude = float(0.0)
            if not check_valid or load_id != -1:
                amplitude = opacity[load_id]
            for inner in range(4):
                if p.head_count >= 4:
                    # Drain before checking this mid slot. Moving this below
                    # the validity/alpha tests changes CUDA's streaming order.
                    p = blend_step(p, colors)
                id = id_at(mid, inner)
                if check_valid and id == -1:
                    continue
                gid = shuffle_int(group.head_mask, load_id, inner, 4)
                weight = shuffle_float(group.head_mask, amplitude, inner, 4)
                if not p.active:
                    continue
                p.contributors += 1
                value = sample(
                    load_matrix(transforms, gid), float(group.x), float(group.y)
                )
                depth = value[1]
                if depth < -1.0 or depth > 1.0:
                    continue
                gaussian = wp.exp(value[0])
                alpha = wp.min(0.99, weight * gaussian)
                if alpha < 1.0 / 255.0:
                    continue
                # Strict comparison retains existing equal-depth head entries.
                p = insert_head(p, depth, alpha, id, gaussian)
        p.mid_count -= 4
        sync(group.half_mask)
        return p

    @wp.func
    def push_mid(
        pixel: Pixel,
        group: Group,
        tail: Queue,
        mid: Queue,
        check_valid: bool,
        transforms: wp.array(dtype=wp.mat44),
        opacity: wp.array(dtype=float),
        colors: wp.array(dtype=feature_type),
    ):
        """Move sixteen tail entries through the four 2x2 mid groups."""
        p = pixel
        load_id = id_at(tail, group.half_lane)
        for batch in range(4):
            if check_valid and p.tail_count == 0:
                break
            source = 4 * batch + group.mid_rank
            id = id_at(tail, source)
            gid = shuffle_int(group.half_mask, load_id, source, 16)
            depth = float(INF)
            if not check_valid or gid != -1:
                mid_group = group.half_lane // 4
                depth = sample(
                    load_matrix(transforms, gid),
                    float(group.x0 + 1 + 2 * (mid_group & 1)),
                    float(group.y0 + 1 + 2 * (mid_group // 2)),
                )[1]
            if check_valid and id == -1:
                depth = INF
            rank = rank_mid(group, depth)
            store(mid, rank, depth, id)
            sync(group.head_mask)
            depth, id = depth_at(mid, group.mid_rank), id_at(mid, group.mid_rank)
            p.mid_count += 4
            if batch != 0 or p.mid_count > 4:
                merge(mid, 4, group.head_mask, group.mid_rank, 4, depth, id)
                p = front_mid(p, group, mid, False, transforms, opacity, colors)
            else:
                store(mid, 4 + group.mid_rank, depth, id)
            if check_valid:
                p.tail_count -= wp.min(4, p.tail_count)
        if not check_valid:
            p.tail_count -= 16
        return p

    return front_mid, push_mid


@wp.func
def compare_exchange(q: Queue, a: int, b: int):
    da, db = depth_at(q, a), depth_at(q, b)
    if da > db:
        ia, ib = id_at(q, a), id_at(q, b)
        store(q, a, db, ib)
        store(q, b, da, ia)


@wp.func
def sort_batch(tail: Queue, group: Group):
    """CUDA's Batcher network on the 32 incoming tail entries.

    Preserve the network's tie order; a general tile_sort is not equivalent.
    Sixteen lanes each own one compare/exchange per network step.
    """
    size = int(2)
    while size <= 32:
        stride = size // 2
        offset = group.half_lane & (stride - 1)
        sync(group.half_mask)
        pos = 2 * group.half_lane - (group.half_lane & (stride - 1))
        compare_exchange(tail, 32 + pos, 32 + pos + stride)
        stride //= 2
        while stride > 0:
            sync(group.half_mask)
            pos = 2 * group.half_lane - (group.half_lane & (stride - 1))
            if offset >= stride:
                compare_exchange(tail, 32 + pos - stride, 32 + pos)
            stride //= 2
        size *= 2


def _make_evaluation(
    cull_4x4: bool,
    blend_step: wp.Function | None = None,
    backward: bool = False,
    channels: int = 3,
) -> wp.Function:
    Pixel = _pixel_type(channels)
    feature_type = wp.types.vector(channels, wp.float32)
    if blend_step is None:
        blend_step = _make_blend(channels)
    front_mid, push_mid = _make_stream_functions(blend_step, channels)

    # Keep streaming loops rolled. Unroll only the fixed vector accesses in
    # the helpers with wp.static: broad unrolling inflates register pressure.
    @wp.func
    def render(
        tid: int,
        width: int,
        height: int,
        ranges: wp.array(dtype=int),
        point_list: wp.array(dtype=int),
        transforms: wp.array(dtype=wp.mat44),
        opacity: wp.array(dtype=float),
        colors: wp.array(dtype=feature_type),
        bg: wp.array(dtype=float),
        output: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        adjoint: RenderAdjoint,
    ):
        cull = wp.static(cull_4x4)
        replay_backward = wp.static(backward)
        tile, local = tid // 256, tid % 256
        grid_width = (width + 15) // 16
        tile_x, tile_y = tile % grid_width * 16, tile // grid_width * 16
        group = Group()
        group.lane = local % 32
        group.half_lane = local % 16
        group.tail = local // 16
        group.mid = local // 4
        group.mid_rank = local % 4
        group.x0 = tile_x + 4 * (group.tail % 4)
        group.y0 = tile_y + 4 * (group.tail // 4)
        mid_group = group.half_lane // 4
        group.x = group.x0 + 2 * (mid_group & 1) + (group.mid_rank & 1)
        group.y = group.y0 + 2 * (mid_group // 2) + (group.mid_rank // 2)
        group.half_mask = wp.uint32(0xFFFF) << wp.uint32(group.lane & 16)
        group.head_mask = wp.uint32(0xF) << wp.uint32(group.lane & 28)
        p = Pixel()
        p.active = group.x < width and group.y < height
        p.transmittance = 1.0
        p.head_depths = wp.vec4(INF)
        p.head_ids = wp.vec4i(-1)
        if replay_backward:
            p.x = float(group.x)
            p.y = float(group.y)
            p.adjoint = adjoint
            if p.active:
                pixel_index = group.y * width + group.x
                p.final_transmittance = final_t[pixel_index]
                for channel in range(wp.static(channels)):
                    ch = wp.static(channel)
                    offset = ch * width * height + pixel_index
                    p.gradient[ch] = adjoint.pixel_gradient[offset]
                    p.final_color[ch] = output[offset] - p.final_transmittance * bg[ch]
                    p.background_gradient += bg[ch] * p.gradient[ch]
        depths = shared_depths()
        ids = shared_ids()
        tail, mid = tail_queue(depths, ids, group.tail), mid_queue(
            depths, ids, group.mid
        )
        first, last = int(0), int(0)
        if group.lane == 0:
            first = ranges[tile * 2]
            last = ranges[tile * 2 + 1]
        first = shuffle_int(FULL, first, 0, 32)
        last = shuffle_int(FULL, last, 0, 32)
        # Each warp loads 32 candidates for BOTH of its 4x4 regions, culls
        # and sorts them, then merges with the two retained tail halves.
        for progress in range(first, last, 32):
            if not any_active(FULL, p.active):
                break
            gid = int(-1)
            if progress + group.lane < last:
                gid = point_list[progress + group.lane]
            g = wp.mat44(0.0)
            amplitude = float(0.0)
            if gid != -1:
                g = load_matrix(transforms, gid)
                if cull:
                    amplitude = opacity[gid]
            culled = int(0)
            for half in range(2):
                other_group = (group.tail & ~1) + half
                if cull:
                    if gid != -1:
                        power = frustum(
                            g,
                            float(tile_x + 4 * (other_group & 3)),
                            float(group.y0),
                            3.0,
                            3.0,
                        )
                        if wp.min(0.99, amplitude * wp.exp(-power)) < 1.0 / 255.0:
                            culled |= 1 << half
            for half in range(2):
                other_group = (group.tail & ~1) + half
                depth = float(INF)
                if gid != -1 and (not cull or not (culled & (1 << half))):
                    depth = sample(
                        g,
                        float(tile_x + 4 * (other_group & 3) + 2),
                        float(group.y0 + 2),
                    )[1]
                id = gid
                if depth == INF:
                    id = -1
                store(tail_queue(depths, ids, other_group), 32 + group.lane, depth, id)
            sort_batch(tail, group)
            for half in range(2):
                q = tail_queue(depths, ids, (group.tail & ~1) + half)
                old_count = shuffle_int(FULL, p.tail_count, half * 16, 32)
                key, id = depth_at(q, 32 + group.lane), id_at(q, 32 + group.lane)
                added = count_active(FULL, id != -1)
                if half == group.lane // 16:
                    p.tail_count += added
                if old_count != 0:
                    merge(q, 32, FULL, group.lane, 0, key, id)
                else:
                    store(q, group.lane, key, id)
            for half in range(2):
                if p.tail_count > 32:
                    # Drain sixteen to mid/head, retaining at most 32 in tail.
                    p = push_mid(
                        p, group, tail, mid, False, transforms, opacity, colors
                    )
                    sync(group.half_mask)
                    for index in range(3 - half):
                        dest = group.half_lane + index * 16
                        store(
                            tail,
                            dest,
                            depth_at(tail, dest + 16),
                            id_at(tail, dest + 16),
                        )
                    sync(group.half_mask)
        if any_active(FULL, p.active):
            # End of tile stream: drain tail -> mid -> pixel heads. The valid
            # checks handle padded -1 entries in the final incoming batch.
            if p.tail_count != 0:
                for half in range(2):
                    p = push_mid(p, group, tail, mid, True, transforms, opacity, colors)
                    if half == 0 and p.tail_count == 0:
                        break
                    if half == 0:
                        store(
                            tail,
                            group.half_lane,
                            depth_at(tail, group.half_lane + 16),
                            id_at(tail, group.half_lane + 16),
                        )
            if any_active(FULL, p.active):
                if p.mid_count != 0:
                    store(
                        mid,
                        group.mid_rank,
                        depth_at(mid, 4 + group.mid_rank),
                        id_at(mid, 4 + group.mid_rank),
                    )
                    p = front_mid(p, group, mid, True, transforms, opacity, colors)
                while p.active and p.head_count != 0:
                    p = blend_step(p, colors)
        if not replay_backward and group.x < width and group.y < height:
            index = group.y * width + group.x
            for channel in range(wp.static(channels)):
                ch = wp.static(channel)
                output[ch * width * height + index] = (
                    p.color[ch] + p.transmittance * bg[ch]
                )
            final_t[index] = p.transmittance
            contributors[index] = p.contributors

    return render


@cache
def _make_kernel(cull_4x4: bool, channels: int = 3) -> wp.Kernel:
    feature_type = wp.types.vector(channels, wp.float32)
    evaluate = _make_evaluation(cull_4x4, channels=channels)

    @wp.kernel(
        module="unique",
        enable_backward=False,
        launch_bounds=256,
        module_options={"max_unroll": 0},
    )
    def render(
        width: int,
        height: int,
        ranges: wp.array(dtype=int),
        point_list: wp.array(dtype=int),
        transforms: wp.array(dtype=wp.mat44),
        opacity: wp.array(dtype=float),
        colors: wp.array(dtype=feature_type),
        bg: wp.array(dtype=float),
        output: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
    ):
        evaluate(
            wp.tid(),
            width,
            height,
            ranges,
            point_list,
            transforms,
            opacity,
            colors,
            bg,
            output,
            final_t,
            contributors,
            RenderAdjoint(),
        )

    return render


@torch.no_grad()
def render_hierarchical_3d(
    preprocessed: PreprocessedGaussians,
    bins: TileBins,
    raster_settings: GaussianRasterizationSettings,
    *,
    output: RenderOutput | None = None,
    _bindings: FrameBindings | None = None,
) -> RenderOutput:
    """Blend (N,C) features in one traversal using a cached C specialization.

    Output reuse is explicitly opt-in; ``color`` has shape (C,H,W).
    """
    wp.init()
    settings = raster_settings.settings
    if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
        raise ValueError("requires eval_3D=True and sort_mode=HIER")
    sizes = settings.sort_settings.queue_sizes
    if (sizes.tile_4x4, sizes.tile_2x2, sizes.per_pixel) != (64, 8, 4):
        raise ValueError("only queue sizes 64/8/4 are supported")
    if raster_settings.render_depth:
        raise ValueError("render_depth is not supported")
    if "gauss2screen" not in preprocessed or "opacity" not in preprocessed:
        raise ValueError("preprocessed data must contain 3D transforms and opacity")
    width, height = raster_settings.image_width, raster_settings.image_height
    device = preprocessed["gauss2screen"].device
    channels = preprocessed["rgb"].shape[1]
    if channels <= 0:
        raise ValueError("features must have at least one channel")
    if (
        raster_settings.bg.shape != (channels,)
        or raster_settings.bg.device != device
        or raster_settings.bg.dtype != torch.float32
    ):
        raise ValueError("bg must be float32 (C,) on the rendering device")
    feature_type = wp.types.vector(channels, wp.float32)
    layouts = {
        "color": ((channels, height, width), torch.float32),
        "final_T": ((height, width), torch.float32),
        "contributors": ((height, width), torch.int32),
    }
    if output is None:
        output = cast(
            RenderOutput,
            {
                name: torch.empty(shape, dtype=dtype, device=device)
                for name, (shape, dtype) in layouts.items()
            },
        )
    else:
        for name, (shape, dtype) in layouts.items():
            value = output.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.device != device
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or not value.is_contiguous()
                or value.requires_grad
            ):
                raise ValueError(
                    f"output[{name!r}] must be contiguous {dtype} with shape {shape} on {device}, without gradients"
                )
    transforms = preprocessed["gauss2screen"].contiguous()
    if transforms.data_ptr() % 16:
        transforms = transforms.clone()
    kernel = _make_kernel(
        bool(settings.culling_settings.hierarchical_4x4_culling), channels
    )
    launch(
        kernel,
        dim=256 * ((width + 15) // 16) * ((height + 15) // 16),
        inputs=[
            width,
            height,
            kernel_arg(bins["ranges"].contiguous().view(-1), wp.int32, _bindings),
            kernel_arg(bins["point_list"].contiguous(), wp.int32, _bindings),
            kernel_arg(transforms, wp.mat44, _bindings),
            kernel_arg(preprocessed["opacity"].contiguous(), wp.float32, _bindings),
            kernel_arg(
                contiguous_features(preprocessed["rgb"]), feature_type, _bindings
            ),
            kernel_arg(raster_settings.bg.contiguous().view(-1), wp.float32, _bindings),
        ],
        outputs=[
            kernel_arg(output["color"].view(-1), wp.float32, _bindings),
            kernel_arg(output["final_T"].view(-1), wp.float32, _bindings),
            kernel_arg(output["contributors"].view(-1), wp.int32, _bindings),
        ],
        block_dim=256,
        stream=current_stream(device, bindings=_bindings),
        bindings=_bindings,
    )
    return output
