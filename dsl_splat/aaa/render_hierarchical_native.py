"""CUDA-organized 3D hierarchical forward, compiled automatically by Warp.

Port of StopThePop hierarchical_render.cuh (MIT, Graz University of
Technology, 2024). One 256-thread block per 16x16 tile: paired 4x4 tails per
warp, shared 8-entry mids per four lanes and register heads per pixel.
Native CUDA is embedded in Python; no reference headers or extension build
are required at runtime. Forward only, default queues 64/8/4.
"""
import torch
import warp as wp

from .interop import current_stream
from .dispatch import kernel_arg, launch
from .native_geometry import GEOMETRY


BODY = r"""
    constexpr float INF = 3.4028234663852886e38f;
    constexpr unsigned FULL = 0xffffffffu;
    constexpr unsigned HEAD = 0x1000000u, MID = 0x10000u;
    const int lane = threadIdx.x & 31, half_lane = threadIdx.x & 15;
    const int group = threadIdx.x >> 4, group_x = group & 3, group_y = group >> 2;
    const int mid_group = half_lane >> 2, mid_rank = half_lane & 3;
    const unsigned half_mask = 0xffffu << (lane & 16);
    const unsigned head_mask = 0xfu << (lane & 28);
    const int grid_width = (width+15)/16;
    const int tile_x = tile%grid_width*16, tile_y = tile/grid_width*16;
    const int x0 = tile_x+4*group_x, y0 = tile_y+4*group_y;
    const int ix = x0+2*(mid_group&1)+(mid_rank&1);
    const int iy = y0+2*(mid_group>>1)+(mid_rank>>1);
    const int mg = group*4+mid_group;
    bool active = ix < width && iy < height;
    float T = 1.0f, C[3] = {0.0f, 0.0f, 0.0f};
    int contributor = 0;
    unsigned fill = 0;
    float head_depths[4] = {INF, INF, INF, INF};
    float head_alphas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int head_ids[4] = {-1, -1, -1, -1};
    __shared__ float tail_depths[16][64], mid_depths[64][8];
    __shared__ int tail_ids[16][64], mid_ids[64][8];
    // Match auxiliary.h::loadMatrix4x4's aligned float4 loads. A plain Warp
    // matrix array access has only float alignment in its C++ type and can
    // otherwise emit sixteen scalar loads for the same 64-byte transform.
    const float4* __restrict__ matrix_data = reinterpret_cast<const float4*>(transforms.data);
    auto load_matrix = [&](int id) {
        M4 matrix;
        #pragma unroll
        for (int i=0; i<4; ++i) {
            float4 row = matrix_data[id*4+i];
            matrix.data[i][0] = row.x; matrix.data[i][1] = row.y;
            matrix.data[i][2] = row.z; matrix.data[i][3] = row.w;
        }
        return matrix;
    };
    // Every warp loads the same range and broadcasts within its own lanes.
    // Avoid the reference's identical concurrent writes to shared range.
    int first = lane == 0 ? ranges[tile*2] : 0;
    int last = lane == 0 ? ranges[tile*2+1] : 0;
    first = __shfl_sync(FULL, first, 0);
    last = __shfl_sync(FULL, last, 0);

    // The reference merge: new entries go after equal old entries. The
    // initial local sorting network supplies its own ordering of depth ties.
    auto merge = [&](int n, unsigned mask, int rank, float* keys, int* ids,
                     float* final_keys, int* final_ids, float key, int id) {
        int s0 = 0;
        #pragma unroll
        for (int stride=n/2; stride>0; stride/=2)
            if (keys[s0+stride] <= key) s0 += stride;
        s0 += 1;
        if (keys[0] > key) s0 = 0;
        s0 += rank;
        float old_key = keys[rank];
        keys[rank] = key;
        __syncwarp(mask);
        int s1 = 0;
        #pragma unroll
        for (int stride=n/2; stride>0; stride/=2)
            if (keys[s1+stride] < old_key) s1 += stride;
        s1 += 1;
        if (keys[0] >= old_key) s1 = 0;
        s1 += rank;
        int old_id = ids[rank];
        __syncwarp(mask);
        final_keys[s0] = key;
        final_keys[s1] = old_key;
        final_ids[s0] = id;
        final_ids[s1] = old_id;
        __syncwarp(mask);
    };
    auto blend = [&]() {
        fill -= HEAD;
        if (!active) return;
        const float alpha = head_alphas[0];
        float next_T = T*(1.0f-alpha);
        if (next_T < 0.0001f) { active = false; return; }
        auto color = colors[head_ids[0]];
        #pragma unroll
        for (int ch=0; ch<3; ++ch) C[ch] += color[ch]*alpha*T;
        T = next_T;
        #pragma unroll
        for (int j=1; j<4; ++j) {
            head_depths[j-1] = head_depths[j];
            head_alphas[j-1] = head_alphas[j];
            head_ids[j-1] = head_ids[j];
        }
        head_depths[3] = INF;
    };
    auto front_mid = [&](bool check_valid) {
        if (__any_sync(head_mask, active)) {
            int load_id = mid_ids[mg][mid_rank];
            float load_opacity = (!check_valid || load_id != -1) ? opacity[load_id] : 0.0f;
            // Keep nested streaming loops as in CUDA. Forcing their full
            // unrolling duplicates the ray evaluation and queue code and
            // substantially slows the generated renderer.
            for (int inner=0; inner<4; ++inner) {
                if (fill >= 4*HEAD) blend();
                int id = mid_ids[mg][inner];
                if (check_valid && id == -1) continue;
                int gid = __shfl_sync(head_mask, load_id, inner, 4);
                float amplitude = __shfl_sync(head_mask, load_opacity, inner, 4);
                if (!active) continue;
                contributor++;
                auto value = sample(load_matrix(gid), static_cast<float>(ix), static_cast<float>(iy));
                float depth = value[1];
                if (depth < -1.0f || depth > 1.0f) continue;
                float alpha = fminf(0.99f, amplitude*expf(value[0]));
                if (alpha < 1.0f/255.0f) continue;
                #pragma unroll
                for (int s=0; s<4; ++s) {
                    if (depth < head_depths[s]) {
                        float old_depth = head_depths[s], old_alpha = head_alphas[s];
                        int old_id = head_ids[s];
                        head_depths[s] = depth; head_alphas[s] = alpha; head_ids[s] = id;
                        depth = old_depth; alpha = old_alpha; id = old_id;
                    }
                }
                fill += HEAD;
            }
        }
        fill -= 4*MID;
        __syncwarp(half_mask);
    };
    auto push_mid = [&](bool check_valid) {
        int load_id = tail_ids[group][half_lane];
        for (int m=0; m<4; ++m) {
            if (check_valid && (fill & 0xffffu) == 0) break;
            const int source = 4*m+mid_rank;
            int id = tail_ids[group][source];
            int gid = __shfl_sync(half_mask, load_id, source, 16);
            float depth = INF;
            if (!check_valid || gid != -1)
                depth = sample(load_matrix(gid), static_cast<float>(x0+1+2*(mid_group&1)),
                               static_cast<float>(y0+1+2*(mid_group>>1)))[1];
            if (check_valid && id == -1) depth = INF;
            int rank = 0;
            #pragma unroll
            for (int j=1; j<4; ++j) {
                int other_rank = (mid_rank+j)%4;
                float other = __shfl_sync(head_mask, depth, other_rank, 4);
                if (other < depth || (other == depth && other_rank < mid_rank)) rank++;
            }
            mid_depths[mg][rank] = depth; mid_ids[mg][rank] = id;
            __syncwarp(head_mask);
            depth = mid_depths[mg][mid_rank]; id = mid_ids[mg][mid_rank];
            fill += 4*MID;
            if (m != 0 || (fill & 0xff0000u) > 4*MID) {
                merge(4, head_mask, mid_rank, mid_depths[mg]+4, mid_ids[mg]+4,
                      mid_depths[mg], mid_ids[mg], depth, id);
                front_mid(false);
            } else {
                mid_depths[mg][4+mid_rank] = depth; mid_ids[mg][4+mid_rank] = id;
            }
            if (check_valid) fill -= min(4u, fill & 0xffffu);
        }
        if (!check_valid) fill -= 16;
    };

    for (int progress=first; progress<last; progress+=32) {
        if (!__any_sync(FULL, active)) break;
        int gid = progress+lane < last ? point_list[progress+lane] : -1;
        M4 g;
        float amplitude = 0.0f;
        if (gid != -1) {
            g = load_matrix(gid);
            if (CULL_ALPHA) amplitude = opacity[gid];
        }
        unsigned culled = 0;
        for (int half=0; half<2; ++half) {
            int other_group = (group & ~1) + half;
            int other_x = tile_x+4*(other_group&3);
            if (gid != -1 && CULL_ALPHA) {
                float power = frustum(g, static_cast<float>(other_x), static_cast<float>(y0), 3.0f, 3.0f);
                if (fminf(0.99f, amplitude*expf(-power)) < 1.0f/255.0f) culled |= 1u<<half;
            }
        }
        for (int half=0; half<2; ++half) {
            int other_group = (group & ~1) + half;
            float depth = INF;
            if (gid != -1 && (!CULL_ALPHA || !(culled & (1u<<half))))
                depth = sample(g, static_cast<float>(tile_x+4*(other_group&3)+2), static_cast<float>(y0+2))[1];
            tail_depths[other_group][32+lane] = depth;
            tail_ids[other_group][32+lane] = depth == INF ? -1 : gid;
        }
        // Original 32-key batcher network, executed by the 16 half-warp lanes.
        for (unsigned size=2; size<=32; size*=2) {
            unsigned stride = size/2, offset = half_lane & (stride-1);
            __syncwarp(half_mask);
            unsigned pos = 2*half_lane-(half_lane & (stride-1));
            if (tail_depths[group][32+pos] > tail_depths[group][32+pos+stride]) {
                float d = tail_depths[group][32+pos]; int id = tail_ids[group][32+pos];
                tail_depths[group][32+pos] = tail_depths[group][32+pos+stride];
                tail_ids[group][32+pos] = tail_ids[group][32+pos+stride];
                tail_depths[group][32+pos+stride] = d; tail_ids[group][32+pos+stride] = id;
            }
            for (stride/=2; stride>0; stride/=2) {
                __syncwarp(half_mask);
                pos = 2*half_lane-(half_lane & (stride-1));
                if (offset >= stride && tail_depths[group][32+pos-stride] > tail_depths[group][32+pos]) {
                    float d = tail_depths[group][32+pos-stride]; int id = tail_ids[group][32+pos-stride];
                    tail_depths[group][32+pos-stride] = tail_depths[group][32+pos];
                    tail_ids[group][32+pos-stride] = tail_ids[group][32+pos];
                    tail_depths[group][32+pos] = d; tail_ids[group][32+pos] = id;
                }
            }
        }
        for (int half=0; half<2; ++half) {
            int other_group = (group & ~1) + half;
            float* keys = tail_depths[other_group]; int* ids = tail_ids[other_group];
            if ((__shfl_sync(FULL, fill, half*16) & 0xffffu) != 0) {
                float key = keys[32+lane]; int id = ids[32+lane];
                unsigned valid = __popc(__ballot_sync(FULL, id != -1));
                if (half == lane/16) fill += valid;
                merge(32, FULL, lane, keys, ids, keys, ids, key, id);
            } else {
                keys[lane] = keys[32+lane]; int id = ids[32+lane]; ids[lane] = id;
                unsigned valid = __popc(__ballot_sync(FULL, id != -1));
                if (half == lane/16) fill += valid;
            }
        }
        for (int half=0; half<2; ++half) {
            if ((fill & 0xffffu) > 32) {
                push_mid(false);
                __syncwarp(half_mask);
                for (int i=0; i<3-half; ++i) {
                    tail_ids[group][half_lane+i*16] = tail_ids[group][half_lane+(i+1)*16];
                    tail_depths[group][half_lane+i*16] = tail_depths[group][half_lane+(i+1)*16];
                }
                __syncwarp(half_mask);
            }
        }
    }
    if (__any_sync(FULL, active)) {
        if ((fill & 0xffffu) != 0) {
            for (int half=0; half<2; ++half) {
                push_mid(true);
                if (half == 0 && (fill & 0xffffu) == 0) break;
                if (half == 0) {
                    tail_ids[group][half_lane] = tail_ids[group][half_lane+16];
                    tail_depths[group][half_lane] = tail_depths[group][half_lane+16];
                }
            }
        }
        if (__any_sync(FULL, active)) {
            if ((fill & 0xff0000u) != 0) {
                mid_ids[mg][mid_rank] = mid_ids[mg][4+mid_rank];
                mid_depths[mg][mid_rank] = mid_depths[mg][4+mid_rank];
                front_mid(true);
            }
            while (active && fill != 0) blend();
        }
    }
    if (ix < width && iy < height) {
        const int p = iy*width+ix;
        #pragma unroll
        for (int ch=0; ch<3; ++ch) output[ch*width*height+p] = C[ch]+T*bg[ch];
        final_t[p] = T; contributors[p] = contributor;
    }
"""


@wp.func_native(GEOMETRY + 'constexpr bool CULL_ALPHA = true;\n' + BODY)
def _evaluate_cull(tile: int, width: int, height: int,
                   ranges: wp.array(dtype=int), point_list: wp.array(dtype=int),
                   transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
                   colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=float),
                   output: wp.array(dtype=float), final_t: wp.array(dtype=float),
                   contributors: wp.array(dtype=int)):
    pass


@wp.func_native(GEOMETRY + 'constexpr bool CULL_ALPHA = false;\n' + BODY)
def _evaluate_unculled(tile: int, width: int, height: int,
                       ranges: wp.array(dtype=int), point_list: wp.array(dtype=int),
                       transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
                       colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=float),
                       output: wp.array(dtype=float), final_t: wp.array(dtype=float),
                       contributors: wp.array(dtype=int)):
    pass


@wp.kernel(module='unique', enable_backward=False, launch_bounds=256)
def _render_cull(width: int, height: int,
                  ranges: wp.array(dtype=int), point_list: wp.array(dtype=int),
                  transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
                  colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=float),
                  output: wp.array(dtype=float), final_t: wp.array(dtype=float),
                  contributors: wp.array(dtype=int)):
    _evaluate_cull(wp.tid()//256, width, height, ranges, point_list, transforms,
                   opacity, colors, bg, output, final_t, contributors)


@wp.kernel(module='unique', enable_backward=False, launch_bounds=256)
def _render_unculled(width: int, height: int,
                      ranges: wp.array(dtype=int), point_list: wp.array(dtype=int),
                      transforms: wp.array(dtype=wp.mat44), opacity: wp.array(dtype=float),
                      colors: wp.array(dtype=wp.vec3), bg: wp.array(dtype=float),
                      output: wp.array(dtype=float), final_t: wp.array(dtype=float),
                      contributors: wp.array(dtype=int)):
    _evaluate_unculled(wp.tid()//256, width, height, ranges, point_list, transforms,
                       opacity, colors, bg, output, final_t, contributors)


@torch.no_grad()
def render_hierarchical_3d(preprocessed, bins, raster_settings, *, output=None,
                           share_mid=False, cooperative_tail=True, _bindings=None):
    """Render a fresh frame on the current Torch stream; storage may vary.

    ``output=`` is an explicit opt-in to reusable buffers. Diagnostic staged
    options retain their previous meaning; production uses shared native queues.
    """
    if share_mid or not cooperative_tail:
        from .render_hierarchical_shared import render_hierarchical_3d as staged
        if output is not None:
            raise ValueError('reusable output buffers require the native renderer')
        return staged(preprocessed, bins, raster_settings, share_mid=share_mid,
                       cooperative_tail=cooperative_tail)
    wp.init()
    settings = raster_settings.settings
    if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
        raise ValueError('requires eval_3D=True and sort_mode=HIER')
    sizes = settings.sort_settings.queue_sizes
    if (sizes.tile_4x4, sizes.tile_2x2, sizes.per_pixel) != (64, 8, 4):
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
    kernel = _render_cull if settings.culling_settings.hierarchical_4x4_culling else _render_unculled
    transforms = preprocessed['gauss2screen'].contiguous()
    if transforms.data_ptr() % 16:
        # A contiguous view can still have a nonaligned storage offset.
        transforms = transforms.clone()
    launch(kernel, dim=256*((width+15)//16)*((height+15)//16),
              inputs=[width, height, kernel_arg(bins['ranges'].contiguous().view(-1),wp.int32,_bindings),
                      kernel_arg(bins['point_list'].contiguous(),wp.int32,_bindings),
                      kernel_arg(transforms,wp.mat44,_bindings),
                      kernel_arg(preprocessed['opacity'].contiguous(),wp.float32,_bindings),
                      kernel_arg(preprocessed['rgb'].contiguous(),wp.vec3,_bindings),
                      kernel_arg(raster_settings.bg.contiguous().view(-1),wp.float32,_bindings)],
              outputs=[kernel_arg(output['color'].view(-1),wp.float32,_bindings),
                       kernel_arg(output['final_T'].view(-1),wp.float32,_bindings),
                       kernel_arg(output['contributors'].view(-1),wp.int32,_bindings)],
              block_dim=256, stream=current_stream(device),bindings=_bindings)
    return output
