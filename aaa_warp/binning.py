"""Gaussian/tile duplication, radix sorting, and tile ranges for AAA forward.

The sort key stores tile ID in the high 32 bits and the raw float32 depth bits
in the low 32 bits, matching CUDA's ``constructSortKey``.
"""
import torch
import warp as wp
from .interop import current_stream
from .dispatch import kernel_arg, launch

from .geometry import load_camera
from .native_geometry import frustum_depth_native
from .binning_native import duplicate_aaa_default as _duplicate_aaa_default, duplicate_dim


@wp.func_native('return __float_as_uint(value);')
def _float_bits(value: wp.float32) -> wp.uint32:
    ...


@wp.func
def _sort_key(tile_id: int, depth: float):
    return (wp.uint64(wp.uint32(tile_id)) << wp.uint64(32)) | wp.uint64(_float_bits(depth))


@wp.func
def _tile_bounds(center: wp.vec2, extent: wp.vec2, gx: int, gy: int):
    xmin = wp.clamp(int(wp.floor((center[0]-extent[0])/16.0)), 0, gx)
    ymin = wp.clamp(int(wp.floor((center[1]-extent[1])/16.0)), 0, gy)
    xmax = wp.clamp(int(wp.ceil((center[0]+extent[0])/16.0)), 0, gx)
    ymax = wp.clamp(int(wp.ceil((center[1]+extent[1])/16.0)), 0, gy)
    return xmin, ymin, xmax, ymax


@wp.func
def _tile_minimum_with_pos(co: wp.vec4, mean: wp.vec2, lo: wp.vec2, hi: wp.vec2):
    xdiff, ydiff = lo[0]-mean[0], lo[1]-mean[1]
    left, above = float(xdiff>0.0), float(ydiff>0.0)
    notx = left+float(mean[0]>hi[0])
    noty = above+float(mean[1]>hi[1])
    pos = mean
    result = float(0.0)
    if notx+noty > 0.0:
        px = left*lo[0]+(1.0-left)*hi[0]
        py = above*lo[1]+(1.0-above)*hi[1]
        dx,dy = wp.copysign(15.0,xdiff),wp.copysign(15.0,ydiff)
        diffx,diffy = mean[0]-px,mean[1]-py
        tx = noty*wp.clamp((dx*co[0]*diffx+dx*co[1]*diffy)*(1.0/(225.0*co[0])),0.0,1.0)
        ty = notx*wp.clamp((dy*co[1]*diffx+dy*co[2]*diffy)*(1.0/(225.0*co[2])),0.0,1.0)
        pos = wp.vec2(px+tx*dx, py+ty*dy)
        mx,my = mean[0]-pos[0],mean[1]-pos[1]
        result = 0.5*(co[0]*mx*mx+co[2]*my*my)+co[1]*mx*my
    return result, pos


@wp.func
def _frustum_minimum_with_depth(lo: wp.vec2, hi: wp.vec2, g2s: wp.mat44, sequential: bool = False):
    result = frustum_depth_native(g2s,lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1],sequential)
    return result[0],result[1]


@wp.func
def _ray_depth(pos: wp.vec2, width: int, height: int,
               inv_vp: wp.mat44, camera: wp.vec3,
               inverse: wp.array3d(dtype=float), i: int):
    ndc = wp.vec2(pos[0]*(2.0/float(width))-1.0,
                  pos[1]*(2.0/float(height))-1.0)
    world = inv_vp*wp.vec4(ndc[0],ndc[1],0.0,1.0)
    direction = wp.normalize(wp.vec3(world[0],world[1],world[2])*(1.0/world[3])-camera)
    x,y,z = direction[0],direction[1],direction[2]
    a,b,c = inverse[i,0,0],inverse[i,0,1],inverse[i,0,2]
    d,e,f = inverse[i,1,0],inverse[i,1,1],inverse[i,1,2]
    num = inverse[i,2,0]*x+inverse[i,2,1]*y+inverse[i,2,2]*z
    den = (a*x+b*y+c*z)*x+(b*x+d*y+e*z)*y+(c*x+e*y+f*z)*z
    return wp.max(0.0,num/wp.max(0.00001,den)+8.0)


@wp.kernel(enable_backward=False)
def _duplicate(
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float), radii: wp.array(dtype=int),
    conics: wp.array(dtype=wp.vec4), transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float), inverse: wp.array3d(dtype=float),
    inverse_vp: wp.array2d(dtype=float), camera: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int), width: int, height: int,
    eval3d: bool, tile_culling: bool, load_balancing: bool, order: int,
    keys: wp.array(dtype=wp.uint64), values: wp.array(dtype=int),
):
    i = wp.tid()
    if radii[i] <= 0:
        return
    gx,gy = (width+15)//16,(height+15)//16
    xmin,ymin,xmax,ymax = _tile_bounds(centers[i],extents[i],gx,gy)
    off = int(0)
    if i > 0:
        off = offsets[i-1]
    stop = offsets[i]
    simple = not tile_culling and not load_balancing and order < 2
    threshold = float(0.0)
    co = wp.vec4(0.0)
    transform = wp.mat44(0.0)
    if tile_culling or order >= 2:
        if eval3d:
            threshold = wp.log(opacity3d[i]/(1.0/255.0))
            transform = transforms[i]
        else:
            co = conics[i]
            threshold = wp.log(co[3]/(1.0/255.0))
    inv_vp = wp.mat44(0.0)
    cam = wp.vec3(0.0)
    if order >= 2 and not eval3d:
        inv_vp = load_camera(inverse_vp)
        cam = camera[0]
    for y in range(ymin,ymax):
        for x in range(xmin,xmax):
            lo = wp.vec2(float(x*16),float(y*16))
            hi = lo+wp.vec2(15.0)
            power = float(0.0)
            max_pos = centers[i]
            depth = depths[i]
            if not simple:
                if eval3d:
                    if order >= 2:
                        power,tile_depth = _frustum_minimum_with_depth(lo,hi,transform,not load_balancing or (y-ymin)*(xmax-xmin)+x-xmin < 32)
                        depth = tile_depth+8.0
                    else:
                        if tile_culling:
                            power = frustum_depth_native(transform,lo[0],lo[1],15.0,15.0,
                                not load_balancing or (y-ymin)*(xmax-xmin)+x-xmin < 32)[0]
                        depth += 8.0
                else:
                    if tile_culling or order == 3:
                        power,max_pos = _tile_minimum_with_pos(co,centers[i],lo,hi)
                    if order >= 2:
                        target = lo+wp.vec2(7.5)
                        if order == 3:
                            target = max_pos
                        depth = _ray_depth(target,width,height,inv_vp,cam,inverse,i)
                    else:
                        depth += 8.0
            if not tile_culling or power <= threshold:
                if off < stop and off < keys.shape[0]:
                    keys[off] = _sort_key(y*gx+x, depth)
                    values[off] = i
                off += 1
    # CUDA reserves according to preprocess and pads any shortfall with an
    # invalid tile key. This can happen at contribution threshold boundaries.
    for j in range(off,wp.min(stop,keys.shape[0])):
        keys[j] = _sort_key(-1, 3.4028234663852886e38)
        values[j] = -1



def _uses_aaa_default_duplicate(settings):
    # Only flags affecting this stage select its specialization. EWA, bounds,
    # HIER queues and 4x4 culling are still handled by their respective stages.
    return (bool(settings.eval_3D) and bool(settings.load_balancing)
            and bool(settings.culling_settings.tile_based_culling)
            and int(settings.sort_settings.sort_order) == 3)


def _duplicate_launch(preprocessed, raster_settings, offsets, bindings=None):
    if _uses_aaa_default_duplicate(raster_settings.settings):
        inputs = [kernel_arg(preprocessed['means2D'],wp.vec2,bindings),
                  kernel_arg(preprocessed['rects2D'],wp.vec2,bindings),
                  kernel_arg(preprocessed['radii'],wp.int32,bindings),
                  kernel_arg(preprocessed['gauss2screen'],wp.mat44,bindings),
                  kernel_arg(preprocessed['opacity'],wp.float32,bindings),
                  kernel_arg(offsets,wp.int32,bindings),
                  raster_settings.image_width,raster_settings.image_height]
        return _duplicate_aaa_default, inputs, 32
    return _duplicate, _duplicate_inputs(preprocessed,raster_settings,offsets,bindings), 256


@wp.kernel(enable_backward=False)
def _identify_ranges(keys: wp.array(dtype=wp.uint64), ranges: wp.array2d(dtype=int)):
    i = wp.tid()
    current = wp.uint32(keys[i] >> wp.uint64(32))
    if i == 0:
        if current != wp.uint32(0xffffffff):
            ranges[int(current),0] = 0
    else:
        previous = wp.uint32(keys[i-1] >> wp.uint64(32))
        if current != previous:
            if previous != wp.uint32(0xffffffff):
                ranges[int(previous),1] = i
            if current != wp.uint32(0xffffffff):
                ranges[int(current),0] = i
    if i == keys.shape[0]-1 and current != wp.uint32(0xffffffff):
        ranges[int(current),1] = keys.shape[0]


def _duplicate_inputs(preprocessed, raster_settings, offsets, bindings=None):
    device = preprocessed['radii'].device
    def field(name, dtype, shape):
        value = preprocessed.get(name)
        if value is None:
            value = torch.empty(shape, dtype=torch.float32, device=device)
        return kernel_arg(value,dtype,bindings)
    settings = raster_settings.settings
    return [field('means2D',wp.vec2,(0,2)),field('rects2D',wp.vec2,(0,2)),
            field('depths',wp.float32,(0,)),field('radii',wp.int32,(0,)),
            field('conic_opacity',wp.vec4,(0,4)),field('gauss2screen',wp.mat44,(0,4,4)),
            field('opacity',wp.float32,(0,)),field('cov3D_inv',wp.float32,(0,3,4)),
            kernel_arg(raster_settings.inv_viewprojmatrix,wp.float32,bindings),
            kernel_arg(raster_settings.campos.reshape(1,3),wp.vec3,bindings),
            kernel_arg(offsets,bindings=bindings),raster_settings.image_width,raster_settings.image_height,
            bool(settings.eval_3D),bool(settings.culling_settings.tile_based_culling),
            bool(settings.load_balancing),int(settings.sort_settings.sort_order)]


@wp.kernel(enable_backward=False)
def _initialize_fixed(offsets: wp.array(dtype=int), n: int,
                       keys: wp.array(dtype=wp.uint64), values: wp.array(dtype=int),
                       ranges: wp.array2d(dtype=int), pair_count: wp.array(dtype=int)):
    i = wp.tid()
    if i < keys.shape[0]:
        keys[i] = wp.uint64(-1)
        values[i] = -1
    if i < ranges.shape[0]:
        ranges[i,0] = 0
        ranges[i,1] = 0
    if i == 0:
        total = int(0)
        if n:
            total = offsets[n-1]
        pair_count[0] = total


class _FixedBinning:
    """Capture-compatible storage; callers must check pair_count for overflow.

    Writes are bounded even on overflow. Padded keys sort after every valid
    tile. The renderer sees only ranges for the materialized prefix until
    its caller grows this workspace and reruns the complete forward.
    """
    def __init__(self, n, raster_settings, device, capacity):
        self.capacity = capacity
        self.n = n
        tile_count = ((raster_settings.image_width+15)//16) * ((raster_settings.image_height+15)//16)
        self.offsets = torch.empty(n, dtype=torch.int32, device=device)
        self.keys = torch.empty(2*capacity, dtype=torch.int64, device=device)
        self.values = torch.empty(2*capacity, dtype=torch.int32, device=device)
        self.ranges = torch.empty((tile_count,2), dtype=torch.int32, device=device)
        self.pair_count = torch.empty(1, dtype=torch.int32, device=device)
        self.result = {'point_list': self.values[:capacity], 'ranges': self.ranges}
        self._offsets = wp.from_torch(self.offsets)
        self._keys = wp.from_torch(self.keys, dtype=wp.uint64)
        self._values = wp.from_torch(self.values)
        self._prefix_keys = wp.from_torch(self.keys[:capacity], dtype=wp.uint64)
        self._prefix_values = wp.from_torch(self.values[:capacity])
        self._ranges = wp.from_torch(self.ranges)
        self._count = wp.from_torch(self.pair_count)
        self._end_bit = 32+tile_count.bit_length()

    def launch(self, preprocessed, raster_settings):
        stream = current_stream(self.offsets.device)
        with wp.ScopedStream(stream, sync_enter=False):
            if self.n:
                wp.utils.array_scan(wp.from_torch(preprocessed['tiles_touched']), self._offsets)
            wp.launch(_initialize_fixed, dim=max(1,self.capacity,self.ranges.shape[0]),
                      inputs=[self._offsets,self.n],
                      outputs=[self._prefix_keys,self._prefix_values,self._ranges,self._count],
                      stream=stream)
            if self.capacity:
                if self.n:
                    kernel, inputs, block_dim = _duplicate_launch(preprocessed,raster_settings,self.offsets)
                    wp.launch(kernel, dim=duplicate_dim(kernel,self.n), inputs=inputs, block_dim=block_dim,
                              outputs=[self._prefix_keys,self._prefix_values],stream=stream)
                wp.utils.radix_sort_pairs(self._keys,self._values,self.capacity,end_bit=self._end_bit)
                wp.launch(_identify_ranges,dim=self.capacity,inputs=[self._prefix_keys],
                          outputs=[self._ranges],stream=stream)
        return self.result


@torch.no_grad()
def bin_and_sort(preprocessed, raster_settings, *, diagnostics=True, _bindings=None):
    """Build AAA's duplicated Gaussian list, sorted keys, and tile ranges.

    The call synchronizes once to obtain the total number of Gaussian/tile
    pairs, as the CUDA reference does before allocating its binning buffers.
    Set ``diagnostics=False`` when only sorted indices and tile ranges are
    needed for rendering; this avoids two device copies of unsorted pairs.
    """
    wp.init()
    n = preprocessed['radii'].shape[0]
    device = preprocessed['radii'].device
    width,height = raster_settings.image_width,raster_settings.image_height
    gx,gy = (width+15)//16,(height+15)//16
    offsets = torch.empty((n,),dtype=torch.int32,device=device)
    stream = current_stream(device)
    if n:
        with wp.ScopedStream(stream, sync_enter=False):
            wp.utils.array_scan(wp.from_torch(preprocessed['tiles_touched']),wp.from_torch(offsets))
        count = int(offsets[-1].item())
    else:
        count = 0
    keys_storage = torch.empty((2*count,),dtype=torch.int64,device=device)
    values_storage = torch.empty((2*count,),dtype=torch.int32,device=device)
    ranges = torch.zeros((gx*gy,2),dtype=torch.int32,device=device)
    if count:
        keys_prefix, values_prefix = keys_storage[:count], values_storage[:count]
        kernel, inputs, block_dim = _duplicate_launch(preprocessed,raster_settings,offsets,_bindings)
        launch(kernel,dim=duplicate_dim(kernel,n),inputs=inputs,block_dim=block_dim,
            outputs=[kernel_arg(keys_prefix,wp.uint64,_bindings),
                     kernel_arg(values_prefix,bindings=_bindings)],stream=stream,bindings=_bindings)
        if diagnostics:
            keys_unsorted = keys_prefix.clone()
            values_unsorted = values_prefix.clone()
        with wp.ScopedStream(stream, sync_enter=False):
            wp.utils.radix_sort_pairs(wp.from_torch(keys_storage,dtype=wp.uint64),
                                      wp.from_torch(values_storage),count,
                                      end_bit=32+(gx*gy).bit_length())
        launch(_identify_ranges,dim=count,
                  inputs=[kernel_arg(keys_prefix,wp.uint64,_bindings)],
                  outputs=[kernel_arg(ranges,bindings=_bindings)],stream=stream,bindings=_bindings)
    elif diagnostics:
        keys_unsorted = keys_storage[:0]
        values_unsorted = values_storage[:0]
    result = dict(point_list=values_prefix if count else values_storage[:0], ranges=ranges)
    if diagnostics:
        result.update(point_offsets=offsets,
                      point_list_keys_unsorted=keys_unsorted,
                      point_list_unsorted=values_unsorted,
                      point_list_keys=keys_storage[:count])
    return result
