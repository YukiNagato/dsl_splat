"""AAA preprocessing, ported from cuda_rasterizer/forward{,_common}.{cu,h}.

Original formulas: Inria / AAA-Gaussians; see the reference project's LICENSE.md.
This is a forward-only diagnostic implementation, with no autograd registration.
"""
import torch
import warp as wp
from .interop import current_stream, contiguous_features
from .dispatch import kernel_arg, launch, launch_tiled
from .geometry import rotation_matrix
from .preprocess3d import preprocess_3d
from .preprocess_common import finish_preprocess
from .cooperative_culling import cull_first_32, cull_remainder, cull_remainder_3d


@wp.func
def _transform(p: wp.vec3, m: wp.array2d(dtype=float)):
    # The reference Torch matrices contain GLM columns in their rows.
    return wp.vec4(
        m[0, 0]*p[0] + m[1, 0]*p[1] + m[2, 0]*p[2] + m[3, 0],
        m[0, 1]*p[0] + m[1, 1]*p[1] + m[2, 1]*p[2] + m[3, 1],
        m[0, 2]*p[0] + m[1, 2]*p[1] + m[2, 2]*p[2] + m[3, 2],
        m[0, 3]*p[0] + m[1, 3]*p[1] + m[2, 3]*p[2] + m[3, 3])


@wp.kernel(enable_backward=False)
def _preprocess(
    means: wp.array(dtype=wp.vec3), scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4), opacities: wp.array(dtype=float),
    cov_precomp: wp.array2d(dtype=float), use_cov: bool, view: wp.array2d(dtype=float),
    proj: wp.array2d(dtype=float), width: int, height: int,
    tanx: float, tany: float, modifier: float, ewa: bool, rect: bool, tight: bool,
    means2d: wp.array(dtype=wp.vec2), rects: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float), covs: wp.array2d(dtype=float),
    conics: wp.array(dtype=wp.vec4),
    radii: wp.array(dtype=int), tiles: wp.array(dtype=int),
):
    i = wp.tid()
    radii[i] = 0
    tiles[i] = 0
    p = means[i]
    t = _transform(p, view)
    if t[2] < 0.2:
        return
    cov = wp.mat33(0.0)
    if use_cov:
        cov = wp.mat33(cov_precomp[i,0],cov_precomp[i,1],cov_precomp[i,2],
                       cov_precomp[i,1],cov_precomp[i,3],cov_precomp[i,4],
                       cov_precomp[i,2],cov_precomp[i,4],cov_precomp[i,5])
    else:
        rotation = rotation_matrix(rotations[i])
        s = scales[i]*modifier
        scaling = wp.mat33(s[0],0.0,0.0,0.0,s[1],0.0,0.0,0.0,s[2])
        m = scaling*rotation
        cov = wp.transpose(m)*m
    fx = float(width) / (2.0*tanx)
    fy = float(height) / (2.0*tany)
    tx = wp.clamp(t[0]/t[2], -1.3*tanx, 1.3*tanx)*t[2]
    ty = wp.clamp(t[1]/t[2], -1.3*tany, 1.3*tany)*t[2]
    j = wp.mat33(fx/t[2], 0.0, 0.0,
                 0.0, fy/t[2], 0.0,
                 -(fx*tx)/(t[2]*t[2]), -(fy*ty)/(t[2]*t[2]), 0.0)
    v = wp.mat33(view[0,0], view[0,1], view[0,2],
                 view[1,0], view[1,1], view[1,2],
                 view[2,0], view[2,1], view[2,2])
    transform = v*j
    projected = wp.transpose(transform)*wp.transpose(cov)*transform
    a, b, c = projected[0,0]+0.3, projected[0,1], projected[1,1]+0.3
    det = a*c-b*b
    if det == 0.0:
        return
    compensation = float(1.0)
    if ewa:
        original_det = projected[0,0]*projected[1,1]-b*b
        compensation = wp.sqrt(wp.max(0.000025, original_det/det))
    opacity = opacities[i]*compensation
    if opacity < 1.0/255.0:
        return
    extent = float(3.33)
    if tight:
        extent = wp.min(3.33, wp.sqrt(2.0*wp.log(opacity/(1.0/255.0))))
    mid = 0.5*(a+c)
    eigenvalue = mid+wp.sqrt(wp.max(0.01, mid*mid-det))
    radius = extent*wp.sqrt(eigenvalue)
    if radius <= 0.0:
        return
    clip = _transform(p, proj)
    invw = 1.0/(clip[3]+0.0000001)
    center = wp.vec2(((clip[0]*invw+1.0)*float(width)-1.0)*0.5,
                     ((clip[1]*invw+1.0)*float(height)-1.0)*0.5)
    ex, ey = radius, radius
    if rect:
        ex = wp.min(extent*wp.sqrt(a), radius)
        ey = wp.min(extent*wp.sqrt(c), radius)
    gx, gy = (width+15)//16, (height+15)//16
    xmin = wp.clamp(int(wp.floor((center[0]-ex)/16.0)), 0, gx)
    ymin = wp.clamp(int(wp.floor((center[1]-ey)/16.0)), 0, gy)
    xmax = wp.clamp(int(wp.ceil((center[0]+ex)/16.0)), 0, gx)
    ymax = wp.clamp(int(wp.ceil((center[1]+ey)/16.0)), 0, gy)
    count = (xmax-xmin)*(ymax-ymin)
    if count == 0:
        return
    means2d[i] = center
    rects[i] = wp.vec2(ex, ey)
    depths[i] = t[2]
    covs[i,0] = cov[0,0]
    covs[i,1] = cov[0,1]
    covs[i,2] = cov[0,2]
    covs[i,3] = cov[1,1]
    covs[i,4] = cov[1,2]
    covs[i,5] = cov[2,2]
    invdet = 1.0/det
    conics[i] = wp.vec4(c*invdet, -b*invdet, a*invdet, opacity)
    radii[i] = int(wp.ceil(radius))
    tiles[i] = count


@torch.no_grad()
def preprocess(*, means3D, opacities, raster_settings, scales=None, rotations=None,
               colors_precomp=None, shs=None, cov3D_precomp=None, filter3D=None, features=None, _bindings=None):
    """Run AAA forward preprocessing and return detached, named CUDA tensors.

    Mirrors the supported input branches of AAA's preprocessCUDA, including 3D
    filtering/bounds/culling. With load_balancing and tile culling enabled,
    the first 32 tiles are checked per Gaussian and the remainder is distributed
    across a 32-thread GPU block.
    Matrix and quaternion conventions match the original Python interface.
    ``radii``, ``tiles_touched`` and ``valid`` are defined for every row.
    Other fields are meaningful only where ``valid`` is True, as in CUDA;
    rejected geometry/SH rows are not initialized or cleared.
    ``features`` aliases ``colors_precomp`` and accepts (N,C), C > 0. The
    legacy result key ``rgb`` contains these contiguous features directly
    (aliasing input storage when contiguous); SH generates a fresh (N,3) array.
    """
    wp.init()
    if features is not None:
        if colors_precomp is not None:
            raise ValueError('provide features OR colors_precomp, not both')
        colors_precomp = features
    settings = raster_settings.settings
    culling = settings.culling_settings
    mode, order = int(settings.sort_settings.sort_mode), int(settings.sort_settings.sort_order)
    if mode not in range(4) or order not in range(4):
        raise ValueError('invalid sort mode/order')
    eval3d = bool(settings.eval_3D)
    need_inverse = mode != 0 or order in (2,3)
    if not isinstance(means3D, torch.Tensor) or not means3D.is_cuda:
        raise ValueError('means3D must be a CUDA tensor')
    if means3D.ndim != 2 or means3D.shape[1] != 3:
        raise ValueError('means3D must have shape (N, 3)')
    device, n = means3D.device, means3D.shape[0]
    def check(name, value, shape):
        if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != torch.float32:
            raise ValueError(f'{name} must be float32 on {device}')
        if tuple(value.shape) != shape:
            raise ValueError(f'{name} must have shape {shape}')
        # This function runs under no_grad, and kernel_arg explicitly disables
        # Warp gradients. Avoid an extra Tensor view for already-contiguous inputs.
        return value.contiguous()
    def optional(name, value, shape):
        # Empty arrays are never accessed by inactive branches.
        return torch.empty((0,)+shape[1:],device=device) if value is None else check(name,value,shape)
    if (shs is None) == (colors_precomp is None):
        raise ValueError('provide exactly one of shs or colors_precomp')
    channels = 3
    if colors_precomp is not None:
        if not isinstance(colors_precomp,torch.Tensor) or colors_precomp.ndim != 2 or colors_precomp.shape[0] != n or colors_precomp.shape[1] <= 0:
            raise ValueError('features/colors_precomp must have shape (N,C), C > 0')
        channels = colors_precomp.shape[1]
        colors_precomp = contiguous_features(check('colors_precomp',colors_precomp,(n,channels)))
    check('bg',raster_settings.bg,(channels,))
    if cov3D_precomp is None:
        if scales is None or rotations is None:
            raise ValueError('provide scales and rotations')
    elif scales is not None or rotations is not None:
        raise ValueError('provide covariance OR scales/rotations')
    if cov3D_precomp is not None and (eval3d or need_inverse):
        raise ValueError('3D evaluation and ray-depth helpers require scales/rotations')
    degree = int(raster_settings.sh_degree)
    if degree not in range(4):
        raise ValueError('sh_degree must be in [0,3]')
    if shs is not None:
        if shs.ndim != 3 or shs.shape[0] != n or shs.shape[2] != 3 or shs.shape[1] < (degree+1)**2:
            raise ValueError('shs must be (N,M,3), M >= (sh_degree+1)**2')
    def scalar_input(name, value):
        if not isinstance(value,torch.Tensor):
            raise ValueError(f'{name} must be a tensor')
        if value.shape == (n,1):
            value = value.reshape(n)
        return check(name,value,(n,))
    tensors = {
        'means': check('means3D',means3D,(n,3)),
        'scales': optional('scales',scales,(n,3)),
        'rotations': optional('rotations',rotations,(n,4)),
        'opacities': scalar_input('opacities',opacities),
        'cov': optional('cov3D_precomp',cov3D_precomp,(n,6)),
        'filter': torch.empty(0,device=device) if filter3D is None else scalar_input('filter3D',filter3D),
        'sh': optional('shs',shs,tuple(shs.shape) if shs is not None else (n,0,3)),
        'view': check('viewmatrix',raster_settings.viewmatrix,(4,4)),
        'proj': check('projmatrix',raster_settings.projmatrix,(4,4)),
        'camera': check('campos',raster_settings.campos,(3,)).reshape(1,3),
    }
    w,h = raster_settings.image_width,raster_settings.image_height
    if w <= 0 or h <= 0 or raster_settings.tanfovx <= 0 or raster_settings.tanfovy <= 0:
        raise ValueError('image dimensions and field of view tangents must be positive')
    if raster_settings.scale_modifier <= 0:
        raise ValueError('scale_modifier must be positive')
    shapes = {'means2D':(n,2),'rects2D':(n,2),'depths':(n,),'radii':(n,),'tiles_touched':(n,)}
    if eval3d:
        shapes.update(gauss2screen=(n,4,4),opacity=(n,))
    else:
        shapes.update(cov3D=(n,6),conic_opacity=(n,4))
        if need_inverse:
            shapes['cov3D_inv'] = (n,3,4)
    if shs is not None:
        shapes['clamped'] = (n,3)
        shapes['rgb'] = (n,3)
    shapes['valid'] = (n,)
    cooperative = bool(settings.load_balancing and culling.tile_based_culling)
    complete_3d = eval3d and not cooperative
    result = {key:torch.empty(
        shape, device=device, dtype=torch.int32 if key in ('radii','tiles_touched')
        else torch.bool if key in ('clamped','valid') else torch.float32)
        for key,shape in shapes.items()}
    if n:
        stream = current_stream(device, bindings=_bindings)
        refs = []
        def arr(key,dtype=wp.float32):
            return kernel_arg(tensors[key],dtype,_bindings)
        def out(key,dtype=wp.float32,empty_shape=(0,)):
            if key in result:
                return kernel_arg(result[key],dtype,_bindings)
            if key == 'rgb':
                empty_shape = (0,3)
            t = torch.empty(empty_shape,device=device,dtype=torch.bool if dtype==wp.bool else torch.float32)
            refs.append(t)
            return kernel_arg(t,dtype,_bindings)
        means,scale,rotation = arr('means',wp.vec3),arr('scales',wp.vec3),arr('rotations',wp.vec4)
        common = [out('means2D',wp.vec2),out('rects2D',wp.vec2),out('depths')]
        if eval3d:
            queue = torch.empty(n if cooperative else 0,device=device,dtype=torch.int32)
            queue_size = torch.zeros(1,device=device,dtype=torch.int32) if cooperative else queue
            refs.extend((queue,queue_size))
            queued_arg = kernel_arg(queue,bindings=_bindings)
            queue_size_arg = kernel_arg(queue_size,bindings=_bindings)
            launch(preprocess_3d,dim=n,inputs=[means,scale,rotation,arr('opacities'),
                arr('sh',wp.vec3),shs is not None,degree,order,arr('filter'),
                filter3D is not None,arr('camera',wp.vec3),arr('view'),arr('proj'),w,h,
                raster_settings.tanfovx,raster_settings.tanfovy,raster_settings.scale_modifier,
                settings.proper_ewa_scaling,getattr(settings,'new_aabb',True),getattr(settings,'near_clipping',False),
                culling.tile_based_culling,complete_3d,cooperative],
                outputs=common+[out('gauss2screen',wp.mat44),out('opacity'),out('radii',wp.int32),
                                out('tiles_touched',wp.int32),out('rgb',wp.vec3),out('valid',wp.bool),
                                out('clamped',wp.bool,(0,3)),queued_arg,queue_size_arg],
                block_dim=32,stream=stream,bindings=_bindings)
        else:
            launch(_preprocess,dim=n,inputs=[means,scale,rotation,arr('opacities'),arr('cov'),cov3D_precomp is not None,
                arr('view'),arr('proj'),w,h,raster_settings.tanfovx,raster_settings.tanfovy,raster_settings.scale_modifier,
                settings.proper_ewa_scaling,culling.rect_bounding,culling.tight_opacity_bounding],
                outputs=common+[out('cov3D'),out('conic_opacity',wp.vec4),out('radii',wp.int32),out('tiles_touched',wp.int32)],stream=stream,bindings=_bindings)
        if cooperative and eval3d:
            remainder_blocks = min(n,1024)
            launch_tiled(cull_remainder_3d,dim=remainder_blocks,
                         inputs=[out('means2D',wp.vec2),out('rects2D',wp.vec2),
                                 out('gauss2screen',wp.mat44),out('opacity'),w,h,remainder_blocks],
                         outputs=[out('radii',wp.int32),out('tiles_touched',wp.int32),
                                  out('valid',wp.bool),queued_arg,queue_size_arg],
                         block_dim=32,stream=stream,bindings=_bindings)
        elif cooperative:
            queue = torch.empty(n, device=device, dtype=torch.int32)
            queue_size = torch.zeros(1, device=device, dtype=torch.int32)
            refs.extend((queue, queue_size))
            culling_inputs = [out('means2D',wp.vec2), out('rects2D',wp.vec2),
                # The predicate loads only the active geometry layout.
                out('conic_opacity',wp.vec4,(0,4)), out('gauss2screen',wp.mat44,(0,4,4)),
                out('opacity'), eval3d, w, h]
            culling_outputs = [out('radii',wp.int32), out('tiles_touched',wp.int32),
                kernel_arg(queue,bindings=_bindings), kernel_arg(queue_size,bindings=_bindings)]
            launch(cull_first_32, dim=n, inputs=culling_inputs,
                      outputs=culling_outputs, stream=stream,bindings=_bindings)
            remainder_blocks = min(n, 32768)
            launch_tiled(cull_remainder, dim=remainder_blocks,
                            inputs=culling_inputs+[remainder_blocks],
                            outputs=culling_outputs, block_dim=32, stream=stream,bindings=_bindings)
        if not eval3d:
            launch(finish_preprocess,dim=n,inputs=[means,scale,rotation,arr('camera',wp.vec3),
                arr('sh',wp.vec3),shs is not None,degree,raster_settings.scale_modifier,order,need_inverse,eval3d,
                culling.tile_based_culling and not cooperative,w,h,out('means2D',wp.vec2),out('rects2D',wp.vec2),
                out('conic_opacity',wp.vec4,(0,4)),out('gauss2screen',wp.mat44,(0,4,4)),out('opacity'),
                out('cov3D',empty_shape=(0,6))],
                outputs=[out('depths'),out('rgb',wp.vec3),out('clamped',wp.bool,(0,3)),out('cov3D_inv',wp.float32,(0,3,4)),
                         out('radii',wp.int32),out('tiles_touched',wp.int32),out('valid',wp.bool)],stream=stream,bindings=_bindings)
    if colors_precomp is not None:
        # Legacy key retained for stage/API compatibility; it can now hold C
        # channels. No extra feature copy or per-channel geometry pass.
        result['rgb'] = colors_precomp.detach() if colors_precomp.requires_grad else colors_precomp
    return result
