"""AAA adaptive 3D filter and perspective-correct Gaussian bounds."""
import warp as wp
from .geometry import rotation_matrix, load_camera, xyz, frustum_minimum, bound_axis, screen_bounds


@wp.func
def _zero_final_3d(i: int, centers: wp.array(dtype=wp.vec2), rects: wp.array(dtype=wp.vec2),
                   depths: wp.array(dtype=float), transforms: wp.array(dtype=wp.mat44),
                   effective_opacity: wp.array(dtype=float), radii: wp.array(dtype=int),
                   tiles: wp.array(dtype=int), rgb: wp.array(dtype=wp.vec3),
                   valid: wp.array(dtype=wp.bool)):
    centers[i] = wp.vec2(0.0)
    rects[i] = wp.vec2(0.0)
    depths[i] = 0.0
    transforms[i] = wp.mat44(0.0)
    effective_opacity[i] = 0.0
    radii[i] = 0
    tiles[i] = 0
    rgb[i] = wp.vec3(0.0)
    valid[i] = False


@wp.kernel(enable_backward=False)
def preprocess_3d(
    means: wp.array(dtype=wp.vec3), scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4), opacities: wp.array(dtype=float),
    colors: wp.array(dtype=wp.vec3),
    filters: wp.array(dtype=float), has_filter: bool, camera: wp.array(dtype=wp.vec3),
    view: wp.array2d(dtype=float), proj: wp.array2d(dtype=float),
    width: int, height: int, tanx: float, tany: float, modifier: float,
    ewa: bool, new_aabb: bool, near_clipping: bool, complete: bool,
    centers: wp.array(dtype=wp.vec2), rects: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float), transforms: wp.array(dtype=wp.mat44),
    effective_opacity: wp.array(dtype=float), radii: wp.array(dtype=int), tiles: wp.array(dtype=int),
    rgb: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    p = means[i]
    viewmat = load_camera(view)
    t = viewmat*wp.vec4(p[0],p[1],p[2],1.0)
    if near_clipping and t[2] < 0.2:
        if complete:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
        return
    fx,fy = float(width)/(2.0*tanx),float(height)/(2.0*tany)
    r = rotation_matrix(rotations[i])
    ray = r*wp.normalize(p-camera[0])
    variance = wp.cw_mul(scales[i],scales[i])
    filter_variance = (t[2]/wp.max(fx,fy))*(t[2]/wp.max(fx,fy))*0.3
    if has_filter:
        filter_variance = wp.max(filters[i]*filters[i],filter_variance)
    dilated_variance = variance+wp.vec3(filter_variance)
    dilated = wp.vec3(wp.sqrt(dilated_variance[0]),wp.sqrt(dilated_variance[1]),wp.sqrt(dilated_variance[2]))
    ray2 = wp.cw_mul(ray,ray)
    area_before = wp.dot(ray2,wp.vec3(variance[1]*variance[2],variance[0]*variance[2],variance[0]*variance[1]))
    # CUDA squares sqrt(dilated_variance) again for the compensation denominator.
    sd = wp.cw_mul(dilated,dilated)
    area_after = wp.dot(ray2,wp.vec3(sd[1]*sd[2],sd[0]*sd[2],sd[0]*sd[1]))
    opacity = opacities[i]
    if ewa:
        opacity *= wp.sqrt(area_before/area_after)
    if opacity < 1.0/255.0:
        if complete:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
        return
    threshold = wp.log(opacity/(1.0/255.0))
    cutoff = wp.min(11.11,2.0*threshold)
    scale = dilated*wp.sqrt(modifier)
    # Preserve CUDA operation order: inverse scale, then divide by sqrt(modifier).
    invscale = wp.vec3(1.0/dilated[0],1.0/dilated[1],1.0/dilated[2])/wp.sqrt(modifier)
    world2gauss = wp.mat33(invscale[0],0.0,0.0,0.0,invscale[1],0.0,0.0,0.0,invscale[2])*r
    cam_gauss = world2gauss*(camera[0]-p)
    if wp.dot(cam_gauss,cam_gauss) < cutoff:
        if complete:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
        return
    s = wp.mat33(scale[0],0.0,0.0,0.0,scale[1],0.0,0.0,0.0,scale[2])
    linear = wp.transpose(s*r)
    world = wp.mat44(linear[0,0],linear[0,1],linear[0,2],p[0],
                     linear[1,0],linear[1,1],linear[1,2],p[1],
                     linear[2,0],linear[2,1],linear[2,2],p[2],
                     0.0,0.0,0.0,1.0)
    viewport = wp.mat44(float(width)/2.0,0.0,0.0,float(width)/2.0-0.5,
                        0.0,float(height)/2.0,0.0,float(height)/2.0-0.5,
                        0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0)
    g2s = (viewport*load_camera(proj))*world
    if frustum_minimum(wp.vec2(0.0),wp.vec2(float(width)-1.0,float(height)-1.0),g2s) > threshold:
        if complete:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
        return
    center,extent = wp.vec2(0.0),wp.vec2(0.0)
    if new_aabb:
        g2v = viewmat*world
        bx = wp.vec2(float(width)/2.0)+fx*bound_axis(g2v,xyz(t),cutoff,0)
        by = wp.vec2(float(height)/2.0)+fy*bound_axis(g2v,xyz(t),cutoff,1)
        center = wp.vec2((bx[1]+bx[0])/2.0,(by[1]+by[0])/2.0)
        extent = wp.vec2((bx[1]-bx[0])/2.0,(by[1]-by[0])/2.0)
    else:
        bounds_valid,center,extent = screen_bounds(g2s,cutoff)
        if not bounds_valid:
            if complete:
                _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
            return
    gx,gy = (width+15)//16,(height+15)//16
    xmin = wp.clamp(int(wp.floor((center[0]-extent[0])/16.0)),0,gx)
    ymin = wp.clamp(int(wp.floor((center[1]-extent[1])/16.0)),0,gy)
    xmax = wp.clamp(int(wp.ceil((center[0]+extent[0])/16.0)),0,gx)
    ymax = wp.clamp(int(wp.ceil((center[1]+extent[1])/16.0)),0,gy)
    count = (xmax-xmin)*(ymax-ymin)
    if count == 0:
        if complete:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
        return
    centers[i] = center
    rects[i] = extent
    depths[i] = t[2]
    transforms[i] = g2s  # Row-major bytes equal CUDA's transposed GLM storage.
    effective_opacity[i] = opacity
    radius_i = int(wp.ceil(wp.max(extent[0],extent[1])))
    radii[i] = radius_i
    tiles[i] = count
    if complete:
        if radius_i > 0:
            rgb[i] = colors[i]
            valid[i] = True
        else:
            _zero_final_3d(i,centers,rects,depths,transforms,effective_opacity,radii,tiles,rgb,valid)
