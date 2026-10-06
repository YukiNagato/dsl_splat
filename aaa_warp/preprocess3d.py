"""AAA adaptive 3D filter and perspective-correct Gaussian bounds."""
import warp as wp
from .sh import evaluate_sh
from .native_math import (glm_matmul44, glm_filter_view_depth, glm_normalize3,
                          glm_rotation, glm_dilated_scale, glm_view_position)
from .geometry import load_camera, frustum_minimum, bound_axis, screen_bounds
from .native_geometry import frustum_depth_native


@wp.kernel(enable_backward=False)
def preprocess_3d(
    means: wp.array(dtype=wp.vec3), scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4), opacities: wp.array(dtype=float),
    sh: wp.array2d(dtype=wp.vec3),
    use_sh: bool, degree: int, sort_order: int,
    filters: wp.array(dtype=float), has_filter: bool, camera: wp.array(dtype=wp.vec3),
    view: wp.array2d(dtype=float), proj: wp.array2d(dtype=float),
    width: int, height: int, tanx: float, tany: float,
    principal_x: float, principal_y: float, modifier: float,
    ewa: bool, new_aabb: bool, near_clipping: bool, tile_culling: bool, complete: bool,
    cooperative: bool,
    centers: wp.array(dtype=wp.vec2), rects: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float), transforms: wp.array(dtype=wp.mat44),
    effective_opacity: wp.array(dtype=float), radii: wp.array(dtype=int), tiles: wp.array(dtype=int),
    rgb: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.bool),
    clamped: wp.array2d(dtype=wp.bool),
    queued: wp.array(dtype=int), queue_size: wp.array(dtype=int),
):
    i = wp.tid()
    # CUDA initializes culling sentinels, not rejected geometry rows.
    # No downstream stage reads geometry or SH flags when radius is zero.
    radii[i] = 0
    tiles[i] = 0
    valid[i] = False
    p = means[i]
    viewmat = load_camera(view)
    t = glm_view_position(viewmat,p,not complete)
    if near_clipping and t[2] < 0.2:
        return
    fx,fy = float(width)/(2.0*tanx),float(height)/(2.0*tany)
    r = glm_rotation(rotations[i])
    ray = r*glm_normalize3(p-camera[0])
    variance = wp.cw_mul(scales[i],scales[i])
    # forward uses mat4x3 for its depth, but compute_gauss2screen converts
    # that view to mat4 and uses GLM's pairwise mat4-vector sum for filtering.
    filter_depth = glm_filter_view_depth(viewmat,p)
    filter_variance = (filter_depth/wp.max(fx,fy))*(filter_depth/wp.max(fx,fy))*0.3
    if has_filter:
        filter_variance = wp.max(filters[i]*filters[i],filter_variance)
    dilated = glm_dilated_scale(scales[i],filter_variance)
    ray2 = wp.cw_mul(ray,ray)
    area_before = wp.dot(ray2,wp.vec3(variance[1]*variance[2],variance[0]*variance[2],variance[0]*variance[1]))
    # CUDA squares sqrt(dilated_variance) again for the compensation denominator.
    sd = wp.cw_mul(dilated,dilated)
    area_after = wp.dot(ray2,wp.vec3(sd[1]*sd[2],sd[0]*sd[2],sd[0]*sd[1]))
    opacity = opacities[i]
    if ewa:
        opacity *= wp.sqrt(area_before/area_after)
    if opacity < 1.0/255.0:
        return
    threshold = wp.log(opacity/(1.0/255.0))
    cutoff = wp.min(11.11,2.0*threshold)
    scale = dilated*wp.sqrt(modifier)
    # Preserve CUDA operation order: inverse scale, then divide by sqrt(modifier).
    invscale = wp.vec3(1.0/dilated[0],1.0/dilated[1],1.0/dilated[2])/wp.sqrt(modifier)
    world2gauss = wp.mat33(invscale[0],0.0,0.0,0.0,invscale[1],0.0,0.0,0.0,invscale[2])*r
    cam_gauss = world2gauss*(camera[0]-p)
    if wp.dot(cam_gauss,cam_gauss) < cutoff:
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
    g2s = glm_matmul44(glm_matmul44(viewport,load_camera(proj)),world)
    if frustum_minimum(wp.vec2(0.0),wp.vec2(float(width)-1.0,float(height)-1.0),g2s) > threshold:
        return
    center,extent = wp.vec2(0.0),wp.vec2(0.0)
    if new_aabb:
        g2v = glm_matmul44(viewmat,world)
        # NVCC reuses the mat4x3 view position for the translation column of
        # gauss2view. Its cooperative specialization retains the shared z
        # product before summation, unlike an independent mat4 multiply.
        g2v[0,3] = t[0]
        g2v[1,3] = t[1]
        g2v[2,3] = t[2]
        bx = wp.vec2(principal_x)+fx*bound_axis(g2v,t,cutoff,0)
        by = wp.vec2(principal_y)+fy*bound_axis(g2v,t,cutoff,1)
        center = wp.vec2((bx[1]+bx[0])/2.0,(by[1]+by[0])/2.0)
        extent = wp.vec2((bx[1]-bx[0])/2.0,(by[1]-by[0])/2.0)
    else:
        bounds_valid,center,extent = screen_bounds(g2s,cutoff)
        if not bounds_valid:
            return
    gx,gy = (width+15)//16,(height+15)//16
    xmin = wp.clamp(int(wp.floor((center[0]-extent[0])/16.0)),0,gx)
    ymin = wp.clamp(int(wp.floor((center[1]-extent[1])/16.0)),0,gy)
    xmax = wp.clamp(int(wp.ceil((center[0]+extent[0])/16.0)),0,gx)
    ymax = wp.clamp(int(wp.ceil((center[1]+extent[1])/16.0)),0,gy)
    count = (xmax-xmin)*(ymax-ymin)
    if count == 0:
        return
    if complete and tile_culling:
        count = int(0)
        for y in range(ymin,ymax):
            for x in range(xmin,xmax):
                lo = wp.vec2(float(x*16),float(y*16))
                # Match the reference's serial tile loop specialization.
                if frustum_depth_native(g2s,lo[0],lo[1],15.0,15.0,True)[0] <= threshold:
                    count += 1
        if count == 0:
            return
    radius_i = int(wp.ceil(wp.max(extent[0],extent[1])))
    if complete and radius_i <= 0:
        return
    if cooperative:
        # Preserve cooperative view-position arithmetic above even though SH
        # and the first tile checks now share this launch with geometry.
        if radius_i == 0:
            return
        rect_count = count
        rect_width = xmax-xmin
        count = int(0)
        for tile_index in range(wp.min(rect_count,32)):
            x = xmin+tile_index%rect_width
            y = ymin+tile_index//rect_width
            if frustum_depth_native(g2s,float(x*16),float(y*16),15.0,15.0,False)[0] <= threshold:
                count += 1
        if rect_count > 32:
            slot = wp.atomic_add(queue_size,0,1)
            queued[slot] = i
        elif count == 0:
            return
    centers[i] = center
    rects[i] = extent
    depths[i] = t[2]
    transforms[i] = g2s  # Row-major bytes equal CUDA's transposed GLM storage.
    effective_opacity[i] = opacity
    radii[i] = radius_i
    tiles[i] = count
    # Long rectangles may still be rejected by the following collective
    # kernel; their geometry/color is then undefined and valid is reset there.
    if complete or cooperative:
        if radius_i != 0:
            if sort_order != 0:
                depths[i] = wp.length(camera[0]-p)
            if use_sh:
                color = evaluate_sh(i,degree,p,camera[0],sh)
                for c in range(3):
                    clamped[i,c] = color[c] < 0.0
                rgb[i] = wp.vec3(wp.max(color[0],0.0),wp.max(color[1],0.0),wp.max(color[2],0.0))
            # Precomputed features alias their Torch input in the host wrapper.
            # Only SH-generated RGB needs a geometry-stage output buffer.
            valid[i] = True
