"""AAA geometry from consistent_common.cuh and stopthepop_common.cuh.

Matrices here have ordinary mathematical rows. The CUDA GLM helpers store the
transpose, so g2s[col] in CUDA corresponds to g2s[row] here.
"""
import warp as wp


@wp.func
def rotation_matrix(q: wp.vec4):
    r, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33(
        1.0-2.0*(y*y+z*z), 2.0*(x*y+r*z), 2.0*(x*z-r*y),
        2.0*(x*y-r*z), 1.0-2.0*(x*x+z*z), 2.0*(y*z+r*x),
        2.0*(x*z+r*y), 2.0*(y*z-r*x), 1.0-2.0*(x*x+y*y))


@wp.func
def load_camera(a: wp.array2d(dtype=float)):
    return wp.mat44(a[0,0], a[1,0], a[2,0], a[3,0],
                    a[0,1], a[1,1], a[2,1], a[3,1],
                    a[0,2], a[1,2], a[2,2], a[3,2],
                    a[0,3], a[1,3], a[2,3], a[3,3])


@wp.func
def xyz(v: wp.vec4):
    return wp.vec3(v[0], v[1], v[2])


@wp.func
def in_range(p: wp.vec4, start: wp.vec3, extent: wp.vec3, axis: int):
    d = p[axis]-start[axis]*p[3]
    return d > 0.0 and d < extent[axis]*p[3]


@wp.func
def plane_minimum(plane: wp.vec4, g2s: wp.mat44):
    n = xyz(plane)
    factor = plane[3]/wp.dot(n,n)
    p = -plane*factor
    return -p[3], g2s*wp.vec4(p[0],p[1],p[2],1.0)


@wp.func
def ray_minimum(a: wp.vec4, b: wp.vec4, g2s: wp.mat44):
    d = wp.cross(xyz(a), xyz(b))
    m = a[3]*xyz(b)-b[3]*xyz(a)
    md = m/wp.dot(d,d)
    p = wp.cross(d,md)
    return wp.dot(m,md), g2s*wp.vec4(p[0],p[1],p[2],1.0)


@wp.func
def frustum_minimum(lo: wp.vec2, hi: wp.vec2, g2s: wp.mat44):
    start = wp.vec3(lo[0],lo[1],-1.0)
    extent = wp.vec3(hi[0]-lo[0],hi[1]-lo[1],2.0)
    mean = wp.vec4(g2s[0,3],g2s[1,3],g2s[2,3],g2s[3,3])
    best = float(3.4028234663852886e38)
    if in_range(mean,start,extent,0) and in_range(mean,start,extent,1) and in_range(mean,start,extent,2):
        best = 0.0
    else:
        dx = wp.copysign(extent[0]*0.5, (mean[0]-start[0]*mean[3])-(extent[0]*0.5*mean[3]))
        dy = wp.copysign(extent[1]*0.5, (mean[1]-start[1]*mean[3])-(extent[1]*0.5*mean[3]))
        ax = g2s[0]-g2s[3]*(start[0]+extent[0]*0.5+dx)
        ay = g2s[1]-g2s[3]*(start[1]+extent[1]*0.5+dy)
        v, p = plane_minimum(ax,g2s)
        if v < best and in_range(p,start,extent,1) and in_range(p,start,extent,2):
            best = v
        v, p = plane_minimum(ay,g2s)
        if v < best and in_range(p,start,extent,0) and in_range(p,start,extent,2):
            best = v
        v, p = ray_minimum(ax,ay,g2s)
        if v < best and in_range(p,start,extent,2):
            best = v
        other_y = g2s[1]-g2s[3]*(start[1]+extent[1]*0.5-dy)
        v, p = ray_minimum(ax,other_y,g2s)
        if v < best and in_range(p,start,extent,2):
            best = v
        other_x = g2s[0]-g2s[3]*(start[0]+extent[0]*0.5-dx)
        v, p = ray_minimum(other_x,ay,g2s)
        if v < best and in_range(p,start,extent,2):
            best = v
    return 0.5*best


@wp.func
def normalize_angle(theta: float):
    value = wp.mod(theta,2.0*wp.pi)
    if value > wp.pi:
        value -= 2.0*wp.pi
    if value <= -wp.pi:
        value += 2.0*wp.pi
    return value


@wp.func
def bound_axis(g2v: wp.mat44, mean: wp.vec3, cutoff: float, axis: int):
    direction = wp.normalize(mean)
    theta_mu = wp.atan2(direction[axis],direction[2])
    t = wp.vec4(cutoff,cutoff,cutoff,-1.0)
    aa = wp.dot(t,wp.cw_mul(g2v[axis],g2v[axis]))
    zz = wp.dot(t,wp.cw_mul(g2v[2],g2v[2]))
    az = wp.dot(t,wp.cw_mul(g2v[axis],g2v[2]))
    discriminant = az*az-zz*aa
    lower = -(wp.pi/2.0-1.0e-5)
    upper = wp.pi/2.0-1.0e-5
    if discriminant > 0.0:
        root = wp.sqrt(discriminant)
        a = wp.atan2(-(az+root),-zz)
        b = wp.atan2(-(az-root),-zz)
        while a > theta_mu:
            a -= wp.pi
        while a < theta_mu-wp.pi:
            a += wp.pi
        while b < theta_mu:
            b += wp.pi
        while b > theta_mu+wp.pi:
            b -= wp.pi
        na, nb = normalize_angle(a), normalize_angle(b)
        if theta_mu < 0.0 and wp.abs(na) < wp.abs(nb):
            a += 2.0*wp.pi
            b += 2.0*wp.pi
        elif theta_mu > 0.0 and wp.abs(nb) < wp.abs(na):
            a -= 2.0*wp.pi
            b -= 2.0*wp.pi
        lower = wp.max(lower,a)
        upper = wp.min(upper,b)
    return wp.vec2(wp.tan(lower),wp.tan(upper))


@wp.func
def screen_bounds(g2s: wp.mat44, cutoff: float):
    t = wp.vec4(cutoff,cutoff,cutoff,-1.0)
    s = wp.dot(t,wp.cw_mul(g2s[3],g2s[3]))
    center, extent = wp.vec2(0.0), wp.vec2(0.0)
    valid = False
    if s < 0.0:
        f = t/s
        p = wp.vec3(wp.dot(f,wp.cw_mul(g2s[0],g2s[3])),
                    wp.dot(f,wp.cw_mul(g2s[1],g2s[3])),
                    wp.dot(f,wp.cw_mul(g2s[2],g2s[3])))
        h = wp.cw_mul(p,p)-wp.vec3(wp.dot(f,wp.cw_mul(g2s[0],g2s[0])),
                                  wp.dot(f,wp.cw_mul(g2s[1],g2s[1])),
                                  wp.dot(f,wp.cw_mul(g2s[2],g2s[2])))
        ez = wp.sqrt(wp.max(h[2],0.0))
        valid = not (p[2]-ez < -1.0 or p[2]+ez > 1.0)
        center = wp.vec2(p[0],p[1])
        extent = wp.vec2(wp.sqrt(wp.max(h[0],0.0)),wp.sqrt(wp.max(h[1],0.0)))
    return valid, center, extent


@wp.func
def tile_minimum_2d(co: wp.vec4, mean: wp.vec2, lo: wp.vec2, hi: wp.vec2):
    xdiff, ydiff = lo[0]-mean[0], lo[1]-mean[1]
    left, above = float(xdiff>0.0), float(ydiff>0.0)
    notx = left+float(mean[0]>hi[0])
    noty = above+float(mean[1]>hi[1])
    result = float(0.0)
    if notx+noty > 0.0:
        px = left*lo[0]+(1.0-left)*hi[0]
        py = above*lo[1]+(1.0-above)*hi[1]
        dx,dy = wp.copysign(15.0,xdiff),wp.copysign(15.0,ydiff)
        diffx,diffy = mean[0]-px,mean[1]-py
        tx = noty*wp.clamp((dx*co[0]*diffx+dx*co[1]*diffy)*(1.0/(225.0*co[0])),0.0,1.0)
        ty = notx*wp.clamp((dy*co[1]*diffx+dy*co[2]*diffy)*(1.0/(225.0*co[2])),0.0,1.0)
        mx,my = mean[0]-(px+tx*dx),mean[1]-(py+ty*dy)
        result = 0.5*(co[0]*mx*mx+co[2]*my*my)+co[1]*mx*my
    return result
