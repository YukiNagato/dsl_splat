"""CUDA-order ray/frustum arithmetic, automatically compiled by Warp.

Formulas from AAA/StopThePop consistent_common.cuh (MIT, Graz University of
Technology, 2024). The snippet is self-contained and requires no GLM headers
or separately built extension. Warp rows correspond to reference GLM columns.
"""
import warp as wp


GEOMETRY = r"""
    using V3 = wp::vec_t<3, float>;
    using V4 = wp::vec_t<4, float>;
    using M4 = wp::mat_t<4, 4, float>;
    struct Ray { V4 position; float squared_distance; };
    auto dot3 = [](V3 a, V3 b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };
    auto dot4 = [](V4 a, V4 b) {
        return (a[0]*b[0] + a[1]*b[1]) + (a[2]*b[2] + a[3]*b[3]);
    };
    auto transform = [&](const M4& g, V4 p) {
        return V4(dot4(g.get_row(0), p), dot4(g.get_row(1), p),
                  dot4(g.get_row(2), p), dot4(g.get_row(3), p));
    };
    auto ray = [&](V4 a, V4 b) {
        V3 d(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
        V3 m(a[3]*b[0]-a[0]*b[3], a[3]*b[1]-a[1]*b[3], a[3]*b[2]-a[2]*b[3]);
        float dd = dot3(d, d);
        V3 md(m[0]/dd, m[1]/dd, m[2]/dd);
        V4 p(d[1]*md[2]-md[1]*d[2], d[2]*md[0]-md[2]*d[0],
             d[0]*md[1]-md[0]*d[1], 1.0f);
        return Ray{p, dot3(m, md)};
    };
    auto sample = [&](const M4& g, float x, float y) {
        V4 a = g.get_row(0) - g.get_row(3)*x;
        V4 b = g.get_row(1) - g.get_row(3)*y;
        Ray r = ray(a, b);
        float depth = dot4(g.get_row(2), r.position)*__frcp_rn(dot4(g.get_row(3), r.position));
        return wp::vec_t<2, float>(-0.5f*r.squared_distance, depth);
    };
    auto frustum = [&](const M4& g, float x, float y, float w, float h) {
        const V3 start(x, y, -1.0f), extent(w, h, 2.0f);
        auto inside = [&](V4 p, int dim) {
            float delta = p[dim]-start[dim]*p[3];
            return delta > 0.0f && delta < extent[dim]*p[3];
        };
        V4 mean(g.data[0][3], g.data[1][3], g.data[2][3], g.data[3][3]);
        float best = 3.4028234663852886e38f;
        if (inside(mean, 0) && inside(mean, 1) && inside(mean, 2)) {
            best = 0.0f;
        } else {
            float dx = copysignf(w*0.5f, (mean[0]-x*mean[3])-w*0.5f*mean[3]);
            float dy = copysignf(h*0.5f, (mean[1]-y*mean[3])-h*0.5f*mean[3]);
            V4 ax = g.get_row(0)-g.get_row(3)*(x+w*0.5f+dx);
            V4 ay = g.get_row(1)-g.get_row(3)*(y+h*0.5f+dy);
            auto plane = [&](V4 a, int other_dim) {
                float norm = a[3]/(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
                V4 p(-a[0]*norm, -a[1]*norm, -a[2]*norm, 1.0f);
                float value = a[3]*norm;
                V4 screen = transform(g, p);
                if (value < best && inside(screen, other_dim) && inside(screen, 2)) best = value;
            };
            plane(ax, 1);
            plane(ay, 0);
            auto edge = [&](V4 a, V4 b) {
                Ray r = ray(a, b);
                V4 screen = transform(g, r.position);
                if (r.squared_distance < best && inside(screen, 2)) best = r.squared_distance;
            };
            edge(ax, ay);
            edge(ax, g.get_row(1)-g.get_row(3)*(y+h*0.5f-dy));
            edge(g.get_row(0)-g.get_row(3)*(x+w*0.5f-dx), ay);
        }
        return 0.5f*best;
    };
"""


@wp.func_native(GEOMETRY + 'return sample(g2s, x, y);')
def sample_native(g2s: wp.mat44, x: float, y: float) -> wp.vec2:
    pass


@wp.func_native(GEOMETRY + 'return frustum(g2s, x, y, width, height);')
def frustum_native(g2s: wp.mat44, x: float, y: float, width: float, height: float) -> float:
    pass


# The binning depth is a division, unlike sample()'s reciprocal multiplication.
# Keep its candidate positions together with the culling power, so preprocess
# and duplicate do not independently reimplement the geometric predicates.
# NVCC retains the closer y-plane products in the serial duplicate path,
# including the first 32 tiles of its load-balanced path. The farther plane
# and cooperative remaining tiles use FMAs. Preserve those boundaries without
# changing predicates.
FRUSTUM_DEPTH = r'''

    using V3 = wp::vec_t<3, float>;
    using V4 = wp::vec_t<4, float>;
    using M4 = wp::mat_t<4, 4, float>;
    struct Ray { V4 position; float squared_distance; };
    auto dot3 = [](V3 a, V3 b) {
        return fmaf(a[2],b[2],fmaf(a[0],b[0],__fmul_rn(a[1],b[1])));
    };
    auto dot4 = [](V4 a, V4 b) {
        return __fadd_rn(fmaf(a[0],b[0],__fmul_rn(a[1],b[1])), fmaf(a[2],b[2],__fmul_rn(a[3],b[3])));
    };
    auto transform = [&](const M4& g, V4 p) {
        return V4(dot4(g.get_row(0), p), dot4(g.get_row(1), p),
                  dot4(g.get_row(2), p), dot4(g.get_row(3), p));
    };
    auto ray = [&](V4 a, V4 b) {
        V3 d(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]);
        V3 m(a[3]*b[0]-a[0]*b[3], a[3]*b[1]-a[1]*b[3], a[3]*b[2]-a[2]*b[3]);
        float dd = dot3(d, d);
        V3 md(m[0]/dd, m[1]/dd, m[2]/dd);
        V4 p(d[1]*md[2]-md[1]*d[2], d[2]*md[0]-md[2]*d[0],
             d[0]*md[1]-md[0]*d[1], 1.0f);
        return Ray{p, dot3(m, md)};
    };
    const V3 start(x,y,-1.0f), extent(width,height,2.0f);
    auto inside = [&](V4 p, int dim) {
        float delta = p[dim]-start[dim]*p[3];
        return delta > 0.0f && delta < extent[dim]*p[3];
    };
    
    const M4& g = g2s;
    auto plane_at=[&](int axis,float coordinate,bool far=false) {
        V4 result;
        #pragma unroll
        for (int j=0;j<4;++j) result[j] = sequential && axis == 1 && !far
            ? __fsub_rn(g.data[axis][j],__fmul_rn(g.data[3][j],coordinate))
            : fmaf(-g.data[3][j],coordinate,g.data[axis][j]);
        return result;
    };

    V4 mean(g.data[0][3],g.data[1][3],g.data[2][3],g.data[3][3]);
    V4 best_pos(1.0f);
    float best = 3.4028234663852886e38f;
    if (inside(mean,0) && inside(mean,1) && inside(mean,2)) {
        best = 0.0f;
        best_pos = mean;
    } else {
        float dx = copysignf(width*0.5f,(mean[0]-x*mean[3])-width*0.5f*mean[3]);
        float dy = copysignf(height*0.5f,(mean[1]-y*mean[3])-height*0.5f*mean[3]);
        V4 ax = plane_at(0,x+width*0.5f+dx);
        V4 ay = plane_at(1,y+height*0.5f+dy);
        auto plane = [&](V4 a,int other) {
            float norm = a[3]/dot3(V3(a[0],a[1],a[2]),V3(a[0],a[1],a[2]));
            V4 p(-a[0]*norm,-a[1]*norm,-a[2]*norm,1.0f);
            V4 screen = transform(g,p);
            float value = a[3]*norm;
            if (value < best && inside(screen,other) && inside(screen,2)) {
                best = value;
                best_pos = screen;
            }
        };
        plane(ax,1);
        plane(ay,0);
        auto edge = [&](V4 a,V4 b) {
            Ray r = ray(a,b);
            V4 screen = transform(g,r.position);
            if (r.squared_distance < best && inside(screen,2)) {
                best = r.squared_distance;
                best_pos = screen;
            }
        };
        edge(ax,ay);
        edge(ax,plane_at(1,y+height*0.5f-dy,true));
        edge(plane_at(0,x+width*0.5f-dx),ay);
    }
    return wp::vec_t<2,float>(0.5f*best,best_pos[2]/best_pos[3]);
'''


@wp.func_native(FRUSTUM_DEPTH)
def frustum_depth_native(g2s:wp.mat44,x:float,y:float,width:float,height:float,sequential:bool)->wp.vec2:
    pass


@wp.func_native(r'''
    using V4=wp::vec_t<4,float>;
    const float pi=3.14159265358979323846f;
    auto dot4=[](V4 a,V4 b) {
        return (a[0]*b[0]+a[1]*b[1])+(a[2]*b[2]+a[3]*b[3]);
    };
    // All three squared view coordinates are reused by the two axis bounds.
    const float length_squared=__fadd_rn(__fadd_rn(__fmul_rn(mean[0],mean[0]),
        __fmul_rn(mean[1],mean[1])),__fmul_rn(mean[2],mean[2]));
    const float norm=1.0f/sqrtf(length_squared);
    const auto direction=mean*norm;
    const float theta_mu=atan2f(direction[axis],direction[2]);
    const V4 t(cutoff,cutoff,cutoff,-1.0f), a=g2v.get_row(axis), z=g2v.get_row(2);
    float aa=dot4(t,wp::cw_mul(a,a));
    float zz=dot4(t,wp::cw_mul(z,z));
    float az=dot4(t,wp::cw_mul(a,z));
    float discriminant=az*az-zz*aa;
    float lower=-(1.57079632679489661923f-1e-5f), upper=-lower;
    if (discriminant > 0.0f) {
        float root=sqrtf(discriminant);
        float lo=atan2f(-(az+root),-zz), hi=atan2f(-(az-root),-zz);
        while (lo > theta_mu) lo-=pi;
        while (lo < theta_mu-pi) lo+=pi;
        while (hi < theta_mu) hi+=pi;
        while (hi > theta_mu+pi) hi-=pi;
        auto angle=[&](float theta) {
            theta=fmodf(theta,2.0f*pi);
            if (theta > pi) theta-=2.0f*pi;
            if (theta <= -pi) theta+=2.0f*pi;
            return theta;
        };
        float nl=angle(lo), nh=angle(hi);
        if (theta_mu < 0.0f && fabsf(nl) < fabsf(nh)) { lo+=2.0f*pi; hi+=2.0f*pi; }
        else if (theta_mu > 0.0f && fabsf(nh) < fabsf(nl)) { lo-=2.0f*pi; hi-=2.0f*pi; }
        lower=fmaxf(lower,lo);
        upper=fminf(upper,hi);
    }
    return wp::vec_t<2,float>(tanf(lower),tanf(upper));
''')
def bound_axis_native(g2v:wp.mat44,mean:wp.vec3,cutoff:float,axis:int)->wp.vec2:
    pass


@wp.func_native(r'''
    using V4=wp::vec_t<4,float>;
    auto dot4=[](V4 a,V4 b) {
        return (a[0]*b[0]+a[1]*b[1])+(a[2]*b[2]+a[3]*b[3]);
    };
    const V4 t(cutoff,cutoff,cutoff,-1.0f);
    const V4 x=g2s.get_row(0), y=g2s.get_row(1), z=g2s.get_row(2), w=g2s.get_row(3);
    float s=dot4(t,wp::cw_mul(w,w));
    if (s >= 0.0f) return V4(0.0f,0.0f,-1.0f,-1.0f);
    V4 f=(1.0f/s)*t;
    float px=dot4(f,wp::cw_mul(x,w)), py=dot4(f,wp::cw_mul(y,w)), pz=dot4(f,wp::cw_mul(z,w));
    float hz=pz*pz-dot4(f,wp::cw_mul(z,z));
    float ez=sqrtf(fmaxf(hz,0.0f));
    if (pz-ez < -1.0f || pz+ez > 1.0f) return V4(0.0f,0.0f,-1.0f,-1.0f);
    float hx=px*px-dot4(f,wp::cw_mul(x,x)), hy=py*py-dot4(f,wp::cw_mul(y,y));
    return V4(px,py,sqrtf(fmaxf(hx,0.0f)),sqrtf(fmaxf(hy,0.0f)));
''')
def screen_bounds_native(g2s:wp.mat44,cutoff:float)->wp.vec4:
    pass
