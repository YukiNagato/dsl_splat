"""Reference GLM arithmetic for AAA's 3D preprocessing.

Warp's generic matrix multiply uses a different FMA accumulation order, and its
normalize divides each component instead of multiplying by a reciprocal length.
Explicit rounding keeps shared products from being contracted differently by
NVRTC. Quaternion FMA choices were checked against the CUDA oracle's actual
rotation, not just PTX (PTXAS can further fuse mul/add). See ANOMALY_ANALYSIS.md.
These self-contained snippets are automatically JIT compiled by Warp.
"""

import warp as wp


@wp.func_native(r"""
    return wp::vec_t<3,float>(
        sqrtf(__fadd_rn(__fmul_rn(scale[0],scale[0]),filter_variance)),
        sqrtf(__fadd_rn(__fmul_rn(scale[1],scale[1]),filter_variance)),
        sqrtf(__fadd_rn(__fmul_rn(scale[2],scale[2]),filter_variance)));
""")
def glm_dilated_scale(scale: wp.vec3, filter_variance: float) -> wp.vec3:
    pass


@wp.func_native(r"""
    const float r=q[0], x=q[1], y=q[2], z=q[3];
    const float yy=__fmul_rn(y,y), zz=__fmul_rn(z,z);
    // xz is reused by both signs; xy/yz contract with the rounded r products.
    const float rz=__fmul_rn(r,z), xz=__fmul_rn(x,z), rx=__fmul_rn(r,x);
    const float d0=__fadd_rn(yy,zz), d1=fmaf(x,x,zz), d2=fmaf(x,x,yy);
    return wp::mat_t<3,3,float>(
        __fsub_rn(1.0f,__fadd_rn(d0,d0)), 2.0f*fmaf(x,y,rz), 2.0f*fmaf(-r,y,xz),
        2.0f*fmaf(x,y,-rz), __fsub_rn(1.0f,__fadd_rn(d1,d1)), 2.0f*fmaf(y,z,rx),
        2.0f*fmaf(r,y,xz), 2.0f*fmaf(y,z,-rx), __fsub_rn(1.0f,__fadd_rn(d2,d2)));
""")
def glm_rotation(q: wp.vec4) -> wp.mat33:
    pass


@wp.func_native(r"""
    wp::mat_t<4, 4, float> result;
    #pragma unroll
    for (int i=0; i<4; ++i) {
        #pragma unroll
        for (int j=0; j<4; ++j)
            result.data[i][j] = fmaf(a.data[i][3],b.data[3][j],
                fmaf(a.data[i][2],b.data[2][j],
                     fmaf(a.data[i][0],b.data[0][j],__fmul_rn(a.data[i][1],b.data[1][j]))));
    }
    return result;
""")
def glm_matmul44(a: wp.mat44, b: wp.mat44) -> wp.mat44:
    pass


@wp.func_native(r"""
    float xy = fmaf(view.data[2][0],p[0],__fmul_rn(view.data[2][1],p[1]));
    // The reference reuses the z product in its mat4x3 and mat4 view sums;
    // it is rounded before adding the translation in the filtering path.
    float z = __fmul_rn(view.data[2][2],p[2]);
    return __fadd_rn(xy,__fadd_rn(z,view.data[2][3]));
""")
def glm_filter_view_depth(view: wp.mat44, p: wp.vec3) -> float:
    pass


@wp.func_native(r"""
    wp::vec_t<3,float> result;
    #pragma unroll
    for (int i=0;i<3;++i) {
        float xy=fmaf(view.data[i][0],p[0],__fmul_rn(view.data[i][1],p[1]));
        float total;
        if (cooperative && i == 2)
            total=__fadd_rn(xy,__fmul_rn(view.data[i][2],p[2]));
        else
            total=fmaf(view.data[i][2],p[2],xy);
        result[i]=__fadd_rn(total,view.data[i][3]);
    }
    return result;
""")
def glm_view_position(view: wp.mat44, p: wp.vec3, cooperative: bool) -> wp.vec3:
    pass


@wp.func_native(r"""
    float inverse_length = 1.0f/sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    return a*inverse_length;
""")
def glm_normalize3(a: wp.vec3) -> wp.vec3:
    pass
