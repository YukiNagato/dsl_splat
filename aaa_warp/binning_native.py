"""AAA duplicate scheduling, compiled automatically by Warp's native JIT."""
import warp as wp

from .native_geometry import FRUSTUM_DEPTH


_BODY = r'''
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int gx = (width + 15) / 16, gy = (height + 15) / 16;
    const bool active = i < radii.shape[0] && radii.data[i] > 0;
    int xmin=0, ymin=0, rw=0, count=0, off=0, stop=0;
    float threshold=0.0f;
    wp::mat_t<4,4,float> local(0.0f);
    if (active) {
        const auto center=centers.data[i], extent=extents.data[i];
        xmin = max(0,min(gx,static_cast<int>(floorf((center[0]-extent[0])/16.0f))));
        ymin = max(0,min(gy,static_cast<int>(floorf((center[1]-extent[1])/16.0f))));
        const int xmax=max(0,min(gx,static_cast<int>(ceilf((center[0]+extent[0])/16.0f))));
        const int ymax=max(0,min(gy,static_cast<int>(ceilf((center[1]+extent[1])/16.0f))));
        rw=xmax-xmin;
        count=rw*(ymax-ymin);
        off=i ? offsets.data[i-1] : 0;
        stop=offsets.data[i];
        threshold=logf(opacity3d.data[i]/(1.0f/255.0f));
        local=transforms.data[i];
        for (int t=0;t<min(count,32);++t) {
            const int x=xmin+t%rw, y=ymin+t/rw;
            const auto result=evaluate(local,static_cast<float>(x*16),static_cast<float>(y*16),15.0f,15.0f,true);
            if (result[0]<=threshold) {
                if (off<stop && off<keys.shape[0]) {
                    keys.data[off]=(uint64_t(y*gx+x)<<32)|__float_as_uint(result[1]+8.0f);
                    values.data[off]=i;
                }
                ++off;
            }
        }
    }
    // Each lane owns one Gaussian initially. Visit long rectangles in lane
    // order and use ballot ranks to retain row-major tile order within each.
    unsigned remaining=__ballot_sync(mask,active && count>32);
    while (remaining) {
        const int owner=__ffs(remaining)-1;
        const int id=__shfl_sync(mask,i,owner);
        const int x0=__shfl_sync(mask,xmin,owner), y0=__shfl_sync(mask,ymin,owner);
        const int row_width=__shfl_sync(mask,rw,owner), total=__shfl_sync(mask,count,owner);
        int pos=__shfl_sync(mask,off,owner);
        const int end=__shfl_sync(mask,stop,owner);
        const float limit=__shfl_sync(mask,threshold,owner);
        wp::mat_t<4,4,float> shared;
        #pragma unroll
        for (int r=0;r<4;++r) {
            #pragma unroll
            for (int c=0;c<4;++c) shared.data[r][c]=__shfl_sync(mask,local.data[r][c],owner);
        }
        for (int base=32;base<total;base+=32) {
            const int t=base+lane;
            const int x=x0+t%row_width, y=y0+t/row_width;
            wp::vec_t<2,float> result(0.0f);
            bool contributes=false;
            if (t<total) {
                result=evaluate(shared,static_cast<float>(x*16),static_cast<float>(y*16),15.0f,15.0f,false);
                contributes=result[0]<=limit;
            }
            const unsigned hits=__ballot_sync(mask,contributes);
            const int target=pos+__popc(hits & ((1u<<lane)-1u));
            if (contributes && target<end && target<keys.shape[0]) {
                keys.data[target]=(uint64_t(y*gx+x)<<32)|__float_as_uint(result[1]+8.0f);
                values.data[target]=id;
            }
            pos+=__popc(hits);
        }
        if (lane==owner) off=pos;
        remaining &= remaining-1;
    }
    if (active) {
        for (int j=off;j<min(stop,keys.shape[0]);++j) {
            keys.data[j]=(uint64_t(0xffffffffu)<<32)|__float_as_uint(3.4028234663852886e38f);
            values.data[j]=-1;
        }
    }
'''


@wp.func_native('auto evaluate = [](const wp::mat_t<4,4,float>& g2s, float x, float y, '
                'float width, float height, bool sequential) {\n' + FRUSTUM_DEPTH + '\n};\n' + _BODY)
def _duplicate_cooperative(i: int,
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    radii: wp.array(dtype=int), transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float), offsets: wp.array(dtype=int),
    width: int, height: int,
    keys: wp.array(dtype=wp.uint64), values: wp.array(dtype=int)):
    pass


@wp.kernel(module='unique', enable_backward=False, launch_bounds=32)
def duplicate_aaa_default(
    centers: wp.array(dtype=wp.vec2), extents: wp.array(dtype=wp.vec2),
    radii: wp.array(dtype=int), transforms: wp.array(dtype=wp.mat44),
    opacity3d: wp.array(dtype=float), offsets: wp.array(dtype=int),
    width: int, height: int,
    keys: wp.array(dtype=wp.uint64), values: wp.array(dtype=int)):
    # Launch a rounded number of threads: padded lanes must join collectives.
    _duplicate_cooperative(wp.tid(),centers,extents,radii,transforms,opacity3d,
                           offsets,width,height,keys,values)


def duplicate_dim(kernel, n):
    return ((n+31)//32)*32 if kernel is duplicate_aaa_default else n
