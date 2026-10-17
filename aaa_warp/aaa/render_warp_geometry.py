"""Readable ray/frustum equations with the renderer's arithmetic order.

The dot products are deliberately explicit: generic vec4 dot has a different
accumulation order. Only CUDA's rounded reciprocal needs a native primitive.
"""

import warp as wp


@wp.func_native("return __frcp_rn(value);")
def reciprocal(value: float) -> float:
    pass


@wp.struct
class Ray:
    position: wp.vec4
    squared_distance: float


@wp.func
def dot3(a: wp.vec3, b: wp.vec3):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def dot4(a: wp.vec4, b: wp.vec4):
    return (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3])


@wp.func
def transform(g: wp.mat44, p: wp.vec4):
    return wp.vec4(dot4(g[0], p), dot4(g[1], p), dot4(g[2], p), dot4(g[3], p))


@wp.func
def closest_ray(a: wp.vec4, b: wp.vec4):
    direction = wp.vec3(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )
    moment = wp.vec3(
        a[3] * b[0] - a[0] * b[3], a[3] * b[1] - a[1] * b[3], a[3] * b[2] - a[2] * b[3]
    )
    dd = dot3(direction, direction)
    scaled = wp.vec3(moment[0] / dd, moment[1] / dd, moment[2] / dd)
    r = Ray()
    r.position = wp.vec4(
        direction[1] * scaled[2] - scaled[1] * direction[2],
        direction[2] * scaled[0] - scaled[2] * direction[0],
        direction[0] * scaled[1] - scaled[0] * direction[1],
        1.0,
    )
    r.squared_distance = dot3(moment, scaled)
    return r


@wp.func
def sample(g: wp.mat44, x: float, y: float):
    r = closest_ray(g[0] - g[3] * x, g[1] - g[3] * y)
    depth = dot4(g[2], r.position) * reciprocal(dot4(g[3], r.position))
    return wp.vec2(-0.5 * r.squared_distance, depth)


@wp.func
def inside(p: wp.vec4, start: wp.vec3, extent: wp.vec3, axis: int):
    delta = p[axis] - start[axis] * p[3]
    return delta > 0.0 and delta < extent[axis] * p[3]


@wp.func
def test_plane(
    g: wp.mat44,
    a: wp.vec4,
    other_axis: int,
    start: wp.vec3,
    extent: wp.vec3,
    best: float,
):
    norm = a[3] / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    screen = transform(g, wp.vec4(-a[0] * norm, -a[1] * norm, -a[2] * norm, 1.0))
    value = a[3] * norm
    if (
        value < best
        and inside(screen, start, extent, other_axis)
        and inside(screen, start, extent, 2)
    ):
        best = value
    return best


@wp.func
def test_edge(
    g: wp.mat44, a: wp.vec4, b: wp.vec4, start: wp.vec3, extent: wp.vec3, best: float
):
    r = closest_ray(a, b)
    if r.squared_distance < best and inside(transform(g, r.position), start, extent, 2):
        best = r.squared_distance
    return best


@wp.func
def frustum(g: wp.mat44, x: float, y: float, width: float, height: float):
    start, extent = wp.vec3(x, y, -1.0), wp.vec3(width, height, 2.0)
    mean = wp.vec4(g[0, 3], g[1, 3], g[2, 3], g[3, 3])
    best = float(3.4028234663852886e38)
    if (
        inside(mean, start, extent, 0)
        and inside(mean, start, extent, 1)
        and inside(mean, start, extent, 2)
    ):
        best = 0.0
    else:
        dx = wp.copysign(width * 0.5, (mean[0] - x * mean[3]) - width * 0.5 * mean[3])
        dy = wp.copysign(height * 0.5, (mean[1] - y * mean[3]) - height * 0.5 * mean[3])
        ax = g[0] - g[3] * (x + width * 0.5 + dx)
        ay = g[1] - g[3] * (y + height * 0.5 + dy)
        best = test_plane(g, ax, 1, start, extent, best)
        best = test_plane(g, ay, 0, start, extent, best)
        best = test_edge(g, ax, ay, start, extent, best)
        best = test_edge(
            g, ax, g[1] - g[3] * (y + height * 0.5 - dy), start, extent, best
        )
        best = test_edge(
            g, g[0] - g[3] * (x + width * 0.5 - dx), ay, start, extent, best
        )
    return 0.5 * best
