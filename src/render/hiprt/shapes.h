/*
    hiprt/shapes.h -- HIPRT custom function indices for Mitsuba shapes.
*/

#pragma once

#if defined(MI_ENABLE_AMD)

#include <mitsuba/render/fwd.h>
#include <cstdint>

NAMESPACE_BEGIN(mitsuba)

enum HiprtIntersectionFn : uint32_t {
    HIPRT_ISECT_FN_TRIANGLE_CULL = 0,
    HIPRT_ISECT_FN_SPHERE        = 1,
    HIPRT_ISECT_FN_DISK          = 2,
    HIPRT_ISECT_FN_CYLINDER      = 3,
    HIPRT_ISECT_FN_ELLIPSOIDS    = 4,
    HIPRT_ISECT_FN_SDFGRID       = 5,
    HIPRT_ISECT_FN_LINEAR_CURVE  = 6,
    HIPRT_ISECT_FN_BSPLINE_CURVE = 7,
    HIPRT_ISECT_FN_COUNT         = 8
};

inline uint32_t hiprt_fn_index(ShapeType type) {
    switch (type) {
        case ShapeType::Sphere:     return HIPRT_ISECT_FN_SPHERE;
        case ShapeType::Disk:       return HIPRT_ISECT_FN_DISK;
        case ShapeType::Cylinder:   return HIPRT_ISECT_FN_CYLINDER;
        case ShapeType::Ellipsoids: return HIPRT_ISECT_FN_ELLIPSOIDS;
        case ShapeType::SDFGrid:    return HIPRT_ISECT_FN_SDFGRID;
        case ShapeType::LinearCurve: return HIPRT_ISECT_FN_LINEAR_CURVE;
        case ShapeType::BSplineCurve: return HIPRT_ISECT_FN_BSPLINE_CURVE;
        default:                    return HIPRT_ISECT_FN_COUNT;
    }
}

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_AMD
