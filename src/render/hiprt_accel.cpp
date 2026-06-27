/*
    hiprt_accel.cpp -- HIPRT acceleration structure builder.
*/

#include "hiprt/accel.h"
#include "hiprt/shapes.h"

#if defined(MI_ENABLE_AMD)

#include <mitsuba/core/logger.h>
#include <drjit-core/amd.h>
#include <drjit-core/jit.h>
#include <hiprt/hiprt.h>

#include <cstring>
#include <array>
#include <memory>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

static void hiprt_check(hiprtError result, const char *expr) {
    if (result != hiprtSuccess)
        Throw("HiprtAccel: %s failed with HIPRT error %u.",
              expr, (uint32_t) result);
}

#define MI_HIPRT_CHECK(expr) hiprt_check((expr), #expr)

struct DeviceAllocation {
    void *ptr = nullptr;

    DeviceAllocation() = default;

    explicit DeviceAllocation(size_t size) {
        if (size)
            ptr = jit_malloc(JitBackend::AMD, size);
    }

    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;

    DeviceAllocation(DeviceAllocation &&other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~DeviceAllocation() { reset(); }

    void reset() {
        if (ptr) {
            jit_free(ptr);
            ptr = nullptr;
        }
    }
};

template <typename T> DeviceAllocation upload_vector(const std::vector<T> &v) {
    DeviceAllocation result(v.size() * sizeof(T));
    if (!v.empty())
        jit_memcpy(JitBackend::AMD, result.ptr, v.data(),
                   v.size() * sizeof(T));
    return result;
}

struct HiprtFuncData {
    const void *data = nullptr;
    const uint32_t *lookup = nullptr;
};

static const char *hiprt_intersect_fn_names[HIPRT_ISECT_FN_COUNT] = {
    nullptr,
    "mi_hiprt_intersect_sphere",
    "mi_hiprt_intersect_disk",
    "mi_hiprt_intersect_cylinder",
    "mi_hiprt_intersect_ellipsoids",
    "mi_hiprt_intersect_sdfgrid",
    "mi_hiprt_intersect_linear_curve",
    "mi_hiprt_intersect_bspline_curve"
};

static const char *hiprt_filter_fn_names[HIPRT_ISECT_FN_COUNT] = {
    "mi_hiprt_filter_backface",
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

static const char *const hiprt_device_source = R"(
struct MiHiprtFuncData {
    const void *data;
    const uint32_t *lookup;
};

struct MiSphereData { float4 center_radius; };
struct MiDiskData { float4 to_object[3]; };
struct MiCylinderData { float4 to_object[3]; float4 params; };
struct MiEllipsoidData { float4 to_object[3]; };
struct MiLinearCurveData { float4 p0, p1; };
struct MiBSplineCurveData { float4 p0, p1, p2, p3; };
struct MiSDFGridHeader {
    uint32_t res_x, res_y, res_z, n_voxels;
    float voxel_size[3];
    float pad;
    float4 to_object[3];
};

__device__ inline const MiHiprtFuncData *mi_hiprt_func_data(const void *data) {
    return (const MiHiprtFuncData *) data;
}

__device__ inline uint32_t mi_hiprt_data_offset(const void *data,
                                                uint32_t instance_id) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    return fd->lookup ? fd->lookup[instance_id] : 0u;
}

__device__ inline float mi_hiprt_dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 mi_hiprt_add3(float3 a, float3 b) {
    return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ inline float3 mi_hiprt_sub3(float3 a, float3 b) {
    return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ inline float3 mi_hiprt_mul3(float3 a, float b) {
    return float3{ a.x * b, a.y * b, a.z * b };
}

__device__ inline float3 mi_hiprt_div3(float3 a, float3 b) {
    return float3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

__device__ inline float3 mi_hiprt_min3(float3 a, float3 b) {
    return float3{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

__device__ inline float3 mi_hiprt_max3(float3 a, float3 b) {
    return float3{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

__device__ inline float mi_hiprt_length3(float3 a) {
    return sqrtf(mi_hiprt_dot3(a, a));
}

__device__ inline bool mi_hiprt_solve_quadratic(float a, float b, float c,
                                                float &x0, float &x1) {
    bool linear_case = (a == 0.0f);
    if (linear_case && b == 0.0f)
        return false;
    x0 = x1 = -c / b;
    if (linear_case)
        return true;

    float discrim = fmaf(b, b, -4.0f * a * c);
    if (discrim < 0.0f)
        return false;

    float temp = -0.5f * (b + copysignf(sqrtf(discrim), b));
    float x0p = temp / a;
    float x1p = c / temp;
    x0 = fminf(x0p, x1p);
    x1 = fmaxf(x0p, x1p);
    return true;
}

__device__ inline float3 mi_hiprt_apply_point(const float4 *m, float3 p) {
    return float3{
        m[0].x*p.x + m[0].y*p.y + m[0].z*p.z + m[0].w,
        m[1].x*p.x + m[1].y*p.y + m[1].z*p.z + m[1].w,
        m[2].x*p.x + m[2].y*p.y + m[2].z*p.z + m[2].w
    };
}

__device__ inline float3 mi_hiprt_apply_vector(const float4 *m, float3 v) {
    return float3{
        m[0].x*v.x + m[0].y*v.y + m[0].z*v.z,
        m[1].x*v.x + m[1].y*v.y + m[1].z*v.z,
        m[2].x*v.x + m[2].y*v.y + m[2].z*v.z
    };
}

__device__ __attribute__((noinline))
bool mi_hiprt_filter_backface(const hiprtRay& ray,
                              const void*, void*,
                              const hiprtHit& hit) {
    return mi_hiprt_dot3(hit.normal, ray.direction) > 0.0f;
}

__device__ inline float3 mi_hiprt_float3(float4 v) {
    return float3{ v.x, v.y, v.z };
}

__device__ inline float3 mi_hiprt_normalize3(float3 v) {
    float len = mi_hiprt_length3(v);
    return len > 0.0f ? mi_hiprt_mul3(v, 1.0f / len)
                      : float3{ 0.0f, 0.0f, 0.0f };
}

__device__ inline bool mi_hiprt_curve_accept_hit(
        const hiprtRay& ray, float t, float v, float3 normal,
        float &best_t, float &best_v, float3 &best_n) {
    if (!(t >= ray.minT && t <= ray.maxT && t < best_t))
        return false;
    if (mi_hiprt_dot3(normal, ray.direction) >= 0.0f)
        return false;

    best_t = t;
    best_v = v;
    best_n = mi_hiprt_normalize3(normal);
    return true;
}

__device__ inline bool mi_hiprt_curve_sphere_hit(
        const hiprtRay& ray, float3 center, float radius, float v,
        float &best_t, float &best_v, float3 &best_n) {
    if (radius <= 0.0f)
        return false;

    float3 oc = mi_hiprt_sub3(ray.origin, center);
    float A = mi_hiprt_dot3(ray.direction, ray.direction);
    float B = 2.0f * mi_hiprt_dot3(oc, ray.direction);
    float C = mi_hiprt_dot3(oc, oc) - radius * radius;

    float t0, t1;
    if (!mi_hiprt_solve_quadratic(A, B, C, t0, t1))
        return false;

    bool found = false;
    float3 p0 = mi_hiprt_add3(oc, mi_hiprt_mul3(ray.direction, t0));
    found |= mi_hiprt_curve_accept_hit(
        ray, t0, v, mi_hiprt_mul3(p0, 1.0f / radius),
        best_t, best_v, best_n);
    float3 p1 = mi_hiprt_add3(oc, mi_hiprt_mul3(ray.direction, t1));
    found |= mi_hiprt_curve_accept_hit(
        ray, t1, v, mi_hiprt_mul3(p1, 1.0f / radius),
        best_t, best_v, best_n);
    return found;
}

__device__ inline bool mi_hiprt_curve_segment_hit(
        const hiprtRay& ray, float3 p0, float r0, float3 p1, float r1,
        float v0, float v1, bool endcaps,
        float &best_t, float &best_v, float3 &best_n) {
    float3 axis = mi_hiprt_sub3(p1, p0);
    float len = mi_hiprt_length3(axis);
    bool found = false;

    if (len <= 1e-7f) {
        return endcaps && mi_hiprt_curve_sphere_hit(
            ray, p0, fmaxf(r0, r1), v0, best_t, best_v, best_n);
    }

    float3 w = mi_hiprt_mul3(axis, 1.0f / len);
    float3 oc = mi_hiprt_sub3(ray.origin, p0);
    float oz = mi_hiprt_dot3(oc, w);
    float dz = mi_hiprt_dot3(ray.direction, w);
    float3 op = mi_hiprt_sub3(oc, mi_hiprt_mul3(w, oz));
    float3 dp = mi_hiprt_sub3(ray.direction, mi_hiprt_mul3(w, dz));
    float k = (r1 - r0) / len;
    float r_base = r0 + k * oz;

    float A = mi_hiprt_dot3(dp, dp) - k * k * dz * dz;
    float B = 2.0f * (mi_hiprt_dot3(op, dp) - r_base * k * dz);
    float C = mi_hiprt_dot3(op, op) - r_base * r_base;

    float t0, t1;
    if (mi_hiprt_solve_quadratic(A, B, C, t0, t1)) {
        float roots[2] = { t0, t1 };
        for (int i = 0; i < 2; ++i) {
            float t = roots[i];
            float h = oz + dz * t;
            if (!(h >= 0.0f && h <= len))
                continue;
            float local = h / len;
            float radius = r0 + (r1 - r0) * local;
            float3 q = mi_hiprt_add3(op, mi_hiprt_mul3(dp, t));
            float3 normal = mi_hiprt_sub3(q, mi_hiprt_mul3(w, radius * k));
            float v = v0 + (v1 - v0) * local;
            found |= mi_hiprt_curve_accept_hit(
                ray, t, v, normal, best_t, best_v, best_n);
        }
    }

    if (endcaps) {
        found |= mi_hiprt_curve_sphere_hit(
            ray, p0, r0, v0, best_t, best_v, best_n);
        found |= mi_hiprt_curve_sphere_hit(
            ray, p1, r1, v1, best_t, best_v, best_n);
    }

    return found;
}

__device__ inline float4 mi_hiprt_bspline_eval(
        const MiBSplineCurveData& c, float v) {
    float v2 = v * v, v3 = v2 * v;
    float b0 = (-v3 + 3.0f * v2 - 3.0f * v + 1.0f) * (1.0f / 6.0f);
    float b1 = (3.0f * v3 - 6.0f * v2 + 4.0f) * (1.0f / 6.0f);
    float b2 = (-3.0f * v3 + 3.0f * v2 + 3.0f * v + 1.0f) *
               (1.0f / 6.0f);
    float b3 = v3 * (1.0f / 6.0f);
    return float4{
        b0 * c.p0.x + b1 * c.p1.x + b2 * c.p2.x + b3 * c.p3.x,
        b0 * c.p0.y + b1 * c.p1.y + b2 * c.p2.y + b3 * c.p3.y,
        b0 * c.p0.z + b1 * c.p1.z + b2 * c.p2.z + b3 * c.p3.z,
        b0 * c.p0.w + b1 * c.p1.w + b2 * c.p2.w + b3 * c.p3.w
    };
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_sphere(const hiprtRay& ray,
                               const void* data, void*,
                               hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiSphereData *spheres = (const MiSphereData *) fd->data;
    const MiSphereData &s = spheres[mi_hiprt_data_offset(data, hit.instanceID) +
                                    hit.primID];
    float3 center = float3{ s.center_radius.x, s.center_radius.y,
                            s.center_radius.z };
    float radius = s.center_radius.w;

    float3 l = mi_hiprt_sub3(ray.origin, center);
    float3 d = ray.direction;
    float plane_t = mi_hiprt_dot3(mi_hiprt_mul3(l, -1.0f), d) /
                    mi_hiprt_length3(d);
    float3 plane_p = mi_hiprt_add3(ray.origin, mi_hiprt_mul3(d, plane_t));

    if (plane_t == 0.0f &&
        mi_hiprt_length3(mi_hiprt_sub3(plane_p, center)) > radius)
        return false;

    float3 o = mi_hiprt_sub3(plane_p, center);
    float A = mi_hiprt_dot3(d, d);
    float B = 2.0f * mi_hiprt_dot3(o, d);
    float C = mi_hiprt_dot3(o, o) - radius * radius;

    float near_t, far_t;
    bool ok = mi_hiprt_solve_quadratic(A, B, C, near_t, far_t);
    near_t += plane_t;
    far_t += plane_t;

    bool out_bounds = !(near_t <= ray.maxT && far_t >= 0.0f);
    bool in_bounds = near_t < ray.minT && far_t > ray.maxT;
    float t = near_t < 0.0f ? far_t : near_t;

    if (ok && !out_bounds && !in_bounds &&
        t >= ray.minT && t <= ray.maxT) {
        hit.t = t;
        hit.normal = mi_hiprt_mul3(
            mi_hiprt_sub3(mi_hiprt_add3(ray.origin,
                                        mi_hiprt_mul3(ray.direction, t)),
                          center),
            1.0f / radius);
        return true;
    }
    return false;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_disk(const hiprtRay& ray,
                             const void* data, void*,
                             hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiDiskData *disks = (const MiDiskData *) fd->data;
    const MiDiskData &d = disks[mi_hiprt_data_offset(data, hit.instanceID) +
                                hit.primID];

    float3 ro = mi_hiprt_apply_point(d.to_object, ray.origin);
    float3 rd = mi_hiprt_apply_vector(d.to_object, ray.direction);

    float t = -ro.z / rd.z;
    float3 local = mi_hiprt_add3(ro, mi_hiprt_mul3(rd, t));
    if (local.x * local.x + local.y * local.y <= 1.0f &&
        t >= ray.minT && t <= ray.maxT) {
        hit.t = t;
        hit.uv = float2{ local.x, local.y };
        hit.normal = float3{ 0.0f, 0.0f, 1.0f };
        return true;
    }
    return false;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_cylinder(const hiprtRay& ray,
                                 const void* data, void*,
                                 hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiCylinderData *cyls = (const MiCylinderData *) fd->data;
    const MiCylinderData &c = cyls[mi_hiprt_data_offset(data, hit.instanceID) +
                                  hit.primID];
    float length = c.params.x, radius = c.params.y;

    float3 ro = mi_hiprt_apply_point(c.to_object, ray.origin);
    float3 rd = mi_hiprt_apply_vector(c.to_object, ray.direction);

    float A = rd.x * rd.x + rd.y * rd.y;
    float B = 2.0f * (rd.x * ro.x + rd.y * ro.y);
    float C = ro.x * ro.x + ro.y * ro.y - radius * radius;

    float near_t, far_t;
    bool ok = mi_hiprt_solve_quadratic(A, B, C, near_t, far_t);

    bool out_bounds = !(near_t <= ray.maxT && far_t >= ray.minT);
    float z_pos_near = ro.z + rd.z * near_t;
    float z_pos_far = ro.z + rd.z * far_t;
    bool in_bounds = near_t < ray.minT && far_t > ray.maxT;
    bool valid =
        ok && !out_bounds && !in_bounds &&
        ((z_pos_near >= 0.0f && z_pos_near <= length && near_t > ray.minT) ||
         (z_pos_far >= 0.0f && z_pos_far <= length && far_t < ray.maxT));
    float t = (z_pos_near >= 0.0f && z_pos_near <= length && near_t >= 0.0f)
              ? near_t : far_t;

    if (valid && t >= ray.minT && t <= ray.maxT) {
        hit.t = t;
        hit.normal = float3{ ro.x + rd.x * t, ro.y + rd.y * t, 0.0f };
        return true;
    }
    return false;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_linear_curve(const hiprtRay& ray,
                                     const void* data, void*,
                                     hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiLinearCurveData *curves = (const MiLinearCurveData *) fd->data;
    const MiLinearCurveData &c =
        curves[mi_hiprt_data_offset(data, hit.instanceID) + hit.primID];

    float best_t = ray.maxT;
    float best_v = 0.0f;
    float3 best_n = float3{ 0.0f, 0.0f, 0.0f };
    bool found = mi_hiprt_curve_segment_hit(
        ray, mi_hiprt_float3(c.p0), c.p0.w,
        mi_hiprt_float3(c.p1), c.p1.w,
        0.0f, 1.0f, true, best_t, best_v, best_n);

    if (found) {
        hit.t = best_t;
        hit.uv = float2{ best_v, 0.0f };
        hit.normal = best_n;
    }
    return found;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_bspline_curve(const hiprtRay& ray,
                                      const void* data, void*,
                                      hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiBSplineCurveData *curves = (const MiBSplineCurveData *) fd->data;
    const MiBSplineCurveData &c =
        curves[mi_hiprt_data_offset(data, hit.instanceID) + hit.primID];

    float best_t = ray.maxT;
    float best_v = 0.0f;
    float3 best_n = float3{ 0.0f, 0.0f, 0.0f };
    bool found = false;

    constexpr int steps = 128;
    float u0 = 0.0f;
    float4 q0 = mi_hiprt_bspline_eval(c, u0);
    for (int i = 0; i < steps; ++i) {
        float u1 = (float) (i + 1) * (1.0f / (float) steps);
        float4 q1 = mi_hiprt_bspline_eval(c, u1);
        found |= mi_hiprt_curve_segment_hit(
            ray, mi_hiprt_float3(q0), q0.w,
            mi_hiprt_float3(q1), q1.w,
            u0, u1, false, best_t, best_v, best_n);
        q0 = q1;
        u0 = u1;
    }

    if (found) {
        hit.t = best_t;
        hit.uv = float2{ best_v, 0.0f };
        hit.normal = best_n;
    }
    return found;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_ellipsoids(const hiprtRay& ray,
                                   const void* data, void*,
                                   hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const MiEllipsoidData *ellis = (const MiEllipsoidData *) fd->data;
    const MiEllipsoidData &e =
        ellis[mi_hiprt_data_offset(data, hit.instanceID) + hit.primID];

    float3 ro = mi_hiprt_apply_point(e.to_object, ray.origin);
    float3 rd = mi_hiprt_apply_vector(e.to_object, ray.direction);

    float plane_t = mi_hiprt_dot3(mi_hiprt_mul3(ro, -1.0f), rd) /
                    mi_hiprt_length3(rd);
    float3 plane_p = mi_hiprt_add3(ro, mi_hiprt_mul3(rd, plane_t));

    if (plane_t == 0.0f && mi_hiprt_length3(plane_p) > 1.0f)
        return false;

    float A = mi_hiprt_dot3(rd, rd);
    float B = 2.0f * mi_hiprt_dot3(plane_p, rd);
    float C = mi_hiprt_dot3(plane_p, plane_p) - 1.0f;

    float near_t, far_t;
    bool ok = mi_hiprt_solve_quadratic(A, B, C, near_t, far_t);
    near_t += plane_t;
    far_t += plane_t;

    bool out_bounds = !(near_t <= ray.maxT && far_t >= 0.0f);
    bool in_bounds = near_t < 0.0f && far_t > ray.maxT;
    bool backfacing = near_t < 0.0f;

    if (ok && !out_bounds && !in_bounds && !backfacing &&
        near_t >= ray.minT && near_t <= ray.maxT) {
        hit.t = near_t;
        hit.normal = mi_hiprt_add3(ro, mi_hiprt_mul3(rd, near_t));
        return true;
    }
    return false;
}

__device__ inline bool mi_hiprt_sdf_intersect_aabb(
        float3 ro, float3 rd, float3 bbmin, float3 bbmax,
        float &t_min, float &t_max) {
    float3 inv_d = float3{ 1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z };
    float3 t1 = float3{ (bbmin.x - ro.x) * inv_d.x,
                        (bbmin.y - ro.y) * inv_d.y,
                        (bbmin.z - ro.z) * inv_d.z };
    float3 t2 = float3{ (bbmax.x - ro.x) * inv_d.x,
                        (bbmax.y - ro.y) * inv_d.y,
                        (bbmax.z - ro.z) * inv_d.z };
    float3 t_lo = mi_hiprt_min3(t1, t2);
    float3 t_hi = mi_hiprt_max3(t1, t2);
    t_min = fmaxf(fmaxf(t_lo.x, t_lo.y), t_lo.z);
    t_max = fminf(fminf(t_hi.x, t_hi.y), t_hi.z);
    return t_max >= fmaxf(t_min, 0.0f);
}

__device__ inline float mi_hiprt_sdf_eval(float t, float c3, float c2,
                                          float c1, float c0) {
    return fmaf(fmaf(fmaf(c3, t, c2), t, c1), t, c0);
}

__device__ inline float mi_hiprt_sdf_numerical_solve(
        float t_near, float t_far, float f_near, float f_far,
        float c3, float c2, float c1, float c0) {
    float t = 0.0f;
    for (int i = 0; i < 50; ++i) {
        t = t_near + (t_far - t_near) * (-f_near / (f_far - f_near));
        float f_t = mi_hiprt_sdf_eval(t, c3, c2, c1, c0);
        if (f_t * f_near <= 0.0f) {
            t_far = t;
            f_far = f_t;
        } else {
            t_near = t;
            f_near = f_t;
        }
        if (fabsf(t_far - t_near) < 1e-5f)
            break;
    }
    return t;
}

__device__ inline bool mi_hiprt_sdf_solve_cubic(
        float t_beg, float t_end, float c3, float c2, float c1, float c0,
        float &t) {
    float f_t_beg = mi_hiprt_sdf_eval(t_beg, c3, c2, c1, c0);
    float f_t_end = mi_hiprt_sdf_eval(t_end, c3, c2, c1, c0);
    float t_near = t_beg, f_near = f_t_beg;
    float t_far = t_end, f_far = f_t_end;

    float root_0, root_1;
    bool has_drv =
        mi_hiprt_solve_quadratic(c3 * 3.0f, c2 * 2.0f, c1,
                                 root_0, root_1);
    if (has_drv) {
        if (t_near <= root_0 && root_0 <= t_far) {
            float f_root_0 = mi_hiprt_sdf_eval(root_0, c3, c2, c1, c0);
            if (f_near * f_root_0 <= 0.0f) {
                t_far = root_0;
                f_far = f_root_0;
            } else {
                t_near = root_0;
                f_near = f_root_0;
            }
        }
        if (t_near <= root_1 && root_1 <= t_far) {
            float f_root_1 = mi_hiprt_sdf_eval(root_1, c3, c2, c1, c0);
            if (f_near * f_root_1 <= 0.0f) {
                t_far = root_1;
                f_far = f_root_1;
            } else {
                t_near = root_1;
                f_near = f_root_1;
            }
        }
    }

    if (f_near * f_far > 0.0f)
        return false;

    t = mi_hiprt_sdf_numerical_solve(t_near, t_far, f_near, f_far,
                                     c3, c2, c1, c0);
    return true;
}

__device__ __attribute__((noinline))
bool mi_hiprt_intersect_sdfgrid(const hiprtRay& ray,
                                const void* data, void*,
                                hiprtHit& hit) {
    const MiHiprtFuncData *fd = mi_hiprt_func_data(data);
    const uint8_t *buf = (const uint8_t *) fd->data;
    buf += mi_hiprt_data_offset(data, hit.instanceID);

    const MiSDFGridHeader *h = (const MiSDFGridHeader *) buf;
    const uint32_t *voxel_indices =
        (const uint32_t *) (buf + sizeof(MiSDFGridHeader));
    const float *grid = (const float *) (voxel_indices + h->n_voxels);

    uint32_t vi = voxel_indices[hit.primID];
    uint32_t x_len = h->res_x - 1u;
    uint32_t y_len = h->res_y - 1u;
    uint32_t vx = vi % x_len;
    uint32_t vy = ((vi - vx) / x_len) % y_len;
    uint32_t vz = (vi - vx - vy * x_len) / (x_len * y_len);

    float3 ro = mi_hiprt_apply_point(h->to_object, ray.origin);
    float3 rd = mi_hiprt_apply_vector(h->to_object, ray.direction);

    float3 vox = float3{ (float) vx, (float) vy, (float) vz };
    float3 vsize = float3{ h->voxel_size[0], h->voxel_size[1],
                           h->voxel_size[2] };
    float3 bbmin = float3{ vox.x * vsize.x, vox.y * vsize.y,
                           vox.z * vsize.z };
    float3 bbmax = mi_hiprt_add3(bbmin, vsize);

    float t_bbox_beg = 0.0f, t_bbox_end = 0.0f;
    if (!mi_hiprt_sdf_intersect_aabb(ro, rd, bbmin, bbmax,
                                     t_bbox_beg, t_bbox_end))
        return false;

    t_bbox_beg = fmaxf(t_bbox_beg, ray.minT);
    if (t_bbox_end < t_bbox_beg)
        return false;

    float3 inv_vs = float3{ 1.0f / vsize.x, 1.0f / vsize.y,
                            1.0f / vsize.z };
    float3 ro_v = mi_hiprt_sub3(mi_hiprt_div3(ro, vsize), vox);
    float3 rd_v = mi_hiprt_div3(rd, vsize);

    uint32_t sx = h->res_x, sy = h->res_y;
#define MI_SDF_SAMPLE(ix, iy, iz) grid[(iz) * sy * sx + (iy) * sx + (ix)]
    float s000 = MI_SDF_SAMPLE(vx,     vy,     vz);
    float s100 = MI_SDF_SAMPLE(vx + 1, vy,     vz);
    float s010 = MI_SDF_SAMPLE(vx,     vy + 1, vz);
    float s110 = MI_SDF_SAMPLE(vx + 1, vy + 1, vz);
    float s001 = MI_SDF_SAMPLE(vx,     vy,     vz + 1);
    float s101 = MI_SDF_SAMPLE(vx + 1, vy,     vz + 1);
    float s011 = MI_SDF_SAMPLE(vx,     vy + 1, vz + 1);
    float s111 = MI_SDF_SAMPLE(vx + 1, vy + 1, vz + 1);
#undef MI_SDF_SAMPLE

    float3 p0 = mi_hiprt_add3(ro_v, mi_hiprt_mul3(rd_v, t_bbox_beg));
    float o_x = p0.x, o_y = p0.y, o_z = p0.z;
    float d_x = rd_v.x, d_y = rd_v.y, d_z = rd_v.z;

    float a  = s101 - s001;
    float k0 = s000;
    float k1 = s100 - s000;
    float k2 = s010 - s000;
    float k3 = s110 - s010 - k1;
    float k4 = k0 - s001;
    float k5 = k1 - a;
    float k6 = k2 - (s011 - s001);
    float k7 = k3 - (s111 - s011 - a);
    float m0 = o_x * o_y;
    float m1 = d_x * d_y;
    float m2 = fmaf(o_x, d_y, o_y * d_x);
    float m3 = fmaf(k5, o_z, -k1);
    float m4 = fmaf(k6, o_z, -k2);
    float m5 = fmaf(k7, o_z, -k3);

    float c0 = fmaf(k4, o_z, -k0) +
               fmaf(o_x, m3, fmaf(o_y, m4, m0 * m5));
    float c1 = fmaf(d_x, m3, d_y * m4) + m2 * m5 +
               d_z * (k4 + fmaf(k5, o_x, fmaf(k6, o_y, k7 * m0)));
    float c2 = fmaf(m1, m5,
                    d_z * (fmaf(k5, d_x, fmaf(k6, d_y, k7 * m2))));
    float c3 = k7 * m1 * d_z;

    float t_beg = 0.0f;
    float t_end = t_bbox_end - t_bbox_beg;
    float t = 0.0f;
    bool ok = false;
    if (c3 != 0.0f) {
        ok = mi_hiprt_sdf_solve_cubic(t_beg, t_end, c3, c2, c1, c0, t);
    } else {
        float root_0, root_1;
        ok = mi_hiprt_solve_quadratic(c2, c1, c0, root_0, root_1);
        if (ok && t_beg <= root_0 && root_0 <= t_end)
            t = root_0;
        else if (ok && t_beg <= root_1 && root_1 <= t_end)
            t = root_1;
        else
            ok = false;
    }

    float final_t = t_bbox_beg + t;
    if (ok && t_beg <= t && t <= t_end &&
        final_t >= ray.minT && final_t <= ray.maxT) {
        hit.t = final_t;
        hit.normal = float3{ 0.0f, 0.0f, 1.0f };
        return true;
    }
    return false;
}
)";

struct HiprtAccelData {
    hiprtContext context = nullptr;
    hiprtScene scene = nullptr;
    hiprtFuncTable func_table = nullptr;
    std::vector<hiprtGeometry> geometries;
    std::vector<DeviceAllocation> allocations;

    ~HiprtAccelData() {
        if (context) {
            if (func_table)
                hiprtDestroyFuncTable(context, func_table);
            if (scene)
                hiprtDestroyScene(context, scene);
            for (hiprtGeometry geometry : geometries)
                if (geometry)
                    hiprtDestroyGeometry(context, geometry);
            hiprtDestroyContext(context);
        }
    }
};

static void fill_matrix_frame(hiprtFrameMatrix &frame, const float to_world[12]) {
    std::memset(&frame, 0, sizeof(frame));
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 4; ++col)
            frame.matrix[row][col] = to_world[col * 3 + row];
    frame.time = 0.f;
}

static hiprtGeometry build_triangle_geometry(hiprtContext context,
                                             const ShapeIR &g, bool compact,
                                             uint32_t geom_type) {
    if (g.kind != ShapeIR::Kind::Triangles &&
        g.kind != ShapeIR::Kind::TrianglesCulled)
        Throw("HiprtAccel: expected triangle mesh descriptor.");
    if (!g.vertex_ptr || !g.index_ptr || g.vertex_count == 0 ||
        g.face_count == 0)
        Throw("HiprtAccel: invalid triangle mesh descriptor.");
    if (g.vertex_count > UINT32_MAX || g.face_count > UINT32_MAX)
        Throw("HiprtAccel: triangle mesh is too large for HIPRT.");

    hiprtTriangleMeshPrimitive mesh {};
    mesh.vertices = const_cast<void *>(g.vertex_ptr);
    mesh.vertexCount = (uint32_t) g.vertex_count;
    mesh.vertexStride = 3 * sizeof(float);
    mesh.triangleIndices = const_cast<void *>(g.index_ptr);
    mesh.triangleCount = (uint32_t) g.face_count;
    mesh.triangleStride = 3 * sizeof(uint32_t);

    hiprtGeometryBuildInput input {};
    input.type = hiprtPrimitiveTypeTriangleMesh;
    input.geomType = geom_type;
    input.primitive.triangleMesh = mesh;

    hiprtBuildOptions options {};
    options.buildFlags = compact ? hiprtBuildFlagBitPreferHighQualityBuild
                                 : hiprtBuildFlagBitPreferFastBuild;

    size_t temp_size = 0;
    MI_HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(
        context, input, options, temp_size));
    DeviceAllocation temp(temp_size);

    hiprtGeometry geometry = nullptr;
    MI_HIPRT_CHECK(hiprtCreateGeometry(context, input, options, geometry));
    MI_HIPRT_CHECK(hiprtBuildGeometry(context, hiprtBuildOperationBuild,
                                      input, options, temp.ptr, nullptr,
                                      geometry));
    return geometry;
}

static hiprtGeometry build_aabb_geometry(hiprtContext context,
                                         const ShapeIR &g, bool compact,
                                         void *aabb_ptr,
                                         uint32_t geom_type) {
    bool aabb_kind = g.kind == ShapeIR::Kind::Custom ||
                     g.kind == ShapeIR::Kind::LinearCurve ||
                     g.kind == ShapeIR::Kind::BSplineCurve;
    if (!aabb_kind || !aabb_ptr || g.prim_count == 0)
        Throw("HiprtAccel: invalid custom AABB descriptor.");
    if (g.prim_count > UINT32_MAX)
        Throw("HiprtAccel: custom geometry is too large for HIPRT.");

    hiprtAABBListPrimitive list {};
    list.aabbs = aabb_ptr;
    list.aabbCount = (uint32_t) g.prim_count;
    list.aabbStride = 6 * sizeof(float);

    hiprtGeometryBuildInput input {};
    input.type = hiprtPrimitiveTypeAABBList;
    input.geomType = geom_type;
    input.primitive.aabbList = list;

    hiprtBuildOptions options {};
    options.buildFlags = compact ? hiprtBuildFlagBitPreferHighQualityBuild
                                 : hiprtBuildFlagBitPreferFastBuild;

    size_t temp_size = 0;
    MI_HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(
        context, input, options, temp_size));
    DeviceAllocation temp(temp_size);

    hiprtGeometry geometry = nullptr;
    MI_HIPRT_CHECK(hiprtCreateGeometry(context, input, options, geometry));
    MI_HIPRT_CHECK(hiprtBuildGeometry(context, hiprtBuildOperationBuild,
                                      input, options, temp.ptr, nullptr,
                                      geometry));
    return geometry;
}

static size_t align16(size_t v) { return (v + 15) & ~(size_t) 15; }

static uint32_t append_custom_data(
        const ShapeIR &g, uint32_t fn,
        std::array<std::vector<uint8_t>, HIPRT_ISECT_FN_COUNT> &data,
        std::array<size_t, HIPRT_ISECT_FN_COUNT> &elem_size,
        std::array<size_t, HIPRT_ISECT_FN_COUNT> &cursor) {
    size_t total = g.data_size_bytes();
    if (!g.fill_data || total == 0)
        Throw("HiprtAccel: custom shape type 0x%x has no primitive data.",
              (uint32_t) g.type);

    if (fn == HIPRT_ISECT_FN_SDFGRID) {
        size_t offset = cursor[fn];
        data[fn].resize(offset + align16(total));
        g.fill_data(g.ctx, data[fn].data() + offset);
        cursor[fn] += align16(total);
        return (uint32_t) offset;
    }

    if (!g.pdata_size || !g.prim_count)
        Throw("HiprtAccel: custom shape type 0x%x has invalid data layout.",
              (uint32_t) g.type);
    if (elem_size[fn] && elem_size[fn] != g.pdata_size)
        Throw("HiprtAccel: shape type 0x%x disagrees on data size.",
              (uint32_t) g.type);
    elem_size[fn] = g.pdata_size;

    size_t elem = cursor[fn];
    data[fn].resize((elem + g.prim_count) * elem_size[fn]);
    g.fill_data(g.ctx, data[fn].data() + elem * elem_size[fn]);
    cursor[fn] += g.prim_count;
    return (uint32_t) elem;
}

static std::pair<HiprtAccelData *, uint32_t>
build_impl(const SceneIR &sd, bool compact) {
    if (sd.instances.empty())
        Throw("HiprtAccel: scene description contains no instances.");

    // HIPRT builds consume buffers on the HIP context directly. Synchronize
    // pending Dr.Jit producers before handing their device pointers to HIPRT.
    jit_flush_thread();
    jit_eval();
    jit_amd_sync_device();

    auto accel = std::make_unique<HiprtAccelData>();

    hiprtContextCreationInput context_input {};
    context_input.ctxt = (hiprtApiCtx) jit_amd_context();
    context_input.device = (hiprtApiDevice) jit_amd_device_raw();
    context_input.deviceType = hiprtDeviceAMD;

    MI_HIPRT_CHECK(hiprtCreateContext(HIPRT_API_VERSION, context_input,
                                      accel->context));
    MI_HIPRT_CHECK(hiprtSetLogLevel(accel->context,
                                    hiprtLogLevelError | hiprtLogLevelWarn));

    std::vector<hiprtInstance> instances;
    std::vector<hiprtFrameMatrix> frames;
    std::array<std::vector<uint8_t>, HIPRT_ISECT_FN_COUNT> custom_data;
    std::array<size_t, HIPRT_ISECT_FN_COUNT> custom_elem_size {};
    std::array<size_t, HIPRT_ISECT_FN_COUNT> custom_cursor {};
    std::array<std::vector<uint32_t>, HIPRT_ISECT_FN_COUNT> custom_lookup;

    bool any_triangles = false;
    bool any_custom = false;
    bool any_backface_culled_triangles = false;
    size_t custom_geometry_count = 0;

    for (const InstanceEntry &inst : sd.instances) {
        const BlasEntry &blas = sd.blases[inst.blas_index];
        for (const ShapeIR &g : blas.geoms) {
            hiprtGeometry geometry = nullptr;
            bool lookup_pushed = false;

            switch (g.kind) {
                case ShapeIR::Kind::Triangles:
                    geometry = build_triangle_geometry(
                        accel->context, g, compact, hiprtInvalidValue);
                    any_triangles = true;
                    break;

                case ShapeIR::Kind::TrianglesCulled:
                    geometry = build_triangle_geometry(
                        accel->context, g, compact,
                        HIPRT_ISECT_FN_TRIANGLE_CULL);
                    any_triangles = true;
                    any_backface_culled_triangles = true;
                    break;

                case ShapeIR::Kind::Custom:
                case ShapeIR::Kind::BSplineCurve:
                case ShapeIR::Kind::LinearCurve: {
                    uint32_t fn = hiprt_fn_index(g.type);
                    if (fn >= HIPRT_ISECT_FN_COUNT)
                        Throw("HiprtAccel: shape type 0x%x has no HIPRT "
                              "intersection function.", (uint32_t) g.type);
                    if (g.prim_count == 0 || !g.fill_aabbs)
                        Throw("HiprtAccel: shape type 0x%x has no "
                              "AABB data.", (uint32_t) g.type);
                    if (g.prim_count > UINT32_MAX)
                        Throw("HiprtAccel: shape type 0x%x has too "
                              "many primitives for HIPRT.", (uint32_t) g.type);

                    std::vector<float> aabbs(g.prim_count * 6);
                    g.fill_aabbs(g.ctx, aabbs.data());
                    DeviceAllocation aabb_alloc = upload_vector(aabbs);
                    void *aabb_ptr = aabb_alloc.ptr;
                    accel->allocations.push_back(std::move(aabb_alloc));

                    uint32_t lookup_value = append_custom_data(
                        g, fn, custom_data, custom_elem_size, custom_cursor);

                    geometry = build_aabb_geometry(
                        accel->context, g, compact, aabb_ptr, fn);

                    for (auto &lookup : custom_lookup)
                        lookup.push_back(0u);
                    custom_lookup[fn].back() = lookup_value;
                    lookup_pushed = true;
                    any_custom = true;
                    ++custom_geometry_count;
                    break;
                }

                case ShapeIR::Kind::Instance:
                    Throw("HiprtAccel: instance geometry must be flattened "
                          "before reaching the HIPRT builder.");
            }

            if (!lookup_pushed)
                for (auto &lookup : custom_lookup)
                    lookup.push_back(0u);

            accel->geometries.push_back(geometry);

            hiprtInstance hiprt_instance {};
            hiprt_instance.type = hiprtInstanceTypeGeometry;
            hiprt_instance.geometry = geometry;
            instances.push_back(hiprt_instance);

            hiprtFrameMatrix frame {};
            fill_matrix_frame(frame, inst.to_world);
            frames.push_back(frame);
        }
    }

    if (instances.empty())
        Throw("HiprtAccel: no geometry was emitted.");

    if (custom_lookup[0].size() != instances.size())
        Throw("HiprtAccel: internal lookup table size mismatch.");

    if (any_custom || any_backface_culled_triangles) {
        std::array<const void *, HIPRT_ISECT_FN_COUNT> data_ptr {};
        std::array<const uint32_t *, HIPRT_ISECT_FN_COUNT> lookup_ptr {};

        for (uint32_t fn = 0; fn < HIPRT_ISECT_FN_COUNT; ++fn) {
            if (custom_data[fn].empty())
                continue;

            DeviceAllocation data_alloc(custom_data[fn].size());
            jit_memcpy(JitBackend::AMD, data_alloc.ptr, custom_data[fn].data(),
                       custom_data[fn].size());
            data_ptr[fn] = data_alloc.ptr;
            accel->allocations.push_back(std::move(data_alloc));

            DeviceAllocation lookup_alloc = upload_vector(custom_lookup[fn]);
            lookup_ptr[fn] = (const uint32_t *) lookup_alloc.ptr;
            accel->allocations.push_back(std::move(lookup_alloc));
        }

        std::vector<HiprtFuncData> func_data(HIPRT_ISECT_FN_COUNT);
        for (uint32_t fn = 0; fn < HIPRT_ISECT_FN_COUNT; ++fn)
            func_data[fn] = HiprtFuncData { data_ptr[fn], lookup_ptr[fn] };

        DeviceAllocation func_data_alloc = upload_vector(func_data);
        const HiprtFuncData *func_data_ptr =
            (const HiprtFuncData *) func_data_alloc.ptr;
        accel->allocations.push_back(std::move(func_data_alloc));

        MI_HIPRT_CHECK(hiprtCreateFuncTable(
            accel->context, HIPRT_ISECT_FN_COUNT, 1, accel->func_table));

        for (uint32_t fn = 0; fn < HIPRT_ISECT_FN_COUNT; ++fn) {
            if (fn != HIPRT_ISECT_FN_TRIANGLE_CULL && !data_ptr[fn])
                continue;

            hiprtFuncDataSet set {};
            if (data_ptr[fn])
                set.intersectFuncData = func_data_ptr + fn;
            MI_HIPRT_CHECK(hiprtSetFuncTable(
                accel->context, accel->func_table, fn, 0, set));
        }
    }

    DeviceAllocation instance_alloc = upload_vector(instances);
    DeviceAllocation frame_alloc = upload_vector(frames);

    hiprtSceneBuildInput scene_input {};
    scene_input.instances = instance_alloc.ptr;
    scene_input.instanceFrames = frame_alloc.ptr;
    scene_input.instanceCount = (uint32_t) instances.size();
    scene_input.frameCount = (uint32_t) frames.size();
    scene_input.frameType = hiprtFrameTypeMatrix;

    hiprtBuildOptions options {};
    options.buildFlags = compact ? hiprtBuildFlagBitPreferHighQualityBuild
                                 : hiprtBuildFlagBitPreferFastBuild;

    size_t scene_temp_size = 0;
    MI_HIPRT_CHECK(hiprtGetSceneBuildTemporaryBufferSize(
        accel->context, scene_input, options, scene_temp_size));
    DeviceAllocation scene_temp(scene_temp_size);

    MI_HIPRT_CHECK(hiprtCreateScene(accel->context, scene_input, options,
                                    accel->scene));
    MI_HIPRT_CHECK(hiprtBuildScene(accel->context, hiprtBuildOperationBuild,
                                   scene_input, options, scene_temp.ptr,
                                   nullptr, accel->scene));
    jit_amd_sync_device();

    accel->allocations.push_back(std::move(instance_alloc));
    accel->allocations.push_back(std::move(frame_alloc));

    uint32_t geom_mask = 0;
    if (any_triangles)
        geom_mask |= 0x1u;
    if (any_custom)
        geom_mask |= 0x2u;
    if (any_backface_culled_triangles)
        geom_mask |= 0x8u;

    uint32_t scene_index = accel->func_table
        ? jit_amd_configure_scene_ex(
              (void *) accel->scene, (void *) accel->func_table,
              HIPRT_ISECT_FN_COUNT, 1, hiprt_intersect_fn_names,
              hiprt_filter_fn_names, hiprt_device_source, geom_mask)
        : jit_amd_configure_scene((void *) accel->scene, geom_mask);
    jit_amd_scene_set_cleanup(
        scene_index, [](void *p) { delete (HiprtAccelData *) p; }, accel.get());

    Log(Debug, "HiprtAccel: built acceleration structures (%zu instances, "
               "%zu custom geometries%s)",
        instances.size(), custom_geometry_count,
        any_backface_culled_triangles ? ", culled triangles" : "");

    return { accel.release(), scene_index };
}

std::pair<HiprtAccelData *, uint32_t>
build_hiprt_accel(const SceneIR &sd, bool compact) {
    return build_impl(sd, compact);
}

void release_hiprt_accel(HiprtAccelData *accel, uint32_t scene_index) {
    if (scene_index)
        jit_var_dec_ref(scene_index);
    else
        delete accel;
}

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_AMD
