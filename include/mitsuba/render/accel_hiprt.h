/*
    accel_hiprt.h -- HIPRT acceleration backend declarations.
*/

#pragma once

#if defined(MI_ENABLE_AMD)

#include <mitsuba/render/fwd.h>
#include <drjit/array_traverse.h>

NAMESPACE_BEGIN(mitsuba)

/// Opaque handle owning the native HIPRT objects, see src/render/hiprt_accel.cpp
struct HiprtAccelData;

/// Vectorized GPU ray tracing acceleration via AMD HIPRT
template <typename Float, typename Spectrum>
struct HiprtAccel {
    MI_IMPORT_TYPES(Shape, ShapePtr)
    HiprtAccel() = default;
    DRJIT_NON_COPYABLE(HiprtAccel)

    ~HiprtAccel() { release(); }

    void init(Scene<Float, Spectrum> *scene, const Properties &props);
    void rebuild(Scene<Float, Spectrum> *scene);
    void release();

    static void static_initialization() { }
    static void static_shutdown() { }

    PreliminaryIntersection3f ray_intersect_preliminary(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask coherent,
        bool reorder, UInt32 reorder_hint, uint32_t reorder_hint_bits,
        Mask active) const;
    Mask ray_test(const Scene<Float, Spectrum> *scene, const Ray3f &ray,
                  Mask coherent, Mask active) const;
    SurfaceInteraction3f ray_intersect_naive(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray,
        Mask active) const;

    DRJIT_TRAVERSE(HiprtAccel, accel_handle, func_table_handle,
                   geom_shape_offsets,
                   geom_shape_table, instance_owner_table)

    /// Opaque handle owning the HIPRT context, scene, geometries, and buffers.
    HiprtAccelData *accel = nullptr;
    /// Dr.Jit scene id from jit_amd_configure_scene(), 0 for empty scenes.
    uint32_t scene_index = 0;
    /// Handle variable representing the HIPRT scene for @dr.freeze.
    UInt64 accel_handle;
    /// Handle variable representing the optional HIPRT function table.
    UInt64 func_table_handle;
    /// Per-HIPRT-instance recovery tables resolving \c pi.shape from a hit's
    /// (instance_id, geometry_id). The first milestone flattens every Mitsuba
    /// geometry into one HIPRT scene instance, so geometry_id is normally zero.
    DynamicBuffer<UInt32> geom_shape_offsets;
    DynamicBuffer<UInt32> geom_shape_table;
    /// Per-HIPRT-instance owner registry id, or zero for top-level shapes.
    DynamicBuffer<UInt32> instance_owner_table;

private:
    void trace(const Ray3f &ray, Mask active, uint32_t out[8],
               bool shadow) const;
};

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_AMD
