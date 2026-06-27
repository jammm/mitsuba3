/*
    scene_hiprt.inl -- Templated half of the AMD HIPRT backend.
*/

#include <drjit-core/amd.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/scene_ir.h>
#include "hiprt/accel.h"

NAMESPACE_BEGIN(mitsuba)

/// Build raw recovery tables that match hiprt_accel.cpp's flattened HIPRT scene
/// instance order: for each SceneIR instance, one HIPRT instance per geometry in
/// the referenced BLAS.
static void build_hiprt_recovery_table_data(
    const SceneIR &sd, std::vector<uint32_t> &offsets,
    std::vector<uint32_t> &table, std::vector<uint32_t> &owners) {
    offsets.clear();
    table.clear();
    owners.clear();

    for (const InstanceEntry &inst : sd.instances) {
        uint32_t owner = inst.owner_registry_id == SCENE_IR_NO_OWNER
                             ? 0u
                             : inst.owner_registry_id;
        const BlasEntry &blas = sd.blases[inst.blas_index];
        for (const ShapeIR &g : blas.geoms) {
            offsets.push_back((uint32_t) table.size());
            table.push_back(jit_registry_id(g.ctx));
            owners.push_back(owner);
        }
    }
}

template <typename Float, typename Spectrum>
void HiprtAccel<Float, Spectrum>::init(Scene<Float, Spectrum> *scene,
                                       const Properties & /*props*/) {
    SceneIR sd = SceneIRBuilder<Float, Spectrum>::build(scene);

    if (sd.instances.empty()) {
        Log(Debug, "accel_init_hiprt(): scene contains no shapes.");
        return;
    }

    std::vector<uint32_t> offsets, table, owners;
    build_hiprt_recovery_table_data(sd, offsets, table, owners);
    using UInt32 = dr::uint32_array_t<Float>;
    geom_shape_offsets =
        dr::load<DynamicBuffer<UInt32>>(offsets.data(), offsets.size());
    geom_shape_table =
        dr::load<DynamicBuffer<UInt32>>(table.data(), table.size());
    instance_owner_table =
        dr::load<DynamicBuffer<UInt32>>(owners.data(), owners.size());

    std::tie(accel, scene_index) =
        build_hiprt_accel(sd, scene->m_compact_accel);

    accel_handle = UInt64::steal(jit_amd_scene_owner_handle(scene_index));
    uint32_t ft_handle = jit_amd_scene_func_table_handle(scene_index);
    if (ft_handle)
        func_table_handle = UInt64::steal(ft_handle);
    else
        func_table_handle = 0;
}

template <typename Float, typename Spectrum>
void HiprtAccel<Float, Spectrum>::rebuild(Scene<Float, Spectrum> *scene) {
    release();
    Properties props;
    init(scene, props);
}

template <typename Float, typename Spectrum>
void HiprtAccel<Float, Spectrum>::release() {
    if (!accel && scene_index == 0)
        return;
    accel_handle = 0;
    func_table_handle = 0;
    release_hiprt_accel(accel, scene_index);
    accel = nullptr;
    scene_index = 0;
}

template <typename Float, typename Spectrum>
void HiprtAccel<Float, Spectrum>::trace(const Ray3f &ray, Mask active,
                                        uint32_t out[8], bool shadow) const {
    using Single = dr::float32_array_t<Float>;
    dr::Array<Single, 3> ray_o(ray.o), ray_d(ray.d);
    Single ray_tmin(0.f), ray_tmax(ray.maxt);

    if constexpr (!std::is_same_v<Single, Float>)
        ray_tmax = dr::minimum(ray_tmax, dr::Largest<Single>);

    uint32_t args[8] = {
        ray_o.x().index(), ray_o.y().index(), ray_o.z().index(),
        ray_d.x().index(), ray_d.y().index(), ray_d.z().index(),
        ray_tmin.index(), ray_tmax.index()
    };

    jit_amd_ray_trace(8, args, active.index(), out, 8, scene_index, shadow);
}

template <typename Float, typename Spectrum>
typename HiprtAccel<Float, Spectrum>::PreliminaryIntersection3f
HiprtAccel<Float, Spectrum>::ray_intersect_preliminary(
    const Scene<Float, Spectrum> * /*scene*/, const Ray3f &ray, Mask /*coh*/,
    bool /*reorder*/, UInt32 /*reorder_hint*/, uint32_t /*reorder_hint_bits*/,
    Mask active) const {
    using Single = dr::float32_array_t<Float>;

    PreliminaryIntersection3f pi = dr::zeros<PreliminaryIntersection3f>();
    if (scene_index == 0)
        return pi;

    uint32_t out[8];
    trace(ray, active, out, /* shadow = */ false);

    Mask valid = Mask::steal(out[0]);

    pi.valid      = valid;
    pi.t          = Float(Single::steal(out[1]));
    pi.prim_uv    = Point2f(Float(Single::steal(out[2])),
                            Float(Single::steal(out[3])));
    pi.prim_index = UInt32::steal(out[5]);

    UInt32 instance_id = UInt32::steal(out[4]);
    UInt32 geometry_id = UInt32::steal(out[6]);
    UInt32 owner_instance_id = UInt32::steal(out[7]);

    UInt32 off      = dr::gather<UInt32>(geom_shape_offsets, instance_id, valid);
    UInt32 shape_id = dr::gather<UInt32>(geom_shape_table, off + geometry_id,
                                         valid);
    pi.shape = dr::reinterpret_array<ShapePtr, UInt32>(shape_id);

    UInt32 owner_id = dr::gather<UInt32>(instance_owner_table,
                                         owner_instance_id,
                                         valid);
    pi.instance = dr::reinterpret_array<ShapePtr, UInt32>(owner_id);

    return pi;
}

template <typename Float, typename Spectrum>
typename HiprtAccel<Float, Spectrum>::Mask
HiprtAccel<Float, Spectrum>::ray_test(const Scene<Float, Spectrum> * /*scene*/,
                                      const Ray3f &ray, Mask /*coherent*/,
                                      Mask active) const {
    if (scene_index == 0)
        return dr::zeros<Mask>(dr::width(ray.o));

    uint32_t out[8];
    trace(ray, active, out, /* shadow = */ true);
    return Mask::steal(out[0]);
}

template <typename Float, typename Spectrum>
typename HiprtAccel<Float, Spectrum>::SurfaceInteraction3f
HiprtAccel<Float, Spectrum>::ray_intersect_naive(
    const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask active) const {
    return scene->ray_intersect(ray, active);
}

NAMESPACE_END(mitsuba)
