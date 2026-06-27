/*
    hiprt/accel.h -- Plain-C++ interface to the HIPRT acceleration builder.
*/

#pragma once

#if defined(MI_ENABLE_AMD)

#include <mitsuba/core/platform.h>
#include <mitsuba/render/scene_ir.h>
#include <utility>

NAMESPACE_BEGIN(mitsuba)

struct HiprtAccelData;

/// Build the lowered scene's triangle acceleration structures and register them
/// with Dr.Jit. Returns the owning data object and the scene JIT variable.
extern MI_EXPORT_LIB std::pair<HiprtAccelData *, uint32_t>
build_hiprt_accel(const SceneIR &sd, bool compact);

/// Drop the JIT scene reference and release HIPRT objects when no recording or
/// pending kernel keeps the scene alive.
extern MI_EXPORT_LIB void release_hiprt_accel(HiprtAccelData *accel,
                                              uint32_t scene_index);

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_AMD
