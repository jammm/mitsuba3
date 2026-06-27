# ----------------------------------------------------------
#  Mitsuba library
# ----------------------------------------------------------

# Some parts of Mitsuba use the TSL hash map class
include_directories(../ext/nanobind/ext/robin_map/include)

add_subdirectory(core)
add_subdirectory(render)

add_library(mitsuba SHARED)
target_link_libraries(mitsuba PUBLIC mitsuba-core)
target_link_libraries(mitsuba PUBLIC mitsuba-render)
set_target_properties(mitsuba PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

if (MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(MI_CLANGRT_BUILTINS "")
  if (DEFINED ENV{ROCM_HOME})
    file(TO_CMAKE_PATH "$ENV{ROCM_HOME}" MI_ROCM_HOME)
    file(GLOB MI_CLANGRT_BUILTINS
      "${MI_ROCM_HOME}/lib/llvm/lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
  endif()

  if (MI_CLANGRT_BUILTINS)
    list(GET MI_CLANGRT_BUILTINS 0 MI_CLANGRT_BUILTINS_LIB)
    target_link_libraries(mitsuba PRIVATE "${MI_CLANGRT_BUILTINS_LIB}")
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU)$")
  # Treat undefined symbols as errors when linking
  target_link_libraries(mitsuba PRIVATE -Wl,--no-undefined)
  # Silence an incorrect warning message in GCC LTO builds
  target_link_options(mitsuba PRIVATE -Walloc-size-larger-than=18446744073709551615)
endif()

# ----------------------------------------------------------
#  Mitsuba executable
# ----------------------------------------------------------

add_subdirectory(mitsuba)

# ----------------------------------------------------------
#  Plugins
# ----------------------------------------------------------

add_subdirectory(bsdfs)
add_subdirectory(emitters)
add_subdirectory(films)
add_subdirectory(integrators)
add_subdirectory(media)
add_subdirectory(phase)
add_subdirectory(rfilters)
add_subdirectory(samplers)
add_subdirectory(sensors)
add_subdirectory(shapes)
add_subdirectory(spectra)
add_subdirectory(textures)
add_subdirectory(volumes)
set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)

# ----------------------------------------------------------
#  Python bindings and extensions
# ----------------------------------------------------------

if (MI_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
