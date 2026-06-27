<!--
    AMD/HIPRT port note

    This branch carries an AMD GPU port of Mitsuba 3. Keep these build notes at
    the top so humans and automation agents see the branch-specific workflow
    before the upstream Mitsuba README below.
-->

# Mitsuba 3 AMD/HIPRT Port

This branch is an AMD GPU port of Mitsuba 3. It adds `amd_*` and `amd_ad_*`
variants backed by Dr.Jit's AMD/HIP backend and HIPRT ray tracing acceleration.
Use the steps below to build the port from a fresh checkout on Windows with a
ROCm SDK available through a Python virtual environment.

## AMD Build Steps

### 1. Clone and initialize submodules

```powershell
git clone --recursive -b jam/hip git@github.com:jammm/mitsuba3.git mitsuba3-amd
Set-Location mitsuba3-amd
git submodule update --init --recursive
```

For an existing checkout, update the branch and submodules to the recorded
commits:

```powershell
git checkout jam/hip
git pull --ff-only
git submodule update --init --recursive
```

### 2. Activate the compiler and ROCm environment

Run these commands from the repository root in PowerShell. The virtual
environment must provide the `rocm-sdk` command.

```powershell
# Activate Visual Studio compiler environment.
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' |
  ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
      [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
  }

# Locate and activate your Python virtual environment.
# Adjust $Venv if your venv lives elsewhere.
$Repo = (Get-Location).Path
$Venv = Join-Path (Split-Path $Repo -Parent) "venv"
. "$Venv\Scripts\Activate.ps1"

# Locate ROCm from the active Python environment.
$ROCM_ROOT = (rocm-sdk path --root).Trim()
$ROCM_BIN = (rocm-sdk path --bin).Trim()
$env:ROCM_HOME = $ROCM_ROOT
$env:HIP_PATH = $ROCM_ROOT
$env:PATH = "$ROCM_ROOT\lib\llvm\bin;$ROCM_BIN;$env:PATH"

# Build Python extensions with the same compiler environment.
$env:CC = "clang-cl"
$env:CXX = "clang-cl"
$env:DISTUTILS_USE_SDK = "1"
```

### 3. Build HIPRT once

Mitsuba links against the HIPRT import library in
`ext/hiprt/dist/bin/Release`. If that directory does not already contain
`hiprt*64.lib`, build HIPRT first:

```powershell
$HiprtBuild = Join-Path $Repo "build\hiprt"
cmake -S "$Repo\ext\hiprt" -B $HiprtBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHIP_PATH="$ROCM_ROOT" `
  -DBITCODE=ON `
  -DPRECOMPILE=ON `
  -DNO_UNITTEST=ON `
  -DFORCE_DISABLE_CUDA=ON

cmake --build $HiprtBuild --config Release --parallel $env:NUMBER_OF_PROCESSORS

# Preflight check expected by Mitsuba's AMD CMake path.
Get-ChildItem "$Repo\ext\hiprt\dist\bin\Release\hiprt*64.lib"
```

### 4. Configure Mitsuba with AMD variants

Use a fresh build directory, or delete `build\mitsuba_amd\mitsuba.conf` before
changing `MI_DEFAULT_VARIANTS`. Mitsuba only uses `MI_DEFAULT_VARIANTS` when it
creates a new `mitsuba.conf`.

```powershell
$Build = Join-Path $Repo "build\mitsuba_amd"
cmake -S $Repo -B $Build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DMI_DEFAULT_VARIANTS="scalar_rgb,amd_rgb,amd_ad_rgb,amd_spectral,amd_ad_spectral" `
  -DDRJIT_ENABLE_CUDA=OFF
```

The configure log should include lines similar to:

```text
Mitsuba: building the following variants:
 * amd_rgb
 * amd_ad_rgb
Dr.Jit: building the AMD/HIP backend.
Dr.Jit-Core: AMD/HIP GPU backend enabled.
Mitsuba: using HIPRT for AMD GPU ray tracing.
```

### 5. Build Mitsuba

```powershell
cmake --build $Build --config Release --parallel $env:NUMBER_OF_PROCESSORS
cmake --build $Build --config Release --parallel $env:NUMBER_OF_PROCESSORS --target copy-targets-drjit
cmake --build $Build --config Release --parallel $env:NUMBER_OF_PROCESSORS --target copy-targets
```

### 6. Run an AMD smoke render

Keep the ROCm and build outputs in `PATH` when running Python from the build
tree.

```powershell
$env:PATH = "$Build\Release;$ROCM_ROOT\lib\llvm\bin;$ROCM_BIN;$env:PATH"
$env:PYTHONPATH = "$Build\python;$Build\Release\python"
$env:HIPRT_PATH = "$Repo\ext\hiprt"

@'
import drjit as dr
import mitsuba as mi

mi.set_variant("amd_rgb")
print("variant", mi.variant())
print("amd backend", dr.has_backend(dr.JitBackend.AMD))

scene = mi.load_dict({
    "type": "scene",
    "integrator": { "type": "path" },
    "sensor": {
        "type": "perspective",
        "film": { "type": "hdrfilm", "width": 8, "height": 8 },
        "sampler": { "type": "independent", "sample_count": 1 },
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0])
    },
    "shape": {
        "type": "rectangle",
        "bsdf": {
            "type": "diffuse",
            "reflectance": { "type": "rgb", "value": [0.8, 0.2, 0.2] }
        }
    },
    "emitter": { "type": "constant", "radiance": { "type": "rgb", "value": [1, 1, 1] } }
})

image = mi.render(scene)
total = dr.sum(image.array)
dr.eval(total)
print("sum", total)
'@ | python -
```

Expected result: the script prints `variant amd_rgb`, reports the AMD backend
as available, and finishes with a finite image sum.

### 7. Run tests

For a broad Dr.Jit check, prefer a serial run first. Parallel `pytest-xdist`
runs can put substantial pressure on GPU memory and Windows TDR.

```powershell
python -m pytest "$Repo\ext\drjit\tests" --tb=short -ra --timeout=300
```

Then run Mitsuba tests relevant to your change. Keep the same `PATH`,
`PYTHONPATH`, `ROCM_HOME`, and `HIPRT_PATH` environment from the build step.

### Common pitfalls

- `MI_DEFAULT_VARIANTS` does not modify an existing `mitsuba.conf`; delete the
  build directory or the generated config before reconfiguring variants.
- `hiprt*64.lib` must exist below `ext/hiprt/dist/bin/Release` before Mitsuba
  configures the AMD render backend.
- The active Python environment must put `rocm-sdk` on `PATH`.
- Keep `$ROCM_ROOT\lib\llvm\bin` and the ROCm SDK bin directory in `PATH` while
  building and running tests.

---

<!-- <img src="https://github.com/mitsuba-renderer/mitsuba3/raw/master/docs/images/logo_plain.png" width="120" height="120" alt="Mitsuba logo"> -->

<img src="https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-data/master/docs/images/banners/banner_01.jpg"
alt="Mitsuba banner">

# Mitsuba Renderer 3

| Documentation  | Tutorial videos  | Linux             | MacOS             | Windows           |       PyPI        |
|      :---:     |      :---:       |       :---:       |       :---:       |       :---:       |       :---:       |
| [![docs][1]][2]| [![vids][9]][10] | [![rgl-ci][3]][4] | [![rgl-ci][5]][6] | [![rgl-ci][7]][8] | [![pypi][11]][12] |

[1]: https://readthedocs.org/projects/mitsuba/badge/?version=stable
[2]: https://mitsuba.readthedocs.io/en/stable/
[3]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_LinuxAmd64Clang10)/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_LinuxAmd64Clang10&guest=1
[5]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_LinuxAmd64gcc9)/statusIcon.svg
[6]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_LinuxAmd64gcc9&guest=1
[7]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_WindowsAmd64msvc2020)/statusIcon.svg
[8]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_WindowsAmd64msvc2020&guest=1
[9]: https://img.shields.io/badge/YouTube-View-green?style=plastic&logo=youtube
[10]: https://www.youtube.com/watch?v=9Ja9buZx0Cs&list=PLI9y-85z_Po6da-pyTNGTns2n4fhpbLe5&index=1
[11]: https://img.shields.io/pypi/v/mitsuba.svg?color=green
[12]: https://pypi.org/pypi/mitsuba

## Introduction

Mitsuba 3 is a research-oriented rendering system for forward and inverse light
transport simulation developed at [EPFL](https://www.epfl.ch) in Switzerland.
It consists of a core library and a set of plugins that implement functionality
ranging from materials and light sources to complete rendering algorithms.

Mitsuba 3 is *retargetable*: this means that the underlying implementations and
data structures can transform to accomplish various different tasks. For
example, the same code can simulate both scalar (classic one-ray-at-a-time) RGB transport
or differential spectral transport on the GPU. This all builds on
[Dr.Jit](https://github.com/mitsuba-renderer/drjit), a specialized *just-in-time*
(JIT) compiler developed specifically for this project.

## Main Features

- **Cross-platform**: Mitsuba 3 has been tested on Linux (``x86_64``), macOS
  (``aarch64``, ``x86_64``), and Windows (``x86_64``).

- **High performance**: The underlying Dr.Jit compiler fuses rendering code
  into kernels that achieve state-of-the-art performance using
  an LLVM backend targeting the CPU and a CUDA/OptiX backend
  targeting NVIDIA GPUs with ray tracing hardware acceleration.

- **Python first**: Mitsuba 3 is deeply integrated with Python. Materials,
  textures, and even full rendering algorithms can be developed in Python,
  which the system JIT-compiles (and optionally differentiates) on the fly.
  This enables the experimentation needed for research in computer graphics and
  other disciplines.

- **Differentiation**: Mitsuba 3 is a differentiable renderer, meaning that it
  can compute derivatives of the entire simulation with respect to input
  parameters such as camera pose, geometry, BSDFs, textures, and volumes. It
  implements recent differentiable rendering algorithms developed at EPFL.

- **Spectral & Polarization**: Mitsuba 3 can be used as a monochromatic
  renderer, RGB-based renderer, or spectral renderer. Each variant can
  optionally account for the effects of polarization if desired.

## Tutorial videos, documentation

We've recorded several [YouTube videos][10] that provide a gentle introduction
Mitsuba 3 and Dr.Jit. Beyond this you can find complete Juypter notebooks
covering a variety of applications, how-to guides, and reference documentation
on [readthedocs][2].

## Installation

We provide pre-compiled binary wheels via PyPI. Installing Mitsuba this way is as simple as running

```bash
pip install mitsuba
```

on the command line. The Python package includes thirteen variants by default:

- ``scalar_rgb``
- ``scalar_spectral``
- ``scalar_spectral_polarized``
- ``llvm_ad_rgb``
- ``llvm_ad_mono``
- ``llvm_ad_mono_polarized``
- ``llvm_ad_spectral``
- ``llvm_ad_spectral_polarized``
- ``cuda_ad_rgb``
- ``cuda_ad_mono``
- ``cuda_ad_mono_polarized``
- ``cuda_ad_spectral``
- ``cuda_ad_spectral_polarized``

The scalar variants perform one-ray-at-a-time simulations, while the LLVM and CUDA 
variants can be used for inverse rendering on the CPU or GPU respectively. To access additional 
variants, you will need to compile a custom version of Dr.Jit using CMake. Please see the
[documentation](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html)
for details on this.

### Requirements

- `Python >= 3.9`
- (optional) For computation on the GPU: `Nvidia driver >= 535`
- (optional) For vectorized / parallel computation on the CPU: `LLVM >= 11.1`

## Usage

Here is a simple "Hello World" example that shows how simple it is to render a
scene using Mitsuba 3 from Python:

```python
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
mi.set_variant('scalar_rgb')
# Load a scene
scene = mi.load_dict(mi.cornell_box())
# Render the scene
img = mi.render(scene)
# Write the rendered image to an EXR file
mi.Bitmap(img).write('cbox.exr')
```

Tutorials and example notebooks covering a variety of applications can be found
in the [documentation][2].

## About

This project was created by [Wenzel Jakob](https://rgl.epfl.ch/people/wjakob).
Significant features and/or improvements to the code were contributed by
[Sébastien Speierer](https://speierers.github.io/),
[Nicolas Roussel](https://github.com/njroussel),
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Delio Vicini](https://dvicini.github.io/),
[Tizian Zeltner](https://tizianzeltner.com/),
[Baptiste Nicolet](https://bnicolet.com/),
[Miguel Crespo](https://mcrespo.me/),
[Vincent Leroy](https://github.com/leroyvn), and
[Ziyi Zhang](https://github.com/ziyi-zhang).

When using Mitsuba 3 in academic projects, please cite:

```bibtex
@software{Mitsuba3,
    title = {Mitsuba 3 renderer},
    author = {Wenzel Jakob and Sébastien Speierer and Nicolas Roussel and Merlin Nimier-David and Delio Vicini and Tizian Zeltner and Baptiste Nicolet and Miguel Crespo and Vincent Leroy and Ziyi Zhang},
    note = {https://mitsuba-renderer.org},
    version = {3.8.0},
    year = 2022
}
```
