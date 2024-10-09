# Development

Install required tools:

- go version 1.22 or higher
- gcc version 11.4.0 or higher


### MacOS

[Download Go](https://go.dev/dl/)

Optionally enable debugging and more verbose logging:

```bash
# At build time
export CGO_CFLAGS="-g"

# At runtime
export OLLAMA_DEBUG=1
```

Get the required libraries and build the native LLM code:  (Adjust the job count based on your number of processors for a faster build)

```bash
make -j 5
```

Then build ollama:

```bash
go build .
```

Now you can run `ollama`:

```bash
./ollama
```

#### Xcode 15 warnings

If you are using Xcode newer than version 14, you may see a warning during `go build` about `ld: warning: ignoring duplicate libraries: '-lobjc'` due to Golang issue https://github.com/golang/go/issues/67799 which can be safely ignored.  You can suppress the warning with `export CGO_LDFLAGS="-Wl,-no_warn_duplicate_libraries"`

### Linux

#### Linux CUDA (NVIDIA)

_Your operating system distribution may already have packages for NVIDIA CUDA. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install `make`, `gcc` and `golang` as well as [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
development and runtime packages.

Typically the build scripts will auto-detect CUDA, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `CUDA_LIB_DIR` to the location of the shared
libraries, and `CUDACXX` to the location of the nvcc compiler. You can customize
a set of target CUDA architectures by setting `CMAKE_CUDA_ARCHITECTURES` (e.g. "50;60;70")

Then generate dependencies:  (Adjust the job count based on your number of processors for a faster build)

```
make -j 5
```

Then build the binary:

```
go build .
```

#### Linux ROCm (AMD)

_Your operating system distribution may already have packages for AMD ROCm and CLBlast. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install [CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md) and [ROCm](https://rocm.docs.amd.com/en/latest/) development packages first, as well as `make`, `gcc`, and `golang`.

Typically the build scripts will auto-detect ROCm, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `ROCM_PATH` to the location of the ROCm
install (typically `/opt/rocm`), and `CLBlast_DIR` to the location of the
CLBlast install (typically `/usr/lib/cmake/CLBlast`). You can also customize
the AMD GPU targets by setting AMDGPU_TARGETS (e.g. `AMDGPU_TARGETS="gfx1101;gfx1102"`)

Then generate dependencies:  (Adjust the job count based on your number of processors for a faster build)

```
make -j 5
```

Then build the binary:

```
go build .
```

ROCm requires elevated privileges to access the GPU at runtime. On most distros you can add your user account to the `render` group, or run as root.

#### Advanced CPU Settings

By default, running `make` will compile a few different variations
of the LLM library based on common CPU families and vector math capabilities,
including a lowest-common-denominator which should run on almost any 64 bit CPU
somewhat slowly. At runtime, Ollama will auto-detect the optimal variation to
load. 

Custom CPU settings are not currently supported in the new Go server build but will be added back after we complete the transition.

#### Containerized Linux Build

If you have Docker available, you can build linux binaries with `OLLAMA_NEW_RUNNERS=1 ./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included. The resulting binary is placed in `./dist`

### Windows

The following tools are required as a minimal development environment to build CPU inference support.

- Go version 1.22 or higher
  - https://go.dev/dl/
- Git
  - https://git-scm.com/download/win
- GCC and Make.  There are multiple options on how to go about installing these tools on Windows.  We have verified the following, but others may work as well:  
  - [MSYS2](https://www.msys2.org/)
    - After installing, from an MSYS2 terminal, run `pacman -S mingw-w64-ucrt-x86_64-gcc make` to install the required tools
  - Assuming you used the default install prefix for msys2 above, add `c:\msys64\ucrt64\bin` and `c:\msys64\usr\bin` to your environment variable `PATH` where you will perform the build steps below (e.g. system-wide, account-level, powershell, cmd, etc.)

Then, build the `ollama` binary:

```powershell
$env:CGO_ENABLED="1"
make -j 8
go build .
```

#### GPU Support

The GPU tools require the Microsoft native build tools.  To build either CUDA or ROCm, you must first install MSVC via Visual Studio:

- Make sure to select `Desktop development with C++` as a Workload during the Visual Studio install
- You must complete the Visual Studio install and run it once **BEFORE** installing CUDA or ROCm for the tools to properly register
- Add the location of the **64 bit (x64)** compiler (`cl.exe`) to your `PATH`
- Note: the default Developer Shell may configure the 32 bit (x86) compiler which will lead to build failures.  Ollama requires a 64 bit toolchain.

#### Windows CUDA (NVIDIA)

In addition to the common Windows development tools and MSVC described above:

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

#### Windows ROCm (AMD Radeon)

In addition to the common Windows development tools and MSVC described above:

- [AMD HIP](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)

#### Windows arm64

The default `Developer PowerShell for VS 2022` may default to x86 which is not what you want.  To ensure you get an arm64 development environment, start a plain PowerShell terminal and run:

```powershell
import-module 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -Arch arm64 -vsinstallpath 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community' -skipautomaticlocation
```

You can confirm with `write-host $env:VSCMD_ARG_TGT_ARCH`

Follow the instructions at https://www.msys2.org/wiki/arm64/ to set up an arm64 msys2 environment.  Ollama requires gcc and mingw32-make to compile, which is not currently available on Windows arm64, but a gcc compatibility adapter is available via `mingw-w64-clang-aarch64-gcc-compat`. At a minimum you will need to install the following:

```
pacman -S mingw-w64-clang-aarch64-clang mingw-w64-clang-aarch64-gcc-compat mingw-w64-clang-aarch64-make make
```

You will need to ensure your PATH includes go, cmake, gcc and clang mingw32-make to build ollama from source. (typically `C:\msys64\clangarm64\bin\`)

## Vendoring

Ollama is designed to support multiple LLM backends, and currently utilizes [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [ggml](https://github.com/ggerganov/ggml) through a vendoring model.  While we generally strive to contribute changes back upstream to avoid drift, we cary a small set of patches which are applied to the tracking commit.  A set of make targets are available to aid developers in updating to a newer tracking commit, or to work on changes.

If you will be updating the vendoring code, you should start by running the following command to establish the tracking llama.cpp repo at the top of your ollama repo.

```
make apply-patches
```

### Updating Base Commit

**Pin to new base commit**

To update to a newer base commit, select the upstream git tag or commit and update `llama/vendoring.env` at the top of the ollama tree

**Apply Ollama Patches**

When updating to a newer base commit, the existing patches may not apply cleanly and require manual merge resolution.  In the following example, we'll assume patch 0001 applies cleanly, 0002 needs adjustments, and 0003 is clean as well.

Start by applying the patches.  In our example scenario, 0001 applies, but 0002 fails.  The `apply-patches` target will stop at first failure.

```
make apply-patches
```

Now go into the llama.cpp tracking repo at the top of the ollama repo, and perform merge resolution using your preferred tool to the patch commit which failed (e.g. 0002)  Once that commit is resolved and commited, bring the refreshed patch back to ./llama/patches by running:

```
make patch
```

This will refresh both 0001 and 0002 in our example scenario.  0001 may have minor line number changes (or no changes at all) while 0002 now contains your adjustments so it applies cleanly.  No other patches will be affected.  Now you can re-run the `apply-patches` target, which will succeed for 0001, 0002, and 0003

```
make apply-patches
```

Continue iterating until `apply-patches` succeeds at applying **all** the patches.  Once finished, run a final `patch` target to ensure everything is updated.

```
make patch
```

Build and test Ollama, and make any necessary changes to the Go code based on the new base commit.  Submit your PR to the Ollama repo.

### New Development

When working on new fixes or features that impact vendored code, use the following model.  First get a clean tracking repo with all current patches applied:

```
make apply-patches
```

Now edit the upstream native code in the llama.cpp tracking repo at the top of the ollama repo.  You do not need to commit every change in order to build, a dirty working tree in the tracking repo is OK while developing.  Simply save in your editor, and run the following to refresh the vendored code with your changes, build the backend(s) and build ollama:

```
make sync
make -j 8
go build .
```

> [!IMPORTANT]
> Do **NOT** run `apply-patches` while you're iterating as that will reset the tracking repo.  It will detect a dirty tree and abort, but if your tree is clean and you accidentally ran this target, use `git reflog` to recover your commit(s).

Iterate until you're ready to submit PRs.  Once your code is ready, commit a change in the llama.cpp tracking repo, then generate a new patch for ollama with

```
make patch
```

> [!IMPORTANT]
> Once you have completed this step, it is safe to run `apply-patches` since your change is preserved as a new patch.

In your llama.cpp tracking repo, create a branch, and cherry-pick the new commit to that branch, then submit a PR upstream to llama.cpp.

Commit the changes in the ollama repo and submit a PR to Ollama, which will include the vendored code update with your change, along with a new patch.

After your PR upstream is merged, follow the **Updating Base Commit** instructions above, however first remove your patch before running `apply-patches` since the new base commit contains your change already.