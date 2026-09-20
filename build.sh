#!/usr/bin/env bash
# Release x64 build of Magpie with the AMD / FSR3 / XeSS-FG feature set this fork needs.
#
# The feature flags are not defaults in BuildOptions.props, so a plain MSBuild of the
# project compiles the FSR3 stub instead of the real implementation and then fails on
# Magpie.Core's unused `resources` parameter. These are the flags the release build uses.
#
# Usage (from magpie/):  ./build.sh [Magpie.Core|Magpie|all]      default: all
set -o pipefail

ROOT='H:\orcaworkspace\magpie-dlss5-amd'
MSB="/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/amd64/MSBuild.exe"
FSR3_SDK="$ROOT\\presr-src\\dlss-5-amd-project-1.7.3\\OptiScaler-DLSSNR-PreSR-Multipass-main\\external\\FidelityFX-SDK-v2"

ARGS=(
  -p:Configuration=Release
  -p:Platform=x64
  -p:SolutionDir="$ROOT\\magpie\\"
  -p:EnableFSR3ZeroMV=true
  -p:EnableAmdOpticalFlow=true
  -p:EnableXeSSFrameGeneration=true
  -p:FSR3SdkDir="$FSR3_SDK"
  -p:XeSSSdkDir="$ROOT\\deps\\xess"
  -v:m -nologo
)

target="${1:-all}"
if [ "$target" != "Magpie" ]; then
  "$MSB" src/Magpie.Core/Magpie.Core.vcxproj "${ARGS[@]}" || exit 1
fi
if [ "$target" != "Magpie.Core" ]; then
  taskkill //IM Magpie.exe //F >/dev/null 2>&1
  "$MSB" src/Magpie/Magpie.vcxproj "${ARGS[@]}" || exit 1
fi
