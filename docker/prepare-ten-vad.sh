#!/bin/sh
# Bundle pinned ARM64 build inputs so container startup needs no network.
set -eu

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) exit 0 ;;
esac

ten_vad_revision=d68ab71c8d5fbd8f98e97ae7c9986af76b5d72b1
onnxruntime_version=1.22.0
mkdir -p /opt/ten-vad/source /opt/ten-vad/onnxruntime
cd /opt/ten-vad

curl --fail --location --retry 3 --output source.tar.gz \
    "https://codeload.github.com/TEN-framework/ten-vad/tar.gz/$ten_vad_revision"
echo '65b711fff6086c35d7f7255ec32bde22a80b7b52d4ead5e14097d28d4d562f85  source.tar.gz' | sha256sum --check -
tar -xzf source.tar.gz --strip-components=1 -C source

curl --fail --location --retry 3 --output onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v$onnxruntime_version/onnxruntime-linux-aarch64-$onnxruntime_version.tgz"
echo 'bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702  onnxruntime.tgz' | sha256sum --check -
tar -xzf onnxruntime.tgz --strip-components=1 -C onnxruntime

# Upstream uses a model path relative to cwd. Make it independent of WORKDIR.
sed -i 's|"onnx_model/ten-vad.onnx"|"/opt/ten-vad/source/src/onnx_model/ten-vad.onnx"|' source/src/aed.cc
rm source.tar.gz onnxruntime.tgz
