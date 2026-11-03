"""Select TEN VAD's native library or its Linux ARM64 ONNX extension."""

import platform
import sys

if sys.platform == "linux" and platform.machine() in {"aarch64", "arm64"}:
    from ten_vad_python import VAD as TenVad
else:
    from ten_vad import TenVad
