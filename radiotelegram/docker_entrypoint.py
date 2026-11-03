"""Build the bundled TEN ONNX backend on ARM64 before starting the container."""

import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys


def prepare_vad():
    if sys.platform != "linux" or platform.machine() not in {"aarch64", "arm64"}:
        return

    root = Path("/opt/ten-vad")
    print("ARM64 detected: preparing TEN VAD ONNX backend", flush=True)
    subprocess.run(
        [
            "cmake",
            "-S", str(root / "source/examples_onnx/python"),
            "-B", str(root / "build"),
            f"-DORT_ROOT={root / 'onnxruntime'}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    # CMake reuses the compiled extension on subsequent container starts.
    subprocess.run(
        ["cmake", "--build", str(root / "build"), "--parallel", "2"],
        check=True,
    )
    # The build directory may have been absent when Python cached sys.path.
    importlib.invalidate_caches()

    import numpy as np
    from ten_vad_python import VAD

    # Verify the extension, runtime library and model before starting audio I/O.
    VAD().process(np.zeros(256, dtype=np.int16))
    print("TEN VAD ONNX backend ready", flush=True)


def main():
    prepare_vad()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
