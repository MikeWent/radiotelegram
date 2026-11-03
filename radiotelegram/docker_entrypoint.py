"""Cache the bundled TEN ONNX backend on ARM64 before starting the container."""

import importlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys


def prepare_vad():
    if sys.platform != "linux" or platform.machine() not in {"aarch64", "arm64"}:
        return

    root = Path("/opt/ten-vad")
    build = root / "build"
    stamp = build / ".ready"
    cache_key = "\n".join(
        [(root / "build-inputs.sha256").read_text(), sys.version, version("pybind11")]
    )
    if (
        stamp.is_file()
        and stamp.read_text() == cache_key
        and any(build.glob("ten_vad_python*.so"))
    ):
        print("Using cached TEN VAD ONNX backend", flush=True)
    else:
        stamp.unlink(missing_ok=True)
        print("ARM64 detected: building TEN VAD ONNX backend", flush=True)
        subprocess.run(
            [
                "cmake",
                "-S", str(root / "source/examples_onnx/python"),
                "-B", str(build),
                f"-DORT_ROOT={root / 'onnxruntime'}",
                f"-DPython_EXECUTABLE={sys.executable}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--clean-first", "--parallel", "2"],
            check=True,
        )
    # The build directory may have been absent when Python cached sys.path.
    importlib.invalidate_caches()

    import numpy as np
    from ten_vad_python import VAD

    # Verify the extension, runtime library and model before starting audio I/O.
    VAD().process(np.zeros(256, dtype=np.int16))
    stamp.write_text(cache_key)
    print("TEN VAD ONNX backend ready", flush=True)


def main():
    prepare_vad()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
