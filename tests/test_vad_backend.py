"""Architecture selection and fail-fast Docker startup for TEN VAD."""

import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from radiotelegram import docker_entrypoint


@pytest.mark.parametrize(
    "system,machine,backend",
    [
        ("linux", "aarch64", "ten_vad_python"),
        ("linux", "arm64", "ten_vad_python"),
        ("linux", "x86_64", "ten_vad"),
        ("darwin", "arm64", "ten_vad"),
    ],
)
def test_backend_selection(monkeypatch, system, machine, backend):
    selected = object()
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(docker_entrypoint.platform, "machine", lambda: machine)
    monkeypatch.setitem(sys.modules, "ten_vad", None)
    monkeypatch.setitem(sys.modules, "ten_vad_python", None)
    monkeypatch.setitem(
        sys.modules, backend, SimpleNamespace(TenVad=selected, VAD=selected)
    )

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "radiotelegram/vad.py")
    )
    assert module["TenVad"] is selected


@pytest.fixture
def arm_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(docker_entrypoint.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(sys, "argv", ["entrypoint", "python", "-m", "radiotelegram.main"])
    run = MagicMock()
    vad = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(docker_entrypoint.subprocess, "run", run)
    monkeypatch.setitem(sys.modules, "ten_vad_python", SimpleNamespace(VAD=vad))
    monkeypatch.setattr(docker_entrypoint.os, "execvp", execute)
    monkeypatch.setattr(docker_entrypoint, "Path", lambda _: tmp_path)
    monkeypatch.setattr(docker_entrypoint, "version", lambda _: "test-pybind11")
    (tmp_path / "build-inputs.sha256").write_text("inputs-v1")

    def compile_backend(*args, **kwargs):
        build = tmp_path / "build"
        build.mkdir(exist_ok=True)
        (build / "ten_vad_python.test.so").touch()

    run.side_effect = compile_backend
    return run, vad, execute


def test_arm_build_and_model_check_precede_app_start(arm_startup):
    run, vad, execute = arm_startup
    order = MagicMock()
    order.attach_mock(run, "build")
    order.attach_mock(vad, "vad")
    order.attach_mock(execute, "execute")

    docker_entrypoint.main()

    assert run.call_count == 2
    configure = run.call_args_list[0]
    assert f"-DPython_EXECUTABLE={sys.executable}" in configure.args[0]
    assert configure.kwargs["check"] is True
    assert run.call_args_list[1].kwargs["check"] is True
    np.testing.assert_array_equal(
        vad.return_value.process.call_args.args[0], np.zeros(256, dtype=np.int16)
    )
    assert vad.return_value.process.call_args.args[0].dtype == np.int16
    assert [call[0] for call in order.mock_calls] == [
        "build", "build", "vad", "vad().process", "execute"
    ]
    execute.assert_called_once_with("python", ["python", "-m", "radiotelegram.main"])


def test_failed_arm_build_stops_startup(arm_startup):
    run, vad, execute = arm_startup
    run.side_effect = subprocess.CalledProcessError(1, "cmake")
    with pytest.raises(subprocess.CalledProcessError):
        docker_entrypoint.main()
    vad.assert_not_called()
    execute.assert_not_called()


def test_cached_arm_backend_skips_cmake(arm_startup):
    run, vad, execute = arm_startup
    docker_entrypoint.main()
    run.reset_mock()
    vad.reset_mock()
    execute.reset_mock()

    docker_entrypoint.main()

    run.assert_not_called()
    vad.return_value.process.assert_called_once()
    execute.assert_called_once()


@pytest.mark.parametrize("changed", ["inputs", "python", "pybind11", "extension"])
def test_stale_arm_cache_rebuilds(monkeypatch, tmp_path, arm_startup, changed):
    run, _, _ = arm_startup
    docker_entrypoint.main()
    run.reset_mock()
    if changed == "inputs":
        (tmp_path / "build-inputs.sha256").write_text("inputs-v2")
    elif changed == "python":
        monkeypatch.setattr(sys, "version", "new-python-version")
    elif changed == "pybind11":
        monkeypatch.setattr(docker_entrypoint, "version", lambda _: "new-pybind11")
    else:
        (tmp_path / "build/ten_vad_python.test.so").unlink()

    docker_entrypoint.main()

    assert run.call_count == 2
    assert "--clean-first" in run.call_args.args[0]


def test_failed_rebuild_invalidates_old_cache(tmp_path, arm_startup):
    run, _, execute = arm_startup
    docker_entrypoint.main()
    execute.reset_mock()
    (tmp_path / "build-inputs.sha256").write_text("inputs-v2")
    run.side_effect = subprocess.CalledProcessError(1, "cmake")

    with pytest.raises(subprocess.CalledProcessError):
        docker_entrypoint.main()

    assert not (tmp_path / "build/.ready").exists()
    execute.assert_not_called()


def test_newly_built_backend_can_be_imported(monkeypatch, tmp_path, arm_startup):
    run, _, execute = arm_startup
    monkeypatch.delitem(sys.modules, "ten_vad_python")
    build = tmp_path / "build"
    # Isolate discovery from any real extension already installed in Docker.
    monkeypatch.setattr(sys, "path", [str(build)])
    assert importlib.util.find_spec("ten_vad_python") is None
    assert sys.path_importer_cache[str(build)] is None

    def compile_backend(*args, **kwargs):
        build.mkdir(exist_ok=True)
        (build / "ten_vad_python.py").write_text(
            "class VAD:\n"
            "    def process(self, frame):\n"
            "        return 0.0, False\n"
        )

    run.side_effect = compile_backend
    docker_entrypoint.main()
    execute.assert_called_once()


def test_failed_model_check_stops_startup(tmp_path, arm_startup):
    _, vad, execute = arm_startup
    vad.return_value.process.side_effect = RuntimeError("Model could not be loaded")
    with pytest.raises(RuntimeError, match="Model could not be loaded"):
        docker_entrypoint.main()
    execute.assert_not_called()
    assert not (tmp_path / "build/.ready").exists()


def test_x64_skips_build_and_passes_through_command(monkeypatch, arm_startup):
    run, vad, execute = arm_startup
    monkeypatch.setattr(docker_entrypoint.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "argv", ["entrypoint", "sh", "-c", "echo ready"])

    docker_entrypoint.main()

    run.assert_not_called()
    vad.assert_not_called()
    execute.assert_called_once_with("sh", ["sh", "-c", "echo ready"])
