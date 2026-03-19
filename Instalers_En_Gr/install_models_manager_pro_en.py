#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models Manager Pro Installer (English)

Installs the Python packages used by Models Manager Pro and automatically chooses
CPU or NVIDIA GPU variants where appropriate.
"""

from __future__ import annotations

import argparse
import importlib
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

SCRIPT_VERSION = "1.2"

MESSAGES = {'title': 'Models Manager Pro - Python Package Installer', 'subtitle': 'English installer with automatic CPU / NVIDIA GPU package selection', 'argparse_desc': 'Install the Python dependencies required by Models Manager Pro.', 'help_force_gpu': 'Force GPU package installation even if auto-detection fails.', 'help_force_cpu': 'Force CPU-only package installation.', 'help_skip_optional': 'Skip optional feature packages such as TensorRT, Triton, NCNN, ONNXSIM and tiktoken.', 'help_dry_run': 'Print the commands without executing them.', 'run': '> {command}', 'cmd_failed': 'Command failed with exit code {code}: {command}', 'force_gpu_no_detect': 'GPU mode was forced even though no NVIDIA GPU was auto-detected.', 'gpu_platform_not_supported': "GPU auto-install is not enabled for platform '{platform}', falling back to CPU packages.", 'cuda_too_old': "Detected CUDA capability '{cuda}' is too old for the current prebuilt PyTorch CUDA wheels; falling back to CPU packages.", 'installing_torch': 'Installing PyTorch and torchvision ({variant})...', 'installing_ort_gpu': 'Installing ONNX Runtime GPU package...', 'installing_ort_cpu': 'Installing ONNX Runtime CPU package...', 'installing_base': 'Installing core application packages...', 'skip_optional': 'Optional packages were skipped by request.', 'installing_optional': 'Installing optional / feature-specific packages when they are missing...', 'skip_tensorrt': 'Skipping TensorRT automatic install on platform={platform}, machine={machine}.', 'py_info': 'Python: {python} | Executable: {exe}', 'os_info': 'OS: {os_name} {release} | Architecture: {machine}', 'gpu_found': 'NVIDIA GPU detected: {gpus} | CUDA reported by nvidia-smi: {cuda}', 'unknown_gpu': 'Unknown NVIDIA GPU', 'unknown': 'unknown', 'gpu_not_found': 'No NVIDIA GPU was detected. CPU packages will be used.', 'chosen_variant': 'Selected compute variant: {variant}', 'python_too_old': 'Python {version} is too old. Current stable PyTorch requires Python 3.10 or later.', 'upgrade_tools': 'Upgrading pip / setuptools / wheel...', 'required_failures': 'Required package installation errors:', 'optional_failures': 'Optional package installation warnings:', 'missing_required_imports': 'Missing required imports after installation: {items}', 'missing_optional_imports': 'Optional imports that are still unavailable: {items}', 'torch_version': 'Installed torch version: {version}', 'torch_cuda': 'torch.cuda.is_available(): {value}', 'torch_gpu_name': 'PyTorch GPU device: {name}', 'done_with_errors': 'Installation finished with errors. Please review the messages above.', 'done_partial': 'Installation finished. Core packages are installed, but some optional feature packages are unavailable.', 'done_success': 'Installation completed successfully. All core packages imported correctly.', 'done_dry_run': 'Dry run completed. No changes were made.', 'help_skip_cmake': 'Skip the automatic CMake check and installation step.', 'help_cmake_msi_path': 'Use a specific local path to cmake-4.3.0-windows-x86_64.msi.', 'checking_cmake': 'Checking whether CMake 4.3.0 or newer is already available...', 'cmake_found': 'CMake is already available: {version}', 'cmake_too_old': 'CMake {version} is installed, but version 4.3.0 or newer is required. The installer will upgrade it.', 'cmake_missing': 'CMake was not found. The script will install CMake 4.3.0 before installing Python packages.', 'cmake_skip_requested': 'CMake installation was skipped by request.', 'cmake_skipped_non_windows': 'Automatic CMake MSI installation is only implemented for Windows. Current platform: {platform}.', 'cmake_skipped_arch': 'Automatic CMake MSI installation is configured for Windows x64 only. Current architecture: {machine}.', 'cmake_local_msi': 'Using local CMake MSI: {path}', 'cmake_downloading': 'Downloading CMake MSI from the official release source...', 'cmake_downloaded': 'CMake MSI downloaded to: {path}', 'cmake_installing': 'Installing CMake silently via msiexec...', 'cmake_install_mode_admin': 'Administrator rights detected. CMake will be added to the system PATH for all users.', 'cmake_install_mode_user': 'Administrator rights were not detected. CMake will be installed for the current user and added to the user PATH.', 'cmake_path_added': 'Added CMake to the current process PATH: {path}', 'cmake_ready': 'CMake is now available: {version}', 'cmake_not_found_postinstall': 'CMake installation finished, but cmake.exe was not found in PATH immediately afterwards.', 'cmake_download_failed': 'Failed to download the CMake MSI: {error}', 'cmake_missing_file': 'The specified CMake MSI path does not exist: {path}', 'installing_triton_windows': 'Installing triton-windows for Windows GPU support...', 'skip_triton_windows': 'Skipping triton-windows automatic install on platform={platform}, machine={machine}, gpu_mode={gpu_mode}.', 'triton_already_present': 'Triton is already available. triton-windows installation is not needed.', 'installing_tensorrt': 'TensorRT is not available. Installing NVIDIA TensorRT package...', 'tensorrt_already_present': 'TensorRT is already available. Automatic installation is not needed.', 'tensorrt_fallback': 'Primary TensorRT package was not usable. Trying fallback package: {package}'}


def msg(key: str, **kwargs) -> str:
    text = MESSAGES[key]
    if kwargs:
        return text.format(**kwargs)
    return text


def print_header() -> None:
    print("=" * 78)
    print(msg("title"))
    print(msg("subtitle"))
    print("=" * 78)


@dataclass
class CommandResult:
    ok: bool
    code: int
    command: list[str]


class InstallerError(RuntimeError):
    pass


class Installer:
    def __init__(self, dry_run: bool = False, force_gpu: bool = False, force_cpu: bool = False, skip_optional: bool = False) -> None:
        self.dry_run = dry_run
        self.force_gpu = force_gpu
        self.force_cpu = force_cpu
        self.skip_optional = skip_optional
        self.python = sys.executable
        self.failures: list[str] = []
        self.optional_failures: list[str] = []

    def run(self, cmd: list[str], *, required: bool = True) -> CommandResult:
        rendered = " ".join(cmd)
        print(msg("run", command=rendered))
        if self.dry_run:
            return CommandResult(True, 0, cmd)
        completed = subprocess.run(cmd)
        ok = completed.returncode == 0
        if not ok:
            text = msg("cmd_failed", code=completed.returncode, command=rendered)
            print(text)
            if required:
                self.failures.append(text)
            else:
                self.optional_failures.append(text)
        return CommandResult(ok, completed.returncode, cmd)

    def pip(self, args: list[str], *, required: bool = True) -> CommandResult:
        return self.run([self.python, "-m", "pip", *args], required=required)

    def install(self, packages: Iterable[str], *, index_url: str | None = None, required: bool = True) -> CommandResult:
        cmd = ["install", "--upgrade", *packages]
        if index_url:
            cmd += ["--index-url", index_url]
        return self.pip(cmd, required=required)

    def uninstall(self, packages: Iterable[str]) -> CommandResult:
        return self.pip(["uninstall", "-y", *packages], required=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=msg("argparse_desc"))
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--force-gpu", action="store_true", help=msg("help_force_gpu"))
    grp.add_argument("--force-cpu", action="store_true", help=msg("help_force_cpu"))
    parser.add_argument("--skip-optional", action="store_true", help=msg("help_skip_optional"))
    parser.add_argument("--skip-cmake", action="store_true", help=msg("help_skip_cmake"))
    parser.add_argument("--cmake-msi-path", type=str, default="", help=msg("help_cmake_msi_path"))
    parser.add_argument("--dry-run", action="store_true", help=msg("help_dry_run"))
    return parser.parse_args()


def python_ok() -> bool:
    return sys.version_info >= (3, 10)


CMAKE_REQUIRED_VERSION = (4, 3, 0)
CMAKE_MSI_NAME = "cmake-4.3.0-windows-x86_64.msi"
CMAKE_MSI_URL = "https://github.com/Kitware/CMake/releases/download/v4.3.0/cmake-4.3.0-windows-x86_64.msi"


def parse_version_tuple(text: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", text)
    return tuple(int(x) for x in nums[:4]) if nums else ()


def get_cmake_version() -> str | None:
    cmake = shutil.which("cmake")
    if not cmake:
        return None
    try:
        out = subprocess.run([cmake, "--version"], capture_output=True, text=True, check=False)
        m = re.search(r"cmake version\s+([0-9]+(?:\.[0-9]+){1,3})", out.stdout + "\n" + out.stderr, re.I)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def is_cmake_sufficient(version: str | None) -> bool:
    if not version:
        return False
    parts = list(parse_version_tuple(version))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) >= CMAKE_REQUIRED_VERSION


def is_windows_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def add_cmake_to_current_path() -> None:
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "CMake" / "bin",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "CMake" / "bin",
        Path.home() / "AppData" / "Local" / "Programs" / "CMake" / "bin",
        Path.home() / "AppData" / "Local" / "CMake" / "bin",
    ]
    existing = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        cand = str(candidate)
        if candidate.exists() and cand not in existing:
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            print(msg("cmake_path_added", path=cand))
            return


def resolve_cmake_msi_path(inst: Installer, explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.exists():
            raise InstallerError(msg("cmake_missing_file", path=str(p)))
        print(msg("cmake_local_msi", path=str(p)))
        return p

    script_dir = Path(__file__).resolve().parent
    local_candidates = [
        script_dir / CMAKE_MSI_NAME,
        Path.cwd() / CMAKE_MSI_NAME,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            print(msg("cmake_local_msi", path=str(candidate)))
            return candidate

    download_path = Path(tempfile.gettempdir()) / CMAKE_MSI_NAME
    if download_path.exists():
        print(msg("cmake_local_msi", path=str(download_path)))
        return download_path

    print(msg("cmake_downloading"))
    if inst.dry_run:
        print(msg("cmake_downloaded", path=str(download_path)))
        return download_path
    try:
        urllib.request.urlretrieve(CMAKE_MSI_URL, download_path)
    except Exception as exc:
        raise InstallerError(msg("cmake_download_failed", error=str(exc))) from exc
    print(msg("cmake_downloaded", path=str(download_path)))
    return download_path


def ensure_cmake(inst: Installer, explicit_path: str = "", skip_cmake: bool = False) -> None:
    if skip_cmake:
        print(msg("cmake_skip_requested"))
        return

    print(msg("checking_cmake"))
    existing_version = get_cmake_version()
    if is_cmake_sufficient(existing_version):
        print(msg("cmake_found", version=existing_version))
        return
    if existing_version:
        print(msg("cmake_too_old", version=existing_version))
    else:
        print(msg("cmake_missing"))

    if sys.platform != "win32":
        print(msg("cmake_skipped_non_windows", platform=sys.platform))
        return

    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        print(msg("cmake_skipped_arch", machine=platform.machine()))
        return

    try:
        msi_path = resolve_cmake_msi_path(inst, explicit_path)
    except InstallerError as exc:
        print(str(exc))
        inst.failures.append(str(exc))
        return

    print(msg("cmake_installing"))
    cmd = ["msiexec", "/i", str(msi_path)]
    if is_windows_admin():
        print(msg("cmake_install_mode_admin"))
        cmd += ["ALLUSERS=1", "ADD_CMAKE_TO_PATH=System"]
    else:
        print(msg("cmake_install_mode_user"))
        cmd += ["ADD_CMAKE_TO_PATH=User"]
    cmd += ["/qn", "/norestart"]
    inst.run(cmd, required=True)

    add_cmake_to_current_path()
    version_after = get_cmake_version()
    if version_after:
        print(msg("cmake_ready", version=version_after))
    else:
        print(msg("cmake_not_found_postinstall"))


def detect_nvidia_gpu() -> tuple[bool, str, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, "", None
    try:
        q = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        names = [line.strip() for line in q.stdout.splitlines() if line.strip()]
        has_gpu = bool(names)
    except Exception:
        names = []
        has_gpu = False
    cuda_version = None
    try:
        out = subprocess.run([smi], capture_output=True, text=True, check=False)
        m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", out.stdout + "\n" + out.stderr)
        if m:
            cuda_version = m.group(1)
    except Exception:
        cuda_version = None
    return has_gpu, ", ".join(names), cuda_version


def module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def ensure_triton_windows(inst: Installer, gpu_mode: bool) -> None:
    machine = platform.machine().lower()
    if not (gpu_mode and sys.platform == "win32" and machine in ("amd64", "x86_64")):
        print(msg("skip_triton_windows", platform=sys.platform, machine=platform.machine(), gpu_mode=gpu_mode))
        return

    if module_available("triton"):
        print(msg("triton_already_present"))
        return

    print(msg("installing_triton_windows"))
    inst.install(["triton-windows"], required=False)


def ensure_tensorrt(inst: Installer, gpu_mode: bool) -> None:
    machine = platform.machine().lower()
    if not (gpu_mode and sys.platform in ("win32", "linux") and machine in ("amd64", "x86_64")):
        print(msg("skip_tensorrt", platform=sys.platform, machine=platform.machine()))
        return

    if module_available("tensorrt"):
        print(msg("tensorrt_already_present"))
        return

    print(msg("installing_tensorrt"))
    inst.install(["tensorrt-cu12"], required=False)
    if inst.dry_run or module_available("tensorrt"):
        return

    print(msg("tensorrt_fallback", package="tensorrt"))
    inst.install(["tensorrt"], required=False)


def choose_torch_variant(has_nvidia: bool, cuda_version: str | None, force_gpu: bool, force_cpu: bool) -> tuple[str, str | None]:
    if force_cpu:
        return "cpu", None
    if not has_nvidia and not force_gpu:
        return "cpu", None

    if force_gpu and not has_nvidia:
        print(msg("force_gpu_no_detect"))

    if sys.platform not in ("win32", "linux"):
        print(msg("gpu_platform_not_supported", platform=sys.platform))
        return "cpu", None

    if cuda_version:
        parts = [int(p) for p in cuda_version.split(".")[:2]]
        while len(parts) < 2:
            parts.append(0)
        vv = tuple(parts[:2])
    else:
        vv = (12, 6)

    if vv >= (12, 8):
        return "cu128", "https://download.pytorch.org/whl/cu128"
    if vv >= (12, 6):
        return "cu126", "https://download.pytorch.org/whl/cu126"
    if vv >= (11, 8):
        return "cu118", "https://download.pytorch.org/whl/cu118"

    print(msg("cuda_too_old", cuda=cuda_version or "unknown"))
    return "cpu", None


def install_pytorch(inst: Installer, variant: str, index_url: str | None) -> None:
    print(msg("installing_torch", variant=variant))
    inst.uninstall([
        "torch", "torchvision", "torchaudio",
        "onnxruntime", "onnxruntime-gpu",
    ])

    if variant == "cpu":
        if sys.platform == "darwin":
            inst.install(["torch", "torchvision"], required=True)
        else:
            inst.install(["torch", "torchvision"], index_url="https://download.pytorch.org/whl/cpu", required=True)
    else:
        inst.install(["torch", "torchvision"], index_url=index_url, required=True)


def install_runtime_packages(inst: Installer, gpu_mode: bool) -> None:
    if gpu_mode:
        print(msg("installing_ort_gpu"))
        inst.install(["onnxruntime-gpu[cuda,cudnn]"], required=True)
    else:
        print(msg("installing_ort_cpu"))
        inst.install(["onnxruntime"], required=True)

    base_packages = [
        "numpy",
        "opencv-python",
        "PySide6",
        "ultralytics",
        "reportlab",
        "pillow",
        "psutil",
        "openai",
        "pandas",
        "matplotlib",
        "onnx",
    ]
    print(msg("installing_base"))
    inst.install(base_packages, required=True)

    if inst.skip_optional:
        print(msg("skip_optional"))
        return

    print(msg("installing_optional"))
    optional_groups: list[tuple[str, list[str], bool]] = [
        ("onnxsim", ["onnxsim"], False),
        ("tiktoken", ["tiktoken"], False),
        ("ncnn", ["ncnn"], False),
    ]

    for module_name, pkgs, required in optional_groups:
        if not module_available(module_name):
            inst.install(pkgs, required=required)

    ensure_triton_windows(inst, gpu_mode)
    ensure_tensorrt(inst, gpu_mode)


def verify_imports(gpu_mode: bool) -> tuple[list[str], list[str]]:
    required = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
        "onnxruntime": "onnxruntime / onnxruntime-gpu",
        "PySide6": "PySide6",
        "reportlab": "reportlab",
        "PIL": "pillow",
        "psutil": "psutil",
        "openai": "openai",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "onnx": "onnx",
    }
    optional = {
        "onnxsim": "onnxsim",
        "tiktoken": "tiktoken",
        "ncnn": "ncnn",
    }
    if gpu_mode:
        optional["tensorrt"] = "tensorrt-cu12 / tensorrt"
        if sys.platform == "win32":
            optional["triton"] = "triton-windows / triton"

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for mod, label in required.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing_required.append(label)

    for mod, label in optional.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing_optional.append(label)

    return missing_required, missing_optional


def show_hardware_summary(has_nvidia: bool, gpu_names: str, cuda_version: str | None, variant: str) -> None:
    print(msg("py_info", python=platform.python_version(), exe=sys.executable))
    print(msg("os_info", os_name=platform.system(), release=platform.release(), machine=platform.machine()))
    if has_nvidia:
        print(msg("gpu_found", gpus=gpu_names or msg("unknown_gpu"), cuda=cuda_version or msg("unknown")))
    else:
        print(msg("gpu_not_found"))
    print(msg("chosen_variant", variant=variant))


def main() -> int:
    print_header()
    args = parse_args()

    if not python_ok():
        print(msg("python_too_old", version=platform.python_version()))
        return 2

    inst = Installer(
        dry_run=bool(args.dry_run),
        force_gpu=bool(args.force_gpu),
        force_cpu=bool(args.force_cpu),
        skip_optional=bool(args.skip_optional),
    )

    ensure_cmake(inst, explicit_path=str(args.cmake_msi_path or ""), skip_cmake=bool(args.skip_cmake))

    print(msg("upgrade_tools"))
    inst.pip(["install", "--upgrade", "pip", "setuptools", "wheel"], required=True)

    has_nvidia, gpu_names, cuda_version = detect_nvidia_gpu()
    variant, torch_index = choose_torch_variant(has_nvidia, cuda_version, args.force_gpu, args.force_cpu)
    gpu_mode = variant != "cpu"
    show_hardware_summary(has_nvidia, gpu_names, cuda_version, variant)

    install_pytorch(inst, variant, torch_index)
    install_runtime_packages(inst, gpu_mode)

    if inst.dry_run:
        print(msg("done_dry_run"))
        return 0

    missing_required, missing_optional = verify_imports(gpu_mode)

    print("-" * 78)
    if inst.failures:
        print(msg("required_failures"))
        for item in inst.failures:
            print(f"  - {item}")
    if inst.optional_failures:
        print(msg("optional_failures"))
        for item in inst.optional_failures:
            print(f"  - {item}")
    if missing_required:
        print(msg("missing_required_imports", items=", ".join(missing_required)))
    if missing_optional:
        print(msg("missing_optional_imports", items=", ".join(missing_optional)))

    try:
        import torch
        print(msg("torch_version", version=getattr(torch, "__version__", "unknown")))
        print(msg("torch_cuda", value=str(torch.cuda.is_available())))
        if torch.cuda.is_available():
            try:
                print(msg("torch_gpu_name", name=torch.cuda.get_device_name(0)))
            except Exception:
                pass
    except Exception:
        pass

    if inst.failures or missing_required:
        print(msg("done_with_errors"))
        return 1

    if inst.optional_failures or missing_optional:
        print(msg("done_partial"))
        return 0

    print(msg("done_success"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
