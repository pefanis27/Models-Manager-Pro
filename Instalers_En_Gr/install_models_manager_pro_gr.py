#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Εγκαταστάτης Models Manager Pro (Ελληνικά)

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

MESSAGES = {'title': 'Models Manager Pro - Εγκατάσταση Python Πακέτων', 'subtitle': 'Ελληνικό script με αυτόματη επιλογή CPU / NVIDIA GPU πακέτων', 'argparse_desc': 'Εγκαθιστά τις Python εξαρτήσεις που απαιτεί το Models Manager Pro.', 'help_force_gpu': 'Εξαναγκάζει εγκατάσταση πακέτων GPU ακόμη κι αν αποτύχει η αυτόματη ανίχνευση.', 'help_force_cpu': 'Εξαναγκάζει εγκατάσταση μόνο CPU πακέτων.', 'help_skip_optional': 'Παραλείπει προαιρετικά πακέτα δυνατοτήτων όπως TensorRT, Triton, NCNN, ONNXSIM και tiktoken.', 'help_dry_run': 'Εμφανίζει μόνο τις εντολές χωρίς να τις εκτελέσει.', 'run': '> {command}', 'cmd_failed': 'Η εντολή απέτυχε με κωδικό εξόδου {code}: {command}', 'force_gpu_no_detect': 'Έγινε εξαναγκασμός λειτουργίας GPU παρότι δεν εντοπίστηκε αυτόματα NVIDIA GPU.', 'gpu_platform_not_supported': "Η αυτόματη εγκατάσταση GPU δεν υποστηρίζεται για την πλατφόρμα '{platform}', γίνεται επιστροφή σε CPU πακέτα.", 'cuda_too_old': "Η ανιχνευμένη δυνατότητα CUDA '{cuda}' είναι πολύ παλιά για τα τρέχοντα prebuilt PyTorch CUDA wheels· γίνεται επιστροφή σε CPU πακέτα.", 'installing_torch': 'Γίνεται εγκατάσταση PyTorch και torchvision ({variant})...', 'installing_ort_gpu': 'Γίνεται εγκατάσταση του ONNX Runtime GPU πακέτου...', 'installing_ort_cpu': 'Γίνεται εγκατάσταση του ONNX Runtime CPU πακέτου...', 'installing_base': 'Γίνεται εγκατάσταση των βασικών πακέτων της εφαρμογής...', 'skip_optional': 'Τα προαιρετικά πακέτα παραλείφθηκαν μετά από επιλογή.', 'installing_optional': 'Γίνεται εγκατάσταση προαιρετικών / ειδικών πακέτων δυνατοτήτων όταν λείπουν...', 'skip_tensorrt': 'Παραλείπεται η αυτόματη εγκατάσταση TensorRT για platform={platform}, machine={machine}.', 'py_info': 'Python: {python} | Εκτελέσιμο: {exe}', 'os_info': 'Λειτουργικό: {os_name} {release} | Αρχιτεκτονική: {machine}', 'gpu_found': 'Εντοπίστηκε NVIDIA GPU: {gpus} | CUDA από το nvidia-smi: {cuda}', 'unknown_gpu': 'Άγνωστη NVIDIA GPU', 'unknown': 'άγνωστο', 'gpu_not_found': 'Δεν εντοπίστηκε NVIDIA GPU. Θα χρησιμοποιηθούν CPU πακέτα.', 'chosen_variant': 'Επιλεγμένο compute variant: {variant}', 'python_too_old': 'Η Python {version} είναι πολύ παλιά. Το τρέχον stable PyTorch απαιτεί Python 3.10 ή νεότερη.', 'upgrade_tools': 'Γίνεται αναβάθμιση pip / setuptools / wheel...', 'required_failures': 'Σφάλματα εγκατάστασης απαιτούμενων πακέτων:', 'optional_failures': 'Προειδοποιήσεις εγκατάστασης προαιρετικών πακέτων:', 'missing_required_imports': 'Απουσιάζουν απαιτούμενα imports μετά την εγκατάσταση: {items}', 'missing_optional_imports': 'Προαιρετικά imports που παραμένουν μη διαθέσιμα: {items}', 'torch_version': 'Εγκατεστημένη έκδοση torch: {version}', 'torch_cuda': 'torch.cuda.is_available(): {value}', 'torch_gpu_name': 'GPU συσκευή που βλέπει το PyTorch: {name}', 'done_with_errors': 'Η εγκατάσταση ολοκληρώθηκε με σφάλματα. Δες τα παραπάνω μηνύματα.', 'done_partial': 'Η εγκατάσταση ολοκληρώθηκε. Τα βασικά πακέτα εγκαταστάθηκαν, αλλά κάποια προαιρετικά πακέτα δυνατοτήτων δεν είναι διαθέσιμα.', 'done_success': 'Η εγκατάσταση ολοκληρώθηκε επιτυχώς. Όλα τα βασικά πακέτα έγιναν import σωστά.', 'done_dry_run': 'Η προσομοίωση ολοκληρώθηκε. Δεν έγιναν αλλαγές.', 'help_skip_cmake': 'Παραλείπει τον αυτόματο έλεγχο και την εγκατάσταση του CMake.', 'help_cmake_msi_path': 'Χρησιμοποιεί συγκεκριμένη τοπική διαδρομή για το cmake-4.3.0-windows-x86_64.msi.', 'checking_cmake': 'Γίνεται έλεγχος αν υπάρχει ήδη διαθέσιμο το CMake 4.3.0 ή νεότερο...', 'cmake_found': 'Το CMake είναι ήδη διαθέσιμο: {version}', 'cmake_too_old': 'Βρέθηκε εγκατεστημένο το CMake {version}, αλλά απαιτείται η έκδοση 4.3.0 ή νεότερη. Θα γίνει αναβάθμιση.', 'cmake_missing': 'Το CMake δεν βρέθηκε. Το script θα εγκαταστήσει πρώτα το CMake 4.3.0 και μετά τα Python πακέτα.', 'cmake_skip_requested': 'Η εγκατάσταση του CMake παραλείφθηκε μετά από επιλογή.', 'cmake_skipped_non_windows': 'Η αυτόματη εγκατάσταση του CMake μέσω MSI υλοποιείται μόνο για Windows. Τρέχουσα πλατφόρμα: {platform}.', 'cmake_skipped_arch': 'Η αυτόματη εγκατάσταση του CMake μέσω MSI είναι ρυθμισμένη μόνο για Windows x64. Τρέχουσα αρχιτεκτονική: {machine}.', 'cmake_local_msi': 'Θα χρησιμοποιηθεί το τοπικό CMake MSI: {path}', 'cmake_downloading': 'Γίνεται λήψη του CMake MSI από την επίσημη πηγή έκδοσης...', 'cmake_downloaded': 'Το CMake MSI αποθηκεύτηκε στο: {path}', 'cmake_installing': 'Γίνεται αθόρυβη εγκατάσταση του CMake μέσω msiexec...', 'cmake_install_mode_admin': 'Εντοπίστηκαν δικαιώματα διαχειριστή. Το CMake θα προστεθεί στο system PATH για όλους τους χρήστες.', 'cmake_install_mode_user': 'Δεν εντοπίστηκαν δικαιώματα διαχειριστή. Το CMake θα εγκατασταθεί για τον τρέχοντα χρήστη και θα προστεθεί στο user PATH.', 'cmake_path_added': 'Το CMake προστέθηκε στο PATH της τρέχουσας διεργασίας: {path}', 'cmake_ready': 'Το CMake είναι πλέον διαθέσιμο: {version}', 'cmake_not_found_postinstall': 'Η εγκατάσταση του CMake ολοκληρώθηκε, αλλά το cmake.exe δεν βρέθηκε αμέσως μετά στο PATH.', 'cmake_download_failed': 'Απέτυχε η λήψη του CMake MSI: {error}', 'cmake_missing_file': 'Η δηλωμένη διαδρομή για το CMake MSI δεν υπάρχει: {path}', 'installing_triton_windows': 'Γίνεται εγκατάσταση του triton-windows για υποστήριξη Windows GPU...', 'skip_triton_windows': 'Παραλείπεται η αυτόματη εγκατάσταση του triton-windows για platform={platform}, machine={machine}, gpu_mode={gpu_mode}.', 'triton_already_present': 'Το Triton είναι ήδη διαθέσιμο. Δεν χρειάζεται εγκατάσταση του triton-windows.', 'installing_tensorrt': 'Το TensorRT δεν είναι διαθέσιμο. Γίνεται εγκατάσταση του NVIDIA TensorRT πακέτου...', 'tensorrt_already_present': 'Το TensorRT είναι ήδη διαθέσιμο. Δεν χρειάζεται αυτόματη εγκατάσταση.', 'tensorrt_fallback': 'Το βασικό πακέτο TensorRT δεν ήταν αξιοποιήσιμο. Δοκιμάζεται εναλλακτικό πακέτο: {package}'}


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
