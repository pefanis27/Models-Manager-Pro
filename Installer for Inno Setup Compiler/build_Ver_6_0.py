# -*- coding: utf-8 -*-
"""
Models Manager Pro Ver 6.0 - Interactive Builder
-------------------------------------------------
-----------------------------------------------------------
Τι βελτιώθηκε σε σχέση με το αρχικό:
✅ Σταθερότερη εκτέλεση PyInstaller: τρέχει μέσω `python -m PyInstaller` (σωστό venv)
✅ Πραγματικός "pre-clean" καθαρισμός (build/dist/spec) όταν το επιλέξεις
✅ Αυτόματη ανίχνευση UPX (PATH ή γνωστοί φάκελοι) — χωρίς σπασίματα
✅ Αυτόματο --add-data για app_icon.png + assets/ (αν υπάρχουν)
✅ Extra συλλογές μόνο αν είναι εγκατεστημένες (onnxruntime/tensorrt/matplotlib/reportlab)
✅ Hooks σε απομονωμένο φάκελο (.mmpro_pyinstaller/) και ασφαλής cleanup
✅ Καλύτερα μηνύματα σφάλματος + πιο "σφιχτό" command

Changelog (v6.0):
  - SYNC: Ευθυγράμμιση με Models_Manager_Pro_Ver_4_0.py
  - FIX:  Προστέθηκαν hidden imports για torchvision (CNN μοντέλα: MobileNet/ResNet)
  - FIX:  Προστέθηκαν hidden imports για openai (A.I Copilot / Groq API)
  - FIX:  Προστέθηκε --collect-all=torchvision (απαραίτητο για CNNInferenceHelper)
  - FIX:  Προστέθηκαν matplotlib backends: backend_qtagg (PySide6 charts)
          και backend_pdf (PDF Report generation) — έλειπαν και έσπαγαν τα charts/reports
  - FIX:  Δυναμική ανίχνευση TensorRT φακέλου (αντί για hardcoded v10.13.3.9)
  - FIX:  Προστέθηκε --icon= argument (exe icon για Windows taskbar/explorer)
  - FIX:  Προστέθηκε onnxruntime_logging στα hidden imports
  - FIX:  Έλεγχος torchvision / openai ως optional dep hints
  - NEW:  Δυναμική ανίχνευση .ico αρχείου (προτεραιότητα έναντι .png)
  - NEW:  Βήμα επιλογής Console/Windowed με προεπιλογή Windowed για release
  - CLEAN: Ενημερώθηκε header/version, αφαιρέθηκε hardcoded TensorRT path
  - RETAINED: Όλες οι βελτιώσεις v3.2 (FIX warning flags, env copy, importlib top-level κ.λπ.)

Σημείωση:
  Το Models_Manager_Pro_Ver_4_0.py κάνει νωρίς DLL/ROOT_DIR/OpenCV/ORT tweaks
  σε frozen mode, οπότε το runtime hook είναι intentionally "μικρό" και ασφαλές.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


# ================== ΡΥΘΜΙΣΕΙΣ ==================
APP_NAME     = "Models_Manager_Pro"
ENTRY_SCRIPT = "Models_Manager_Pro_Ver_6_0.py"
SPEC_FILE    = f"{APP_NAME}.spec"

# Hooks (δημιουργούνται προσωρινά — διαγράφονται μετά το build)
HOOK_BASE_DIRNAME = ".mmpro_pyinstaller"
RUNTIME_HOOK_FILE = "pyi_rth_mmpro.py"
HOOKS_DIRNAME     = "extra_hooks"

# Resources που μπαίνουν στο dist
ICON_PNG  = "app_icon.png"   # fallback αν δεν υπάρχει .ico
ICON_ICO  = "app_icon.ico"   # προτεραιότητα για Windows taskbar/exe
ASSETS_DIR = "assets"

# ================== ANSI COLORS ==================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ================== FLAGS ==================
# Suppress SyntaxWarning "invalid escape sequence" από TensorRT docstrings (Python 3.12+)
SUPPRESS_TENSORRT_SYNTAXWARNINGS = True
# Suppress DeprecationWarning από torch.distributed legacy sharding modules
SUPPRESS_TORCH_DISTRIBUTED_DEPRECATIONWARNINGS = True
# Εμφάνιση hint για optional deps που δεν βρέθηκαν
HINT_MISSING_OPTIONAL_DEPS = True
# Συμπερίληψη torch.compile/dynamo/inductor stack
# True = μεγαλύτερο exe αλλά torch.compile() λειτουργεί (χρησιμοποιείται στο training)
INCLUDE_TORCH_COMPILE_STACK = True


# ================== ANSI WINDOWS ==================
def _enable_ansi_windows_best_effort() -> None:
    """Enable ANSI colors in Windows console (best-effort)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)
            return
    except Exception:
        pass
    try:
        os.system("color")
    except Exception:
        pass


_enable_ansi_windows_best_effort()


# ================== HOOK CONTENTS ==================
RUNTIME_HOOK_CONTENT = r"""
import os
import sys
import multiprocessing

# 1) Windows multiprocessing fix (PyInstaller onefile/onedir)
try:
    multiprocessing.freeze_support()
except Exception:
    pass

# 2) Ultralytics settings/config dir redirect (αποφυγή εγγραφής σε protected paths)
if sys.platform.startswith('win'):
    try:
        base = os.getenv('APPDATA') or os.getenv('LOCALAPPDATA')
        if base:
            ultra_dir = os.path.join(base, 'Ultralytics')
            os.makedirs(ultra_dir, exist_ok=True)
            os.environ.setdefault('ULTRALYTICS_SETTINGS_DIR', ultra_dir)
            os.environ.setdefault('YOLO_CONFIG_DIR', ultra_dir)
    except Exception:
        pass

# 3) DLL path assist (Windows frozen)
if getattr(sys, 'frozen', False) and sys.platform.startswith('win'):
    try:
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            add_dir = getattr(os, 'add_dll_directory', None)
            if callable(add_dir):
                try:
                    add_dir(meipass)
                except Exception:
                    pass
            os.environ['PATH'] = meipass + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

# 4) torchvision frozen path fix (CNN inference)
if getattr(sys, 'frozen', False):
    try:
        import torchvision
    except Exception:
        pass
"""

HOOK_ULTRALYTICS_CONTENT = r"""
from PyInstaller.utils.hooks import collect_all
datas, binaries, hiddenimports = collect_all('ultralytics')
"""

HOOK_ONNX_CONTENT = r"""
# Custom ONNX hook — εξαιρεί onnx.backend.test.* (optional deps / noise)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("onnx")
hiddenimports = [m for m in hiddenimports if not m.startswith("onnx.backend.test")]
excludedimports = ["onnx.backend.test"]
warn_on_missing_hiddenimports = False
"""

HOOK_TORCHVISION_CONTENT = r"""
# torchvision hook για CNN μοντέλα (MobileNet V2/V3, ResNet-50/101)
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = collect_all('torchvision')
hiddenimports += collect_submodules('torchvision.models')
hiddenimports += collect_submodules('torchvision.transforms')
hiddenimports += [
    'torchvision.models.mobilenet',
    'torchvision.models.mobilenetv2',
    'torchvision.models.mobilenetv3',
    'torchvision.models.resnet',
    'torchvision.datasets',
    'torchvision.datasets.folder',
]
"""


# ================== HELPERS ==================
def print_header() -> None:
    print(f"\n{BOLD}{CYAN}======================================================{RESET}")
    print(f"{BOLD}{CYAN}   Models Manager Pro Ver 4.0 — Builder Tool v4.0    {RESET}")
    print(f"{BOLD}{CYAN}======================================================{RESET}\n")


def ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    """Επιστρέφει 1..len(options). Default όταν πατηθεί Enter."""
    default = max(1, min(len(options), int(default)))
    print(f"{YELLOW}{prompt}{RESET}")
    for i, opt in enumerate(options, 1):
        d = " (default)" if i == default else ""
        print(f"  {BOLD}{i}.{RESET} {opt}{CYAN}{d}{RESET}")
    while True:
        raw = input(f"\n{BOLD}Επίλεξε (1-{len(options)}) ή Enter για {default}: {RESET}").strip()
        if raw == "":
            return default
        if raw.isdigit():
            v = int(raw)
            if 1 <= v <= len(options):
                return v
        print(f"{RED}Μη έγκυρη επιλογή.{RESET}")


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _safe_rmtree(p: Path) -> None:
    try:
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _pyi_data_sep() -> str:
    return ";" if os.name == "nt" else ":"


def _add_data_args(cmd: list[str], src: Path, dest_rel: str) -> None:
    """Append --add-data αν το src υπάρχει."""
    try:
        if not src.exists():
            return
    except Exception:
        return
    sep = _pyi_data_sep()
    cmd.append(f"--add-data={str(src)}{sep}{dest_rel}")


def _importable(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except Exception:
        return False


def _find_upx_dir() -> Optional[Path]:
    """Βρίσκει τον φάκελο UPX (PATH ή γνωστές τοποθεσίες)."""
    try:
        exe = shutil.which("upx") or shutil.which("upx.exe")
        if exe:
            return Path(exe).resolve().parent
    except Exception:
        pass
    if os.name == "nt":
        candidates: list[Path] = [
            Path("C:/upx"),
            Path("C:/Tools/upx"),
            Path(os.getenv("ProgramFiles", "C:/Program Files")) / "upx",
            Path(os.getenv("LOCALAPPDATA", "")) / "upx",
        ]
        for d in candidates:
            try:
                if (d / "upx.exe").exists():
                    return d.resolve()
            except Exception:
                pass
    return None


def _find_tensorrt_dir(project_root: Path) -> Optional[Path]:
    """
    Δυναμική ανίχνευση TensorRT φακέλου στο project root.
    Αναζητά φακέλους με pattern 'TensorRT-*' αντί για hardcoded version.
    """
    try:
        candidates = sorted(project_root.glob("TensorRT-*"), reverse=True)
        for d in candidates:
            if d.is_dir():
                return d
    except Exception:
        pass
    return None


def _find_icon(project_root: Path) -> Optional[Path]:
    """
    Βρίσκει το icon για το exe.
    Προτεραιότητα: .ico (Windows taskbar) > .png (fallback, δεν δουλεύει σωστά στο taskbar)
    """
    ico = project_root / ICON_ICO
    if ico.exists():
        return ico
    png = project_root / ICON_PNG
    if png.exists():
        return png
    return None


def _check_pyinstaller() -> bool:
    return _importable("PyInstaller")


def create_hooks(hook_base: Path) -> tuple[Path, Path]:
    """Δημιουργεί runtime hook + extra hooks dir. Επιστρέφει (runtime_hook_path, hooks_dir_path)."""
    hook_base.mkdir(parents=True, exist_ok=True)

    runtime_hook_path = hook_base / RUNTIME_HOOK_FILE
    hooks_dir_path    = hook_base / HOOKS_DIRNAME
    hooks_dir_path.mkdir(parents=True, exist_ok=True)

    runtime_hook_path.write_text(RUNTIME_HOOK_CONTENT, encoding="utf-8")
    (hooks_dir_path / "hook-ultralytics.py").write_text(HOOK_ULTRALYTICS_CONTENT, encoding="utf-8")
    (hooks_dir_path / "hook-onnx.py").write_text(HOOK_ONNX_CONTENT, encoding="utf-8")
    (hooks_dir_path / "hook-torchvision.py").write_text(HOOK_TORCHVISION_CONTENT, encoding="utf-8")

    return runtime_hook_path, hooks_dir_path


def pre_clean(project_root: Path) -> None:
    """Αφαιρεί build/dist/spec από προηγούμενα runs."""
    _safe_rmtree(project_root / "build")
    _safe_rmtree(project_root / "dist")
    _safe_unlink(project_root / SPEC_FILE)


def _join_cmd_for_print(cmd: list[str]) -> str:
    """Εκτυπώσιμη μορφή command για copy/paste."""
    try:
        import shlex
        return shlex.join(cmd)
    except Exception:
        return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])


# ================== MAIN ==================
def main() -> None:
    print_header()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    entry = project_root / ENTRY_SCRIPT
    if not entry.exists():
        print(f"{RED}❌ Δεν βρέθηκε το {ENTRY_SCRIPT} στο: {project_root}{RESET}")
        input(f"\n{BOLD}Πάτα Enter για έξοδο...{RESET}")
        return

    # Έλεγχος PyInstaller πριν τα ερωτήματα
    if not _check_pyinstaller():
        print(f"{RED}❌ Το PyInstaller δεν βρέθηκε.{RESET}")
        print(f"{YELLOW}💡 Εγκατάστησέ το: pip install pyinstaller{RESET}")
        input(f"\n{BOLD}Πάτα Enter για έξοδο...{RESET}")
        return

    # Hint για optional deps
    if HINT_MISSING_OPTIONAL_DEPS:
        optional_pkgs = [
            "onnxruntime", "onnx", "tensorrt",
            "reportlab", "matplotlib", "psutil",
            "torchvision", "openai",          # NEW: torchvision (CNN) + openai (Copilot)
        ]
        missing = [p for p in optional_pkgs if not _importable(p)]
        if missing:
            print(f"{YELLOW}ℹ️  Προαιρετικά πακέτα που δεν βρέθηκαν (θα παραληφθούν): "
                  f"{', '.join(missing)}{RESET}")
            if "torchvision" in missing:
                print(f"{YELLOW}   ⚠️  torchvision λείπει — τα CNN μοντέλα (MobileNet/ResNet) "
                      f"ΔΕΝ θα λειτουργούν στο frozen exe!{RESET}")
            if "openai" in missing:
                print(f"{YELLOW}   ⚠️  openai λείπει — το A.I Copilot (Groq) "
                      f"ΔΕΝ θα λειτουργεί στο frozen exe!{RESET}")
            print()

    # Icon ανίχνευση
    icon_path = _find_icon(project_root)
    if icon_path:
        print(f"{GREEN}🎨 Icon: {icon_path.name}{RESET}")
        if icon_path.suffix.lower() == '.png':
            print(f"{YELLOW}   💡 Βρέθηκε μόνο .png — για σωστό Windows taskbar icon "
                  f"χρησιμοποίησε .ico (app_icon.ico){RESET}")
    else:
        print(f"{YELLOW}⚠️  Δεν βρέθηκε icon (app_icon.ico / app_icon.png) — "
              f"το exe θα έχει default icon{RESET}")

    # TensorRT folder
    trt_pack = _find_tensorrt_dir(project_root)
    if trt_pack:
        print(f"{GREEN}🔥 TensorRT: {trt_pack.name}{RESET}\n")

    # ─── ΒΗΜΑΤΑ ΕΠΙΛΟΓΗΣ ───────────────────────────────────────────────────────

    # ΒΗΜΑ 1: Build type
    build_type = ask_choice(
        "Πώς θέλεις να δημιουργηθεί το αρχείο;", [
            "OneFile (Ένα ενιαίο .exe — Πιο φορητό, αργότερη εκκίνηση)",
            "OneDir  (Φάκελος με αρχεία — Πιο γρήγορη εκκίνηση)",
        ], default=1)

    # ΒΗΜΑ 2: Console/Windowed
    console_type = ask_choice(
        "Θέλεις να εμφανίζεται παράθυρο κονσόλας;", [
            "Όχι — Windowed/GUI μόνο (Για τελικούς χρήστες)",
            "Ναι — Console (Για debugging / εμφάνιση σφαλμάτων)",
        ], default=1)

    # ΒΗΜΑ 3: UPX
    upx_choice = ask_choice(
        "Θέλεις συμπίεση με UPX (μικρότερο μέγεθος);", [
            "Ναι (αν υπάρχει UPX στο PATH)",
            "Όχι (ασφαλέστερο, λιγότερα antivirus false positives)",
        ], default=2)

    # ΒΗΜΑ 4: Pre-clean
    clean_build = ask_choice(
        "Καθαρισμός build/dist/spec πριν το build;", [
            "Ναι — Καθαρό build από μηδέν (Προτείνεται)",
            "Όχι — Κράτα cache για ταχύτητα",
        ], default=1)

    # ─── ΠΡΟΕΤΟΙΜΑΣΙΑ ──────────────────────────────────────────────────────────

    if clean_build == 1:
        print(f"\n{YELLOW}🧹 Pre-clean: build / dist / spec...{RESET}")
        pre_clean(project_root)

    hook_base = project_root / HOOK_BASE_DIRNAME
    runtime_hook_path, hooks_dir_path = create_hooks(hook_base)

    # ─── COMMAND ───────────────────────────────────────────────────────────────

    cmd: list[str] = [sys.executable]

    # Warning suppression flags (ανεξάρτητα, ένα -W ανά φίλτρο)
    if SUPPRESS_TENSORRT_SYNTAXWARNINGS:
        cmd += ["-W", r"ignore:invalid escape sequence:SyntaxWarning:tensorrt\..*"]

    if SUPPRESS_TORCH_DISTRIBUTED_DEPRECATIONWARNINGS:
        cmd += [
            "-W", r"ignore:.*torch\.distributed\._sharding_spec.*:DeprecationWarning",
            "-W", r"ignore:.*torch\.distributed\._sharded_tensor.*:DeprecationWarning",
            "-W", r"ignore:.*torch\.distributed\._shard\.checkpoint.*:DeprecationWarning",
        ]

    cmd += ["-m", "PyInstaller"]

    # ── Core ───────────────────────────────────────────────────────────────────
    cmd += [
        f"--name={APP_NAME}",
        "--noconfirm",
        f"--runtime-hook={str(runtime_hook_path)}",
        f"--additional-hooks-dir={str(hooks_dir_path)}",
    ]

    # Build mode
    cmd.append("--onefile" if build_type == 1 else "--onedir")

    # Console/windowed
    cmd.append("--windowed" if console_type == 1 else "--console")

    # PyInstaller-level clean
    if clean_build == 1:
        cmd.append("--clean")

    # Icon
    if icon_path:
        cmd.append(f"--icon={str(icon_path)}")

    # UPX
    if upx_choice == 2:
        cmd.append("--noupx")
    else:
        upx_dir = _find_upx_dir()
        if upx_dir:
            cmd.append(f"--upx-dir={str(upx_dir)}")
        else:
            print(f"{YELLOW}⚠️ Δεν βρέθηκε UPX. Συνεχίζω χωρίς.{RESET}")
            cmd.append("--noupx")

    # ── Data files ─────────────────────────────────────────────────────────────
    _add_data_args(cmd, project_root / ICON_PNG, ".")
    if icon_path and icon_path.suffix.lower() == ".ico":
        _add_data_args(cmd, icon_path, ".")
    _add_data_args(cmd, project_root / ASSETS_DIR, ASSETS_DIR)

    # TensorRT DLL pack (δυναμική ανίχνευση)
    if trt_pack:
        _add_data_args(cmd, trt_pack, trt_pack.name)

    # ── Collections ────────────────────────────────────────────────────────────

    # torch: collect-all (εξασφαλίζει όλα τα submodules)
    cmd.append("--collect-all=torch")

    # torchvision: απαραίτητο για CNN μοντέλα (MobileNet V2/V3, ResNet-50/101)
    if _importable("torchvision"):
        cmd.append("--collect-all=torchvision")
    else:
        print(f"{YELLOW}⚠️ torchvision δεν βρέθηκε — παραλείπεται{RESET}")

    # onnxruntime
    if _importable("onnxruntime"):
        cmd.append("--collect-all=onnxruntime")

    # ONNX: custom hook (χωρίς backend.test noise) + hidden imports
    if _importable("onnx"):
        cmd.extend([
            "--hidden-import=onnx",
            "--hidden-import=onnx.onnx_cpp2py_export",
        ])

    # TensorRT
    if _importable("tensorrt"):
        cmd.append("--collect-all=tensorrt")

    # reportlab (PDF reports)
    if _importable("reportlab"):
        cmd.append("--collect-all=reportlab")

    # matplotlib — hidden imports αντί collect-all (αποφυγή baseline noise)
    # ΣΗΜΑΝΤΙΚΟ: backend_qtagg (PySide6 charts) + backend_pdf (PDF reports)
    if _importable("matplotlib"):
        cmd.extend([
            "--hidden-import=matplotlib",
            "--hidden-import=matplotlib.pyplot",
            "--hidden-import=matplotlib.figure",
            "--hidden-import=matplotlib.backends.backend_agg",
            "--hidden-import=matplotlib.backends.backend_qtagg",   # FIX: PySide6 charts
            "--hidden-import=matplotlib.backends.backend_pdf",     # FIX: PDF report generation
        ])

    # psutil (live system resources)
    if _importable("psutil"):
        cmd.append("--collect-all=psutil")

    # openai (A.I Copilot / Groq API)
    if _importable("openai"):
        cmd.extend([
            "--hidden-import=openai",
            "--collect-all=openai",
        ])
    else:
        print(f"{YELLOW}⚠️ openai δεν βρέθηκε — A.I Copilot δεν θα λειτουργεί{RESET}")

    # ── Hidden imports ─────────────────────────────────────────────────────────
    cmd.extend([
        # PySide6 / Qt
        "--hidden-import=PySide6",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=shiboken6",
        # Computer vision
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        # Data
        "--hidden-import=pandas",
        # ONNX Runtime logging
        "--hidden-import=onnxruntime",
        "--hidden-import=onnxruntime.capi._pybind_state",  # FIX: onnxruntime_logging
        # torchvision CNN models
        "--hidden-import=torchvision",
        "--hidden-import=torchvision.models",
        "--hidden-import=torchvision.transforms",
        "--hidden-import=torchvision.datasets",
        "--hidden-import=torchvision.datasets.folder",
    ])

    # ── Excludes ───────────────────────────────────────────────────────────────
    cmd.extend([
        # ONNX test suite
        "--exclude-module=onnx.backend.test",
        "--exclude-module=onnx.backend.test.report",
        "--exclude-module=onnx.backend.test.runner",
        # matplotlib test suite
        "--exclude-module=matplotlib.tests",
        # Qt bindings (χρησιμοποιούμε μόνο PySide6)
        "--exclude-module=PyQt6",
        "--exclude-module=PyQt6-Qt6",
        "--exclude-module=PyQt6-sip",
        "--exclude-module=PyQt5",
        # Άλλα περιττά
        "--exclude-module=tkinter",
        "--exclude-module=IPython",
    ])

    # torch compile stack
    if not INCLUDE_TORCH_COMPILE_STACK:
        cmd.extend([
            "--exclude-module=torch._dynamo",
            "--exclude-module=torch._inductor",
            "--exclude-module=torch._inductor.kernel",
            "--exclude-module=torch._inductor.codegen",
            "--exclude-module=torch._inductor.codegen.cpp_gemm_template",
        ])

    # ── Entry point ────────────────────────────────────────────────────────────
    cmd.append(str(entry))

    # ─── ΕΚΤΕΛΕΣΗ ──────────────────────────────────────────────────────────────
    print(f"\n{GREEN}🚀 Ξεκινάει το Build...{RESET}")
    print(f"{BOLD}Command:{RESET} {_join_cmd_for_print(cmd)}\n")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        mins, secs = divmod(int(duration), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        print(f"\n{GREEN}{'='*54}{RESET}")
        print(f"{GREEN}✅ BUILD ΕΠΙΤΥΧΗΣ!{RESET}")
        print(f"   Χρόνος:      {time_str}")
        print(f"   Αποτέλεσμα:  {BOLD}{project_root / 'dist'}{RESET}")
        if build_type == 1:
            exe_path = project_root / "dist" / f"{APP_NAME}.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / 1024 / 1024
                print(f"   Μέγεθος exe: {size_mb:.1f} MB")
        print(f"{GREEN}{'='*54}{RESET}")
    except FileNotFoundError as e:
        print(f"\n{RED}❌ ΑΠΟΤΥΧΙΑ: Δεν βρέθηκε εργαλείο: {e}{RESET}")
        print(f"{YELLOW}💡 Βεβαιώσου ότι έχεις PyInstaller: pip install pyinstaller{RESET}")
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}❌ ΑΠΟΤΥΧΙΑ: PyInstaller exit code={e.returncode}{RESET}")
        print(f"{YELLOW}💡 Δες τα logs παραπάνω για το πρώτο ERROR.{RESET}")
    finally:
        _safe_rmtree(hook_base)
        input(f"\n{BOLD}Πάτα Enter για έξοδο...{RESET}")


if __name__ == "__main__":
    main()
