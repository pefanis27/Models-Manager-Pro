# -*- coding: utf-8 -*-
"""
(c) Πεφάνης Ευάγγελος
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Models Manager Pro  –  v4.0                            ║
║          A.I Copilot Edition  (YOLO / Ultralytics + CNN Classifiers)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Κεντρική εφαρμογή διαχείρισης μοντέλων YOLO & CNN (εκπαίδευση, εξαγωγή,     ║
║  live detection/classification, video inference, benchmark, AI Copilot).     ║
║                                                                              ║
║  Υποστηριζόμενα μοντέλα CNN (torchvision):                                   ║
║    • MobileNet V2, MobileNet V3 Small, MobileNet V3 Large                    ║
║    • ResNet-50, ResNet-101                                                   ║
║                                                                              ║
║  Δομή αρχείου:                                                               ║
║    1.  Imports & αρχικοποίηση περιβάλλοντος                                  ║
║    2.  Logging & crash-log utilities                                         ║
║    3.  Runner / subprocess helpers                                           ║
║    4.  Path constants                                                        ║
║    5.  Export target helpers (ONNX / TensorRT / NCNN)                        ║
║    6.  PDF report generation (ReportLab)                                     ║
║    7.  Diagnostics utilities                                                 ║
║    8.  TensorRT cache management                                             ║
║    9.  Camera / OpenCV utilities                                             ║
║   10.  PySide6 custom widgets                                                ║
║   11.  Χρώματα, HTML log formatting, constants                               ║
║   12.  Dataset & model configuration                                         ║
║   13.  CNN helpers (CNNInferenceHelper, CNNTrainingWorker)                   ║
║   14.  Κύριο παράθυρο – YOLOProManager                                       ║
║   15.  Entry point – main()                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# ΕΓΚΑΤΑΣΤΑΣΗ ΑΠΑΙΤΟΥΜΕΝΩΝ ΠΑΚΕΤΩΝ – Python 3.12
# Εκτέλεσε τις παρακάτω εντολές στο terminal (PowerShell / CMD / Linux shell)
# =============================================================================

# 1️⃣ Προαιρετικά: δημιουργία virtual environment
# python -m venv venv
# venv\Scripts\activate          (Windows)
# source venv/bin/activate       (Linux / Mac)

# 2️⃣ Αναβάθμιση pip
# python -m pip install --upgrade pip

# 3️⃣ Βασικές αριθμητικές βιβλιοθήκες
# pip install numpy

# 4️⃣ PyTorch + torchvision (CUDA ή CPU)
# CPU version:
# pip install torch torchvision

# Αν έχεις NVIDIA GPU (π.χ. RTX 3060):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5️⃣ YOLO / Ultralytics
# pip install ultralytics

# 6️⃣ OpenCV για camera / video processing
# pip install opencv-python

# 7️⃣ ONNX Runtime (για inference ONNX models)
# pip install onnxruntime

# 8️⃣ GUI framework
# pip install PySide6

# 9️⃣ Δημιουργία PDF reports
# pip install reportlab

# 🔟 Image handling (χρησιμοποιείται στο reportlab helper)
# pip install pillow

# 1️⃣1️⃣ System diagnostics (CPU/RAM monitoring)
# pip install psutil

# =============================================================================
# Προαιρετικά αλλά χρήσιμα για YOLO pipelines
# =============================================================================

# ONNX tools
# pip install onnx

# Export utilities
# pip install onnxsim

# =============================================================================
# Έλεγχος εγκατάστασης
# =============================================================================

# python -c "import torch, ultralytics, cv2, numpy, PySide6, reportlab, onnxruntime; print('All packages OK')"

from __future__ import annotations
import collections
import faulthandler
import gc
import hashlib
import html
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 1 – Αρχικοποίηση συστήματος καταγραφής (logging)
# ═══════════════════════════════════════════════════════════════════════

# ── Αρχικοποίηση κεντρικού logger της εφαρμογής ────────────────────────────
# Δημιουργεί έναν singleton Logger με το όνομα 'MMPro'.
# Χρησιμοποιείται παντού στην εφαρμογή για καταγραφή πληροφοριών, προειδοποιήσεων και σφαλμάτων.
def _setup_mmpro_logging() -> logging.Logger:
    """Αρχικοποιεί και επιστρέφει τον κεντρικό logger (MMPro) της εφαρμογής."""
    lg = logging.getLogger("MMPro")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter( "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    lg.addHandler(handler)
    return lg
_MMPRO_LOGGER: logging.Logger = _setup_mmpro_logging()


def _mmpro_get_logger() -> logging.Logger:
    """Επιστρέφει τον singleton logger της εφαρμογής."""
    return _MMPRO_LOGGER
"""Environment configuration.
Ρυθμίσεις/guards για σταθερότητα (OpenCV videoio, ONNXRuntime logs, threads) και checks.
"""


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 2 – Ασφαλής καταγραφή σφαλμάτων & crash logs
# ═══════════════════════════════════════════════════════════════════════

# ── Ασφαλής καταγραφή σφάλματος με προαιρετική δημιουργία crash log ─────────
# Αν crash_log=True, αποθηκεύει λεπτομερές αρχείο σφάλματος στον δίσκο
# (περιέχει traceback + dump όλων των threads). Επιστρέφει το path του αρχείου ή None.
def safe_log_error( msg: str, exc: Exception | None = None, level: int | None = None, *, crash_log: bool = False, crash_tag: str = "crash", dump_all_threads: bool = True,) -> str | None:
    """
    Καταγράφει σφάλμα με ασφάλεια.

    Παράμετροι:
        msg       – Μήνυμα σφάλματος.
        exc       – Exception αντικείμενο (προαιρετικό).
        crash_log – Αν True, αποθηκεύει crash log στον δίσκο.
    Επιστρέφει το path του crash log ή None.
    """
    if level is None:
        level = logging.ERROR
    crash_path: str | None = None

    # Capture traceback NOW — format_exc() returns correct data only when
    # called from within (or shortly after) an except block.
    _tb_str: str | None = None
    try:
        _fe = traceback.format_exc()
        if _fe and 'NoneType: None' not in _fe:
            _tb_str = _fe
    except Exception:
        pass
    # If format_exc gave nothing, try exc.__traceback__
    if not _tb_str and exc is not None and getattr(exc, '__traceback__', None) is not None:
        try:
            _tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            pass

    try:
        lg = _MMPRO_LOGGER
        if exc is None:
            lg.log(level, str(msg))
        else:
            lg.log(level, "%s: %s", msg, exc, exc_info=True)
    except Exception:
        try:
            if exc is None:
                print(f"[MMPro][ERROR] {msg}", file=sys.stderr)
            else:
                print(f"[MMPro][ERROR] {msg}: {exc}", file=sys.stderr)
                traceback.print_exc()
        except Exception:
            pass
    if crash_log:
        try:
            crash_path = _mmpro_write_crash_log(
                msg=str(msg),
                exc=exc,
                tag=str(crash_tag or "crash"),
                dump_all_threads=bool(dump_all_threads),
                tb_str=_tb_str,
            )
            if crash_path:
                _MMPRO_LOGGER.error("🧾 Crash log saved: %s", crash_path)
                try:
                    _mmpro_emit_crash_log_created(crash_path)
                except Exception:
                    pass
        except Exception:
            pass
    return crash_path
_LOG_ONCE_STATE: dict[str, dict] = {}
_LAST_CRASH_LOG_PATH: str | None = None


def get_last_crash_log_path() -> str | None:
    return _LAST_CRASH_LOG_PATH


def safe_log_once( key: str, msg: str, exc: Exception | None = None, *, group: str | None = None, level: int | None = None,) -> None:
    """Καταγράφει warning μόνο την πρώτη φορά ανά key (suppresses επαναλήψεις)."""
    if level is None:
        level = logging.WARNING
    k = str(key or "unknown")
    g = str(group or (k.split(":", 1)[0] if ":" in k else "General"))
    st = _LOG_ONCE_STATE.get(k)
    if st is None:
        st = {"count": 0, "group": g, "msg": str(msg), "exc": None}
        _LOG_ONCE_STATE[k] = st
    st["count"] = int(st.get("count", 0)) + 1
    if exc is not None:
        st["exc"] = repr(exc)
    if st["count"] == 1:
        try:
            if exc is None:
                _MMPRO_LOGGER.log(level, "[ONCE][%s] %s", g, msg)
            else:
                _MMPRO_LOGGER.log(level, "[ONCE][%s] %s: %s", g, msg, exc, exc_info=True)
        except Exception:
            try:
                print(f"[ONCE][{g}] {msg}", file=sys.stderr)
            except Exception:
                pass


def flush_log_once_summary( context: str = "", *, reset: bool = True, top_n: int = 10, min_total: int = 1,) -> str | None:
    """Εκτυπώνει σύνοψη όλων των suppressed warnings που έχουν καταγραφεί."""
    if not _LOG_ONCE_STATE:
        if _LAST_CRASH_LOG_PATH:
            return f"🧾 Τελευταίο Crash Log: {_LAST_CRASH_LOG_PATH}"
    items = list(_LOG_ONCE_STATE.items())
    total = sum(int(v.get("count", 0)) for _, v in items)
    if total < max(1, min_total):
        if reset:
            _LOG_ONCE_STATE.clear()
    groups: dict[str, list[tuple[str, dict]]] = {}
    for k, v in items:
        g = str(v.get("group") or "General")
        groups.setdefault(g, []).append((k, v))
    lines: list[str] = []
    title = ( f"📌 Σύνοψη suppressed warnings ({context})" if context else "📌 Σύνοψη suppressed warnings")
    lines.append(title)
    lines.append(f"Σύνολο εμφανίσεων: {total} · Μοναδικά keys: {len(items)}")
    for g in sorted(groups.keys()):
        rows = sorted(groups[g], key=lambda kv: int(kv[1].get("count", 0)), reverse=True)
        lines.append("")
        lines.append(f"🔸 {g}:")
        for k, v in rows[:top_n]:
            cnt = int(v.get("count", 0))
            entry_msg = str(v.get("msg") or k)
            lines.append(f"  • x{cnt} — {entry_msg}")
        remaining = len(rows) - top_n
        if remaining > 0:
            lines.append(f"  … +{remaining} ακόμη (suppressed)")
    if _LAST_CRASH_LOG_PATH:
        lines.append("")
        lines.append(f"🧾 Τελευταίο Crash Log: {_LAST_CRASH_LOG_PATH}")
    out = "\n".join(lines)
    try:
        _MMPRO_LOGGER.info(out)
    except Exception:
        pass
    if reset:
        _LOG_ONCE_STATE.clear()
    return out


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 3 – Υπολογισμός στατιστικών ανίχνευσης
# ═══════════════════════════════════════════════════════════════════════

# ── Υπολογισμός στατιστικών από χρόνους inference και confidence scores ───────
# Χρησιμοποιεί NumPy όταν είναι διαθέσιμο, αλλιώς pure-Python fallback.
# Επιστρέφει: mean/std/min/max χρόνου, μέσο confidence, συνολικές ανιχνεύσεις.
def calculate_safe_statistics( detection_times: list[float], conf_scores: dict[str, list[float]],) -> dict[str, Any]:
    """
    Υπολογίζει στατιστικά από χρόνους ανίχνευσης και confidence scores.

    Χρησιμοποιεί NumPy αν είναι διαθέσιμο, αλλιώς pure-Python fallback.
    Επιστρέφει dict με mean/std/min/max_time, avg_conf, total_detections.
    """
    try:
        import numpy as np
        dt_arr = np.array(detection_times, dtype=np.float64) if detection_times else np.array([], dtype=np.float64)
        all_scores: list[float] = []
        if conf_scores:
            for v in conf_scores.values():
                if v:
                    all_scores.extend(v)
        cs_arr = np.array(all_scores, dtype=np.float64) if all_scores else np.array([], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_time  = float(np.mean(dt_arr))   if dt_arr.size  else 0.0
            std_time   = float(np.std(dt_arr))    if dt_arr.size  else 0.0
            min_time   = float(np.min(dt_arr))    if dt_arr.size  else 0.0
            max_time   = float(np.max(dt_arr))    if dt_arr.size  else 0.0
            avg_conf   = float(np.mean(cs_arr))   if cs_arr.size  else 0.0
            total_det  = int(cs_arr.size)
        return { "mean_time": mean_time, "std_time":  std_time, "min_time":  min_time, "max_time":  max_time, "avg_conf":  avg_conf, "total_detections": total_det,}
    except ImportError:
        dt_list = list(detection_times or [])
        all_sc: list[float] = []
        for v in (conf_scores or {}).values():
            all_sc.extend(v or [])

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0
        return {
            "mean_time": _mean(dt_list),
            "std_time":  0.0,
            "min_time":  min(dt_list) if dt_list else 0.0,
            "max_time":  max(dt_list) if dt_list else 0.0,
            "avg_conf":  _mean(all_sc),
            "total_detections": len(all_sc),
        }


def _mmpro_get_crash_dir() -> Path | None:
    try:
        if root := os.environ.get("MM_PRO_ROOT_DIR"):
            base = Path(root).expanduser().resolve()
        else:
            base = Path.home() / "Models_Manager_Pro"
        return base / "Crash_Logs"
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None


def _mmpro_dump_all_threads_text() -> str:
    try:
        frames = getattr(sys, "_current_frames", None)
        if not callable(frames):
            return "Thread dump: sys._current_frames() not available."
        cur = frames()
        out_lines = []
        for t in threading.enumerate():
            ident = getattr(t, "ident", None)
            out_lines.append(f"--- Thread: {t.name} (id={ident}) ---")
            fr = cur.get(ident)
            if fr is None:
                out_lines.append("  <no frame>")
                continue
            out_lines.extend(traceback.format_stack(fr))
            out_lines.append("")
        return "".join(out_lines) if out_lines else "Thread dump: (empty)."
    except Exception as e:
        return f"Thread dump failed: {e}"


def _mmpro_write_crash_log(
    msg: str,
    exc: Exception | None,
    tag: str,
    dump_all_threads: bool = True,
    tb_str: str | None = None,         # Pre-formatted traceback (captured inside except block)
    exc_type: type | None = None,       # Exception type (from sys.excepthook args)
    exc_tb=None,                        # Traceback object (from sys.excepthook args)
) -> str | None:
    global _LAST_CRASH_LOG_PATH
    try:
        crash_dir = _mmpro_get_crash_dir()
        if crash_dir is None:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (tag or "crash"))
        path = crash_dir / f"crash_{safe_tag}_{ts}.log"

        import platform as _plat
        lines = []
        lines.append("=" * 72)
        lines.append(f"  Models Manager Pro — Crash Log")
        lines.append(f"  Tag:       {safe_tag}")
        lines.append(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  OS:        {_plat.system()} {_plat.release()} ({_plat.machine()})")
        lines.append(f"  Python:    {_plat.python_version()}")
        try:
            import torch as _t
            lines.append(f"  PyTorch:   {_t.__version__} | CUDA: {_t.version.cuda or 'N/A'}")
        except Exception:
            pass
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"MESSAGE: {msg}")
        lines.append("")

        # Exception info
        if exc is not None or exc_type is not None:
            _etype = exc_type or type(exc)
            _eval  = exc
            lines.append("─── Exception ─────────────────────────────────────────────────────")
            lines.append(f"  Type:    {_etype.__name__ if _etype else 'Unknown'}")
            lines.append(f"  Value:   {_eval!r}")
            lines.append("")

        # Traceback — in priority order:
        # 1. Pre-formatted string passed from caller (most reliable)
        # 2. Traceback object from excepthook
        # 3. Fallback: format_exc() (may be empty outside except block)
        _tb_text = None
        if tb_str and tb_str.strip() and 'NoneType: None' not in tb_str:
            _tb_text = tb_str.strip()
        elif exc_tb is not None:
            try:
                _tb_text = "".join(traceback.format_tb(exc_tb)).strip()
            except Exception:
                pass
        elif exc is not None and getattr(exc, '__traceback__', None) is not None:
            try:
                _tb_text = "".join(
                    traceback.format_tb(exc.__traceback__)
                ).strip()
            except Exception:
                pass
        # Last resort: format_exc (only useful if called inside an except block)
        if not _tb_text:
            try:
                _fe = traceback.format_exc()
                if _fe and 'NoneType: None' not in _fe:
                    _tb_text = _fe.strip()
            except Exception:
                pass

        if _tb_text:
            lines.append("─── Traceback ──────────────────────────────────────────────────────")
            for line in _tb_text.splitlines():
                lines.append(line)
            lines.append("")

        if dump_all_threads:
            lines.append("─── Thread Dump ────────────────────────────────────────────────────")
            lines.append(_mmpro_dump_all_threads_text())
            lines.append("")

        lines.append("=" * 72)
        try:
            path.write_text("\n".join(lines), encoding="utf-8", errors="replace")
        except Exception:
            with open(str(path), "wb") as f:
                f.write("\n".join(lines).encode("utf-8", errors="replace"))
        _LAST_CRASH_LOG_PATH = str(path)
        return str(path)
    except Exception:
        return None


def _mmpro_emit_crash_log_created(path_str: str) -> None:
    sigs = _mmpro_get_signals()
    if sigs is not None:
        try:
            sigs.crash_log_created.emit(str(path_str))
        except Exception:
            pass


def _mmpro_get_signals() -> "Any | None":
    try:
        from PySide6.QtCore import QObject as _QObject, Signal as _Signal
    except Exception:
        return None
    if getattr(_mmpro_get_signals, "_inst", None) is not None:
        return _mmpro_get_signals._inst

    class _MMProSignals(_QObject):
        crash_log_created = _Signal(str)
    try:
        _mmpro_get_signals._inst = _MMProSignals()
    except Exception:
        _mmpro_get_signals._inst = None
    return _mmpro_get_signals._inst


def _mmpro_install_global_exception_hooks() -> None:

    def _hook(exctype, value, tb):
        try:
            # format_tb while we still have the traceback object
            _tb_str = "".join(traceback.format_exception(exctype, value, tb))
            _mmpro_write_crash_log(
                msg="Uncaught Exception",
                exc=value,
                tag="uncaught",
                dump_all_threads=True,
                tb_str=_tb_str,
                exc_type=exctype,
                exc_tb=tb,
            )
        except Exception:
            try:
                safe_log_error("Uncaught Exception", value, crash_log=True,
                               crash_tag="uncaught", dump_all_threads=True)
            except Exception:
                pass
        try:
            sys.__excepthook__(exctype, value, tb)
        except Exception:
            pass
    sys.excepthook = _hook
    if hasattr(threading, "excepthook"):

        def _th_hook(args):
            try:
                _exc_val  = getattr(args, 'exc_value', None)
                _exc_type = getattr(args, 'exc_type', None)
                _exc_tb   = getattr(args, 'exc_traceback', None)
                _tb_str = None
                try:
                    if _exc_type and _exc_val and _exc_tb:
                        _tb_str = "".join(
                            traceback.format_exception(_exc_type, _exc_val, _exc_tb))
                    elif _exc_val and getattr(_exc_val, '__traceback__', None):
                        _tb_str = "".join(
                            traceback.format_exception(
                                type(_exc_val), _exc_val, _exc_val.__traceback__))
                except Exception:
                    pass
                _thread_name = str(getattr(args, 'thread', None) or 'Unknown')
                _mmpro_write_crash_log(
                    msg=f"Thread Exception ({_thread_name})",
                    exc=_exc_val,
                    tag="thread",
                    dump_all_threads=True,
                    tb_str=_tb_str,
                    exc_type=_exc_type,
                    exc_tb=_exc_tb,
                )
            except Exception:
                try:
                    safe_log_error(
                        f"Thread Exception ({getattr(args, 'thread', None)})",
                        getattr(args, "exc_value", None),
                        crash_log=True, crash_tag="thread", dump_all_threads=True,
                    )
                except Exception:
                    pass
            try:
                threading.__excepthook__(args)
            except Exception:
                try:
                    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
                except Exception:
                    pass
        try:
            threading.excepthook = _th_hook
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 4 – Διαμόρφωση περιβάλλοντος (OpenCV / ONNXRuntime / DLLs)
# ═══════════════════════════════════════════════════════════════════════

def configure_opencv_videoio_env() -> None:
    """Ρυθμίζει env variables για σταθερή λειτουργία του OpenCV VideoIO."""
    if os.name != "nt":
        return
    backend = str(os.environ.get("MM_PRO_CAM_BACKEND", "dshow")).strip().lower()
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    if backend in ("dshow", "directshow"):
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "10000")
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "10000")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "5000")


def configure_onnxruntime_logging(severity: int = 3, verbosity: int = 0) -> None:
    """Ορίζει το επίπεδο καταγραφής (severity) του ONNX Runtime."""
    try:
        sev = int(severity)
    except Exception:
        sev = 3
    try:
        verb = int(verbosity)
    except Exception:
        verb = 0
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(sev)
        if sev == 0:
            ort.set_default_logger_verbosity(verb)
    except Exception:
        return
_DLL_DIRS_ADDED: set[str] = set()


def _add_dll_search_dir(p: Path) -> None:
    if os.name != "nt":
        return
    try:
        p = Path(p).resolve()
        if not p.exists() or not p.is_dir():
            return
        key = str(p)
        if key in _DLL_DIRS_ADDED:
            return
        _DLL_DIRS_ADDED.add(key)
        add_dir = getattr(os, "add_dll_directory", None)
        if callable(add_dir):
            try:
                add_dir(key)
            except Exception:
                pass
        try:
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = key + (os.pathsep + old if old else "")
        except Exception:
            pass
    except Exception:
        pass


def _add_dll_search_paths_best_effort() -> None:
    if os.name != "nt":
        return
    base_dirs: list[Path] = []
    try:
        base_dirs.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    try:
        if meipass := getattr(sys, "_MEIPASS", None):
            base_dirs.append(Path(meipass).resolve())
    except Exception:
        pass
    extra_subs = [ "_internal", "tensorrt", "ncnn", "pnnx", "onnxruntime", "torch", str(Path("torch") / "lib"),]
    for bd in list(dict.fromkeys([d for d in base_dirs if isinstance(d, Path)])):
        _add_dll_search_dir(bd)
        for s in extra_subs:
            try:
                _add_dll_search_dir(bd / s)
            except Exception:
                pass
    for bd in list(dict.fromkeys([d for d in base_dirs if isinstance(d, Path)])):
        try:
            bd = bd.resolve()
            if not bd.exists():
                continue
            for root, dirs, files in os.walk(str(bd)):
                try:
                    rel = Path(root).resolve().relative_to(bd)
                    if len(rel.parts) > 2:
                        dirs[:] = []
                        continue
                except Exception:
                    pass
                lower_files = [f.lower() for f in files]
                if any(x.endswith((".dll", ".pyd")) for x in lower_files):
                    _add_dll_search_dir(Path(root))
        except Exception:
            pass


def _safe_mkdir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _pick_root_dir() -> Path:
    exe_dir = Path(sys.executable).resolve().parent
    if _safe_mkdir(exe_dir / "Data_Sets"):
        return exe_dir
    if base := os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA"):
        return Path(base).expanduser().resolve() / "Models_Manager_Pro"
        _safe_mkdir(root / "Data_Sets")
    return Path.home().resolve() / "Models_Manager_Pro"
    _safe_mkdir(root / "Data_Sets")


def _mmpro_early_init() -> None:
    try:
        if os.name == "nt" and getattr(sys, "frozen", False):
            _add_dll_search_paths_best_effort()
    except Exception:
        pass
    try:
        if getattr(sys, "frozen", False):
            root = _pick_root_dir()
            os.environ.setdefault("MM_PRO_ROOT_DIR", str(root))
            os.environ.setdefault("MM_PRO_DATA_DIR", str(root))
            try:
                os.chdir(str(root))
            except Exception:
                pass
    except Exception:
        pass
    try:
        configure_opencv_videoio_env()
    except Exception:
        if os.name == "nt":
            os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_DSHOW", "10000")
            os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
            os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        lvl = int(os.environ.get("MM_PRO_ORT_LOG_LEVEL", "3"))
    except Exception:
        lvl = 3
    try:
        configure_onnxruntime_logging(severity=lvl, verbosity=0)
    except Exception:
        pass
_mmpro_early_init()
"""Runner utilities.
Helpers για subprocess execution, timeouts, safe env και parsing αποτελεσμάτων.
"""


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 5 – Runner utilities: subprocess, faulthandler, workers
# ═══════════════════════════════════════════════════════════════════════

def enable_faulthandler(crash_dir: Path, tag: str) -> Path | None:
    """Ενεργοποιεί τον faulthandler Python για καταγραφή C-level crashes σε αρχείο."""
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    try:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = crash_dir / f'faulthandler_{tag}_{stamp}.log'
        fh = log_path.open('w', encoding='utf-8', errors='replace')
        faulthandler.enable(file=fh, all_threads=True)
        return log_path
    except Exception as e:
        _MMPRO_LOGGER.debug("enable_faulthandler error: %s", e)


def _run_cmd(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', **kw)


def print_line(prefix: str, payload: str = '') -> None:
    """Εκτυπώνει γραμμή με prefix στο stdout (thread-safe)· χρήση από workers."""
    try:
        safe_payload = str(payload or '').replace('\r', '\\r').replace('\n', '\\n')
        sys.stdout.write(f"{prefix}{safe_payload}\n")
        sys.stdout.flush()
    except Exception:
        pass
"""Path constants.
Κεντρικοποιεί όλα τα paths φακέλων του project (models, datasets, exports, reports).
"""
"""Κοινός helper για paths + overwrite policy (μία πηγή αλήθειας).
Το module είναι stdlib-only (Path/shutil). Δεν κάνει import το core, ώστε να; αποφεύγονται κύκλοι.
UI usage (π.χ. training_tab):
    target = export_target_for(model_path, export_format); overwrite = target_exists_and_nonempty(target) and ask_overwrite(self, 'Εξαγωγή', target)
    # γράψε overwrite στο job
Worker usage (export_worker):
    target = export_target_for(self.model_path, self.export_format); ok, msg = ensure_overwrite_target(target, overwrite=self.overwrite)
"""
from typing import Literal
ExportKind = Literal["file", "dir"]


@dataclass(frozen=True)


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 6 – Export target helpers (ONNX / TensorRT / NCNN)
# ═══════════════════════════════════════════════════════════════════════

class ExportTarget:
    """Frozen dataclass που αντιπροσωπεύει τον στόχο εξαγωγής (path + kind)."""
    kind: ExportKind
    path: Path


# ── Επιστρέφει το ExportTarget (path + kind) για το δοθέν format εξαγωγής ────
# Υποστηριζόμενα formats: 'onnx' → .onnx, 'tensorrt'/'engine' → .engine, 'ncnn' → _ncnn_model/
def export_target_for(model_path: Path, export_format: str) -> ExportTarget:
    """Επιστρέφει ExportTarget (path + kind) για το δοθέν format εξαγωγής."""
    fmt = (export_format or "").strip().lower()
    mp = Path(model_path)
    if fmt == "onnx":
        return ExportTarget("file", mp.with_suffix(".onnx"))
    if fmt in {"tensorrt", "engine", "trt"}:
        return ExportTarget("file", mp.with_suffix(".engine"))
    if fmt == "ncnn":
        return ExportTarget("dir", mp.parent / f"{mp.stem}_ncnn_model")
    raise ValueError(f"Unknown export_format: {export_format!r}")


def dir_has_files(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        return any(p.iterdir())
    except Exception:
        return True


def target_exists_and_nonempty(target: ExportTarget) -> bool:
    """Ελέγχει αν το export target υπάρχει ήδη και δεν είναι άδειο."""
    try:
        if target.kind == "file":
            return target.path.exists()
        return target.path.exists() and dir_has_files(target.path)
    except Exception:
        return True


def _unlink_missing_ok(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except TypeError:
        if p.exists():
            p.unlink()


def safe_unlink(p: Path) -> tuple[bool, str]:
    try:
        if not p.exists():
            return True, ""
    except Exception:
        pass
    try:
        _unlink_missing_ok(p)
        return True, ""
    except Exception as e:
        return False, str(e)


def _on_rm_error(func, path, exc_info) -> None:
    try:
        os.chmod(path, 0o666)
        func(path)
    except Exception:
        raise


def safe_rmtree(p: Path) -> tuple[bool, str]:
    try:
        if not p.exists():
            return True, ""
    except Exception:
        pass
    try:
        shutil.rmtree(p, onerror=_on_rm_error)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Διαγράφει ή επαληθεύει ότι το export target είναι ελεύθερο για overwrite ─
# Αν overwrite=False και το target υπάρχει, επιστρέφει (False, 'exists').
# Αν overwrite=True, διαγράφει το αρχείο ή τον φάκελο.
def ensure_overwrite_target( target: ExportTarget, overwrite: bool, require_nonempty_dir: bool = True,) -> tuple[bool, str]:
    """Διαγράφει ή επαληθεύει ότι το target είναι ελεύθερο για overwrite."""
    p = target.path
    try:
        if not p.exists():
            return True, ""
    except Exception:
        pass
    if target.kind == "dir" and require_nonempty_dir:
        try:
            if p.is_dir() and not dir_has_files(p):
                return True, ""
        except Exception:
            pass
    if not overwrite:
        return False, "exists"
    if target.kind == "file":
        return safe_unlink(p)
    return safe_rmtree(p)


def ask_overwrite(parent, title: str, target: ExportTarget, default_no: bool = True) -> bool:
    """Εμφανίζει QMessageBox για επιβεβαίωση αντικατάστασης υπάρχοντος target."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except Exception:
        return False
    p = target.path
    if target.kind == "file":
        text = f"Υπάρχει ήδη αρχείο: {p.name}\n\nΘέλεις να το αντικαταστήσεις;"
    else:
        text = f"Υπάρχει ήδη φάκελος: {p.name}\n\nΘέλεις να τον αντικαταστήσεις;"
    default_btn = QMessageBox.StandardButton.No if default_no else QMessageBox.StandardButton.Yes
    reply = QMessageBox.question( parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, default_btn,)
    return reply == QMessageBox.StandardButton.Yes


def best_effort_unlink(p: Path) -> None:
    try:
        _unlink_missing_ok(p)
    except Exception:
        pass


def best_effort_rmtree(p: Path) -> None:
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def best_effort_remove(p: Path) -> None:
    try:
        if p.is_dir():
            best_effort_rmtree(p)
        else:
            best_effort_unlink(p)
    except Exception:
        pass
"""PDF reports.
Δημιουργία PDF αναφορών (training/detection) με reportlab.
"""


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 7 – Δημιουργία PDF αναφορών (ReportLab)
# ═══════════════════════════════════════════════════════════════════════

def _safe_str(x: Any) -> str:
    if x is None:
        return ''
    try:
        return str(x)
    except Exception:
        return ''


def _rl_imports() -> dict[str, Any]:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import ( BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, NextPageTemplate, KeepTogether,)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    return {
        "A4": A4,
        "landscape": landscape,
        "mm": mm,
        "colors": colors,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle,
        "BaseDocTemplate": BaseDocTemplate,
        "Frame": Frame,
        "PageTemplate": PageTemplate,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "Image": Image,
        "Table": Table,
        "TableStyle": TableStyle,
        "PageBreak": PageBreak,
        "NextPageTemplate": NextPageTemplate,
        "KeepTogether": KeepTogether,
        "pdfmetrics": pdfmetrics,
        "TTFont": TTFont,
    }


@dataclass


class PdfTheme:
    """Dataclass με τα χρώματα (brand/ink/muted/grid) για τις PDF αναφορές."""
    brand_dark:   str = '#0b1220'
    brand_accent: str = '#7c3aed'
    ink:          str = '#0f172a'
    muted:        str = '#475569'
    grid:         str = '#e2e8f0'
    soft:         str = '#f1f5f9'


def _register_fonts(resource_root: Path) -> tuple[str, str]:
    rl = _rl_imports()
    pdfmetrics = rl['pdfmetrics']
    TTFont = rl['TTFont']
    reg_name = 'MMProSans'
    bold_name = 'MMProSans-Bold'
    roots: list[Path] = []
    try:
        roots.append(Path(resource_root))
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    try:
        roots.append(Path(getattr(sys, '_MEIPASS')))
    except Exception:
        pass
    seen: set[str] = set()
    roots = [r for r in roots if str(r) not in seen and not seen.add(str(r))]
    candidates: list[tuple[Path, Path]] = []
    for r in roots:
        candidates.extend([
            (r / 'assets' / 'fonts' / 'DejaVuSans.ttf', r / 'assets' / 'fonts' / 'DejaVuSans-Bold.ttf'),
            (r / 'assets' / 'fonts' / 'NotoSans-Regular.ttf', r / 'assets' / 'fonts' / 'NotoSans-Bold.ttf'),
            (r / 'assets' / 'fonts' / 'LiberationSans-Regular.ttf', r / 'assets' / 'fonts' / 'LiberationSans-Bold.ttf'),
        ])
    windir = os.environ.get('WINDIR', r'C:\Windows').strip()
    win_fonts = Path(windir) / 'Fonts'
    candidates.extend([
        (win_fonts / 'arial.ttf', win_fonts / 'arialbd.ttf'),
        (win_fonts / 'segoeui.ttf', win_fonts / 'segoeuib.ttf'),
        (win_fonts / 'calibri.ttf', win_fonts / 'calibrib.ttf'),
        (win_fonts / 'tahoma.ttf', win_fonts / 'tahomabd.ttf'),
        (win_fonts / 'times.ttf', win_fonts / 'timesbd.ttf'),
    ])
    candidates.extend([
        (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
        (Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'), Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')),
        (Path('/Library/Fonts/Arial.ttf'), Path('/Library/Fonts/Arial Bold.ttf')),
        (Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')),
    ])
    env_reg = (os.environ.get('MM_PRO_PDF_FONT_REG') or '').strip()
    env_bold = (os.environ.get('MM_PRO_PDF_FONT_BOLD') or '').strip()
    if env_reg:
        try:
            candidates.insert(0, (Path(env_reg), Path(env_bold) if env_bold else Path(env_reg)))
        except Exception:
            pass
    tried: set[str] = set()
    for reg_path, bold_path in candidates:
        key = f'{reg_path}|{bold_path}'
        if key in tried:
            continue
        tried.add(key)
        try:
            if not reg_path.is_file():
                continue
            pdfmetrics.registerFont(TTFont(reg_name, str(reg_path)))
            use_bold = reg_name
            if bold_path.is_file():
                try:
                    pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
                    use_bold = bold_name
                except Exception:
                    use_bold = reg_name
            try:
                pdfmetrics.registerFontFamily(reg_name, normal=reg_name, bold=use_bold, italic=reg_name, boldItalic=use_bold)
            except Exception:
                pass
            return (reg_name, use_bold)
        except Exception:
            continue
    return ('Helvetica', 'Helvetica-Bold')


def _mk_styles(resource_root: Path, theme: PdfTheme):
    rl = _rl_imports()
    colors = rl['colors']
    getSampleStyleSheet = rl['getSampleStyleSheet']
    ParagraphStyle = rl['ParagraphStyle']
    font_reg, font_bold = _register_fonts(resource_root)
    base = getSampleStyleSheet()
    return {'font_reg': font_reg, 'font_bold': font_bold, 'title': ParagraphStyle(name='MMProTitle', parent=base['Title'], fontName=font_bold, fontSize=22, leading=26, textColor=colors.HexColor(theme.ink), spaceAfter=10), 'subtitle': ParagraphStyle(name='MMProSubtitle', parent=base['Normal'], fontName=font_reg, fontSize=11, leading=14, textColor=colors.HexColor(theme.muted), spaceAfter=10), 'h1': ParagraphStyle(name='MMProH1', parent=base['Heading2'], fontName=font_bold, fontSize=14, leading=18, textColor=colors.HexColor(theme.ink), spaceBefore=10, spaceAfter=6), 'p': ParagraphStyle(name='MMProP', parent=base['Normal'], fontName=font_reg, fontSize=10, leading=14, textColor=colors.HexColor(theme.ink)), 'small': ParagraphStyle(name='MMProSmall', parent=base['Normal'], fontName=font_reg, fontSize=8.8, leading=12, textColor=colors.HexColor(theme.muted)), 'chip': ParagraphStyle(name='MMProChip', parent=base['Normal'], fontName=font_bold, fontSize=9.5, leading=12, textColor=colors.HexColor(theme.brand_accent)), 'chart_title': ParagraphStyle(name='MMProChartTitle', parent=base['Normal'], fontName=font_bold, fontSize=10.5, leading=12.5, textColor=colors.HexColor(theme.ink), spaceAfter=4)}


def _draw_header_footer(canvas, doc, theme: PdfTheme, styles, title: str, run_id: str, margin_x: float | None=None):
    rl = _rl_imports()
    mm = rl['mm']
    colors = rl['colors']
    page_w, page_h = doc.pagesize
    mx = 18 * mm if margin_x is None else float(margin_x)
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(theme.grid))
    canvas.setLineWidth(0.8)
    canvas.line(mx, page_h - 18 * mm, page_w - mx, page_h - 18 * mm)
    canvas.setFillColor(colors.HexColor(theme.ink))
    canvas.setFont(styles['font_bold'], 9.5)
    canvas.drawString(mx, page_h - 13.5 * mm, title)
    canvas.setFillColor(colors.HexColor(theme.muted))
    canvas.setFont(styles['font_reg'], 8.8)
    canvas.drawRightString(page_w - mx, page_h - 13.5 * mm, run_id)
    canvas.setStrokeColor(colors.HexColor(theme.grid))
    canvas.setLineWidth(0.8)
    canvas.line(mx, 16 * mm, page_w - mx, 16 * mm)
    canvas.setFillColor(colors.HexColor(theme.muted))
    canvas.setFont(styles['font_reg'], 8.5)
    canvas.drawString(mx, 10.5 * mm, 'Models Manager Pro - PDF Report')
    canvas.drawRightString(page_w - mx, 10.5 * mm, f'Page {doc.page}')
    canvas.restoreState()


def _fit_image(path: Path, max_w: float, max_h: float):
    rl = _rl_imports()
    Image = rl['Image']
    w = h = 0
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path)
        w, h = img.size
        img.close()
    except Exception:
        try:
            from reportlab.lib.utils import ImageReader
            w, h = ImageReader(str(path)).getSize()
        except Exception:
            w = h = 0
    if w <= 0 or h <= 0:
        img_flow = Image(str(path))
        try:
            img_flow.hAlign = 'CENTER'
        except Exception:
            pass
        return img_flow
    scale = min(max_w / float(w), max_h / float(h))
    img_flow = Image(str(path), width=w * scale, height=h * scale)
    try:
        img_flow.hAlign = 'CENTER'
    except Exception:
        pass
    return img_flow


def _table(data: Sequence[Sequence[str]], col_widths: Sequence[float], theme: PdfTheme, styles):
    rl = _rl_imports()
    Table = rl['Table']
    TableStyle = rl['TableStyle']
    colors = rl['colors']
    t = Table(data, colWidths=list(col_widths), hAlign='LEFT')
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), styles['font_reg']), ('FONTSIZE', (0, 0), (-1, -1), 9.2), ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(theme.ink)), ('GRID', (0, 0), (-1, -1), 0.6, colors.HexColor(theme.grid)), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(theme.soft)]), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(theme.brand_dark)), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONT', (0, 0), (-1, 0), styles['font_bold']), ('FONTSIZE', (0, 0), (-1, 0), 9.6), ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def _cover(story, title: str, subtitle: str, meta_lines: Sequence[str], logo_path: Path | None, theme: PdfTheme, styles):
    rl = _rl_imports()
    Spacer = rl['Spacer']
    Paragraph = rl['Paragraph']
    mm = rl['mm']
    KeepTogether = rl['KeepTogether']
    if logo_path and logo_path.is_file():
        try:
            story.append(_fit_image(logo_path, max_w=32 * mm, max_h=32 * mm))
            story.append(Spacer(1, 10))
        except Exception:
            pass
    story.append(Paragraph(title, styles['title']))
    story.append(Paragraph(subtitle, styles['subtitle']))
    story.append(Spacer(1, 10))
    lines = '<br/>'.join([_safe_str(x) for x in meta_lines if _safe_str(x)])
    if lines:
        story.append(KeepTogether([Paragraph('<b>Run snapshot</b>', styles['h1']), Paragraph(lines, styles['p'])]))
    story.append(Spacer(1, 18))
    story.append(Paragraph('Generated by Models Manager Pro', styles['small']))


def build_training_report_pdf(output_pdf: Path, resource_root: Path, run_id: str, model_name: str, dataset_name: str, device: str, imgsz: int, run_info_rows: Sequence[tuple[str, str]], metrics_rows: Sequence[tuple[str, str]], charts: Sequence[tuple[str, Path]], extra_pages: Sequence[tuple[str, Path]]=(), notes: Sequence[str | None]=None, model_type: str = 'yolo_detect') -> Path:
    """
    Δημιουργεί PDF αναφορά εκπαίδευσης (Training Report).

    Παράμετρος model_type:
      'yolo_detect'   → YOLO Object Detection Training Report
      'yolo_classify' → YOLO Classification Training Report
      'cnn'           → CNN (torchvision) Training Report

    Περιλαμβάνει cover page, run overview, key metrics και charts
    σε landscape pages.
    """
    rl = _rl_imports()
    A4 = rl['A4']
    landscape = rl['landscape']
    mm = rl['mm']
    BaseDocTemplate = rl['BaseDocTemplate']
    Frame = rl['Frame']
    PageTemplate = rl['PageTemplate']
    Paragraph = rl['Paragraph']
    Spacer = rl['Spacer']
    PageBreak = rl['PageBreak']
    NextPageTemplate = rl['NextPageTemplate']
    KeepTogether = rl['KeepTogether']
    theme = PdfTheme()
    styles = _mk_styles(resource_root, theme)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    left = right = 18 * mm
    top = 22 * mm
    bottom = 20 * mm
    frame = Frame(left, bottom, A4[0] - left - right, A4[1] - top - bottom, id='normal')
    doc = BaseDocTemplate(str(output_pdf), pagesize=A4, leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom, title=f'Training Report - {run_id}', author='Models Manager Pro')
    doc.report_title = 'Training Report'

    def _on_page(canvas, doc_):
        _title = getattr(doc_, 'report_title', 'Training Report')
        _draw_header_footer(canvas, doc_, theme, styles, _title, run_id)
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame], onPage=_on_page)])
    chart_pagesize = landscape(A4)
    chart_left = chart_right = 12 * mm
    chart_top = 22 * mm
    chart_bottom = 20 * mm
    chart_max_w = chart_pagesize[0] - chart_left - chart_right
    chart_max_h = chart_pagesize[1] - chart_top - chart_bottom
    chart_frame = Frame(chart_left, chart_bottom, chart_pagesize[0] - chart_left - chart_right, chart_pagesize[1] - chart_top - chart_bottom, id='chart')

    def _on_page_chart(canvas, doc_):
        _title = getattr(doc_, 'report_title', 'Training Report')
        _draw_header_footer(canvas, doc_, theme, styles, f'{_title} (Charts)', run_id, margin_x=12 * mm)
    doc.addPageTemplates([PageTemplate(id='chart', frames=[chart_frame], onPage=_on_page_chart, pagesize=chart_pagesize)])
    story: list = []
    logo = None
    try:
        logo = resource_root / 'app_icon.png'
    except Exception:
        logo = None
    cover_meta = [f'Model: {model_name}', f'Dataset: {dataset_name}', f'Device: {device}', f'Image size: {imgsz}px']
    _mt = str(model_type or 'yolo_detect').lower()
    if _mt == 'cnn':
        _cover_title    = 'CNN Training Report'
        _cover_subtitle = 'CNN (torchvision) classification training summary — loss & accuracy curves'
        doc.report_title = 'CNN Training Report'
    elif _mt == 'yolo_classify':
        _cover_title    = 'YOLO Classification Training Report'
        _cover_subtitle = 'YOLO-CLS classification training summary — loss & accuracy curves'
        doc.report_title = 'YOLO Classification Training Report'
    else:
        _cover_title    = 'YOLO Detection Training Report'
        _cover_subtitle = 'YOLO object detection training summary — loss, mAP & precision/recall curves'
        doc.report_title = 'YOLO Detection Training Report'
    _cover(story, title=_cover_title, subtitle=_cover_subtitle, meta_lines=cover_meta, logo_path=logo, theme=theme, styles=styles)
    story.append(PageBreak())
    story.append(Paragraph('Run overview', styles['h1']))
    run_table = _table([['Field', 'Value']] + [[k, v] for k, v in run_info_rows], col_widths=[60 * mm, 110 * mm], theme=theme, styles=styles)
    story.append(run_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph('Key metrics', styles['h1']))
    met_table = _table([['Metric', 'Value']] + [[k, v] for k, v in metrics_rows], col_widths=[75 * mm, 95 * mm], theme=theme, styles=styles)
    story.append(met_table)
    story.append(Spacer(1, 10))
    if notes:
        story.append(Paragraph('Notes', styles['h1']))
        story.append(Paragraph('<br/>'.join([_safe_str(n) for n in notes]), styles['p']))
    story.append(NextPageTemplate('chart'))
    story.append(PageBreak())
    for title, img_path in charts:
        story.append(Paragraph(_safe_str(title), styles['chart_title']))
        try:
            story.append(_fit_image(img_path, max_w=chart_max_w, max_h=chart_max_h - 14 * mm))
        except Exception:
            story.append(Paragraph(f'(Failed to embed image: {img_path})', styles['small']))
        story.append(PageBreak())
    for title, img_path in extra_pages:
        story.append(Paragraph(_safe_str(title), styles['chart_title']))
        try:
            story.append(_fit_image(img_path, max_w=chart_max_w, max_h=chart_max_h - 14 * mm))
        except Exception:
            story.append(Paragraph(f'(Failed to embed image: {img_path})', styles['small']))
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    doc.build(story)
    return output_pdf


def build_detection_report_pdf(output_pdf: Path, resource_root: Path, run_id: str, run_info_rows: Sequence[tuple[str, str]], metrics_rows: Sequence[tuple[str, str]], top_classes: Sequence[Sequence[str]], charts: Sequence[tuple[str, Path]], notes: Sequence[str | None]=None, model_type: str = 'yolo_detect') -> Path:
    """
    Δημιουργεί PDF αναφορά ανίχνευσης / αξιολόγησης μοντέλου (Detection / Stats Report).

    Παράμετρος model_type:
      'yolo_detect'   → YOLO Detection Statistics Report
      'yolo_classify' → YOLO Classification Statistics Report
      'cnn'           → CNN Classification Statistics Report

    Περιλαμβάνει: run overview, metrics, top classes, charts.
    """
    rl = _rl_imports()
    A4 = rl['A4']
    landscape = rl['landscape']
    mm = rl['mm']
    BaseDocTemplate = rl['BaseDocTemplate']
    Frame = rl['Frame']
    PageTemplate = rl['PageTemplate']
    Paragraph = rl['Paragraph']
    Spacer = rl['Spacer']
    PageBreak = rl['PageBreak']
    NextPageTemplate = rl['NextPageTemplate']
    theme = PdfTheme()
    styles = _mk_styles(resource_root, theme)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    left = right = 18 * mm
    top = 22 * mm
    bottom = 20 * mm
    frame = Frame(left, bottom, A4[0] - left - right, A4[1] - top - bottom, id='normal')
    doc = BaseDocTemplate(str(output_pdf), pagesize=A4, leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom, title=f'Detection Report - {run_id}', author='Models Manager Pro')
    _mt_d = str(model_type or 'yolo_detect').lower()
    if _mt_d == 'cnn':
        _det_title    = 'CNN Classification Statistics Report'
        _det_subtitle = 'CNN (torchvision) model evaluation — per-class accuracy, confidence & charts'
    elif _mt_d == 'yolo_classify':
        _det_title    = 'YOLO Classification Statistics Report'
        _det_subtitle = 'YOLO-CLS model evaluation — per-class accuracy & distribution charts'
    else:
        _det_title    = 'YOLO Detection Statistics Report'
        _det_subtitle = 'YOLO object detection model evaluation — per-class precision, recall & mAP charts'
    doc.report_title = _det_title

    def _on_page(canvas, doc_):
        _title = getattr(doc_, 'report_title', _det_title)
        _draw_header_footer(canvas, doc_, theme, styles, _title, run_id)
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame], onPage=_on_page)])
    chart_pagesize = landscape(A4)
    chart_left = chart_right = 12 * mm
    chart_top = 22 * mm
    chart_bottom = 20 * mm
    chart_max_w = chart_pagesize[0] - chart_left - chart_right
    chart_max_h = chart_pagesize[1] - chart_top - chart_bottom
    chart_frame = Frame(chart_left, chart_bottom, chart_pagesize[0] - chart_left - chart_right, chart_pagesize[1] - chart_top - chart_bottom, id='chart')

    def _on_page_chart(canvas, doc_):
        _title = getattr(doc_, 'report_title', _det_title)
        _draw_header_footer(canvas, doc_, theme, styles, f'{_title} (Charts)', run_id, margin_x=12 * mm)
    doc.addPageTemplates([PageTemplate(id='chart', frames=[chart_frame], onPage=_on_page_chart, pagesize=chart_pagesize)])
    story: list = []
    logo = None
    try:
        logo = resource_root / 'app_icon.png'
    except Exception:
        logo = None
    _cover(story, title=_det_title, subtitle=_det_subtitle, meta_lines=[f'Run: {run_id}'], logo_path=logo, theme=theme, styles=styles)
    story.append(PageBreak())
    story.append(Paragraph('Run overview', styles['h1']))
    story.append(_table([['Field', 'Value']] + [[k, v] for k, v in run_info_rows], col_widths=[60 * mm, 110 * mm], theme=theme, styles=styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph('Key metrics', styles['h1']))
    story.append(_table([['Metric', 'Value']] + [[k, v] for k, v in metrics_rows], col_widths=[75 * mm, 95 * mm], theme=theme, styles=styles))
    story.append(Spacer(1, 10))
    if top_classes:
        story.append(Paragraph('Top classes', styles['h1']))
        story.append(_table(list(top_classes), col_widths=[70 * mm, 35 * mm, 30 * mm, 35 * mm], theme=theme, styles=styles))
    if notes:
        story.append(Spacer(1, 10))
        story.append(Paragraph('Notes', styles['h1']))
        story.append(Paragraph('<br/>'.join([_safe_str(n) for n in notes]), styles['p']))
    story.append(NextPageTemplate('chart'))
    story.append(PageBreak())
    for title, img_path in charts:
        story.append(Paragraph(_safe_str(title), styles['chart_title']))
        try:
            story.append(_fit_image(img_path, max_w=chart_max_w, max_h=chart_max_h - 14 * mm))
        except Exception:
            story.append(Paragraph(f'(Failed to embed image: {img_path})', styles['small']))
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    doc.build(story)
    return output_pdf
"""Diagnostics utilities.
Μη-UI βοηθητικές συναρτήσεις για συλλογή πληροφοριών συστήματος και checks.
"""


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 8 – Διαγνωστικά utilities (σύστημα, hardware, packages)
# ═══════════════════════════════════════════════════════════════════════

def _build_diagnostics_log_lines(data: dict[str, Any], *, for_view: bool=False, view_pip_freeze_max_lines: int=250) -> list[tuple[str, str | None, bool, bool]]:
    Colors = globals().get('Colors', None)
    if Colors is None:

        class _FallbackColors:
            HEADER = None
            BLUE = None
            CYAN = None
            GREEN = None
            YELLOW = None
            MAGENTA = None
            RED = None
            LIGHT = None
        Colors = _FallbackColors()
    app = data.get('app', {})
    osd = data.get('os', {})
    hd = data.get('hardware', {})
    gd = data.get('gpu', {})
    pd = data.get('packages', {})
    ed = data.get('env', {})
    sep = '═' * 72
    sub_sep = '─' * 72

    def L(text: str, color: str | None=None, bold: bool=False, underline: bool=False):
        return (text, color, bool(bold), bool(underline))
    lines: list[tuple[str, str | None, bool, bool]] = []
    lines.append(L(sep, getattr(Colors, 'HEADER', None), bold=True))
    lines.append(L(f"🧪 DIAGNOSTICS ΑΝΑΦΟΡΑ – {app.get('name', 'Models Manager Pro')}", getattr(Colors, 'CYAN', None), bold=True))
    lines.append(L(sub_sep, getattr(Colors, 'HEADER', None), bold=True))
    lines.append(L(f"Timestamp: {data.get('timestamp', '')}", getattr(Colors, 'LIGHT', None)))
    lines.append(L(f"App Version: v{app.get('version', '')}", getattr(Colors, 'LIGHT', None)))
    py_full = str(data.get('python', {}).get('version', '') or '')
    py_short = (py_full.split() or [''])[0]
    torch_ver = str(pd.get('torch', 'N/A'))
    ultra_ver = str(pd.get('ultralytics', 'N/A'))
    lines.append(L(f'Python: {py_short} | PyTorch: {torch_ver} | Ultralytics: {ultra_ver}', getattr(Colors, 'LIGHT', None)))
    if app.get('python_executable'):
        lines.append(L(f"Python exe: {app.get('python_executable', '')}", getattr(Colors, 'LIGHT', None)))
    if app.get('cwd'):
        lines.append(L(f"CWD: {app.get('cwd', '')}", getattr(Colors, 'LIGHT', None)))
    if app.get('root_dir'):
        lines.append(L(f"Root: {app.get('root_dir', '')}", getattr(Colors, 'LIGHT', None)))
    lines.append(L('', None))
    lines.append(L('=== 🪟 ΛΕΙΤΟΥΡΓΙΚΟ ΣΥΣΤΗΜΑ ===', getattr(Colors, 'HEADER', None), bold=True))
    for k in ['platform', 'system', 'release', 'version', 'machine', 'processor']:
        v = osd.get(k)
        if v:
            lines.append(L(f'{k:<12}: {v}', getattr(Colors, 'LIGHT', None)))
    lines.append(L('', None))
    lines.append(L('=== 🧠 HARDWARE ===', getattr(Colors, 'HEADER', None), bold=True))
    if hd.get('cpu_cores') is not None:
        lines.append(L(f"CPU Cores  : {hd.get('cpu_cores')}", getattr(Colors, 'LIGHT', None)))
    if hd.get('cpu_percent') is not None:
        lines.append(L(f"CPU Load   : {hd.get('cpu_percent')}%", getattr(Colors, 'LIGHT', None)))
    if hd.get('ram_total'):
        lines.append(L(f"RAM Total  : {hd.get('ram_total')}", getattr(Colors, 'LIGHT', None)))
    if hd.get('ram_used'):
        lines.append(L(f"RAM Used   : {hd.get('ram_used')}", getattr(Colors, 'LIGHT', None)))
    if hd.get('ram_percent') is not None:
        lines.append(L(f"RAM Load   : {hd.get('ram_percent')}%", getattr(Colors, 'LIGHT', None)))
    if hd.get('psutil_error'):
        lines.append(L(f"⚠️ psutil_error: {hd.get('psutil_error')}", getattr(Colors, 'YELLOW', None)))
    lines.append(L('', None))
    lines.append(L('=== 🎮 GPU / CUDA ===', getattr(Colors, 'HEADER', None), bold=True))
    cuda_ok = bool(gd.get('cuda_available'))
    if cuda_ok:
        name = gd.get('gpu_name')
        mem = gd.get('gpu_total_memory')
        if name and mem:
            lines.append(L(f'🟢 CUDA διαθέσιμο – GPU: {name} ({mem})', getattr(Colors, 'GREEN', None)))
        elif name:
            lines.append(L(f'🟢 CUDA διαθέσιμο – GPU: {name}', getattr(Colors, 'GREEN', None)))
        else:
            lines.append(L('🟢 CUDA διαθέσιμο', getattr(Colors, 'GREEN', None)))
    else:
        if gd.get('torch_error'):
            lines.append(L(f"❌ Torch error: {gd.get('torch_error')}", getattr(Colors, 'RED', None), bold=True))
        lines.append(L('⚠️ CUDA ΜΗ διαθέσιμο – εκτέλεση σε CPU.', getattr(Colors, 'YELLOW', None)))
    for k in ['torch', 'cuda_version', 'gpu_count', 'gpu_allocated_now', 'gpu_reserved_now', 'nvidia_smi']:
        if k in gd and gd.get(k) not in (None, '', 'unknown'):
            lines.append(L(f'{k:<16}: {gd.get(k)}', getattr(Colors, 'LIGHT', None)))
    lines.append(L('', None))
    lines.append(L('=== 📦 PACKAGES ===', getattr(Colors, 'HEADER', None), bold=True))
    pkg_order = ['PySide6', 'torch', 'torchvision', 'ultralytics', 'opencv-python', 'numpy', 'onnx']
    for k in pkg_order:
        if k in pd:
            v = pd.get(k)
            if isinstance(v, str) and 'not installed' in v:
                lines.append(L(f'⚠️ {k:<14}: {v}', getattr(Colors, 'YELLOW', None)))
            else:
                lines.append(L(f'{k:<14}: {v}', getattr(Colors, 'LIGHT', None)))
    for k in sorted([x for x in pd.keys() if x not in pkg_order]):
        lines.append(L(f'{k:<14}: {pd.get(k)}', getattr(Colors, 'LIGHT', None)))
    lines.append(L('', None))
    lines.append(L('=== ⚙️ ENVIRONMENT ===', getattr(Colors, 'HEADER', None), bold=True))
    env_order = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_PATH_V12_1', 'CUDA_PATH_V12_0', 'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV', 'ULTRALYTICS_SETTINGS_DIR', 'PATH']
    for k in env_order:
        v = ed.get(k, '')
        if not v:
            continue
        lines.append(L(f'{k:<22}: {v}', getattr(Colors, 'LIGHT', None)))
    lines.append(L('', None))
    lines.append(L('📋 PIP FREEZE (για πλήρη συμβατότητα)', getattr(Colors, 'HEADER', None), bold=True, underline=True))
    pf = str(data.get('pip_freeze', '') or '').strip()
    if not pf:
        lines.append(L('(empty)', getattr(Colors, 'LIGHT', None)))
    else:
        pf_lines = pf.splitlines()
        if for_view:
            show = pf_lines[:max(1, int(view_pip_freeze_max_lines))]
            for line in show:
                lines.append(L(line, getattr(Colors, 'LIGHT', None)))
            if len(pf_lines) > len(show):
                lines.append(L(f'… (εμφάνιση {len(show)} / {len(pf_lines)} γραμμών – το πλήρες υπάρχει στο Text/JSON/ZIP)', getattr(Colors, 'CYAN', None)))
        else:
            for line in pf_lines:
                lines.append(L(line, getattr(Colors, 'LIGHT', None)))
    lines.append(L(sep, getattr(Colors, 'HEADER', None), bold=True))
    return lines


def _safe_cmd(cmd: list[str], timeout: float=2.0) -> tuple[bool, str]:
    try:
        p = subprocess.run( cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout, shell=False,)
        out = (p.stdout or '').strip() or (p.stderr or '').strip()
        return (p.returncode == 0, out)
    except Exception as e:
        return (False, str(e))


def _safe_import_version(dist_name: str) -> str:
    try:
        from importlib import metadata
        return metadata.version(dist_name)
    except Exception:
        return '(not installed)'


def _bytes_to_gb(num_bytes: float) -> str:
    try:
        return f'{num_bytes / 1024 ** 3:.2f} GB'
    except Exception:
        return 'N/A'


def collect_diagnostics_data(app_name: str='Models Manager Pro (A.I Copilot)', app_version: str='3.1') -> dict[str, Any]:
    ts = datetime.now().isoformat(timespec='seconds')
    root_dir = Path(__file__).resolve().parent
    data: dict[str, Any] = {'timestamp': ts, 'app': {'name': app_name, 'version': app_version, 'cwd': str(Path.cwd()), 'root_dir': str(root_dir), 'python_executable': sys.executable}, 'os': {'platform': platform.platform(), 'system': platform.system(), 'release': platform.release(), 'version': platform.version(), 'machine': platform.machine(), 'processor': platform.processor()}, 'python': {'version': sys.version.replace('\n', ' '), 'implementation': platform.python_implementation()}, 'hardware': {'cpu_cores': os.cpu_count()}, 'packages': {}, 'gpu': {}, 'env': {}}
    try:
        import psutil
        vm = psutil.virtual_memory()
        data['hardware'].update({'ram_total': _bytes_to_gb(float(vm.total)), 'ram_used': _bytes_to_gb(float(vm.used)), 'ram_percent': float(vm.percent), 'cpu_percent': float(psutil.cpu_percent(interval=0.2))})
    except Exception as e:
        data['hardware']['psutil_error'] = str(e)
    pkgs = {'PySide6': 'PySide6', 'torch': 'torch', 'torchvision': 'torchvision', 'ultralytics': 'ultralytics', 'opencv-python': 'opencv-python', 'numpy': 'numpy', 'onnx': 'onnx'}
    data['packages'] = {label: _safe_import_version(dist) for label, dist in pkgs.items()}
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        gpu: dict[str, Any] = {'torch': getattr(torch, '__version__', 'unknown'), 'cuda_available': cuda_ok, 'cuda_version': getattr(getattr(torch, 'version', object()), 'cuda', None)}
        if cuda_ok:
            gpu.update({'gpu_count': int(torch.cuda.device_count()), 'gpu_name': torch.cuda.get_device_name(0)})
            try:
                props = torch.cuda.get_device_properties(0)
                gpu['gpu_total_memory'] = _bytes_to_gb(float(props.total_memory))
                gpu['gpu_allocated_now'] = _bytes_to_gb(float(torch.cuda.memory_allocated(0)))
                gpu['gpu_reserved_now'] = _bytes_to_gb(float(torch.cuda.memory_reserved(0)))
            except Exception:
                pass
        data['gpu'] = gpu
    except Exception as e:
        data['gpu']['torch_error'] = str(e)
    ok, out = _safe_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], timeout=2.0)
    if ok:
        data['gpu']['nvidia_smi'] = out
    env_keys = ['PATH', 'CUDA_PATH', 'CUDA_HOME', 'CUDA_PATH_V12_0', 'CUDA_PATH_V12_1', 'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV', 'ULTRALYTICS_SETTINGS_DIR']
    env_map: dict[str, str] = {}
    for k in env_keys:
        v = os.environ.get(k, '')
        if k == 'PATH' and len(v) > 1500:
            env_map[k] = v[:1500] + ' ... (truncated)'
        else:
            env_map[k] = v
    data['env'] = env_map
    ok, out = _safe_cmd([sys.executable, '-m', 'pip', 'freeze'], timeout=6.0)
    data['pip_freeze'] = out if ok else f'(failed) {out}'
    return data


def diagnostics_to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def diagnostics_to_text(data: dict[str, Any]) -> str:
    out_lines: list[str] = []
    for text, _color, _bold, _underline in _build_diagnostics_log_lines(data, for_view=False):
        out_lines.append(text)
    return '\n'.join(out_lines)


def diagnostics_to_html(data: dict[str, Any]) -> str:
    try:
        html_lines: list[str] = []
        for text, color, bold, underline in _build_diagnostics_log_lines(data, for_view=True):
            html_lines.append(format_html_log(text, color, bold=bold, underline=underline))
        body = '<br>'.join(html_lines)
        return f"<html><head><meta charset='utf-8'></head><body style='margin:0; padding:0; background: transparent;'>{body}</body></html>"
    except Exception:
        safe = (diagnostics_to_text(data) or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f"""<html><head><meta charset='utf-8'></head><body><pre style="font-family: Consolas, 'Courier New', monospace; font-size: 11.5px;">{safe}</pre></body></html>"""
"""Models Manager Pro – core utilities.
Κεντρικό module με κοινές βοηθητικές συναρτήσεις/σταθερές που χρησιμοποιούνται; από όλα τα tabs/workers (training/export/benchmark/live camera).
Στόχοι:
- Ενιαία συμπεριφορά (logging, paths, safety wrappers).; - Σταθερότητα σε Windows (OpenCV/COM, camera backends).; - Ασφαλές wrapper για YOLO predict με auto-recover σε fixed input sizes.
"""
import torch
try:
    import psutil
except Exception:
    psutil = None
try:
    import cv2
except Exception:
    cv2 = None
_BADGE_STYLE_QSS = ("QLabel{background-color:rgba(0,0,0,.08);border:1px solid rgba(0,0,0,.14);color:#1a1a1a;border-radius:12px;padding:4px 10px;font-weight:600;}")
IS_WINDOWS: bool = os.name == "nt"
_MM_PRO_CAMERA_LOCK: threading.Lock = threading.Lock()
_MM_PRO_CAMERA_LOCK_OWNER: str | None = None


def acquire_camera_lock(owner: str, timeout_sec: float = 0.0) -> bool:
    global _MM_PRO_CAMERA_LOCK_OWNER
    try:
        if timeout_sec > 0:
            ok = _MM_PRO_CAMERA_LOCK.acquire(timeout=float(timeout_sec))
        else:
            ok = _MM_PRO_CAMERA_LOCK.acquire(blocking=False)
        if ok:
            _MM_PRO_CAMERA_LOCK_OWNER = str(owner or "unknown")
        return bool(ok)
    except Exception:
        return True


def release_camera_lock() -> None:
    global _MM_PRO_CAMERA_LOCK_OWNER
    try:
        if _MM_PRO_CAMERA_LOCK.locked():
            _MM_PRO_CAMERA_LOCK_OWNER = None
            _MM_PRO_CAMERA_LOCK.release()
    except Exception:
        pass


def get_camera_lock_owner() -> str:
    try:
        return str(_MM_PRO_CAMERA_LOCK_OWNER or '')
    except Exception:
        return ''
_OPENCV_RUNTIME_CONFIGURED = False
_COM_INITIALIZED_THREADS: set[int] = set()
_MF_STARTED = False


def configure_opencv_runtime() -> None:
    global _OPENCV_RUNTIME_CONFIGURED
    if _OPENCV_RUNTIME_CONFIGURED:
        return
    if cv2 is None:
        return
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        if hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    _OPENCV_RUNTIME_CONFIGURED = True
_CUDA_WARMED_UP = False


def warmup_torch_cuda(context: str = "warmup") -> None:
    global _CUDA_WARMED_UP
    if _CUDA_WARMED_UP:
        return
    if _env_bool("MM_PRO_DISABLE_CUDA_WARMUP", False):
        _CUDA_WARMED_UP = True
    try:
        _ = torch.cuda.is_available()
    except Exception:
        _CUDA_WARMED_UP = True
    if torch.cuda.is_available():
        try:
            x = torch.empty((1,), device="cuda")
            torch.cuda.synchronize()
            del x
        except Exception as e:
            safe_log_once( "cuda_warmup", f"⚠️ CUDA warmup failed ({context})", e, group="CUDA", level=logging.WARNING,)
    _CUDA_WARMED_UP = True
_CUDA_THREAD_READY: set[int] = set()
_CUDA_THREAD_LOCK: threading.Lock = threading.Lock()


def ensure_cuda_ready_for_thread(context: str = "thread") -> None:
    if not torch.cuda.is_available():
        return
    tid = threading.get_ident()
    with _CUDA_THREAD_LOCK:
        if tid in _CUDA_THREAD_READY:
            return
    try:
        _ = torch.cuda.current_device()
        x = torch.empty((1,), device="cuda")
        torch.cuda.synchronize()
        del x
    except Exception as e:
        _MMPRO_LOGGER.debug("ensure_cuda_ready_for_thread(%s) failed: %s", context, e)
    finally:
        with _CUDA_THREAD_LOCK:
            _CUDA_THREAD_READY.add(tid)


def ensure_windows_com_initialized() -> None:
    if not IS_WINDOWS:
        return
    try:
        import threading
        tid = int(threading.get_ident())
        if tid in _COM_INITIALIZED_THREADS:
            return
    except Exception:
        tid = None
    try:
        import ctypes
        COINIT_APARTMENTTHREADED = 0x2
        RPC_E_CHANGED_MODE = 0x80010106
        hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        if int(hr) == int(RPC_E_CHANGED_MODE):
            pass
    except Exception:
        pass
    finally:
        try:
            if tid is not None:
                _COM_INITIALIZED_THREADS.add(tid)
        except Exception:
            pass


def ensure_windows_media_foundation_started() -> None:
    global _MF_STARTED
    if not IS_WINDOWS:
        return
    if _MF_STARTED:
        return
    try:
        import ctypes
        MF_VERSION = 0x00020070
        mfplat = ctypes.windll.mfplat
        hr = mfplat.MFStartup(ctypes.c_uint(MF_VERSION), ctypes.c_uint(0))
        if int(hr) >= 0:
            _MF_STARTED = True
    except Exception:
        _MF_STARTED = True
MMPRO_CACHE_DIR_NAME = '_mmpro_cache'  # Όνομα υποφακέλου cache της εφαρμογής
TRT_CACHE_ENABLED = False  # Ενεργοποίηση/απενεργοποίηση TensorRT cache


def _env_get(name, default, cast=None):
    try:
        v = os.environ.get(name, "").strip()
        if cast is bool:
            return v.lower() in ("1","true","yes","on")
        return cast(v) if cast else (v if os.environ.get(name) is not None else default)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool: return _env_get(name, default, bool)


def _env_int(name: str, default: int) -> int: return _env_get(name, default, int)


def _env_float(name: str, default: float) -> float: return _env_get(name, default, float)


def _env_str(name: str, default: str = "") -> str: return _env_get(name, default)


def ensure_dir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _json_read(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
    except Exception:
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _json_write(path: Path, data: dict) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _MMPRO_LOGGER.warning("_json_write mkdir failed: %s", e)
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = Path(str(path) + ".tmp")
    try:
        tmp.write_text(json_str, encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        _MMPRO_LOGGER.debug("_json_write atomic write failed, falling back: %s", e)
        try:
            path.write_text(json_str, encoding="utf-8")
        except Exception as e2:
            _MMPRO_LOGGER.warning("_json_write fallback also failed: %s", e2)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def json_write(path: Path, data: dict) -> None:
    _json_write(path, data)


def _file_fingerprint(path: Path, chunk_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    try:
        st = path.stat()
        size = st.st_size
        mtime_ns: int = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    except Exception:
        size = 0
        mtime_ns = 0
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            head = f.read(chunk_bytes)
            if head:
                h.update(head)
            if size > chunk_bytes:
                try:
                    f.seek(max(0, size - chunk_bytes))
                    tail = f.read(chunk_bytes)
                    if tail:
                        h.update(tail)
                except Exception:
                    pass
    except Exception:
        pass
    return {"size": size, "mtime_ns": mtime_ns, "sha256_approx": h.hexdigest()}


def _env_versions() -> dict[str, Any]:

    def _safe_version(fn: Callable[[], Any], default: Any = "") -> Any:
        try:
            return fn()
        except Exception:
            return default
    cuda_available = _safe_version(lambda: bool(torch.cuda.is_available()), False)
    v: dict[str, Any] = {
        "python":       _safe_version(lambda: sys.version.split()[0]),
        "torch":        _safe_version(lambda: getattr(torch, "__version__", "")),
        "cuda":         _safe_version(lambda: getattr(getattr(torch, "version", None), "cuda", "") or ""),
        "cuda_available": cuda_available,
        "gpu":          _safe_version(lambda: torch.cuda.get_device_name(0) if cuda_available else ""),
        "gpu_cc":       _safe_version(lambda: "{}.{}".format(*torch.cuda.get_device_capability(0)) if cuda_available else ""),
    }
    for pkg, attr in (("ultralytics", "__version__"), ("tensorrt", "__version__")):
        try:
            mod = __import__(pkg)
            v[pkg] = getattr(mod, attr, "")
        except Exception:
            v[pkg] = ""
    return v


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 9 – TensorRT cache management & signatures
# ═══════════════════════════════════════════════════════════════════════

# ── Δημιουργία μοναδικού fingerprint (signature) για TensorRT engine ─────────
# Συνδυάζει: hash αρχείου μοντέλου, παραμέτρους inference (imgsz/half/batch/...),
# εκδόσεις (TRT/GPU/CUDA). Χρησιμοποιείται για αξιόπιστο cache invalidation.
def trt_signature(model_path: Path, imgsz: int, half: bool = True, batch: int = 1, dynamic: bool = False, workspace: int = 4, device: str = 'cuda:0') -> dict:
    """
    Δημιουργεί μοναδικό fingerprint για TensorRT engine.

    Περιλαμβάνει: model hash, inference params (imgsz/half/batch/...),
    versions (TRT/GPU/CUDA) για αξιόπιστο cache invalidation.
    """
    fp = _file_fingerprint(model_path)
    sig = {
        'kind': 'tensorrt_engine',
        'model': str(Path(model_path).resolve()),
        'model_fingerprint': fp,
        'params': { 'imgsz': int(imgsz), 'half': bool(half), 'batch': int(batch), 'dynamic': bool(dynamic), 'workspace': int(workspace), 'device': str(device),},
        'versions': _env_versions(),
    }
    try:
        sig['signature_id'] = hashlib.sha256(json.dumps(sig, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()[:16]
    except Exception:
        sig['signature_id'] = ''
    return sig


def trt_signature_path_for_engine(engine_path: Path) -> Path:
    return Path(str(engine_path) + '.mmpro.json')


def trt_cache_dir_for_model(model_path: Path) -> Path:
    base = Path(model_path).resolve().parent
    if not TRT_CACHE_ENABLED:
        return base
    return ensure_dir(base / MMPRO_CACHE_DIR_NAME / 'tensorrt')


def trt_cached_engine_paths(model_path: Path) -> list[Path]:
    if not TRT_CACHE_ENABLED:
        return []
    cache_dir = trt_cache_dir_for_model(model_path)
    try:
        return sorted(cache_dir.glob('*.engine'), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []


def trt_signature_matches(a: dict, b: dict) -> bool:
    try:
        if not a or not b:
            return False
        if (a.get('kind') or '') != (b.get('kind') or ''):
            return False
        af = (a.get('model_fingerprint') or {})
        bf = (b.get('model_fingerprint') or {})
        for k in ('size', 'mtime_ns', 'sha256_approx'):
            if str(af.get(k)) != str(bf.get(k)):
                return False
        ap = (a.get('params') or {})
        bp = (b.get('params') or {})
        keys = ('imgsz', 'half', 'batch', 'dynamic', 'workspace', 'device')
        for k in keys:
            if str(ap.get(k)) != str(bp.get(k)):
                return False
        av = (a.get('versions') or {})
        bv = (b.get('versions') or {})
        for k in ('tensorrt', 'gpu', 'gpu_cc'):
            if str(av.get(k) or '') != str(bv.get(k) or ''):
                return False
        return True
    except Exception:
        return False


def trt_engine_is_up_to_date(model_path: Path, engine_path: Path, imgsz: int, half: bool = True, batch: int = 1, dynamic: bool = False, workspace: int = 4, device: str = 'cuda:0') -> bool:
    """Ελέγχει αν υπάρχον TensorRT engine είναι ενημερωμένο βάσει signature."""
    try:
        if not Path(engine_path).exists():
            return False
    except Exception:
        return False
    expected = trt_signature(model_path, imgsz, half=half, batch=batch, dynamic=dynamic, workspace=workspace, device=device)
    sig_path = trt_signature_path_for_engine(engine_path)
    existing = _json_read(sig_path)
    if trt_signature_matches(existing, expected):
        return True
    if not existing:
        try:
            _preflight_tensorrt_engine(Path(engine_path))
            _json_write(sig_path, expected)
            return True
        except Exception:
            return False
    return False


def trt_try_restore_from_cache(model_path: Path, engine_path: Path, imgsz: int, half: bool = True, batch: int = 1, dynamic: bool = False, workspace: int = 4, device: str = 'cuda:0') -> bool:
    """Αναζητά και επαναφέρει cached TensorRT engine αν ταιριάζει το signature."""
    if not TRT_CACHE_ENABLED:
        return False
    expected = trt_signature(model_path, imgsz, half=half, batch=batch, dynamic=dynamic, workspace=workspace, device=device)
    cache_dir = trt_cache_dir_for_model(model_path)
    candidates = trt_cached_engine_paths(model_path)
    for eng in candidates:
        sig = _json_read(trt_signature_path_for_engine(eng))
        if trt_signature_matches(sig, expected):
            try:
                ensure_dir(Path(engine_path).parent)
                shutil.copy2(str(eng), str(engine_path))
                _json_write(trt_signature_path_for_engine(engine_path), expected)
                return True
            except Exception:
                continue
    return False


def trt_store_to_cache(model_path: Path, engine_path: Path, imgsz: int, half: bool = True, batch: int = 1, dynamic: bool = False, workspace: int = 4, device: str = 'cuda:0') -> Path | None:
    if not TRT_CACHE_ENABLED:
        return None
    try:
        if not Path(engine_path).exists():
            return None
    except Exception:
        return None
    sig = trt_signature(model_path, imgsz, half=half, batch=batch, dynamic=dynamic, workspace=workspace, device=device)
    cache_dir = trt_cache_dir_for_model(model_path)
    sid = str(sig.get('signature_id') or '')
    name = f"{Path(model_path).stem}__{sid}__imgsz{int(imgsz)}__bs{int(batch)}__{'fp16' if half else 'fp32'}{'__dyn' if dynamic else ''}.engine"
    dst = cache_dir / name
    try:
        if dst.exists() and dst.stat().st_size == Path(engine_path).stat().st_size:
            _json_write(trt_signature_path_for_engine(dst), sig)
            return dst
    except Exception:
        pass
    try:
        shutil.copy2(str(engine_path), str(dst))
        _json_write(trt_signature_path_for_engine(dst), sig)
        return dst
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 10 – OpenCV runtime configuration & camera utilities
# ═══════════════════════════════════════════════════════════════════════

def _configure_cv2_runtime() -> None:
    if cv2 is None:
        return
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
_configure_cv2_runtime()


def _tune_video_capture(cap) -> None:
    if cv2 is None or cap is None:
        return
    if _env_bool('MM_PRO_NO_CAM_TUNE', False):
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        fps_env = _env_str('MM_PRO_CAM_FPS', '')
        fps_max = _env_float('MM_PRO_CAM_FPS_MAX', 120.0)
        fps_req = float(fps_env) if fps_env else fps_max
        if fps_req > 0 and hasattr(cv2, 'CAP_PROP_FPS'):
            cap.set(cv2.CAP_PROP_FPS, fps_req)
    except Exception:
        pass
    if _env_bool('MM_PRO_CAM_MJPG', False) or _env_bool('MM_PRO_CAM_FORCE_MJPG', False):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
_MM_PRO_CAM_RES_CACHE: dict[tuple[int, int], tuple[int, int]] = {}


def _mmpro_parse_resolution(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    s = str(text).strip().lower().replace('×', 'x')
    for sep in ('x', ',', ' '):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    w = int(float(parts[0]))
                    h = int(float(parts[1]))
                    if w > 0 and h > 0:
                        return w, h
                except Exception:
                    return None


def _mmpro_get_capture_res(cap) -> tuple[int, int]:
    if cv2 is None or cap is None:
        return 0, 0
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    except Exception:
        w = 0
    try:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:
        h = 0
    return w, h


def _mmpro_set_capture_res(cap, w: int, h: int) -> tuple[int, int]:
    if cv2 is None or cap is None:
        return 0, 0
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    except Exception:
        pass
    try:
        cap.grab()
        cap.grab()
    except Exception:
        pass
    return _mmpro_get_capture_res(cap)


def _mmpro_try_resolution(cap, w: int, h: int) -> bool:
    if cap is None:
        return False
    aw, ah = _mmpro_set_capture_res(cap, w, h)
    if int(aw) != int(w) or int(ah) != int(h):
        return False
    good = 0
    needed = 3
    tries = 10
    try:
        for _ in range(int(tries)):
            try:
                ok = cap.grab()
            except Exception:
                ok = False
            if not ok:
                continue
            try:
                ret, frame = cap.retrieve()
            except Exception:
                ret, frame = False, None
            if (not ret) or (frame is None) or (getattr(frame, 'size', 0) == 0):
                good = 0
                continue
            try:
                fh, fw = frame.shape[:2]
            except Exception:
                good = 0
                continue
            if int(fw) == int(w) and int(fh) == int(h):
                good += 1
                if good >= int(needed):
                    return True
            else:
                good = 0
        return False
    except Exception:
        return False


def _mmpro_apply_best_camera_resolution(cap, camera_index: int, backend_id: int) -> tuple[int, int]:
    if cv2 is None or cap is None:
        return 0, 0
    auto_on = _env_bool('MM_PRO_CAM_RES_AUTO', True)
    if not auto_on:
        return _mmpro_get_capture_res(cap)
    key = (int(camera_index), int(backend_id or 0))
    if key in _MM_PRO_CAM_RES_CACHE:
        cw, ch = _MM_PRO_CAM_RES_CACHE[key]
        try:
            _mmpro_set_capture_res(cap, int(cw), int(ch))
        except Exception:
            pass
        return _mmpro_get_capture_res(cap)
    forced = None
    try:
        forced = _mmpro_parse_resolution(str(os.environ.get('MM_PRO_CAM_RES', '')).strip())
    except Exception:
        forced = None
    if forced is not None:
        fw, fh = forced
        try:
            _mmpro_set_capture_res(cap, int(fw), int(fh))
        except Exception:
            pass
        aw, ah = _mmpro_get_capture_res(cap)
        _MM_PRO_CAM_RES_CACHE[key] = (int(aw), int(ah))
        return aw, ah
    prefer = None
    try:
        prefer_raw = str(os.environ.get('MM_PRO_CAM_PREFER_RES', '1920x1080')).strip()
        if prefer_raw and prefer_raw.lower() not in ('auto', 'none', 'off', '0'):
            prefer = _mmpro_parse_resolution(prefer_raw)
    except Exception:
        prefer = None
    if prefer is not None:
        try:
            pw, ph = int(prefer[0]), int(prefer[1])
            if pw > 0 and ph > 0:
                if _mmpro_try_resolution(cap, pw, ph):
                    _MM_PRO_CAM_RES_CACHE[key] = (int(pw), int(ph))
                    return _mmpro_get_capture_res(cap)
        except Exception:
            pass
    prefer_aspect = None
    try:
        prefer_aspect_raw = str(os.environ.get('MM_PRO_CAM_PREFER_ASPECT', '16:9')).strip().lower()
        if prefer_aspect_raw and prefer_aspect_raw not in ('auto', 'none', 'off', '0'):
            if ':' in prefer_aspect_raw:
                a, b = prefer_aspect_raw.split(':', 1)
                prefer_aspect = float(a) / float(b)
            else:
                prefer_aspect = float(prefer_aspect_raw)
    except Exception:
        prefer_aspect = (16.0 / 9.0)
    try:
        aspect_tol = float(os.environ.get('MM_PRO_CAM_ASPECT_TOL', '0.03'))
    except Exception:
        aspect_tol = 0.03
    aspect_tol = max(0.005, min(0.20, aspect_tol))
    candidates: list[tuple[int, int]] = [
        (3840, 2160), (2560, 1440), (1920, 1080), (1600, 900), (1366, 768), (1280, 720),
        (1024, 576), (960, 540), (854, 480), (640, 360),
        (2592, 1944), (2048, 1536), (1600, 1200), (1280, 960), (1024, 768),
        (800, 600), (640, 480),
    ]
    seen = set()
    uniq: list[tuple[int, int]] = []
    for w, h in candidates:
        if (w, h) not in seen:
            uniq.append((w, h))
            seen.add((w, h))
    uniq.sort(key=lambda x: int(x[0]) * int(x[1]), reverse=True)
    if prefer_aspect is not None:
        pref_list: list[tuple[int, int]] = []
        other_list: list[tuple[int, int]] = []
        for w, h in uniq:
            try:
                ar = float(w) / float(h)
            except Exception:
                ar = 0.0
            if ar and abs(ar - float(prefer_aspect)) <= aspect_tol:
                pref_list.append((w, h))
            else:
                other_list.append((w, h))
        ordered = pref_list + other_list
    else:
        ordered = uniq
    orig_w, orig_h = _mmpro_get_capture_res(cap)
    chosen = (orig_w, orig_h)
    for w, h in ordered:
        if _mmpro_try_resolution(cap, int(w), int(h)):
            chosen = (int(w), int(h))
            break
    if chosen == (orig_w, orig_h):
        try:
            _mmpro_set_capture_res(cap, int(orig_w), int(orig_h))
        except Exception:
            pass
    _MM_PRO_CAM_RES_CACHE[key] = (int(chosen[0] or 0), int(chosen[1] or 0))
    return _mmpro_get_capture_res(cap)


def apply_best_camera_resolution(cap, camera_index: int = 0, backend_id: int = 0) -> tuple[int, int]:
    """Εφαρμόζει την καλύτερη διαθέσιμη ανάλυση για το capture device."""
    try:
        return _mmpro_apply_best_camera_resolution(cap, int(camera_index), int(backend_id or 0))
    except Exception:
        return _mmpro_get_capture_res(cap)


def force_capture_resolution(cap, w: int = 1920, h: int = 1080, verify: bool = True) -> bool:
    if cv2 is None or cap is None:
        return False
    try:
        _mmpro_set_capture_res(cap, int(w), int(h))
        if not verify:
            return True
        good = 0
        needed = 3
        tries = 10
        for _ in range(int(tries)):
            try:
                ok = cap.grab()
            except Exception:
                ok = False
            if not ok:
                continue
            try:
                ret, frame = cap.retrieve()
            except Exception:
                ret, frame = False, None
            if (not ret) or (frame is None) or (getattr(frame, 'size', 0) == 0):
                good = 0
                continue
            try:
                fh, fw = frame.shape[:2]
            except Exception:
                good = 0
                continue
            if int(fw) == int(w) and int(fh) == int(h):
                good += 1
                if good >= int(needed):
                    return True
            else:
                good = 0
        return False
    except Exception:
        return False


def _validate_capture(cap, warmup_reads: int = 3) -> bool:
    if cap is None:
        return False
    try:
        for _ in range(max(1, int(warmup_reads))):
            try:
                ok = cap.grab()
            except Exception:
                ok = False
            if ok:
                try:
                    ret, frame = cap.retrieve()
                except Exception:
                    ret, frame = False, None
                try:
                    if ret and frame is not None and getattr(frame, "size", 0) > 0:
                        return True
                except Exception:
                    if ret and frame is not None:
                        return True
            try:
                time.sleep(0.03)
            except Exception:
                pass
    except Exception:
        pass
    return False


def open_video_capture(index: int = 0, allow_msmf: bool | None = None, forbid_dshow: bool | None = None) -> Any | None:
    """
    Ανοίγει VideoCapture επιλέγοντας αυτόματα το καλύτερο backend.

    Windows: δοκιμάζει DSHOW > MSMF > ANY σύμφωνα με env variables.
    Non-Windows: χρησιμοποιεί default backend.
    """
    if cv2 is None:
        raise ModuleNotFoundError('Λείπει το OpenCV (cv2). Εγκατάστησέ το με: pip install opencv-python')
    configure_opencv_runtime()
    ensure_windows_com_initialized()
    ensure_windows_media_foundation_started()
    default_backend = 'dshow' if IS_WINDOWS else 'auto'
    backend_pref = _env_str('MM_PRO_CAM_BACKEND', default_backend).lower()
    if allow_msmf is None:
        allow_msmf = _env_bool('MM_PRO_ALLOW_MSMF', False)
    if forbid_dshow is None:
        if 'MM_PRO_ALLOW_DSHOW' in os.environ:
            forbid_dshow = not _env_bool('MM_PRO_ALLOW_DSHOW', True)
        else:
            forbid_dshow = _env_bool('MM_PRO_FORBID_DSHOW', False)
    tried: list[tuple] = []

    def _backend_info(cap) -> tuple[str, int]:
        name = ''
        bid = 0
        try:
            if hasattr(cap, 'getBackendName'):
                name = str(cap.getBackendName() or '')
        except Exception:
            name = ''
        try:
            if hasattr(cv2, 'CAP_PROP_BACKEND'):
                bid = int(cap.get(int(cv2.CAP_PROP_BACKEND)) or 0)
        except Exception:
            bid = 0
        return name, bid

    def _is_dshow(name: str, bid: int) -> bool:
        try:
            if bid and hasattr(cv2, 'CAP_DSHOW') and int(bid) == int(cv2.CAP_DSHOW):
                return True
        except Exception:
            pass
        return 'DSHOW' in (name or '').upper()

    def _open(cam_index: int, api: int):
        cap = None
        try:
            cap = cv2.VideoCapture(cam_index, api) if api else cv2.VideoCapture(cam_index)
            if cap is None or (not cap.isOpened()):
                tried.append((cam_index, int(api or 0), '', 0, 'OPEN_FAIL'))
            else:
                _tune_video_capture(cap)
                if not _validate_capture(cap, warmup_reads=5):
                    name, bid = _backend_info(cap)
                    tried.append((cam_index, int(api or 0), name, int(bid or 0), 'NO_FRAME'))
                else:
                    name, bid = _backend_info(cap)
                    if forbid_dshow and _is_dshow(name, bid):
                        tried.append((cam_index, int(api or 0), name or 'DSHOW', int(bid or 0), 'DSHOW(REJECT)'))
                    else:
                        try:
                            _mmpro_apply_best_camera_resolution(cap, int(cam_index), int(bid or (api or 0)))
                        except Exception:
                            pass
                        tried.append((cam_index, int(api or 0), name, int(bid or 0), 'OK'))
                        return cap, int(bid or (api or 0))
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None, 0
    if IS_WINDOWS:
        apis: list[int] = []
        if backend_pref in ('msmf', 'cap_msmf'):
            if hasattr(cv2, 'CAP_MSMF') and allow_msmf:
                apis = [int(cv2.CAP_MSMF)]
        elif backend_pref in ('dshow', 'directshow', 'cap_dshow'):
            if hasattr(cv2, 'CAP_DSHOW'):
                apis = [int(cv2.CAP_DSHOW)]
        elif backend_pref in ('any', 'cap_any'):
            apis = [0]
        else:
            if (not forbid_dshow) and hasattr(cv2, 'CAP_DSHOW'):
                apis.append(int(cv2.CAP_DSHOW))
            if allow_msmf and hasattr(cv2, 'CAP_MSMF'):
                apis.append(int(cv2.CAP_MSMF))
            try:
                allow_any = str(os.environ.get('MM_PRO_CAM_ALLOW_ANY_FALLBACK', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                allow_any = False
            if allow_any:
                apis.append(0)
            if not apis:
                apis = [0]
        indices = [int(index)]
        try:
            try_neg1 = str(os.environ.get('MM_PRO_CAM_TRY_NEG1', '0')).strip().lower() in ('1','true','yes','on')
        except Exception:
            try_neg1 = False
        if try_neg1 and int(index) == 0:
            indices.append(-1)
        for cam_index in indices:
            for api in apis:
                cap, actual = _open(cam_index, api)
                if cap is not None:
                    return cap, int(actual)
        raise RuntimeError(f'❌ Δεν ήταν δυνατό το άνοιγμα της κάμερας με MSMF/ANY' + ('' if not forbid_dshow else ' (χωρίς DSHOW)') + f'. (Δοκιμές: {tried})')
    cap, _api = _open(int(index), 0)
    if cap is None:
        raise RuntimeError('❌ Δεν άνοιξε κάμερα.')
    return cap, int(_api or 0)
import numpy as np
import matplotlib
matplotlib.use('Agg')
try:
    if getattr(sys, 'frozen', False):
        freeze_support()
except Exception:
    pass
try:
    import faulthandler as _faulthandler
    _crash_dir = None
    try:
        _base_runtime_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
        _crash_dir = _base_runtime_dir / 'Crash_Logs'
        _crash_dir.mkdir(parents=True, exist_ok=True)
        _fh_path = _crash_dir / 'faulthandler.log'
        _fh_file = open(_fh_path, 'a', encoding='utf-8')
        _faulthandler.enable(file=_fh_file, all_threads=True)
    except Exception:
        pass
except Exception:
    pass


def _mmpro_setup_subprocess_no_console_windows() -> None:
    if os.name != 'nt':
        return
    import subprocess as _sp
    if getattr(_sp, '_mmpro_no_console_patched', False):
        return
    _orig_popen = _sp.Popen

    def _popen_no_console(*args, **kwargs):
        flags = kwargs.get('creationflags', 0)
        try:
            flags |= _sp.CREATE_NO_WINDOW
        except AttributeError:
            pass
        kwargs['creationflags'] = flags
        startupinfo = kwargs.get('startupinfo')
        if startupinfo is None:
            startupinfo = _sp.STARTUPINFO()
        startupinfo.dwFlags |= _sp.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs['startupinfo'] = startupinfo
        return _orig_popen(*args, **kwargs)
    _sp.Popen = _popen_no_console
    _sp._mmpro_no_console_patched = True
_mmpro_setup_subprocess_no_console_windows()


class _NullWriter:

    def write(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        raise OSError("_NullWriter: no file descriptor")
if getattr(sys, 'frozen', False):
    if getattr(sys, 'stdout', None) is None:
        sys.stdout = _NullWriter()
    if getattr(sys, 'stderr', None) is None:
        sys.stderr = _NullWriter()
try:
    from ultralytics import YOLO as _YOLO, settings as _ultra_settings
    YOLO = _YOLO
    settings = _ultra_settings
    _ULTRALYTICS_IMPORT_ERROR = None
except Exception as _e:
    _ULTRALYTICS_IMPORT_ERROR = _e

    def YOLO(*args, **kwargs):
        raise ModuleNotFoundError("Λείπει το 'ultralytics'. Εγκατάστησέ το με: pip install ultralytics") from _ULTRALYTICS_IMPORT_ERROR

    class _DummySettings(dict):

        def reset(self):
            return None

        def update(self, *a, **k):
            return None
    settings = _DummySettings()
try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
    _OPENAI_IMPORT_ERROR = None
except Exception as _e:
    _OPENAI_IMPORT_ERROR = _e

    class OpenAI:

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("Λείπει το 'openai'. Εγκατάστησέ το με: pip install openai") from _OPENAI_IMPORT_ERROR
try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit, QTabWidget, QGroupBox, QCheckBox, QLineEdit, QMessageBox, QGridLayout, QFrame, QSizePolicy, QProgressBar, QScrollArea, QFormLayout, QPlainTextEdit, QTableWidget, QTableWidgetItem, QSlider, QAbstractSpinBox, QDoubleSpinBox, QHeaderView, QAbstractItemView
    from PySide6.QtCore import QObject, QThread, Signal, Qt, QSize, QTimer, QMutex, Slot, QUrl
    from PySide6 import __version__ as PYSIDE_VERSION
    from PySide6.QtGui import QPalette, QColor, QFont, QImage, QPixmap, QIcon, QPainter, QPen, QBrush, QTextCursor, QDesktopServices, QTextOption
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError('Λείπει το PySide6 (GUI). Εγκατάστησέ το με: pip install PySide6') from _e


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 11 – PySide6 custom widgets
# ═══════════════════════════════════════════════════════════════════════

class RoundedMaskLabel(QLabel):
    """QLabel με στρογγυλεμένες γωνίες μέσω QPainterPath mask."""
    __slots__ = ("_radius",)

    def __init__(self, *a, radius=18, **k):
        super().__init__(*a, **k)
        try:
            self._radius = int(radius)
        except Exception:
            self._radius = 18
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAutoFillBackground(True)
        except Exception:
            pass
        try:
            self._update_mask()
        except Exception:
            pass

    def setRadius(self, r):
        try:
            self._radius = int(r)
        except Exception:
            self._radius = 18
        self._update_mask()

    def radius(self):
        return int(getattr(self, "_radius", 18))

    def resizeEvent(self, e):
        try:
            super().resizeEvent(e)
        except Exception:
            pass
        self._update_mask()

    def _update_mask(self):
        try:
            r = max(0, int(getattr(self, "_radius", 18)))
        except Exception:
            r = 18
        if r <= 0:
            try:
                self.clearMask()
            except Exception:
                pass
        try:
            from PySide6.QtGui import QPainterPath, QRegion
        except Exception:
            try:
                from PySide6.QtGui import QPainterPath, QRegion
            except Exception:
                return
        try:
            path = QPainterPath()
            path.addRoundedRect(self.rect(), r, r)
            self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        except Exception:
            pass


class AspectRatioFrame(QWidget):
    """QWidget που διατηρεί σταθερό aspect ratio στο child widget."""
    __slots__ = ("_aw", "_ah", "_child")

    def __init__(self, child: QWidget, aw: int = 16, ah: int = 9, parent=None):
        super().__init__(parent)
        self._child = child
        try:
            self._aw = max(1, int(aw))
            self._ah = max(1, int(ah))
        except Exception:
            self._aw, self._ah = 16, 9
        try:
            self._child.setParent(self)
            self._child.show()
        except Exception:
            pass
        try:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        except Exception:
            pass

    def set_aspect(self, aw: int, ah: int) -> None:
        try:
            self._aw = max(1, int(aw))
            self._ah = max(1, int(ah))
        except Exception:
            self._aw, self._ah = 16, 9
        self._apply_geometry()

    def resizeEvent(self, e):
        try:
            super().resizeEvent(e)
        except Exception:
            pass
        self._apply_geometry()

    def _apply_geometry(self) -> None:
        try:
            child = getattr(self, "_child", None)
            if child is None:
                return
            W = int(self.width())
            H = int(self.height())
            if W <= 2 or H <= 2:
                return
            aw = int(getattr(self, "_aw", 16))
            ah = int(getattr(self, "_ah", 9))
            target_h = int(round(W * (ah / aw)))
            if target_h <= H:
                new_w = W
                new_h = target_h
            else:
                new_h = H
                new_w = int(round(H * (aw / ah)))
            x = int((W - new_w) / 2)
            y = int((H - new_h) / 2)
            child.setGeometry(x, y, new_w, new_h)
        except Exception:
            pass
warnings.filterwarnings('ignore', message="Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*", category=UserWarning)
warnings.filterwarnings('ignore', message='record_context_cpp is not support on non-linux non-x86_64 platforms.*', category=UserWarning)
warnings.filterwarnings('ignore', message='Enable tracemalloc to get the object allocation traceback', category=UserWarning)
warnings.filterwarnings('ignore', message='Unable to automatically guess model task, assuming.*', category=UserWarning)
warnings.filterwarnings('once', category=UserWarning, module='torch\\._inductor\\.runtime\\.triton_helpers')
warnings.filterwarnings('once', category=UserWarning, module='torch\\._dynamo\\.convert_frame')
logging.getLogger('torch._inductor').setLevel(logging.ERROR)
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)


def setup_triton_windows_compat(logger=None) -> None:
    try:
        base_cache = Path.home() / '.cache' / 'torchinductor'
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(base_cache))
        os.environ.setdefault('TRITON_CACHE_DIR', str(base_cache / 'triton'))
        if platform.system() == 'Windows':
            try:
                import triton
                if logger:
                    logger.info('ℹ️ Triton (Windows build) βρέθηκε στο περιβάλλον.')
            except Exception:
                if logger:
                    logger.warning('⚠️ Triton δεν βρέθηκε σε αυτό το περιβάλλον Windows. Θα προβώ σε ασφαλή fallback ρυθμίσεις.')
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        try:
            if hasattr(torch, '_inductor'):
                try:
                    torch._inductor.config.triton.cudagraphs = False
                except Exception:
                    pass
                try:
                    torch._inductor.config.max_autotune_gemm_backends = 'ATEN,CPP'
                except Exception:
                    pass
                try:
                    torch._inductor.config.max_autotune = False
                except Exception:
                    pass
                if logger:
                    logger.info('ℹ️ TorchInductor config τροποποιήθηκε για μεγαλύτερη συμβατότητα (π.χ. απενεργοποίηση cudagraphs).')
        except BaseException as e:
            if logger:
                logger.warning(f'⚠️ Δεν ήταν δυνατή η τροποποίηση torch._inductor.config: {e}')
    except Exception as e:
        if logger:
            logger.error(f'❌ Σφάλμα κατά τη ρύθμιση Triton/Inductor: {e}')
setup_triton_windows_compat()


def _mmpro_code_dir() -> Path:
    if getattr(sys, 'frozen', False):
        try:
            return Path(getattr(sys, '_MEIPASS'))
        except Exception:
            return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
CODE_DIR = _mmpro_code_dir()


def is_portable_mode() -> bool:
    flag = (os.environ.get('MM_PRO_PORTABLE', '') or '').strip().lower()
    if flag in {'1', 'true', 'yes', 'on'}:
        return True
    marker_base = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else CODE_DIR
    return (marker_base / 'portable_mode.txt').exists()


def _default_user_data_dir() -> Path:
    try:
        if sys.platform.startswith("win"):
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
            if not base:
                base = str(Path.home() / "AppData" / "Local")
            return Path(base).expanduser().resolve() / "Models_Manager_Pro"
        if sys.platform == "darwin":
            return (Path.home() / "Library" / "Application Support" / "Models_Manager_Pro").resolve()
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser().resolve() if xdg else (Path.home() / ".local" / "share").resolve()
        return base / "models_manager_pro"
    except Exception:
        return (Path.home().resolve() / "Models_Manager_Pro")


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    test = p / ".__mmpro_write_test__"
    try:
        test.write_text("ok", encoding="utf-8")
        try:
            test.unlink(missing_ok=True)
        except TypeError:
            if test.exists():
                test.unlink()
        return True
    except Exception:
        try:
            if test.exists():
                test.unlink()
        except Exception:
            pass
        return False


def _looks_like_mmpro_root(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        for d in ("Data_Sets", "Trained_Models", "Crash_Logs", "Train_Reports"):
            if (p / d).exists():
                return True
    except Exception:
        return False
    return False


def get_app_data_root() -> Path:
    override = (os.environ.get('MM_PRO_DATA_DIR') or os.environ.get('MM_PRO_ROOT_DIR') or '').strip()
    if override:
        return Path(override).expanduser().resolve()
    frozen = bool(getattr(sys, 'frozen', False))
    exe_dir = Path(sys.executable).resolve().parent if frozen else CODE_DIR
    if is_portable_mode():
        return exe_dir
    if not frozen:
        return CODE_DIR
    try:
        if _looks_like_mmpro_root(exe_dir) and _is_writable_dir(exe_dir):
            return exe_dir
    except Exception:
        pass
    return _default_user_data_dir()


def resource_path(relative_path: str) -> str:
    rel = str(relative_path).lstrip("/\\")
    candidates: list[Path] = []
    try:
        candidates.append(Path(getattr(sys, '_MEIPASS')))
    except Exception:
        pass
    try:
        candidates.append(CODE_DIR)
    except Exception:
        pass
    if getattr(sys, 'frozen', False):
        try:
            candidates.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
    for base in candidates:
        try:
            p = (base / rel).resolve()
            if p.exists():
                return str(p)
        except Exception:
            continue
    base = candidates[0] if candidates else Path('.')
    return str((base / rel).resolve())
# ════════════════════════════════════════════════════════════════════════════════
# ΠΑΘΕΣ ΦΑΚΕΛΩΝ ΕΦΑΡΜΟΓΗΣ
# ROOT_DIR:               Ριζικός φάκελος δεδομένων (π.χ. %LOCALAPPDATA%/Models_Manager_Pro)
# DATASETS_DIR:           Data_Sets/     – αποθήκη datasets
# MODELS_DIR_INITIAL:     Base_Models/   – pretrained YOLO weights
# TRAINED_MODELS_DIR:     Trained_Models/– εκπαιδευμένα μοντέλα
# TRAIN_REPORTS_DIR:      Train_Reports/ – PDF αναφορές εκπαίδευσης
# DETECTION_REPORTS_DIR:  Detection_Reports/
# ════════════════════════════════════════════════════════════════════════════════
ROOT_DIR = get_app_data_root()
DATASETS_DIR = ROOT_DIR / 'Data_Sets'
MODELS_DIR_INITIAL = ROOT_DIR / 'Base_Models'
CRASH_LOGS_DIR = ROOT_DIR / 'Crash_Logs'
TRAINED_MODELS_DIR = ROOT_DIR / 'Trained_Models'
MODELS_DIR_TRAINED_PT = TRAINED_MODELS_DIR
TRAIN_REPORTS_DIR = ROOT_DIR / 'Train_Reports'
DETECTION_REPORTS_DIR = ROOT_DIR / 'Detection_Reports'
DETECTION_PREVIEW_DIR = ROOT_DIR / '_stats_preview'
for _p in [ROOT_DIR, MODELS_DIR_INITIAL, CRASH_LOGS_DIR, DATASETS_DIR, TRAINED_MODELS_DIR, TRAIN_REPORTS_DIR, DETECTION_REPORTS_DIR, DETECTION_PREVIEW_DIR]:
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ── Persistent Settings ─────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# AppSettings – Persistent ρυθμίσεις εφαρμογής (JSON αρχείο)
# ════════════════════════════════════════════════════════════════════════════════
# Αποθηκεύει/φορτώνει ρυθμίσεις χρήστη στο ROOT_DIR/app_settings.json.
# Χρησιμοποιεί singleton pattern (AppSettings.instance()).
# Convenience methods: restore_combo/save_combo, restore_spin/save_spin, κ.λπ.
class AppSettings:
    """Αποθηκεύει/φορτώνει ρυθμίσεις σε JSON αρχείο στο ROOT_DIR."""
    _FILE = 'app_settings.json'
    _inst: AppSettings | None = None

    def __init__(self):
        self._path = ROOT_DIR / self._FILE
        self._data: dict = {}
        self._load()

    @classmethod

    def instance(cls) -> AppSettings:
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._data = json.loads(self._path.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            self._data = {}

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def set_many(self, mapping: dict) -> None:
        self._data.update(mapping)
        self.save()

    # ── Convenience helpers ──────────────────────────────────────────────────

    def restore_combo(self, combo: QComboBox, key: str) -> None:
        v = self.get(key)
        if v is None:
            return
        idx = combo.findText(str(v))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def save_combo(self, combo: QComboBox, key: str) -> None:
        self.set(key, combo.currentText())

    def restore_spin(self, spin, key: str) -> None:
        v = self.get(key)
        if v is None:
            return
        try:
            spin.setValue(type(spin.value())(v))
        except Exception:
            pass

    def save_spin(self, spin, key: str) -> None:
        self.set(key, spin.value())

    def restore_check(self, cb: QCheckBox, key: str) -> None:
        v = self.get(key)
        if v is None:
            return
        try:
            cb.setChecked(bool(v))
        except Exception:
            pass

    def save_check(self, cb: QCheckBox, key: str) -> None:
        self.set(key, cb.isChecked())

def _settings() -> AppSettings:
    return AppSettings.instance()


def is_frozen_app() -> bool:
    return bool(getattr(sys, 'frozen', False))


def should_print_traceback(_exc: Exception | None = None) -> bool:
    flag = str(os.environ.get('MM_DEBUG_TRACEBACK', '0')).strip().lower()
    return flag in ('1', 'true', 'yes', 'on')


def guess_ultralytics_task(model_path: str | Path, default: str = 'detect') -> str:
    """Εκτιμά το task (detect/classify/segment/pose/obb/cnn_classify) από το όνομα μοντέλου."""
    try:
        name = str(model_path).lower()
    except Exception:
        return default
    # CNN torchvision models
    for cnn in _CNN_MODEL_KEYS:
        if cnn in name:
            return 'classify'
    if any(tok in name for tok in ('-cls', '_cls', 'cls.', 'classify', 'classification')):
        return 'classify'
    if any(tok in name for tok in ('-seg', '_seg', 'seg.', 'segment', 'segmentation')):
        return 'segment'
    if 'pose' in name:
        return 'pose'
    if 'obb' in name:
        return 'obb'
    return default


def _windows_add_dll_dirs() -> None:
    if os.name != 'nt':
        return
    if str(os.environ.get('MM_DISABLE_DLL_DIRS', '0')).strip().lower() in ('1', 'true', 'yes', 'on'):
        return
    enable_in_dev = str(os.environ.get('MM_ENABLE_DLL_DIRS', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
    if (not is_frozen_app()) and (not enable_in_dev):
        return
    add = getattr(os, 'add_dll_directory', None)
    if not callable(add):
        return
    candidates: list[Path] = []
    try:
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            p = Path(str(meipass)).resolve()
            if p.exists():
                candidates.append(p)
    except Exception:
        pass
    if is_frozen_app():
        try:
            candidates.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
    candidates += [ROOT_DIR / 'TensorRT-10.13.3.9' / 'bin']
    if enable_in_dev:
        try:
            candidates.append(CODE_DIR)
        except Exception:
            pass
    seen: set[str] = set()
    for d in candidates:
        try:
            if not d.exists() or not d.is_dir():
                continue
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            add(str(d))
        except Exception:
            pass
_windows_add_dll_dirs()
try:
    _old_pt_dir = ROOT_DIR / 'PyTorch_Trained_Models'
    if _old_pt_dir.exists() and _old_pt_dir.is_dir():
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for _item in _old_pt_dir.iterdir():
            _dst = TRAINED_MODELS_DIR / _item.name
            if _dst.exists():
                continue
            try:
                shutil.move(str(_item), str(_dst))
            except Exception:
                pass
        try:
            if not any(_old_pt_dir.iterdir()):
                _old_pt_dir.rmdir()
        except Exception:
            pass
except Exception:
    pass
try:
    if settings is not None:
        try:
            settings.update({ 'datasets_dir': str(DATASETS_DIR), 'weights_dir': str(MODELS_DIR_INITIAL), 'runs_dir': str(ROOT_DIR / 'Runs')})
        except Exception:
            pass
except Exception as e:
    print(f'Warning: Could not update ultralytics settings: {e}')


def configure_ultralytics_logging(level: int = logging.WARNING) -> None:
    """Φιλτράρει verbose logs από τη βιβλιοθήκη Ultralytics/YOLO."""

    class _MMProUltraNoiseFilter(logging.Filter):

        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage() or ''
                if 'Unable to automatically guess model task' in msg:
                    return False
                if msg.startswith('Loading ') and ' inference' in msg:
                    return False
                if msg.startswith('Using ONNX Runtime'):
                    return False
            except Exception:
                return True
            return True
    _noise_filter = _MMProUltraNoiseFilter()
    try:
        ul = logging.getLogger('ultralytics')
        ul.setLevel(level)
        try:
            ul.addFilter(_noise_filter)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from ultralytics.utils import LOGGER as _ULTRA_LOGGER
        try:
            _ULTRA_LOGGER.setLevel(level)
            _ULTRA_LOGGER.addFilter(_noise_filter)
        except Exception:
            pass
        try:
            for h in (getattr(_ULTRA_LOGGER, 'handlers', None) or []):
                try:
                    h.setLevel(level)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
try:
    configure_ultralytics_logging(logging.WARNING)
except Exception:
    pass


@dataclass


@dataclass


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 13 – Dataset & model configuration constants
# ═══════════════════════════════════════════════════════════════════════

class DatasetConfig:
    """Dataclass που περιγράφει ένα training dataset (URLs, train/val dirs)."""
    name: str
    zip_url: str | None = None
    yaml_url: str | None = None
    train_dir: str | None = None
    val_dir: str | None = None
    yaml_name: str | None = None
# ════════════════════════════════════════════════════════════════════════════════
# ΠΡΟΚΑΘΟΡΙΣΜΕΝΑ DATASETS ΕΚΠΑΙΔΕΥΣΗΣ (YOLO)
# coco8: 8 εικόνες – για γρήγορες δοκιμές
# coco128: 128 εικόνες – για σύντομη εκπαίδευση
# coco: πλήρες COCO 2017 (~20GB) – για production εκπαίδευση
# Για CNN: χρησιμοποιούνται custom datasets από Data_Sets/<name>/train/val/
# ════════════════════════════════════════════════════════════════════════════════
TRAIN_DATASETS: dict[str, DatasetConfig] = {
    "coco8": DatasetConfig(
        name="coco8",
        zip_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        yaml_url="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml",
        train_dir=f"{DATASETS_DIR}/coco8/images/train",
        val_dir=f"{DATASETS_DIR}/coco8/images/val",
        yaml_name="coco8.yaml",
    ),
    "coco128": DatasetConfig(
        name="coco128",
        zip_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        yaml_url="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml",
        train_dir=f"{DATASETS_DIR}/coco128/images/train2017",
        val_dir=f"{DATASETS_DIR}/coco128/images/val2017",
        yaml_name="coco128.yaml",
    ),
    "coco": DatasetConfig(
        name="coco",
        yaml_url="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml",
        train_dir=f"{DATASETS_DIR}/coco/images/train2017",
        val_dir=f"{DATASETS_DIR}/coco/images/val2017",
        yaml_name="coco.yaml",
    ),
}
# ════════════════════════════════════════════════════════════════════════════════
# ΛΙΣΤΑ ΜΟΝΤΕΛΩΝ ΕΚΠΑΙΔΕΥΣΗΣ (YOLO / Ultralytics)
# Περιέχει όλες τις παραλλαγές YOLO v5–v12 για detection.
# Χρησιμοποιείται από το Tab '🎓 Εκπαίδευση Μοντέλου' όταν επιλεγεί 'Yolo (Detection)'.
# ════════════════════════════════════════════════════════════════════════════════
TRAIN_MODELS: list[str] = [
    "yolov5nu", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
    "yolov6n", "yolov6s", "yolov6m", "yolov6l", "yolov6x",
    "yolov7", "yolov7x", "yolov7-w6", "yolov7-e6", "yolov7-d6", "yolov7-e6e",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov9c", "yolov9e", "yolov9t", "yolov9s", "yolov9m", "yolov9l", "yolov9x",
    "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
]
TRAIN_CLS_MODELS: list[str] = [ "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls", "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",]
ALL_TRAIN_MODELS: list[str] = TRAIN_MODELS + TRAIN_CLS_MODELS
# ── CNN Classifier models (torchvision) ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════
# ΛΙΣΤΑ CNN ΜΟΝΤΕΛΩΝ (torchvision) – MobileNet / ResNet
# Εκπαιδεύονται με CNNTrainingWorker (PyTorch native, ΟΧΙ Ultralytics).
# Απαιτούν dataset ταξινόμησης: Data_Sets/<dataset>/train/<class>/...
#                                               val/<class>/...
# ΔΕΝ υποστηρίζουν: Triton/TorchCompile, TensorRT export, NCNN export.
# ΥΠΟΣΤΗΡΙΖΟΥΝ: ONNX export (torch.onnx.export), live camera classification.
# ════════════════════════════════════════════════════════════════════════════════
TRAIN_CNN_MODELS: list[str] = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet50",
    "resnet101",
]
_CNN_MODEL_KEYS: frozenset[str] = frozenset(TRAIN_CNN_MODELS)


def is_cnn_model(name: str) -> bool:
    """Επιστρέφει True αν το μοντέλο είναι CNN classifier (torchvision), όχι YOLO."""
    return str(name).lower().strip() in _CNN_MODEL_KEYS


# ════════════════════════════════════════════════════════════════════════════════
# CNN PATH DETECTION – Ελέγχει αν ένα αρχείο μοντέλου είναι CNN torchvision
# ════════════════════════════════════════════════════════════════════════════════
# Ελέγχει (κατά προτεραιότητα):
#   1. Stem name του αρχείου (mobilenet_v2, resnet50 κ.λπ.)
#   2. Sibling class_names.json  → model_name key
#   3. Για .pt: peek στο checkpoint dict  → 'model_name' key
# ΣΗΜΑΝΤΙΚΟ: Δεν καλεί ποτέ YOLO() (θα έδινε KeyError 'model' σε CNN checkpoints).
def _is_cnn_path(model_path) -> bool:
    """
    Ελέγχει αν ένα αρχείο μοντέλου (.pt ή .onnx) είναι CNN torchvision.
    Ελέγχει (κατά προτεραιότητα):
      1. Stem name του αρχείου (mobilenet_v2, resnet50 κ.λπ.)
      2. Sibling class_names.json  → model_name key
      3. Για .pt: peek στο checkpoint dict  → 'model_name' key
    Δεν κάνει ποτέ YOLO() load.
    """
    try:
        p = Path(model_path)
        stem = p.stem.lower()
        # Βήμα 1: Έλεγχος στο όνομα του αρχείου (π.χ. 'mobilenet_v2_GPU_grape_224.pt')
        if any(c in stem for c in _CNN_MODEL_KEYS):
            return True
        # Βήμα 2: Έλεγχος αν υπάρχει class_names.json στον ίδιο φάκελο
        cj = p.parent / 'class_names.json'
        if cj.is_file():
            try:
                d = json.loads(cj.read_text(encoding='utf-8', errors='replace'))
                mn = str(d.get('model_name', '')).lower()
                if any(c in mn for c in _CNN_MODEL_KEYS):
                    return True
            except Exception:
                pass
        # Βήμα 3: Γρήγορο peek στο checkpoint (μόνο για .pt) – ελέγχει το 'model_name' key
        if p.suffix.lower() == '.pt':
            try:
                import torch as _t
                ck = _t.load(str(p), map_location='cpu', weights_only=False)
                if isinstance(ck, dict) and 'model_name' in ck:
                    mn = str(ck.get('model_name', '')).lower()
                    return any(c in mn for c in _CNN_MODEL_KEYS)
            except Exception:
                pass
    except Exception:
        pass
    return False


# ── Εξαγωγή model_name (π.χ. 'mobilenet_v2') από path CNN μοντέλου ─────────
# Ελέγχει JSON → checkpoint → stem name. Fallback: επιστρέφει κενό string.
def _cnn_model_name_from_path(model_path) -> str:
    """
    Εξάγει το model_name (π.χ. 'mobilenet_v2') από path CNN μοντέλου.
    Fallback: stem name.
    """
    try:
        p = Path(model_path)
        # From JSON
        cj = p.parent / 'class_names.json'
        if cj.is_file():
            try:
                d = json.loads(cj.read_text(encoding='utf-8', errors='replace'))
                mn = str(d.get('model_name', '')).lower()
                if mn and any(c in mn for c in _CNN_MODEL_KEYS):
                    return mn
            except Exception:
                pass
        # From checkpoint
        if p.suffix.lower() == '.pt':
            try:
                import torch as _t
                ck = _t.load(str(p), map_location='cpu', weights_only=False)
                mn = str(ck.get('model_name', '')).lower()
                if mn and any(c in mn for c in _CNN_MODEL_KEYS):
                    return mn
            except Exception:
                pass
        # From stem
        stem = p.stem.lower()
        for c in TRAIN_CNN_MODELS:
            if c in stem:
                return c
    except Exception:
        pass
    return ''


# ── Φόρτωση class names για CNN μοντέλο (χωρίς να φορτωθεί ολόκληρο το μοντέλο)
# Πηγές: 1. class_names.json  2. checkpoint dict 'class_names' key
# Επιστρέφει κενή λίστα αν δεν βρεθούν.
def _cnn_class_names_from_path(model_path) -> list[str]:
    """
    Φορτώνει λίστα class names για CNN μοντέλο από:
      1. class_names.json
      2. checkpoint dict
    Επιστρέφει κενή λίστα αν δεν βρεθούν.
    """
    try:
        p = Path(model_path)
        # 1. JSON
        cj = p.parent / 'class_names.json'
        if cj.is_file():
            try:
                d = json.loads(cj.read_text(encoding='utf-8', errors='replace'))
                names = list(d.get('class_names', []))
                if names:
                    return names
            except Exception:
                pass
        # 2. checkpoint
        if p.suffix.lower() == '.pt':
            try:
                import torch as _t
                ck = _t.load(str(p), map_location='cpu', weights_only=False)
                names = list(ck.get('class_names', []))
                if names:
                    return names
            except Exception:
                pass
    except Exception:
        pass
    return []


# ── Φόρτωση torchvision μοντέλου (MobileNet V2/V3, ResNet 50/101) ────────────
# Αν num_classes != 1000, αντικαθιστά το τελευταίο linear layer (classifier head)
# ώστε να ταιριάζει με τον αριθμό κλάσεων του dataset.
# pretrained=True: χρησιμοποιεί ImageNet pre-trained weights (DEFAULT).
# pretrained=False: τυχαία αρχικοποίηση (χρησιμοποιείται στο inference loading).
def _load_torchvision_model(model_name: str, num_classes: int = 1000, pretrained: bool = True):
    """Φορτώνει torchvision μοντέλο (MobileNet/ResNet) με pretrained weights."""
    import torchvision.models as tv_models
    import torch
    name = str(model_name).lower().strip()
    weights_arg = "DEFAULT" if pretrained else None
    if name == "mobilenet_v2":
        w = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = tv_models.mobilenet_v2(weights=w)
        if num_classes != 1000:
            in_f = m.classifier[1].in_features
            m.classifier[1] = torch.nn.Linear(in_f, num_classes)
    elif name == "mobilenet_v3_small":
        w = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = tv_models.mobilenet_v3_small(weights=w)
        if num_classes != 1000:
            in_f = m.classifier[3].in_features
            m.classifier[3] = torch.nn.Linear(in_f, num_classes)
    elif name == "mobilenet_v3_large":
        w = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = tv_models.mobilenet_v3_large(weights=w)
        if num_classes != 1000:
            in_f = m.classifier[3].in_features
            m.classifier[3] = torch.nn.Linear(in_f, num_classes)
    elif name == "resnet50":
        w = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        m = tv_models.resnet50(weights=w)
        if num_classes != 1000:
            in_f = m.fc.in_features
            m.fc = torch.nn.Linear(in_f, num_classes)
    elif name == "resnet101":
        w = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
        m = tv_models.resnet101(weights=w)
        if num_classes != 1000:
            in_f = m.fc.in_features
            m.fc = torch.nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Άγνωστο CNN μοντέλο: {model_name!r}")
    return m


# ── Δημιουργία torchvision transforms για CNN εκπαίδευση / inference ─────────
# train=True:  RandomResizedCrop + RandomHorizontalFlip + ColorJitter + Normalize
# train=False: Resize(imgsz*1.143) + CenterCrop(imgsz) + Normalize
# Normalization: ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
def _cnn_get_transforms(imgsz: int = 224, train: bool = False):
    """Επιστρέφει torchvision transforms για CNN inference/training."""
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("Λείπει το torchvision. Εγκατάστησέ το: pip install torchvision")
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(imgsz * 1.143)),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
TRAIN_IMAGE_SIZES: list[int] = [224, 320, 480, 640]
CAMERA_IMAGE_SIZES: list[int] = [224, 320, 480, 640]
TARGET_FPS: int = 60  # Στόχος frame rate για live detection
STATS_IMAGE_SIZE: int = 640  # Μέγεθος εικόνας (px) για στατιστική ανάλυση
STATS_CONFIDENCE_THRESHOLD: float = 0.5  # Κατώφλι confidence για φιλτράρισμα
STATS_IOU_THRESHOLD: float = 0.5  # Κατώφλι IoU για υπολογισμό mAP
STATS_IMG_EXTS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 12 – Χρώματα & HTML log formatting
# ═══════════════════════════════════════════════════════════════════════

class Colors:
    """Κλάση σταθερών χρωμάτων για το HTML log output (dark/light theme)."""
    HEADER = "#9b59b6"
    BLUE = "#569cd6"
    CYAN = "#1abc9c"
    GREEN = "#2ecc71"
    YELLOW = "#f1c40f"
    MAGENTA = "#e74c3c"
    RED = "#f44747"
    LIGHT = "#f0f0f0"
    ORANGE = "#e67e22"


def set_log_colors_for_theme(mode: str) -> None:
    m = (mode or '').strip().lower()
    if m == 'dark':
        Colors.HEADER = '#ffe6ff'
        Colors.BLUE = '#99ccff'
        Colors.CYAN = '#99ffff'
        Colors.GREEN = '#ccff99'
        Colors.YELLOW = '#ffefa3'
        Colors.MAGENTA = '#ffb3dd'
        Colors.RED = '#ff9999'
        Colors.LIGHT = '#ffffff'
    else:
        Colors.HEADER = '#000080'
        Colors.BLUE = '#003399'
        Colors.CYAN = '#005f73'
        Colors.GREEN = '#004225'
        Colors.YELLOW = '#7a5c00'
        Colors.MAGENTA = '#7b004b'
        Colors.RED = '#8b0000'
        Colors.LIGHT = '#000000'
_ANSI_RE = re.compile('\\x1b\\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    try:
        return _ANSI_RE.sub('', text or '')
    except Exception:
        return text or ''


def _is_separator_line(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if len(t) < 8:
        return False
    allowed = set('=-_·─━═⎯')
    return set(t) <= allowed
_HTML_LOG_BASE_STYLES: tuple[str, ...] = ( "white-space: pre", "font-family: 'Consolas', 'Courier New', monospace", "font-size: 11pt", "line-height: 1.2", "display: inline-block",)


def format_html_log( text: str, color: str | None = None, bold: bool = False, underline: bool = False,) -> str:
    raw = strip_ansi(text)
    safe = html.escape(raw, quote=False)
    styles: list[str] = list(_HTML_LOG_BASE_STYLES)
    is_sep = _is_separator_line(raw)
    if color:
        styles.append(f"color: {color}")
        if not is_sep:
            styles.append(f"border-left: 3px solid {color}")
            styles.append("padding: 2px 6px")
            styles.append("margin: 1px 0")
            styles.append("padding-left: 8px")
    else:
        styles.append("padding: 1px 2px")
    if bold:
        styles.append("font-weight: bold")
    if underline:
        styles.append("text-decoration: underline")
    style_attr = f' style="{" ; ".join(styles)}"' if styles else ""
    return f"<span{style_attr}>{safe}</span>"
BACKEND_PRETTY_NAMES: dict[str, str] = { "pytorch":  "PyTorch (.pt)", "onnx":     "ONNX (.onnx)", "tensorrt": "TensorRT (.engine)", "ncnn":     "NCNN (_ncnn_model)",}


def backend_pretty_name(backend: str) -> str:
    return BACKEND_PRETTY_NAMES.get((backend or '').lower(), backend)


def find_available_backends(models_dir: Path, base_name: str) -> dict[str, Path]:
    backends: dict[str, Path] = {}
    pt_path = models_dir / f'{base_name}.pt'
    onnx_path = models_dir / f'{base_name}.onnx'
    engine_path = models_dir / f'{base_name}.engine'
    ncnn_dir = models_dir / f'{base_name}_ncnn_model'
    if pt_path.exists():
        backends['pytorch'] = pt_path
    if onnx_path.exists():
        backends['onnx'] = onnx_path
    if engine_path.exists():
        backends['tensorrt'] = engine_path
    if ncnn_dir.exists() and ncnn_dir.is_dir():
        backends['ncnn'] = ncnn_dir
    return backends


def cuda_sync() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def yolo_is_classification(model: object) -> bool:
    try:
        return getattr(model, 'task', None) == 'classify'
    except Exception:
        return False


def _mmpro_parse_expected_imgsz_from_error(msg: str):
    try:
        s = str(msg or '')
        pairs = re.findall(r"index:\s*(\d+)\s*Got:\s*(\d+)\s*Expected:\s*(\d+)", s)
        exp = {}
        for idx, _got, ex in pairs:
            try:
                exp[int(idx)] = int(ex)
            except Exception:
                pass
        if 2 in exp and 3 in exp and exp[2] > 0 and exp[3] > 0:
            h = int(exp[2])
            w = int(exp[3])
            return int(h) if h == w else (h, w)
        m_trt = re.search(r"max\s+model\s+size\s*[\(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]", s, re.IGNORECASE)
        if m_trt:
            try:
                h = int(m_trt.group(3))
                w = int(m_trt.group(4))
                if h > 0 and w > 0:
                    return int(h) if h == w else (h, w)
            except Exception:
                pass
        m_trt_hw = re.search(r"max\s+model\s+size\s*[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]", s, re.IGNORECASE)
        if m_trt_hw:
            try:
                h = int(m_trt_hw.group(1))
                w = int(m_trt_hw.group(2))
                if h > 0 and w > 0:
                    return int(h) if h == w else (h, w)
            except Exception:
                pass
        exps = re.findall(r"Expected:\s*(\d+)", s)
        if exps:
            try:
                v = int(exps[0])
                if v > 0:
                    return int(v)
            except Exception:
                pass
    except Exception:
        pass


def _mmpro_parse_imgsz_from_name(name: str):
    try:
        if not name:
            return None
        s = str(name)
        m = re.search(r'imgsz(\d+)', s, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                return v if v > 0 else None
            except Exception:
                return None
        m2 = re.search(r'_(\d+)$', s)
        if m2:
            try:
                v = int(m2.group(1))
                return v if v > 0 else None
            except Exception:
                return None
    except Exception:
        pass


def _mmpro_get_forced_imgsz(model: object):
    try:
        v = getattr(model, '_mmpro_forced_imgsz', None)
        if isinstance(v, int) and v > 0:
            return int(v)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            h, w = v
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return (int(h), int(w))
    except Exception:
        pass


def _mmpro_read_export_meta_for_path(path: 'Path') -> dict:
    try:
        from pathlib import Path
        import json
        p = Path(path)
        cand: list[Path] = []
        if p.is_dir():
            cand.append(p / 'mmpro_export_meta.json')
            cand.append(p / 'export_meta.mmpro.json')
        else:
            cand.append(p.with_suffix(p.suffix + '.mmpro.json'))
            cand.append(p.parent / (p.stem + '.mmpro.json'))
        for c in cand:
            try:
                if c.exists() and c.is_file():
                    return json.loads(c.read_text(encoding='utf-8', errors='ignore') or '{}') or {}
            except Exception:
                continue
    except Exception:
        pass
    return {}


def _mmpro_try_infer_ncnn_imgsz_from_param(ncnn_dir: 'Path'):
    try:
        from pathlib import Path
        d = Path(ncnn_dir)
        if not d.exists() or not d.is_dir():
            return None
        param_files = list(d.glob('*.param'))
        if not param_files:
            return None
        param_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        txt = param_files[0].read_text(encoding='utf-8', errors='ignore')
        lines = [ln for ln in txt.splitlines() if 'Input' in ln]
        if not lines:
            lines = txt.splitlines()[:50]
        nums: list[int] = []
        for ln in lines:
            for m in re.findall(r"=\s*(\d{2,4})", ln):
                try:
                    v = int(m)
                    if 32 <= v <= 2048:
                        nums.append(v)
                except Exception:
                    pass
        if not nums:
            return None
        from collections import Counter
        c = Counter(nums)
        sq = [v for v, k in c.items() if k >= 2 and v % 8 == 0]
        if sq:
            return int(max(sq))
        cand = [v for v in nums if v % 8 == 0]
        return int(max(cand)) if cand else int(max(nums))
    except Exception:
        return None


def yolo_predict_first(model: object, frame: np.ndarray, imgsz: int | tuple[int, int], conf: float=0.25, iou: float=0.45, verbose: bool=False, classes=None):
    forced = _mmpro_get_forced_imgsz(model)
    imgsz_use = forced if forced is not None else imgsz

    def _mmpro_normalize_imgsz_for_task(v, is_classify: bool):
        if not is_classify:
            return v
        try:
            if isinstance(v, (tuple, list)) and len(v) == 2:
                h, w = v
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return int(h) if h == w else (int(h), int(w))
        except Exception:
            pass
        return v

    def _call_predict(_imgsz):
        is_cls = yolo_is_classification(model)
        _imgsz = _mmpro_normalize_imgsz_for_task(_imgsz, is_cls)
        if is_cls:
            return model.predict(frame, imgsz=_imgsz, verbose=verbose)
        kw = dict(imgsz=_imgsz, conf=conf, iou=iou, verbose=verbose)
        if classes is not None:
            kw['classes'] = classes
        return model.predict(frame, **kw)
    try:
        out = _call_predict(imgsz_use)
    except Exception as e:
        exp = _mmpro_parse_expected_imgsz_from_error(str(e))
        if exp is not None:
            try:
                setattr(model, '_mmpro_forced_imgsz', exp)
            except Exception:
                pass
            try:
                out = _call_predict(exp)
            except Exception:
                raise
        else:
            raise
    try:
        return out[0]
    except Exception:
        return out


def _mmpro_fix_ncnn_results(results, orig_frame: np.ndarray, model_imgsz: int) -> object:
    try:
        if results is None:
            return results
        if orig_frame is None:
            return results
        orig_h, orig_w = orig_frame.shape[:2]
        try:
            r_orig = getattr(results, 'orig_img', None)
            if r_orig is None:
                return results
            r_h, r_w = np.asarray(r_orig).shape[:2]
        except Exception:
            return results
        if r_h == orig_h and r_w == orig_w:
            return results
        reported_orig_shape = None
        try:
            reported_orig_shape = getattr(results, 'orig_shape', None)
        except Exception:
            pass
        imgsz_for_check = int(model_imgsz) if model_imgsz and int(model_imgsz) > 0 else 640
        boxes_already_in_orig_space = False
        if reported_orig_shape is not None:
            try:
                rs_h = int(reported_orig_shape[0])
                rs_w = int(reported_orig_shape[1])
                if rs_h == orig_h and rs_w == orig_w:
                    _extent_confirmed = False
                    try:
                        _boxes_obj = getattr(results, 'boxes', None)
                        if _boxes_obj is not None and hasattr(_boxes_obj, 'xyxy'):
                            _xyxy = _boxes_obj.xyxy
                            if hasattr(_xyxy, 'cpu'):
                                _xyxy = _xyxy.cpu().numpy()
                            else:
                                _xyxy = np.asarray(_xyxy)
                            if _xyxy.shape[0] == 0:
                                _extent_confirmed = True
                            elif float(np.max(_xyxy)) > imgsz_for_check * 1.05:
                                _extent_confirmed = True
                    except Exception:
                        pass
                    boxes_already_in_orig_space = _extent_confirmed
            except Exception:
                pass
        if not boxes_already_in_orig_space:
            lbsz_h = r_h
            lbsz_w = r_w
            scale = min(lbsz_w / max(1, orig_w), lbsz_h / max(1, orig_h))
            pad_w = (lbsz_w - orig_w * scale) / 2.0
            pad_h = (lbsz_h - orig_h * scale) / 2.0
            try:
                boxes = getattr(results, 'boxes', None)
                if boxes is not None and hasattr(boxes, 'data') and boxes.data is not None:
                    _d = boxes.data
                    try:
                        if hasattr(_d, 'cpu'):
                            _arr = _d.cpu().numpy()
                        else:
                            _arr = np.asarray(_d)
                        if _arr.shape[0] > 0:
                            try:
                                if hasattr(_d, 'clone'):
                                    _fixed = _d.clone().float()
                                    _fixed[:, 0] = (_d[:, 0].float() - pad_w) / scale
                                    _fixed[:, 1] = (_d[:, 1].float() - pad_h) / scale
                                    _fixed[:, 2] = (_d[:, 2].float() - pad_w) / scale
                                    _fixed[:, 3] = (_d[:, 3].float() - pad_h) / scale
                                    _fixed[:, 0].clamp_(0, orig_w)
                                    _fixed[:, 1].clamp_(0, orig_h)
                                    _fixed[:, 2].clamp_(0, orig_w)
                                    _fixed[:, 3].clamp_(0, orig_h)
                                    boxes.data = _fixed
                                else:
                                    _farr = np.array(_arr, dtype=np.float32)
                                    _farr[:, 0] = (_farr[:, 0] - pad_w) / scale
                                    _farr[:, 1] = (_farr[:, 1] - pad_h) / scale
                                    _farr[:, 2] = (_farr[:, 2] - pad_w) / scale
                                    _farr[:, 3] = (_farr[:, 3] - pad_h) / scale
                                    _farr[:, 0] = np.clip(_farr[:, 0], 0, orig_w)
                                    _farr[:, 1] = np.clip(_farr[:, 1], 0, orig_h)
                                    _farr[:, 2] = np.clip(_farr[:, 2], 0, orig_w)
                                    _farr[:, 3] = np.clip(_farr[:, 3], 0, orig_h)
                                    boxes.data = _farr
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                results.orig_shape = (int(orig_h), int(orig_w))
            except Exception:
                pass
        try:
            results.orig_img = orig_frame
        except Exception:
            pass
    except Exception:
        pass
    return results


def _ncnn_manual_annotate( frame: np.ndarray, results, model_imgsz: int,) -> np.ndarray:
    try:
        annotated = frame.copy()
        orig_h, orig_w = frame.shape[:2]
        raw_data = None
        try:
            boxes_obj = getattr(results, 'boxes', None)
            if boxes_obj is None:
                return annotated
            for attr in ('_data', 'data'):
                v = getattr(boxes_obj, attr, None)
                if v is not None:
                    try:
                        arr = v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
                        if arr.ndim == 2 and arr.shape[1] >= 6:
                            raw_data = arr
                            break
                    except Exception:
                        pass
            if raw_data is None or raw_data.ndim != 2 or raw_data.shape[1] < 6:
                xyxy_raw = getattr(boxes_obj, 'xyxy', None)
                conf_raw = getattr(boxes_obj, 'conf', None)
                cls_raw  = getattr(boxes_obj, 'cls', None)
                if xyxy_raw is not None and conf_raw is not None and cls_raw is not None:

                    def _to_np(t):
                        return t.cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                    raw_data = np.concatenate([ _to_np(xyxy_raw), _to_np(conf_raw).reshape(-1, 1), _to_np(cls_raw).reshape(-1, 1),], axis=1)
        except Exception:
            return annotated
        if raw_data is None or raw_data.shape[0] == 0:
            return annotated
        ros_h, ros_w = orig_h, orig_w
        try:
            ros = getattr(results, 'orig_shape', None)
            if ros is not None and len(ros) >= 2:
                ros_h = int(ros[0])
                ros_w = int(ros[1])
        except Exception:
            pass
        coords = raw_data[:, :4].copy().astype(np.float32)
        if ros_h != orig_h or ros_w != orig_w:
            lbsz_h = ros_h if ros_h > 0 else (int(model_imgsz) if model_imgsz else 640)
            lbsz_w = ros_w if ros_w > 0 else (int(model_imgsz) if model_imgsz else 640)
            scale  = min(lbsz_w / max(orig_w, 1), lbsz_h / max(orig_h, 1))
            pad_w  = (lbsz_w - orig_w * scale) / 2.0
            pad_h  = (lbsz_h - orig_h * scale) / 2.0
            coords[:, 0] = (coords[:, 0] - pad_w) / scale
            coords[:, 1] = (coords[:, 1] - pad_h) / scale
            coords[:, 2] = (coords[:, 2] - pad_w) / scale
            coords[:, 3] = (coords[:, 3] - pad_h) / scale
        coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, orig_w)
        coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, orig_h)
        _PERSON_CLASS = 0
        for i in range(coords.shape[0]):
            x1, y1, x2, y2 = coords[i]
            if y2 <= y1:
                w = max(x2 - x1, 1.0)
                cls_i = int(raw_data[i, 5]) if raw_data.shape[1] > 5 else -1
                ar = 2.0 if cls_i == _PERSON_CLASS else 1.0
                y2_est = float(np.clip(y1 + w * ar, 0, orig_h))
                coords[i, 3] = y2_est
        try:
            confs = raw_data[:, 4].astype(np.float32)
            order = np.argsort(confs)[::-1]
            keep = []
            suppressed = np.zeros(len(order), dtype=bool)
            for ii in range(len(order)):
                if suppressed[ii]:
                    continue
                keep.append(order[ii])
                bx1, by1, bx2, by2 = coords[order[ii]]
                b_area = max(bx2 - bx1, 0) * max(by2 - by1, 0)
                for jj in range(ii + 1, len(order)):
                    if suppressed[jj]:
                        continue
                    ox1, oy1, ox2, oy2 = coords[order[jj]]
                    ix1 = max(bx1, ox1)
                    iy1 = max(by1, oy1)
                    ix2 = min(bx2, ox2)
                    iy2 = min(by2, oy2)
                    iw = max(ix2 - ix1, 0)
                    ih = max(iy2 - iy1, 0)
                    inter = iw * ih
                    o_area = max(ox2 - ox1, 0) * max(oy2 - oy1, 0)
                    union = b_area + o_area - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > 0.45:
                        suppressed[jj] = True
            raw_data  = raw_data[keep]
            coords    = coords[keep]
        except Exception:
            pass
        names: dict = {}
        try:
            n = getattr(results, 'names', None)
            if isinstance(n, dict):
                names = n
            elif isinstance(n, (list, tuple)):
                names = {i: str(v) for i, v in enumerate(n)}
        except Exception:
            pass
        _PALETTE = [
            (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
            (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
            (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
            (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
            (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
        ]
        lw = max(1, int(round(0.002 * max(orig_h, orig_w))))
        for row, coord in zip(raw_data, coords):
            try:
                x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])
                conf  = float(row[4]) if len(row) > 4 else 1.0
                cls_i = int(row[5])   if len(row) > 5 else 0
                color = _PALETTE[cls_i % len(_PALETTE)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)
                cls_name = names.get(cls_i, f'cls{cls_i}')
                label = f'{cls_name} {conf:.2f}'
                font_scale = max(0.35, lw * 0.4)
                font_thick = max(1, lw - 1)
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                lbl_y1 = max(y1 - th - baseline - 4, 0)
                lbl_y2 = max(y1, th + baseline + 4)
                cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 4, lbl_y2), color, -1)
                cv2.putText( annotated, label, (x1 + 2, lbl_y2 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,)
            except Exception:
                continue
        return annotated
    except Exception:
        try:
            return frame.copy()
        except Exception:
            return frame


def _find_first_matching_dll(patterns: list[str], dirs: list[Path]) -> Path | None:
    for d in dirs:
        try:
            if not d or not d.exists() or not d.is_dir():
                continue
        except Exception:
            continue
        for pat in patterns:
            try:
                for p in d.glob(pat):
                    if p.is_file():
                        return p
            except Exception:
                continue


def _ensure_tensorrt_plugins_loaded_windows() -> dict:
    info = { 'plugin_found': '', 'plugin_loaded': False, 'add_dir_ok': False, 'add_dir_error': '', 'plugin_load_error': '',}
    if os.name != 'nt':
        return info
    if str(os.environ.get('MM_DISABLE_TRT_DLL_LOAD', '0')).strip().lower() in ('1', 'true', 'yes', 'on'):
        return info
    cand: list[Path] = []
    try:
        for s in os.environ.get('PATH', '').split(os.pathsep):
            if s:
                cand.append(Path(s))
    except Exception:
        pass
    try:
        import tensorrt as _trt
        trt_dir = Path(getattr(_trt, '__file__', '')).resolve().parent
        cand += [trt_dir, trt_dir / 'lib', trt_dir / 'libs', trt_dir / '.libs', trt_dir.parent]
    except Exception:
        pass
    if is_frozen_app():
        try:
            cand.append(Path(getattr(sys, '_MEIPASS', '')))
            cand.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
    try:
        cand += [ROOT_DIR / 'TensorRT-10.13.3.9' / 'bin']
    except Exception:
        pass
    seen: set[str] = set()
    dirs: list[Path] = []
    for d in cand:
        try:
            if not d or not d.exists() or not d.is_dir():
                continue
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            dirs.append(d)
        except Exception:
            continue
    plugin = _find_first_matching_dll(['nvinfer_plugin*.dll'], dirs)
    if not plugin:
        return info
    info['plugin_found'] = str(plugin)
    add = getattr(os, 'add_dll_directory', None)
    if callable(add):
        try:
            add(str(plugin.parent))
            info['add_dir_ok'] = True
        except Exception as e:
            info['add_dir_error'] = str(e)
    try:
        import ctypes
        ctypes.CDLL(str(plugin))
        info['plugin_loaded'] = True
    except Exception as e:
        info['plugin_load_error'] = str(e)
    return info


def load_yolo_for_backend(backend: str, path: Path) -> "Any":
    b = (backend or '').lower()
    task = guess_ultralytics_task(path)
    if b in ('pytorch',):
        return (YOLO(str(path)), backend_pretty_name(b).split(' ')[0])
    if b in ('onnx',):
        try:
            return (YOLO(str(path), task=task), backend_pretty_name(b).split(' ')[0])
        except TypeError:
            return (YOLO(str(path)), backend_pretty_name(b).split(' ')[0])
    if b == 'tensorrt':
        _preflight_tensorrt_engine(path)
        try:
            return (YOLO(str(path), task=task), backend_pretty_name(b).split(' ')[0])
        except TypeError:
            return (YOLO(str(path)), backend_pretty_name(b).split(' ')[0])
    if b == 'ncnn':
        _preflight_ncnn_import()
        try:
            m = YOLO(str(path), task=task)
            try:
                meta = _mmpro_read_export_meta_for_path(path)
                v = meta.get('imgsz') if isinstance(meta, dict) else None
                if isinstance(v, int) and v > 0:
                    setattr(m, '_mmpro_forced_imgsz', int(v))
                elif isinstance(v, (tuple, list)) and len(v) == 2:
                    h, w = v
                    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                        setattr(m, '_mmpro_forced_imgsz', (int(h), int(w)))
            except Exception:
                pass
            try:
                if getattr(m, '_mmpro_forced_imgsz', None) is None and path.is_dir():
                    g = _mmpro_try_infer_ncnn_imgsz_from_param(path)
                    if isinstance(g, int) and g > 0:
                        setattr(m, '_mmpro_forced_imgsz', int(g))
            except Exception:
                pass
            return (m, backend_pretty_name(b).split(' ')[0])
        except TypeError:
            m = YOLO(str(path))
            try:
                meta = _mmpro_read_export_meta_for_path(path)
                v = meta.get('imgsz') if isinstance(meta, dict) else None
                if isinstance(v, int) and v > 0:
                    setattr(m, '_mmpro_forced_imgsz', int(v))
            except Exception:
                pass
            try:
                if getattr(m, '_mmpro_forced_imgsz', None) is None and path.is_dir():
                    g = _mmpro_try_infer_ncnn_imgsz_from_param(path)
                    if isinstance(g, int) and g > 0:
                        setattr(m, '_mmpro_forced_imgsz', int(g))
            except Exception:
                pass
            return (m, backend_pretty_name(b).split(' ')[0])
    raise ValueError(f'Άγνωστο backend: {backend}')


def perform_smart_memory_cleanup(context: str = '') -> str:
    info_parts: list[str] = []
    try:
        import gc
        collected = gc.collect()
        info_parts.append(f'GC objects: {collected}')
    except Exception:
        info_parts.append('GC: σφάλμα')
    try:
        import torch
        if torch.cuda.is_available():
            before_reserved = None
            after_reserved = None
            try:
                before_reserved = int(torch.cuda.memory_reserved() or 0)
            except Exception:
                before_reserved = None
            try:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                after_reserved = int(torch.cuda.memory_reserved() or 0)
            except Exception:
                after_reserved = None
            if isinstance(before_reserved, int) and isinstance(after_reserved, int):
                freed = max(0, before_reserved - after_reserved)
                mb = freed / (1024 ** 2)
                info_parts.append(f'GPU cache ~{mb:.1f} MB')
            else:
                info_parts.append('GPU cache καθαρίστηκε (χωρίς ακριβή μέτρηση)')
        else:
            info_parts.append('CUDA: όχι διαθέσιμη')
    except Exception:
        info_parts.append('CUDA cleanup: σφάλμα')
    try:
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            process = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(process, -1, -1)
            info_parts.append('WorkingSet trimmed')
    except Exception:
        pass
    base = ' · '.join(info_parts) if info_parts else 'χωρίς διαθέσιμες πληροφορίες'
    return f'{context} → {base}' if context else base


def format_html_summary(summary_text: str) -> str:
    if not summary_text:
        return '<div style="font-family:\'Consolas\',\'Courier New\',monospace; white-space:pre-wrap; font-size:11pt; color:#888;"><em>Δεν υπάρχουν διαθέσιμα αποτελέσματα ανάλυσης.</em></div>'
    lines = summary_text.splitlines()
    html_lines: list[str] = []
    heading_emojis = ('🧪', '📦', '🗂', '📊', '⚙️', '🏷', '⏱', '🎯', '💡')
    for raw_line in lines:
        line = raw_line.rstrip('\r')
        stripped = line.strip()
        if not stripped:
            html_lines.append('<div style="height:6px;"></div>')
            continue
        if set(stripped) <= {'=', '-', '·'}:
            html_lines.append('<hr style="border:0;border-top:1px solid #ccc;margin:8px 0;">')
            continue
        esc = html.escape(line)
        if stripped[0] in heading_emojis:
            html_lines.append(f'<p style="margin:4px 0 2px 0; font-weight:bold;">{esc}</p>')
        elif stripped.startswith('•') or stripped.startswith('-') or stripped.startswith('• '):
            html_lines.append(f'<p style="margin:1px 0 1px 22px;">{esc}</p>')
        else:
            html_lines.append(f'<p style="margin:1px 0;">{esc}</p>')
    return '<div style="font-family:\'Consolas\',\'Courier New\',monospace; white-space:pre-wrap; font-size:11pt; line-height:1.35;">' + '\n'.join(html_lines) + '</div>'


def open_file_externally(filepath: str) -> bool:
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', filepath], check=True)
        else:
            subprocess.run(['xdg-open', filepath], check=True)
    except Exception as e:
        safe_log_error('Failed to open file', e)
        return False
    return True


def open_folder_externally(folder_path: str) -> bool:
    try:
        if platform.system() == 'Windows':
            os.startfile(folder_path)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', folder_path], check=True)
        else:
            subprocess.run(['xdg-open', folder_path], check=True)
    except Exception as e:
        safe_log_error('Failed to open folder', e)
        return False
    return True
BASE_URL = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
API_KEY = os.getenv('GROQ_API_KEY', '').strip()
DEFAULT_MODEL = 'moonshotai/kimi-k2-instruct-0905'
CURRENT_LLM_MODEL = DEFAULT_MODEL
# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT για το AI Copilot LLM (Groq API)
# Καθορίζει τη συμπεριφορά του LLM για προτάσεις ρυθμίσεων εκπαίδευσης.
# Γνωρίζει: YOLO μοντέλα (detection + classification) + CNN torchvision μοντέλα.
# Σημαντικοί κανόνες:
#   - CNN: triton_enabled=false ΠΑΝΤΑ, optimizer=adam/adamw/sgd
#   - YOLO: τιμές compile_mode σε Ελληνικά ('Προεπιλογή', 'Μείωση επιβάρυνσης', ...)
#   - Παράγει 2 YAML blocks: gui_config + train_hyperparams
# ════════════════════════════════════════════════════════════════════════════════
TRAINING_COPILOT_SYSTEM_PROMPT = (
    '\nΕίσαι εξειδικευμένος βοηθός εκπαίδευσης μοντέλων (YOLO + CNN torchvision) στην εφαρμογή'
    ' "Models Manager Pro v4.0 Copilot".\n'
    'Λαμβάνεις πάντα ένα κείμενο με:\n'
    '- "ΤΡΕΧΟΥΣΕΣ ΡΥΘΜΙΣΕΙΣ" (τρέχον μοντέλο, dataset, εικόνα, epochs, patience, device κ.λπ.)\n'
    '- "ΠΕΡΙΒΑΛΛΟΝ HARDWARE" (CPU/GPU, μνήμη, CUDA, Triton, compile_mode)\n'
    '- "ΣΗΜΕΙΩΣΗ CNN ΜΟΝΤΕΛΑ" (αν το επιλεγμένο μοντέλο είναι CNN torchvision)\n'
    '- Πιθανές επιπλέον οδηγίες του χρήστη.\n\n'
    '### Κατηγορίες μοντέλων\n'
    'YOLO μοντέλα (Ultralytics): yolov5, yolov8, yolo11, yolo12 κ.λπ.\n'
    'CNN μοντέλα (torchvision): mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, resnet50, resnet101.\n'
    '- Τα CNN μοντέλα ΔΕΝ υποστηρίζουν Triton/TorchCompile (triton_enabled=false πάντα).\n'
    '- Τα CNN μοντέλα χρησιμοποιούν optimizer: adam/adamw/sgd (ΟΧΙ "auto").\n'
    '- Τα CNN μοντέλα απαιτούν dataset ταξινόμησης με δομή train/<class>/... val/<class>/...\n\n'
    'Στόχος σου είναι:\n'
    '1) Να δώσεις μια σύντομη επεξήγηση (2–4 προτάσεις στα Ελληνικά) για το ΤΙ προτείνεις και ΓΙΑΤΙ,\n'
    '   με έμφαση στη σταθερότητα, την ταχύτητα και την ποιότητα του μοντέλου.\n'
    '2) ΣΤΟ ΤΕΛΟΣ της απάντησης να παράγεις ΠΑΝΤΑ **δύο** YAML blocks μέσα σε ```yaml```:\n'
    '   - Το πρώτο block θα ονομάζεται **gui_config** και ρυθμίζει τα βασικά πεδία της φόρμας.\n'
    '   - Το δεύτερο block θα ονομάζεται **train_hyperparams** και ρυθμίζει τους hyperparameters.\n\n'
    '### Κανόνες για το πρώτο YAML block (gui_config)\n'
    'Το πρώτο μπλοκ πρέπει να έχει ΑΚΡΙΒΩΣ το αντικείμενο:\n'
    '```yaml\ngui_config:\n  model_name: ...\n  dataset_name: ...\n  image_size: ...\n'
    '  epochs: ...\n  patience: ...\n  device: ...\n```\n\n'
    '- Χρησιμοποίησε **ΜΟΝΟ** αυτά τα κλειδιά: `model_name`, `dataset_name`, `image_size`, `epochs`, `patience`, `device`.\n'
    '- Αν δεν θέλεις να αλλάξεις κάποια τιμή, μπορείς είτε να την παραλείψεις ΕΝΤΕΛΩΣ,\n'
    '  είτε να βάλεις την τρέχουσα τιμή όπως δίνεται στις "ΤΡΕΧΟΥΣΕΣ ΡΥΘΜΙΣΕΙΣ".\n'
    '- Για το `device` χρησιμοποίησε τιμές όπως `\'CPU\'`, `\'GPU\'`, `\'cuda\'`, `\'cpu\'`.\n'
    '- ΜΗΝ προσθέτεις άλλα κλειδιά στο gui_config.\n\n'
    '### Κανόνες για το δεύτερο YAML block (train_hyperparams)\n'
    '```yaml\ntrain_hyperparams:\n  batch: ...\n  optimizer: ...\n  lr0: ...\n  lrf: ...\n'
    '  momentum: ...\n  weight_decay: ...\n  warmup_epochs: ...\n  workers: ...\n'
    '  triton_enabled: ...\n  compile_mode: ...\n```\n\n'
    '- Για **CNN μοντέλα**: θέσε `triton_enabled: false` και `compile_mode: \'Προεπιλογή\'` ΠΑΝΤΑ.\n'
    '- Για **YOLO μοντέλα**: Το `triton_enabled` είναι boolean (`true`/`false`).\n'
    '- Για το `optimizer` σε YOLO χρησιμοποίησε: `SGD`, `Adam`, `AdamW`, ή `auto`.\n'
    '- Για το `optimizer` σε CNN χρησιμοποίησε: `adam`, `adamw`, ή `sgd`.\n\n'
    '#### Σχέση Triton και compile_mode (YOLO μόνο)\n'
    '- Για το `compile_mode` χρησιμοποίησε **ΑΠΟΚΛΕΙΣΤΙΚΑ** μία από τις παρακάτω τιμές:\n'
    '  - `\'Προεπιλογή\'` → torch.compile default.\n'
    '  - `\'Μείωση επιβάρυνσης\'` → mode `reduce-overhead`.\n'
    '  - `\'Μέγιστος αυτόματος συντονισμός\'` → mode `max-autotune`.\n\n'
    '### Συμπεριφορά που πρέπει να αποφεύγεις\n'
    '- ΜΗΝ εφευρίσκεις νέα ονόματα παραμέτρων ή καινούρια κλειδιά YAML.\n'
    '- ΜΗΝ αλλάζεις το format των YAML blocks.\n'
    '- ΜΗΝ προτείνεις Triton για CNN μοντέλα.\n\n'
    '### Τελική δομή απάντησης\n'
    '1) Σύντομη ανάλυση/εξήγηση στα Ελληνικά (2–4 προτάσεις).\n'
    '2) Ακριβώς δύο YAML blocks (gui_config + train_hyperparams).\n'
)
try:
    _LLM_CONFIG_PATH = ROOT_DIR / 'llm_settings.json'
except Exception:
    _LLM_CONFIG_PATH = Path('llm_settings.json')


def _load_llm_settings_from_file() -> None:
    global BASE_URL, API_KEY, CURRENT_LLM_MODEL
    try:
        if _LLM_CONFIG_PATH.is_file():
            with _LLM_CONFIG_PATH.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                base_url = data.get('base_url') or BASE_URL
                api_key = data.get('api_key') or API_KEY
                model = data.get('model') or CURRENT_LLM_MODEL
                BASE_URL = str(base_url).strip()
                API_KEY = str(api_key).strip()
                CURRENT_LLM_MODEL = str(model).strip() or DEFAULT_MODEL
    except Exception:
        pass


def _save_llm_settings_to_file() -> None:
    try:
        data = {'base_url': BASE_URL, 'api_key': API_KEY, 'model': CURRENT_LLM_MODEL or DEFAULT_MODEL}
        try:
            json_write(_LLM_CONFIG_PATH, data)
        except Exception:
            with _LLM_CONFIG_PATH.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_current_llm_model() -> str:
    global CURRENT_LLM_MODEL
    return CURRENT_LLM_MODEL or DEFAULT_MODEL


def configure_llm(base_url: str | None=None, api_key: str | None=None, model: str | None=None) -> None:
    global BASE_URL, API_KEY, CURRENT_LLM_MODEL
    if base_url:
        BASE_URL = base_url.strip()
    if api_key is not None:
        API_KEY = api_key.strip()
    if model:
        CURRENT_LLM_MODEL = model.strip() or DEFAULT_MODEL
    _save_llm_settings_to_file()
_load_llm_settings_from_file()


def has_valid_groq_api_key() -> bool:
    key = (API_KEY or '').strip()
    if not key:
        return False
    placeholder_tokens = {'YOUR_GROQ_API_KEY_HERE', 'YOUR_API_KEY_HERE', 'YOUR_GROQ_KEY_HERE'}
    upper = key.upper()
    for token in placeholder_tokens:
        if upper == token.upper():
            return False
    return True
WINDOWS_11_LIGHT_STYLE = '\nQWidget {\n    background-color: #f3f3f3;\n    color: #202020;\n    font-family: "Segoe UI", Arial, sans-serif;\n    font-size: 13pt;\n}\nQLabel {\n    background-color: transparent;\n    border: none;\n}\nQLineEdit,\nQPlainTextEdit,\nQTextEdit,\nQComboBox,\nQSpinBox,\nQDoubleSpinBox {\n    background-color: #f3f3f3;\n}\nQMainWindow {\n    background-color: #f3f3f3;\n}\nQTabWidget::pane {\n    border: 1px solid #7fc7ff;\n    border-radius: 8px;\n    background: #f3f3f3;\n}\nQTabBar::tab {\n    background: #e0e0e0;\n    color: #202020;\n    padding: 10px 20px;\n    min-height: 36px;\n    max-height: 36px;\n    border-top-left-radius: 8px;\n    border-top-right-radius: 8px;\n    margin-right: 3px;\n    font-weight: 500;\n}\nQTabBar::tab:hover {\n    background: #f0f7ff;\n    color: #0078d4;\n}\nQTabBar::tab:selected {\n    background: #ffffff;\n    color: #0078d4;\n    font-weight: 700;\n    border-bottom: 2px solid #0078d4;\n}\nQGroupBox {\n    background-color: #f3f3f3;\n    border: 1px solid #7fc7ff;\n    border-radius: 10px;\n    margin-top: 14px;\n    padding: 14px 10px 10px 10px;\n}\nQGroupBox#FeaturesGroup {\n    background-color: transparent;\n}\nQWidget#FeaturesInner {\n    background-color: transparent;\n}\nQGroupBox::title {\n    subcontrol-origin: margin;\n    left: 12px;\n    padding: 0 6px;\n    color: #0050a0;\n    font-weight: 700;\n    font-size: 12pt;\n}\nQFrame#Card {\n    background: #f3f3f3;\n    border: 1px solid #7fc7ff;\n    border-radius: 12px;\n}\nQFrame#ResourceCard {\n    background: #E8FFE8;\n    border: 1px solid #7fc7ff;\n    border-radius: 12px;\n}\nQFrame#HeaderFrame {\n    background: #e0e0e0;\n    border: 1px solid #7fc7ff;\n    border-radius: 12px;\n    padding: 8px;\n}\nQPushButton {\n    background-color: #0078d4;\n    color: #ffffff;\n    border: 1px solid #006cbe;\n    border-radius: 8px;\n    padding: 8px 18px;\n    font-weight: 600;\n    font-size: 12.5pt;\n    min-height: 32px;\n}\nQPushButton#RefreshButton {\n    padding: 3px 6px;\n    min-height: 24px;\n    font-size: 11pt;\n}\nQPushButton:hover {\n    background-color: #0b8cf0;\n    border-color: #0a7ad0;\n}\nQPushButton:pressed {\n    background-color: #005a9e;\n    border-color: #004f8f;\n}\nQPushButton:disabled {\n    background-color: #c8c8c8;\n    color: #7a7a7a;\n    border-color: #c8c8c8;\n}\nQPushButton#ThemeButton, #DashboardButton {\n    min-height: 28px;\n    padding: 7px 18px;\n    font-weight: 600;\n}\nQComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {\n    background: #ffffff;\n    color: #202020;\n    border: 1.5px solid #c8c8c8;\n    border-radius: 7px;\n    padding: 5px 7px;\n}\nQComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,\nQLineEdit:hover, QTextEdit:hover {\n    border-color: #7fc7ff;\n    background-color: #ffffff;\n}\nQComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,\nQLineEdit:focus {\n    border-color: #0078d4;\n}\nQComboBox::drop-down {\n    border: none;\n}\nQCheckBox, QRadioButton {\n    spacing: 8px;\n    font-size: 13pt;\n}\nQCheckBox::indicator, QRadioButton::indicator {\n    width: 18px;\n    height: 18px;\n}\nQCheckBox::indicator {\n    width: 18px;\n    height: 18px;\n    border-radius: 4px;\n    border: 2px solid #2196f3;\n    background-color: #ffffff;\n}\nQCheckBox::indicator:checked {\n    background-color: #2196f3;\n    border: 2px solid #2196f3;\n}\nQCheckBox::indicator:hover {\n    border-color: #4dabf7;\n}\nQCheckBox:hover, QRadioButton:hover {\n    background-color: #e8f3ff;\n    border-radius: 5px;\n}\nQRadioButton::indicator {\n    border-radius: 9px;\n    border: 1px solid #c6c6c6;\n    background: #ffffff;\n}\nQRadioButton::indicator:checked {\n    background-color: #0078d4;\n    border-color: #006cbe;\n}\nQScrollBar:vertical {\n    background: #ebebeb;\n    width: 10px;\n    border-radius: 5px;\n    margin: 0;\n}\nQScrollBar::handle:vertical {\n    background: #b0b0b0;\n    border-radius: 5px;\n    min-height: 28px;\n}\nQScrollBar::handle:vertical:hover {\n    background: #0078d4;\n}\nQScrollBar::add-line:vertical,\nQScrollBar::sub-line:vertical { height: 0; }\nQScrollBar:horizontal {\n    background: #ebebeb;\n    height: 10px;\n    border-radius: 5px;\n    margin: 0;\n}\nQScrollBar::handle:horizontal {\n    background: #b0b0b0;\n    border-radius: 5px;\n    min-width: 28px;\n}\nQScrollBar::handle:horizontal:hover {\n    background: #0078d4;\n}\nQScrollBar::add-line:horizontal,\nQScrollBar::sub-line:horizontal { width: 0; }\nQScrollArea {\n    background: transparent;\n    border: none;\n}\nQListWidget, QTreeWidget, QTableWidget {\n    background-color: #f5f5f5;\n    alternate-background-color: #eaeaea;\n    border: 1px solid #7fc7ff;\n    border-radius: 8px;\n    gridline-color: #e0e0e0;\n    outline: none;\n}\nQTableWidget::item, QListWidget::item, QTreeWidget::item {\n    padding: 4px 6px;\n}\nQTableWidget::item:selected, QListWidget::item:selected,\nQTreeWidget::item:selected {\n    background-color: #cce4ff;\n    color: #202020;\n}\nQTableWidget::item:hover, QListWidget::item:hover,\nQTreeWidget::item:hover {\n    background-color: #e3f0ff;\n}\nQHeaderView::section {\n    background-color: #e8e8e8;\n    color: #202020;\n    padding: 7px 8px;\n    border: none;\n    border-right: 1px solid #d0d0d0;\n    border-bottom: 1px solid #c0c0c0;\n    font-weight: 700;\n    font-size: 12pt;\n}\nQProgressBar {\n    border: 1px solid #7fc7ff;\n    border-radius: 8px;\n    text-align: center;\n    background: #ffffff;\n    color: #202020;\n    font-weight: 600;\n    min-height: 20px;\n}\nQProgressBar::chunk {\n    background-color: #0078d4;\n    border-radius: 8px;\n}\nQSlider::groove:horizontal {\n    height: 6px;\n    background: #d0d0d0;\n    border-radius: 3px;\n}\nQSlider::handle:horizontal {\n    background: #0078d4;\n    width: 18px;\n    height: 18px;\n    margin: -6px 0;\n    border-radius: 9px;\n    border: 2px solid #ffffff;\n}\nQSlider::sub-page:horizontal {\n    background: #0078d4;\n    border-radius: 3px;\n}\nQToolTip {\n    background: #1a1a1a;\n    color: #f0f0f0;\n    border: 1px solid #7fc7ff;\n    border-radius: 6px;\n    padding: 5px 10px;\n    font-size: 11pt;\n}\nQSplitter::handle {\n    background: #c8c8c8;\n}\nQSplitter::handle:hover {\n    background: #0078d4;\n}\nQSplitter::handle:horizontal { width: 3px; }\nQSplitter::handle:vertical   { height: 3px; }\nQLabel#Title {\n    font-size: 23pt;\n    font-weight: 700;\n    color: #202020;\n    background-color: #e0e0e0;\n}\nQLabel#Subtitle {\n    font-size: 13pt;\n    color: #606060;\n    background-color: #e0e0e0;\n}\nQLabel#CardTitle {\n    background-color: #f3f3f3;\n    color: #202020;\n    font-weight: 700;\n    padding: 4px 8px;\n    border-radius: 6px;\n}\nQLabel#ResourceTitle {\n    background-color: #E8FFE8;\n    color: #202020;\n    font-weight: 700;\n    padding: 4px 8px;\n    border-radius: 6px;\n}\nQLabel#ResourceLabel {\n    font-weight: 600;\n    color: #202020;\n}\nQLabel#FooterLabel {\n    font-weight: 600;\n    color: #000000;\n}\nQTextEdit#LogOutput {\n    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;\n    background-color: #dadada;\n    color: #000000;\n    font-size: 11pt;\n    border-radius: 10px;\n    border: 1px solid #7fc7ff;\n    padding: 10px;\n}\nQTextEdit#BenchmarkLogOutput, QTextEdit#CameraBenchmarkLogOutput {\n    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;\n    background-color: #dadada;\n    color: #000000;\n    font-size: 11pt;\n    border-radius: 10px;\n    border: none;\n    padding: 10px;\n}\nQFrame[panelFrame="true"] {\n    border: 1px solid #7fc7ff;\n    border-radius: 8px;\n}\nQFrame[frameShape="4"],\nQFrame[frameShape="5"] {\n    border: none;\n    border-radius: 0px;\n}\n'
WINDOWS_11_DARK_STYLE = '\nQWidget {\n    background-color: #121212;\n    color: #f5f5f5;\n    font-family: "Segoe UI", Arial, sans-serif;\n    font-size: 13pt;\n}\nQLabel {\n    background-color: transparent;\n    border: none;\n}\nQLineEdit,\nQPlainTextEdit,\nQTextEdit,\nQComboBox,\nQSpinBox,\nQDoubleSpinBox {\n    background-color: #1c1c1c;\n}\nQMainWindow {\n    background-color: #121212;\n}\nQTabWidget::pane {\n    border: 1px solid #333333;\n    border-radius: 8px;\n    background: #181818;\n}\nQTabBar::tab {\n    background: #181818;\n    color: #c0c0c0;\n    padding: 10px 20px;\n    min-height: 36px;\n    max-height: 36px;\n    border-top-left-radius: 8px;\n    border-top-right-radius: 8px;\n    margin-right: 3px;\n    font-weight: 500;\n}\nQTabBar::tab:hover {\n    background: #222222;\n    color: #9fddff;\n}\nQTabBar::tab:selected {\n    background: #262626;\n    color: #9fddff;\n    font-weight: 700;\n    border-bottom: 2px solid #9fddff;\n}\nQGroupBox {\n    background-color: #1c1c1c;\n    border: 1px solid #9fddff;\n    border-radius: 10px;\n    margin-top: 14px;\n    padding: 14px 10px 10px 10px;\n    color: #f5f5f5;\n}\nQGroupBox#FeaturesGroup {\n    background-color: transparent;\n}\nQWidget#FeaturesInner {\n    background-color: transparent;\n}\nQGroupBox::title {\n    subcontrol-origin: margin;\n    left: 12px;\n    padding: 0 6px;\n    color: #9fddff;\n    font-weight: 700;\n    font-size: 12pt;\n}\nQFrame#Card {\n    background: #1c1c1c;\n    border: 1px solid #9fddff;\n    border-radius: 12px;\n}\nQFrame#ResourceCard {\n    background: #101c10;\n    border: 1px solid #9fddff;\n    border-radius: 12px;\n}\nQFrame#ResourceCard QLabel {\n    color: #7CFC00;\n}\nQFrame#HeaderFrame {\n    background: #222222;\n    border: 1px solid #9fddff;\n    border-radius: 12px;\n    padding: 8px;\n}\nQPushButton {\n    background-color: #3a3a3a;\n    color: #f5f5f5;\n    border: 1px solid #4a4a4a;\n    border-radius: 8px;\n    padding: 8px 18px;\n    font-weight: 600;\n    font-size: 12.5pt;\n    min-height: 32px;\n}\nQPushButton#RefreshButton {\n    padding: 3px 6px;\n    min-height: 24px;\n    font-size: 11pt;\n}\nQPushButton:hover {\n    background-color: #4a4a4a;\n    border-color: #9fddff;\n    color: #ffffff;\n}\nQPushButton:pressed {\n    background-color: #292929;\n    border-color: #666666;\n}\nQPushButton:disabled {\n    background-color: #2a2a2a;\n    color: #555555;\n    border-color: #2a2a2a;\n}\nQPushButton#ThemeButton, #DashboardButton {\n    min-height: 28px;\n    padding: 7px 18px;\n    font-weight: 600;\n}\nQComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {\n    background: #1f1f1f;\n    color: #f5f5f5;\n    border: 1.5px solid #444444;\n    border-radius: 7px;\n    padding: 5px 7px;\n}\nQComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,\nQLineEdit:hover, QTextEdit:hover {\n    border-color: #9fddff;\n    background-color: #262626;\n}\nQComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,\nQLineEdit:focus {\n    border-color: #9fddff;\n}\nQComboBox::drop-down {\n    border: none;\n}\nQComboBox QAbstractItemView {\n    background: #1f1f1f;\n    border: 1px solid #444444;\n    border-radius: 6px;\n    selection-background-color: #333333;\n    selection-color: #f5f5f5;\n    outline: none;\n}\nQCheckBox, QRadioButton {\n    spacing: 8px;\n    font-size: 13pt;\n}\nQCheckBox::indicator {\n    width: 18px;\n    height: 18px;\n    border-radius: 4px;\n    border: 2px solid #2196f3;\n    background-color: #1e1e1e;\n}\nQCheckBox::indicator:checked {\n    background-color: #2196f3;\n    border: 2px solid #2196f3;\n}\nQCheckBox::indicator:hover {\n    border-color: #4dabf7;\n}\nQCheckBox:hover, QRadioButton:hover {\n    background-color: #262626;\n    border-radius: 5px;\n}\nQRadioButton::indicator {\n    width: 18px;\n    height: 18px;\n    border-radius: 9px;\n    border: 1px solid #3a3a3a;\n    background: #262626;\n}\nQRadioButton::indicator:checked {\n    background-color: #0a84ff;\n    border-color: #0a84ff;\n}\nQScrollBar:vertical {\n    background: #1a1a1a;\n    width: 10px;\n    border-radius: 5px;\n    margin: 0;\n}\nQScrollBar::handle:vertical {\n    background: #3a3a3a;\n    border-radius: 5px;\n    min-height: 28px;\n}\nQScrollBar::handle:vertical:hover {\n    background: #9fddff;\n}\nQScrollBar::add-line:vertical,\nQScrollBar::sub-line:vertical { height: 0; }\nQScrollBar:horizontal {\n    background: #1a1a1a;\n    height: 10px;\n    border-radius: 5px;\n    margin: 0;\n}\nQScrollBar::handle:horizontal {\n    background: #3a3a3a;\n    border-radius: 5px;\n    min-width: 28px;\n}\nQScrollBar::handle:horizontal:hover {\n    background: #9fddff;\n}\nQScrollBar::add-line:horizontal,\nQScrollBar::sub-line:horizontal { width: 0; }\nQScrollArea {\n    background: transparent;\n    border: none;\n}\nQListWidget, QTreeWidget, QTableWidget {\n    background-color: #1c1c1c;\n    color: #f5f5f5;\n    alternate-background-color: #222222;\n    border: 1px solid #9fddff;\n    border-radius: 8px;\n    gridline-color: #2a2a2a;\n    outline: none;\n}\nQTableWidget::item, QListWidget::item, QTreeWidget::item {\n    padding: 4px 6px;\n}\nQTableWidget::item:selected, QListWidget::item:selected,\nQTreeWidget::item:selected {\n    background-color: #2a3a4a;\n    color: #ffffff;\n}\nQTableWidget::item:hover, QListWidget::item:hover,\nQTreeWidget::item:hover {\n    background-color: #242424;\n}\nQHeaderView::section {\n    background-color: #202020;\n    color: #9fddff;\n    padding: 7px 8px;\n    border: none;\n    border-right: 1px solid #2a2a2a;\n    border-bottom: 1px solid #333333;\n    font-weight: 700;\n    font-size: 12pt;\n}\nQProgressBar {\n    border: 1px solid #9fddff;\n    border-radius: 8px;\n    text-align: center;\n    background: #181818;\n    color: #f5f5f5;\n    font-weight: 600;\n    min-height: 20px;\n}\nQProgressBar::chunk {\n    background-color: #0a84ff;\n    border-radius: 8px;\n}\nQSlider::groove:horizontal {\n    height: 6px;\n    background: #333333;\n    border-radius: 3px;\n}\nQSlider::handle:horizontal {\n    background: #0a84ff;\n    width: 18px;\n    height: 18px;\n    margin: -6px 0;\n    border-radius: 9px;\n    border: 2px solid #121212;\n}\nQSlider::sub-page:horizontal {\n    background: #0a84ff;\n    border-radius: 3px;\n}\nQToolTip {\n    background: #1e1e1e;\n    color: #f5f5f5;\n    border: 1px solid #9fddff;\n    border-radius: 6px;\n    padding: 5px 10px;\n    font-size: 11pt;\n}\nQSplitter::handle {\n    background: #333333;\n}\nQSplitter::handle:hover {\n    background: #9fddff;\n}\nQSplitter::handle:horizontal { width: 3px; }\nQSplitter::handle:vertical   { height: 3px; }\nQLabel#Title {\n    font-size: 23pt;\n    font-weight: 700;\n    color: #f5f5f5;\n    background-color: #222222;\n}\nQLabel#Subtitle {\n    font-size: 13pt;\n    color: #dddddd;\n    background-color: #222222;\n}\nQLabel#CardTitle {\n    background-color: #1c1c1c;\n    color: #f5f5f5;\n    font-weight: 700;\n    padding: 4px 8px;\n    border-radius: 6px;\n}\nQLabel#ResourceTitle {\n    background-color: #101c10;\n    color: #7CFC00;\n    font-weight: 700;\n    padding: 4px 8px;\n    border-radius: 6px;\n}\nQLabel#ResourceLabel {\n    font-weight: 600;\n    color: #7CFC00;\n}\nQLabel#FooterLabel {\n    font-weight: 600;\n    color: #f5f5f5;\n}\nQTextEdit#LogOutput {\n    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;\n    background-color: #181818;\n    color: #ffffff;\n    font-size: 11pt;\n    border-radius: 10px;\n    border: 1px solid #9fddff;\n    padding: 10px;\n}\nQTextEdit#BenchmarkLogOutput, QTextEdit#CameraBenchmarkLogOutput {\n    font-family: "Consolas", "Cascadia Code", "Courier New", monospace;\n    background-color: #181818;\n    color: #ffffff;\n    font-size: 11pt;\n    border-radius: 10px;\n    border: none;\n    padding: 10px;\n}\nQFrame[panelFrame="true"] {\n    border: 1px solid #9fddff;\n    border-radius: 8px;\n}\nQFrame[frameShape="4"],\nQFrame[frameShape="5"] {\n    border: none;\n    border-radius: 0px;\n}\nQListWidget::item:selected,\nQTreeWidget::item:selected,\nQTableWidget::item:selected {\n    background-color: #2a3a4a;\n    color: #ffffff;\n}\nQListWidget::item:hover,\nQTreeWidget::item:hover,\nQTableWidget::item:hover {\n    background-color: #242424;\n}\n'



def apply_light_theme_to_window(window=None) -> None:
    from PySide6.QtWidgets import QApplication, QTextEdit
    from PySide6.QtGui import QFont
    try:
        QApplication.setStyle('Fusion')
    except Exception:
        pass
    if window is not None:
        try:
            window.setStyleSheet(WINDOWS_11_LIGHT_STYLE)
        except Exception:
            pass
        try:
            console_font = QFont('Consolas')
            console_font.setStyleHint(QFont.StyleHint.Monospace)
            console_font.setPointSize(11)
            for _name in ('LogOutput', 'BenchmarkLogOutput', 'CameraBenchmarkLogOutput'):
                for log_widget in window.findChildren(QTextEdit, _name):
                    log_widget.setFont(console_font)
        except Exception:
            pass
    set_log_colors_for_theme('light')


class LLMClient:
    _NO_KEY_ERROR = (
        "Δεν έχει οριστεί API key για το Groq API.\n"
        "Επιλογές:\n"
        "  1. Ορίστε τη μεταβλητή GROQ_API_KEY στο περιβάλλον.\n"
        "  2. Περάστε api_key= στον LLMClient().\n"
        "  3. Χρησιμοποιήστε το κουμπί «⚙️ Ρυθμίσεις LLM» στην καρτέλα Copilot."
    )

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or API_KEY
        if not self.api_key or self.api_key == "YOUR_GROQ_API_KEY_HERE":
            raise RuntimeError(self._NO_KEY_ERROR)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat( self, system_prompt: str, user_message: str, model: str = DEFAULT_MODEL, temperature: float = 0.2,) -> str:
        messages = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_message},]
        response = self.client.chat.completions.create( model=model, messages=messages, temperature=temperature, stream=False,)
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError):
            return str(response)


def add_dashboard_and_theme_buttons(owner, layout, go_to_dashboard_callback, toggle_theme_callback):
    dashboard_button = QPushButton('🏠 Πίνακας Ελέγχου')
    dashboard_button.setObjectName('DashboardButton')
    dashboard_button.setFixedHeight(24)
    try:
        dashboard_button.clicked.connect(go_to_dashboard_callback)
    except Exception:
        pass
    theme_button = QPushButton('Light/Dark')
    theme_button.setObjectName('ThemeButton')
    theme_button.setFixedHeight(24)
    try:
        theme_button.clicked.connect(toggle_theme_callback)
    except Exception:
        pass
    layout.addWidget(dashboard_button)
    layout.addWidget(theme_button)
    try:
        owner.dashboard_button = dashboard_button
        owner.theme_button = theme_button
    except Exception:
        pass
    return (dashboard_button, theme_button)


def add_blue_separator(outer_layout) -> None:
    top_separator = QFrame()
    top_separator.setFrameShape(QFrame.Shape.HLine)
    top_separator.setFrameShadow(QFrame.Shadow.Plain)
    top_separator.setStyleSheet('color: #7fc7ff; background-color: #7fc7ff; max-height: 1px; min-height: 1px;')
    outer_layout.addWidget(top_separator)
    return top_separator


class JobManager(QObject):
    job_started = Signal(str)
    job_finished = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self._queue: collections.deque[tuple] = collections.deque()
        self._current: tuple | None = None

    def is_running(self) -> bool:
        return self._current is not None

    def current_name(self) -> str:
        return self._current[0] if self._current else ""

    def try_start(self, name: str, start_cb: Callable, cancel_cb: Callable | None = None) -> bool:
        if self._current is None:
            self._current = (name, start_cb, cancel_cb)
            try:
                self.job_started.emit(name)
            except Exception:
                pass
            try:
                start_cb()
            except Exception:
                self._current = None
                raise
            return True
        self._queue.append((name, start_cb, cancel_cb))
        return False

    def done(self, ok: bool = True) -> None:
        name = self.current_name()
        self._current = None
        try:
            self.job_finished.emit(name, ok)
        except Exception:
            pass
        if self._queue:
            n, cb, cc = self._queue.popleft()
            self.try_start(n, cb, cc)

    def cancel_current(self) -> None:
        if not self._current:
            return
        _, _, cancel_cb = self._current
        if callable(cancel_cb):
            try:
                cancel_cb()
            except Exception:
                pass
JOB_MANAGER = JobManager()


class LogRingBuffer:

    def __init__(self, max_lines: int = 5000) -> None:
        self.max_lines = int(max_lines)
        self._lines: collections.deque[str] = collections.deque(maxlen=self.max_lines)

    def append(self, line: str) -> None:
        if line is None:
            return
        self._lines.append(str(line))

    def get(self, contains: str = "") -> list[str]:
        if not contains:
            return list(self._lines)
        needle = contains.lower()
        return [ln for ln in self._lines if needle in ln.lower()]
GLOBAL_LOG_BUFFER = LogRingBuffer(max_lines=8000)


def generate_diagnostics_zip(zip_path: str) -> str:
    import zipfile
    try:
        data = collect_diagnostics_data()
        report_json = diagnostics_to_json(data)
        report_txt = diagnostics_to_text(data)
        pip_freeze = str(data.get("pip_freeze", ""))
    except Exception as e:
        fallback: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "cwd": str(Path.cwd()),
            "error": str(e),
        }
        report_json = json.dumps(fallback, indent=2, ensure_ascii=False)
        report_txt = "MODELS MANAGER PRO – DIAGNOSTICS\n" + report_json
        pip_freeze = ""
    p = Path(zip_path).expanduser().resolve()
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("diagnostics/report.json", report_json)
        z.writestr("diagnostics/report.txt", report_txt)
        if pip_freeze:
            z.writestr("diagnostics/pip_freeze.txt", pip_freeze)
    return str(p)


class TabNavigationMixin:

    def toggle_theme(self):
        main_window = self.window()
        if hasattr(main_window, 'toggle_theme_global'):
            main_window.toggle_theme_global()

    def go_to_dashboard(self):
        main_window = self.window()
        try:
            tabs = getattr(main_window, 'tabs', None)
            if tabs is not None:
                tabs.setCurrentIndex(0)
        except Exception:
            pass


class BenchmarkUIHelpersMixin:

    def append_log(self, html_text: str):
        self.log_edit.append(html_text)

    @staticmethod

    def _parse_imgsz_from_name(name: str):
        if not name:
            return None
        m = re.search(r'imgsz(\d+)', name, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        m2 = re.search(r'_(\d+)$', name)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                return None

    def on_worker_results(self, results: list):
        self.results_table.setRowCount(0)
        for backend, fps, ms_per_image in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            backend_item = QTableWidgetItem(backend_pretty_name(backend))
            fps_item = QTableWidgetItem(f'{fps:.2f}')
            ms_item = QTableWidgetItem(f'{ms_per_image:.2f}')
            self.results_table.setItem(row, 0, backend_item)
            self.results_table.setItem(row, 1, fps_item)
            self.results_table.setItem(row, 2, ms_item)


class LogEmitMixin:

    def _cprint(self, text: str, color: str=Colors.CYAN, bold: bool=False, underline: bool=False):
        self.log.emit(format_html_log(text, color, bold, underline))

    def _log_exc(self, context: str, exc: BaseException, extra: dict | None = None) -> None:
        """
        Κεντρική μέθοδος καταγραφής σφαλμάτων με πλήρες traceback στο Log.
        Χρησιμοποιείται από όλους τους workers για ομοιόμορφη αναφορά σφαλμάτων.

        Args:
            context:  Περιγραφή πού συνέβη το σφάλμα (π.χ. 'load_model', 'inference')
            exc:      Η εξαίρεση που πιάστηκε
            extra:    Προαιρετικό dict με επιπλέον πληροφορίες (model_path, backend κ.λπ.)
        """
        import traceback as _tb
        tb_str = _tb.format_exc()
        lines = [
            f'━━━ ❌ ΣΦΑΛΜΑ: {context} ━━━',
            f'   Τύπος:    {type(exc).__name__}',
            f'   Μήνυμα:   {exc}',
        ]
        if extra:
            for k, v in extra.items():
                lines.append(f'   {k}: {v}')
        lines.append('   Traceback:')
        for tb_line in tb_str.strip().splitlines():
            lines.append(f'   {tb_line}')
        lines.append('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        full_msg = '\n'.join(lines)
        try:
            self._cprint(full_msg, Colors.RED, bold=False)
        except Exception:
            pass
        # Επίσης εκπέμπουμε error signal αν υπάρχει
        try:
            short = f'{context}: {type(exc).__name__}: {exc}'
            if hasattr(self, 'error') and callable(getattr(self.error, 'emit', None)):
                self.error.emit(short)
        except Exception:
            pass


class StoppableMixin:
    _is_running: bool = False

    def stop(self) -> None:
        self._is_running = False


def trt_purge_cache_for_model(model_path: Path) -> None:
    try:
        base = Path(model_path).resolve().parent
        cache_dir = base / MMPRO_CACHE_DIR_NAME / 'tensorrt'
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception:
        pass
"""Export worker.
Worker που κάνει export μοντέλων σε ONNX/TensorRT/NCNN και γράφει meta για σωστή χρήση μετά.
"""


# ════════════════════════════════════════════════════════════════════════════════
# ExportWorker – QObject worker για εξαγωγή μοντέλου σε ONNX / TensorRT / NCNN
# ════════════════════════════════════════════════════════════════════════════════
# Εκτελείται σε ξεχωριστό QThread για να μην κλειδώνει το GUI.
# Σήματα: log(str), finished(), error(str)
# Μέθοδοι εξαγωγής:
#   export_onnx()    → .onnx  (YOLO ή CNN torchvision)
#   export_ncnn()    → _ncnn_model/ (μόνο YOLO)
#   export_tensorrt()→ .engine (μόνο YOLO + GPU + TensorRT installed)
# ════════════════════════════════════════════════════════════════════════════════
class ExportWorker(QObject, LogEmitMixin):
    finished = Signal()
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int, str)

    def __init__( self, model_path, imgsz: int, export_format: str, parent=None, overwrite: bool = False,) -> None:
        super().__init__(parent)
        self.model_path = Path(model_path)
        self.imgsz = int(imgsz)
        self.export_format = str(export_format).lower().strip()
        self.overwrite = bool(overwrite)

    def _log_html(self, text: str, color=None, bold: bool=False):
        try:
            self.log.emit(format_html_log(str(text), color, bold=bold))
        except Exception:
            pass

    def _log_export_environment(self) -> None:
        py_ver = sys.version.split()[0]
        try:
            import torch as _torch
            torch_ver = getattr(_torch, "__version__", "unknown")
            cuda_available = _torch.cuda.is_available()
        except ImportError:
            torch_ver = "unknown"
            cuda_available = False
            _torch = None
        try:
            import ultralytics as _ultra
            ultra_ver = getattr(_ultra, "__version__", "unknown")
        except ImportError:
            ultra_ver = "unknown"
        self._log_html("=== ΠΕΡΙΒΑΛΛΟΝ ΕΞΑΓΩΓΗΣ ΜΟΝΤΕΛΟΥ ===", Colors.HEADER, bold=True)
        self._log_html( f"Python: {py_ver} | PyTorch: {torch_ver} | Ultralytics: {ultra_ver}", Colors.CYAN,)
        if cuda_available and _torch is not None:
            try:
                dev_name = _torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "CUDA GPU"
            self._log_html(f"🟢 Διαθέσιμη GPU (CUDA): {dev_name}", Colors.GREEN)
        else:
            self._log_html( "⚠️ Δεν βρέθηκε διαθέσιμη GPU (CUDA). Χρήση CPU μόνο.", Colors.YELLOW,)

    def _log_process_tree(self) -> None:
        try:
            selected_formats: list[str] = []
            if getattr(self, 'export_format', '') == 'ncnn':
                selected_formats = ['ncnn']
            elif getattr(self, 'export_format', '') == 'onnx':
                selected_formats = ['onnx']
            elif getattr(self, 'export_format', '') == 'tensorrt':
                selected_formats = ['tensorrt']
            else:
                fmt = getattr(self, 'export_format', '')
                if fmt:
                    selected_formats = [fmt]
            fmt_titles = {'ncnn': 'NCNN (.param + .bin)', 'onnx': 'ONNX (.onnx)', 'tensorrt': 'TensorRT Engine (.engine)', 'cnn_onnx': 'CNN → ONNX (torchvision)'}
            self._log_html('🌳 Δομή λογικής ροής εξαγωγής', Colors.CYAN, bold=True)
            self._log_html('└─ 🧠 Κύρια διαδικασία: Export (GUI)', Colors.CYAN, bold=False)
            self._log_html('   ├─ 📁 Έλεγχος διαδρομής & τύπου μοντέλου (.pt)', Colors.CYAN, bold=False)
            self._log_html('   ├─ 🧪 Έλεγχος περιβάλλοντος Python / PyTorch / Ultralytics / CUDA', Colors.CYAN, bold=False)
            self._log_html('   ├─ 🧹 Προετοιμασία μνήμης (GC / CUDA cache)', Colors.CYAN, bold=False)
            for i, fmt in enumerate(selected_formats):
                is_last = i == len(selected_formats) - 1
                branch = '└─' if is_last else '├─'
                if fmt == 'ncnn':
                    icon = '📦'
                elif fmt == 'onnx':
                    icon = '📦'
                else:
                    icon = '🔥'
                title = fmt_titles.get(fmt, fmt.upper())
                line = f'   {branch} {icon} Εξαγωγή σε {title}'
                self._log_html(line, Colors.CYAN, bold=False)
        except Exception:
            pass

    def run(self):
        import time, gc, traceback
        try:
            time.sleep(1.0)
            if not self.model_path.exists():
                msg = f'Το μοντέλο δεν βρέθηκε: {self.model_path}'
                self._log_html(msg, Colors.RED, bold=True)
                self.error.emit(msg)
                return
            # ── CNN ONNX export ──────────────────────────────────────────────
            if self.export_format == 'cnn_onnx' or _is_cnn_path(self.model_path):
                try:
                    self._log_export_environment()
                    self._log_process_tree()
                except Exception:
                    pass
                self._export_cnn_onnx()
                return
            # ── YOLO exports (NCNN / ONNX / TensorRT) ───────────────────────
            if self.model_path.suffix.lower() != '.pt':
                msg = 'Η εξαγωγή NCNN/ONNX υποστηρίζεται μόνο από PyTorch (.pt) μοντέλα.'
                self._log_html(msg, Colors.RED, bold=True)
                self.error.emit(msg)
                return
            try:
                self._log_export_environment()
                self._log_process_tree()
            except Exception:
                pass
            if self.export_format == 'ncnn':
                self._export_ncnn_subprocess()
            elif self.export_format == 'onnx':
                self._export_onnx_subprocess()
            elif self.export_format == 'tensorrt':
                ok = self.export_tensorrt()
                if not ok:
                    msg = 'Αποτυχία εξαγωγής TensorRT (δες λεπτομέρειες στο Export Log).'
                    try:
                        self.error.emit(msg)
                    except Exception:
                        pass
            else:
                msg = f'Άγνωστο format εξαγωγής: {self.export_format}'
                self._log_html(msg, Colors.RED, bold=True)
                try:
                    self.error.emit(msg)
                except Exception:
                    pass
            try:
                info = perform_smart_memory_cleanup('Μετά την εξαγωγή μοντέλου')
                self._log_html(f'🧠 Καθαρισμός μνήμης: {info}', Colors.MAGENTA, bold=False)
            except Exception:
                self._log_html('⚠️ Αποτυχία καθαρισμού μνήμης (export)', Colors.YELLOW, bold=False)
        except Exception as e:
            try:
                tb = traceback.format_exc()
            except Exception:
                tb = ''
            self._log_html(f'Απρόσμενο σφάλμα εξαγωγής: {e}', Colors.RED, bold=True)
            if tb:
                self._log_html(tb, Colors.RED, bold=False)
            try:
                self._log_exc('Εξαγωγή μοντέλου', e, extra={
                'Μοντέλο': str(getattr(self, 'model_path', '?')),
                'Μορφή':   str(getattr(self, 'export_format', '?')),
            })
            except Exception:
                pass
        finally:
            try:
                self.finished.emit()
            except Exception:
                pass

    def _run_subprocess(self, script: str, args: list, label: str, log_file=None, show_success_output: bool = False):
        ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
        verbose_mode = _env_bool("MM_PRO_VERBOSE_EXPORT", False)
        if getattr(sys, "frozen", False):
            try:
                self._log_html( f"ℹ️ (NCNN/ONNX) Υπο-διεργασία {label}: χρήση εξωτερικού Python interpreter (frozen .exe).", Colors.CYAN, bold=False,)
            except Exception:
                pass
            candidates = ["py", "python", "python3"] if sys.platform.startswith("win") else ["python3", "python"]
            python_exe = None
            for prog in candidates:
                if shutil.which(prog):
                    python_exe = prog
                    break
            if python_exe is None:
                from io import StringIO
                from contextlib import redirect_stdout, redirect_stderr
                try:
                    self._log_html( f"⚠️ Δεν βρέθηκε external Python στο PATH. Εκτέλεση {label} in-process (frozen mode).", Colors.YELLOW, bold=True,)
                except Exception:
                    pass
                old_argv = list(sys.argv)
                sys.argv = ["mmpro_inprocess", *[str(a) for a in args]]
                buf_out, buf_err = StringIO(), StringIO()
                err: Exception | None = None
                try:
                    with redirect_stdout(buf_out), redirect_stderr(buf_err):
                        exec(compile(script, "<mmpro_subprocess>", "exec"), {"__name__": "__main__"})
                except Exception as e:
                    err = e
                finally:
                    sys.argv = old_argv
                out_text = buf_out.getvalue()
                err_text = buf_err.getvalue()
                if log_file is not None:
                    try:
                        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                        Path(log_file).write_text((out_text or "") + "\n" + (err_text or ""), encoding="utf-8", errors="replace")
                    except Exception:
                        pass
                if err is not None:
                    if out_text.strip():
                        self._log_html(f"=== {label} STDOUT ===\n{out_text}", Colors.YELLOW, bold=False)
                    if err_text.strip():
                        self._log_html(f"=== {label} STDERR ===\n{err_text}", Colors.YELLOW, bold=False)
                    raise RuntimeError(f"{label} in-process απέτυχε: {err}") from err
                if verbose_mode and (out_text.strip() or err_text.strip()):
                    if out_text.strip():
                        self._log_html(f"=== {label} STDOUT ===\n{out_text}", Colors.CYAN, bold=False)
                    if err_text.strip():
                        self._log_html(f"=== {label} STDERR ===\n{err_text}", Colors.CYAN, bold=False)
                else:
                    self._log_html(f"✅ Υπο-διεργασία {label} ολοκληρώθηκε (in-process).", Colors.GREEN, bold=False)
            if python_exe == "py":
                cmd = [python_exe, "-3", "-c", script] + [str(a) for a in args]
            else:
                cmd = [python_exe, "-c", script] + [str(a) for a in args]
        else:
            cmd = [sys.executable, "-c", script] + [str(a) for a in args]
        if log_file is None:
            try:
                log_file = Path(getattr(self, "model_path", Path.cwd())).parent / f"{str(label).lower()}_subprocess.log"
            except Exception:
                log_file = None
        try:
            self._log_html(f"⏳ Εκτέλεση υπο-διεργασίας {label}...", Colors.CYAN, bold=False)
        except Exception:
            pass
        env = os.environ.copy()
        env.setdefault('PYTHONUTF8', '1')
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        try:
            proc = _run_cmd( cmd, env=env,)
        except Exception as e:
            raise RuntimeError(f"Αποτυχία εκκίνησης υπο-διεργασίας {label}: {e}")

        def _clean_text(s: str) -> str:
            try:
                return "\n".join([ANSI_ESCAPE_RE.sub("", ln) for ln in str(s).splitlines()])
            except Exception:
                return str(s)
        if log_file is not None and (proc.stdout or proc.stderr):
            try:
                lf = Path(str(log_file))
                lf.parent.mkdir(parents=True, exist_ok=True)
                parts = []
                if proc.stdout:
                    parts.append("=== STDOUT ===")
                    parts.append(_clean_text(proc.stdout))
                if proc.stderr:
                    parts.append("=== STDERR ===")
                    parts.append(_clean_text(proc.stderr))
                lf.write_text("\n".join(parts).strip() + "\n", encoding="utf-8", errors="ignore")
            except Exception:
                pass
        show_output = bool(verbose_mode or show_success_output or (proc.returncode != 0))
        if show_output:
            if proc.stdout:
                try:
                    self._log_html(f"=== {label} STDOUT ===", Colors.CYAN, bold=True)
                except Exception:
                    pass
                for line in proc.stdout.splitlines():
                    clean = ANSI_ESCAPE_RE.sub("", line)
                    if clean.strip():
                        try:
                            self._log_html(clean, Colors.CYAN, bold=False)
                        except Exception:
                            pass
            if proc.stderr:
                try:
                    self._log_html(f"=== {label} STDERR ===", Colors.YELLOW, bold=True)
                except Exception:
                    pass
                for line in proc.stderr.splitlines():
                    clean = ANSI_ESCAPE_RE.sub("", line)
                    if clean.strip():
                        try:
                            self._log_html(clean, Colors.YELLOW, bold=False)
                        except Exception:
                            pass
        else:
            if proc.stdout or proc.stderr:
                try:
                    self._log_html( f"ℹ️ {label}: ολοκληρώθηκε. (Verbose log: {log_file})", Colors.CYAN, bold=False,)
                except Exception:
                    pass
        if proc.returncode != 0:
            raise RuntimeError(f"{label} υπο-διεργασία απέτυχε με κωδικό {proc.returncode}")

    def _read_imgsz_from_model(self, fallback: int = 640) -> int:
        try:
            from ultralytics import YOLO as _YOLO
            _m = _YOLO(str(self.model_path))
            for attr in ('args', 'overrides'):
                try:
                    d = getattr(_m, attr, None)
                    if isinstance(d, dict):
                        v = d.get('imgsz')
                        if v is not None:
                            if isinstance(v, (list, tuple)) and len(v) >= 1:
                                return int(v[0])
                            return int(v)
                except Exception:
                    continue
        except Exception:
            pass
        return int(fallback)

    def _export_ncnn_subprocess(self):
        from pathlib import Path
        export_dir = self.model_path.parent / f'{self.model_path.stem}_ncnn_model'
        target = ExportTarget('dir', Path(export_dir))
        if target_exists_and_nonempty(target):
            ok, msg = ensure_overwrite_target(target, overwrite=bool(getattr(self, 'overwrite', False)), require_nonempty_dir=True)
            if not ok:
                if msg == 'exists':
                    self._log_html(f'⚠️ Υπάρχει ήδη φάκελος NCNN export: {export_dir.name}', Colors.YELLOW, bold=False)
                else:
                    self._log_html(f'⚠️ Αποτυχία διαγραφής παλιού φακέλου NCNN ({msg}). Η εξαγωγή ακυρώνεται.', Colors.RED, bold=False)
        export_dir.mkdir(parents=True, exist_ok=True)
        self._log_html('--- Έναρξη Εξαγωγής NCNN ---', Colors.CYAN, bold=True)
        script = "import os ,gc ,shutil ,sys\nfrom pathlib import Path\nfrom ultralytics import YOLO\n\n\nos .environ ['CUDA_VISIBLE_DEVICES']=''\n\nmodel_path =Path (sys .argv [1 ])\nimgsz =int (sys .argv [2 ])\nexport_dir =Path (sys .argv [3 ])\nexport_dir .mkdir (parents =True ,exist_ok =True )\n\nmodel =YOLO (str (model_path ))\nresult =model .export (format ='ncnn',imgsz =imgsz ,half =False ,simplify =True ,verbose =True )\n\n\ncandidates =[]\n\ntry :\n    if result :\n        out_path =Path (str (result ))\n        if out_path .is_dir ():\n            candidates .append (out_path )\n        else :\n            candidates .append (out_path .parent )\nexcept Exception :\n    pass\n\n\nsearch_roots =[Path .cwd ()/'runs',Path .cwd (),model_path .parent ]\nfor root in search_roots :\n    try :\n        if root .exists ():\n            for d in root .rglob ('*'):\n                if d .is_dir ()and 'ncnn'in d .name .lower ():\n                    try :\n                        has_param =any (f .suffix =='.param'for f in d .glob ('*.param'))\n                        has_bin =any (f .suffix =='.bin'for f in d .glob ('*.bin'))\n                        if has_param or has_bin :\n                            candidates .append (d )\n                    except Exception :\n                        pass\n    except Exception :\n        pass\n\nsrc_dir =None\nif candidates :\n    try :\n        src_dir =max (candidates ,key =lambda d :d .stat ().st_mtime )\n    except Exception :\n        src_dir =candidates [-1 ]\n\n\nmoved_any =False\nif src_dir and src_dir .exists ():\n    for f in list (src_dir .glob ('*')):\n        try :\n            if f .is_file ()and (f .suffix in {'.param','.bin','.json','.txt'}or 'ncnn'in f .name .lower ()):\n                shutil .move (str (f ),str (export_dir /f .name ))\n                moved_any =True\n        except Exception :\n            pass\n\n\nif not moved_any :\n    for f in Path .cwd ().glob ('*'):\n        try :\n            if f .is_file ()and (f .suffix in {'.param','.bin'}and 'ncnn'in f .name .lower ()):\n                shutil .move (str (f ),str (export_dir /f .name ))\n                moved_any =True\n        except Exception :\n            pass\n"
        args = [self.model_path, self.imgsz, export_dir]
        self._run_subprocess(script, args, 'NCNN', log_file=export_dir / 'ncnn_verbose.log')
        try:
            from datetime import datetime
            import json
            has_param = any(p.suffix == '.param' for p in export_dir.glob('*.param'))
            has_bin = any(p.suffix == '.bin' for p in export_dir.glob('*.bin'))
            if has_param or has_bin:
                task = ''
                try:
                    task = str(guess_ultralytics_task(self.model_path) or '')
                except Exception:
                    task = ''
                resolved_imgsz = int(getattr(self, 'imgsz', 640))
                try:
                    from_param = _mmpro_try_infer_ncnn_imgsz_from_param(export_dir)
                    if isinstance(from_param, int) and from_param > 0:
                        resolved_imgsz = from_param
                    else:
                        resolved_imgsz = self._read_imgsz_from_model(fallback=resolved_imgsz)
                except Exception:
                    pass
                meta = { 'format': 'ncnn', 'imgsz': resolved_imgsz, 'task': task, 'created': datetime.now().isoformat(timespec='seconds'),}
                (export_dir / 'mmpro_export_meta.json').write_text( json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
        self._log_html('--- Ολοκλήρωση Εξαγωγής NCNN ---', Colors.GREEN, bold=True)
        self._log_html('✅ Εξαγωγή NCNN ολοκληρώθηκε με επιτυχία.', Colors.GREEN, bold=True)
        self._log_html(f'📁 Φάκελος εξαγόμενου μοντέλου: {export_dir}', Colors.CYAN, bold=False)

    def export_tensorrt(self) -> bool:
        from pathlib import Path
        try:
            try:
                import torch
                if not torch.cuda.is_available():
                    self._log_html('❌ Δεν εντοπίστηκε διαθέσιμη συσκευή CUDA. Η εξαγωγή TensorRT ακυρώνεται.', Colors.RED, bold=True)
                    return False
            except Exception as e:
                self._log_html(f'⚠️ Αποτυχία ελέγχου CUDA πριν την εξαγωγή TensorRT: {e}', Colors.RED, bold=False)
                return False
            try:
                import tensorrt
            except Exception as e:
                self._log_html(
                    '❌ Δεν βρέθηκαν τα TensorRT Python bindings στο περιβάλλον.\n'
                    '   • Εγκατάστησε TensorRT για την έκδοση Python σου ή χρησιμοποίησε υποστηριζόμενη έκδοση Python.\n'
                    f'   • Λεπτομέρεια: {e}',
                    Colors.RED,
                    bold=True,
                )
                return False
            self._log_html('--- Έναρξη Εξαγωγής 🔥 TensorRT Engine (.engine) ---', Colors.CYAN, bold=True)
            export_path = self.model_path.with_suffix('.engine')
            if bool(getattr(self, 'overwrite', False)):
                try:
                    ok, msg = safe_unlink(export_path)
                    if not ok:
                        self._log_html(f'⚠️ Αποτυχία διαγραφής παλιού TensorRT engine ({msg}). Η εξαγωγή ακυρώνεται.', Colors.RED, bold=False)
                        return False
                except Exception:
                    pass
            imgsz = int(getattr(self, 'imgsz', 640))
            batch = 1
            dynamic = False
            workspace = 4
            device_sig = 'cuda:0'
            try:
                if (not bool(getattr(self, 'overwrite', False))) and trt_try_restore_from_cache(self.model_path, export_path, imgsz, half=True, batch=batch, dynamic=dynamic, workspace=workspace, device=device_sig):
                    self._log_html(f'♻️🔥 Επαναφορά TensorRT engine από cache: {export_path.name}', Colors.GREEN, bold=True)
                    try:
                        _preflight_tensorrt_engine(export_path)
                    except Exception:
                        try:
                            export_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    else:
                        return True
            except Exception:
                pass
            try:
                if (not bool(getattr(self, 'overwrite', False))) and trt_engine_is_up_to_date(self.model_path, export_path, imgsz, half=True, batch=batch, dynamic=dynamic, workspace=workspace, device=device_sig):
                    try:
                        _preflight_tensorrt_engine(export_path)
                        self._log_html(f'♻️🔥 Το TensorRT engine είναι ήδη έτοιμο & ενημερωμένο: {export_path.name}', Colors.GREEN, bold=False)
                        return True
                    except Exception:
                        try:
                            export_path.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception:
                pass
            t0 = time.time()
            self._log_html('🔥 Δημιουργία/αναδόμηση TensorRT engine…', Colors.CYAN, bold=False)

            def _validate_engine_or_raise(p: Path) -> None:
                if not p.exists():
                    raise RuntimeError('Δεν δημιουργήθηκε αρχείο .engine.')
                _preflight_tensorrt_engine(p)

            def _export_inprocess(half: bool) -> None:
                model = YOLO(str(self.model_path))
                result = model.export( format='engine', imgsz=imgsz, device=0, half=half, workspace=workspace, verbose=True, batch=batch, dynamic=dynamic,)
                engine_path = Path(result) if result else export_path
                if engine_path.exists() and engine_path != export_path:
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if export_path.exists():
                            export_path.unlink()
                    except Exception:
                        pass
                    try:
                        shutil.move(str(engine_path), str(export_path))
                    except Exception:
                        shutil.copy2(str(engine_path), str(export_path))

            def _export_subprocess(half: bool) -> None:
                sig = trt_signature(self.model_path, imgsz, half=half, batch=batch, dynamic=dynamic, workspace=workspace, device=device_sig)
                sig_json = json.dumps(sig, ensure_ascii=True, sort_keys=True)
                script = r"""
import gc
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from ultralytics import YOLO
warnings.filterwarnings('ignore', message='Unable to automatically guess model task, assuming.*', category=UserWarning); logging.getLogger('ultralytics').setLevel(logging.WARNING)
try:
    from ultralytics.utils import LOGGER as _ULTRA_LOGGER
    _ULTRA_LOGGER.setLevel(logging.WARNING)
except Exception: pass
model_path = Path(sys.argv[1]); imgsz = int(sys.argv[2]); export_path = Path(sys.argv[3]); cache_dir = Path(sys.argv[4]); sig = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
os.environ['ULTRALYTICS_ONNXSLIM_DISABLED'] = '1'; os.environ['ONNXSLIM_DISABLE'] = '1'; export_path.parent.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(str(model_path)); result = model.export(format='engine', imgsz=imgsz, device=0, half=%s, workspace=%d, verbose=True, batch=%d, dynamic=%s); engine_path = Path(result) if result else export_path
if engine_path.exists() and engine_path != export_path:
    if export_path.exists():
        try: export_path.unlink()
        except Exception: pass
    try: shutil.move(str(engine_path), str(export_path))
    except Exception: shutil.copy2(str(engine_path), str(export_path))
if export_path.exists():
    # Signature + cache
    sig_path = Path(str(export_path) + '.mmpro.json')
    try: sig_path.write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception: pass
    sid = str(sig.get('signature_id') or ''); name = f'{model_path.stem}__{sid}__imgsz{imgsz}__bs1__fp{16 if %s else 32}.engine'; cached = cache_dir / name
    try: shutil.copy2(str(export_path), str(cached)); Path(str(cached) + '.mmpro.json').write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception: pass
""" % (str(bool(half)), int(workspace), int(batch), str(bool(dynamic)), str(bool(half)))
                args = [self.model_path, imgsz, export_path, trt_cache_dir_for_model(self.model_path), sig_json]
                self._run_subprocess(script, args, 'TensorRT')
            import shutil
            last_err: Exception | None = None
            for half in (True, False):
                try:
                    try:
                        if export_path.exists():
                            export_path.unlink()
                    except Exception:
                        pass
                    if is_frozen_app():
                        _export_inprocess(half=half)
                    else:
                        _export_subprocess(half=half)
                    _validate_engine_or_raise(export_path)
                    sig = trt_signature(self.model_path, imgsz, half=half, batch=batch, dynamic=dynamic, workspace=workspace, device=device_sig)
                    try:
                        json_write(trt_signature_path_for_engine(export_path), sig)
                        trt_purge_cache_for_model(self.model_path)
                    except Exception:
                        pass
                    dt = max(0.0, time.time() - t0)
                    self._log_html(f'✅🔥 TensorRT Engine δημιουργήθηκε: {export_path.name}  (⏱ {dt:.1f}s, fp{16 if half else 32})', Colors.GREEN, bold=True)
                    return True
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f'Αποτυχία δημιουργίας έγκυρου TensorRT engine: {last_err}')
        except Exception as e:
            self._log_html(f'🔥 Σφάλμα εξαγωγής TensorRT: {e}', Colors.RED, bold=True)
            return False

    def _export_cnn_onnx(self) -> None:
        """
        Εξαγωγή CNN torchvision μοντέλου (.pt) σε ONNX μέσω torch.onnx.export.
        Γράφει επίσης ένα cnn_onnx_meta.json δίπλα στο .onnx αρχείο
        ώστε η CNNInferenceHelper να μπορεί να το φορτώσει αυτόματα.
        """
        from pathlib import Path
        import json as _json
        export_path = self.model_path.with_suffix('.onnx')
        target = ExportTarget('file', Path(export_path))
        # ── Έλεγχος overwrite ────────────────────────────────────────────────
        if target_exists_and_nonempty(target):
            ok, msg = ensure_overwrite_target(
                target,
                overwrite=bool(getattr(self, 'overwrite', False)),
                require_nonempty_dir=False,
            )
            if not ok:
                if msg == 'exists':
                    self._log_html(
                        f'⚠️ Υπάρχει ήδη αρχείο ONNX: {export_path.name} — '
                        'χρησιμοποίησε Overwrite για αντικατάσταση.',
                        Colors.YELLOW, bold=False)
                else:
                    self._log_html(
                        f'⚠️ Αποτυχία διαγραφής παλιού ONNX ({msg}). '
                        'Η εξαγωγή ακυρώνεται.',
                        Colors.RED, bold=False)
                return
        self._log_html('━━━ Έναρξη CNN → ONNX Export ━━━', Colors.CYAN, bold=True)
        try:
            import torch
            # Φόρτωση μοντέλου μέσω CNNInferenceHelper (ήδη δοκιμασμένος)
            helper = CNNInferenceHelper(self.model_path, device='cpu')
            helper.load()
            self._log_html(
                f'🧠 Μοντέλο: {helper.model_name_str} | '
                f'{helper.num_classes} κλάσεις | imgsz={helper.imgsz}',
                Colors.GREEN)
            # ── torch.onnx.export ─────────────────────────────────────────────
            dummy = torch.randn(1, 3, helper.imgsz, helper.imgsz)
            torch.onnx.export(
                helper.model_nn,
                dummy,
                str(export_path),
                export_params=True,
                opset_version=12,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                verbose=False,
            )
            self._log_html(
                f'✅ CNN ONNX export ολοκληρώθηκε: {export_path.name}',
                Colors.GREEN, bold=True)
            # ── Γράψε cnn_onnx_meta.json δίπλα στο .onnx ────────────────────
            # Το CNNInferenceHelper.load() το διαβάζει αυτόματα (class_names.json
            # ή cnn_onnx_meta.json) οπότε η inference δουλεύει αμέσως.
            meta = {
                'model_name':  helper.model_name_str,
                'num_classes': helper.num_classes,
                'class_names': helper.class_names,
                'imgsz':       helper.imgsz,
                'source_pt':   self.model_path.name,
                'opset':       12,
            }
            meta_path = export_path.parent / (export_path.stem + '_onnx_meta.json')
            try:
                meta_path.write_text(
                    _json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding='utf-8')
                self._log_html(
                    f'📄 Metadata αποθηκεύτηκαν: {meta_path.name}',
                    Colors.CYAN)
            except Exception as me:
                self._log_html(
                    f'⚠️ Αδυναμία αποθήκευσης metadata ({me}). '
                    'Το ONNX αρχείο είναι OK.',
                    Colors.YELLOW)
            self._log_html('━━━ CNN → ONNX Export Ολοκληρώθηκε ━━━',
                           Colors.GREEN, bold=True)
        except Exception as e:
            import traceback as _tb
            self._log_html(
                f'❌ Σφάλμα CNN ONNX export: {e}', Colors.RED, bold=True)
            self._log_html(_tb.format_exc(), Colors.RED)
            try:
                self.error.emit(f'Σφάλμα CNN ONNX export: {e}')
            except Exception:
                pass

    def _export_onnx_subprocess(self):
        from pathlib import Path
        export_path = self.model_path.with_suffix('.onnx')
        target = ExportTarget('file', Path(export_path))
        if target_exists_and_nonempty(target):
            ok, msg = ensure_overwrite_target(target, overwrite=bool(getattr(self, 'overwrite', False)), require_nonempty_dir=False)
            if not ok:
                if msg == 'exists':
                    self._log_html(f'⚠️ Υπάρχει ήδη αρχείο ONNX: {export_path.name}', Colors.YELLOW, bold=False)
                else:
                    self._log_html(f'⚠️ Αποτυχία διαγραφής παλιού ONNX αρχείου ({msg}). Η εξαγωγή ακυρώνεται.', Colors.RED, bold=False)
        # ── YOLO ONNX export ──────────────────────────────────────────────
        # CNN μοντέλα έχουν ήδη φιλτραριστεί από το ExportWorker.run().
        self._log_html('--- Έναρξη Εξαγωγής ONNX ---', Colors.CYAN, bold=True)
        script = "import os ,gc ,shutil ,sys\nfrom pathlib import Path\nfrom ultralytics import YOLO\n\nmodel_path =Path (sys .argv [1 ])\nimgsz =int (sys .argv [2 ])\nexport_path =Path (sys .argv [3 ])\n\n\nos .environ ['ULTRALYTICS_ONNXSLIM_DISABLED']='1'\nos .environ ['ONNXSLIM_DISABLE']='1'\nos .environ ['ORT_DISABLE_ALL']='1'\nos .environ ['CUDA_VISIBLE_DEVICES']=''\n\nmodel =YOLO (str (model_path ))\nresult =model .export (\nformat ='onnx',\nimgsz =imgsz ,\nhalf =False ,\ndynamic =False ,\nsimplify =False ,\nopset =12 ,\nverbose =True ,\n)\nif result :\n    onnx_path =Path (result )\nelse :\n    onnx_path =export_path\nif onnx_path .exists ()and onnx_path !=export_path :\n    export_path .parent .mkdir (parents =True ,exist_ok =True )\n    if export_path .exists ():\n        export_path .unlink ()\n    shutil .move (str (onnx_path ),str (export_path ))\n"
        args = [self.model_path, self.imgsz, export_path]
        self._run_subprocess(script, args, 'ONNX', log_file=export_path.parent / 'onnx_verbose.log')
        self._log_html(f'Εξαγωγή ONNX ολοκληρώθηκε: {export_path}', Colors.GREEN, bold=True)
"""Training worker.
Worker που εκτελεί training, συλλέγει logs/metrics και γράφει artifacts (charts/reports).
"""
from PySide6.QtCore import QEvent


class _NoInteractFilter(QObject):

    def eventFilter(self, obj, event):
        try:
            et = event.type()
            if et in (QEvent.Wheel, QEvent.KeyPress, QEvent.KeyRelease, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
                return True
        except Exception:
            pass
        try:
            return super().eventFilter(obj, event)
        except Exception:
            return False


# ════════════════════════════════════════════════════════════════════════════════
# TrainingWorker – QObject worker για εκπαίδευση YOLO μοντέλων (Ultralytics)
# ════════════════════════════════════════════════════════════════════════════════
# Εκτελείται σε ξεχωριστό QThread (ή subprocess μέσω QProcess).
# Σήματα: log(str), finished(), error(str), progress(int, str), report_ready(str)
#
# Ροή εκπαίδευσης:
#   1. Αρχικοποίηση περιβάλλοντος (logging, seed, Triton check)
#   2. Λήψη/επαλήθευση dataset (YAML ή classification folder)
#   3. Φόρτωση YOLO μοντέλου (τοπικό ή αυτόματη λήψη)
#   4. model.train(**kwargs) με callbacks (on_epoch_end, on_model_save)
#   5. Αποθήκευση best.pt στο Trained_Models/
#   6. Δημιουργία PDF αναφοράς εκπαίδευσης (matplotlib + ReportLab)
# ════════════════════════════════════════════════════════════════════════════════
class TrainingWorker(QObject, LogEmitMixin):
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, str)
    report_ready = Signal(str)

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        imgsz: int,
        device: str,
        epochs: int,
        patience: int,
        use_triton: bool,
        compile_mode: str,
        extra_hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.dataset_name = str(dataset_name)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.use_triton = bool(use_triton)
        self.compile_mode = str(compile_mode)
        self.extra_hparams: dict[str, Any] = extra_hparams or {}
        self._stop_requested = False
        self.config = TRAIN_DATASETS.get(dataset_name)
        device_label = "CUDA" if str(self.device).lower().startswith("cuda") else "CPU"
        self.project_prefix = f"Runs_{self.model_name}_{device_label}"
        self.project_name = f"Finetuned_{self.model_name}_{self.dataset_name}_imgsz{self.imgsz}"
        self.project_dir = ROOT_DIR / self.project_prefix / self.project_name
        self.log_file = self.project_dir / "training.log"
        self.current_progress: int = 0
        self.progress_mutex = QMutex()
        self.best_map5095: float | None = None
        self.best_top1: float | None = None
        self.last_epoch_logged: int = -1
        self.triton_available: bool = False
        self.triton_compatible: bool = False
        self.file_logger: logging.Logger | None = None
        self.setup_triton_compat()

    def _parse_progress_from_text(self, text: str):
        if not text:
            return None
        t = ' '.join(str(text).split())
        m = re.search(r'(\d{1,3})\s*%', t)
        if m:
            try:
                pct = max(0, min(100, int(m.group(1))))
                return (pct, t)
            except Exception:
                return None
        m = re.search(r'epoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)', t, re.IGNORECASE)
        if m:
            try:
                cur = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    pct = max(0, min(100, int(cur * 100 / total)))
                    return (pct, t)
            except Exception:
                pass
        m = re.search(r'(step|iter|iters|batch)\s*[:=]?\s*(\d+)\s*/\s*(\d+)', t, re.IGNORECASE)
        if m:
            try:
                cur = int(m.group(2))
                total = int(m.group(3))
                if total > 0:
                    pct = max(0, min(100, int(cur * 100 / total)))
                    return (pct, t)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_requested = True

    def _setup_file_logging(self) -> None:
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger(f'training_{id(self)}')
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            self.file_logger = logger
        except Exception:
            self.file_logger = None

    def _cprint(self, text: str, color: Colors = Colors.LIGHT, bold: bool = False, underline: bool = False, inline: bool = False) -> None:
        try:
            html_text = format_html_log(text, color, bold=bold, underline=underline)
            if inline:
                html_text = "__INLINE__" + html_text
            self.log.emit(html_text)
            try:
                plain = re.sub(r"<[^>]+>", " ", html_text)
                plain = plain.replace("&nbsp;", " ").replace("&amp;", "&")
                parsed = self._parse_progress_from_text(plain)
                if parsed:
                    pct, pmsg = parsed
                    self.progress.emit(int(pct), str(pmsg))
            except Exception:
                pass
        except Exception:
            try:
                self.log.emit(text)
            except Exception:
                pass
        try:
            if self.file_logger:
                self.file_logger.info(text)
        except Exception:
            pass

    def setup_triton_compat(self) -> None:
        try:
            cuda_ok = torch.cuda.is_available() and str(self.device).lower().startswith('cuda')
        except Exception:
            cuda_ok = False
        has_compile = hasattr(torch, 'compile')
        self.triton_available = bool(cuda_ok and has_compile)
        self.triton_compatible = bool(cuda_ok and has_compile)
        if self.use_triton and self.triton_available:
            self._cprint('ℹ️ Το σύστημα υποστηρίζει torch.compile / Triton. Θα γίνει προσπάθεια χρήσης του κατά την έναρξη της εκπαίδευσης.', Colors.CYAN)
        elif self.use_triton:
            self._cprint('⚠️ Ζητήθηκε χρήση Triton, αλλά το σύστημα / PyTorch δεν το υποστηρίζει. Συνέχιση χωρίς compile.', Colors.YELLOW)
            self.use_triton = False

    def compile_model_safe(self, model):
        if not (self.use_triton and self.triton_available and self.triton_compatible):
            return model
        if not hasattr(torch, 'compile'):
            self._cprint('⚠️ Η τρέχουσα έκδοση PyTorch δεν υποστηρίζει torch.compile. Συνέχιση χωρίς compile.', Colors.YELLOW)
            self.use_triton = False
            return model
        try:
            return torch.compile(model, mode='max-autotune')
            self._cprint("🔥 TorchInductor ενεργό (torch.compile, mode='max-autotune').", Colors.MAGENTA, bold=True)
        except Exception as e:
            self._cprint(f'⚠️ Αποτυχία compilation: {e}', Colors.YELLOW)
            self._cprint('🔧 Συνέχιση με μη compiled μοντέλο (eager mode).', Colors.YELLOW)
            self.use_triton = False
            return model

    def on_train_start_safe(self, trainer):
        if self._stop_requested:
            raise StopIteration('Training stopped by user request')
        try:
            already_compiled = bool(getattr(trainer.args, 'compile', False))
        except Exception:
            already_compiled = False
        if self.use_triton and self.triton_available and self.triton_compatible:
            if already_compiled:
                self._cprint('🔄 Εκπαίδευση με ενεργό torch.compile (μέσω YOLO trainer).', Colors.CYAN)
            else:
                self._cprint('🔄 Ενεργοποίηση torch.compile (fallback από GUI)...', Colors.CYAN)
                try:
                    trainer.model = self.compile_model_safe(trainer.model)
                except Exception as e:
                    self._cprint(f'⚠️ Σφάλμα κατά το fallback torch.compile: {e}', Colors.YELLOW)
                    self._cprint('🔧 Συνέχιση χωρίς TorchInductor (fallback απέτυχε).', Colors.YELLOW)
                    self.use_triton = False

    def _log_training_environment(self) -> None:
        try:
            python_ver = platform.python_version()
            torch_ver = getattr(torch, "__version__", "N/A")
            try:
                import ultralytics
                ul_ver = getattr(ultralytics, "__version__", "N/A")
            except ImportError:
                ul_ver = "N/A"
            cuda_available = torch.cuda.is_available()
            gpu_name: str | None = None
            gpu_mem: str | None = None
            if cuda_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_mem = f"{total / 1024 ** 3:.1f} GB"
                except Exception:
                    pass
            self._cprint("=== ΠΕΡΙΒΑΛΛΟΝ ΕΚΠΑΙΔΕΥΣΗΣ ΜΟΝΤΕΛΟΥ ===", Colors.HEADER, bold=True)
            self._cprint(f"Python: {python_ver} | PyTorch: {torch_ver} | Ultralytics: {ul_ver}", Colors.LIGHT)
            if cuda_available and gpu_name:
                self._cprint(f"🟢 CUDA διαθέσιμο – GPU: {gpu_name} ({gpu_mem})", Colors.GREEN)
            elif cuda_available:
                self._cprint("🟢 CUDA διαθέσιμο", Colors.GREEN)
            else:
                self._cprint("⚠️ CUDA ΜΗ διαθέσιμο – εκπαίδευση σε CPU.", Colors.YELLOW)
        except Exception as e:
            _MMPRO_LOGGER.debug("_log_training_environment error: %s", e)

    def _log_training_flow(self) -> None:
        try:
            lines = ['🌳 Δομή λογικής ροής εκπαίδευσης', '└─ 🧠 Κύρια διαδικασία: Training (GUI)', '   ├─ 📁 Έλεγχος διαδρομής & ρυθμίσεων dataset', '   ├─ 🧪 Έλεγχος περιβάλλοντος Python / PyTorch / Ultralytics / CUDA', '   ├─ 🧹 Προετοιμασία μνήμης (GC / CUDA cache, seed, καθάρισμα παλιών runs)', '   ├─ 📥 Προετοιμασία / κατέβασμα dataset (YOLO ή ταξινόμηση)', '   ├─ ⚙️ Ρύθμιση Triton / torch.compile (αν είναι ενεργό)', '   ├─ 🏋️ Εκπαίδευση μοντέλου (training epochs)', '   └─ 📊 Αξιολόγηση & παραγωγή αναφοράς (metrics, γραφήματα, logs)']
            for line in lines:
                self._cprint(line, Colors.CYAN)
        except Exception:
            pass

    def _print_timing_summary(self, overall_start_time: float | None, overall_end_time: float | None, startup_end_time: float | None, dataset_start_time: float | None, dataset_end_time: float | None, train_start_time: float | None, train_end_time: float | None, eval_start_time: float | None, eval_end_time: float | None) -> None:
        if overall_start_time is None or overall_end_time is None:
            return

        def safe_delta(start: float | None, end: float | None) -> float | None:
            if start is None or end is None:
                return None
            try:
                delta = float(end) - float(start)
            except Exception:
                return None
            return max(delta, 0.0)
        total_time = safe_delta(overall_start_time, overall_end_time)
        startup_time = safe_delta(overall_start_time, startup_end_time)
        dataset_time = safe_delta(dataset_start_time, dataset_end_time)
        train_time = safe_delta(train_start_time, train_end_time)
        eval_time = safe_delta(eval_start_time, eval_end_time)
        self._cprint('\n⏱️ ΧΡΟΝΟΜΕΤΡΗΣΗ ΕΚΠΑΙΔΕΥΣΗΣ (πραγματικός χρόνος)', Colors.HEADER, bold=True, underline=True)
        label_width = 40
        hms_width = 10
        secs_width = 12
        header = f"{'Στάδιο':<{label_width}} | {'Διάρκεια':^{hms_width}} | {'Σύνολο (sec)':^{secs_width}}"
        separator = '-' * len(header)
        self._cprint(separator, Colors.LIGHT)
        self._cprint(header, Colors.LIGHT, bold=True)
        self._cprint(separator, Colors.LIGHT)

        def format_row(label: str, value: float | None, highlight: bool=False) -> None:
            if value is None:
                return
            try:
                seconds = float(value)
            except Exception:
                return
            mins, secs = divmod(seconds, 60)
            hours, mins = divmod(int(mins), 60)
            hms = f'{hours:02d}:{mins:02d}:{int(secs):02d}'
            text = f'{label:<{label_width}} | {hms:>{hms_width}} | {seconds:>{secs_width}.1f}'
            color = Colors.GREEN if highlight else Colors.LIGHT
            self._cprint(text, color, bold=highlight)
        format_row('Χρόνος εκκίνησης:', startup_time)
        format_row('Χρόνος προετοιμασίας dataset:', dataset_time)
        format_row('Χρόνος εκπαίδευσης εποχών:', train_time)
        format_row('Χρόνος αξιολόγησης μοντέλου:', eval_time)
        format_row('Συνολικός χρόνος:', total_time, highlight=True)

    def set_seed(self, seed: int=42) -> None:
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    def download_file(self, url: str, dest: Path) -> None:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._cprint(f'⬇️ Λήψη: {url}', Colors.LIGHT)
        last_progress = {'value': -1}
        first_line = {'done': False}

        def _report_hook(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            progress = min(100, int(downloaded * 100 / total_size))
            if progress == last_progress['value']:
                return
            last_progress['value'] = progress
            if progress not in (0, 100) and progress % 10 != 0:
                return
            bar_len = 20
            filled = int(progress * bar_len / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            msg = f'   ▓ Πρόοδος λήψης: [{bar}] {progress}%'
            if not first_line['done']:
                self._cprint(msg, Colors.LIGHT)
                first_line['done'] = True
            else:
                self._cprint(msg, Colors.LIGHT, inline=True)
        urllib.request.urlretrieve(url, dest, _report_hook)
        final = last_progress['value'] if last_progress['value'] >= 0 else 100
        bar_len = 20
        filled = int(final * bar_len / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        self._cprint(f'   ▓ Πρόοδος λήψης: [{bar}] {final}%', Colors.LIGHT, inline=True)
        self._cprint(f'✅ Ολοκληρώθηκε η λήψη: {dest}', Colors.GREEN)

    def ensure_dataset(self, cfg: DatasetConfig) -> None:
        if not isinstance(cfg, DatasetConfig):
            return
        train_dir = Path(cfg.train_dir) if cfg.train_dir else None
        val_dir = Path(cfg.val_dir) if cfg.val_dir else None
        dataset_root: Path | None = None
        if train_dir is not None:
            p = train_dir
            try:
                while True:
                    if p.parent == DATASETS_DIR:
                        dataset_root = p
                        break
                    if p.parent == p:
                        break
                    p = p.parent
            except Exception:
                dataset_root = None
        if dataset_root is None:
            if getattr(cfg, 'name', None):
                dataset_root = DATASETS_DIR / cfg.name
            else:
                dataset_root = DATASETS_DIR
        yaml_path = dataset_root / (cfg.yaml_name or '')
        if train_dir and val_dir and train_dir.is_dir() and val_dir.is_dir() and yaml_path.is_file():
            return
        if cfg.zip_url and train_dir and val_dir and (not train_dir.is_dir() or not val_dir.is_dir()):
            zip_dest = DATASETS_DIR / f'{cfg.name}.zip'
            if not zip_dest.is_file():
                self.download_file(cfg.zip_url, zip_dest)
            self._cprint(f'📦 Αποσυμπίεση {zip_dest.name}...', Colors.LIGHT)
            import zipfile
            with zipfile.ZipFile(zip_dest, 'r') as zf:
                zf.extractall(DATASETS_DIR)
            self._cprint('✅ Ολοκληρώθηκε η αποσυμπίεση dataset.', Colors.GREEN)
            try:
                if zip_dest.is_file():
                    zip_dest.unlink()
                    self._cprint(f'🗑️ Διαγράφηκε το αρχείο ZIP: {zip_dest}', Colors.LIGHT)
            except Exception as e:
                self._cprint(f'⚠️ Αποτυχία διαγραφής ZIP {zip_dest}: {e}', Colors.YELLOW)
        if cfg.yaml_url and (not yaml_path.is_file()):
            self.download_yaml(cfg.yaml_url, yaml_path)
        try:
            ds_name = getattr(cfg, 'name', None) or ''
            if str(ds_name).lower() == 'coco' and dataset_root and dataset_root.is_dir():
                has_train = False
                has_val = False
                try:
                    train_images_dir = dataset_root / 'images' / 'train2017'
                    val_images_dir = dataset_root / 'images' / 'val2017'
                    if train_images_dir.is_dir():
                        for _ in train_images_dir.iterdir():
                            has_train = True
                            break
                    train_list = dataset_root / 'train2017.txt'
                    if train_list.is_file():
                        has_train = True
                    if val_images_dir.is_dir():
                        for _ in val_images_dir.iterdir():
                            has_val = True
                            break
                    val_list = dataset_root / 'val2017.txt'
                    if val_list.is_file():
                        has_val = True
                except Exception:
                    pass
                if has_train and has_val:
                    deleted_any = False
                    for zip_path in dataset_root.rglob('*.zip'):
                        try:
                            zip_path.unlink()
                            deleted_any = True
                            self._cprint(f'🗑️ Διαγράφηκε ZIP του COCO: {zip_path}', Colors.LIGHT)
                        except Exception as e:
                            self._cprint(f'⚠️ Αποτυχία διαγραφής ZIP {zip_path}: {e}', Colors.YELLOW)
                    if deleted_any:
                        self._cprint('✅ Καθαρισμός ZIP για το πλήρες COCO ολοκληρώθηκε.', Colors.GREEN)
        except Exception as e:
            self._cprint(f'⚠️ Σφάλμα στον καθαρισμό ZIP του COCO: {e}', Colors.YELLOW)

    def _prepare_full_coco_dataset(self) -> bool:
        ds_root = DATASETS_DIR / 'coco'
        ds_root.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics import settings
            settings.update({'datasets_dir': str(DATASETS_DIR)})
        except Exception:
            pass
        yaml_path = ds_root / 'coco.yaml'
        images_train = ds_root / 'images' / 'train2017'
        images_val = ds_root / 'images' / 'val2017'
        labels_train = ds_root / 'labels' / 'train2017'
        labels_val = ds_root / 'labels' / 'val2017'

        def _has_any_file(d: Path, exts: set[str] | None = None) -> bool:
            try:
                if not d.is_dir():
                    return False
                for fp in d.iterdir():
                    if not fp.is_file():
                        continue
                    if exts is None:
                        return True
                    if fp.suffix.lower() in exts:
                        return True
            except Exception:
                return False
            return False

        def _purge_caches(root: Path) -> int:
            removed = 0
            try:
                for p in root.rglob('*.cache'):
                    try:
                        p.unlink(missing_ok=True)
                        removed += 1
                    except Exception:
                        pass
            except Exception:
                pass
            return removed

        def _write_or_patch_coco_yaml(dst: Path, root_dir: Path) -> None:
            coco_names = [
                'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
                'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
                'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
            ]
            try:
                if dst.is_file():
                    t = dst.read_text(encoding='utf-8', errors='ignore')
                    ok_path = (str(root_dir) in t) or (root_dir.as_posix() in t)
                    uses_dirs = ('train: images/train2017' in t) and ('val: images/val2017' in t)
                    if ok_path and uses_dirs:
                        return
            except Exception:
                pass
            root_str = str(root_dir).replace('\\', '/')
            lines = []
            lines.append('# COCO 2017 dataset (pinned inside Data_Sets/coco)')
            lines.append(f'path: {root_str}')
            lines.append('train: images/train2017')
            lines.append('val: images/val2017')
            lines.append('test: images/test2017')
            lines.append('nc: 80')
            lines.append('names:')
            for i, name in enumerate(coco_names):
                lines.append(f'  {i}: {name}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        try:
            _write_or_patch_coco_yaml(yaml_path, ds_root)
        except Exception as e:
            self.error.emit(f'❌ Αποτυχία δημιουργίας/διόρθωσης coco.yaml στο Data_Sets/coco: {e}')
            return False
        removed = _purge_caches(ds_root)
        if removed:
            self._cprint(f'🧹 Διαγράφηκαν {removed} παλιά *.cache μέσα στο COCO (Data_Sets/coco).', Colors.YELLOW)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        has_train_imgs = _has_any_file(images_train, img_exts)
        has_val_imgs = _has_any_file(images_val, img_exts)
        has_train_lbls = _has_any_file(labels_train, {'.txt'})
        has_val_lbls = _has_any_file(labels_val, {'.txt'})
        if not (has_train_imgs and has_val_imgs and has_train_lbls and has_val_lbls):
            self._cprint( '⬇️ Το COCO δεν βρέθηκε πλήρες στο Data_Sets/coco. ' 'Θα επιχειρήσω αυτόματο download ΜΕΣΑ στο Data_Sets...', Colors.YELLOW,)

            def _ultra_download(url: str, dest_dir: Path, unzip: bool = True) -> bool:
                dest_dir.mkdir(parents=True, exist_ok=True)
                fname = url.rsplit('/', 1)[-1]
                dest_file = dest_dir / fname
                try:
                    from ultralytics.utils.downloads import safe_download as _sdl
                    self._cprint(f'⬇️ safe_download: {fname}', Colors.LIGHT)
                    _sdl( url=url, file=str(dest_file), unzip=unzip, delete=True, retry=3, progress=True,)
                    if unzip:
                        return True
                    return dest_file.exists()
                except Exception as e1:
                    self._cprint(f'⚠️ safe_download αποτυχία ({e1}), fallback urllib...', Colors.YELLOW)
                    try:
                        if dest_file.exists():
                            dest_file.unlink()
                    except Exception:
                        pass
                try:
                    import urllib.request
                    import zipfile as _zf
                    self._cprint(f'⬇️ urllib.request: {url} → {dest_file.name}', Colors.LIGHT)
                    last_pct = [-1]

                    def _reporthook(count, block_size, total_size):
                        if total_size <= 0:
                            return
                        pct = min(100, int(count * block_size * 100 / total_size))
                        if pct != last_pct[0] and pct % 10 == 0:
                            self._cprint(f'  {fname}: {pct}%', Colors.LIGHT)
                            last_pct[0] = pct
                    urllib.request.urlretrieve(url, str(dest_file), reporthook=_reporthook)
                    if not dest_file.exists() or dest_file.stat().st_size == 0:
                        raise RuntimeError(f'Downloaded file is empty: {dest_file}')
                    if unzip and dest_file.suffix.lower() == '.zip':
                        self._cprint(f'📦 Αποσυμπίεση {dest_file.name}...', Colors.LIGHT)
                        with _zf.ZipFile(str(dest_file), 'r') as zf:
                            zf.extractall(str(dest_dir))
                        try:
                            dest_file.unlink()
                        except Exception:
                            pass
                    return True
                except Exception as e2:
                    self._cprint(f'❌ urllib download αποτυχία: {e2}', Colors.RED)
                    return False
            URL_LABELS = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip'
            URL_VAL    = 'http://images.cocodataset.org/zips/val2017.zip'
            URL_TRAIN  = 'http://images.cocodataset.org/zips/train2017.zip'
            need_labels = not has_train_lbls or not has_val_lbls
            need_val    = not has_val_imgs
            need_train  = not has_train_imgs
            if need_labels:
                self._cprint('⬇️ Λήψη COCO labels (~241 MB)...', Colors.CYAN)
                ok = _ultra_download(URL_LABELS, ds_root, unzip=True)
                if not ok:
                    self.error.emit(
                        '❌ Αποτυχία download COCO labels.\n'
                        f'Κατέβασέ τα χειροκίνητα από:\n  {URL_LABELS}\n'
                        f'και αποσυμπίεσέ τα χειροκίνητα στο: {ds_root}\n'
                        f'(ώστε να δημιουργηθεί: {labels_train})'
                    )
                    return False
                if not labels_train.exists():
                    for candidate in [ ds_root.parent / 'labels', ds_root.parent / 'coco' / 'labels', ds_root / 'coco' / 'labels',]:
                        if candidate.is_dir() and any(candidate.rglob('*.txt')):
                            target = ds_root / 'labels'
                            if candidate.resolve() != target.resolve():
                                try:
                                    if target.exists():
                                        shutil.rmtree(str(target), ignore_errors=True)
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.move(str(candidate), str(target))
                                    self._cprint( f'🔧 Labels μετακινήθηκαν: {candidate} → {target}', Colors.YELLOW,)
                                except Exception as mv_e:
                                    self._cprint(f'⚠️ Αποτυχία μετακίνησης labels: {mv_e}', Colors.YELLOW)
                            break
            if need_val:
                self._cprint('⬇️ Λήψη COCO val2017 images (~1 GB)...', Colors.CYAN)
                images_dir = ds_root / 'images'
                ok = _ultra_download(URL_VAL, images_dir, unzip=True)
                if not ok:
                    self.error.emit( '❌ Αποτυχία download COCO val images.\n' f'Κατέβασέ τα χειροκίνητα από:\n  {URL_VAL}\n' f'και αποσυμπίεσέ τα μέσα στο: {images_dir}')
                    return False
                wrong = ds_root / 'val2017'
                right = images_dir / 'val2017'
                if wrong.exists() and not right.exists():
                    try:
                        right.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(wrong), str(right))
                    except Exception:
                        pass
            if need_train:
                self._cprint('⬇️ Λήψη COCO train2017 images (~18 GB, μπορεί να πάρει ώρα)...', Colors.CYAN)
                images_dir = ds_root / 'images'
                ok = _ultra_download(URL_TRAIN, images_dir, unzip=True)
                if not ok:
                    self.error.emit( '❌ Αποτυχία download COCO train images.\n' f'Κατέβασέ τα χειροκίνητα από:\n  {URL_TRAIN}\n' f'και αποσυμπίεσέ τα μέσα στο: {images_dir}')
                    return False
                wrong = ds_root / 'train2017'
                right = images_dir / 'train2017'
                if wrong.exists() and not right.exists():
                    try:
                        right.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(wrong), str(right))
                    except Exception:
                        pass
            _purge_caches(ds_root)
            has_train_imgs = _has_any_file(images_train, img_exts)
            has_val_imgs   = _has_any_file(images_val, img_exts)
            has_train_lbls = _has_any_file(labels_train, {'.txt'})
            has_val_lbls   = _has_any_file(labels_val, {'.txt'})
        if not has_train_imgs:
            self.error.emit(f'❌ Δεν βρέθηκαν εικόνες στο: {images_train}\nΒάλε το COCO μέσα στο Data_Sets/coco (images/train2017).')
            return False
        if not has_val_imgs:
            self.error.emit(f'❌ Δεν βρέθηκαν εικόνες στο: {images_val}\nΒάλε το COCO μέσα στο Data_Sets/coco (images/val2017).')
            return False
        if not has_train_lbls:
            self.error.emit(f'❌ Δεν βρέθηκαν labels στο: {labels_train}\nΠεριμένω YOLO labels (*.txt) μέσα στο Data_Sets/coco/labels/train2017.')
            return False
        if not has_val_lbls:
            self.error.emit(f'❌ Δεν βρέθηκαν labels στο: {labels_val}\nΠεριμένω YOLO labels (*.txt) μέσα στο Data_Sets/coco/labels/val2017.')
            return False
        try:
            sample = []
            for p in labels_train.glob('*.txt'):
                sample.append(p)
                if len(sample) >= 25:
                    break
            bad = 0
            good_with_objects = 0
            for p in sample:
                try:
                    raw = p.read_text(encoding='utf-8', errors='ignore').strip()
                    if not raw:
                        continue
                    for line in raw.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            bad += 1
                            break
                        cls, x, y, w, h = map(float, parts)
                        if not (0 <= cls <= 79):
                            bad += 1
                            break
                        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                            bad += 1
                            break
                        good_with_objects += 1
                        break
                except Exception:
                    bad += 1
            if sample and bad == len(sample) and good_with_objects == 0:
                self.error.emit(
                    '❌ Τα labels στο Data_Sets/coco/labels/train2017 φαίνονται να ΜΗΝ είναι σε YOLO format.\n'
                    'Περιμένω γραμμές: <class_id> <x_center> <y_center> <width> <height> (όλα normalized 0..1).'
                )
                return False
        except Exception:
            pass
        self._cprint('✅ COCO έτοιμο μέσα στο Data_Sets/coco (χρήση τοπικού coco.yaml).', Colors.GREEN)
        return True

    def download_yaml(self, url: str, dest: Path) -> None:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._cprint(f'⬇️ Λήψη YAML: {url}', Colors.LIGHT)
        with urllib.request.urlopen(url) as resp, dest.open('wb') as f:
            f.write(resp.read())
        self._cprint(f'✅ Αποθηκεύτηκε YAML: {dest}', Colors.GREEN)

    def remove_old_runs(self, project_prefix: str) -> None:
        try:
            base = self.project_dir
            if base.is_dir():
                shutil.rmtree(base, ignore_errors=True)
                self._cprint(f'🗑️ Διαγράφηκε παλιός φάκελος run: {base}', Colors.YELLOW)
        except Exception as e:
            self._cprint(f'⚠️ Αποτυχία καθαρισμού παλιού run εκπαίδευσης: {e}', Colors.YELLOW)

    # ── Callback που καλείται στο τέλος κάθε epoch από το YOLO trainer ──────────
    # Ενημερώνει το progress bar, εξάγει metrics (mAP50-95 ή Top-1 accuracy)
    # και εκπέμπει log μήνυμα με τη μορφή:
    # "--- Ολοκληρώθηκε η Εποχή N/T (Progress: P%) – loss=X | mAP50-95=X ---"
    def on_epoch_end(self, trainer):
        if self._stop_requested:
            self._cprint('🛑 Ελήφθη σήμα διακοπής. Τερματισμός εκπαίδευσης...', Colors.RED, bold=True)
            raise StopIteration('Training stopped by user')
        try:
            epoch_idx = int(getattr(trainer, 'epoch', 0))
            if epoch_idx == self.last_epoch_logged:
                return
            self.last_epoch_logged = epoch_idx
            display_epoch = epoch_idx + 1
            total_epochs = int(getattr(getattr(trainer, 'args', None), 'epochs', self.epochs or 0) or self.epochs or 1)
            try:
                prog = 20 + int(epoch_idx / max(total_epochs, 1) * 70)
                prog = max(20, min(90, prog))
            except Exception:
                prog = 20
            self.progress_mutex.lock()
            self.current_progress = prog
            self.progress_mutex.unlock()
            metrics = getattr(trainer, 'metrics', {}) or {}
            is_classification_model = isinstance(self.model_name, str) and self.model_name.endswith('-cls')
            metric_parts = []
            loss_val = None
            for key in ('train/box_loss', 'train/loss', 'loss'):
                if key in metrics:
                    loss_val = metrics.get(key)
                    break
            if isinstance(loss_val, (int, float)):
                metric_parts.append(f'loss={loss_val:.4f}')
            if is_classification_model:
                top1_keys = ('metrics/accuracy_top1', 'metrics/acc_top1', 'metrics/top1', 'top1')
                top5_keys = ('metrics/accuracy_top5', 'metrics/acc_top5', 'metrics/top5', 'top5')
                top1 = next((metrics[k] for k in top1_keys if k in metrics), None)
                top5 = next((metrics[k] for k in top5_keys if k in metrics), None)
                if isinstance(top1, (int, float)):
                    self.best_top1 = max(self.best_top1 or 0.0, float(top1))
                    metric_parts.append(f'Top-1={float(top1):.4f}')
                if isinstance(top5, (int, float)):
                    metric_parts.append(f'Top-5={float(top5):.4f}')
            else:
                map_keys = ('metrics/mAP50-95(B)', 'metrics/mAP50-95(M)', 'metrics/mAP50-95')
                map50_keys = ('metrics/mAP50(B)', 'metrics/mAP50(M)', 'metrics/mAP50')
                map5095 = next((metrics[k] for k in map_keys if k in metrics), None)
                map50 = next((metrics[k] for k in map50_keys if k in metrics), None)
                if isinstance(map5095, (int, float)):
                    self.best_map5095 = max(self.best_map5095 or 0.0, float(map5095))
                    metric_parts.append(f'mAP50-95={float(map5095):.4f}')
                if isinstance(map50, (int, float)):
                    metric_parts.append(f'mAP50={float(map50):.4f}')
            metrics_str = ' | '.join(metric_parts) if metric_parts else 'χωρίς διαθέσιμα metrics'
            self._cprint(f'--- Ολοκληρώθηκε η Εποχή {display_epoch}/{total_epochs} (Progress: {prog}%) – {metrics_str} ---', Colors.MAGENTA)
        except StopIteration:
            raise
        except Exception as e:
            try:
                self._cprint(f'⚠️ Σφάλμα στο on_epoch_end: {e}', Colors.YELLOW)
            except Exception:
                pass

    # ── Callback που καλείται όταν αποθηκεύεται νέο καλύτερο μοντέλο ────────────
    # Εμφανίζει banner "ΝΕΟ ΚΑΛΥΤΕΡΟ ΜΟΝΤΕΛΟ" με το νέο score.
    # Για classification: ελέγχει Top-1 Accuracy. Για detection: mAP@50-95.
    def on_model_save(self, trainer):
        if self._stop_requested:
            return
        metrics = getattr(trainer, 'metrics', None) or {}
        epoch = int(getattr(trainer, 'epoch', 0))
        is_classification_model = isinstance(self.model_name, str) and self.model_name.endswith('-cls')
        best_value = None
        metric_label = None
        if is_classification_model:
            top1_keys = ('metrics/accuracy_top1', 'metrics/acc_top1', 'metrics/top1', 'top1')
            top1 = next((metrics[k] for k in top1_keys if k in metrics), None)
            if isinstance(top1, (int, float)):
                prev_best = self.best_top1 or 0.0
                if top1 <= prev_best:
                    return
                self.best_top1 = float(top1)
                best_value = float(top1)
                metric_label = 'Top-1 Accuracy'
        else:
            map_keys = ('metrics/mAP50-95(B)', 'metrics/mAP50-95(M)', 'metrics/mAP50-95')
            map5095 = next((metrics[k] for k in map_keys if k in metrics), None)
            if isinstance(map5095, (int, float)):
                prev_best = self.best_map5095 or 0.0
                if map5095 <= prev_best:
                    return
                self.best_map5095 = float(map5095)
                best_value = float(map5095)
                metric_label = 'mAP@50-95'
        if best_value is None:
            return
        self._cprint(f"\n{'=' * 70}", Colors.GREEN, bold=True)
        self._cprint(f'ΝΕΟ ΚΑΛΥΤΕΡΟ ΜΟΝΤΕΛΟ (Epoch: {epoch + 1})!', Colors.YELLOW, bold=True)
        self._cprint(f'   {metric_label}: {best_value:.4f}', Colors.GREEN)
        self._cprint(f"{'=' * 70}\n", Colors.GREEN, bold=True)

    # ── Δημιουργία γραφημάτων και PDF αναφοράς εκπαίδευσης ─────────────────────
    # Διαβάζει το results.csv, δημιουργεί:
    #   - Loss curves (train/val loss)
    #   - Detection metrics (mAP50, mAP50-95, Precision, Recall)
    #   - Training dynamics (LR, fitness)
    # Αποθηκεύει PDF στο Train_Reports/ και JSON metrics στο μοντέλο.
    def plot_metrics(self, csv_path: Path, project_dir: Path, model_name: str, dataset_name: str, device: str, imgsz: int, final_metrics: dict, requested_epochs: int, model_output_dir: Path | None=None) -> Path | None:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from datetime import datetime
            import numpy as np
        except Exception as e:
            self._cprint(f'⚠️ Δεν είναι δυνατή η δημιουργία PDF αναφοράς (λείπει matplotlib/pandas/numpy): {e}', Colors.YELLOW)
        try:
            plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
        except Exception:
            pass
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self._cprint(f'⚠️ Αποτυχία ανάγνωσης results.csv: {e}', Colors.YELLOW)
        if df.empty:
            self._cprint('⚠️ Το αρχείο results.csv είναι κενό – δεν δημιουργείται αναφορά.', Colors.YELLOW)
        epochs = df['epoch'] if 'epoch' in df.columns else pd.Series(range(len(df)), name='epoch')
        last_row = df.iloc[-1]
        cols = list(df.columns)

        def _num(v):
            try:
                if v is None:
                    return None
                if pd.isna(v):
                    return None
                return float(v)
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        def _get_last(name: str, default=None):
            if name in last_row:
                return _num(last_row[name])
            return default
        has_map = any((c.startswith('metrics/mAP') for c in cols))
        has_precision_recall = any((c.startswith('metrics/precision') or c.startswith('metrics/recall') for c in cols))
        has_accuracy = any(('metrics/accuracy' in c for c in cols))
        task_type = 'generic'
        if has_map or has_precision_recall:
            task_type = 'detect'
        if has_accuracy and (not (has_map or has_precision_recall)):
            task_type = 'classify'
        if has_accuracy and (has_map or has_precision_recall):
            name_lower = (model_name or '').lower()
            task_type = 'classify' if 'cls' in name_lower or 'class' in name_lower else 'detect'
        final_metrics.clear()
        try:
            if 'epoch' in df.columns:
                unique_epochs = sorted(df['epoch'].unique())
                epochs_ran_raw = len(unique_epochs)
            else:
                epochs_ran_raw = len(df)
        except Exception:
            epochs_ran_raw = len(df)
        if requested_epochs is not None:
            try:
                epochs_ran = min(int(epochs_ran_raw), int(requested_epochs))
            except Exception:
                epochs_ran = int(epochs_ran_raw)
        else:
            epochs_ran = int(epochs_ran_raw)
        final_metrics['epochs_ran'] = epochs_ran
        train_box_loss = _get_last('train/box_loss')
        val_box_loss = _get_last('val/box_loss')
        train_loss_generic = _get_last('train/loss')
        val_loss_generic = _get_last('val/loss')
        map50_col = None
        map5095_col = None
        for cand in ('metrics/mAP50(B)', 'metrics/mAP50'):
            if cand in df.columns:
                map50_col = cand
                break
        for cand in ('metrics/mAP50-95(B)', 'metrics/mAP50-95'):
            if cand in df.columns:
                map5095_col = cand
                break
        best_map50 = float(df[map50_col].max()) if map50_col else None
        best_map5095 = float(df[map5095_col].max()) if map5095_col else None
        prec_col = 'metrics/precision(B)' if 'metrics/precision(B)' in df.columns else 'metrics/precision' if 'metrics/precision' in df.columns else None
        rec_col = 'metrics/recall(B)' if 'metrics/recall(B)' in df.columns else 'metrics/recall' if 'metrics/recall' in df.columns else None
        best_precision = float(df[prec_col].max()) if prec_col else None
        best_recall = float(df[rec_col].max()) if rec_col else None
        acc1_col = None
        acc5_col = None
        for c in cols:
            if 'metrics/accuracy_top1' in c:
                acc1_col = c
            if 'metrics/accuracy_top5' in c:
                acc5_col = c
        best_acc1 = float(df[acc1_col].max()) if acc1_col else None
        best_acc5 = float(df[acc5_col].max()) if acc5_col else None
        final_metrics['train_box_loss'] = train_box_loss
        final_metrics['val_box_loss'] = val_box_loss
        final_metrics['train_loss'] = train_loss_generic
        final_metrics['val_loss'] = val_loss_generic
        final_metrics['map50'] = best_map50
        final_metrics['map5095'] = best_map5095
        final_metrics['precision'] = best_precision
        final_metrics['recall'] = best_recall
        final_metrics['acc_top1'] = best_acc1
        final_metrics['acc_top5'] = best_acc5
        final_metrics['task_type'] = task_type
        try:
            reports_dir = TRAIN_REPORTS_DIR
        except Exception:
            try:
                from pathlib import Path as _Path
                base_dir = _Path(project_dir).parent
            except Exception:
                from pathlib import Path as _Path
                base_dir = _Path('.')
            reports_dir = base_dir / 'Train_Reports'
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        safe_model = str(model_name).replace('/', '_')
        safe_dataset = str(dataset_name).replace('/', '_')
        pdf_name = f'TrainReport_{safe_model}_{safe_dataset}_{imgsz}.pdf'
        pdf_path = reports_dir / pdf_name

        def fmt(v, digits: int=4) -> str:
            if v is None:
                return '–'
            try:
                return f'{float(v):.{digits}f}'
            except Exception:
                return str(v)
        results_png = None
        candidates = [project_dir / 'results.png', project_dir / 'train' / 'results.png', project_dir.parent / 'results.png']
        for cand in candidates:
            if cand.is_file():
                results_png = cand
                break
        try:
            pass
        except Exception as e:
            self._cprint(f'⚠️ Δεν είναι δυνατή η δημιουργία πλούσιου PDF (λείπει reportlab/pdf_reports): {e}', Colors.YELLOW)
        assets_dir = reports_dir / f'_assets_{safe_model}_{safe_dataset}_{imgsz}'
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _save_fig(fig, name: str) -> Path:
            out = assets_dir / name
            try:
                fig.savefig(out, dpi=200, bbox_inches='tight')
            finally:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            return out
        charts: list[tuple[str, Path]] = []
        extra_pages: list[tuple[str, Path]] = []
        fig_loss, ax_loss = plt.subplots(figsize=(11.0, 6.2))
        ax_loss.set_title('Loss Curves')
        plotted_any_loss = False
        if task_type == 'detect':
            for col_name, label in [('train/box_loss', 'train/box_loss'), ('val/box_loss', 'val/box_loss'), ('train/cls_loss', 'train/cls_loss'), ('val/cls_loss', 'val/cls_loss'), ('train/dfl_loss', 'train/dfl_loss'), ('val/dfl_loss', 'val/dfl_loss')]:
                if col_name in df.columns:
                    ax_loss.plot(epochs, df[col_name], label=label)
                    plotted_any_loss = True
        elif task_type == 'classify':
            for col_name, label in [('train/loss', 'train/loss'), ('val/loss', 'val/loss')]:
                if col_name in df.columns:
                    ax_loss.plot(epochs, df[col_name], label=label)
                    plotted_any_loss = True
        else:
            for col_name in df.columns:
                if 'loss' in col_name:
                    ax_loss.plot(epochs, df[col_name], label=col_name)
                    plotted_any_loss = True
        if plotted_any_loss:
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.grid(True, linestyle='--', alpha=0.25)
            ax_loss.legend(loc='best')
        else:
            ax_loss.text(0.5, 0.5, 'No loss columns found in results.csv', ha='center', va='center')
        charts.append(('Loss curves', _save_fig(fig_loss, 'loss_curves.png')))
        fig_met, ax_met = plt.subplots(figsize=(11.0, 6.2))
        if task_type == 'detect':
            ax_met.set_title('Detection Metrics')
            plotted_any = False
            if map50_col:
                ax_met.plot(epochs, df[map50_col], label=map50_col)
                plotted_any = True
            if map5095_col:
                ax_met.plot(epochs, df[map5095_col], label=map5095_col)
                plotted_any = True
            if prec_col:
                ax_met.plot(epochs, df[prec_col], label=prec_col)
                plotted_any = True
            if rec_col:
                ax_met.plot(epochs, df[rec_col], label=rec_col)
                plotted_any = True
            if plotted_any:
                ax_met.set_xlabel('Epoch')
                ax_met.set_ylabel('Score')
                ax_met.set_ylim(0.0, 1.05)
                ax_met.grid(True, linestyle='--', alpha=0.25)
                ax_met.legend(loc='best')
            else:
                ax_met.text(0.5, 0.5, 'No detection metrics in results.csv', ha='center', va='center')
        elif task_type == 'classify':
            ax_met.set_title('Classification Metrics')
            plotted_any = False
            if acc1_col:
                ax_met.plot(epochs, df[acc1_col], label=acc1_col)
                plotted_any = True
            if acc5_col:
                ax_met.plot(epochs, df[acc5_col], label=acc5_col)
                plotted_any = True
            if plotted_any:
                ax_met.set_xlabel('Epoch')
                ax_met.set_ylabel('Accuracy')
                ax_met.set_ylim(0.0, 1.05)
                ax_met.grid(True, linestyle='--', alpha=0.25)
                ax_met.legend(loc='best')
            else:
                ax_met.text(0.5, 0.5, 'No accuracy metrics in results.csv', ha='center', va='center')
        else:
            ax_met.set_title('Metrics')
            ax_met.text(0.5, 0.5, 'No known metrics found', ha='center', va='center')
        charts.append(('Key metrics', _save_fig(fig_met, 'metrics.png')))
        fig_dyn, ax_dyn = plt.subplots(figsize=(11.0, 6.2))
        ax_dyn.set_title('Training Dynamics (LR / Fitness)')
        plotted_any_dyn = False
        for col_name in ['lr0', 'lr1', 'lr2', 'lr', 'lr/pg', 'metrics/fitness']:
            if col_name in df.columns:
                ax_dyn.plot(epochs, df[col_name], label=col_name)
                plotted_any_dyn = True
        if plotted_any_dyn:
            ax_dyn.set_xlabel('Epoch')
            ax_dyn.set_ylabel('Value')
            ax_dyn.grid(True, linestyle='--', alpha=0.25)
            ax_dyn.legend(loc='best')
        else:
            ax_dyn.text(0.5, 0.5, 'No LR / fitness columns in results.csv', ha='center', va='center')
        charts.append(('Training dynamics', _save_fig(fig_dyn, 'dynamics.png')))
        if results_png is not None and results_png.is_file():
            extra_pages.append(('Ultralytics results', results_png))
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        run_info_rows = [('Model', str(model_name)), ('Dataset', str(dataset_name)), ('Device', str(device)), ('Image size', f'{imgsz}px'), ('Epochs (run / requested)', f"{final_metrics['epochs_ran']} / {requested_epochs}"), ('Generated at', now_str)]
        metrics_rows: list[tuple[str, str]] = []
        if task_type == 'detect':
            metrics_rows.extend([('Final train box loss', fmt(train_box_loss, 5)), ('Final val box loss', fmt(val_box_loss, 5)), ('Best mAP50', fmt(best_map50, 4)), ('Best mAP50-95', fmt(best_map5095, 4)), ('Best precision', fmt(best_precision, 4)), ('Best recall', fmt(best_recall, 4))])
        elif task_type == 'classify':
            metrics_rows.extend([('Final train loss', fmt(train_loss_generic, 5)), ('Final val loss', fmt(val_loss_generic, 5)), ('Best accuracy top1', fmt(best_acc1, 4)), ('Best accuracy top5', fmt(best_acc5, 4))])
        else:
            metrics_rows.extend([('Final train box loss', fmt(train_box_loss, 5)), ('Final val box loss', fmt(val_box_loss, 5)), ('Final train loss', fmt(train_loss_generic, 5)), ('Final val loss', fmt(val_loss_generic, 5))])
        run_id = f'{model_name}_{device}_{dataset_name}_{imgsz}'
        try:
            _pdf_model_type = 'yolo_classify' if task_type == 'classify' else 'yolo_detect'
            build_training_report_pdf(output_pdf=pdf_path, resource_root=CODE_DIR, run_id=str(run_id), model_name=str(model_name), dataset_name=str(dataset_name), device=str(device), imgsz=int(imgsz), run_info_rows=run_info_rows, metrics_rows=metrics_rows, charts=charts, extra_pages=extra_pages, notes=[f'results.csv: {csv_path}', f'run folder: {project_dir}'], model_type=_pdf_model_type)
        except Exception as e:
            self._cprint(f'⚠️ Σφάλμα δημιουργίας πλούσιου PDF: {e}', Colors.YELLOW)
        try:
            if model_output_dir is not None:
                import json
                from datetime import datetime
                import shutil
                model_output_dir = Path(model_output_dir)
                metrics_dir = model_output_dir / 'metrics'
                metrics_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if Path(csv_path).is_file():
                        shutil.copy2(csv_path, metrics_dir / Path(csv_path).name)
                except Exception:
                    pass
                charts_dir = metrics_dir / 'charts'
                charts_dir.mkdir(parents=True, exist_ok=True)

                def _copy_img(src: Path) -> str | None:
                    try:
                        if not src or not Path(src).is_file():
                            return None
                        dst = charts_dir / Path(src).name
                        if dst.exists():
                            stem = dst.stem
                            suf = dst.suffix
                            i = 2
                            while (charts_dir / f'{stem}_{i}{suf}').exists():
                                i += 1
                            dst = charts_dir / f'{stem}_{i}{suf}'
                        shutil.copy2(src, dst)
                        return dst.name
                    except Exception:
                        return None
                charts_out = []
                for t, p in list(charts) + list(extra_pages):
                    name = _copy_img(Path(p))
                    if name:
                        charts_out.append({'title': str(t), 'file': name})
                payload = {'kind': 'training', 'generated_at': datetime.now().isoformat(timespec='seconds'), 'run_id': str(run_id), 'model_name': str(model_name), 'dataset_name': str(dataset_name), 'device': str(device), 'imgsz': int(imgsz), 'final_metrics': dict(final_metrics or {}), 'report_pdf': pdf_path.name, 'report_pdf_path': str(pdf_path), 'results_csv': Path(csv_path).name, 'charts': charts_out}
                with open(metrics_dir / 'training_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._cprint(f'📄 Δημιουργήθηκε αναφορά PDF εκπαίδευσης: {pdf_path.name}', Colors.GREEN)
        return pdf_path

    def run(self) -> None:
        try:
            overall_start_time = time.time()
            startup_end_time = None
            dataset_start_time = None
            dataset_end_time = None
            train_start_time = None
            train_end_time = None
            eval_start_time = None
            eval_end_time = None
            device_label = 'GPU' if str(self.device).lower().startswith('cuda') else 'CPU'
            run_id = f'{self.model_name}_{device_label}_{self.dataset_name}_{self.imgsz}'
            sep = '═' * 72
            sub_sep = '─' * 72
            self._cprint(sep, Colors.HEADER, bold=True)
            self._cprint(f'🚀 ΕΝΑΡΞΗ ΕΚΠΑΙΔΕΥΣΗΣ ΜΟΝΤΕΛΟΥ: {run_id}', Colors.CYAN, bold=True)
            self._cprint(sub_sep, Colors.HEADER, bold=True)
            self.remove_old_runs(self.project_prefix)
            self.project_dir.mkdir(parents=True, exist_ok=True)
            try:
                device_type = 'GPU' if 'cuda' in str(self.device).lower() else 'CPU'
                run_name = f'{self.model_name}_{device_type}_{self.dataset_name}_{self.imgsz}'
                old_trained_dir = MODELS_DIR_TRAINED_PT / run_name
                if old_trained_dir.is_dir():
                    best_effort_rmtree(old_trained_dir)
                    self._cprint(f'🗑️ Διαγράφηκε παλιός φάκελος εκπαιδευμένου μοντέλου: {old_trained_dir}', Colors.YELLOW)
            except Exception as e:
                self._cprint(f'⚠️ Αποτυχία διαγραφής παλιού εκπαιδευμένου μοντέλου: {e}', Colors.YELLOW)
            self._setup_file_logging()
            self._log_training_environment()
            self._log_training_flow()
            if self._stop_requested:
                raise StopIteration('User cancelled at start')
            self.set_seed(42)
            startup_end_time = time.time()
            dataset_start_time = startup_end_time
            is_classification_model = isinstance(self.model_name, str) and self.model_name.endswith('-cls')
            yaml_path: Path | None = None
            if is_classification_model and self.config is None:
                dataset_root = DATASETS_DIR / self.dataset_name
                train_dir = dataset_root / 'train'
                val_dir = dataset_root / 'val'
                if not train_dir.is_dir() or not val_dir.is_dir():
                    self.error.emit(f'Το dataset ταξινόμησης δεν είναι σε μορφή:\n  {dataset_root}/train/<class>/...\n  {dataset_root}/val/<class>/...')
            elif isinstance(self.config, DatasetConfig):
                ds_name = getattr(self.config, 'name', None) or str(self.dataset_name)
                ds_name_lower = str(ds_name).lower()
                if ds_name_lower == 'coco':
                    self._cprint('📥 Επιλέχθηκε πλήρες COCO dataset.\n   Η Ultralytics μπορεί τώρα να κατεβάσει μεγάλα αρχεία (GB) κατά την πρώτη εκτέλεση του training.\n   Αυτό μπορεί να διαρκέσει αρκετά λεπτά ανάλογα με τη σύνδεσή σου.\nℹ️ Η αναλυτική πρόοδος σε MB/GB φαίνεται στην κονσόλα.\n   Στο Log Εκπαίδευσης εμφανίζονται βασικά μηνύματα λήψης.', Colors.LIGHT)
                    self.ensure_dataset(self.config)
                    if not self._prepare_full_coco_dataset():
                        return
                    yaml_path = (DATASETS_DIR / 'coco' / 'coco.yaml')
                    if not yaml_path.is_file():
                        self.error.emit(f"❌ Δεν βρέθηκε το τοπικό COCO YAML στο: {yaml_path}")
                else:
                    self.ensure_dataset(self.config)
                    if self.config.yaml_name:
                        dataset_root_cfg: Path | None = None
                        train_dir_cfg: Path | None = None
                        try:
                            if self.config.train_dir:
                                train_dir_cfg = Path(self.config.train_dir)
                        except Exception:
                            train_dir_cfg = None
                        if train_dir_cfg is not None:
                            p = train_dir_cfg
                            try:
                                while True:
                                    if p.parent == DATASETS_DIR:
                                        dataset_root_cfg = p
                                        break
                                    if p.parent == p:
                                        break
                                    p = p.parent
                            except Exception:
                                dataset_root_cfg = None
                        if dataset_root_cfg is None:
                            if getattr(self.config, 'name', None):
                                dataset_root_cfg = DATASETS_DIR / self.config.name
                            else:
                                dataset_root_cfg = DATASETS_DIR
                        yaml_path = dataset_root_cfg / self.config.yaml_name
            else:
                yaml_path = DATASETS_DIR / f'{self.dataset_name}.yaml'
                if not yaml_path.is_file():
                    self.error.emit(f"Δεν βρέθηκε YAML για το dataset '{self.dataset_name}'. Περίμενα: {yaml_path}")
            if self._stop_requested:
                raise StopIteration('User cancelled before training')
            dataset_end_time = time.time()
            model_file = f'{self.model_name}.pt'
            model_file_path = MODELS_DIR_INITIAL / model_file
            self._cprint('\n🏁 Έναρξη εκπαίδευσης...', Colors.GREEN)
            self._cprint(f'Μοντέλο: {self.model_name.upper()}', Colors.CYAN)
            self._cprint(f'Dataset: {self.dataset_name.upper()}', Colors.CYAN)
            self._cprint(f'Image Size: {self.imgsz}', Colors.CYAN)
            self._cprint(f'Συσκευή: {self.device.upper()}', Colors.CYAN)
            self._cprint(f'Εποχές: {self.epochs}', Colors.CYAN)
            self._cprint(f'Patience: {self.patience}', Colors.CYAN)
            if yaml_path is not None:
                self._cprint(f'YAML path: {yaml_path}', Colors.YELLOW)
            else:
                ds_name_for_log = str(self.dataset_name).upper()
                if isinstance(self.config, DatasetConfig):
                    try:
                        ds_name_for_log = str(getattr(self.config, 'name', self.dataset_name)).upper()
                    except Exception:
                        pass
                if ds_name_for_log.upper() == 'COCO':
                    self._cprint('YAML path: coco.yaml (Ultralytics built-in)', Colors.YELLOW)
                else:
                    self._cprint('YAML path: N/A (task ταξινόμησης χωρίς YAML)', Colors.YELLOW)
            if self.use_triton and str(self.device).lower().startswith('cuda') and self.triton_available:
                self._cprint(f"🔥 TRITON ΕΝΕΡΓΟ (λειτουργία: {self.compile_mode or 'Προεπιλογή'})", Colors.MAGENTA, bold=True)
            else:
                self._cprint('⚡ TRITON ΑΝΕΝΕΡΓΟ: Κανονική εκπαίδευση χωρίς compile.', Colors.YELLOW)
                self.use_triton = False
            train_start_time = time.time()
            pre_existing_root_pts = {p.name for p in Path.cwd().glob('*.pt')}
            if model_file_path.exists():
                self._cprint(f'✅ Τοπικό βάρος βρέθηκε: {model_file_path}.', Colors.GREEN)
                model_source = str(model_file_path)
            else:
                self._cprint(f'⚠️ Δεν βρέθηκε τοπικό {model_file}. Θα κατέβει αυτόματα από Ultralytics.', Colors.YELLOW)
                model_source = self.model_name
            model = YOLO(model_source)
            model.add_callback('on_fit_epoch_end', self.on_epoch_end)
            model.add_callback('on_model_save', self.on_model_save)
            model.add_callback('on_train_start', self.on_train_start_safe)
            self.progress_mutex.lock()
            self.current_progress = 20
            self.progress_mutex.unlock()
            if is_classification_model and isinstance(self.config, DatasetConfig) and self.config.train_dir:
                cls_root = Path(self.config.train_dir).parent
                data_arg = str(cls_root)
            elif is_classification_model:
                data_arg = str(DATASETS_DIR / self.dataset_name)
            else:
                ds_name = None
                if isinstance(self.config, DatasetConfig):
                    ds_name = getattr(self.config, 'name', None)
                if not ds_name:
                    ds_name = str(self.dataset_name)
                data_arg = str(yaml_path)
            compile_flag: bool = False
            if self.use_triton and self.triton_available and str(self.device).lower().startswith('cuda'):
                mapping = {'Προεπιλογή': True, 'Μείωση επιβάρυνσης': 'reduce-overhead', 'Μέγιστος αυτόματος συντονισμός': 'max-autotune'}
                selected = mapping.get(self.compile_mode, True)
                compile_flag = bool(selected)
            amp_flag: bool = False
            try:
                dev = str(self.device).lower()
                amp_flag = dev.startswith('cuda') or dev.isdigit()
            except Exception:
                amp_flag = False

            def _ensure_amp_probe_weight() -> bool:
                try:
                    import shutil
                    target = Path.cwd() / 'yolo11n.pt'
                    if target.is_file():
                        return True
                    src = MODELS_DIR_INITIAL / 'yolo11n.pt'
                    if src.is_file():
                        try:
                            shutil.copyfile(src, target)
                            return target.is_file()
                        except Exception:
                            pass
                    try:
                        from ultralytics.utils.downloads import attempt_download_asset
                        MODELS_DIR_INITIAL.mkdir(parents=True, exist_ok=True)
                        _dl = Path(attempt_download_asset(str(src)))
                        if _dl.is_file() and not src.is_file():
                            try:
                                shutil.copyfile(_dl, src)
                            except Exception:
                                pass
                        if src.is_file():
                            try:
                                shutil.copyfile(src, target)
                            except Exception:
                                pass
                        elif _dl.is_file():
                            try:
                                shutil.copyfile(_dl, target)
                            except Exception:
                                pass
                        return target.is_file()
                    except Exception as e:
                        self._cprint(f"⚠️ AMP: Αδυναμία εύρεσης/λήψης 'yolo11n.pt' για AMP check ({e}).", Colors.YELLOW)
                        return False
                except Exception:
                    return False
            if amp_flag:
                if not _ensure_amp_probe_weight():
                    amp_flag = False
                    self._cprint("⚠️ AMP απενεργοποιήθηκε αυτόματα για να αποφευχθεί crash (λείπει 'yolo11n.pt').", Colors.YELLOW)
            train_kwargs = {
                'data': data_arg,
                'epochs': int(self.epochs),
                'patience': int(self.patience),
                'imgsz': int(self.imgsz),
                'device': self.device,
                'project': str(ROOT_DIR / self.project_prefix),
                'name': self.project_name,
                'exist_ok': True,
                'save': True,
                'verbose': True,
                'amp': bool(amp_flag),
                'compile': compile_flag,
            }
            if isinstance(self.extra_hparams, dict):
                safe_hparams = {}
                for k, v in self.extra_hparams.items():
                    if v is None:
                        continue
                    safe_hparams[str(k)] = v
                train_kwargs.update(safe_hparams)
            try:
                _classes = train_kwargs.get('classes', None)
                if isinstance(_classes, str):
                    s = _classes.strip().lower()
                    if s in ('', '[]', 'none', 'null'):
                        train_kwargs.pop('classes', None)
                        self._cprint('ℹ️ Φίλτρο κλάσεων (classes): κενό → απενεργοποιείται (όλες οι κλάσεις).', Colors.CYAN)
                elif isinstance(_classes, (list, tuple, set)):
                    if not _classes:
                        train_kwargs.pop('classes', None)
                        self._cprint('ℹ️ Φίλτρο κλάσεων (classes): καμία επιλογή → απενεργοποιείται (όλες οι κλάσεις).', Colors.CYAN)
                    else:
                        _fixed = []
                        for x in list(_classes):
                            try:
                                _fixed.append(int(x))
                            except Exception:
                                pass
                        if _fixed:
                            train_kwargs['classes'] = _fixed
                            self._cprint(f'🎯 Φίλτρο κλάσεων ενεργό (classes): {_fixed}', Colors.CYAN)
            except Exception:
                pass
            try:
                if getattr(sys, 'frozen', False):
                    prev_workers = train_kwargs.get('workers', None)
                    train_kwargs['workers'] = 0
                    if prev_workers not in (None, 0):
                        self._cprint(f'ℹ️ Για λόγους σταθερότητας (Installer/Exe), οι DataLoader workers κλειδώνονται σε 0 (ήταν {prev_workers}).', Colors.YELLOW)
            except Exception:
                pass
            try:
                results = model.train(**train_kwargs)
            except Exception as e:
                emsg = str(e)
                if bool(train_kwargs.get('amp')) and ('yolo11n.pt' in emsg) and ('No such file' in emsg or 'Errno 2' in emsg or 'not found' in emsg.lower()):
                    self._cprint(f"⚠️ AMP check failed (λείπει 'yolo11n.pt'). Συνεχίζω χωρίς AMP...", Colors.YELLOW)
                    train_kwargs['amp'] = False
                    results = model.train(**train_kwargs)
                elif bool(train_kwargs.get('compile')) and self.use_triton:
                    self._cprint(f'⚠️ Σφάλμα κατά την εκπαίδευση με torch.compile: {e}', Colors.YELLOW)
                    self._cprint('🔁 Επαναπροσπάθεια εκπαίδευσης χωρίς torch.compile...', Colors.YELLOW)
                    self.use_triton = False
                    train_kwargs['compile'] = False
                    results = model.train(**train_kwargs)
                else:
                    raise
            if self._stop_requested:
                raise StopIteration('User cancelled')
            self.progress_mutex.lock()
            self.current_progress = 90
            self.progress_mutex.unlock()
            train_end_time = time.time()
            for p in Path.cwd().glob('*.pt'):
                if p.name not in pre_existing_root_pts and re.match('^yolo.*\\.pt$', p.name, re.IGNORECASE):
                    try:
                        MODELS_DIR_INITIAL.mkdir(parents=True, exist_ok=True)
                        dst = MODELS_DIR_INITIAL / p.name
                        if not dst.exists():
                            shutil.copyfile(p, dst)
                            self._cprint(f'💾 Αποθηκεύτηκε base weight στο Base_Models: {dst.name}', Colors.GREEN)
                    except Exception:
                        pass
                    try:
                        p.unlink(missing_ok=True)
                        self._cprint(f'🧹 Καθαρισμός: {p.name}', Colors.MAGENTA)
                    except Exception:
                        pass
            best_pt = self.project_dir / 'weights' / 'best.pt'
            device_type = 'GPU' if 'cuda' in str(self.device).lower() else 'CPU'
            target_dir: Path | None = None
            if best_pt.exists():
                self._cprint(f'✅ Εκπαίδευση ολοκληρώθηκε! Καλύτερο μοντέλο: {best_pt.name} ({device_type}).', Colors.GREEN, bold=True)
                try:
                    run_name = f'{self.model_name}_{device_type}_{self.dataset_name}_{self.imgsz}'
                    target_dir = MODELS_DIR_TRAINED_PT / run_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_pt = target_dir / f'{run_name}.pt'
                    shutil.copy2(best_pt, target_pt)
                    self._cprint(f'📁 Το εκπαιδευμένο μοντέλο αποθηκεύτηκε στο Trained_Models: {target_pt}', Colors.CYAN)
                except Exception as e:
                    self._cprint(f'⚠️ Δεν μπόρεσα να αντιγράψω το best.pt στον φάκελο Trained_Models: {e}', Colors.YELLOW)
            else:
                self._cprint('⚠️ Η εκπαίδευση ολοκληρώθηκε, αλλά δεν βρέθηκε best.pt. Έλεγξε τα logs.', Colors.YELLOW)
            eval_start_time = time.time()
            csv_path = self.project_dir / 'results.csv'
            final_metrics = {}
            if csv_path.is_file():
                try:
                    pdf_path = self.plot_metrics(csv_path, self.project_dir, self.model_name, self.dataset_name, self.device, self.imgsz, final_metrics, requested_epochs=self.epochs, model_output_dir=target_dir)
                    if pdf_path:
                        self.report_ready.emit(str(pdf_path))
                except Exception as e:
                    self._cprint(f'⚠️ Αποτυχία δημιουργίας PDF αναφοράς: {e}', Colors.YELLOW)
            eval_end_time = time.time()
            overall_end_time = time.time()
            try:
                self._print_timing_summary(overall_start_time, overall_end_time, startup_end_time, dataset_start_time, dataset_end_time, train_start_time, train_end_time, eval_start_time, eval_end_time)
            except Exception:
                pass
            try:
                info = perform_smart_memory_cleanup('Μετά την εκπαίδευση')
                self._cprint(f'🧠 Καθαρισμός μνήμης: {info}', Colors.MAGENTA)
            except Exception:
                self._cprint('⚠️ Αποτυχία έξυπνου καθαρισμού μνήμης (training)', Colors.YELLOW)
            sep = '═' * 72
            self._cprint('\n' + sep, Colors.HEADER, bold=True)
            self._cprint('✅ ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!', Colors.GREEN, bold=True)
            self._cprint(sep + '\n', Colors.HEADER, bold=True)
        except StopIteration:
            self._cprint('\n🛑 Η εκπαίδευση διακόπηκε από τον χρήστη.', Colors.YELLOW, bold=True)
        except Exception as e:
            tb = traceback.format_exc()
            self._log_exc('Εκπαίδευση YOLO', e, extra={
                'Μοντέλο': getattr(self, 'model_name', '?'),
                'Dataset': getattr(self, 'dataset_name', '?'),
                'Device':  getattr(self, 'device', '?'),
                'Epochs':  getattr(self, 'epochs', '?'),
            })
            if self.file_logger:
                try:
                    self.file_logger.error('Training error', exc_info=True)
                except Exception:
                    pass
        finally:
            self.finished.emit()


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 13 – CNN helpers (CNNTrainingWorker, CNNInferenceHelper)
# ═══════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# CNNTrainingWorker – QObject worker για εκπαίδευση CNN μοντέλων (torchvision)
# ════════════════════════════════════════════════════════════════════════════════
# Χρησιμοποιεί PyTorch native training loop (ΟΧΙ Ultralytics).
# Εκπέμπει τα ίδια signals με TrainingWorker για συμβατότητα με το GUI.
# Λειτουργίες:
#   - ImageFolder datasets (train/<class>/... val/<class>/...)
#   - Pretrained torchvision weights (ImageNet DEFAULT)
#   - CrossEntropyLoss + Adam/AdamW/SGD + CosineAnnealingLR
#   - Early stopping (patience)
#   - Αποθήκευση checkpoint με class_names + state_dict
#   - PDF αναφορά (loss curves + accuracy curves)
# Logging: ίδια μορφοποίηση με TrainingWorker (separators, epoch table, timing)
# ════════════════════════════════════════════════════════════════════════════════
class CNNTrainingWorker(QObject, LogEmitMixin):
    """
    QObject worker για εκπαίδευση CNN μοντέλων (torchvision: MobileNet/ResNet).
    Εκπέμπει τα ίδια signals με τον TrainingWorker ώστε το GUI να λειτουργεί
    χωρίς αλλαγές στη λογική σύνδεσης.
    Το logging ακολουθεί ακριβώς την ίδια μορφοποίηση με τον YOLO TrainingWorker.
    """
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    progress = Signal(int, str)
    report_ready = Signal(str)

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        imgsz: int,
        device: str,
        epochs: int,
        patience: int,
        extra_hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name    = str(model_name)
        self.dataset_name  = str(dataset_name)
        self.imgsz         = int(imgsz)
        self.device        = str(device)
        self.epochs        = int(epochs)
        self.patience      = int(patience)
        self.extra_hparams: dict[str, Any] = extra_hparams or {}
        self._stop_requested = False
        device_label = "CUDA" if str(self.device).lower().startswith("cuda") else "CPU"
        self.project_prefix = f"Runs_{self.model_name}_{device_label}"
        self.project_name   = f"CNN_{self.model_name}_{self.dataset_name}_imgsz{self.imgsz}"
        self.project_dir    = ROOT_DIR / self.project_prefix / self.project_name
        self.log_file       = self.project_dir / "training.log"
        self.file_logger: logging.Logger | None = None
        self.progress_mutex = QMutex()
        self.current_progress: int = 0
        self.best_top1: float | None = None
        self.last_epoch_logged: int = -1

    # ─── signal helpers ──────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_requested = True

    def _parse_progress_from_text(self, text: str):
        """Ίδια λογική με TrainingWorker – εξάγει progress % από plain text."""
        if not text:
            return None
        t = ' '.join(str(text).split())
        m = re.search(r'(\d{1,3})\s*%', t)
        if m:
            try:
                pct = max(0, min(100, int(m.group(1))))
                return (pct, t)
            except Exception:
                return None
        m = re.search(r'epoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)', t, re.IGNORECASE)
        if m:
            try:
                cur, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    return (max(0, min(100, int(cur * 100 / total))), t)
            except Exception:
                pass
        return None

    def _cprint(self, text: str, color: str = Colors.LIGHT,
                bold: bool = False, underline: bool = False,
                inline: bool = False) -> None:
        """Ταυτόσημο με TrainingWorker._cprint – HTML log + progress parse + file log."""
        try:
            html_text = format_html_log(text, color, bold=bold, underline=underline)
            if inline:
                html_text = "__INLINE__" + html_text
            self.log.emit(html_text)
            try:
                plain = re.sub(r"<[^>]+>", " ", html_text)
                plain = plain.replace("&nbsp;", " ").replace("&amp;", "&")
                parsed = self._parse_progress_from_text(plain)
                if parsed:
                    pct, pmsg = parsed
                    self.progress.emit(int(pct), str(pmsg))
            except Exception:
                pass
        except Exception:
            try:
                self.log.emit(text)
            except Exception:
                pass
        try:
            if self.file_logger:
                self.file_logger.info(text)
        except Exception:
            pass

    # ─── file logging ────────────────────────────────────────────────────

    def _setup_file_logging(self) -> None:
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger(f'cnn_training_{id(self)}')
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fmt = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            self.file_logger = logger
        except Exception:
            self.file_logger = None

    # ─── environment / flow banners (ίδια μορφή με YOLO) ─────────────────

    # ── Banner περιβάλλοντος εκπαίδευσης ────────────────────────────────────────
    # Εμφανίζει: Python version, PyTorch version, torchvision version, GPU info.
    # Ίδια μορφοποίηση με YOLO TrainingWorker._log_training_environment().
    def _log_training_environment(self) -> None:
        """Εκτυπώνει banner περιβάλλοντος – ίδια μορφή με YOLO TrainingWorker."""
        try:
            python_ver = platform.python_version()
            torch_ver  = getattr(torch, "__version__", "N/A")
            try:
                import torchvision as _tv
                tv_ver = getattr(_tv, "__version__", "N/A")
            except ImportError:
                tv_ver = "N/A"
            cuda_available = torch.cuda.is_available()
            gpu_name: str | None = None
            gpu_mem:  str | None = None
            if cuda_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    total    = torch.cuda.get_device_properties(0).total_memory
                    gpu_mem  = f"{total / 1024 ** 3:.1f} GB"
                except Exception:
                    pass
            self._cprint("=== ΠΕΡΙΒΑΛΛΟΝ ΕΚΠΑΙΔΕΥΣΗΣ CNN ΜΟΝΤΕΛΟΥ ===",
                         Colors.HEADER, bold=True)
            self._cprint(
                f"Python: {python_ver} | PyTorch: {torch_ver} | torchvision: {tv_ver}",
                Colors.LIGHT)
            if cuda_available and gpu_name:
                self._cprint(
                    f"🟢 CUDA διαθέσιμο – GPU: {gpu_name} ({gpu_mem})", Colors.GREEN)
            elif cuda_available:
                self._cprint("🟢 CUDA διαθέσιμο", Colors.GREEN)
            else:
                self._cprint(
                    "⚠️ CUDA ΜΗ διαθέσιμο – εκπαίδευση σε CPU.", Colors.YELLOW)
        except Exception as e:
            _MMPRO_LOGGER.debug("CNN _log_training_environment error: %s", e)

    def _log_training_flow(self) -> None:
        """Δέντρο ροής εκπαίδευσης – ίδια μορφή με YOLO TrainingWorker."""
        try:
            lines = [
                '🌳 Δομή λογικής ροής CNN εκπαίδευσης',
                '└─ 🧠 Κύρια διαδικασία: CNN Training (GUI)',
                '   ├─ 📁 Έλεγχος dataset (train/<class>/… val/<class>/…)',
                '   ├─ 🧪 Έλεγχος περιβάλλοντος Python / PyTorch / torchvision / CUDA',
                '   ├─ 🧹 Προετοιμασία μνήμης (GC / CUDA cache, seed)',
                '   ├─ 📥 Φόρτωση dataset (ImageFolder + transforms)',
                '   ├─ 🏋️ Φόρτωση pretrained CNN μοντέλου (torchvision)',
                '   ├─ ⚙️ Ρύθμιση optimizer / scheduler / loss',
                '   ├─ 🏋️ Εκπαίδευση μοντέλου (training epochs + early stopping)',
                '   └─ 📊 Αξιολόγηση & παραγωγή αναφοράς (metrics, γραφήματα, logs)',
            ]
            for line in lines:
                self._cprint(line, Colors.CYAN)
        except Exception:
            pass

    def _print_timing_summary(
        self,
        overall_start: float, overall_end: float,
        startup_end:   float,
        dataset_start: float, dataset_end: float,
        train_start:   float, train_end: float,
        eval_start:    float, eval_end:   float,
    ) -> None:
        """Πίνακας χρονομέτρησης – ακριβώς ίδια μορφή με YOLO TrainingWorker."""
        def safe_delta(s, e):
            try:
                return max(0.0, float(e) - float(s))
            except Exception:
                return None

        total_t   = safe_delta(overall_start, overall_end)
        startup_t = safe_delta(overall_start, startup_end)
        dataset_t = safe_delta(dataset_start, dataset_end)
        train_t   = safe_delta(train_start,   train_end)
        eval_t    = safe_delta(eval_start,    eval_end)

        self._cprint('\n⏱️ ΧΡΟΝΟΜΕΤΡΗΣΗ ΕΚΠΑΙΔΕΥΣΗΣ (πραγματικός χρόνος)',
                     Colors.HEADER, bold=True, underline=True)
        lw, hw, sw = 40, 10, 12
        header    = f"{'Στάδιο':<{lw}} | {'Διάρκεια':^{hw}} | {'Σύνολο (sec)':^{sw}}"
        separator = '-' * len(header)
        self._cprint(separator, Colors.LIGHT)
        self._cprint(header, Colors.LIGHT, bold=True)
        self._cprint(separator, Colors.LIGHT)

        def fmt_row(label, val, highlight=False):
            if val is None:
                return
            mins, secs = divmod(float(val), 60)
            hours, mins = divmod(int(mins), 60)
            hms  = f'{hours:02d}:{mins:02d}:{int(secs):02d}'
            text = f'{label:<{lw}} | {hms:>{hw}} | {float(val):>{sw}.1f}'
            self._cprint(text,
                         Colors.GREEN if highlight else Colors.LIGHT,
                         bold=highlight)

        fmt_row('Χρόνος εκκίνησης:',              startup_t)
        fmt_row('Χρόνος προετοιμασίας dataset:',   dataset_t)
        fmt_row('Χρόνος εκπαίδευσης εποχών:',      train_t)
        fmt_row('Χρόνος αξιολόγησης μοντέλου:',    eval_t)
        fmt_row('Συνολικός χρόνος:',               total_t, highlight=True)

    # ─── epoch / model-save callbacks ────────────────────────────────────

    # ── Log τέλους epoch – ίδια μορφή με YOLO on_epoch_end callback ─────────────
    # Εκπέμπει γραμμή: "--- Ολοκληρώθηκε η Εποχή N/T (P%) – loss=X | Top-1=X ---"
    # Χρησιμοποιείται για συνέπεια εμφάνισης μεταξύ YOLO και CNN εκπαίδευσης.
    def _on_epoch_end(self, epoch: int, total_epochs: int, prog: int,
                      train_loss: float, val_loss: float,
                      acc_top1: float, acc_top5: float,
                      curr_lr: float, ep_time: float) -> None:
        """
        Μορφοποιεί το τέλος κάθε εποχής ακριβώς όπως το YOLO on_epoch_end:
        --- Ολοκληρώθηκε η Εποχή N/T (Progress: P%) – loss=X | val_loss=X | Top-1=X | Top-5=X ---
        """
        if epoch == self.last_epoch_logged:
            return
        self.last_epoch_logged = epoch

        metric_parts = [
            f'loss={train_loss:.4f}',
            f'val_loss={val_loss:.4f}',
            f'Top-1={acc_top1:.4f}',
            f'Top-5={acc_top5:.4f}',
            f'lr={curr_lr:.6f}',
            f'time={ep_time:.1f}s',
        ]
        metrics_str = ' | '.join(metric_parts)
        self._cprint(
            f'--- Ολοκληρώθηκε η Εποχή {epoch}/{total_epochs} '
            f'(Progress: {prog}%) – {metrics_str} ---',
            Colors.MAGENTA)

    # ── Banner νέου καλύτερου CNN μοντέλου ──────────────────────────────────────
    # Εμφανίζει: "ΝΕΟ ΚΑΛΥΤΕΡΟ CNN ΜΟΝΤΕΛΟ (Epoch: N)! | Top-1: X.XXXX"
    # Ίδια μορφοποίηση με YOLO TrainingWorker.on_model_save().
    def _on_new_best(self, epoch: int, acc_top1: float) -> None:
        """
        Banner νέου καλύτερου μοντέλου – ακριβώς ίδια μορφή με YOLO on_model_save.
        """
        self._cprint(f"\n{'=' * 70}", Colors.GREEN, bold=True)
        self._cprint(f'ΝΕΟ ΚΑΛΥΤΕΡΟ CNN ΜΟΝΤΕΛΟ (Epoch: {epoch})!',
                     Colors.YELLOW, bold=True)
        self._cprint(f'   Top-1 Accuracy: {acc_top1:.4f}', Colors.GREEN)
        self._cprint(f"{'=' * 70}\n", Colors.GREEN, bold=True)

    # ─── seed ────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int = 42) -> None:
        try:
            random.seed(seed)
            import numpy as _np_s
            _np_s.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass

    # ─── main training loop ───────────────────────────────────────────────

    def run(self) -> None:
        """Κεντρικός βρόχος CNN εκπαίδευσης (torchvision), ίδια ροή με YOLO."""
        overall_start = time.time()
        startup_end   = None
        dataset_start = None
        dataset_end   = None
        train_start   = None
        train_end     = None
        eval_start    = None
        eval_end      = None

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
        except ImportError as e:
            self.error.emit(f"Λείπει το PyTorch: {e}")
            self.finished.emit()
            return
        try:
            from torchvision.datasets import ImageFolder
        except ImportError as e:
            self.error.emit(
                f"Λείπει το torchvision: {e}\nΕγκατάστησέ το: pip install torchvision")
            self.finished.emit()
            return

        device_label = 'GPU' if str(self.device).lower().startswith('cuda') else 'CPU'
        run_id = f'{self.model_name}_{device_label}_{self.dataset_name}_{self.imgsz}'

        # ── Header (ίδιο με YOLO) ─────────────────────────────────────────
        sep     = '═' * 72
        sub_sep = '─' * 72
        self._cprint(sep,       Colors.HEADER, bold=True)
        self._cprint(f'🧠 ΕΝΑΡΞΗ CNN ΕΚΠΑΙΔΕΥΣΗΣ ΜΟΝΤΕΛΟΥ: {run_id}',
                     Colors.CYAN, bold=True)
        self._cprint(sub_sep,   Colors.HEADER, bold=True)

        self._setup_file_logging()
        self._log_training_environment()
        self._log_training_flow()

        if self._stop_requested:
            self._cprint('\n🛑 Η εκπαίδευση διακόπηκε από τον χρήστη.',
                         Colors.YELLOW, bold=True)
            self.finished.emit()
            return

        self._set_seed(42)
        startup_end   = time.time()
        dataset_start = startup_end

        # ── Επαλήθευση δομής dataset (train/<class>/... val/<class>/...) ─────────
        dataset_root = DATASETS_DIR / self.dataset_name
        train_dir = dataset_root / 'train'
        val_dir   = dataset_root / 'val'
        if not train_dir.is_dir() or not val_dir.is_dir():
            self.error.emit(
                f"Δεν βρέθηκε dataset ταξινόμησης στο:\n  {dataset_root}\n"
                f"Απαιτείται δομή: {dataset_root}/train/<class>/… "
                f"και {dataset_root}/val/<class>/…")
            self.finished.emit()
            return

        try:
            train_tf = _cnn_get_transforms(self.imgsz, train=True)
            val_tf   = _cnn_get_transforms(self.imgsz, train=False)
            train_ds = ImageFolder(str(train_dir), transform=train_tf)
            val_ds   = ImageFolder(str(val_dir),   transform=val_tf)
        except Exception as e:
            self.error.emit(f"Σφάλμα φόρτωσης dataset: {e}")
            self.finished.emit()
            return

        num_classes = len(train_ds.classes)
        dataset_end = time.time()

        # ── Startup info block (ίδια μορφή με YOLO) ──────────────────────
        self._cprint('\n🏁 Έναρξη εκπαίδευσης...', Colors.GREEN)
        self._cprint(f'Μοντέλο: {self.model_name.upper()}', Colors.CYAN)
        self._cprint(f'Dataset: {self.dataset_name.upper()}', Colors.CYAN)
        self._cprint(f'Image Size: {self.imgsz}', Colors.CYAN)
        self._cprint(f'Συσκευή: {self.device.upper()}', Colors.CYAN)
        self._cprint(f'Εποχές: {self.epochs}', Colors.CYAN)
        self._cprint(f'Patience: {self.patience}', Colors.CYAN)
        self._cprint(
            f'Κλάσεις ({num_classes}): '
            f'{", ".join(train_ds.classes[:15])}'
            f'{"…" if num_classes > 15 else ""}',
            Colors.CYAN)
        self._cprint(
            f'Train εικόνες: {len(train_ds)}  |  '
            f'Val εικόνες: {len(val_ds)}',
            Colors.YELLOW)
        self._cprint('⚡ TRITON ΑΝΕΝΕΡΓΟ: CNN μοντέλα δεν υποστηρίζουν torch.compile.',
                     Colors.YELLOW)

        # ── Hyperparams ────────────────────────────────────────────────────
        # ── Εξαγωγή hyperparameters από extra_hparams (τιμές από GUI) ────────────
        # Defaults: batch=32, workers=2 (ή 0 για frozen exe), lr0=0.001, lrf=0.01
        batch    = int(self.extra_hparams.get('batch', 32))
        workers  = int(self.extra_hparams.get('workers',
                        0 if getattr(sys, 'frozen', False) else 2))
        lr0      = float(self.extra_hparams.get('lr0', 0.001))
        lrf      = float(self.extra_hparams.get('lrf', 0.01))
        wd       = float(self.extra_hparams.get('weight_decay', 1e-4))
        momentum = float(self.extra_hparams.get('momentum', 0.9))
        optname  = str(self.extra_hparams.get('optimizer', 'adam')).lower()

        self._cprint(sub_sep, Colors.LIGHT)
        self._cprint(
            f'batch={batch} | optimizer={optname} | lr0={lr0} | '
            f'lrf={lrf} | wd={wd} | momentum={momentum} | workers={workers}',
            Colors.LIGHT)
        self._cprint(sub_sep, Colors.LIGHT)

        try:
            train_loader = DataLoader(
                train_ds, batch_size=batch, shuffle=True,
                num_workers=workers, pin_memory=False)
            val_loader   = DataLoader(
                val_ds,   batch_size=batch, shuffle=False,
                num_workers=workers, pin_memory=False)
        except Exception as e:
            self.error.emit(f"Σφάλμα DataLoader: {e}")
            self.finished.emit()
            return

        # ── Model ──────────────────────────────────────────────────────────
        try:
            model_nn = _load_torchvision_model(
                self.model_name, num_classes=num_classes, pretrained=True)
        except Exception as e:
            self.error.emit(
                f'❌ Σφάλμα φόρτωσης CNN μοντέλου {self.model_name}: {e}\n'
                "Βεβαιώσου ότι έχεις: pip install torchvision")
            self.finished.emit()
            return

        self._cprint(
            f'✅ Pretrained {self.model_name.upper()} φορτώθηκε '
            f'(num_classes={num_classes}).', Colors.GREEN)

        # ── Device ─────────────────────────────────────────────────────────
        dev_str = self.device.lower()
        if dev_str.startswith('cuda') and not torch.cuda.is_available():
            self._cprint('⚠️ CUDA μη διαθέσιμο. Fallback σε CPU.', Colors.YELLOW)
            dev_str = 'cpu'
        try:
            device_t = torch.device(dev_str)
        except Exception:
            device_t = torch.device('cpu')
        model_nn = model_nn.to(device_t)

        # ── Αρχικοποίηση loss function, optimizer και scheduler ──────────────────
        # Loss: CrossEntropyLoss (κατάλληλο για multi-class classification)
        # Scheduler: CosineAnnealingLR – μειώνει το LR από lr0 → lr0*lrf
        criterion = nn.CrossEntropyLoss()
        if optname == 'sgd':
            optimizer = optim.SGD(
                model_nn.parameters(), lr=lr0, momentum=momentum, weight_decay=wd)
        elif optname == 'adamw':
            optimizer = optim.AdamW(
                model_nn.parameters(), lr=lr0, weight_decay=wd)
        else:
            optimizer = optim.Adam(
                model_nn.parameters(), lr=lr0, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=lr0 * lrf)

        # ── Output dirs ────────────────────────────────────────────────────
        self.project_dir.mkdir(parents=True, exist_ok=True)
        weights_dir  = self.project_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_pt_path = weights_dir / 'best.pt'
        last_pt_path = weights_dir / 'last.pt'

        # ── Κεντρικός βρόχος εκπαίδευσης (epochs) ────────────────────────────────
        # Σε κάθε epoch: train phase → val phase → log → check best → early stopping
        patience_ct  = 0
        best_epoch   = 0
        results_rows: list[dict] = []

        self._cprint(sub_sep, Colors.LIGHT)
        train_start = time.time()

        self.progress_mutex.lock()
        self.current_progress = 10
        self.progress_mutex.unlock()

        for epoch in range(1, self.epochs + 1):
            if self._stop_requested:
                self._cprint('\n🛑 Η εκπαίδευση διακόπηκε από τον χρήστη.',
                             Colors.YELLOW, bold=True)
                break

            # ── Φάση εκπαίδευσης: forward pass → loss → backward → optimizer.step ──
            model_nn.train()
            running_loss = 0.0
            t_ep = time.perf_counter()

            for _imgs, _labels in train_loader:
                if self._stop_requested:
                    break
                _imgs   = _imgs.to(device_t, non_blocking=True)
                _labels = _labels.to(device_t, non_blocking=True)
                optimizer.zero_grad()
                _out  = model_nn(_imgs)
                _loss = criterion(_out, _labels)
                _loss.backward()
                optimizer.step()
                running_loss += _loss.item() * _imgs.size(0)

            if self._stop_requested:
                break

            train_loss = running_loss / max(len(train_ds), 1)

            # ── Φάση αξιολόγησης: no_grad, υπολογισμός Top-1 / Top-5 accuracy ──────
            model_nn.eval()
            val_loss_sum  = 0.0
            correct_top1  = 0
            correct_top5  = 0
            total_val     = 0

            with torch.no_grad():
                for _imgs, _labels in val_loader:
                    _imgs   = _imgs.to(device_t, non_blocking=True)
                    _labels = _labels.to(device_t, non_blocking=True)
                    _out    = model_nn(_imgs)
                    _vloss  = criterion(_out, _labels)
                    val_loss_sum  += _vloss.item() * _imgs.size(0)
                    _, _p1 = _out.max(1)
                    correct_top1 += _p1.eq(_labels).sum().item()
                    if num_classes >= 5:
                        _, _p5 = _out.topk(5, dim=1)
                        correct_top5 += (
                            _p5.eq(_labels.unsqueeze(1).expand_as(_p5))
                            .any(dim=1).sum().item())
                    else:
                        correct_top5 += correct_top1
                    total_val += _labels.size(0)

            val_loss = val_loss_sum / max(len(val_ds), 1)
            acc_top1 = correct_top1 / max(total_val, 1)
            acc_top5 = correct_top5 / max(total_val, 1)
            ep_time  = time.perf_counter() - t_ep
            curr_lr  = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') \
                       else lr0
            scheduler.step()

            # ── Progress ─────────────────────────────────────────────────
            prog = 10 + int(epoch / max(self.epochs, 1) * 80)
            prog = max(10, min(90, prog))
            self.progress_mutex.lock()
            self.current_progress = prog
            self.progress_mutex.unlock()

            # ── Epoch-end log (ίδιο με YOLO on_epoch_end) ────────────────
            self._on_epoch_end(
                epoch, self.epochs, prog,
                train_loss, val_loss, acc_top1, acc_top5, curr_lr, ep_time)

            results_rows.append({
                'epoch':                   epoch,
                'train/loss':              train_loss,
                'val/loss':                val_loss,
                'metrics/accuracy_top1':   acc_top1,
                'metrics/accuracy_top5':   acc_top5,
                'lr':                      curr_lr,
            })

            # ── Έλεγχος νέου καλύτερου μοντέλου (βάσει Top-1 Accuracy) ─────────────
            is_new_best = (self.best_top1 is None) or (acc_top1 > self.best_top1)
            if is_new_best:
                self.best_top1  = acc_top1
                best_epoch      = epoch
                patience_ct     = 0
                try:
                    torch.save({
                        'epoch':        epoch,
                        'model_name':   self.model_name,
                        'num_classes':  num_classes,
                        'class_names':  train_ds.classes,
                        'imgsz':        self.imgsz,
                        'state_dict':   model_nn.state_dict(),
                        'acc_top1':     acc_top1,
                        'acc_top5':     acc_top5,
                    }, str(best_pt_path))
                    self._on_new_best(epoch, acc_top1)
                except Exception as save_e:
                    self._cprint(
                        f'⚠️ Αποτυχία αποθήκευσης best.pt: {save_e}',
                        Colors.YELLOW)
            else:
                patience_ct += 1
                if self.patience > 0 and patience_ct >= self.patience:
                    self._cprint(
                        f'⏹️ Early stopping (patience={self.patience}) '
                        f'at epoch {epoch}. '
                        f'Best acc@1={self.best_top1:.4f} @ epoch {best_epoch}',
                        Colors.YELLOW, bold=True)
                    break

            # ── last.pt ────────────────────────────────────────────────────
            try:
                torch.save({
                    'epoch':       epoch,
                    'model_name':  self.model_name,
                    'num_classes': num_classes,
                    'class_names': train_ds.classes,
                    'imgsz':       self.imgsz,
                    'state_dict':  model_nn.state_dict(),
                }, str(last_pt_path))
            except Exception:
                pass

        train_end = time.time()

        self.progress_mutex.lock()
        self.current_progress = 92
        self.progress_mutex.unlock()
        self.progress.emit(92, 'Ολοκλήρωση εκπαίδευσης…')

        # ── Save results.csv ────────────────────────────────────────────────
        csv_path = self.project_dir / 'results.csv'
        try:
            import csv as _csv
            if results_rows:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = _csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
                    w.writeheader()
                    w.writerows(results_rows)
        except Exception as csv_e:
            self._cprint(f'⚠️ Αποτυχία αποθήκευσης results.csv: {csv_e}',
                         Colors.YELLOW)

        # ── Copy best → Trained_Models ──────────────────────────────────────
        device_type = 'GPU' if 'cuda' in dev_str else 'CPU'
        target_dir: Path | None = None
        if best_pt_path.exists():
            self._cprint(
                f'✅ Εκπαίδευση ολοκληρώθηκε! Best acc@1='
                f'{self.best_top1:.4f} @ epoch {best_epoch} ({device_type}).',
                Colors.GREEN, bold=True)
            try:
                run_name   = f'{self.model_name}_{device_type}_{self.dataset_name}_{self.imgsz}'
                target_dir = MODELS_DIR_TRAINED_PT / run_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_pt  = target_dir / f'{run_name}.pt'
                shutil.copy2(best_pt_path, target_pt)
                # class names JSON
                cj = target_dir / 'class_names.json'
                try:
                    cj.write_text(
                        json.dumps({
                            'class_names': train_ds.classes,
                            'num_classes': num_classes,
                            'model_name':  self.model_name,
                            'imgsz':       self.imgsz,
                        }, ensure_ascii=False, indent=2),
                        encoding='utf-8')
                except Exception:
                    pass
                self._cprint(
                    f'📁 Αποθηκεύτηκε στο Trained_Models: {target_pt}',
                    Colors.CYAN)
            except Exception as cp_e:
                self._cprint(
                    f'⚠️ Αδυναμία αντιγραφής best.pt: {cp_e}', Colors.YELLOW)
        else:
            self._cprint('⚠️ Ολοκλήρωση χωρίς best.pt!', Colors.YELLOW)

        # ── PDF report ──────────────────────────────────────────────────────
        eval_start = time.time()
        if csv_path.is_file():
            try:
                final_metrics_cnn: dict = {}
                pdf_path_cnn = self._build_cnn_pdf_report(
                    csv_path, results_rows, train_ds.classes,
                    final_metrics_cnn, target_dir,
                    self.best_top1 or 0.0, train_end - train_start)
                if pdf_path_cnn:
                    self.report_ready.emit(str(pdf_path_cnn))
            except Exception as pdf_e:
                self._cprint(
                    f'⚠️ Αποτυχία δημιουργίας PDF αναφοράς: {pdf_e}',
                    Colors.YELLOW)
        eval_end    = time.time()
        overall_end = time.time()

        # ── FINAL METRICS summary (ίδιο στυλ με YOLO) ─────────────────────
        self._cprint(sub_sep, Colors.LIGHT)
        self._cprint('FINAL METRICS (CNN)', Colors.HEADER, bold=True, underline=True)
        self._cprint(
            f'  • Best Top-1 Accuracy : {self.best_top1:.4f}' if self.best_top1 else
            '  • Best Top-1 Accuracy : –', Colors.GREEN)
        self._cprint(
            f'  • Εποχές που εκτελέσθηκαν : {len(results_rows)}/{self.epochs}',
            Colors.LIGHT)
        self._cprint(sub_sep, Colors.LIGHT)

        # ── Timing table ────────────────────────────────────────────────────
        try:
            self._print_timing_summary(
                overall_start, overall_end,
                startup_end or overall_start,
                dataset_start or overall_start, dataset_end or overall_start,
                train_start or overall_start,   train_end or time.time(),
                eval_start or overall_start,    eval_end or time.time())
        except Exception:
            pass

        # ── Memory cleanup ──────────────────────────────────────────────────
        try:
            info = perform_smart_memory_cleanup('Μετά από CNN εκπαίδευση')
            self._cprint(f'🧠 Καθαρισμός μνήμης: {info}', Colors.MAGENTA)
        except Exception:
            pass

        # ── Footer (ίδιο με YOLO) ───────────────────────────────────────────
        self._cprint('\n' + sep, Colors.HEADER, bold=True)
        self._cprint('✅ CNN ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!',
                     Colors.GREEN, bold=True)
        self._cprint(sep + '\n', Colors.HEADER, bold=True)

        self.finished.emit()

    # ─── PDF report builder ────────────────────────────────────────────────

    def _build_cnn_pdf_report(
        self,
        csv_path: Path,
        results_rows: list[dict],
        class_names: list[str],
        final_metrics: dict,
        model_output_dir: Path | None,
        best_acc1: float,
        total_time_s: float,
    ) -> Path | None:
        """Δημιουργεί PDF αναφορά εκπαίδευσης CNN (loss + accuracy plots)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self._cprint('⚠️ Λείπει matplotlib – δεν δημιουργείται PDF.',
                         Colors.YELLOW)
            return None

        try:
            safe_model   = self.model_name.replace('/', '_')
            safe_dataset = self.dataset_name.replace('/', '_')
            reports_dir  = TRAIN_REPORTS_DIR
            reports_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = reports_dir / f'_assets_cnn_{safe_model}_{safe_dataset}_{self.imgsz}'
            assets_dir.mkdir(parents=True, exist_ok=True)
            pdf_name = f'TrainReport_CNN_{safe_model}_{safe_dataset}_{self.imgsz}.pdf'
            pdf_path = reports_dir / pdf_name

            epochs_arr = [r['epoch']                    for r in results_rows]
            tl_arr     = [r['train/loss']               for r in results_rows]
            vl_arr     = [r['val/loss']                 for r in results_rows]
            a1_arr     = [r['metrics/accuracy_top1']    for r in results_rows]
            a5_arr     = [r['metrics/accuracy_top5']    for r in results_rows]

            def _sfig(fig, name: str) -> Path:
                out = assets_dir / name
                try:
                    fig.savefig(out, dpi=180, bbox_inches='tight')
                finally:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                return out

            charts = []
            fig1, ax1 = plt.subplots(figsize=(11.0, 5.0))
            ax1.set_title('CNN Loss Curves')
            ax1.plot(epochs_arr, tl_arr, label='train/loss')
            ax1.plot(epochs_arr, vl_arr, label='val/loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.25)
            charts.append(('Loss curves', _sfig(fig1, 'cnn_loss.png')))

            fig2, ax2 = plt.subplots(figsize=(11.0, 5.0))
            ax2.set_title('CNN Accuracy')
            ax2.plot(epochs_arr, a1_arr, label='acc@1')
            ax2.plot(epochs_arr, a5_arr, label='acc@5')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1.05)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.25)
            charts.append(('Accuracy', _sfig(fig2, 'cnn_accuracy.png')))

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
            _cls_preview = ', '.join(class_names[:8]) + ('...' if len(class_names) > 8 else '')
            run_info_rows = [
                ('Τύπος μοντέλου', f'CNN torchvision ({self.model_name})'),
                ('Dataset',         self.dataset_name),
                ('Device',          self.device),
                ('Image size',      f'{self.imgsz}px'),
                ('Αριθμός κλάσεων', str(len(class_names))),
                ('Κλάσεις',         _cls_preview or '–'),
                ('Epochs ran',      str(len(results_rows))),
                ('Χρόνος εκπ/σης',  f'{total_time_s/60:.1f} min'),
                ('Generated at',    now_str),
            ]
            metrics_rows_rpt = [
                ('Best acc@1 (top-1)', f'{best_acc1:.4f}'),
                ('Best acc@5 (top-5)', f'{max(a5_arr, default=0.0):.4f}'),
                ('Final train loss',   f'{tl_arr[-1]:.5f}' if tl_arr else '–'),
                ('Final val loss',     f'{vl_arr[-1]:.5f}' if vl_arr else '–'),
                ('Training time',      f'{total_time_s/60:.1f} min'),
            ]
            final_metrics['acc_top1']   = best_acc1
            final_metrics['acc_top5']   = max(a5_arr, default=0.0)
            final_metrics['epochs_ran'] = len(results_rows)
            final_metrics['task_type']  = 'classify'

            # ── Per-class accuracy bar chart (CNN-specific) ─────────────
            try:
                if class_names and a1_arr:
                    # Τελευταία γνωστή per-class accuracy από το csv αν υπάρχει,
                    # αλλιώς placeholder με best_acc1 για κάθε κλάση
                    fig_cls, ax_cls = plt.subplots(figsize=(max(8.0, len(class_names) * 0.55), 5.5))
                    ax_cls.set_title(f'Class Names ({len(class_names)} κλάσεις)')
                    x_pos = range(len(class_names))
                    ax_cls.barh(list(x_pos), [1.0] * len(class_names),
                                color='#e2e8f0', height=0.6)
                    ax_cls.set_yticks(list(x_pos))
                    ax_cls.set_yticklabels(class_names, fontsize=max(6, 10 - len(class_names) // 8))
                    ax_cls.set_xlabel('Κλάσεις μοντέλου')
                    ax_cls.set_xlim(0, 1.1)
                    ax_cls.set_title('Κλάσεις μοντέλου (class list)')
                    ax_cls.grid(True, linestyle='--', alpha=0.2, axis='x')
                    charts.append(('Class list', _sfig(fig_cls, 'cnn_classes.png')))
            except Exception:
                pass
            try:
                build_training_report_pdf(
                    output_pdf=pdf_path,
                    resource_root=CODE_DIR,
                    run_id=f'CNN_{self.model_name}_{self.dataset_name}_{self.imgsz}',
                    model_name=self.model_name,
                    dataset_name=self.dataset_name,
                    device=self.device,
                    imgsz=self.imgsz,
                    run_info_rows=run_info_rows,
                    metrics_rows=metrics_rows_rpt,
                    charts=charts,
                    extra_pages=[],
                    model_type='cnn',
                    notes=[
                        f'results.csv: {csv_path}',
                        f'CNN model: {self.model_name}',
                        f'Classes: {", ".join(class_names[:20])}{"..." if len(class_names) > 20 else ""}',
                    ])
            except Exception as pdf_e:
                self._cprint(f'⚠️ Σφάλμα build PDF: {pdf_e}', Colors.YELLOW)

            # Copy artifacts to model output dir
            if model_output_dir is not None:
                try:
                    mdir = Path(model_output_dir) / 'metrics'
                    mdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(csv_path, mdir / csv_path.name)
                    payload = {
                        'kind':           'cnn_training',
                        'generated_at':   datetime.now().isoformat(timespec='seconds'),
                        'model_name':     self.model_name,
                        'dataset_name':   self.dataset_name,
                        'device':         self.device,
                        'imgsz':          self.imgsz,
                        'final_metrics':  final_metrics,
                        'report_pdf':     pdf_path.name,
                    }
                    (mdir / 'training_metrics.json').write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')
                except Exception:
                    pass

            self._cprint(f'📄 PDF αναφορά CNN: {pdf_path.name}', Colors.GREEN)
            return pdf_path
        except Exception as e:
            self._cprint(
                f'⚠️ Σφάλμα δημιουργίας PDF αναφοράς CNN: {e}', Colors.YELLOW)
            return None


# ════════════════════════════════════════════════════════════════════════════════
# CNNInferenceHelper – Helper για inference CNN μοντέλων (torchvision)
# ════════════════════════════════════════════════════════════════════════════════
# Φορτώνει checkpoint .pt (state_dict + class_names + imgsz) και εκτελεί
# softmax top-k classification σε BGR frames (OpenCV format).
#
# Χρήση:
#   helper = CNNInferenceHelper(model_path, device='cpu')
#   helper.load()
#   preds = helper.predict_frame(bgr_frame, top_k=5)
#   # → [(class_name, confidence), ...]
#   annotated = helper.annotate_frame(bgr_frame)
#   # → numpy BGR frame με overlay στο κάτω-αριστερό μέρος
#
# Μέθοδοι:
#   load()            – Φορτώνει checkpoint και αρχικοποιεί transforms
#   predict_frame()   – Εκτελεί inference, επιστρέφει top-k predictions
#   annotate_frame()  – Public API: predict + draw overlay
#   _draw_predictions() – Σχεδιάζει το overlay (bottom-left panel)
# ════════════════════════════════════════════════════════════════════════════════
class CNNInferenceHelper:
    """
    Helper για inference CNN μοντέλων (torchvision) σε frames κάμερας/video.
    Φορτώνει το .pt αρχείο (checkpoint dict) και εκτελεί top-k classification.
    """

    def __init__(self, model_path: Path, device: str = 'cpu') -> None:
        self.model_path  = Path(model_path)
        self.device_str  = str(device)
        self.model_nn    = None
        self._ort_session = None   # ONNX Runtime session (για .onnx αρχεία)
        self._is_onnx    = False   # True αν το μοντέλο είναι .onnx
        self.class_names: list[str] = []
        self.num_classes: int = 0
        self.imgsz: int  = 224
        self.model_name_str: str = ''
        self._transform  = None
        self._loaded     = False

    # ── Φόρτωση CNN checkpoint από .pt αρχείο ────────────────────────────────────
    # Διαβάζει: model_name, num_classes, class_names, imgsz, state_dict.
    # Αν δεν βρεθούν class_names στο checkpoint, ψάχνει στο sibling class_names.json.
    # Αρχικοποιεί το torchvision μοντέλο και τα inference transforms.
    def load(self) -> None:
        """Φορτώνει το μοντέλο από .pt checkpoint ή .onnx αρχείο."""
        suffix = self.model_path.suffix.lower()
        self._is_onnx = (suffix == '.onnx')

        if self._is_onnx:
            # ── ONNX Runtime branch ──────────────────────────────────────────
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            dev_str = self.device_str.lower()
            if dev_str.startswith('cuda'):
                try:
                    available = ort.get_available_providers()
                    if 'CUDAExecutionProvider' in available:
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                except Exception:
                    pass
            self._ort_session = ort.InferenceSession(str(self.model_path), providers=providers)
            # Ανάγνωση μεταδεδομένων μοντέλου:
            # Προτεραιότητα: *_onnx_meta.json > class_names.json
            _meta_candidates = [
                self.model_path.parent / (self.model_path.stem + '_onnx_meta.json'),
                self.model_path.parent / 'class_names.json',
            ]
            for cj in _meta_candidates:
                if cj.is_file():
                    try:
                        d = json.loads(cj.read_text(encoding='utf-8'))
                        self.class_names    = list(d.get('class_names', []))
                        self.model_name_str = str(d.get('model_name', ''))
                        self.imgsz          = int(d.get('imgsz', 224))
                        self.num_classes    = int(d.get('num_classes', len(self.class_names) or 1000))
                        break
                    except Exception:
                        pass
            # Fallback: infer από όνομα αρχείου
            if not self.model_name_str:
                stem = self.model_path.stem.lower()
                for cnn in TRAIN_CNN_MODELS:
                    if cnn in stem:
                        self.model_name_str = cnn
                        break
            if not self.num_classes and self.class_names:
                self.num_classes = len(self.class_names)
            self._transform = _cnn_get_transforms(self.imgsz, train=False)
            self._loaded = True
        else:
            # ── PyTorch (.pt) branch ─────────────────────────────────────────
            import torch
            # Φόρτωση checkpoint – πρώτα με weights_only=False (για παλιά PyTorch),
            # fallback σε strict mode αν αποτύχει.
            try:
                ckpt = torch.load(str(self.model_path), map_location='cpu', weights_only=False)
            except Exception:
                ckpt = torch.load(str(self.model_path), map_location='cpu')

            self.model_name_str = str(ckpt.get('model_name', ''))
            self.num_classes    = int(ckpt.get('num_classes', 1000))
            self.class_names    = list(ckpt.get('class_names', []))
            self.imgsz          = int(ckpt.get('imgsz', 224))

            # Try to load class names from sibling JSON if not in checkpoint
            if not self.class_names:
                try:
                    cj = self.model_path.parent / 'class_names.json'
                    if cj.is_file():
                        d = json.loads(cj.read_text(encoding='utf-8'))
                        self.class_names = list(d.get('class_names', []))
                        if not self.model_name_str:
                            self.model_name_str = str(d.get('model_name', ''))
                        if self.imgsz == 224:
                            self.imgsz = int(d.get('imgsz', 224))
                except Exception:
                    pass

            if not self.model_name_str:
                # Infer from filename
                stem = self.model_path.stem.lower()
                for cnn in TRAIN_CNN_MODELS:
                    if cnn in stem:
                        self.model_name_str = cnn
                        break

            model_nn = _load_torchvision_model(self.model_name_str, num_classes=self.num_classes, pretrained=False)
            model_nn.load_state_dict(ckpt['state_dict'])
            model_nn.eval()

            dev_str = self.device_str.lower()
            if dev_str.startswith('cuda') and not torch.cuda.is_available():
                dev_str = 'cpu'
            self._device_t = torch.device(dev_str)
            self.model_nn  = model_nn.to(self._device_t)
            self._transform = _cnn_get_transforms(self.imgsz, train=False)
            self._loaded = True

    # ── Inference σε BGR frame (OpenCV format) ────────────────────────────────────
    # Μετατρέπει BGR → RGB → PIL → tensor → softmax → top-k.
    # Επιστρέφει: [(class_name, confidence, class_id), ...] ταξινομημένο κατά conf (desc).
    def predict_frame(self, frame, top_k: int = 5) -> list[tuple[str, float, int]]:
        """
        Εκτελεί classification σε BGR frame (numpy array από OpenCV).
        Επιστρέφει λίστα από (class_name, confidence, class_id) ταξινομημένη κατά conf.
        Υποστηρίζει PyTorch (.pt) και ONNX Runtime (.onnx).
        Το class_id είναι ο αριθμητικός δείκτης της κλάσης (χρησιμοποιείται από classes_filter).
        """
        if not self._loaded:
            return []
        try:
            import numpy as np
            from PIL import Image as _PIL_Image
            # BGR → RGB → PIL
            rgb = frame[:, :, ::-1].copy()
            pil_img = _PIL_Image.fromarray(rgb)

            if self._is_onnx:
                # ── ONNX Runtime inference ───────────────────────────────────
                import numpy as _np
                tensor = self._transform(pil_img).unsqueeze(0)  # [1, C, H, W]
                input_name = self._ort_session.get_inputs()[0].name
                logits_np = self._ort_session.run(None, {input_name: tensor.numpy()})[0]  # [1, num_classes]
                # softmax
                e = _np.exp(logits_np[0] - _np.max(logits_np[0]))
                probs = e / e.sum()
                top_k = min(top_k, len(probs))
                idxs = _np.argsort(probs)[::-1][:top_k]
                results = []
                for i in idxs:
                    name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
                    results.append((name, float(probs[i]), int(i)))
                return results
            else:
                # ── PyTorch inference ────────────────────────────────────────
                import torch
                if self.model_nn is None:
                    return []
                tensor = self._transform(pil_img).unsqueeze(0).to(self._device_t)
                with torch.no_grad():
                    logits = self.model_nn(tensor)
                    probs  = torch.softmax(logits, dim=1)[0]
                top_k = min(top_k, len(probs))
                vals, idxs = probs.topk(top_k)
                results = []
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    name = self.class_names[i] if i < len(self.class_names) else f'class_{i}'
                    results.append((name, float(v), int(i)))
                return results
        except Exception:
            return []

    def annotate_frame(self, frame, top_k: int = 5) -> 'np.ndarray':
        """Public API: predict + draw. Delegates to _draw_predictions."""
        try:
            preds = self.predict_frame(frame, top_k=top_k)
            return self._draw_predictions(frame, preds)
        except Exception:
            try:
                return frame.copy()
            except Exception:
                return frame

    # ── Σχεδιασμός classification overlay στο κάτω-αριστερό τμήμα του frame ─────
    # Δομή panel (bottom-left):
    #   ┌─ CNN  mobilenet_v2  |  5 classes ─────────────────────────┐
    #   │ ①  grape       ████████████░░  87.3%                      │
    #   │ ②  wine        ████░░░░░░░░░░  31.2%                      │
    #   └───────────────────────────────────────────────────────────┘
    # Στοιχεία:
    #   - Semi-transparent dark card (addWeighted 75%/25%)
    #   - Πράσινη γραμμή accent στην κορυφή
    #   - Header με model name + αριθμό κλάσεων
    #   - Rank badge (κύκλος με αριθμό)
    #   - Class name (DUPLEX για top-1, SIMPLEX για τα υπόλοιπα)
    #   - Οριζόντια μπάρα confidence (πράσινη #1, μπλε #2, γκρι #3+)
    #   - Ποσοστό (π.χ. "87.3%")
    # Μεγέθη: adaptive βάσει frame resolution (480p → 1080p scaling).
    def _draw_predictions(self, frame, preds: list) -> 'np.ndarray':
        """
        Σχεδιάζει classification overlay στο κάτω-αριστερό μέρος του frame.
        Μπορεί να κληθεί και με pre-computed preds (αποφυγή διπλής inference).
        Δέχεται tuples (name, conf) ή (name, conf, class_id) — αγνοεί το class_id.

        Σχεδιαστική λογική (κάτω-αριστερά):
          ┌─────────────────────────────────────────────┐ ← panel border
          │ CNN  mobilenet_v2  |  5 classes             │ ← header
          ├─────────────────────────────────────────────┤
          │ ①  grape          ████████████░░░  87.3%    │ ← top-1 (green)
          │ ②  wine           ████░░░░░░░░░░░  31.2%    │ ← rest  (blue)
          │ ③  berry          ██░░░░░░░░░░░░░  18.5%    │
          └─────────────────────────────────────────────┘
        """
        # Κανονικοποίηση tuples: (name, conf[, class_id]) → (name, conf)
        preds = [(p[0], p[1]) for p in preds if len(p) >= 2]
        try:
            if not preds:
                try:
                    return frame.copy()
                except Exception:
                    return frame

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # ── Υπολογισμός μεγεθών που κλιμακώνονται με την ανάλυση frame ─────────
            # Στα 1080p: row_h≈45px, στα 480p: row_h≈24px
            row_h       = max(24, int(h * 0.042))   # row height  (μειωμένο)
            pad         = max(8,  int(h * 0.013))   # inner padding
            margin      = max(12, int(h * 0.016))   # frame edge margin
            fs_main     = max(0.45, row_h / 46.0)   # class name font scale (μειωμένο)
            fs_pct      = max(0.42, row_h / 50.0)   # percentage font scale  (μειωμένο)
            fs_hdr      = max(0.34, row_h / 64.0)   # header font scale      (μειωμένο)
            ft_main     = max(1, int(fs_main * 2.0))
            ft_pct      = max(1, int(fs_pct  * 1.8))
            ft_hdr      = max(1, int(fs_hdr  * 1.6))
            bar_max_w   = max(110, int(w * 0.22))   # confidence bar max width
            bar_h       = max(5,   int(row_h * 0.20))
            badge_r     = max(8,   int(row_h * 0.30))
            label_col_w = max(140, int(w * 0.18))   # text column width
            pct_col_w   = max(60,  int(w * 0.060))  # percentage column width
            row_spacing = row_h + max(4, int(row_h * 0.18))
            hdr_h       = max(16, int(row_h * 0.58))

            n       = len(preds)
            panel_w = (pad
                       + badge_r * 2 + pad
                       + label_col_w + pad
                       + bar_max_w + pad
                       + pct_col_w + pad)
            panel_h = pad + hdr_h + pad + n * row_spacing + pad

            # ── Τοποθέτηση panel στο κάτω-αριστερό τμήμα του frame ───────────────────
            # margin: απόσταση από τα άκρα του frame
            x_panel = margin
            y_panel = max(0, h - margin - panel_h)
            rx1 = x_panel
            ry1 = y_panel
            rx2 = min(w - 1, x_panel + panel_w)
            ry2 = min(h - 1, y_panel + panel_h)

            # ── Σχεδιασμός σκούρου semi-transparent panel (75% αδιαφανές) ──────────
            card = annotated.copy()
            cv2.rectangle(card, (rx1, ry1), (rx2, ry2), (10, 10, 10), -1)
            cv2.addWeighted(card, 0.75, annotated, 0.25, 0, annotated)
            # Outer border
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (60, 60, 60), 1)
            # Top accent line (cyan-green)
            cv2.line(annotated, (rx1 + 1, ry1 + 1),
                     (rx2 - 1, ry1 + 1), (40, 200, 140), 2)

            # ── Header row ────────────────────────────────────────────────
            hdr_y = ry1 + pad + hdr_h - 2
            hdr_txt = f'CNN  {self.model_name_str}  |  {self.num_classes} classes'
            cv2.putText(annotated, hdr_txt,
                        (rx1 + pad, max(0, hdr_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs_hdr, (120, 200, 255), ft_hdr, cv2.LINE_AA)
            # Separator line under header
            sep_y = ry1 + pad + hdr_h + 2
            cv2.line(annotated, (rx1 + 4, sep_y), (rx2 - 4, sep_y), (50, 50, 50), 1)

            # ── Prediction rows ────────────────────────────────────────────
            y_rows_start = sep_y + pad + int(row_h * 0.80)

            for k, (cls_name, conf) in enumerate(preds):
                yr     = y_rows_start + k * row_spacing
                is_top = (k == 0)

                # Color palette
                if is_top:
                    txt_col = ( 55, 255,  95)   # vivid green  (BGR)
                    bar_col = ( 35, 195,  65)
                    pct_col = ( 75, 255, 115)
                    bdg_col = ( 35, 185,  65)
                elif k == 1:
                    txt_col = (100, 180, 255)   # light blue
                    bar_col = ( 80, 140, 220)
                    pct_col = (130, 190, 255)
                    bdg_col = ( 70, 120, 200)
                else:
                    txt_col = (195, 195, 195)   # light grey
                    bar_col = ( 70, 110, 185)
                    pct_col = (165, 165, 165)
                    bdg_col = ( 55,  85, 160)

                # ── Πράσινο glow background για την top-1 κλάση ────────────────────
                if is_top:
                    hl = annotated.copy()
                    hl_y1 = max(ry1 + 1, yr - int(row_h * 0.82))
                    hl_y2 = min(ry2 - 1, yr + int(row_h * 0.28))
                    cv2.rectangle(hl, (rx1 + 1, hl_y1), (rx2 - 1, hl_y2),
                                  (16, 48, 18), -1)
                    cv2.addWeighted(hl, 0.44, annotated, 0.56, 0, annotated)

                # ── Rank badge (circle with number) ──────────────────────
                bdg_cx = rx1 + pad + badge_r
                bdg_cy = yr - int(row_h * 0.20)
                cv2.circle(annotated, (bdg_cx, bdg_cy), badge_r, bdg_col, -1)
                # Badge border for top-1
                if is_top:
                    cv2.circle(annotated, (bdg_cx, bdg_cy), badge_r,
                               (80, 255, 120), 1)
                bdg_fs  = max(0.36, badge_r / 17.0)
                bdg_ft  = max(1, int(bdg_fs * 2.0))
                btxt    = str(k + 1)
                (btw, bth), _ = cv2.getTextSize(
                    btxt, cv2.FONT_HERSHEY_SIMPLEX, bdg_fs, bdg_ft)
                cv2.putText(annotated, btxt,
                            (bdg_cx - btw // 2, bdg_cy + bth // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            bdg_fs, (255, 255, 255), bdg_ft, cv2.LINE_AA)

                # ── Class name ────────────────────────────────────────────
                txt_x  = rx1 + pad + badge_r * 2 + max(7, int(row_h * 0.22))
                # Truncate if too long
                char_px = max(1, int(fs_main * 13))
                max_ch  = max(10, label_col_w // char_px)
                disp    = (cls_name if len(cls_name) <= max_ch
                           else cls_name[:max_ch - 1] + '\u2026')  # …
                lbl_fnt = (cv2.FONT_HERSHEY_DUPLEX if is_top
                           else cv2.FONT_HERSHEY_SIMPLEX)
                cv2.putText(annotated, disp,
                            (txt_x, yr),
                            lbl_fnt,
                            fs_main * (1.06 if is_top else 1.0),
                            txt_col,
                            ft_main + (1 if is_top else 0),
                            cv2.LINE_AA)

                # ── Οριζόντια μπάρα confidence (track + filled + border) ─────────────
                # fw: πλάτος γεμάτου τμήματος αναλογικά με το confidence (0.0–1.0)
                bar_x  = rx1 + pad + badge_r * 2 + pad + label_col_w
                bar_yt = yr - bar_h - max(2, int(row_h * 0.13))
                fw     = max(2, int(bar_max_w * conf))

                # Track background
                cv2.rectangle(annotated,
                              (bar_x, bar_yt),
                              (bar_x + bar_max_w, bar_yt + bar_h),
                              (38, 38, 38), -1)
                # Filled segment
                cv2.rectangle(annotated,
                              (bar_x, bar_yt),
                              (bar_x + fw, bar_yt + bar_h),
                              bar_col, -1)
                # Top-1: bright end-cap
                if is_top and fw >= 4:
                    cv2.rectangle(annotated,
                                  (bar_x + fw - 2, bar_yt),
                                  (bar_x + fw, bar_yt + bar_h),
                                  (120, 255, 160), -1)
                # Track border
                cv2.rectangle(annotated,
                              (bar_x, bar_yt),
                              (bar_x + bar_max_w, bar_yt + bar_h),
                              (72, 72, 72), 1)

                # ── Percentage text ────────────────────────────────────────
                pct_str = f'{conf * 100:.1f}%'
                pct_x   = bar_x + bar_max_w + max(7, int(row_h * 0.16))
                cv2.putText(annotated, pct_str,
                            (pct_x, yr),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fs_pct * (1.06 if is_top else 1.0),
                            pct_col,
                            ft_pct + (1 if is_top else 0),
                            cv2.LINE_AA)

            return annotated
        except Exception:
            try:
                return frame.copy()
            except Exception:
                return frame

"""Μονωμένος runner benchmark (subprocess-safe).
Σκοπός:
  - Να τρέχει το benchmark κάθε backend σε *ξεχωριστή διεργασία*.
  - Αν κάποιο backend (π.χ. TensorRT/ONNX/NCNN) κάνει native crash (access violation),
    να *μην* πέφτει όλο το GUI.
Χρήση (μέσω Models_Manager_Pro.py):
  python Models_Manager_Pro.py --mmpro-mode=bench <job.json>; Models_Manager_Pro.exe --mmpro-mode=bench <job.json>
Το job.json πρέπει να περιέχει:
  { "backend": "pytorch"|"onnx"|"tensorrt"|"ncnn", "path": "C:/.../model.pt|model.onnx|model.engine|dir_ncnn_model", "imgsz": 640, "num_warmup": 10, "num_runs": 50, "conf": 0.25, "iou": 0.45}
Ο runner εκτυπώνει *ένα* JSON line στο stdout με αποτέλεσμα.
"""
try:
    configure_opencv_videoio_env()
    try:
        _lvl = int(os.environ.get('MM_PRO_ORT_LOG_LEVEL', '3'))
    except Exception:
        _lvl = 3
    configure_onnxruntime_logging(severity=_lvl, verbosity=0)
except Exception:
    pass


def _truthy_env(name: str) -> bool:
    v = str(os.environ.get(name, '')).strip().lower()
    return v in ('1', 'true', 'yes', 'y', 'on')


def _scan_dlls_in_dirs(dirs: list[Path], patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    try:
        for d in dirs:
            if not d or not d.exists() or not d.is_dir():
                continue
            for pat in patterns:
                try:
                    for p in d.glob(pat):
                        if p.is_file():
                            found.append(p.name)
                except Exception:
                    continue
    except Exception:
        return []
    seen = set()
    out = []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _trt_signature_path_for_engine(engine_path: Path) -> Path:
    return Path(str(engine_path) + '.mmpro.json')


def _bench_json_read(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return None


def _collect_trt_candidate_dirs(trt_module_file: str | None) -> list[Path]:
    dirs: list[Path] = []
    try:
        dirs.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    try:
        meip = getattr(sys, '_MEIPASS', None)
        if meip:
            dirs.append(Path(meip))
    except Exception:
        pass
    try:
        if trt_module_file:
            td = Path(trt_module_file).resolve().parent
            dirs.append(td)
            dirs.append(td / "lib")
            dirs.append(td.parent / "tensorrt_libs")
            dirs.append(td.parent / "tensorrt_libs" / "lib")
    except Exception:
        pass
    try:
        import shutil
        t = shutil.which("trtexec")
        if t:
            tdir = Path(t).resolve().parent
            dirs.append(tdir)
            dirs.append(tdir.parent / "lib")
            dirs.append(tdir.parent / "bin")
    except Exception:
        pass
    for env in ("TENSORRT_PATH", "CUDA_PATH"):
        try:
            v = os.environ.get(env, "")
            if v:
                vp = Path(v).resolve()
                dirs.append(vp)
                dirs.append(vp / "lib")
                dirs.append(vp / "bin")
        except Exception:
            continue
    out: list[Path] = []
    seen = set()
    for d in dirs:
        try:
            dd = d.resolve()
        except Exception:
            dd = d
        if str(dd) in seen:
            continue
        seen.add(str(dd))
        if dd.exists() and dd.is_dir():
            out.append(dd)
    return out


def _bench_ensure_tensorrt_plugins(candidate_dirs: list[Path]) -> dict:
    info: dict = {"plugin_found": "", "plugin_loaded": False, "plugin_load_error": ""}
    if os.name != "nt":
        return info
    handles = []
    try:
        try:
            add_dir = getattr(os, "add_dll_directory", None)
            if add_dir:
                for d in candidate_dirs:
                    try:
                        handles.append(add_dir(str(d)))
                    except Exception:
                        continue
        except Exception:
            pass
        plugin_path: Path | None = None
        for d in candidate_dirs:
            try:
                hits = sorted([p for p in d.glob("nvinfer_plugin*.dll") if p.is_file()], key=lambda x: x.name, reverse=True)
                if hits:
                    plugin_path = hits[0]
                    break
            except Exception:
                continue
        if plugin_path is None:
            return info
        info["plugin_found"] = str(plugin_path)
        try:
            import ctypes
            ctypes.CDLL(str(plugin_path))
            info["plugin_loaded"] = True
        except Exception as e:
            info["plugin_loaded"] = False
            info["plugin_load_error"] = str(e)
        return info
    finally:
        info["_dll_dir_handles"] = handles


def _preflight_tensorrt_engine(engine_path: Path) -> None:
    p = Path(engine_path)
    if not p.exists():
        raise RuntimeError(f"TensorRT backend: δεν βρέθηκε engine: {p}")
    try:
        size = int(p.stat().st_size)
    except Exception:
        size = 0
    if size <= 0:
        raise RuntimeError(f"TensorRT backend: το engine είναι άδειο/0 bytes: {p}")
    sig_path = _trt_signature_path_for_engine(p)
    sig = _bench_json_read(sig_path) or {}
    built_v = (sig.get("versions") or {}) if isinstance(sig, dict) else {}
    built_trt = str(built_v.get("tensorrt") or "")
    built_gpu = str(built_v.get("gpu") or "")
    built_cc = str(built_v.get("gpu_cc") or "")
    is_frozen = bool(getattr(sys, "frozen", False)) or getattr(sys, "_MEIPASS", None) is not None
    try:
        import tensorrt as trt
    except Exception as e:
        extra = ""
        if is_frozen:
            dirs: list[Path] = []
            try:
                dirs.append(Path(sys.executable).resolve().parent)
            except Exception:
                pass
            try:
                meip = getattr(sys, "_MEIPASS", None)
                if meip:
                    dirs.append(Path(meip))
            except Exception:
                pass
            dlls = _scan_dlls_in_dirs(dirs, ("nvinfer*.dll", "nvinfer_plugin*.dll", "nvonnxparser*.dll"))
            if dlls:
                extra = "\nΒρέθηκαν στο dist/MEIPASS: " + ", ".join(dlls[:30])
            else:
                extra = "\nΔεν βρέθηκαν nvinfer*.dll / nvinfer_plugin*.dll στο dist/MEIPASS."
        raise RuntimeError( "TensorRT backend: αποτυχία import του 'tensorrt'.\n" "Αυτό συνήθως σημαίνει ότι λείπουν τα TensorRT DLLs (nvinfer*.dll) ή δεν είναι στο DLL search path." + extra) from e
    rt_trt = str(getattr(trt, "__version__", "?"))
    rt_gpu = ""
    rt_cc = ""
    try:
        import torch
        if torch.cuda.is_available():
            rt_gpu = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            rt_cc = f"{cc[0]}.{cc[1]}"
    except Exception:
        pass
    trt_file = getattr(trt, "__file__", None)
    cand_dirs = _collect_trt_candidate_dirs(trt_file)
    plugin_info = _bench_ensure_tensorrt_plugins(cand_dirs)
    logger_level = trt.Logger.ERROR
    if os.environ.get("MM_TRT_VERBOSE", "0").strip() == "1":
        logger_level = trt.Logger.VERBOSE
    logger = trt.Logger(logger_level)
    plugin_init_ok = ""
    plugin_init_err = ""
    try:
        if hasattr(trt, "init_libnvinfer_plugins"):
            ok = trt.init_libnvinfer_plugins(logger, "")
            plugin_init_ok = str(ok)
    except Exception as e:
        plugin_init_err = str(e)
    data_full = p.read_bytes()

    def _strip_ultralytics_engine_prefix(blob: bytes) -> tuple[bytes, bool]:
        try:
            if len(blob) < 16:
                return blob, False
            meta_len = int.from_bytes(blob[:4], byteorder='little', signed=True)
            if meta_len <= 0 or meta_len > 200_000:
                return blob, False
            end = 4 + meta_len
            if end >= len(blob):
                return blob, False
            meta_bytes = blob[4:end]
            if not meta_bytes.lstrip().startswith(b'{'):
                return blob, False
            meta_txt = meta_bytes.decode('utf-8', errors='ignore')
            if ('"description"' not in meta_txt) and ('"task"' not in meta_txt) and ('"imgsz"' not in meta_txt):
                return blob, False
            return blob[end:], True
        except Exception:
            return blob, False
    data_engine, stripped = _strip_ultralytics_engine_prefix(data_full)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(data_engine)
    if engine is None and stripped:
        engine = runtime.deserialize_cuda_engine(data_full)
    if engine is not None:
        return
    lines: list[str] = []
    lines.append("TensorRT backend: αποτυχία απο-σειριοποίησης του engine (deserialize_cuda_engine επέστρεψε None).")
    lines.append(f"• engine: {p} ({size} bytes)")
    lines.append(f"• TensorRT runtime: {rt_trt}")
    if stripped:
        lines.append("• Ultralytics metadata prefix: detected (attempted strip πριν το deserialize)")
    if rt_gpu or rt_cc:
        lines.append(f"• GPU runtime: {rt_gpu} | CC: {rt_cc}")
    if built_trt or built_gpu or built_cc:
        lines.append(f"• engine built-with: TensorRT={built_trt} | GPU={built_gpu} | CC={built_cc}")
    else:
        lines.append("• engine built-with: (δεν βρέθηκε signature .mmpro.json ή είναι κενό)")
    pf = str(plugin_info.get("plugin_found") or "")
    pl = bool(plugin_info.get("plugin_loaded") or False)
    if pf:
        lines.append(f"• nvinfer_plugin: {pf} | loaded={pl}")
    if plugin_info.get("plugin_load_error"):
        lines.append(f"• plugin load error: {plugin_info.get('plugin_load_error')}")
    if plugin_init_ok:
        lines.append(f"• init_libnvinfer_plugins: {plugin_init_ok}")
    if plugin_init_err:
        lines.append(f"• init_libnvinfer_plugins error: {plugin_init_err}")
    mismatch_lines: list[str] = []
    if built_trt and built_trt != rt_trt:
        mismatch_lines.append(f"  • mismatch TensorRT: engine={built_trt} vs runtime={rt_trt}")
    if built_cc and rt_cc and built_cc != rt_cc:
        mismatch_lines.append(f"  • mismatch GPU CC: engine={built_cc} vs runtime={rt_cc}")
    if built_gpu and rt_gpu and built_gpu != rt_gpu:
        mismatch_lines.append(f"  • mismatch GPU: engine='{built_gpu}' vs runtime='{rt_gpu}'")
    if mismatch_lines:
        lines.append("Πιθανή αιτία: ασυμβατό engine (mismatch).")
        lines.extend(mismatch_lines)
        lines.append("Λύση: διέγραψε το .engine ΚΑΙ το .engine.mmpro.json και κάνε rebuild/export στο ίδιο περιβάλλον.")
    else:
        lines.append("Πιθανές αιτίες:")
        lines.append("  • plugins που δεν φορτώνονται (nvinfer_plugin*.dll)")
        lines.append("  • λείπουν DLL dependencies (CUDA/cuDNN/cuBLAS) που απαιτούνται από το plugin/runtime")
        lines.append("  • engine corrupted/μισογραμμένο")
        lines.append("Λύση: βεβαιώσου ότι τα TensorRT DLLs είναι προσβάσιμα (PATH ή σωστό dist/bin) και ξανακάνε export.")
        lines.append("Tip: βάλε MM_TRT_VERBOSE=1 για πιο αναλυτικά TRT logs στο subprocess.")
    raise RuntimeError("\n".join(lines))


def _preflight_ncnn_import() -> None:
    try:
        import ncnn
    except Exception as e:
        raise RuntimeError( "NCNN backend: αποτυχία εισαγωγής του 'ncnn'.\n" "Συνήθως λείπει κάποιο DLL (π.χ. ncnn.dll ή Visual C++/OpenMP runtime) ή δεν είναι στο DLL search path.") from e


def _safe_print_json(payload: dict[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _bench_once(model: Any, frame, imgsz: int, conf: float, iou: float) -> float:
    cuda_sync()
    t0 = time.perf_counter()
    try:
        is_cls = getattr(model, 'task', None) == 'classify'
    except Exception:
        is_cls = False

    def _parse_expected(msg: str):
        try:
            s = str(msg or '')
            pairs = re.findall(r"index:\s*(\d+)\s*Got:\s*(\d+)\s*Expected:\s*(\d+)", s)
            exp = {}
            for idx, _got, ex in pairs:
                try:
                    exp[int(idx)] = int(ex)
                except Exception:
                    pass
            if 2 in exp and 3 in exp and exp[2] > 0 and exp[3] > 0:
                h = int(exp[2])
                w = int(exp[3])
                return int(h) if h == w else (h, w)
            exps = re.findall(r"Expected:\s*(\d+)", s)
            if exps:
                v = int(exps[0])
                return v if v > 0 else None
        except Exception:
            pass
    try:
        if is_cls:
            _ = model.predict(frame, imgsz=imgsz, verbose=False)
        else:
            _ = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        exp = _parse_expected(str(e))
        if exp is not None:
            exp_use = exp
            try:
                if is_cls and isinstance(exp, (tuple, list)) and len(exp) == 2 and int(exp[0]) == int(exp[1]):
                    exp_use = int(exp[0])
            except Exception:
                exp_use = exp
            if is_cls:
                _ = model.predict(frame, imgsz=exp_use, verbose=False)
            else:
                _ = model.predict(frame, imgsz=exp_use, conf=conf, iou=iou, verbose=False)
        else:
            raise
    cuda_sync()
    return time.perf_counter() - t0


def benchmark_runner_main() -> int:
    if len(sys.argv) < 2:
        _safe_print_json({"ok": False, "error": "Missing job.json path"})
        return 2
    job_path = Path(sys.argv[1]).resolve()
    if not job_path.exists():
        _safe_print_json({"ok": False, "error": f"job.json not found: {job_path}"})
        return 2
    try:
        job = json.loads(job_path.read_text(encoding='utf-8'))
    except Exception as e:
        _safe_print_json({"ok": False, "error": f"Failed to read job.json: {e}"})
        return 2
    backend = str(job.get('backend', '')).strip().lower()
    model_path = Path(str(job.get('path', '')).strip())
    imgsz = int(job.get('imgsz', 640) or 640)
    num_warmup = int(job.get('num_warmup', 10) or 10)
    num_runs = int(job.get('num_runs', 50) or 50)
    conf = float(job.get('conf', 0.25) or 0.25)
    iou = float(job.get('iou', 0.45) or 0.45)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    debug_tb = _truthy_env('MM_DEBUG_TRACEBACK')
    try:
        import numpy as np
    except Exception as e:
        _safe_print_json({"ok": False, "backend": backend, "error": f"numpy missing: {e}"})
        return 3
    if not str(model_path):
        _safe_print_json({"ok": False, "backend": backend, "error": "Missing model path"})
        return 2

    # ── Ανίχνευση CNN torchvision μοντέλου (π.χ. mobilenet_v2_GPU_grape_224.onnx) ──
    # Αν πρόκειται για CNN μοντέλο, χρησιμοποιούμε CNNInferenceHelper
    # αντί για YOLO (που θα αποτύχει με "amax(): Expected reduction dim 1").
    is_cnn = _is_cnn_path(model_path)
    if is_cnn:
        try:
            helper = CNNInferenceHelper(model_path, device='cpu')
            helper.load()
        except Exception as e:
            payload = {"ok": False, "backend": backend, "error": f"Failed to load CNN model: {e}"}
            if debug_tb:
                payload["traceback"] = traceback.format_exc()
            _safe_print_json(payload)
            return 4
        try:
            # CNN μοντέλα έχουν imgsz που ορίζεται στο checkpoint/JSON (συνήθως 224)
            cnn_imgsz = helper.imgsz
            frame = np.random.randint(0, 255, (cnn_imgsz, cnn_imgsz, 3), dtype=np.uint8)
            # Warmup
            for _ in range(max(0, num_warmup)):
                helper.predict_frame(frame)
            # Benchmark
            times = []
            for _ in range(max(1, num_runs)):
                t0 = time.perf_counter()
                helper.predict_frame(frame)
                times.append(time.perf_counter() - t0)
            avg = (sum(times) / len(times)) if times else 0.001
            fps = (1.0 / avg) if avg > 0 else 0.0
            ms = avg * 1000.0
            _safe_print_json({"ok": True, "backend": backend, "fps": float(fps), "ms": float(ms)})
            return 0
        except Exception as e:
            payload = {"ok": False, "backend": backend, "error": str(e)}
            if debug_tb:
                payload["traceback"] = traceback.format_exc()
            _safe_print_json(payload)
            return 5
        finally:
            try:
                del helper
            except Exception:
                pass
        return 0

    # ── YOLO backends (PyTorch / ONNX YOLO / TensorRT / NCNN) ──────────────────
    try:
        from ultralytics import YOLO
    except Exception as e:
        _safe_print_json({"ok": False, "backend": backend, "error": f"ultralytics missing: {e}"})
        return 3
    try:
        if backend == 'tensorrt':
            _preflight_tensorrt_engine(model_path)
        elif backend == 'ncnn':
            _preflight_ncnn_import()
    except Exception as e:
        payload = {"ok": False, "backend": backend, "error": str(e)}
        if debug_tb:
            payload["traceback"] = traceback.format_exc()
        _safe_print_json(payload)
        return 5
    try:
        try:
            model = YOLO(str(model_path), task='detect')
        except TypeError:
            model = YOLO(str(model_path))
    except Exception as e:
        payload = {"ok": False, "backend": backend, "error": f"Failed to load model: {e}"}
        if debug_tb:
            payload["traceback"] = traceback.format_exc()
        _safe_print_json(payload)
        return 4
    try:
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(max(0, num_warmup)):
            _ = _bench_once(model, frame, imgsz, conf, iou)
        times = []
        for _ in range(max(1, num_runs)):
            times.append(_bench_once(model, frame, imgsz, conf, iou))
        avg = (sum(times) / len(times)) if times else 0.001
        fps = (1.0 / avg) if avg > 0 else 0.0
        ms = avg * 1000.0
        _safe_print_json({ "ok": True, "backend": backend, "fps": float(fps), "ms": float(ms),})
        return 0
    except Exception as e:
        payload = {"ok": False, "backend": backend, "error": str(e)}
        if debug_tb:
            payload["traceback"] = traceback.format_exc()
        _safe_print_json(payload)
        return 5
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
import multiprocessing as _mp


def export_runner_main() -> int:
    import json as _json
    import traceback as _traceback
    from pathlib import Path
    if len(sys.argv) < 2:
        print_line('__MM_ERR__', 'Missing job json path.')
        return 2
    job_path = Path(sys.argv[1])
    root_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    try:
        os.chdir(str(globals().get("ROOT_DIR", root_dir)))
    except Exception:
        try:
            os.chdir(str(root_dir))
        except Exception:
            pass
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    crash_dir = Path(globals().get("CRASH_LOGS_DIR", root_dir / 'Crash_Logs'))
    try:
        enable_faulthandler(crash_dir, 'export')
    except Exception:
        pass
    try:
        job = _json.loads(job_path.read_text(encoding='utf-8'))
    except Exception as e:
        print_line('__MM_ERR__', f'Failed to load job json: {e}')
        return 3
    try:
        set_log_colors_for_theme(str(job.get('ui_theme', 'light')))
    except Exception:
        pass
    try:
        worker = ExportWorker( job.get('model_path', ''), int(job.get('imgsz', 640)), job.get('export_format', 'onnx'), overwrite=bool(job.get('overwrite', False)),)
    except Exception as e:
        print_line('__MM_ERR__', f'Failed to init ExportWorker: {e}')
        print_line('__MM_EXCEPTION__', _traceback.format_exc())
        return 5
    try:
        worker.log.connect(lambda html: print_line('__MM_LOG__', html))
        worker.error.connect(lambda msg: print_line('__MM_ERR__', msg))
    except Exception:
        pass
    try:
        worker.run()
        print_line('__MM_DONE__', '')
        return 0
    except Exception as e:
        print_line('__MM_ERR__', f'Exception during export: {e}')
        print_line('__MM_EXCEPTION__', _traceback.format_exc())
        return 1


def training_runner_main() -> int:
    import json as _json
    import traceback as _traceback
    from pathlib import Path
    if len(sys.argv) < 2:
        print_line('__MM_ERR__', 'Missing job json path.')
        return 2
    job_path = Path(sys.argv[1])
    root_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    try:
        os.chdir(str(globals().get("ROOT_DIR", root_dir)))
    except Exception:
        try:
            os.chdir(str(root_dir))
        except Exception:
            pass
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    crash_dir = Path(globals().get("CRASH_LOGS_DIR", root_dir / 'Crash_Logs'))
    try:
        enable_faulthandler(crash_dir, 'train')
    except Exception:
        pass
    try:
        job = _json.loads(job_path.read_text(encoding='utf-8'))
    except Exception as e:
        print_line('__MM_ERR__', f'Failed to load job json: {e}')
        return 3
    try:
        set_log_colors_for_theme(str(job.get('ui_theme', 'light')))
    except Exception:
        pass
    try:
        model_name = job.get('model', job.get('model_name', ''))
        dataset_name = job.get('dataset', job.get('dataset_name', ''))
        imgsz = int(job.get('imgsz', 640))
        device = str(job.get('device', 'cpu'))
        epochs = int(job.get('epochs', 100))
        patience = int(job.get('patience', 50))
        use_triton = bool(job.get('use_triton', False))
        compile_mode = str(job.get('compile_mode', 'none'))
        extra_hparams = job.get('extra_hparams') or {}
        try:
            worker = TrainingWorker(model_name, dataset_name, imgsz, device, epochs, patience, use_triton, compile_mode, extra_hparams=extra_hparams)
        except TypeError:
            worker = TrainingWorker(model_name, dataset_name, imgsz, device, epochs, patience, use_triton, compile_mode)
    except Exception as e:
        print_line('__MM_ERR__', f'Failed to init TrainingWorker: {e}')
        print_line('__MM_EXCEPTION__', _traceback.format_exc())
        return 5
    try:
        worker.log.connect(lambda html: print_line('__MM_LOG__', html))
        worker.error.connect(lambda msg: print_line('__MM_ERR__', msg))
        worker.report_ready.connect(lambda path: print_line('__MM_REPORT__', path))
    except Exception:
        pass
    try:
        worker.run()
        print_line('__MM_DONE__', '')
        return 0
    except Exception as e:
        print_line('__MM_ERR__', f'Exception during training: {e}')
        print_line('__MM_EXCEPTION__', _traceback.format_exc())
        return 1
"""Benchmark workers.
Τρέχει benchmarks για διαθέσιμα backends (PyTorch/ONNX/TensorRT/NCNN) σε ασφαλές; subprocess (για προστασία από native crashes) και συλλέγει FPS/ms.
Σημείωση:
Για inference χρησιμοποιούμε τον κοινό wrapper `yolo_predict_first()` ώστε να υπάρχει; ενιαίο auto-recover σε fixed input sizes (π.χ. ONNX expects 224 αλλά δίνεται 640).
"""
import tempfile


class BenchmarkWorker(QObject, LogEmitMixin):
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    results_ready = Signal(list)

    def __init__(self, base_name: str, imgsz: int, models_dir: Path, parent: QObject | None=None):
        super().__init__(parent)
        self.base_name = base_name
        self.imgsz = imgsz
        self.models_dir = models_dir
        self.num_warmup = 10
        self.num_runs = 50
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        self._cprint('🛑 Λήφθηκε εντολή διακοπής Benchmark.', Colors.RED, bold=True)

    def _run_single_inference(self, model: Any, frame: np.ndarray) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = yolo_predict_first( model, frame, imgsz=int(self.imgsz), conf=float(self.conf_threshold), iou=float(self.iou_threshold), verbose=False,)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0

    def _benchmark_backend(self, backend: str, path: Path) -> tuple[float, float | None]:
        if self._stop_requested:
            return None
        self._cprint(f'🧪 Εκτέλεση {backend_pretty_name(backend)} σε ασφαλές subprocess...', Colors.CYAN)
        job = {
            'backend': (backend or '').lower(),
            'path': str(path),
            'imgsz': int(self.imgsz),
            'num_warmup': int(self.num_warmup),
            'num_runs': int(self.num_runs),
            'conf': float(self.conf_threshold),
            'iou': float(self.iou_threshold),
        }
        job_file = None
        try:
            tmp_dir = Path(tempfile.gettempdir())
            job_file = tmp_dir / f'mmpro_bench_{os.getpid()}_{int(time.time()*1000)}.json'
            job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            self._cprint(f'❌ Αδυναμία δημιουργίας προσωρινού job για benchmark: {e}', Colors.RED, bold=True)
            return (0.0, 0.0)
        try:
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--mmpro-mode=bench', str(job_file)]
            else:
                entry = Path(__file__).resolve()
                cmd = [sys.executable, str(entry), '--mmpro-mode=bench', str(job_file)]
        except Exception:
            cmd = [sys.executable, '--mmpro-mode=bench', str(job_file)]
        env = os.environ.copy()
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')
        env.setdefault('PYTHONUTF8', '1')
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        creationflags = 0
        try:
            if os.name == 'nt':
                creationflags |= subprocess.CREATE_NO_WINDOW
        except Exception:
            pass
        try:
            proc = _run_cmd( cmd, timeout=600, env=env, creationflags=creationflags,)
        except subprocess.TimeoutExpired:
            self._cprint(f'⏳ Timeout στο benchmark για {backend_pretty_name(backend)} (πιθανό hang στο load).', Colors.YELLOW, bold=True)
            return (0.0, 0.0)
        except Exception as e:
            self._cprint(f'❌ Αποτυχία εκτέλεσης subprocess benchmark ({backend}): {e}', Colors.RED, bold=True)
            return (0.0, 0.0)
        finally:
            try:
                if job_file and job_file.exists():
                    job_file.unlink(missing_ok=True)
            except Exception:
                pass

        def _extract_payload(stdout_text: str) -> dict | None:
            try:
                out_lines = [ln.strip() for ln in (stdout_text or '').splitlines() if ln.strip()]
                for ln in reversed(out_lines):
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and 'ok' in obj:
                            return obj
                    except Exception:
                        continue
            except Exception:
                return None
        if getattr(proc, 'returncode', 1) != 0:
            payload = _extract_payload(proc.stdout or '')
            if isinstance(payload, dict) and payload.get('ok') is False:
                err = str(payload.get('error', 'Άγνωστο σφάλμα'))
                self._cprint(f'💥 {backend_pretty_name(backend)} απέτυχε: {err}', Colors.RED, bold=True)
                if os.environ.get('MM_DEBUG_TRACEBACK', '').strip() in ('1', 'true', 'TRUE', 'yes', 'YES'):
                    tb = payload.get('traceback')
                    if isinstance(tb, str) and tb.strip():
                        self._cprint(tb, Colors.RED)
                return (0.0, 0.0)
            stderr = (proc.stderr or '').strip()
            hint = (stderr.splitlines()[-1] if stderr else 'native crash ή σφάλμα φόρτωσης backend')
            rc = int(getattr(proc, 'returncode', 1) or 1)
            if rc in (3221225477, -1073741819):
                hint = 'Access violation (0xC0000005) – native crash (DLL mismatch / driver / TensorRT)'
            self._cprint(f'💥 Το backend {backend_pretty_name(backend)} απέτυχε (exit={rc}): {hint}', Colors.RED, bold=True)
            return (0.0, 0.0)
        payload = _extract_payload(proc.stdout or '')
        if not payload:
            self._cprint(f'❌ Δεν επέστρεψε έγκυρο αποτέλεσμα το subprocess για {backend_pretty_name(backend)}.', Colors.RED)
            return (0.0, 0.0)
        if not payload.get('ok', False):
            self._cprint(f"❌ Σφάλμα {backend_pretty_name(backend)}: {payload.get('error','Άγνωστο σφάλμα')}", Colors.RED)
            return (0.0, 0.0)
        try:
            fps = float(payload.get('fps', 0.0) or 0.0)
            ms_per_image = float(payload.get('ms', 0.0) or 0.0)
        except Exception:
            fps, ms_per_image = 0.0, 0.0
        self._cprint(f'Αποτέλεσμα {backend_pretty_name(backend)}: {fps:.2f} FPS, {ms_per_image:.2f} ms/εικόνα', Colors.GREEN, bold=True)
        return (fps, ms_per_image)

    def run(self):
        try:
            ensure_cuda_ready_for_thread("BenchmarkWorker")
        except Exception:
            pass
        try:
            self._cprint('🚀 Έναρξη Auto-Benchmark PyTorch / ONNX / TensorRT / NCNN...', Colors.CYAN, bold=True)
            self._cprint(f'🧠 Μοντέλο βάσης: {self.base_name}', Colors.CYAN)
            self._cprint(f'📏 Image size: {self.imgsz}', Colors.CYAN)
            backends = find_available_backends(self.models_dir, self.base_name)
            if not backends:
                msg = f"Δεν βρέθηκε κανένα backend για το '{self.base_name}' στον φάκελο: {self.models_dir}"
                self._cprint(msg, Colors.RED, bold=True)
                self.error.emit(msg)
            self._cprint('🔍 Διαθέσιμα backends:', Colors.BLUE, bold=True)
            for b, p in backends.items():
                self._cprint(f'   • {backend_pretty_name(b)} → {p.name}', Colors.BLUE)
            results: list[tuple[str, float, float]] = []
            for backend, path in backends.items():
                if self._stop_requested:
                    self._cprint('🛑 Διακοπή Benchmark πριν την ολοκλήρωση.', Colors.YELLOW, bold=True)
                    break
                self._cprint('--------------------------------------------------------------------------------', Colors.CYAN)
                self._cprint(f'▶ Benchmark για: {backend_pretty_name(backend)}', Colors.MAGENTA, bold=True)
                try:
                    res = self._benchmark_backend(backend, path)
                    if res is None:
                        self._cprint('🛑 Διακοπή Benchmark.', Colors.YELLOW, bold=True)
                        break
                    fps, ms = res
                    if fps > 0:
                        results.append((backend, fps, ms))
                except Exception as e:
                    err = f'Σφάλμα στο backend {backend}: {e}'
                    self._cprint(err, Colors.RED, bold=True)
                    self._cprint(traceback.format_exc(), Colors.RED)
            if not results and (not self._stop_requested):
                msg = 'Δεν προέκυψαν αποτελέσματα benchmark (όλα τα backends απέτυχαν).'
                self._cprint(msg, Colors.RED, bold=True)
                self.error.emit(msg)
            if results:
                results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
                bar = '═' * 72
                self._cprint(bar, Colors.CYAN, bold=True)
                self._cprint('📊 ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ BENCHMARK', Colors.GREEN, bold=True)
                self._cprint(bar, Colors.CYAN, bold=True)
                medals = ['🥇', '🥈', '🥉']
                for idx, (backend, fps, ms) in enumerate(results_sorted):
                    trophy = medals[idx] if idx < len(medals) else '🏅'
                    self._cprint(f'{trophy} {backend_pretty_name(backend):<20} → {fps:8.2f} FPS | {ms:6.2f} ms/εικόνα', Colors.LIGHT if hasattr(Colors, 'LIGHT') else Colors.CYAN)
                self.results_ready.emit(results_sorted)
            if self._stop_requested:
                self._cprint('⚠️ Το Benchmark διακόπηκε μερικώς.', Colors.YELLOW)
            else:
                self._cprint('✅ Benchmark ολοκληρώθηκε.', Colors.GREEN, bold=True)
        except Exception as e:
            err_msg = f'Σφάλμα στο Benchmark: {e}'
            self._cprint(err_msg, Colors.RED, bold=True)
            self._cprint(traceback.format_exc(), Colors.RED)
            self.error.emit(err_msg)
        finally:
            self.finished.emit()
"""Live Camera worker.
- Ανοίγει/διαχειρίζεται την κάμερα σε ξεχωριστό thread (QThread worker).; - Κρατάει *σταθερό* capture resolution (στόχος: 1920x1080 όπου υποστηρίζεται).
- Τρέχει inference με YOLO backends και κάνει safe auto-recover σε mismatch input sizes.; - Στέλνει προς το UI μόνο το *τελευταίο* frame (drop-frames) για να μην μπουκώνει το GUI.
"""
import cv2


# ════════════════════════════════════════════════════════════════════════════════
# CameraWorker – Worker thread για Live Detection/Classification κάμερας
# ════════════════════════════════════════════════════════════════════════════════
# Εκτελείται σε QThread. Ανοίγει κάμερα (OpenCV), τρέχει inference frame-by-frame
# και εκπέμπει QImage για εμφάνιση στο GUI.
#
# Υποστηρίζει:
#   - YOLO backends: PyTorch (.pt), ONNX (.onnx), TensorRT (.engine), NCNN (_ncnn_model/)
#   - CNN torchvision: mobilenet_v2, mobilenet_v3_*, resnet50, resnet101
#   - Auto-recover σε camera disconnects
#   - Preview crop (black bars removal, 16:9 enforce)
#   - Global camera lock (αποτρέπει ταυτόχρονο άνοιγμα από Live + Benchmark)
# ════════════════════════════════════════════════════════════════════════════════
class CameraWorker(QObject, LogEmitMixin, StoppableMixin):
    log = Signal(str)
    frame_ready = Signal(QImage)
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_info: tuple, imgsz: int, classes_filter: list | None=None, use_tensorrt: bool=False, conf_threshold: float=0.25, camera_index: int = 0):
        super().__init__()
        self.model_path, self.model_type = model_info
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.camera_index = int(camera_index or 0)
        self._is_running = False
        self._latest_lock = threading.Lock()
        self._latest_qimg = None
        self._latest_fps = 0.0
        self._latest_infer_fps = 0.0
        self._latest_loop_fps = 0.0
        self._latest_overlay_text = ''
        self._latest_ts = 0.0
        self.classes_filter = classes_filter
        self.use_tensorrt = use_tensorrt
        self.model = None
        self.cap = None
        self.runtime_label = self.model_type
        self._preview_target_ar = 16.0 / 9.0
        self._preview_crop_base = None
        self._preview_crop_cuts = None
        self._preview_crop_check_counter = 0
        self._cnn_helper: 'CNNInferenceHelper | None' = None  # CNN torchvision helper

    def _cprint(self, text: str, color: str | None=None, bold: bool=False):
        self.log.emit(format_html_log(text, color, bold))

    def stop(self):
        self._is_running = False

    def get_latest_qimage(self):
        try:
            with self._latest_lock:
                return self._latest_qimg
        except Exception:
            return None

    def get_latest_fps(self) -> float:
        try:
            with self._latest_lock:
                return float(self._latest_fps)
        except Exception:
            return 0.0

    def get_latest_overlay_text(self) -> str:
        try:
            with self._latest_lock:
                return str(self._latest_overlay_text or '')
        except Exception:
            return ''

    def _enforce_fixed_camera_resolution(self, when: str = '') -> None:
        if self.cap is None or cv2 is None:
            return
        try:
            fixed_on = str(os.environ.get('MM_PRO_CAM_FIXED_1080', '1')).strip().lower() in ('1','true','yes','on')
        except Exception:
            fixed_on = True
        if not fixed_on:
            return
        try:
            ok = force_capture_resolution(self.cap, 1920, 1080, verify=True)
        except Exception:
            ok = False
        aw = ah = 0
        fw = fh = 0
        try:
            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            aw, ah = 0, 0
        try:
            okg = self.cap.grab()
            if okg:
                retf, fr = self.cap.retrieve()
                if retf and fr is not None and getattr(fr, 'size', 0) != 0:
                    fh, fw = fr.shape[:2]
        except Exception:
            fw, fh = 0, 0
        tag = f' ({when})' if when else ''
        try:
            if ok:
                shown = '1080p'
                if fw and fh:
                    shown = f'{fw} x {fh}'
                elif aw and ah:
                    shown = f'{aw} x {ah}'
                self._cprint(f'🎯 1080p CAPTURE LOCK OK{tag}: {shown}', Colors.GREEN, bold=True)
            else:
                try:
                    apply_best_camera_resolution(self.cap, 0, int(getattr(self, '_camera_api', 0) or 0))
                    aw2 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    ah2 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                except Exception:
                    aw2, ah2 = aw, ah
                self._cprint( f'⚠️ Δεν κλείδωσε 1920x1080{tag}. Actual: {aw2} x {ah2} (η κάμερα/driver ίσως δεν το υποστηρίζει στο τρέχον mode).', Colors.YELLOW, bold=True,)
        except Exception:
            pass

    def _center_crop_to_ar(self, frame, desired_ar: float):
        try:
            if frame is None:
                return frame
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return frame
            cur_ar = float(w) / float(h)
            if abs(cur_ar - desired_ar) < 0.01:
                return frame
            if cur_ar > desired_ar:
                new_w = max(2, int(round(float(h) * desired_ar)))
                x0 = max(0, int((w - new_w) / 2))
                return frame[:, x0:x0 + new_w]
            else:
                new_h = max(2, int(round(float(w) / desired_ar)))
                y0 = max(0, int((h - new_h) / 2))
                return frame[y0:y0 + new_h, :]
        except Exception:
            return frame

    def _detect_black_bars_cuts(self, frame):
        try:
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if h < 60 or w < 80:
                return None
            try:
                if frame.ndim == 2:
                    gray = frame
                elif frame.shape[2] == 4:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = frame[:, :, 0] if frame.ndim == 3 else frame
            cy0 = int(h * 0.40)
            cy1 = int(h * 0.60)
            cx0 = int(w * 0.40)
            cx1 = int(w * 0.60)
            mid = gray[cy0:cy1, cx0:cx1]
            mid_mean = float(mid.mean()) if getattr(mid, 'size', 0) else float(gray.mean())
            mean_th = float(min(20.0, max(6.0, mid_mean * 0.20)))
            std_th = 10.0
            row_mean = gray.mean(axis=1)
            row_std = gray.std(axis=1)
            col_mean = gray.mean(axis=0)
            col_std = gray.std(axis=0)
            max_v_cut = int(h * 0.28)
            max_h_cut = int(w * 0.28)
            is_bar_row = (row_mean <= mean_th) & (row_std <= std_th)
            is_bar_col = (col_mean <= mean_th) & (col_std <= std_th)
            top = 0
            for i in range(max_v_cut):
                if bool(is_bar_row[i]):
                    top += 1
                else:
                    break
            bottom = 0
            for i in range(max_v_cut):
                if bool(is_bar_row[h - 1 - i]):
                    bottom += 1
                else:
                    break
            left = 0
            for i in range(max_h_cut):
                if bool(is_bar_col[i]):
                    left += 1
                else:
                    break
            right = 0
            for i in range(max_h_cut):
                if bool(is_bar_col[w - 1 - i]):
                    right += 1
                else:
                    break
            min_px = 12
            if top < min_px and bottom < min_px and left < min_px and right < min_px:
                return None
            new_h = h - top - bottom
            new_w = w - left - right
            if new_h < int(h * 0.70) or new_w < int(w * 0.70):
                return None
            return (int(top), int(bottom), int(left), int(right))
        except Exception:
            return None

    def _apply_preview_crop_policy(self, frame):
        try:
            if frame is None:
                return frame
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return frame
            try:
                enable_blackbars = str(os.environ.get('MM_PRO_PREVIEW_CROP_BLACKBARS', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                enable_blackbars = True
            try:
                force_169 = str(os.environ.get('MM_PRO_PREVIEW_FORCE_16_9', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                force_169 = True
            self._preview_crop_check_counter = int(getattr(self, '_preview_crop_check_counter', 0)) + 1
            need_redetect = False
            base = getattr(self, '_preview_crop_base', None)
            if base is None or base != (int(w), int(h)):
                need_redetect = True
            elif self._preview_crop_check_counter % 180 == 0:
                need_redetect = True
            if need_redetect:
                cuts = self._detect_black_bars_cuts(frame) if enable_blackbars else None
                if cuts is not None:
                    self._preview_crop_base = (int(w), int(h))
                    self._preview_crop_cuts = cuts
                else:
                    self._preview_crop_base = (int(w), int(h))
                    self._preview_crop_cuts = None
            cuts = getattr(self, '_preview_crop_cuts', None)
            if cuts is not None:
                top, bottom, left, right = cuts
                y0 = max(0, int(top))
                y1 = int(h - bottom) if bottom > 0 else int(h)
                x0 = max(0, int(left))
                x1 = int(w - right) if right > 0 else int(w)
                if (y1 - y0) > 10 and (x1 - x0) > 10:
                    frame = frame[y0:y1, x0:x1]
            if force_169:
                frame = self._center_crop_to_ar(frame, float(getattr(self, '_preview_target_ar', 16.0/9.0)))
            return frame
        except Exception:
            return frame

    # ── Φόρτωση μοντέλου πριν την έναρξη inference ──────────────────────────────
    # Ελέγχει πρώτα αν πρόκειται για CNN (torchvision) μέσω _is_cnn_path().
    # CNN: αρχικοποιεί CNNInferenceHelper και θέτει self._cnn_helper.
    # YOLO: φορτώνει μέσω Ultralytics YOLO() για PyTorch/ONNX/TensorRT/NCNN.
    def load_model(self) -> bool:
        try:
            self._cprint('Φόρτωση μοντέλου.', Colors.CYAN)
            self._cprint(f'Διαδρομή: {self.model_path}', Colors.CYAN)
            if not self.model_path.exists():
                raise FileNotFoundError(f'Το μοντέλο δεν βρέθηκε: {self.model_path}')
            # ── CNN classifier (torchvision) ──────────────────────────────
            stem = self.model_path.stem.lower()
            is_cnn = any(c in stem for c in _CNN_MODEL_KEYS)
            # Also detect by checkpoint keys
            if not is_cnn:
                try:
                    import torch as _torch_chk
                    ckpt_peek = _torch_chk.load(str(self.model_path), map_location='cpu', weights_only=False)
                    if isinstance(ckpt_peek, dict) and 'model_name' in ckpt_peek:
                        mn = str(ckpt_peek.get('model_name', '')).lower()
                        is_cnn = any(c in mn for c in _CNN_MODEL_KEYS)
                except Exception:
                    pass
            if is_cnn:
                self._cprint('🧠 Εντοπίστηκε CNN μοντέλο (torchvision). Φόρτωση CNNInferenceHelper…', Colors.CYAN)
                dev_str = 'cuda:0' if (self.model_type in ('PyTorch', 'CNN') and
                                        __import__('torch').cuda.is_available()) else 'cpu'
                self._cnn_helper = CNNInferenceHelper(self.model_path, device=dev_str)
                self._cnn_helper.load()
                self.model = None  # not a YOLO model
                self.runtime_label = f'CNN/{self._cnn_helper.model_name_str}'
                self._cprint(
                    f'✅ CNN μοντέλο φορτώθηκε: {self._cnn_helper.model_name_str} | '
                    f'{self._cnn_helper.num_classes} κλάσεις | imgsz={self._cnn_helper.imgsz}',
                    Colors.GREEN)
                return True
            # ── YOLO path (existing logic) ────────────────────────────────
            self._cnn_helper = None
            task = guess_ultralytics_task(self.model_path)
            if self.model_type == 'PyTorch':
                self.model = YOLO(str(self.model_path))
                self.runtime_label = 'PyTorch'
            elif self.model_type == 'NCNN':
                self.model = YOLO(str(self.model_path), task=task)
                self.runtime_label = 'NCNN'
                try:
                    forced = None
                    meta = _mmpro_read_export_meta_for_path(self.model_path)
                    v = meta.get('imgsz') if isinstance(meta, dict) else None
                    if isinstance(v, int) and v > 0:
                        forced = int(v)
                    elif isinstance(v, (tuple, list)) and len(v) == 2:
                        h, w = v
                        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                            forced = (int(h), int(w))
                    if forced is None and self.model_path.is_dir():
                        g = _mmpro_try_infer_ncnn_imgsz_from_param(self.model_path)
                        if isinstance(g, int) and g > 0:
                            forced = int(g)
                    if forced is not None:
                        try:
                            setattr(self.model, '_mmpro_forced_imgsz', forced)
                        except Exception:
                            pass
                        if isinstance(forced, int) and forced > 0:
                            self.imgsz = int(forced)
                        self._cprint('🧩 NCNN: auto input lock (meta/param) για σωστό scaling.', Colors.CYAN)
                except Exception:
                    pass
            elif self.model_type == 'ONNX':
                engine_path = self.model_path.with_suffix('.engine')
                if self.use_tensorrt and engine_path.exists():
                    self._cprint( f'Φόρτωση TensorRT engine αντί για ONNX: {engine_path.name}', Colors.MAGENTA, bold=True,)
                    self.model = YOLO(str(engine_path), task=task)
                    self.runtime_label = 'TensorRT'
                    try:
                        exp = None
                        try:
                            sig_path = trt_signature_path_for_engine(engine_path)
                            if sig_path.exists():
                                sig = json.loads(sig_path.read_text(encoding='utf-8', errors='replace'))
                                exp = int((sig.get('params') or {}).get('imgsz') or 0)
                        except Exception:
                            exp = None
                        if not exp:
                            exp = _mmpro_parse_imgsz_from_name(engine_path.stem) or _mmpro_parse_imgsz_from_name(engine_path.parent.name)
                        if exp and int(exp) > 0 and int(exp) != int(self.imgsz):
                            self._cprint( f'🧷 TensorRT fixed input: imgsz={int(exp)} (ήταν {self.imgsz}) – auto-adjust για Live Camera.', Colors.YELLOW, bold=True,)
                            self.imgsz = int(exp)
                    except Exception:
                        pass
                else:
                    if self.use_tensorrt and (not engine_path.exists()):
                        self._cprint( 'Ζητήθηκε TensorRT αλλά δεν βρέθηκε .engine – γίνεται φόρτωση ONNX.', Colors.YELLOW, bold=True,)
                    self.model = YOLO(str(self.model_path), task=task)
                    self.runtime_label = 'ONNX'
            elif self.model_type == 'TensorRT':
                # Άμεση φόρτωση .engine αρχείου (επιλεγμένο απευθείας από combo)
                self._cprint(f'Φόρτωση TensorRT engine: {self.model_path.name}', Colors.MAGENTA, bold=True)
                self.model = YOLO(str(self.model_path), task=task)
                self.runtime_label = 'TensorRT'
                try:
                    exp = None
                    try:
                        sig_path = trt_signature_path_for_engine(self.model_path)
                        if sig_path.exists():
                            sig = json.loads(sig_path.read_text(encoding='utf-8', errors='replace'))
                            exp = int((sig.get('params') or {}).get('imgsz') or 0)
                    except Exception:
                        exp = None
                    if not exp:
                        exp = (_mmpro_parse_imgsz_from_name(self.model_path.stem) or
                               _mmpro_parse_imgsz_from_name(self.model_path.parent.name))
                    if exp and int(exp) > 0 and int(exp) != int(self.imgsz):
                        self._cprint(
                            f'🧷 TensorRT fixed input: imgsz={int(exp)} (ήταν {self.imgsz}) – auto-adjust.',
                            Colors.YELLOW, bold=True)
                        self.imgsz = int(exp)
                except Exception:
                    pass
            else:
                raise ValueError(
                    f'Άγνωστος τύπος μοντέλου: {self.model_type!r}. '
                    f'Αποδεκτές τιμές: PyTorch, CNN, ONNX, TensorRT, NCNN'
                )
            if self.use_tensorrt and self.model_type == 'ONNX':
                self._cprint( 'TensorRT checkbox ενεργό – αν υπάρχει διαθέσιμο backend θα χρησιμοποιηθεί αυτόματα.', Colors.MAGENTA,)
            self._cprint('Μοντέλο φορτώθηκε.', Colors.GREEN, bold=True)
            return True
        except Exception as e:
            self._log_exc('Φόρτωση μοντέλου', e, extra={
                'Αρχείο': getattr(self, 'model_path', '?'),
                'Τύπος':  getattr(self, 'model_type', '?'),
            })
            return False

    def _recover_camera(self, reason: str = '') -> bool:
        try:
            msg = f"♻️ Επανασύνδεση κάμερας… ({reason})" if reason else "♻️ Επανασύνδεση κάμερας…"
            self._cprint(msg, Colors.YELLOW, bold=True)
        except Exception:
            pass
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            time.sleep(0.25)
        except Exception:
            pass
        try:
            self.cap, _api = open_video_capture(self.camera_index)
            self._camera_api = int(_api or 0)
        except Exception as e:
            try:
                self._cprint(f"❌ Αποτυχία ανοίγματος κάμερας στο recover: {e}", Colors.RED, bold=True)
            except Exception:
                pass
            return False
        try:
            good = 0
            for _ in range(10):
                try:
                    ok = self.cap.grab()
                except Exception:
                    ok = False
                if not ok:
                    time.sleep(0.02)
                    continue
                try:
                    ret, fr = self.cap.retrieve()
                except Exception:
                    ret, fr = False, None
                if ret and fr is not None and getattr(fr, 'size', 0) != 0:
                    good += 1
                    if good >= 3:
                        break
                else:
                    good = 0
                time.sleep(0.01)
            if good < 3:
                try:
                    self._cprint("❌ Recover: η κάμερα άνοιξε αλλά εξακολουθεί να δίνει άκυρα frames.", Colors.RED)
                except Exception:
                    pass
                return False
        except Exception:
            return False
        try:
            self._enforce_fixed_camera_resolution('recover')
        except Exception:
            pass
        try:
            self._cprint(f"✅ Recover OK | Camera API={getattr(self,'_camera_api',0)}", Colors.GREEN, bold=True)
        except Exception:
            pass
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                self._cprint(f'📐 Ανάλυση κάμερας (μετά το recover): {w} x {h}', Colors.CYAN)
        except Exception:
            pass
        return True

    def run(self):
        if cv2 is None:
            self.error.emit('Λείπει το OpenCV (cv2). Εγκατάστησέ το με: pip install opencv-python')
            self.finished.emit()
            return
        try:
            ensure_cuda_ready_for_thread("CameraWorker")
        except Exception:
            pass
        try:
            configure_opencv_runtime()
            ensure_windows_com_initialized()
            ensure_windows_media_foundation_started()
        except Exception:
            pass
        if not self.load_model():
            self.finished.emit()
            return
        if not acquire_camera_lock('CameraWorker(Live)'):
            owner = get_camera_lock_owner()
            msg = f'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή: {owner}' if owner else 'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή.'
            self.error.emit(msg + '\nΣταμάτα το άλλο tab και ξαναδοκίμασε.')
            self.finished.emit()
            return
        try:
            self.cap, _api = open_video_capture(self.camera_index)
            self._camera_api = int(_api or 0)
        except Exception as e:
            self.error.emit(f'Δεν ήταν δυνατό το άνοιγμα της κάμερας: {e}')
            try:
                release_camera_lock()
            except Exception:
                pass
            self.finished.emit()
            return
        try:
            self._cprint(f'OpenCV: {getattr(cv2, "__version__", "?")} | Camera API={getattr(self, "_camera_api", 0)}', Colors.CYAN)
        except Exception:
            pass
        self._enforce_fixed_camera_resolution('initial')
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                self._cprint(f'📐 Ανάλυση κάμερας: {w} x {h}', Colors.CYAN)
        except Exception:
            pass
        try:
            if IS_WINDOWS:
                auto_mjpg = str(os.environ.get('MM_PRO_CAM_MJPG_AUTO', '0')).strip().lower() in ('1','true','yes','on')
                if auto_mjpg and hasattr(cv2, 'CAP_DSHOW') and int(getattr(self, '_camera_api', 0)) == int(getattr(cv2, 'CAP_DSHOW')):
                    prev_fourcc = 0
                    prev_fourcc_str = ''
                    try:
                        prev_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
                        prev_fourcc_str = ''.join([chr((prev_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                    except Exception:
                        prev_fourcc = 0
                        prev_fourcc_str = ''
                    if 'MJPG' not in (prev_fourcc_str or ''):
                        try:
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        except Exception:
                            pass
                        self._enforce_fixed_camera_resolution('after MJPG auto')
                        fw = fh = 0
                        try:
                            okg = self.cap.grab()
                            if okg:
                                retf, fr = self.cap.retrieve()
                                if retf and fr is not None and getattr(fr, 'size', 0) != 0:
                                    fh, fw = fr.shape[:2]
                        except Exception:
                            fw, fh = 0, 0
                        if not (fw == 1920 and fh == 1080) and prev_fourcc:
                            try:
                                self._cprint('↩️ MJPG auto κράτησε χαμηλή ανάλυση → rollback FOURCC στην προηγούμενη μορφή.', Colors.YELLOW, bold=True)
                            except Exception:
                                pass
                            try:
                                self.cap.set(cv2.CAP_PROP_FOURCC, int(prev_fourcc))
                            except Exception:
                                pass
                            self._enforce_fixed_camera_resolution('after MJPG rollback')
        except Exception:
            pass
        try:
            w2 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h2 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w2 > 0 and h2 > 0:
                self._cprint(f'📐 Τελική ανάλυση κάμερας: {w2} x {h2}', Colors.CYAN, bold=True)
        except Exception:
            pass
        if not self.cap.isOpened():
            self.error.emit('Δεν ήταν δυνατό το άνοιγμα της κάμερας.')
            try:
                release_camera_lock()
            except Exception:
                pass
            self.finished.emit()
            return
        self._is_running = True
        sep = '═' * 72
        sub_sep = '─' * 72
        header = f'📷 Έναρξη Live Ανίχνευσης | Μοντέλο: {self.model_path.name} | Runtime: {self.runtime_label} | Camera: 1080p | conf={self.conf_threshold:.2f}'
        self._cprint(sep, Colors.CYAN, bold=True)
        self._cprint(header, Colors.CYAN, bold=True)
        self._cprint(sub_sep, Colors.CYAN, bold=False)
        bad_frames = 0
        recover_attempts = 0
        fps_loop = 0.0
        infer_ema = None
        fps_window_start = time.perf_counter()
        fps_window_frames = 0
        ema_alpha = 0.15
        try:
            while self._is_running:
                last_read_exc = None
                try:
                    ok = self.cap.grab()
                    if ok:
                        ret, frame = self.cap.retrieve()
                    else:
                        ret, frame = False, None
                except cv2.error as e:
                    ret, frame = False, None
                    last_read_exc = e
                except Exception as e:
                    ret, frame = False, None
                    last_read_exc = e
                if last_read_exc is not None:
                    safe_log_once('Camera:read', '⚠️ Σφάλμα ανάγνωσης κάμερας (grab/retrieve)', last_read_exc, group='Camera')
                if (not ret) or (frame is None) or (getattr(frame, 'size', 0) == 0):
                    bad_frames += 1
                    try:
                        if last_read_exc is not None and (bad_frames == 1 or bad_frames % 30 == 0):
                            self._cprint(f"⚠️ OpenCV σφάλμα κατά την ανάγνωση frame: {last_read_exc}", Colors.YELLOW)
                    except Exception:
                        pass
                    if bad_frames in (45, 90) and recover_attempts < 2:
                        if self._recover_camera(reason=f'{bad_frames} συνεχόμενα άκυρα frames'):
                            recover_attempts += 1
                            bad_frames = 0
                            continue
                    if bad_frames >= 120:
                        try:
                            safe_log_error( 'Camera stopped λόγω πολλών άκυρων frames (πιθανό hang/driver issue).', None, crash_log=True, crash_tag='camera_invalid_frames', dump_all_threads=True,)
                        except Exception:
                            pass
                        self.error.emit('Η κάμερα δεν παρέχει έγκυρα καρέ (πολλά συνεχόμενα άκυρα frames).\n''✔️ Έλεγξε ότι δεν τρέχει άλλο tab (Benchmark/Live) ή άλλη εφαρμογή που κρατάει την κάμερα.\n''💡 Δοκίμασε: MM_PRO_CAM_MJPG_AUTO=0 (disable MJPG auto) ή MM_PRO_CAM_BACKEND=dshow.')
                        break
                    try:
                        time.sleep(0.005)
                    except Exception:
                        pass
                    continue
                bad_frames = 0
                try:
                    if not isinstance(frame, np.ndarray):
                        frame = np.asarray(frame)
                    if getattr(frame, 'dtype', None) is not None and frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8, copy=False)
                    if hasattr(frame, 'flags') and (not frame.flags['C_CONTIGUOUS']):
                        frame = np.ascontiguousarray(frame)
                except Exception:
                    try:
                        frame = np.ascontiguousarray(np.array(frame, dtype=np.uint8))
                    except Exception:
                        pass
                infer_time = 0.0
                try:
                    try:
                        cuda_sync()
                    except Exception:
                        pass
                    t_inf0 = time.perf_counter()
                    # ── CNN inference: predict_frame() μία φορά → _draw_predictions() ─
                    # Το predict_frame() γίνεται χωριστά ώστε τα preds να χρησιμοποιηθούν
                    # και για το FPS overlay bar (top-1 class + confidence).
                    _cnn_h = getattr(self, '_cnn_helper', None)
                    if _cnn_h is not None:
                        _cnn_preds_all = _cnn_h.predict_frame(frame, top_k=len(_cnn_h.class_names) or 5)
                        # ── Εφαρμογή classes_filter ────────────────────────────
                        # classes_filter: None = εμφάνιση όλων, [] = καμία, [id,...] = επιλεγμένες
                        _cf = self.classes_filter
                        if _cf is None:
                            # Χωρίς φίλτρο: κράτα top-5
                            _cnn_preds = _cnn_preds_all[:5]
                        elif len(_cf) == 0:
                            # Καμία κλάση επιλεγμένη: εμφάνισε κενό frame
                            _cnn_preds = []
                        else:
                            # Κράτα μόνο τις επιλεγμένες κλάσεις (κατά class_id)
                            _cnn_preds = [
                                p for p in _cnn_preds_all
                                if len(p) > 2 and int(p[2]) in _cf
                            ][:5]
                        annotated_frame = _cnn_h._draw_predictions(frame, _cnn_preds)
                        results = None
                        is_classification_model = True
                    else:
                        # ── YOLO inference ─────────────────────────────
                        if self.model is None:
                            raise RuntimeError(
                                'Το μοντέλο YOLO δεν έχει φορτωθεί (model=None). '
                                'Έλεγξε ότι το αρχείο μοντέλου υπάρχει και είναι έγκυρο.')
                        is_classification_model = yolo_is_classification(self.model)
                        is_ncnn = (getattr(self, 'model_type', '') == 'NCNN')
                        if is_classification_model:
                            results = yolo_predict_first(self.model, frame, imgsz=self.imgsz, conf=self.conf_threshold, iou=0.45, verbose=False)
                            if is_ncnn:
                                results = _mmpro_fix_ncnn_results(results, frame, self.imgsz)
                            annotated_frame = results.plot(font_size=0.36)
                        elif self.classes_filter is not None and len(self.classes_filter) == 0:
                            results = None
                            annotated_frame = frame.copy()
                        else:
                            classes_arg = self.classes_filter if self.classes_filter else None
                            results = yolo_predict_first(self.model, frame, imgsz=self.imgsz, conf=self.conf_threshold, iou=0.45, verbose=False, classes=classes_arg)
                            if is_ncnn:
                                annotated_frame = _ncnn_manual_annotate(frame, results, self.imgsz)
                            else:
                                annotated_frame = results.plot(font_size=0.36)
                    try:
                        if hasattr(annotated_frame, 'flags') and (not annotated_frame.flags.writeable):
                            annotated_frame = annotated_frame.copy()
                    except Exception:
                        annotated_frame = annotated_frame.copy()
                    try:
                        cuda_sync()
                    except Exception:
                        pass
                    t_inf1 = time.perf_counter()
                    infer_time = max(0.0, t_inf1 - t_inf0)
                    if infer_time > 0:
                        if infer_ema is None:
                            infer_ema = infer_time
                        else:
                            infer_ema = (1.0 - ema_alpha) * infer_ema + ema_alpha * infer_time
                except Exception as e:
                    self._log_exc('Inference / Πρόβλεψη', e, extra={
                        'Μοντέλο': getattr(self, 'model_path', '?'),
                        'Backend': getattr(self, 'model_type', '?'),
                        'imgsz':   getattr(self, 'imgsz', '?'),
                        'model':   'None (δεν φορτώθηκε)' if self.model is None else 'OK',
                    })
                    break
                infer_fps = (1.0 / infer_ema) if (infer_ema is not None and infer_ema > 0) else 0.0
                fps_window_frames += 1
                _elapsed = time.perf_counter() - fps_window_start
                if _elapsed >= 0.50:
                    fps_loop = fps_window_frames / _elapsed if _elapsed > 0 else 0.0
                    fps_window_frames = 0
                    fps_window_start = time.perf_counter()
                try:
                    fh, fw = frame.shape[:2]
                except Exception:
                    fw, fh = 0, 0
                cap_txt = '1080p'
                base_text = f'FPS : {infer_fps:.0f} | LOOP : {fps_loop:.0f} | {self.runtime_label} | CAP : {cap_txt}'
                overlay_text = base_text
                classification_label = None
                classification_conf = None
                try:
                    # ── CNN: top-1 in FPS bar (μετά το classes_filter) ─────
                    _cnn_h2 = getattr(self, '_cnn_helper', None)
                    if _cnn_h2 is not None:
                        # _cnn_preds είναι ήδη φιλτραρισμένο από το inference block
                        _cp = locals().get('_cnn_preds', [])
                        if _cp:
                            # tuple: (name, conf, class_id)
                            top_cls  = _cp[0][0]
                            top_conf = _cp[0][1]
                            overlay_text = (f'{base_text} | '
                                            f'🏷 {top_cls}  {top_conf*100:.1f}%')
                        else:
                            # Καμία επιλεγμένη κλάση
                            overlay_text = f'{base_text} | 🏷 —'
                    elif is_classification_model:
                        probs = getattr(results, 'probs', None)
                        top1_idx = None
                        top1_conf = None
                        if probs is not None:
                            if hasattr(probs, 'top1'):
                                try:
                                    top1_idx = int(probs.top1)
                                except Exception:
                                    top1_idx = None
                            if hasattr(probs, 'top1conf'):
                                try:
                                    top1_conf = float(probs.top1conf)
                                except Exception:
                                    top1_conf = None
                            if top1_idx is None and hasattr(probs, 'data'):
                                try:
                                    import numpy as _np
                                    data = probs.data
                                    if hasattr(data, 'cpu'):
                                        data = data.cpu().numpy()
                                    top1_idx = int(_np.argmax(data))
                                except Exception:
                                    top1_idx = None
                        if top1_idx is not None:
                            names = getattr(results, 'names', None)
                            class_name = f'Class_{top1_idx}'
                            if isinstance(names, (list, tuple)):
                                if 0 <= top1_idx < len(names):
                                    class_name = str(names[top1_idx])
                            elif isinstance(names, dict):
                                class_name = str(names.get(top1_idx, class_name))
                            classification_label = class_name
                            classification_conf = top1_conf
                            if self.classes_filter is not None:
                                if len(self.classes_filter) == 0 or top1_idx not in self.classes_filter:
                                    classification_label = None
                                    classification_conf = None
                except Exception:
                    overlay_text = base_text
                try:
                    if is_classification_model and classification_label is not None:
                        if classification_conf is not None and 0.0 <= classification_conf <= 1.0:
                            perc_val = classification_conf * 100.0
                            overlay_text = f"{overlay_text} | {classification_label} ({perc_val:.1f}%)"
                        else:
                            overlay_text = f"{overlay_text} | {classification_label}"
                except Exception:
                    pass
                try:
                    if not isinstance(annotated_frame, np.ndarray):
                        annotated_frame = np.asarray(annotated_frame)
                    if getattr(annotated_frame, "dtype", None) is not None and annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8, copy=False)
                    if hasattr(annotated_frame, "flags") and (not annotated_frame.flags["C_CONTIGUOUS"]):
                        annotated_frame = np.ascontiguousarray(annotated_frame)
                except Exception:
                    annotated_frame = np.ascontiguousarray(np.array(annotated_frame))
                try:
                    annotated_frame = self._apply_preview_crop_policy(annotated_frame)
                except Exception:
                    pass
                try:
                    if annotated_frame is not None:
                        _h, _w = annotated_frame.shape[:2]
                        if _w != 1920 or _h != 1080:
                                                    tw, th = 1920, 1080
                                                    scale = min(tw / max(1, _w), th / max(1, _h))
                                                    nw = max(1, int(_w * scale))
                                                    nh = max(1, int(_h * scale))
                                                    resized = cv2.resize(annotated_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                                    canvas = np.zeros((th, tw, resized.shape[2]) if resized.ndim == 3 else (th, tw), dtype=resized.dtype)
                                                    x0 = int((tw - nw) / 2)
                                                    y0 = int((th - nh) / 2)
                                                    canvas[y0:y0+nh, x0:x0+nw] = resized
                                                    annotated_frame = canvas
                except Exception:
                    pass
                height, width = annotated_frame.shape[:2]
                try:
                    if annotated_frame.ndim == 2:
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2RGB)
                    elif annotated_frame.shape[2] == 4:
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2RGB)
                    else:
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    if annotated_frame.ndim == 3 and annotated_frame.shape[2] >= 3:
                        annotated_frame_rgb = annotated_frame[:, :, :3][:, :, ::-1].copy()
                    else:
                        annotated_frame_rgb = np.stack([annotated_frame] * 3, axis=-1).astype(np.uint8, copy=False)
                try:
                    if hasattr(annotated_frame_rgb, "flags") and (not annotated_frame_rgb.flags["C_CONTIGUOUS"]):
                        annotated_frame_rgb = np.ascontiguousarray(annotated_frame_rgb)
                except Exception:
                    annotated_frame_rgb = np.ascontiguousarray(annotated_frame_rgb)
                h2, w2 = annotated_frame_rgb.shape[:2]
                ch = int(annotated_frame_rgb.shape[2]) if annotated_frame_rgb.ndim == 3 else 1
                bytes_per_line = int(annotated_frame_rgb.strides[0])
                q_img = QImage(annotated_frame_rgb.data, w2, h2, bytes_per_line, QImage.Format.Format_RGB888).copy()
                try:
                    with self._latest_lock:
                        self._latest_qimg = q_img
                        self._latest_fps = float(fps_loop) if 'fps_loop' in locals() else 0.0
                        self._latest_infer_fps = float(infer_fps) if 'infer_fps' in locals() else 0.0
                        self._latest_loop_fps = float(fps_loop) if 'fps_loop' in locals() else 0.0
                        self._latest_overlay_text = str(overlay_text) if 'overlay_text' in locals() else ''
                        self._latest_ts = time.time()
                except Exception:
                    pass
                try:
                    time.sleep(0)
                except Exception:
                    pass
        except Exception as e:
            import traceback
            try:
                tb = traceback.format_exc()
                self.error.emit(f'Απρόσμενο σφάλμα Live Κάμερας: {e}')
                print(tb, file=sys.stderr)
            except Exception:
                pass
        try:
            if self.cap:
                self.cap.release()
        finally:
            try:
                release_camera_lock()
            except Exception:
                pass
        self.finished.emit()
"""Camera benchmark worker.
Μετρά FPS σε πραγματική κάμερα για κάθε διαθέσιμο backend του ίδιου μοντέλου
(PyTorch/ONNX/TensorRT/NCNN). Περιλαμβάνει:
- global camera lock (Live vs Benchmark να μην ανοίγουν ταυτόχρονα την ίδια κάμερα); - σταθερό capture policy (στόχος: 1080p όπου γίνεται); - preview-only crop (αφαίρεση baked-in black bars + διατήρηση 16:9)
"""
import cv2
if TYPE_CHECKING:
    from ultralytics import YOLO


# ════════════════════════════════════════════════════════════════════════════════
# CameraBenchmarkWorker – Worker για benchmark μοντέλων σε live κάμερα
# ════════════════════════════════════════════════════════════════════════════════
# Μετρά FPS για κάθε διαθέσιμο backend (PyTorch/ONNX/TensorRT/NCNN/CNN) του
# ίδιου μοντέλου. Χρησιμοποιεί πραγματικά frames κάμερας (όχι dummy frames).
#
# Υποστήριξη CNN:
#   - Εντοπίζει CNN μοντέλα με _is_cnn_path() πριν την φόρτωση.
#   - Φορτώνει CNNInferenceHelper αντί YOLO.
#   - Εμφανίζει το ίδιο bottom-left overlay (κλάσεις + ποσοστά + μπάρες).
#   - Εμπλουτίζει το FPS overlay bar με top-1 κλάση.
# ════════════════════════════════════════════════════════════════════════════════
class CameraBenchmarkWorker(QObject, LogEmitMixin, StoppableMixin):
    log = Signal(str)
    frame_ready = Signal(QImage)
    finished = Signal()
    error = Signal(str)
    results_ready = Signal(list)

    def __init__(self, base_name: str, imgsz: int, duration_sec: int, camera_index: int, models_dir: Path, parent: QObject | None=None):
        super().__init__(parent)
        self.base_name = base_name
        self.imgsz = imgsz
        self.duration_sec = max(1, duration_sec)
        self.camera_index = camera_index
        self.models_dir = models_dir
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self._is_running = False
        self.cap = None
        self.runtime_label = ''
        self._overlay_lock = threading.Lock()
        self._latest_overlay_text = ''
        self._cnn_helper: 'CNNInferenceHelper | None' = None  # CNN torchvision helper

    def get_latest_overlay_text(self) -> str:
        try:
            with self._overlay_lock:
                return str(self._latest_overlay_text or '')
        except Exception:
            return ''

    def _center_crop_to_ar(self, frame, desired_ar: float):
        try:
            if frame is None:
                return frame
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return frame
            cur_ar = float(w) / float(h)
            if abs(cur_ar - desired_ar) < 0.01:
                return frame
            if cur_ar > desired_ar:
                new_w = max(2, int(round(float(h) * desired_ar)))
                x0 = max(0, int((w - new_w) / 2))
                return frame[:, x0:x0 + new_w]
            else:
                new_h = max(2, int(round(float(w) / desired_ar)))
                y0 = max(0, int((h - new_h) / 2))
                return frame[y0:y0 + new_h, :]
        except Exception:
            return frame

    def _detect_black_bars_cuts(self, frame):
        try:
            if frame is None:
                return None
            h, w = frame.shape[:2]
            if h < 60 or w < 80:
                return None
            try:
                if frame.ndim == 2:
                    gray = frame
                elif frame.shape[2] == 4:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = frame[:, :, 0] if frame.ndim == 3 else frame
            cy0 = int(h * 0.40)
            cy1 = int(h * 0.60)
            cx0 = int(w * 0.40)
            cx1 = int(w * 0.60)
            mid = gray[cy0:cy1, cx0:cx1]
            mid_mean = float(mid.mean()) if getattr(mid, 'size', 0) else float(gray.mean())
            mean_th = float(min(20.0, max(6.0, mid_mean * 0.20)))
            std_th = 10.0
            row_mean = gray.mean(axis=1)
            row_std = gray.std(axis=1)
            col_mean = gray.mean(axis=0)
            col_std = gray.std(axis=0)
            max_v_cut = int(h * 0.28)
            max_h_cut = int(w * 0.28)
            is_bar_row = (row_mean <= mean_th) & (row_std <= std_th)
            is_bar_col = (col_mean <= mean_th) & (col_std <= std_th)
            top = 0
            for i in range(max_v_cut):
                if bool(is_bar_row[i]):
                    top += 1
                else:
                    break
            bottom = 0
            for i in range(max_v_cut):
                if bool(is_bar_row[h - 1 - i]):
                    bottom += 1
                else:
                    break
            left = 0
            for i in range(max_h_cut):
                if bool(is_bar_col[i]):
                    left += 1
                else:
                    break
            right = 0
            for i in range(max_h_cut):
                if bool(is_bar_col[w - 1 - i]):
                    right += 1
                else:
                    break
            min_px = 12
            if top < min_px and bottom < min_px and left < min_px and right < min_px:
                return None
            new_h = h - top - bottom
            new_w = w - left - right
            if new_h < int(h * 0.70) or new_w < int(w * 0.70):
                return None
            return (int(top), int(bottom), int(left), int(right))
        except Exception:
            return None

    def _apply_preview_crop_policy(self, frame):
        try:
            if frame is None:
                return frame
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return frame
            try:
                enable_blackbars = str(os.environ.get('MM_PRO_PREVIEW_CROP_BLACKBARS', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                enable_blackbars = True
            try:
                force_169 = str(os.environ.get('MM_PRO_PREVIEW_FORCE_16_9', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                force_169 = True
            if not hasattr(self, '_preview_crop_base'):
                self._preview_crop_base = None
                self._preview_crop_cuts = None
                self._preview_crop_check_counter = 0
            self._preview_crop_check_counter = int(getattr(self, '_preview_crop_check_counter', 0)) + 1
            base = getattr(self, '_preview_crop_base', None)
            need_redetect = (base is None) or (base != (int(w), int(h))) or (self._preview_crop_check_counter % 180 == 0)
            if need_redetect:
                cuts = self._detect_black_bars_cuts(frame) if enable_blackbars else None
                self._preview_crop_base = (int(w), int(h))
                self._preview_crop_cuts = cuts
            cuts = getattr(self, '_preview_crop_cuts', None)
            if cuts is not None:
                top, bottom, left, right = cuts
                y0 = max(0, int(top))
                y1 = int(h - bottom) if bottom > 0 else int(h)
                x0 = max(0, int(left))
                x1 = int(w - right) if right > 0 else int(w)
                if (y1 - y0) > 10 and (x1 - x0) > 10:
                    frame = frame[y0:y1, x0:x1]
            if force_169:
                frame = self._center_crop_to_ar(frame, 16.0 / 9.0)
            return frame
        except Exception:
            return frame

    def _recover_camera(self, reason: str = '', stage: int | None = None) -> bool:
        try:
            msg = f"♻️ Recover κάμερας (Benchmark)… ({reason})" if reason else "♻️ Recover κάμερας (Benchmark)…"
            self._cprint(msg, Colors.YELLOW, bold=True)
        except Exception:
            pass
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            time.sleep(0.25)
        except Exception:
            pass
        if not acquire_camera_lock('CameraWorker(Benchmark)'):
            owner = get_camera_lock_owner()
            msg = f'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή: {owner}' if owner else 'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή.'
            self._cprint(msg, Colors.RED, bold=True)
            self.error.emit(msg + '\nΣταμάτα το άλλο tab και ξαναδοκίμασε.')
            self.finished.emit()
            return False  # FIX: stop execution after emitting finished
        try:
            self.cap, _api = open_video_capture(self.camera_index)
            self._camera_api = int(_api or 0)
        except Exception as e:
            try:
                self._cprint(f"❌ Recover αποτυχία: {e}", Colors.RED, bold=True)
            except Exception:
                pass
            return False
        try:
            good = 0
            for _ in range(10):
                try:
                    ok = self.cap.grab()
                except Exception:
                    ok = False
                if not ok:
                    time.sleep(0.02)
                    continue
                try:
                    ret, fr = self.cap.retrieve()
                except Exception:
                    ret, fr = False, None
                if ret and fr is not None and getattr(fr, 'size', 0) != 0:
                    good += 1
                    if good >= 3:
                        break
                else:
                    good = 0
                time.sleep(0.01)
            return good >= 3
        except Exception:
            return False

    def run(self):
        try:
            ensure_cuda_ready_for_thread("CameraBenchmarkWorker")
        except Exception:
            pass
        try:
            configure_opencv_runtime()
            ensure_windows_com_initialized()
            ensure_windows_media_foundation_started()
        except Exception:
            pass
        bar = '=' * 70
        self._cprint(bar, Colors.HEADER, bold=True)
        self._cprint(f' 🎥 ΕΝΑΡΞΗ BENCHMARK ΚΑΜΕΡΑΣ: {self.base_name} ', Colors.CYAN, bold=True)
        self._cprint(bar, Colors.HEADER, bold=True)
        self._cprint(f'🧩 Ρυθμίσεις: διάρκεια={self.duration_sec} s | κάμερα index={self.camera_index}', Colors.LIGHT)
        self._cprint('🌳 Δομή λογικής ροής Benchmark Κάμερας', Colors.CYAN)
        self._cprint('└─ 🧠 Κύρια διαδικασία: Camera Benchmark (GUI)', Colors.CYAN)
        self._cprint('   ├─ 🎥 Άνοιγμα live ροής κάμερας', Colors.CYAN)
        self._cprint('   ├─ 🔍 Αναζήτηση διαθέσιμων backends (PyTorch / ONNX / TensorRT / NCNN)', Colors.CYAN)
        self._cprint('   ├─ 🧪 Εκτέλεση benchmark για κάθε backend (FPS & ms/εικόνα)', Colors.CYAN)
        self._cprint('   └─ 🏁 Τελική κατάταξη backends με βάση το FPS', Colors.CYAN)
        self._cprint('', Colors.LIGHT)
        self._cprint('🔎 Σάρωση εκπαιδευμένων μοντέλων για διαθέσιμα backends...', Colors.BLUE)
        backends = find_available_backends(self.models_dir, self.base_name)
        if not backends:
            msg = f"Δεν βρέθηκε κανένα backend για το '{self.base_name}' στον φάκελο: {self.models_dir}"
            self._cprint(msg, Colors.RED, bold=True)
            self.error.emit(msg)
            self.finished.emit()
            return  # FIX: stop execution after emitting finished
        self._cprint('Βρέθηκαν τα εξής backends για Benchmark Κάμερας:', Colors.BLUE, bold=True)
        for b, p in backends.items():
            self._cprint(f'  • {backend_pretty_name(b)} → {p.name}', Colors.BLUE)
        if not acquire_camera_lock('CameraWorker(Benchmark)'):
            owner = get_camera_lock_owner()
            msg = f'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή: {owner}' if owner else 'Η κάμερα χρησιμοποιείται ήδη από άλλο tab/worker μέσα στην εφαρμογή.'
            self._cprint(msg, Colors.RED, bold=True)
            self.error.emit(msg + '\nΣταμάτα το άλλο tab και ξαναδοκίμασε.')
            self.finished.emit()
            return  # FIX: stop execution after emitting finished
        try:
            self.cap, _api = open_video_capture(self.camera_index)
            self._camera_api = int(_api or 0)
        except Exception as e:
            self.error.emit(f'Δεν ήταν δυνατό το άνοιγμα της κάμερας: {e}')
            try:
                release_camera_lock()
            except Exception:
                pass
            self.finished.emit()
            return
        try:
            self._cprint(f"OpenCV: {getattr(cv2,'__version__','?')} | Camera API={getattr(self, '_camera_api', 0)}", Colors.CYAN)
        except Exception:
            pass
        try:
            fixed_on = str(os.environ.get('MM_PRO_CAM_FIXED_1080', '1')).strip().lower() in ('1','true','yes','on')
        except Exception:
            fixed_on = True
        if fixed_on:
            try:
                ok = force_capture_resolution(self.cap, 1920, 1080, verify=True)
            except Exception:
                ok = False
            if not ok:
                try:
                    apply_best_camera_resolution(self.cap, int(self.camera_index), int(getattr(self, '_camera_api', 0) or 0))
                except Exception:
                    pass
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                self._cprint(f'📐 Ανάλυση κάμερας: {w} x {h}', Colors.CYAN, bold=True)
        except Exception:
            pass
        if not self.cap.isOpened():
            msg = f'Δεν ήταν δυνατό το άνοιγμα της κάμερας (index: {self.camera_index}).'
            self._cprint(msg, Colors.RED, bold=True)
            self.error.emit(msg)
            try:
                release_camera_lock()
            except Exception:
                pass
            self.finished.emit()
            return
        self._cprint('Η κάμερα άνοιξε επιτυχώς.', Colors.GREEN, bold=True)
        self._is_running = True
        results: list[tuple[str, float, float]] = []
        try:
            for backend, path in backends.items():
                if not self._is_running:
                    break
                self._cprint('--------------------------------------------------------------------------------', Colors.CYAN)
                self._cprint(f'▶ Benchmark κάμερας για: {backend_pretty_name(backend)}', Colors.MAGENTA, bold=True)
                # ── CNN check: if the backend path is a CNN model, use CNNInferenceHelper ──
                _bench_is_cnn = _is_cnn_path(path)
                model = None
                self._cnn_helper = None
                if _bench_is_cnn:
                    try:
                        _dev_bench = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
                        self._cnn_helper = CNNInferenceHelper(Path(path), device=_dev_bench)
                        self._cnn_helper.load()
                        self.runtime_label = f'CNN/{self._cnn_helper.model_name_str}'
                        self._cprint(
                            f'🧠 CNN μοντέλο φορτώθηκε: {self._cnn_helper.model_name_str} | '
                            f'{self._cnn_helper.num_classes} κλάσεις',
                            Colors.GREEN, bold=True)
                    except Exception as e:
                        err = f'Σφάλμα φόρτωσης CNN {path}: {e}'
                        self._cprint(err, Colors.RED, bold=True)
                        continue
                else:
                    try:
                        model, self.runtime_label = load_yolo_for_backend(backend, path)
                    except Exception as e:
                        err = f'Σφάλμα φόρτωσης backend {backend}: {e}'
                        self._cprint(err, Colors.RED, bold=True)
                        if should_print_traceback(e):
                            self._cprint(traceback.format_exc(), Colors.RED)
                        continue
                times: list[float] = []
                frame_count = 0
                start_time = time.time()
                fps_loop = 0.0
                fps_window_start = time.perf_counter()
                fps_window_frames = 0
                infer_ema = None
                ema_alpha = 0.20
                bad_frames = 0
                recover_attempts = 0
                while self._is_running and time.time() - start_time < self.duration_sec:
                    last_read_exc = None
                    try:
                        ok = self.cap.grab()
                        if ok:
                            ret, frame = self.cap.retrieve()
                        else:
                            ret, frame = False, None
                    except cv2.error as e:
                        ret, frame = False, None
                        last_read_exc = e
                    except Exception as e:
                        ret, frame = False, None
                        last_read_exc = e
                    if (not ret) or (frame is None) or (getattr(frame, 'size', 0) == 0):
                        bad_frames += 1
                        try:
                            if last_read_exc is not None and (bad_frames == 1 or bad_frames % 30 == 0):
                                self._cprint(f"⚠️ OpenCV σφάλμα κατά την ανάγνωση frame: {last_read_exc}", Colors.YELLOW)
                        except Exception:
                            pass
                        if bad_frames in (15, 30, 60, 90) and recover_attempts < 5:
                            ok_rec = False
                            try:
                                ok_rec = self._recover_camera(reason=f'{bad_frames} συνεχόμενα άκυρα frames', stage=recover_attempts)
                            except Exception:
                                ok_rec = False
                            recover_attempts += 1
                            if ok_rec:
                                bad_frames = 0
                                continue
                        if bad_frames >= 120:
                            self._cprint('Πάρα πολλά συνεχόμενα άκυρα frames από την κάμερα. Διακοπή benchmark.', Colors.RED, bold=True)
                            break
                        try:
                            time.sleep(0.005)
                        except Exception:
                            pass
                        continue
                    bad_frames = 0
                    try:
                        if not isinstance(frame, np.ndarray):
                            frame = np.asarray(frame)
                        if getattr(frame, 'dtype', None) is not None and frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8, copy=False)
                        if hasattr(frame, 'flags') and (not frame.flags['C_CONTIGUOUS']):
                            frame = np.ascontiguousarray(frame)
                    except Exception:
                        try:
                            frame = np.ascontiguousarray(np.array(frame, dtype=np.uint8))
                        except Exception:
                            pass
                    try:
                        cuda_sync()
                        t0 = time.perf_counter()
                        # ── CNN inference ─────────────────────────────────
                        _cnn_h = getattr(self, '_cnn_helper', None)
                        if _cnn_h is not None:
                            _cnn_preds_bench = _cnn_h.predict_frame(frame, top_k=5)
                            annotated_frame  = _cnn_h._draw_predictions(frame, _cnn_preds_bench)
                            is_classification_model = True
                            results_yolo = None
                        else:
                            # ── YOLO inference ────────────────────────────
                            is_classification_model = yolo_is_classification(model)
                            results_yolo = yolo_predict_first(model, frame, imgsz=self.imgsz, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                            if backend == 'ncnn' and not is_classification_model:
                                annotated_frame = _ncnn_manual_annotate(frame, results_yolo, self.imgsz)
                            else:
                                if backend == 'ncnn':
                                    results_yolo = _mmpro_fix_ncnn_results(results_yolo, frame, self.imgsz)
                                annotated_frame = results_yolo.plot(font_size=0.48)
                        try:
                            if hasattr(annotated_frame, 'flags') and (not annotated_frame.flags.writeable):
                                annotated_frame = annotated_frame.copy()
                        except Exception:
                            annotated_frame = annotated_frame.copy()
                        cuda_sync()
                        t1 = time.perf_counter()
                        infer_time = t1 - t0
                        if infer_time > 0:
                            if infer_ema is None:
                                infer_ema = infer_time
                            else:
                                infer_ema = (1.0 - ema_alpha) * infer_ema + ema_alpha * infer_time
                    except Exception as e:
                        self._log_exc('Inference / Πρόβλεψη', e, extra={
                            'Μοντέλο': getattr(self, 'model_path', '?'),
                            'Backend': getattr(self, 'model_type', '?'),
                        })
                        if should_print_traceback(e):
                            self._cprint(traceback.format_exc(), Colors.RED)
                        break
                    times.append(infer_time)
                    frame_count += 1
                    infer_fps = (1.0 / infer_ema) if (infer_ema is not None and infer_ema > 0) else 0.0
                    fps_window_frames += 1
                    _elapsed = time.perf_counter() - fps_window_start
                    if _elapsed >= 0.50:
                        fps_loop = fps_window_frames / _elapsed if _elapsed > 0 else 0.0
                        fps_window_frames = 0
                        fps_window_start = time.perf_counter()
                    try:
                        fh, fw = frame.shape[:2]
                    except Exception:
                        fw, fh = 0, 0
                    cap_txt = '1080p'
                    base_text = f'FPS : {infer_fps:.0f} | LOOP : {fps_loop:.0f} | {self.runtime_label} | CAP : {cap_txt}'
                    overlay_text = base_text
                    classification_label = None
                    classification_conf = None
                    try:
                        # ── CNN: top-1 in FPS overlay bar ─────────────────
                        _cnn_h_bench = getattr(self, '_cnn_helper', None)
                        if _cnn_h_bench is not None:
                            _cp_bench = locals().get('_cnn_preds_bench', [])
                            if _cp_bench:
                                top_cls  = _cp_bench[0][0]
                                top_conf = _cp_bench[0][1]
                                overlay_text = (f'{base_text} | '
                                                f'\U0001f3f7 {top_cls}  {top_conf*100:.1f}%')
                        elif is_classification_model:
                            probs = getattr(results_yolo, 'probs', None)
                            top1_idx = None
                            top1_conf = None
                            if probs is not None:
                                if hasattr(probs, 'top1'):
                                    try:
                                        top1_idx = int(probs.top1)
                                    except Exception:
                                        top1_idx = None
                                if hasattr(probs, 'top1conf'):
                                    try:
                                        top1_conf = float(probs.top1conf)
                                    except Exception:
                                        top1_conf = None
                                if top1_idx is None and hasattr(probs, 'data'):
                                    try:
                                        import numpy as _np
                                        data = probs.data
                                        if hasattr(data, 'cpu'):
                                            data = data.cpu().numpy()
                                        top1_idx = int(_np.argmax(data))
                                    except Exception:
                                        top1_idx = None
                            if top1_idx is not None:
                                names = getattr(results_yolo, 'names', None)
                                class_name = f'Class_{top1_idx}'
                                if isinstance(names, (list, tuple)):
                                    if 0 <= top1_idx < len(names):
                                        class_name = str(names[top1_idx])
                                elif isinstance(names, dict):
                                    class_name = str(names.get(top1_idx, class_name))
                                classification_label = class_name
                                classification_conf = top1_conf
                    except Exception:
                        overlay_text = base_text
                    try:
                        if is_classification_model and classification_label is not None:
                            if classification_conf is not None and 0.0 <= classification_conf <= 1.0:
                                perc_val = classification_conf * 100.0
                                overlay_text = f"{overlay_text} | {classification_label} ({perc_val:.1f}%)"
                            else:
                                overlay_text = f"{overlay_text} | {classification_label}"
                    except Exception:
                        pass
                    try:
                        if not isinstance(annotated_frame, np.ndarray):
                            annotated_frame = np.asarray(annotated_frame)
                        if getattr(annotated_frame, "dtype", None) is not None and annotated_frame.dtype != np.uint8:
                            annotated_frame = annotated_frame.astype(np.uint8, copy=False)
                        if hasattr(annotated_frame, "flags") and (not annotated_frame.flags["C_CONTIGUOUS"]):
                            annotated_frame = np.ascontiguousarray(annotated_frame)
                    except Exception:
                        annotated_frame = np.ascontiguousarray(np.array(annotated_frame))
                    try:
                        annotated_frame = self._apply_preview_crop_policy(annotated_frame)
                    except Exception:
                        pass
                    try:
                        if annotated_frame is not None:
                            _h, _w = annotated_frame.shape[:2]
                            if _w != 1920 or _h != 1080:
                                                        tw, th = 1920, 1080
                                                        scale = min(tw / max(1, _w), th / max(1, _h))
                                                        nw = max(1, int(_w * scale))
                                                        nh = max(1, int(_h * scale))
                                                        resized = cv2.resize(annotated_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                                        canvas = np.zeros((th, tw, resized.shape[2]) if resized.ndim == 3 else (th, tw), dtype=resized.dtype)
                                                        x0 = int((tw - nw) / 2)
                                                        y0 = int((th - nh) / 2)
                                                        canvas[y0:y0+nh, x0:x0+nw] = resized
                                                        annotated_frame = canvas
                    except Exception:
                        pass
                    height, width = annotated_frame.shape[:2]
                    try:
                        if annotated_frame.ndim == 2:
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2RGB)
                        elif annotated_frame.shape[2] == 4:
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2RGB)
                        else:
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    except Exception:
                        if annotated_frame.ndim == 3 and annotated_frame.shape[2] >= 3:
                            annotated_frame_rgb = annotated_frame[:, :, :3][:, :, ::-1].copy()
                        else:
                            annotated_frame_rgb = np.stack([annotated_frame] * 3, axis=-1).astype(np.uint8, copy=False)
                    try:
                        if hasattr(annotated_frame_rgb, "flags") and (not annotated_frame_rgb.flags["C_CONTIGUOUS"]):
                            annotated_frame_rgb = np.ascontiguousarray(annotated_frame_rgb)
                    except Exception:
                        annotated_frame_rgb = np.ascontiguousarray(annotated_frame_rgb)
                    h2, w2 = annotated_frame_rgb.shape[:2]
                    ch = int(annotated_frame_rgb.shape[2]) if annotated_frame_rgb.ndim == 3 else 1
                    bytes_per_line = int(annotated_frame_rgb.strides[0])
                    q_img = QImage(annotated_frame_rgb.data, w2, h2, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    try:
                        with self._overlay_lock:
                            self._latest_overlay_text = str(overlay_text)
                    except Exception:
                        pass
                    self.frame_ready.emit(q_img)
                    try:
                        time.sleep(0.001)
                    except Exception:
                        pass
                if times and frame_count > 0:
                    avg_time = sum(times) / len(times)
                    fps_backend = 1.0 / avg_time if avg_time > 0 else 0.0
                    ms_per_image = avg_time * 1000.0
                    results.append((backend, fps_backend, ms_per_image))
                    self._cprint(f'Αποτέλεσμα κάμερας {backend_pretty_name(backend)}: {fps_backend:.2f} FPS, {ms_per_image:.2f} ms/εικόνα', Colors.GREEN, bold=True)
                try:
                    del model
                except Exception:
                    pass
            if results:
                results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
                bar = '═' * 72
                self._cprint(bar, Colors.CYAN, bold=True)
                self._cprint('📊 ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ BENCHMARK ΚΑΜΕΡΑΣ', Colors.GREEN, bold=True)
                self._cprint(bar, Colors.CYAN, bold=True)
                medals = ['🥇', '🥈', '🥉']
                for idx, (backend, fps, ms_per_image) in enumerate(results_sorted):
                    trophy = medals[idx] if idx < len(medals) else '🏅'
                    self._cprint(f'{trophy} {backend_pretty_name(backend):<20} → {fps:8.2f} FPS | {ms_per_image:6.2f} ms/εικόνα', Colors.LIGHT if hasattr(Colors, 'LIGHT') else Colors.CYAN)
                best_backend, best_fps, best_ms = results_sorted[0]
                self._cprint(f'🏆 Καλύτερο backend κάμερας: {backend_pretty_name(best_backend)} → {best_fps:.2f} FPS ({best_ms:.2f} ms/εικόνα)', Colors.GREEN, bold=True)
                self.results_ready.emit(results_sorted)
            else:
                self._cprint('Δεν προέκυψαν αποτελέσματα benchmark κάμερας.', Colors.YELLOW, bold=True)
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            try:
                release_camera_lock()
            except Exception:
                pass
            self._is_running = False
            self.finished.emit()
"""Training copilot workers.
Background εργασίες/βοηθοί για training suggestions, checks και UX αυτοματισμούς.
"""

# ════════════════════════════════════════════════════════════════════════════════
# VideoFileWorker – Worker thread για inference σε video αρχείο
# ════════════════════════════════════════════════════════════════════════════════
# Ανοίγει video με OpenCV, τρέχει inference frame-by-frame,
# εκπέμπει QImage για εμφάνιση και προαιρετικά αποθηκεύει annotated video.
# Υποστηρίζει YOLO (detection/classification) και CNN torchvision (classification).
class VideoFileWorker(QObject, LogEmitMixin, StoppableMixin):
    """Εκτελεί YOLO inference σε video αρχείο και εκπέμπει frames/πρόοδο."""
    log = Signal(str)
    frame_ready = Signal(QImage)
    progress = Signal(int)        # 0–100
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_info: tuple, video_path: str, imgsz: int,
                 classes_filter: list | None = None, use_tensorrt: bool = False,
                 conf_threshold: float = 0.25, save_output: bool = False):
        super().__init__()
        self.model_path, self.model_type = model_info
        self.video_path = str(video_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes_filter = classes_filter
        self.use_tensorrt = use_tensorrt
        self.save_output = save_output
        self._is_running = False
        self._latest_lock = threading.Lock()
        self._latest_qimg: QImage | None = None
        self._latest_overlay: str = ''
        self.model = None
        self.output_path: str | None = None
        self._cnn_helper: 'CNNInferenceHelper | None' = None  # CNN torchvision helper

    def _cprint(self, text: str, color: str | None = None, bold: bool = False):
        self.log.emit(format_html_log(text, color, bold))

    def stop(self):
        self._is_running = False

    def get_latest_qimage(self) -> QImage | None:
        with self._latest_lock:
            return self._latest_qimg

    def get_latest_overlay_text(self) -> str:
        with self._latest_lock:
            return self._latest_overlay

    def load_model(self) -> bool:
        try:
            self._cprint(f'Φόρτωση μοντέλου: {Path(self.model_path).name}', Colors.CYAN)
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f'Το μοντέλο δεν βρέθηκε: {self.model_path}')
            # ── CNN torchvision ────────────────────────────────────────────
            if _is_cnn_path(self.model_path):
                self._cprint('🧠 Εντοπίστηκε CNN μοντέλο (torchvision).', Colors.CYAN)
                dev_str = 'cuda:0' if (
                    str(getattr(self, 'device', 'cpu')).lower().startswith('cuda') and
                    __import__('torch').cuda.is_available()
                ) else 'cpu'
                self._cnn_helper = CNNInferenceHelper(Path(self.model_path), device=dev_str)
                self._cnn_helper.load()
                self.model = None
                self.runtime_label = f'CNN/{self._cnn_helper.model_name_str}'
                self._cprint(
                    f'✅ CNN μοντέλο φορτώθηκε: {self._cnn_helper.model_name_str} | '
                    f'{self._cnn_helper.num_classes} κλάσεις | imgsz={self._cnn_helper.imgsz}',
                    Colors.GREEN, bold=True)
                return True
            # ── YOLO ───────────────────────────────────────────────────────
            self._cnn_helper = None
            task = guess_ultralytics_task(Path(self.model_path))
            if self.model_type == 'PyTorch':
                self.model = YOLO(str(self.model_path))
            elif self.model_type == 'ONNX':
                engine_path = Path(self.model_path).with_suffix('.engine')
                if self.use_tensorrt and engine_path.exists():
                    self.model = YOLO(str(engine_path), task=task)
                    self._cprint('TensorRT engine φορτώθηκε.', Colors.MAGENTA, bold=True)
                else:
                    self.model = YOLO(str(self.model_path), task=task)
            elif self.model_type == 'NCNN':
                self.model = YOLO(str(self.model_path), task=task)
            elif self.model_type == 'TensorRT':
                self.model = YOLO(str(self.model_path), task=task)
                self._cprint(f'TensorRT engine φορτώθηκε: {Path(self.model_path).name}', Colors.MAGENTA, bold=True)
            else:
                raise ValueError(f'Άγνωστος τύπος μοντέλου: {self.model_type!r}. Αποδεκτές: PyTorch, CNN, ONNX, TensorRT, NCNN')
            self._cprint('Μοντέλο φορτώθηκε.', Colors.GREEN, bold=True)
            return True
        except Exception as e:
            self._log_exc('Φόρτωση μοντέλου (Video)', e, extra={
                'Αρχείο': str(getattr(self, 'video_path', '?')),
                'Τύπος':  str(getattr(self, 'model_type', '?')),
            })
            return False

    def run(self):
        if cv2 is None:
            self.error.emit('Λείπει το OpenCV (cv2). Εγκατάστησέ το με: pip install opencv-python')
            self.finished.emit()
            return
        try:
            ensure_cuda_ready_for_thread('VideoFileWorker')
        except Exception:
            pass
        if not self.load_model():
            self.finished.emit()
            return
        self._is_running = True
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f'Δεν ήταν δυνατό το άνοιγμα του video: {self.video_path}')
                self.finished.emit()
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            self._cprint(f'Video: {Path(self.video_path).name}', Colors.CYAN)
            self._cprint(f'Ανάλυση: {w_src}x{h_src} | FPS: {fps_src:.1f} | Frames: {total_frames}', Colors.CYAN)
            # Prepare output writer
            if self.save_output:
                out_path = Path(self.video_path)
                out_name = out_path.stem + '_annotated' + out_path.suffix
                out_full = out_path.parent / out_name
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(out_full), fourcc, fps_src, (w_src, h_src))
                    self.output_path = str(out_full)
                    self._cprint(f'💾 Αποθήκευση αποτελέσματος: {out_name}', Colors.GREEN)
                except Exception as e:
                    self._cprint(f'⚠️ Δεν ήταν δυνατή η δημιουργία output video: {e}', Colors.YELLOW)
                    writer = None
            frame_idx = 0
            t0 = time.time()
            recent_fps: list[float] = []
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                t_frame = time.time()
                # Inference
                annotated = frame
                overlay = ''
                try:
                    _cnn_h_vf = getattr(self, '_cnn_helper', None)
                    if _cnn_h_vf is not None:
                        # ── CNN torchvision ─────────────────────────────
                        _preds_vf = _cnn_h_vf.predict_frame(frame, top_k=5)
                        annotated = _cnn_h_vf._draw_predictions(frame, _preds_vf)
                        infer_ms = (time.time() - t_frame) * 1000
                        recent_fps.append(1.0 / max(0.001, time.time() - t_frame))
                        if len(recent_fps) > 15:
                            recent_fps.pop(0)
                        fps_live = sum(recent_fps) / max(1, len(recent_fps))
                        top_cls = _preds_vf[0][0] if _preds_vf else '—'
                        top_conf = _preds_vf[0][1] if _preds_vf else 0.0
                        overlay = f'{fps_live:.1f} FPS | {top_cls} {top_conf:.2f} | {infer_ms:.0f}ms'
                        with self._latest_lock:
                            self._latest_overlay = overlay
                    else:
                        # ── YOLO / TensorRT / ONNX / NCNN ───────────────
                        if self.model is None:
                            raise RuntimeError('Το μοντέλο δεν έχει φορτωθεί (model=None)')
                        kw: dict[str, Any] = dict(imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
                        if self.classes_filter:
                            kw['classes'] = self.classes_filter
                        results = self.model.predict(frame, **kw)
                        if results:
                            annotated = results[0].plot()
                            try:
                                boxes = getattr(results[0], 'boxes', None)
                                n = len(boxes) if boxes is not None else 0
                            except Exception:
                                n = 0
                            infer_ms = (time.time() - t_frame) * 1000
                            recent_fps.append(1.0 / max(0.001, time.time() - t_frame))
                            if len(recent_fps) > 15:
                                recent_fps.pop(0)
                            fps_live = sum(recent_fps) / max(1, len(recent_fps))
                            overlay = f'{fps_live:.1f} FPS | {n} obj | {infer_ms:.0f}ms'
                            with self._latest_lock:
                                self._latest_overlay = overlay
                except Exception as e:
                    self._log_exc('VideoFileWorker inference', e, extra={
                        'frame': getattr(frame, 'shape', '?'),
                        'model': 'CNN' if getattr(self, '_cnn_helper', None) else getattr(self, 'model_type', '?'),
                    })
                # Write output
                if writer is not None:
                    try:
                        writer.write(annotated if annotated is not frame else frame)
                    except Exception:
                        pass
                # Emit frame as QImage (RGB)
                try:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    h2, w2, ch = rgb.shape
                    qi = QImage(rgb.data, w2, h2, w2 * ch, QImage.Format.Format_RGB888).copy()
                    with self._latest_lock:
                        self._latest_qimg = qi
                    self.frame_ready.emit(qi)
                except Exception:
                    pass
                # Progress
                if total_frames > 0:
                    pct = min(99, int(frame_idx / total_frames * 100))
                    self.progress.emit(pct)
                # FPS throttle for UI (don't flood signal queue)
                elapsed = time.time() - t_frame
                sleep_for = max(0.0, (1.0 / 30.0) - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            elapsed_total = time.time() - t0
            self._cprint(
                f'✅ Ολοκλήρωση: {frame_idx} frames σε {elapsed_total:.1f}s '
                f'({frame_idx / max(0.001, elapsed_total):.1f} FPS μέσος όρος)',
                Colors.GREEN, bold=True,
            )
            if self.output_path:
                self._cprint(f'💾 Αποθηκεύτηκε: {self.output_path}', Colors.GREEN)
            self.progress.emit(100)
        except Exception as e:
            safe_log_error('VideoFileWorker run error', e)
            self.error.emit(f'Σφάλμα εκτέλεσης: {e}')
        finally:
            self._is_running = False
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
        self.finished.emit()

class LLMWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, system_prompt: str, user_message: str, model: str | None=None, parent=None):
        super().__init__(parent)
        self.system_prompt = system_prompt
        self.user_message = user_message
        self.model = model or get_current_llm_model()

    @Slot()

    def run(self):
        try:
            client = LLMClient()
            content = client.chat(system_prompt=self.system_prompt, user_message=self.user_message, model=self.model, temperature=0.2)
            self.finished.emit(content)
        except Exception as e:
            self.error.emit(str(e))


class LLMSettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('⚙️ Ρυθμίσεις LLM / Groq API')
        self.setModal(True)
        self.resize(720, 520)
        main_layout = QVBoxLayout(self)
        settings_group = QGroupBox('Βασικές ρυθμίσεις LLM / Groq')
        settings_layout = QFormLayout(settings_group)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText('https://api.groq.com/openai/v1')
        self.base_url_edit.setText(str(BASE_URL))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText('Επικόλλησε εδώ το GROQ_API_KEY σου')
        if API_KEY:
            self.api_key_edit.setText(API_KEY)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        suggested_models = [DEFAULT_MODEL, 'llama-3.3-70b-versatile', 'llama-3.3-8b-instant', 'openai/gpt-oss-20b', 'openai/gpt-oss-120b']
        seen = set()
        current_model = get_current_llm_model()
        for m in suggested_models:
            if m and m not in seen:
                self.model_combo.addItem(m)
                seen.add(m)
        if current_model and current_model not in seen:
            self.model_combo.addItem(current_model)
        if current_model:
            idx = self.model_combo.findText(current_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        settings_layout.addRow('Groq Base URL:', self.base_url_edit)
        settings_layout.addRow('GROQ_API_KEY:', self.api_key_edit)
        settings_layout.addRow('Προεπιλεγμένο LLM model:', self.model_combo)
        main_layout.addWidget(settings_group)
        instructions_group = QGroupBox('Οδηγίες για δημιουργία και την χρήση GROQ API Key')
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_label = QLabel('1️⃣ Πάτησε το κουμπί παρακάτω για να ανοίξει η σελίδα API Keys στο Groq.\n2️⃣ Συνδέσου (ή δημιούργησε λογαριασμό) και πάτησε «Create API Key».\n3️⃣ Αντέγραψε το νέο API key και επικόλλησέ το στο πεδίο «GROQ_API_KEY».\n4️⃣ Επίλεξε από τη λίστα το LLM model που θέλεις (π.χ. llama-3.3-70b-versatile ή Kimi K2).\n5️⃣ Πάτησε «Αποθήκευση» – από εδώ και πέρα το Copilot θα χρησιμοποιεί αυτές τις ρυθμίσεις.')
        instructions_label.setWordWrap(True)
        instructions_layout.addWidget(instructions_label)
        self.open_groq_button = QPushButton('🌐 Άνοιγμα σελίδας δημιουργίας Groq API Key')
        self.open_groq_button.setToolTip('Άνοιγμα του default browser στην σελίδα API Keys του Groq')
        self.open_groq_button.clicked.connect(self.open_groq_keys_page)
        instructions_layout.addWidget(self.open_groq_button)
        self.refresh_models_button = QPushButton('🔄 Φόρτωση διαθέσιμων LLM από Groq')
        self.refresh_models_button.setToolTip('Κάνει κλήση στο Groq API με το τρέχον API key και ενημερώνει τη λίστα με τα διαθέσιμα LLM models.')
        self.refresh_models_button.clicked.connect(self.refresh_models_from_api)
        instructions_layout.addWidget(self.refresh_models_button)
        extra_label = QLabel('💡 Προτείνεται, για λόγους ασφαλείας, να ΜΗΝ κοινοποιείς το API key σου σε \n τρίτους και να το αποθηκεύεις σε password manager. Μπορείς επίσης να χρησιμοποιήσεις μεταβλητές περιβάλλοντος (GROQ_API_KEY) αν προτιμάς.')
        extra_label.setWordWrap(True)
        instructions_layout.addWidget(extra_label)
        main_layout.addWidget(instructions_group)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        self.cancel_button = QPushButton('Άκυρο')
        self.save_button = QPushButton('Αποθήκευση')
        self.cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.save_button)
        main_layout.addLayout(buttons_layout)
        self.center_on_screen()

    def center_on_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geo = self.frameGeometry()
        center_point = screen.availableGeometry().center()
        geo.moveCenter(center_point)
        self.move(geo.topLeft())

    def open_groq_keys_page(self) -> None:
        try:
            QDesktopServices.openUrl(QUrl('https://console.groq.com/keys'))
        except Exception:
            pass

    def refresh_models_from_api(self) -> None:
        local_api_key = self.api_key_edit.text().strip() or API_KEY
        local_base_url = self.base_url_edit.text().strip() or BASE_URL
        if not local_api_key:
            QMessageBox.warning(self, 'Ρυθμίσεις LLM', 'Για να φορτώσεις τα διαθέσιμα LLM models από το Groq, συμπλήρωσε πρώτα ένα έγκυρο GROQ_API_KEY.')
            return
        try:
            client = OpenAI(base_url=local_base_url, api_key=local_api_key)
            models = client.models.list()
            model_names = []
            for m in getattr(models, 'data', []):
                mid = getattr(m, 'id', '') or getattr(m, 'name', '')
                if mid:
                    model_names.append(str(mid).strip())
            if not model_names:
                QMessageBox.warning(self, 'Ρυθμίσεις LLM', 'Δεν βρέθηκαν διαθέσιμα LLM models από το Groq API.\nΈλεγξε αν το API key και το base URL είναι σωστά.')
            self.model_combo.clear()
            unique_sorted = sorted(set(model_names))
            for name in unique_sorted:
                self.model_combo.addItem(name)
            current_model = get_current_llm_model()
            if current_model:
                idx = self.model_combo.findText(current_model)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            QMessageBox.information(self, 'Ρυθμίσεις LLM', f'Φορτώθηκαν {len(unique_sorted)} διαθέσιμα LLM models από το Groq API.')
        except Exception as e:
            QMessageBox.warning(self, 'Σφάλμα κατά τη φόρτωση LLM models', f'Προέκυψε σφάλμα κατά την κλήση του Groq API.\nΈλεγξε το API key και τη σύνδεσή σου στο διαδίκτυο.\n\nΛεπτομέρειες: {e}')

    def get_values(self) -> tuple[str, str, str]:
        base_url = self.base_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        model = self.model_combo.currentText().strip()
        return (base_url, api_key, model)
"""Statistics workers.
Heavy εργασίες (ανάγνωση labels, metrics, plots, reports) εκτός UI thread.
"""


# ════════════════════════════════════════════════════════════════════════════════
# StatisticsWorker – Worker για στατιστική ανάλυση dataset
# ════════════════════════════════════════════════════════════════════════════════
# Τρέχει inference σε όλες τις εικόνες ενός dataset και συλλέγει:
#   - Αριθμό ανιχνεύσεων ανά κλάση (class_counts)
#   - Confidence scores ανά κλάση (conf_scores)
#   - Χρόνους inference (detection_times)
# Παράγει PDF αναφορά (γραφήματα: detections/class, inference time, confidence).
# Υποστηρίζει YOLO (detection/classification) και CNN torchvision.
# Εκπέμπει: log(str), summary(str), progress(int), report_ready(str), preview_sample(...)
# ════════════════════════════════════════════════════════════════════════════════
class StatisticsWorker(QObject, LogEmitMixin):
    log = Signal(str)
    summary = Signal(str)
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)
    report_ready = Signal(str)
    preview_sample = Signal(str, str, str, str)

    def __init__(self, model_path: Path, model_type_str: str, dataset_path: Path, dataset_name: str, max_images: int | None=None) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.model_type = model_type_str
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.max_images = max_images
        self._stop_requested = False
        self.model = None
        self._cnn_helper: 'CNNInferenceHelper | None' = None  # CNN torchvision helper
        self.training_imgsz: int | None = None
        self.detection_imgsz: int = STATS_IMAGE_SIZE
        try:
            stem = getattr(self.model_path, 'stem', '') or self.model_path.name
            m = re.search(r'_(\d{2,4})(?:$|\.|_)', stem)
            if m:
                self.training_imgsz = int(m.group(1))
                self.detection_imgsz = self.training_imgsz
        except Exception:
            self.detection_imgsz = STATS_IMAGE_SIZE
        self.file_logger: logging.Logger | None = None
        self.log_file: Path | None = None

    def stop(self) -> None:
        self._stop_requested = True
        self._cprint('🛑 Λήφθηκε εντολή διακοπής Ανάλυσης.', Colors.RED, bold=True)

    def _setup_logging(self) -> None:
        try:
            self.log_file = DETECTION_REPORTS_DIR / f'stats_log_{self.model_path.stem}_{self.dataset_name}.log'
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            logger_name = f'StatsWorker_{self.model_path.stem}_{self.dataset_name}'
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            for h in logger.handlers:
                logger.removeHandler(h)
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
            self.file_logger = logger
            self.file_logger.info('StatisticsWorker logging initialized.')
        except Exception as e:
            try:
                self.log.emit(format_html_log(f'Σφάλμα logging: {e}', Colors.RED))
            except Exception:
                pass
            self.file_logger = None

    def _cprint(self, text: str, color: str | Colors=Colors.LIGHT, bold: bool=False) -> None:
        try:
            html = format_html_log(text, color, bold=bold)
            self.log.emit(html)
        except Exception:
            try:
                self.log.emit(text)
            except Exception:
                pass
        try:
            if self.file_logger:
                self.file_logger.info(text)
        except Exception:
            pass

    def load_model(self) -> bool:
        try:
            self._cprint(f'Φόρτωση μοντέλου από: {self.model_path}', Colors.CYAN)
            # ── CNN torchvision ────────────────────────────────────────────
            if _is_cnn_path(self.model_path):
                self._cprint('🧠 Εντοπίστηκε CNN μοντέλο (torchvision).', Colors.CYAN)
                self._cnn_helper = CNNInferenceHelper(Path(self.model_path), device='cpu')
                self._cnn_helper.load()
                self.model = None
                self._cprint(
                    f'✅ CNN μοντέλο φορτώθηκε: {self._cnn_helper.model_name_str} | '
                    f'{self._cnn_helper.num_classes} κλάσεις | imgsz={self._cnn_helper.imgsz}',
                    Colors.GREEN, bold=True)
                return True
            # ── YOLO ───────────────────────────────────────────────────────
            self._cnn_helper = None
            task = guess_ultralytics_task(self.model_path)
            if self.model_type == 'PyTorch':
                self.model = YOLO(str(self.model_path))
            elif self.model_type == 'NCNN':
                self.model = YOLO(str(self.model_path), task=task)
            elif self.model_type == 'ONNX':
                try:
                    self.model = YOLO(str(self.model_path), task=task)
                except TypeError:
                    self.model = YOLO(str(self.model_path))
            elif self.model_type == 'TensorRT':
                try:
                    self.model = YOLO(str(self.model_path), task=task)
                except TypeError:
                    self.model = YOLO(str(self.model_path))
            else:
                raise ValueError(f'Άγνωστος τύπος μοντέλου: {self.model_type}')
            self._cprint('✅ Μοντέλο φορτώθηκε επιτυχώς.', Colors.GREEN, bold=True)
            return True
        except Exception as e:
            self._log_exc('Φόρτωση μοντέλου (Stats)', e, extra={
                'Αρχείο': str(getattr(self, 'model_path', '?')),
                'Τύπος':  str(getattr(self, 'model_type', '?')),
            })
            return False

    def _ensure_preview_dir(self) -> Path:
        try:
            DETECTION_PREVIEW_DIR.mkdir(exist_ok=True)
        except Exception:
            pass
        return DETECTION_PREVIEW_DIR

    def _make_classification_preview(self, img_path: Path, class_name: str, conf: float) -> Path | None:
        try:
            import cv2
            import numpy as np
            preview_dir = self._ensure_preview_dir()
            preview_path = preview_dir / f'{img_path.stem}_cls_preview.jpg'
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            max_w, max_h = (1280, 720)
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            bar_height = max(40, int(img.shape[0] * 0.1))
            bar = np.full((bar_height, img.shape[1], 3), 20, dtype=img.dtype)
            combined = np.vstack([img, bar])
            text = f'Pred: {class_name} (conf={conf:.2f})'
            try:
                cv2.putText(combined, text, (10, img.shape[0] + int(bar_height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass
            cv2.imwrite(str(preview_path), combined)
            return preview_path
        except Exception:
            return None

    def _make_detection_preview(self, img_path: Path, results) -> Path | None:
        try:
            import cv2
            import numpy as np
            boxes = getattr(results, 'boxes', None)
            if boxes is None:
                return None
            box_data = getattr(boxes, 'data', None)
            if box_data is None:
                return None
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            max_w, max_h = (1280, 720)
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                sx = new_w / max(w, 1)
                sy = new_h / max(h, 1)
            else:
                sx = sy = 1.0
            names = getattr(results, 'names', None)
            data = box_data.cpu().numpy() if hasattr(box_data, 'cpu') else np.asarray(box_data)
            for row in data:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = row[:6]
                x1 = int(x1 * sx)
                y1 = int(y1 * sy)
                x2 = int(x2 * sx)
                y2 = int(y2 * sy)
                p1 = (max(0, x1), max(0, y1))
                p2 = (min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2))
                color = (0, 255, 0)
                try:
                    cv2.rectangle(img, p1, p2, color, 2)
                except Exception:
                    pass
                class_name = f'cls_{int(cls_id)}'
                if isinstance(names, (list, tuple)):
                    cls_idx = int(cls_id)
                    if 0 <= cls_idx < len(names):
                        class_name = str(names[cls_idx])
                elif isinstance(names, dict):
                    class_name = str(names.get(int(cls_id), class_name))
                label = f'{class_name} {conf:.2f}'
                try:
                    cv2.putText(img, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass
            preview_dir = self._ensure_preview_dir()
            preview_path = preview_dir / f'{img_path.stem}_det_preview.jpg'
            cv2.imwrite(str(preview_path), img)
            return preview_path
        except Exception:
            return None

    def _infer_class_from_path(self, img_path: Path) -> str | None:
        try:
            parent = img_path.parent
            if parent is None:
                return None
            generic = {'images', 'image', 'imgs', 'train', 'val', 'test', 'trainval'}
            if parent.name and parent.name.lower() not in generic:
                return parent.name
            grand = parent.parent
            if grand is not None and grand.name and (grand.name.lower() not in generic):
                return grand.name
        except Exception:
            pass

    def _find_label_file_for_image(self, img_path: Path) -> Path | None:
        try:
            stem = img_path.stem
            candidates: list[Path] = []
            try:
                rel = img_path.relative_to(self.dataset_path)
                parts = list(rel.parts)
                if parts:
                    if parts[0].lower() == 'images':
                        rel_labels = Path('labels', *parts[1:]).with_suffix('.txt')
                    else:
                        rel_labels = rel.with_suffix('.txt')
                    candidates.append(self.dataset_path / rel_labels)
            except Exception:
                pass
            try:
                current = img_path.parent
                for _ in range(4):
                    if current is None:
                        break
                    labels_dir = current / 'labels'
                    if labels_dir.is_dir():
                        candidates.append(labels_dir / f'{stem}.txt')
                    current = current.parent
            except Exception:
                pass
            for c in candidates:
                try:
                    if c.exists():
                        return c
                except Exception:
                    continue
        except Exception:
            pass

    def _get_detection_ground_truth_classes(self, img_path: Path, names) -> set[str | None]:
        label_path = self._find_label_file_for_image(img_path)
        if label_path is None:
            return None
        gt_ids: set[int] = set()
        try:
            with label_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except Exception:
                        continue
                    gt_ids.add(cls_id)
        except Exception:
            return None
        if not gt_ids:
            return set()
        gt_names: set[str] = set()
        if isinstance(names, dict):
            for cid in gt_ids:
                gt_names.add(str(names.get(cid, f'Class_{cid}')))
        elif isinstance(names, (list, tuple)):
            for cid in gt_ids:
                if 0 <= cid < len(names):
                    gt_names.add(str(names[cid]))
                else:
                    gt_names.add(f'Class_{cid}')
        else:
            for cid in gt_ids:
                gt_names.add(f'Class_{cid}')
        return gt_names

    def find_images(self, dir_path: Path, max_images: int=500) -> list[Path]:
        images: list[Path] = []
        try:
            for ext in STATS_IMG_EXTS:
                images.extend(dir_path.rglob(ext))
        except Exception as e:
            self._cprint(f'Σφάλμα ανάγνωσης εικόνων: {e}', Colors.RED)
        random.shuffle(images)
        return images[:max_images]

    # ── Κεντρικός βρόχος inference σε εικόνες dataset ────────────────────────────
    # Για CNN: χρησιμοποιεί self._cnn_helper.predict_frame() (top-k softmax).
    # Για YOLO: χρησιμοποιεί self.model.predict() (detection ή classification).
    # Εκπέμπει preview_sample για κάθε εικόνα (έως max_previews).
    # Επιστρέφει dict με: class_counts, conf_scores, detection_times, task.
    def analyze_images(self, images: list[Path]) -> dict:
        from collections import defaultdict
        import numpy as np
        class_counts: dict[str, int] = defaultdict(int)
        conf_scores: dict[str, list[float]] = defaultdict(list)
        detection_times: list[float] = []
        total_detections = 0
        num_images = len(images)

        # ── Determine inference mode ───────────────────────────────────────
        _cnn_h = getattr(self, '_cnn_helper', None)
        is_cnn_mode = _cnn_h is not None
        if is_cnn_mode:
            model_task = 'classify'
            is_classification_model = True
        else:
            model_task = getattr(self.model, 'task', None)
            is_classification_model = model_task == 'classify'

        try:
            user_max = int(self.max_images) if self.max_images is not None else num_images
        except Exception:
            user_max = num_images
        max_previews = max(0, min(user_max, 500))
        preview_count = 0

        for idx, img_path in enumerate(images):
            if self._stop_requested:
                self._cprint('⚠️ Η ανάλυση διακόπηκε από τον χρήστη.', Colors.YELLOW)
                break
            try:
                start_time = time.time()

                # ── CNN inference ─────────────────────────────────────────
                if is_cnn_mode:
                    try:
                        frame = cv2.imread(str(img_path))
                        if frame is None:
                            continue
                        preds = _cnn_h.predict_frame(frame, top_k=5)
                    except Exception as e:
                        self._cprint(f'Σφάλμα CNN inference {img_path.name}: {e}', Colors.RED)
                        continue
                    end_time = time.time()
                    detection_times.append(max(0.0, float(end_time - start_time)))

                    # Πάντα εκπέμπουμε preview — ανεξάρτητα από confidence threshold
                    gt_name = self._infer_class_from_path(img_path)
                    if preds:
                        # Top-1 only for statistics (matches classification behaviour)
                        # predict_frame επιστρέφει (name, conf, class_id) — κρατάμε name+conf
                        _p0 = preds[0]
                        class_name = _p0[0]
                        conf       = _p0[1]
                        # Στατιστικά μόνο για confidence >= threshold
                        if conf >= STATS_CONFIDENCE_THRESHOLD:
                            class_counts[class_name] += 1
                            conf_scores[class_name].append(conf)
                            total_detections += 1
                        # Preview για ΟΛΕΣ τις εικόνες
                        if preview_count < max_previews:
                            preview_path = self._make_classification_preview(
                                img_path, class_name, conf)
                            if preview_path is not None:
                                conf_tag = '' if conf >= STATS_CONFIDENCE_THRESHOLD                                            else f' ⚠️ χαμηλό confidence (threshold={STATS_CONFIDENCE_THRESHOLD:.2f})'
                                pred_text = f'Πρόβλεψη: {class_name} (conf={conf:.2f}){conf_tag}'
                                if gt_name:
                                    truth_text = (f'Πραγματική κλάση (από φάκελο): {gt_name}')
                                else:
                                    truth_text = ('Πραγματική κλάση: '
                                                  '(δεν μπορεί να εξαχθεί από τη διαδρομή)')
                                self.preview_sample.emit(
                                    str(preview_path), img_path.name,
                                    pred_text, truth_text)
                                preview_count += 1
                    else:
                        # Καμία πρόβλεψη — εμφάνισε την εικόνα με ένδειξη
                        if preview_count < max_previews:
                            preview_path = self._make_classification_preview(
                                img_path, '—', 0.0)
                            if preview_path is not None:
                                pred_text = 'Πρόβλεψη: — (καμία έξοδος μοντέλου)'
                                truth_text = (f'Πραγματική κλάση (από φάκελο): {gt_name}'
                                              if gt_name else
                                              'Πραγματική κλάση: (δεν εξάχθηκε)')
                                self.preview_sample.emit(
                                    str(preview_path), img_path.name,
                                    pred_text, truth_text)
                                preview_count += 1

                # ── YOLO inference ────────────────────────────────────────
                else:
                    results_list = self.model.predict(
                        str(img_path),
                        imgsz=self.detection_imgsz,
                        conf=STATS_CONFIDENCE_THRESHOLD,
                        iou=STATS_IOU_THRESHOLD,
                        verbose=False)
                    end_time = time.time()
                    detection_times.append(max(0.0, float(end_time - start_time)))
                    if not results_list:
                        continue
                    res = results_list[0]
                    names = getattr(res, 'names', {})
                    if is_classification_model:
                        probs = getattr(res, 'probs', None)
                        if probs is not None and hasattr(probs, 'top1'):
                            try:
                                cls_id = int(probs.top1)
                                conf = float(probs.top1conf)
                            except Exception:
                                cls_id = None
                                conf = 0.0
                        else:
                            cls_id = None
                            conf = 0.0
                        gt_name = self._infer_class_from_path(img_path)
                        if cls_id is not None:
                            if isinstance(names, dict):
                                class_name = str(names.get(cls_id, f'Class_{cls_id}'))
                            elif isinstance(names, (list, tuple)):
                                class_name = str(names[cls_id]) if 0 <= cls_id < len(names) \
                                             else f'Class_{cls_id}'
                            else:
                                class_name = f'Class_{cls_id}'
                            class_counts[class_name] += 1
                            conf_scores[class_name].append(conf)
                            total_detections += 1
                            # Preview για ΟΛΕΣ τις εικόνες
                            if preview_count < max_previews:
                                preview_path = self._make_classification_preview(
                                    img_path, class_name, conf)
                                if preview_path is not None:
                                    pred_text = f'Πρόβλεψη: {class_name} (conf={conf:.2f})'
                                    truth_text = (f'Πραγματική κλάση (από φάκελο): {gt_name}'
                                                  if gt_name else
                                                  'Πραγματική κλάση: (δεν μπορεί να εξαχθεί)')
                                    self.preview_sample.emit(
                                        str(preview_path), img_path.name,
                                        pred_text, truth_text)
                                    preview_count += 1
                        else:
                            # cls_id=None: η εικόνα επεξεργάστηκε αλλά δεν βγήκε πρόβλεψη
                            if preview_count < max_previews:
                                preview_path = self._make_classification_preview(
                                    img_path, '—', 0.0)
                                if preview_path is not None:
                                    pred_text = 'Πρόβλεψη: — (καμία έξοδος μοντέλου)'
                                    truth_text = (f'Πραγματική κλάση (από φάκελο): {gt_name}'
                                                  if gt_name else
                                                  'Πραγματική κλάση: (δεν εξάχθηκε)')
                                    self.preview_sample.emit(
                                        str(preview_path), img_path.name,
                                        pred_text, truth_text)
                                    preview_count += 1
                    else:
                        boxes = getattr(res, 'boxes', None)
                        _emitted_det_preview = False
                        if boxes is not None:
                            box_data = getattr(boxes, 'data', None)
                            if box_data is not None:
                                try:
                                    data = box_data.cpu().numpy() \
                                           if hasattr(box_data, 'cpu') else np.asarray(box_data)
                                except Exception:
                                    data = box_data
                                if data is not None:
                                    per_image_dets = 0
                                    for row in data:
                                        if len(row) < 6:
                                            continue
                                        cls_id = int(row[5])
                                        conf = float(row[4])
                                        if isinstance(names, dict):
                                            class_name = str(names.get(cls_id,
                                                             f'Class_{cls_id}'))
                                        elif isinstance(names, (list, tuple)):
                                            class_name = str(names[cls_id]) \
                                                if 0 <= cls_id < len(names) \
                                                else f'Class_{cls_id}'
                                        else:
                                            class_name = f'Class_{cls_id}'
                                        class_counts[class_name] += 1
                                        conf_scores[class_name].append(conf)
                                        per_image_dets += 1
                                    total_detections += per_image_dets
                                    # Preview για ΟΛΕΣ τις εικόνες (ακόμα και με 0 ανιχνεύσεις)
                                    if preview_count < max_previews:
                                        preview_path = self._make_detection_preview(
                                            img_path, res)
                                        if preview_path is not None:
                                            gt_names = \
                                                self._get_detection_ground_truth_classes(
                                                    img_path, names)
                                            pred_text = (f'Ανιχνεύσεις: {per_image_dets}'
                                                         if per_image_dets > 0
                                                         else 'Ανιχνεύσεις: 0 (καμία πάνω από threshold)')
                                            if gt_names is None:
                                                truth_text = ('Πραγματικές ετικέτες: '
                                                              '(δεν βρέθηκαν αρχεία labels)')
                                            elif not gt_names:
                                                truth_text = ('Πραγματικές ετικέτες: '
                                                              '(αρχείο labels κενό)')
                                            else:
                                                truth_text = ('Πραγματικές ετικέτες: '
                                                              + ', '.join(sorted(gt_names)))
                                            self.preview_sample.emit(
                                                str(preview_path), img_path.name,
                                                pred_text, truth_text)
                                            preview_count += 1
                                            _emitted_det_preview = True
                        # Εικόνα χωρίς boxes ή χωρίς δεδομένα — εμφάνισε την εικόνα χωρίς annotation
                        if not _emitted_det_preview and preview_count < max_previews:
                            preview_path = self._make_detection_preview(img_path, res)
                            if preview_path is not None:
                                gt_names = self._get_detection_ground_truth_classes(
                                    img_path, names)
                                pred_text = 'Ανιχνεύσεις: 0 (καμία πάνω από threshold)'
                                truth_text = ('Πραγματικές ετικέτες: ' + ', '.join(sorted(gt_names))
                                              if gt_names else
                                              'Πραγματικές ετικέτες: (δεν βρέθηκαν αρχεία labels)')
                                self.preview_sample.emit(
                                    str(preview_path), img_path.name,
                                    pred_text, truth_text)
                                preview_count += 1

                progress = int((idx + 1) / max(num_images, 1) * 100)
                self.progress.emit(progress)
            except Exception as e:
                self._cprint(f'Σφάλμα ανίχνευσης {img_path.name}: {e}', Colors.RED)

        stats: dict[str, object] = {
            'class_counts':    dict(class_counts),
            'conf_scores':     {k: list(v) for k, v in conf_scores.items()},
            'detection_times': detection_times,
            'total_images':    num_images,
            'total_detections':total_detections,
            'task':            model_task or 'unknown',
        }
        return stats

    def generate_summary(self, stats: dict) -> str:
        total_images = int(stats.get('total_images', 0) or 0)
        total_detections = int(stats.get('total_detections', 0) or 0)
        class_counts: dict[str, int] = stats.get('class_counts', {}) or {}
        conf_scores: dict[str, list[float]] = stats.get('conf_scores', {}) or {}
        detection_times = stats.get('detection_times', []) or []
        safe_stats = calculate_safe_statistics(detection_times, conf_scores)
        mean_time = float(safe_stats.get('mean_time', 0.0) or 0.0)
        std_time  = float(safe_stats.get('std_time', 0.0) or 0.0)
        min_time  = float(safe_stats.get('min_time', 0.0) or 0.0)
        max_time  = float(safe_stats.get('max_time', 0.0) or 0.0)
        avg_conf  = float(safe_stats.get('avg_conf', 0.0) or 0.0)
        try:
            det_times_arr = np.asarray(detection_times, dtype=float)
        except Exception:
            det_times_arr = np.array([], dtype=float)
        all_scores_flat: list[float] = []
        try:
            for _v in conf_scores.values():
                all_scores_flat.extend(_v or [])
        except Exception:
            all_scores_flat = []
        try:
            all_scores_arr = np.asarray(all_scores_flat, dtype=float)
        except Exception:
            all_scores_arr = np.array([], dtype=float)
        lines: list[str] = []
        sep = '═' * 70
        sub_sep = '─' * 70
        lines.append(sep)
        lines.append('🧪 ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΑΝΙΧΝΕΥΣΗΣ')
        lines.append(sep)
        lines.append('')
        model_path = getattr(self, 'model_path', None)
        if model_path is None:
            model_name = '(άγνωστο)'
        else:
            try:
                model_name = model_path.name
            except Exception:
                model_name = str(model_path)
        model_type = getattr(self, 'model_type', 'PyTorch')
        task = stats.get('task', 'unknown') or 'unknown'
        task_lower = str(task).lower()
        # Determine pretty task label
        _cnn_h = getattr(self, '_cnn_helper', None)
        if _cnn_h is not None:
            task_pretty = f'Ταξινόμηση εικόνων CNN ({_cnn_h.model_name_str})'
            model_type  = f'CNN ({_cnn_h.model_name_str})'
        elif 'class' in task_lower:
            task_pretty = 'Ταξινόμηση εικόνων (classification)'
        elif 'segment' in task_lower:
            task_pretty = 'Εντοπισμός με segmentation'
        else:
            task_pretty = 'Ανίχνευση αντικειμένων (detection)'
        training_imgsz = getattr(self, 'training_imgsz', None)
        detection_imgsz = getattr(self, 'detection_imgsz', training_imgsz or STATS_IMAGE_SIZE)
        lines.append('📦 Μοντέλο')
        lines.append(f'  • Όνομα: {model_name}')
        lines.append(f'  • Τύπος: {model_type}')
        lines.append(f'  • Εργασία: {task_pretty}')
        if training_imgsz:
            lines.append(f'  • Training image size: {training_imgsz}')
        lines.append(f'  • Detection image size: {detection_imgsz}')
        lines.append('')
        dataset_name = getattr(self, 'dataset_name', '(άγνωστο)')
        lines.append('🗂 Dataset')
        lines.append(f'  • Όνομα: {dataset_name}')
        try:
            dataset_path = getattr(self, 'dataset_path', None)
            if dataset_path is not None:
                lines.append(f'  • Διαδρομή: {dataset_path}')
        except Exception:
            pass
        max_images = getattr(self, 'max_images', None)
        if isinstance(max_images, int) and max_images > 0:
            lines.append(f'  • Εικόνες που ζητήθηκαν για ανάλυση: {max_images}')
        lines.append(f'  • Εικόνες που αναλύθηκαν τελικά: {total_images}')
        lines.append('')
        lines.append('📊 Στατιστικά ανίχνευσης')
        lines.append(f'  • Συνολικές ανιχνεύσεις: {total_detections}')
        avg_det_per_img = total_detections / total_images if total_images > 0 else 0.0
        lines.append(f'  • Μέσος αριθμός ανιχνεύσεων/εικόνα: {avg_det_per_img:.2f}')
        if not class_counts:
            lines.append('  • Δεν βρέθηκαν ανιχνεύσεις στο dataset.')
        else:
            lines.append(f'  • Σύνολο κλάσεων με ανιχνεύσεις: {len(class_counts)}')
        lines.append('')
        if class_counts:
            lines.append('🏷 Κλάσεις με τις περισσότερες ανιχνεύσεις')
            try:
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            except Exception:
                sorted_classes = list(class_counts.items())
            max_classes_to_show = 10
            for cls_name, count in sorted_classes[:max_classes_to_show]:
                confs = conf_scores.get(cls_name, [])
                try:
                    conf_arr = np.asarray(confs, dtype=float)
                except Exception:
                    conf_arr = np.array([], dtype=float)
                if conf_arr.size > 0:
                    mean_conf = float(np.mean(conf_arr))
                    lines.append(f'  • {cls_name}: {count} ανιχνεύσεις (mean conf={mean_conf:.3f})')
                else:
                    lines.append(f'  • {cls_name}: {count} ανιχνεύσεις')
            if len(sorted_classes) > max_classes_to_show:
                lines.append(f'  • ...και άλλες {len(sorted_classes) - max_classes_to_show} κλάσεις.')
            lines.append('')
        lines.append('⏱ Ταχύτητα ανίχνευσης')
        if det_times_arr.size > 0:
            lines.append(f'  • Μέσος χρόνος/εικόνα: {mean_time:.4f} sec (std={std_time:.4f})')
            lines.append(f'  • Ελάχιστος χρόνος: {min_time:.4f} sec')
            lines.append(f'  • Μέγιστος χρόνος: {max_time:.4f} sec')
            fps = 1.0 / mean_time if mean_time > 0 else 0.0
            lines.append(f'  • Εκτιμώμενα FPS: {fps:.2f}')
            if fps >= 30:
                rating = 'ULTRA FAST (>= 30 FPS)'
                desc = 'Ιδανικό για real-time εφαρμογές.'
            elif fps >= 15:
                rating = 'FAST (15–30 FPS)'
                desc = 'Πολύ καλή ταχύτητα – κατάλληλο για τις περισσότερες εφαρμογές.'
            elif fps >= 5:
                rating = 'MEDIUM (5–15 FPS)'
                desc = 'Μέτρια ταχύτητα – ίσως χρειάζεται βελτιστοποίηση για real-time.'
            else:
                rating = 'SLOW (< 5 FPS)'
                desc = 'Χαμηλή ταχύτητα – προτείνεται βελτιστοποίηση μοντέλου ή hardware.'
            lines.append(f'  • Κατηγορία ταχύτητας: {rating}')
            lines.append(f'  • Σχόλιο: {desc}')
        else:
            lines.append('  • Δεν ήταν δυνατός ο υπολογισμός χρόνων ανίχνευσης.')
        lines.append('')
        if all_scores_arr.size > 0:
            mean_conf_all = float(np.mean(all_scores_arr))
            std_conf_all = float(np.std(all_scores_arr))
            lines.append('🎯 Συνολική κατανομή confidence')
            lines.append(f'  • Μέσο confidence (όλων των ανιχνεύσεων): {mean_conf_all:.3f}')
            lines.append(f'  • Τυπική απόκλιση: {std_conf_all:.3f}')
            try:
                low = float(np.percentile(all_scores_arr, 10))
                mid = float(np.percentile(all_scores_arr, 50))
                hi = float(np.percentile(all_scores_arr, 90))
                lines.append(f'  • Percentiles (10% / 50% / 90%): {low:.3f} / {mid:.3f} / {hi:.3f}')
            except Exception:
                pass
            lines.append('')
        lines.append(sub_sep)
        lines.append('💡 Ερμηνεία')
        if total_detections == 0:
            lines.append('  • Το μοντέλο δεν βρήκε καμία ανίχνευση στο dataset.')
            lines.append('  • Ελέγξτε αν το dataset είναι συμβατό με τις κλάσεις του μοντέλου.')
        else:
            lines.append('  • Δείτε τις κορυφαίες κλάσεις για να εντοπίσετε πιθανές μεροληψίες.')
            lines.append('  • Εξετάστε την ταχύτητα (FPS) σε σχέση με τις απαιτήσεις της εφαρμογής.')
            lines.append('  • Συγκρίνετε την κατανομή confidence με τα αποτελέσματα της εκπαίδευσης.')
        lines.append(sub_sep)
        return '\n'.join(lines)

    def plot_stats(self, stats: dict, output_dir: Path) -> Path | None:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception as e:
            self._cprint(f'Δεν ήταν δυνατή η δημιουργία PDF αναφοράς (λείπει matplotlib/numpy): {e}', Colors.YELLOW)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = output_dir / f'DetectionReport_{self.model_path.stem}_{self.dataset_name}.pdf'
            total_images = int(stats.get('total_images', 0) or 0)
            total_detections = int(stats.get('total_detections', 0) or 0)
            class_counts: dict[str, int] = stats.get('class_counts', {}) or {}
            conf_scores: dict[str, list[float]] = stats.get('conf_scores', {}) or {}
            detection_times = stats.get('detection_times', []) or []
            task = str(stats.get('task') or getattr(self, 'model_task', 'unknown'))
            try:
                det_times_arr = np.asarray(detection_times, dtype=float)
            except Exception:
                det_times_arr = np.array([], dtype=float)
            if det_times_arr.size > 0:
                mean_time = float(np.mean(det_times_arr))
                std_time = float(np.std(det_times_arr))
                min_time = float(np.min(det_times_arr))
                max_time = float(np.max(det_times_arr))
            else:
                mean_time = std_time = min_time = max_time = 0.0
            all_scores: list[float] = []
            for v in conf_scores.values():
                all_scores.extend(v)
            try:
                all_scores_arr = np.asarray(all_scores, dtype=float)
            except Exception:
                all_scores_arr = np.array([], dtype=float)
            model_path = getattr(self, 'model_path', None)
            model_name = model_path.name if model_path is not None else '(unknown)'
            dataset_name = getattr(self, 'dataset_name', '(unknown)')
            dataset_path = getattr(self, 'dataset_path', None)
            training_imgsz = getattr(self, 'training_imgsz', None)
            detection_imgsz = getattr(self, 'detection_imgsz', None)
            model_type = getattr(self, 'model_type', None) or getattr(self, 'model_type_str', '')
            model_type = str(model_type or 'Unknown')
            model_task = getattr(self, 'model', None)
            if model_task is not None and hasattr(model_task, 'task'):
                task_pretty = str(getattr(model_task, 'task', task))
            else:
                task_pretty = task
            avg_det_per_img = total_detections / total_images if total_images > 0 else 0.0
            fps = 1.0 / mean_time if mean_time > 0 else 0.0
            if fps >= 30:
                speed_rating = 'ULTRA FAST (>= 30 FPS)'
                speed_desc = 'Ιδανικό για real-time εφαρμογές.'
            elif fps >= 15:
                speed_rating = 'FAST (15–30 FPS)'
                speed_desc = 'Πολύ καλή ταχύτητα – κατάλληλο για τις περισσότερες εφαρμογές.'
            elif fps >= 5:
                speed_rating = 'MEDIUM (5–15 FPS)'
                speed_desc = 'Μέτρια ταχύτητα – ίσως χρειάζεται βελτιστοποίηση για real-time.'
            elif fps > 0:
                speed_rating = 'SLOW (< 5 FPS)'
                speed_desc = 'Χαμηλή ταχύτητα – προτείνεται βελτιστοποίηση μοντέλου ή hardware.'
            else:
                speed_rating = 'N/A'
                speed_desc = 'Δεν υπάρχουν επαρκή δεδομένα χρόνων ανίχνευσης.'

            def fmt(v, digits: int=4) -> str:
                try:
                    if v is None:
                        return '–'
                    return f'{float(v):.{digits}f}'
                except Exception:
                    return str(v)
            plt.style.use('default')
            plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 9, 'figure.titlesize': 13, 'axes.grid': True, 'grid.alpha': 0.25})
            try:
                pass
            except Exception as e:
                self._cprint(f'Δεν ήταν δυνατή η δημιουργία πλούσιου PDF (λείπει reportlab/pdf_reports): {e}', Colors.YELLOW)
            safe_model = str(getattr(self.model_path, 'stem', 'model'))
            safe_dataset = str(dataset_name).replace('/', '_')
            assets_dir = output_dir / f'_assets_{safe_model}_{safe_dataset}'
            try:
                assets_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            def _save_fig(fig, name: str) -> Path:
                out = assets_dir / name
                try:
                    fig.savefig(out, dpi=200, bbox_inches='tight')
                finally:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                return out
            charts: list[tuple[str, Path]] = []
            # ── Ανίχνευση αν είναι CNN μοντέλο για εξειδικευμένα charts ──────
            _is_cnn_report = getattr(self, '_cnn_helper', None) is not None
            _cnn_all_classes: list[str] = []
            if _is_cnn_report:
                try:
                    _cnn_all_classes = list(getattr(self._cnn_helper, 'class_names', []))
                except Exception:
                    _cnn_all_classes = []

            if class_counts or (_is_cnn_report and _cnn_all_classes):
                # Για CNN: εμφάνισε ΟΛΕΣ τις κλάσεις (ακόμα και με 0 ανιχνεύσεις)
                if _is_cnn_report and _cnn_all_classes:
                    all_cls_names = _cnn_all_classes
                    all_cls_values = [class_counts.get(c, 0) for c in all_cls_names]
                else:
                    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                    all_cls_names  = [k for k, _ in sorted_items]
                    all_cls_values = [v for _, v in sorted_items]
                fig_w = max(9.0, len(all_cls_names) * 0.7)
                fig2, ax2 = plt.subplots(figsize=(fig_w, 6.2))
                chart_title = ('CNN Predictions per Class (all classes)'
                               if _is_cnn_report else 'Detections per Class')
                ax2.set_title(chart_title)
                ax2.bar(range(len(all_cls_names)), all_cls_values,
                        color=['#7c3aed' if v > 0 else '#e2e8f0' for v in all_cls_values])
                ax2.set_xticks(range(len(all_cls_names)))
                ax2.set_xticklabels(
                    all_cls_names,
                    rotation=45, ha='right',
                    fontsize=max(6, 9 - len(all_cls_names) // 6))
                ax2.set_ylabel('Ανιχνεύσεις (predictions above threshold)')
                ax2.grid(True, axis='y', linestyle='--', alpha=0.25)
                charts.append((chart_title, _save_fig(fig2, 'detections_per_class.png')))
            if det_times_arr.size > 0:
                fig3, ax3 = plt.subplots(figsize=(11.0, 6.2))
                ax3.set_title('Per-Image Inference Time')
                ax3.hist(det_times_arr, bins=30)
                ax3.set_xlabel('Time per image (sec)')
                ax3.set_ylabel('Images')
                if mean_time > 0:
                    ax3.axvline(mean_time, linestyle='--', linewidth=1.2)
                charts.append(('Inference time', _save_fig(fig3, 'inference_time.png')))
            if all_scores_arr.size > 0:
                fig4, ax4 = plt.subplots(figsize=(11.0, 6.2))
                ax4.set_title('Overall Confidence Distribution')
                ax4.hist(all_scores_arr, bins=25, range=(0.0, 1.0))
                ax4.set_xlabel('Confidence')
                ax4.set_ylabel('Detections')
                charts.append(('Confidence distribution', _save_fig(fig4, 'confidence_hist.png')))
            run_info_rows_out: list[tuple[str, str]] = [('Model', model_name), ('Model type', model_type), ('Task', task_pretty), ('Dataset', dataset_name)]
            if dataset_path is not None:
                run_info_rows_out.append(('Dataset path', str(dataset_path)))
            if training_imgsz:
                run_info_rows_out.append(('Training image size', f'{training_imgsz}px'))
            if detection_imgsz:
                run_info_rows_out.append(('Detection image size', f'{detection_imgsz}px'))
            metrics_rows_out: list[tuple[str, str]] = [('Images analysed', str(total_images)), ('Total detections', str(total_detections)), ('Avg detections / image', fmt(avg_det_per_img, 2))]
            if det_times_arr.size > 0:
                metrics_rows_out.extend([('Mean time / image (sec)', fmt(mean_time, 4)), ('Std time / image (sec)', fmt(std_time, 4)), ('Min time / image (sec)', fmt(min_time, 4)), ('Max time / image (sec)', fmt(max_time, 4)), ('Estimated FPS', fmt(fps, 2)), ('Speed rating', speed_rating)])
            # ── top_classes_table: CNN → ΟΛΕΣ κλάσεις, YOLO → top-10 ──────────
            top_classes_table: list[list[str]] = [['Class', 'Detections', 'Share', 'Mean conf']]
            if _is_cnn_report and _cnn_all_classes:
                # Εμφάνιση ΟΛΩΝ των κλάσεων του CNN μοντέλου
                for cls_name in _cnn_all_classes:
                    count = class_counts.get(cls_name, 0)
                    share = count / max(total_detections, 1) * 100.0 if total_detections > 0 else 0.0
                    scores = conf_scores.get(cls_name, [])
                    mean_conf = np.mean(scores) if scores else None
                    top_classes_table.append([
                        str(cls_name), str(count),
                        f'{share:.1f}%', fmt(mean_conf, 3)])
            elif class_counts:
                # YOLO: top-10 κλάσεις κατά αριθμό ανιχνεύσεων
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                top_k = sorted_classes[:min(len(sorted_classes), 10)]
                for name, count in top_k:
                    share = count / max(total_detections, 1) * 100.0
                    scores = conf_scores.get(name, [])
                    mean_conf = np.mean(scores) if scores else None
                    top_classes_table.append([str(name), str(count), f'{share:.1f}%', fmt(mean_conf, 3)])
            run_id = f'{safe_model}_{dataset_name}'
            try:
                # Determine model_type for PDF cover
                _cnn_h_pdf = getattr(self, '_cnn_helper', None)
                if _cnn_h_pdf is not None:
                    _pdf_model_type = 'cnn'
                elif 'class' in str(task).lower():
                    _pdf_model_type = 'yolo_classify'
                else:
                    _pdf_model_type = 'yolo_detect'
                build_detection_report_pdf(output_pdf=pdf_path, resource_root=CODE_DIR, run_id=str(run_id), run_info_rows=run_info_rows_out, metrics_rows=metrics_rows_out, top_classes=top_classes_table if len(top_classes_table) > 1 else [], charts=charts, notes=[speed_desc], model_type=_pdf_model_type)
            except Exception as e:
                self._cprint(f'Σφάλμα δημιουργίας πλούσιου PDF αναφοράς: {e}', Colors.RED)
            try:
                from pathlib import Path as _Path
                import json as _json
                from datetime import datetime as _dt
                import shutil as _shutil
                model_dir = _Path(self.model_path).resolve().parent
                try:
                    model_dir.relative_to(_Path(TRAINED_MODELS_DIR).resolve())
                    in_trained_models = True
                except Exception:
                    in_trained_models = False
                if in_trained_models:
                    metrics_dir = model_dir / 'metrics'
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    charts_dir2 = metrics_dir / 'charts_detection'
                    charts_dir2.mkdir(parents=True, exist_ok=True)

                    def _copy_img(src: _Path) -> str | None:
                        try:
                            if not src or not _Path(src).is_file():
                                return None
                            dst = charts_dir2 / _Path(src).name
                            if dst.exists():
                                stem = dst.stem
                                suf = dst.suffix
                                i = 2
                                while (charts_dir2 / f'{stem}_{i}{suf}').exists():
                                    i += 1
                                dst = charts_dir2 / f'{stem}_{i}{suf}'
                            _shutil.copy2(src, dst)
                            return dst.name
                        except Exception:
                            return None
                    charts_out = []
                    for t, p in list(charts):
                        name = _copy_img(_Path(p))
                        if name:
                            charts_out.append({'title': str(t), 'file': name})
                    top_classes_out = []
                    if top_classes_table and len(top_classes_table) > 1:
                        for row in top_classes_table[1:]:
                            if len(row) >= 4:
                                top_classes_out.append({'class': str(row[0]), 'detections': str(row[1]), 'share': str(row[2]), 'mean_conf': str(row[3])})
                    payload = {'kind': 'detection', 'generated_at': _dt.now().isoformat(timespec='seconds'), 'run_id': str(run_id), 'model': str(model_name), 'dataset': str(dataset_name), 'images_analysed': int(total_images), 'total_detections': int(total_detections), 'avg_detections_per_image': float(avg_det_per_img) if avg_det_per_img is not None else None, 'fps': float(fps) if fps is not None else None, 'speed_rating': str(speed_rating), 'report_pdf': _Path(pdf_path).name, 'report_pdf_path': str(_Path(pdf_path)), 'charts': charts_out, 'top_classes': top_classes_out}
                    with open(metrics_dir / 'detection_metrics.json', 'w', encoding='utf-8') as f:
                        _json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self._cprint(f'📄 Δημιουργήθηκε αναφορά PDF: {pdf_path.name}', Colors.GREEN)
            return str(pdf_path)
        except Exception as e:
            self._cprint(f'Σφάλμα δημιουργίας PDF αναφοράς: {e}', Colors.RED)

    def run(self) -> None:
        try:
            self._setup_logging()
            if not self.load_model():
                self.finished.emit()
                return
            _find_limit = int(self.max_images) if isinstance(self.max_images, int) and self.max_images > 0 else 500
            images = self.find_images(self.dataset_path, max_images=_find_limit)
            if not images:
                self.error.emit('Δεν βρέθηκαν εικόνες στο επιλεγμένο dataset.')
                self.finished.emit()
            total_images = len(images)
            effective_images = len(images)
            self._cprint(f"Βρέθηκαν συνολικά {total_images} εικόνες στο dataset. Θα αναλυθούν {effective_images} εικόνες (βάσει της ρύθμισης 'Αριθμός ανιχνεύσεων').", Colors.CYAN)
            self.progress.emit(0)
            stats = self.analyze_images(images)
            if self._stop_requested:
                self._cprint('Η διαδικασία σταμάτησε πριν ολοκληρωθεί πλήρως.', Colors.YELLOW)
            else:
                self._cprint('Η ανάλυση ολοκληρώθηκε επιτυχώς.', Colors.GREEN)
            summary_text = self.generate_summary(stats)
            self.summary.emit(summary_text)
            report_path = self.plot_stats(stats, DETECTION_REPORTS_DIR)
            if report_path:
                self.report_ready.emit(report_path)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f'Σφάλμα κατά την ανάλυση: {e}\n{tb}')
        finally:
            try:
                self.progress.emit(100)
            except Exception:
                pass
            self.finished.emit()

class TrainingRunsComparisonDialog(QDialog):
    """
    Διαλόγος σύγκρισης αποτελεσμάτων εκπαιδεύσεων — Αναβαθμισμένη έκδοση.

    Tabs:
      1. Πίνακας  – πλήρης πίνακας metrics με χρωματισμό + φίλτρο
      2. Γραφικές – 6 charts: Metrics Radar, Bar mAP/Accuracy, Loss Comparison,
                    Training Time, Precision-Recall Scatter, Epochs vs Performance
      3. Καμπύλες – Learning curves (loss/metric per epoch) από results.csv
      4. Καλύτερο – Αυτόματη σύσταση καλύτερου μοντέλου ανά κατηγορία

    Πηγές δεδομένων (κατά προτεραιότητα):
      1. metrics/training_metrics.json
      2. results.csv (run φάκελος ή metrics/)
      3. Runs_* φάκελοι YOLO (project dirs)
    """

    # Ορισμός στηλών
    _COLS: list[tuple[str, str, str, str]] = [
        ('Μοντέλο',      'model',           '',        'Όνομα μοντέλου'),
        ('Τύπος',        'model_type',      '',        'CNN / YOLO-det / YOLO-cls'),
        ('Dataset',      'dataset',         '',        'Dataset εκπαίδευσης'),
        ('Εποχές',       'epochs',          '',        'Εποχές που εκτελέσθηκαν'),
        ('imgsz',        'imgsz',           '',        'Μέγεθος εικόνας (px)'),
        ('Device',       'device',          '',        'CPU / GPU'),
        ('mAP50',        'map50',           '{:.4f}',  'YOLO: mAP @IoU=0.50'),
        ('mAP50-95',     'map5095',         '{:.4f}',  'YOLO: mAP @IoU=0.50:0.95'),
        ('Precision',    'precision',       '{:.4f}',  'YOLO Precision'),
        ('Recall',       'recall',          '{:.4f}',  'YOLO Recall'),
        ('Top-1 Acc',    'acc_top1',        '{:.4f}',  'CNN/YOLO-cls: Top-1 Accuracy'),
        ('Top-5 Acc',    'acc_top5',        '{:.4f}',  'CNN/YOLO-cls: Top-5 Accuracy'),
        ('Train Loss',   'train_loss',      '{:.5f}',  'Τελευταία τιμή train loss'),
        ('Val Loss',     'val_loss',        '{:.5f}',  'Τελευταία τιμή val loss'),
        ('Optimizer',    'optimizer',       '',        'Optimizer'),
        ('Batch',        'batch',           '',        'Batch size'),
        ('Χρόνος (min)', 'train_time_min',  '{:.1f}',  'Διάρκεια εκπαίδευσης σε λεπτά'),
        ('Ημερομηνία',   'date',            '',        'Ημερομηνία / ώρα εκπαίδευσης'),
    ]

    _PALETTE = [
        '#7c3aed', '#0066cc', '#1a7f1a', '#cc5500', '#aa1111',
        '#008080', '#8a6000', '#c72c8e', '#2c7a8a', '#5c5c5c',
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('📊 Σύγκριση Εκπαιδεύσεων Μοντέλων')
        # Μεγαλύτερο παράθυρο — προσαρμόζεται στο διαθέσιμο screen
        try:
            from PySide6.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if screen:
                sg = screen.availableGeometry()
                w  = min(1700, int(sg.width()  * 0.93))
                h  = min(1000, int(sg.height() * 0.90))
                self.resize(w, h)
                self.move(max(0, (sg.width()  - w) // 2),
                          max(0, (sg.height() - h) // 2))
            else:
                self.resize(1700, 1000)
        except Exception:
            self.resize(1700, 1000)
        self.setMinimumSize(1100, 700)
        self._runs: list[dict] = []
        self._all_runs: list[dict] = []
        self._selected_indices: set[int] = set()
        self._init_ui()
        QTimer.singleShot(0, self._load_runs)

    # ══════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════

    class _WheelScrollFilter(QObject):
        """
        Event filter που πιάνει WheelEvent από ΟΠΟΙΟΔΗΠΟΤΕ descendant widget
        και κυλά το parent QScrollArea. Εγκαθίσταται αναδρομικά σε όλο το
        widget tree του scroll area, και ξανά κάθε φορά που προστίθεται νέο
        child (π.χ. FigureCanvas matplotlib μετά τη δημιουργία των charts).
        """
        def __init__(self, scroll_area):
            super().__init__(scroll_area)
            self._sa = scroll_area

        def eventFilter(self, obj, event):
            try:
                etype = event.type()
                # Αποδεχόμαστε και τα δύο: QEvent.Type.Wheel (enum) και int 31
                if etype == QEvent.Type.Wheel or int(etype) == 31:
                    # Αν το obj είναι scrollable (π.χ. QTableWidget) αφήνουμε
                    # το event να περάσει στο ίδιο — μόνο αν φτάσει σε
                    # non-scrollable child το κατευθύνουμε στο scroll area.
                    from PySide6.QtWidgets import QAbstractScrollArea
                    if isinstance(obj, QAbstractScrollArea) and obj is not self._sa:
                        # Άλλο scrollable widget — άφησέ το να χειριστεί μόνο του
                        return False
                    sb = self._sa.verticalScrollBar()
                    if sb is not None:
                        delta = event.angleDelta().y()
                        step  = max(80, sb.singleStep() * 5)
                        sb.setValue(sb.value() + (-step if delta > 0 else step))
                        return True
            except Exception:
                pass
            return False

    def _install_wheel_filter(self, scroll_area):
        """
        Εγκαθιστά WheelScrollFilter αναδρομικά σε όλο το widget tree.
        Αποθηκεύει reference ώστε να μπορεί να ξανακληθεί μετά την
        προσθήκη νέων children (charts, curves κ.λπ.).
        """
        if not hasattr(self, '_wheel_filters'):
            self._wheel_filters: dict = {}
        try:
            f = self._wheel_filters.get(id(scroll_area))
            if f is None:
                f = self._WheelScrollFilter(scroll_area)
                self._wheel_filters[id(scroll_area)] = f
            # Εγκατάσταση σε viewport + inner widget + όλα τα children
            self._apply_filter_recursive(scroll_area.viewport(), f)
            inner = scroll_area.widget()
            if inner:
                self._apply_filter_recursive(inner, f)
            scroll_area.installEventFilter(f)
        except Exception:
            pass

    @staticmethod
    def _apply_filter_recursive(widget, event_filter):
        """Εγκαθιστά event_filter αναδρομικά σε widget και όλα τα children."""
        try:
            if widget is None:
                return
            widget.installEventFilter(event_filter)
            for child in widget.findChildren(QWidget):
                try:
                    child.installEventFilter(event_filter)
                except Exception:
                    pass
        except Exception:
            pass

    def _reapply_wheel_filters(self):
        """
        Ξανά-εφαρμόζει wheel filters μετά από προσθήκη νέων widgets
        (π.χ. αφού δημιουργηθούν τα matplotlib charts).
        """
        for sa_id, f in getattr(self, '_wheel_filters', {}).items():
            try:
                sa = f._sa
                self._apply_filter_recursive(sa.viewport(), f)
                inner = sa.widget()
                if inner:
                    self._apply_filter_recursive(inner, f)
            except Exception:
                pass

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Top toolbar ────────────────────────────────────────────────────
        top = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText('🔍  Φίλτρο (μοντέλο / dataset / τύπος…)')
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.refresh_btn   = QPushButton('🔄 Ανανέωση')
        self.export_csv_btn = QPushButton('💾 CSV')
        self.charts_btn    = QPushButton('📈 Ανανέωση Γραφικών')
        self.refresh_btn.clicked.connect(self._load_runs)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.charts_btn.clicked.connect(self._refresh_all_charts)

        top.addWidget(QLabel('<b>Αποτελέσματα εκπαιδεύσεων (YOLO + CNN):</b>'))
        top.addSpacing(10)
        top.addWidget(self.filter_edit, 1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.export_csv_btn)
        top.addWidget(self.charts_btn)
        layout.addLayout(top)

        # ── Main tab widget ────────────────────────────────────────────────
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Tab 1 – Table
        self._build_table_tab()
        # Tab 2 – Charts
        self._build_charts_tab()
        # Tab 3 – Learning Curves
        self._build_curves_tab()
        # Tab 4 – Best Model
        self._build_best_tab()

        self.tabs.currentChanged.connect(self._on_tab_changed)

        # ── Status + close ────────────────────────────────────────────────
        bottom = QHBoxLayout()
        self.status_label = QLabel('Φόρτωση…')
        bottom.addWidget(self.status_label, 1)
        close_btn = QPushButton('Κλείσιμο')
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    # ── Tab 1: Table ───────────────────────────────────────────────────────
    def _build_table_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Legend
        leg = QHBoxLayout()
        for color, label in [('#1a7f1a', '≥ 0.85 Εξαιρετικό'),
                              ('#8a6000', '≥ 0.70 Καλό'),
                              ('#aa1111', '< 0.70 Χαμηλό')]:
            dot = QLabel('●')
            dot.setStyleSheet(f'color:{color}; font-size:15px;')
            leg.addWidget(dot)
            leg.addWidget(QLabel(label))
            leg.addSpacing(14)
        tip = QLabel('(Επίλεξε γραμμές → ανανεώνονται τα charts)')
        tip.setStyleSheet('color:#888; font-style:italic;')
        leg.addWidget(tip)
        leg.addStretch()
        v.addLayout(leg)

        ncols = len(self._COLS)
        self.table = QTableWidget(0, ncols)
        self.table.setHorizontalHeaderLabels([c[0] for c in self._COLS])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 210)
        self.table.setColumnWidth(2, 140)
        for i, (_, _, _, tip2) in enumerate(self._COLS):
            self.table.horizontalHeaderItem(i).setToolTip(tip2)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        # Βεβαιώνουμε ότι το table δέχεται wheel events κανονικά
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.verticalScrollBar().setSingleStep(20)
        v.addWidget(self.table, 1)
        self.tabs.addTab(w, '📋 Πίνακας')

    # ── Tab 2: Charts ──────────────────────────────────────────────────────
    def _build_charts_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        info = QLabel('Επίλεξε γραμμές στον Πίνακα (ή άφησε χωρίς επιλογή για όλα) → 📈 Ανανέωση Γραφικών')
        info.setStyleSheet('color:#555; font-style:italic; padding:2px;')
        v.addWidget(info)
        self.charts_scroll = QScrollArea()
        self.charts_scroll.setWidgetResizable(True)
        self.charts_scroll.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.charts_inner = QWidget()
        self.charts_grid  = QGridLayout(self.charts_inner)
        self.charts_grid.setSpacing(8)
        self.charts_scroll.setWidget(self.charts_inner)
        self._install_wheel_filter(self.charts_scroll)
        v.addWidget(self.charts_scroll, 1)
        self.tabs.addTab(w, '📈 Γραφικές')

    # ── Tab 3: Learning Curves ─────────────────────────────────────────────
    def _build_curves_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        top.addWidget(QLabel('Επιλογή μοντέλου:'))
        self.curves_combo = QComboBox()
        self.curves_combo.currentIndexChanged.connect(self._refresh_curves)
        top.addWidget(self.curves_combo, 1)
        v.addLayout(top)
        self.curves_scroll = QScrollArea()
        self.curves_scroll.setWidgetResizable(True)
        self.curves_scroll.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.curves_inner = QWidget()
        self.curves_vbox  = QVBoxLayout(self.curves_inner)
        self.curves_scroll.setWidget(self.curves_inner)
        self._install_wheel_filter(self.curves_scroll)
        v.addWidget(self.curves_scroll, 1)
        self.tabs.addTab(w, '📉 Καμπύλες Εκπαίδευσης')

    # ── Tab 4: Best Model ──────────────────────────────────────────────────
    def _build_best_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)
        self.best_scroll = QScrollArea()
        self.best_scroll.setWidgetResizable(True)
        self.best_scroll.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.best_inner = QWidget()
        self.best_vbox  = QVBoxLayout(self.best_inner)
        self.best_vbox.setSpacing(10)
        self.best_scroll.setWidget(self.best_inner)
        self._install_wheel_filter(self.best_scroll)
        v.addWidget(self.best_scroll, 1)
        self.tabs.addTab(w, '🏆 Καλύτερο Μοντέλο')

    # ══════════════════════════════════════════════════════════════════════
    #  Data loading
    # ══════════════════════════════════════════════════════════════════════

    def _load_runs(self):
        self._runs = []
        seen_keys: set[str] = set()

        # 1. Trained_Models/<run>/metrics/training_metrics.json
        try:
            for model_dir in sorted(TRAINED_MODELS_DIR.iterdir()):
                if not model_dir.is_dir():
                    continue
                jpath = model_dir / 'metrics' / 'training_metrics.json'
                if jpath.is_file():
                    try:
                        data = json.loads(jpath.read_text(encoding='utf-8', errors='replace'))
                        row = self._parse_metrics_json(data, model_dir)
                        if row:
                            k = self._dedup_key(row)
                            if k not in seen_keys:
                                seen_keys.add(k)
                                self._runs.append(row)
                            continue
                    except Exception:
                        pass
                for csv_path in (
                    list(model_dir.glob('results.csv')) +
                    list((model_dir / 'metrics').glob('results.csv')
                         if (model_dir / 'metrics').is_dir() else [])
                ):
                    try:
                        row = self._parse_results_csv(csv_path, model_dir)
                        if row:
                            k = self._dedup_key(row)
                            if k not in seen_keys:
                                seen_keys.add(k)
                                self._runs.append(row)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2. Runs_* YOLO project dirs
        try:
            for runs_parent in sorted(ROOT_DIR.iterdir()):
                if not runs_parent.is_dir() or not runs_parent.name.startswith('Runs_'):
                    continue
                for run_dir in sorted(runs_parent.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    for csv_path in run_dir.glob('results.csv'):
                        try:
                            row = self._parse_results_csv(csv_path, run_dir)
                            if row:
                                k = self._dedup_key(row)
                                if k not in seen_keys:
                                    seen_keys.add(k)
                                    self._runs.append(row)
                        except Exception:
                            pass
        except Exception:
            pass

        self._runs.sort(key=lambda r: str(r.get('date', '')), reverse=True)
        self._all_runs = list(self._runs)
        self._populate_table(self._runs)
        self._populate_curves_combo()
        self._refresh_best_panel()

    @staticmethod
    def _dedup_key(row: dict) -> str:
        return '|'.join([str(row.get(k, ''))
                         for k in ('model', 'dataset', 'epochs', 'imgsz', 'device')])

    # ══════════════════════════════════════════════════════════════════════
    #  Parsers
    # ══════════════════════════════════════════════════════════════════════

    def _parse_metrics_json(self, data: dict, model_dir: Path) -> dict | None:
        try:
            kind   = str(data.get('kind', '')).lower()
            is_cnn = 'cnn' in kind
            fm     = data.get('final_metrics', {}) or {}

            def _ri(label: str):
                for pair in (data.get('run_info_rows', []) or []):
                    try:
                        if str(pair[0]).strip().lower().startswith(label.lower()):
                            return str(pair[1]).strip()
                    except Exception:
                        pass
                return None

            model_name   = data.get('model_name') or _ri('model') or model_dir.name
            dataset_name = data.get('dataset_name') or _ri('dataset') or '—'
            device       = data.get('device') or _ri('device') or self._infer_device(model_dir.name)
            imgsz        = data.get('imgsz') or _ri('image size') or self._infer_imgsz(model_dir.name)
            generated_at = data.get('generated_at', '')
            if isinstance(generated_at, str) and 'T' in generated_at:
                try:
                    generated_at = datetime.fromisoformat(generated_at).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass

            epochs_ran = fm.get('epochs_ran') or _ri('epochs')
            task = str(fm.get('task_type', '')).lower()
            if is_cnn:
                model_type = 'CNN'
            elif 'classify' in task or 'cls' in task:
                model_type = 'YOLO-cls'
            elif 'detect' in task:
                model_type = 'YOLO-det'
            else:
                model_type = self._infer_model_type(str(model_name))

            def _fv(*keys):
                for k in keys:
                    v = fm.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
                return None

            map50     = _fv('map50', 'best_map50')
            map5095   = _fv('map5095', 'best_map5095', 'map50-95')
            precision = _fv('precision', 'best_precision')
            recall    = _fv('recall', 'best_recall')
            acc_top1  = _fv('acc_top1', 'accuracy_top1', 'best_top1', 'top1')
            acc_top5  = _fv('acc_top5', 'accuracy_top5', 'best_top5', 'top5')
            train_loss= _fv('train_loss', 'final_train_loss', 'train/loss')
            val_loss  = _fv('val_loss', 'final_val_loss', 'val/loss')

            # Enrich from CSV
            csv_path = model_dir / 'metrics' / 'results.csv'
            if not csv_path.is_file():
                csv_path = model_dir / 'results.csv'
            if csv_path.is_file():
                try:
                    extra = self._read_csv_last_row(csv_path)
                    if extra:
                        if map50     is None: map50     = extra.get('map50')
                        if map5095   is None: map5095   = extra.get('map5095')
                        if precision is None: precision = extra.get('precision')
                        if recall    is None: recall    = extra.get('recall')
                        if acc_top1  is None: acc_top1  = extra.get('acc_top1')
                        if acc_top5  is None: acc_top5  = extra.get('acc_top5')
                        if train_loss is None: train_loss= extra.get('train_loss')
                        if val_loss   is None: val_loss  = extra.get('val_loss')
                        if epochs_ran is None: epochs_ran = extra.get('epochs')
                except Exception:
                    pass

            optimizer = data.get('optimizer') or fm.get('optimizer') or '—'
            batch     = data.get('batch')     or fm.get('batch')     or '—'
            t_sec = fm.get('train_duration_sec') or fm.get('train_time_sec')
            t_min = round(float(t_sec) / 60, 1) if t_sec else self._estimate_time_from_dir(model_dir)

            if not generated_at:
                try:
                    mtime = (model_dir / 'metrics' / 'training_metrics.json').stat().st_mtime
                    generated_at = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    generated_at = '—'

            # CSV path for learning curves
            csv_for_curves = model_dir / 'metrics' / 'results.csv'
            if not csv_for_curves.is_file():
                csv_for_curves = model_dir / 'results.csv'

            return {
                'model':          str(model_name),
                'model_type':     model_type,
                'dataset':        str(dataset_name),
                'epochs':         str(epochs_ran) if epochs_ran is not None else '—',
                'imgsz':          str(imgsz) if imgsz else '—',
                'device':         str(device),
                'map50':          map50,
                'map5095':        map5095,
                'precision':      precision,
                'recall':         recall,
                'acc_top1':       acc_top1,
                'acc_top5':       acc_top5,
                'train_loss':     train_loss,
                'val_loss':       val_loss,
                'optimizer':      str(optimizer),
                'batch':          str(batch),
                'train_time_min': t_min,
                'date':           str(generated_at),
                '_csv_path':      str(csv_for_curves) if csv_for_curves.is_file() else None,
                '_src':           'metrics_json',
            }
        except Exception:
            return None

    def _parse_results_csv(self, csv_path: Path, model_dir: Path) -> dict | None:
        try:
            extra = self._read_csv_last_row(csv_path)
            if extra is None:
                return None
            dir_name   = model_dir.name
            model_name = self._infer_model_name(dir_name)
            model_type = self._infer_model_type(dir_name)
            dataset    = self._infer_dataset(dir_name)
            device     = self._infer_device(dir_name)
            imgsz      = self._infer_imgsz(dir_name)
            try:
                epochs_ran = sum(1 for _ in open(csv_path, encoding='utf-8', errors='replace')) - 1
            except Exception:
                epochs_ran = extra.get('epochs') or '—'
            try:
                date_str = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            except Exception:
                date_str = '—'
            return {
                'model':          model_name,
                'model_type':     model_type,
                'dataset':        str(dataset),
                'epochs':         str(epochs_ran),
                'imgsz':          str(imgsz) if imgsz else '—',
                'device':         str(device),
                'map50':          extra.get('map50'),
                'map5095':        extra.get('map5095'),
                'precision':      extra.get('precision'),
                'recall':         extra.get('recall'),
                'acc_top1':       extra.get('acc_top1'),
                'acc_top5':       extra.get('acc_top5'),
                'train_loss':     extra.get('train_loss'),
                'val_loss':       extra.get('val_loss'),
                'optimizer':      '—',
                'batch':          '—',
                'train_time_min': self._estimate_time_from_dir(model_dir),
                'date':           date_str,
                '_csv_path':      str(csv_path),
                '_src':           'csv',
            }
        except Exception:
            return None

    @staticmethod
    def _read_csv_last_row(csv_path: Path) -> dict | None:
        try:
            import csv as _csv
            rows = []
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                for r in _csv.DictReader(f):
                    rows.append(r)
            if not rows:
                return None
            last = rows[-1]
            def _f(*frags) -> float | None:
                for k, v in last.items():
                    kn = k.strip().lower().replace(' ', '').replace('/', '')
                    for frag in frags:
                        if frag.lower().replace(' ', '').replace('/', '') in kn:
                            try:
                                return float(str(v).strip())
                            except Exception:
                                pass
                return None
            return {
                'map50':      _f('map50'),
                'map5095':    _f('map50-95', 'map5095'),
                'precision':  _f('precision(b)', 'metrics/precision', 'precision'),
                'recall':     _f('recall(b)',    'metrics/recall',    'recall'),
                'acc_top1':   _f('accuracy_top1', 'acctop1', 'top1'),
                'acc_top5':   _f('accuracy_top5', 'acctop5', 'top5'),
                'train_loss': _f('train/loss', 'trainloss', 'train/box_loss'),
                'val_loss':   _f('val/loss',   'valloss',   'val/box_loss'),
                'epochs':     len(rows),
            }
        except Exception:
            return None

    @staticmethod
    def _read_csv_all_rows(csv_path: str) -> dict[str, list]:
        """Επιστρέφει {column_name: [values]} για learning curves."""
        try:
            import csv as _csv
            rows = []
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                for r in _csv.DictReader(f):
                    rows.append(r)
            if not rows:
                return {}
            out: dict[str, list] = {}
            for k in rows[0]:
                vals = []
                for r in rows:
                    try:
                        vals.append(float(str(r[k]).strip()))
                    except Exception:
                        vals.append(None)
                out[k.strip()] = vals
            return out
        except Exception:
            return {}

    # ── Inference helpers ──────────────────────────────────────────────────
    @staticmethod
    def _infer_model_name(dir_name: str) -> str:
        parts = dir_name.split('_')
        out = []
        for p in parts:
            if p.upper() in ('GPU', 'CPU', 'CUDA'):
                break
            out.append(p)
        return '_'.join(out) if out else dir_name

    @staticmethod
    def _infer_device(dir_name: str) -> str:
        n = dir_name.upper()
        if 'GPU' in n or 'CUDA' in n:
            return 'GPU'
        if 'CPU' in n:
            return 'CPU'
        return '—'

    @staticmethod
    def _infer_imgsz(dir_name: str) -> str | None:
        m = re.search(r'imgsz(\d+)', dir_name, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r'_(\d{3,4})$', dir_name)
        if m2:
            return m2.group(1)
        m3 = re.search(r'_(\d{3,4})_', dir_name)
        if m3:
            return m3.group(1)
        return None

    @staticmethod
    def _infer_dataset(dir_name: str) -> str:
        parts = dir_name.split('_')
        collecting = False
        ds_parts = []
        for p in parts:
            if p.upper() in ('GPU', 'CPU', 'CUDA'):
                collecting = True
                continue
            if collecting:
                if p.isdigit() and len(p) in (3, 4):
                    break
                ds_parts.append(p)
        return '_'.join(ds_parts) if ds_parts else dir_name

    @staticmethod
    def _infer_model_type(name: str) -> str:
        n = name.lower()
        for cnn in _CNN_MODEL_KEYS:
            if cnn in n:
                return 'CNN'
        if '-cls' in n or '_cls' in n or 'cls' in n:
            return 'YOLO-cls'
        if any(x in n for x in ('yolo', 'finetuned')):
            return 'YOLO-det'
        return '—'

    @staticmethod
    def _estimate_time_from_dir(model_dir: Path) -> float | None:
        try:
            candidates = list(model_dir.rglob('*.pt')) + list(model_dir.rglob('results.csv'))
            if not candidates:
                return None
            mtimes = sorted(p.stat().st_mtime for p in candidates)
            if len(mtimes) < 2:
                return None
            delta = (mtimes[-1] - mtimes[0]) / 60.0
            return round(delta, 1) if delta > 0.5 else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════
    #  Table
    # ══════════════════════════════════════════════════════════════════════

    def _populate_table(self, runs: list[dict]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        _METRIC_COLS = {'map50', 'map5095', 'precision', 'recall', 'acc_top1', 'acc_top5'}
        _LOSS_COLS   = {'train_loss', 'val_loss'}

        for r in runs:
            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)
            is_cnn    = str(r.get('model_type', '')).upper() == 'CNN'
            is_detect = str(r.get('model_type', '')).upper() == 'YOLO-DET'

            for col_idx, (_, key, fmt, _) in enumerate(self._COLS):
                val  = r.get(key)
                if val is None or val == '' or val == '—':
                    text = '—'
                elif fmt and isinstance(val, float):
                    text = fmt.format(val)
                else:
                    text = str(val)

                item = QTableWidgetItem(text)
                # Numeric sort
                if isinstance(val, float):
                    item.setData(Qt.ItemDataRole.UserRole, val)

                if key in _METRIC_COLS and isinstance(val, float):
                    if val >= 0.85:
                        item.setForeground(QColor('#1a7f1a'))
                    elif val >= 0.70:
                        item.setForeground(QColor('#8a6000'))
                    else:
                        item.setForeground(QColor('#aa1111'))
                elif key in _LOSS_COLS and isinstance(val, float):
                    if val < 0.10:
                        item.setForeground(QColor('#1a7f1a'))
                    elif val < 0.40:
                        item.setForeground(QColor('#8a6000'))
                    else:
                        item.setForeground(QColor('#aa1111'))
                if key in ('map50', 'map5095', 'precision', 'recall') and (is_cnn or not is_detect) and text == '—':
                    item.setForeground(QColor('#bbbbbb'))
                if key in ('acc_top1', 'acc_top5') and is_detect and text == '—':
                    item.setForeground(QColor('#bbbbbb'))
                if key == 'model_type':
                    mt = str(val or '').upper()
                    if mt == 'CNN':
                        item.setForeground(QColor('#7c3aed'))
                    elif mt == 'YOLO-CLS':
                        item.setForeground(QColor('#0066cc'))
                    elif mt == 'YOLO-DET':
                        item.setForeground(QColor('#1a7f1a'))

                self.table.setItem(row_idx, col_idx, item)

        self.table.setSortingEnabled(True)
        n     = len(runs)
        total = len(self._all_runs)
        if n == total:
            self.status_label.setText(f'{n} εκπαίδευση(-σεις) βρέθηκαν.')
        else:
            self.status_label.setText(f'{n} από {total} εκπαιδεύσεις (φίλτρο ενεργό).')

    def _apply_filter(self, text: str):
        q = text.strip().lower()
        src = self._all_runs
        if q:
            src = [r for r in src if any(q in str(v).lower() for v in r.values())]
        self._populate_table(src)

    def _on_selection_changed(self):
        self._selected_indices = {r for r in self.table.selectionModel().selectedRows()}

    # ══════════════════════════════════════════════════════════════════════
    #  Charts (Tab 2)
    # ══════════════════════════════════════════════════════════════════════

    def _on_tab_changed(self, idx: int):
        if idx == 1:
            self._refresh_all_charts()
        elif idx == 2:
            self._refresh_curves()
        elif idx == 3:
            self._refresh_best_panel()

    def _get_chart_runs(self) -> list[dict]:
        """Επιστρέφει επιλεγμένες γραμμές ή όλες αν δεν υπάρχει επιλογή."""
        sel_rows = [r.row() for r in self.table.selectionModel().selectedRows()]
        runs = self._all_runs
        # Map visible table rows → runs (using model name match)
        if sel_rows:
            visible: list[dict] = []
            for idx in sel_rows:
                try:
                    mname = self.table.item(idx, 0)
                    if mname:
                        nm = mname.text()
                        match = next((r for r in runs if r.get('model') == nm), None)
                        if match:
                            visible.append(match)
                except Exception:
                    pass
            if visible:
                return visible
        return runs

    def _refresh_all_charts(self):
        """Δημιουργεί/ανανεώνει όλα τα charts στο Tab Γραφικές."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except ImportError:
            self._show_chart_error('Απαιτείται matplotlib. Εγκατάσταση: pip install matplotlib')
            return

        runs = self._get_chart_runs()
        if not runs:
            self._show_chart_error('Δεν υπάρχουν δεδομένα για γραφικές παραστάσεις.')
            return

        # Clear old charts
        while self.charts_grid.count():
            item = self.charts_grid.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        labels  = [r.get('model', f'Run {i+1}')[:22] for i, r in enumerate(runs)]
        colors  = [self._PALETTE[i % len(self._PALETTE)] for i in range(len(runs))]
        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9,
                              'axes.labelsize': 8, 'font.family': 'DejaVu Sans'})

        charts_created = []

        # ── Chart 1: Bar – mAP50 / Top-1 Accuracy ─────────────────────────
        try:
            fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
            ax1.set_title('mAP50 / Top-1 Accuracy ανά μοντέλο', fontweight='bold')
            x = range(len(runs))
            w = 0.38
            bars_a, bars_b = [], []
            for i, r in enumerate(runs):
                v1 = r.get('map50') if r.get('map50') is not None else r.get('acc_top1')
                v2 = r.get('acc_top1') if r.get('map50') is not None else None
                b1 = ax1.bar(i - w/2, v1 or 0, w, color=colors[i], alpha=0.85, label=labels[i])
                if v2 is not None:
                    b2 = ax1.bar(i + w/2, v2, w, color=colors[i], alpha=0.45, hatch='//')
                    bars_b.append(b2)
                bars_a.append(b1)
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
            ax1.set_ylim(0, 1.08)
            ax1.set_ylabel('Score')
            ax1.axhline(0.85, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='0.85 threshold')
            ax1.axhline(0.70, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='0.70 threshold')
            ax1.legend(fontsize=6, loc='upper right')
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            fig1.tight_layout()
            charts_created.append(('mAP50 / Accuracy', fig1))
        except Exception:
            pass

        # ── Chart 2: Bar – Precision & Recall ─────────────────────────────
        try:
            fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
            ax2.set_title('Precision & Recall ανά μοντέλο', fontweight='bold')
            for i, r in enumerate(runs):
                prec = r.get('precision')
                rec  = r.get('recall')
                if prec is not None:
                    ax2.bar(i - 0.2, prec, 0.35, color=colors[i], alpha=0.85)
                if rec is not None:
                    ax2.bar(i + 0.2, rec,  0.35, color=colors[i], alpha=0.45, hatch='//')
            ax2.set_xticks(list(range(len(runs))))
            ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
            ax2.set_ylim(0, 1.08)
            ax2.set_ylabel('Score')
            from matplotlib.patches import Patch
            ax2.legend(handles=[Patch(color='grey', alpha=0.85, label='Precision'),
                                 Patch(color='grey', alpha=0.45, hatch='//', label='Recall')],
                       fontsize=7)
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            fig2.tight_layout()
            charts_created.append(('Precision & Recall', fig2))
        except Exception:
            pass

        # ── Chart 3: Bar – Train/Val Loss ─────────────────────────────────
        try:
            fig3, ax3 = plt.subplots(figsize=(6.5, 4.0))
            ax3.set_title('Train / Val Loss ανά μοντέλο (χαμηλότερο = καλύτερο)', fontweight='bold')
            for i, r in enumerate(runs):
                tl = r.get('train_loss')
                vl = r.get('val_loss')
                if tl is not None:
                    ax3.bar(i - 0.2, tl, 0.35, color=colors[i], alpha=0.85)
                if vl is not None:
                    ax3.bar(i + 0.2, vl, 0.35, color=colors[i], alpha=0.45, hatch='//')
            ax3.set_xticks(list(range(len(runs))))
            ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
            ax3.set_ylabel('Loss')
            from matplotlib.patches import Patch as _P
            ax3.legend(handles=[_P(color='grey', alpha=0.85, label='Train Loss'),
                                 _P(color='grey', alpha=0.45, hatch='//', label='Val Loss')],
                       fontsize=7)
            ax3.grid(axis='y', linestyle='--', alpha=0.3)
            fig3.tight_layout()
            charts_created.append(('Train / Val Loss', fig3))
        except Exception:
            pass

        # ── Chart 4: Scatter – Precision vs Recall ────────────────────────
        try:
            fig4, ax4 = plt.subplots(figsize=(5.5, 4.5))
            ax4.set_title('Precision vs Recall', fontweight='bold')
            for i, r in enumerate(runs):
                prec = r.get('precision')
                rec  = r.get('recall')
                if prec is not None and rec is not None:
                    ax4.scatter(rec, prec, color=colors[i], s=90, zorder=3)
                    ax4.annotate(labels[i], (rec, prec), fontsize=6,
                                 textcoords='offset points', xytext=(5, 3))
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_xlim(-0.05, 1.1)
            ax4.set_ylim(-0.05, 1.1)
            ax4.axhline(0.85, color='green', linestyle='--', linewidth=0.7, alpha=0.4)
            ax4.axvline(0.85, color='green', linestyle='--', linewidth=0.7, alpha=0.4)
            ax4.grid(linestyle='--', alpha=0.3)
            fig4.tight_layout()
            charts_created.append(('Precision vs Recall', fig4))
        except Exception:
            pass

        # ── Chart 5: Bar – Training Time ──────────────────────────────────
        try:
            times = [(labels[i], r.get('train_time_min'), colors[i])
                     for i, r in enumerate(runs) if r.get('train_time_min')]
            if times:
                fig5, ax5 = plt.subplots(figsize=(6.0, 3.8))
                ax5.set_title('Χρόνος εκπαίδευσης (λεπτά)', fontweight='bold')
                lbs = [t[0] for t in times]
                vals= [t[1] for t in times]
                cols= [t[2] for t in times]
                bars = ax5.bar(range(len(times)), vals, color=cols, alpha=0.85)
                for bar, v in zip(bars, vals):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                             f'{v:.1f}', ha='center', va='bottom', fontsize=7)
                ax5.set_xticks(list(range(len(times))))
                ax5.set_xticklabels(lbs, rotation=30, ha='right', fontsize=7)
                ax5.set_ylabel('Λεπτά')
                ax5.grid(axis='y', linestyle='--', alpha=0.3)
                fig5.tight_layout()
                charts_created.append(('Χρόνος Εκπαίδευσης', fig5))
        except Exception:
            pass

        # ── Chart 6: Radar – Συνολικό προφίλ μοντέλου ────────────────────
        try:
            import numpy as np
            radar_metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'Top-1', 'Top-5']
            radar_keys    = ['map50', 'map5095', 'precision', 'recall', 'acc_top1', 'acc_top5']
            N = len(radar_metrics)
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]

            fig6, ax6 = plt.subplots(figsize=(5.5, 5.0), subplot_kw=dict(polar=True))
            ax6.set_title('Radar: Συνολικό Προφίλ Μοντέλου', fontweight='bold', pad=18)
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(radar_metrics, fontsize=8)
            ax6.set_ylim(0, 1)
            ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax6.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)

            for i, r in enumerate(runs):
                vals = [r.get(k) or 0.0 for k in radar_keys]
                vals += vals[:1]
                ax6.plot(angles, vals, color=colors[i], linewidth=1.5, label=labels[i])
                ax6.fill(angles, vals, color=colors[i], alpha=0.07)
            ax6.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=6)
            fig6.tight_layout()
            charts_created.append(('Radar Προφίλ', fig6))
        except Exception:
            pass

        # ── Chart 7: Epochs vs Best Metric ────────────────────────────────
        try:
            fig7, ax7 = plt.subplots(figsize=(6.5, 4.0))
            ax7.set_title('Εποχές vs Καλύτερο Metric', fontweight='bold')
            for i, r in enumerate(runs):
                try:
                    ep = int(str(r.get('epochs', '0')).replace('—', '0'))
                except Exception:
                    ep = 0
                met = r.get('map50') or r.get('acc_top1')
                if ep > 0 and met is not None:
                    ax7.scatter(ep, met, color=colors[i], s=90, zorder=3)
                    ax7.annotate(labels[i], (ep, met), fontsize=6,
                                 textcoords='offset points', xytext=(4, 3))
            ax7.set_xlabel('Εποχές')
            ax7.set_ylabel('mAP50 / Top-1 Acc')
            ax7.set_ylim(0, 1.1)
            ax7.grid(linestyle='--', alpha=0.3)
            fig7.tight_layout()
            charts_created.append(('Εποχές vs Metric', fig7))
        except Exception:
            pass

        # Render all charts to grid
        for idx, (title, fig) in enumerate(charts_created):
            row_g = idx // 2
            col_g = idx % 2
            try:
                canvas = FigureCanvas(fig)
                canvas.setFixedHeight(320)
                frame = QGroupBox(title)
                fv = QVBoxLayout(frame)
                fv.setContentsMargins(2, 2, 2, 2)
                fv.addWidget(canvas)
                self.charts_grid.addWidget(frame, row_g, col_g)
                plt.close(fig)
            except Exception:
                plt.close(fig)

        if not charts_created:
            self._show_chart_error('Δεν βρέθηκαν αρκετά δεδομένα για γραφικές.')

        # Ξανά-εφαρμογή wheel filters σε όλα τα νέα children (FigureCanvas κ.λπ.)
        QTimer.singleShot(50, self._reapply_wheel_filters)

    def _show_chart_error(self, msg: str):
        while self.charts_grid.count():
            item = self.charts_grid.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        lbl = QLabel(f'⚠️ {msg}')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet('color:#888; font-size:12px; padding:20px;')
        self.charts_grid.addWidget(lbl, 0, 0)

    # ══════════════════════════════════════════════════════════════════════
    #  Learning Curves (Tab 3)
    # ══════════════════════════════════════════════════════════════════════

    def _populate_curves_combo(self):
        self.curves_combo.blockSignals(True)
        self.curves_combo.clear()
        for r in self._all_runs:
            cp = r.get('_csv_path')
            if cp and Path(cp).is_file():
                self.curves_combo.addItem(r.get('model', '?'), cp)
        if self.curves_combo.count() == 0:
            self.curves_combo.addItem('(Δεν βρέθηκαν results.csv)')
        self.curves_combo.blockSignals(False)

    def _refresh_curves(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except ImportError:
            return

        csv_path = self.curves_combo.currentData()
        if not csv_path or not Path(csv_path).is_file():
            return

        # Clear
        while self.curves_vbox.count():
            item = self.curves_vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        data = self._read_csv_all_rows(csv_path)
        if not data:
            lbl = QLabel('⚠️ Δεν βρέθηκαν δεδομένα στο CSV.')
            self.curves_vbox.addWidget(lbl)
            return

        # Find epoch column
        epoch_col = next((k for k in data if 'epoch' in k.lower()), None)
        epochs = data.get(epoch_col, list(range(len(next(iter(data.values()))))))

        def _clean(vals):
            return [v if v is not None else float('nan') for v in vals]

        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'font.family': 'DejaVu Sans'})

        plot_groups = [
            ('Loss Curves',
             [k for k in data if 'loss' in k.lower() and epoch_col and k != epoch_col],
             'Loss'),
            ('Accuracy / mAP',
             [k for k in data if any(x in k.lower() for x in ('accuracy', 'map', 'precision', 'recall'))
              and epoch_col and k != epoch_col],
             'Score'),
            ('Learning Rate',
             [k for k in data if 'lr' in k.lower() and epoch_col and k != epoch_col],
             'LR'),
        ]

        import numpy as np
        for title, cols, ylabel in plot_groups:
            cols = [c for c in cols if c in data]
            if not cols:
                continue
            fig, ax = plt.subplots(figsize=(10.0, 3.8))
            ax.set_title(title, fontweight='bold')
            for i, col in enumerate(cols):
                vals = _clean(data[col])
                if any(not (isinstance(v, float) and v != v) for v in vals):
                    ax.plot(epochs[:len(vals)], vals,
                            label=col, color=self._PALETTE[i % len(self._PALETTE)],
                            linewidth=1.4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc='best')
            ax.grid(linestyle='--', alpha=0.3)
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.setFixedHeight(280)
            frame = QGroupBox(title)
            fv = QVBoxLayout(frame)
            fv.setContentsMargins(2, 2, 2, 2)
            fv.addWidget(canvas)
            self.curves_vbox.addWidget(frame)
            plt.close(fig)

        if self.curves_vbox.count() == 0:
            lbl = QLabel('⚠️ Δεν βρέθηκαν κατάλληλες στήλες για γραφικές.')
            self.curves_vbox.addWidget(lbl)

        # Ξανά-εφαρμογή wheel filters στα νέα children
        QTimer.singleShot(50, self._reapply_wheel_filters)

    # ══════════════════════════════════════════════════════════════════════
    #  Best Model Panel (Tab 4)
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_best_panel(self):
        while self.best_vbox.count():
            item = self.best_vbox.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        runs = self._all_runs
        if not runs:
            self.best_vbox.addWidget(QLabel('⚠️ Δεν υπάρχουν δεδομένα.'))
            return

        def _safe_float(v) -> float:
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        # Criteria definitions: (label, scoring function, explanation)
        criteria = [
            ('🏆 Υψηλότερο mAP50 (YOLO Detection)',
             lambda r: _safe_float(r.get('map50')) if r.get('model_type') == 'YOLO-det' else -1,
             'Ο καλύτερος ανιχνευτής βάσει mean Average Precision @IoU=0.50.'),

            ('🏆 Υψηλότερο mAP50-95 (YOLO Detection)',
             lambda r: _safe_float(r.get('map5095')) if r.get('model_type') == 'YOLO-det' else -1,
             'Αυστηρότερη αξιολόγηση: mAP @IoU=0.50:0.95.'),

            ('🏆 Υψηλότερο Top-1 Accuracy (CNN / YOLO-cls)',
             lambda r: _safe_float(r.get('acc_top1')) if r.get('model_type') in ('CNN', 'YOLO-cls') else -1,
             'Ο καλύτερος classifier βάσει Top-1 ακρίβειας.'),

            ('🏆 Καλύτερη Ισορροπία Precision-Recall (F1)',
             lambda r: (2 * _safe_float(r.get('precision')) * _safe_float(r.get('recall'))
                        / max(_safe_float(r.get('precision')) + _safe_float(r.get('recall')), 1e-9))
                       if r.get('precision') and r.get('recall') else -1,
             'F1 = 2·P·R/(P+R). Ισορροπεί precision και recall.'),

            ('⚡ Ταχύτερη Εκπαίδευση (χαμηλότερος χρόνος)',
             lambda r: -_safe_float(r.get('train_time_min')) if r.get('train_time_min') else -9999,
             'Το μοντέλο με τον μικρότερο χρόνο εκπαίδευσης.'),

            ('📉 Χαμηλότερο Val Loss',
             lambda r: -_safe_float(r.get('val_loss')) if r.get('val_loss') else -9999,
             'Χαμηλό val loss = καλύτερη γενίκευση.'),

            ('🎯 Συνολική Βαθμολογία (weighted)',
             lambda r: (
                 0.35 * _safe_float(r.get('map50') or r.get('acc_top1')) +
                 0.25 * _safe_float(r.get('map5095') or r.get('acc_top5')) +
                 0.20 * _safe_float(r.get('precision')) +
                 0.20 * _safe_float(r.get('recall'))
             ),
             'Σταθμισμένη βαθμολογία: 35% mAP50/Top1 + 25% mAP95/Top5 + 20% Precision + 20% Recall.'),
        ]

        title_lbl = QLabel('<h2 style="margin:0;">🏆 Ανάλυση Καλύτερου Μοντέλου</h2>')
        title_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.best_vbox.addWidget(title_lbl)

        sub_lbl = QLabel(f'Αναλύθηκαν <b>{len(runs)}</b> εκπαιδεύσεις.')
        self.best_vbox.addWidget(sub_lbl)

        for crit_label, score_fn, explanation in criteria:
            try:
                scored = [(r, score_fn(r)) for r in runs]
                scored = [(r, s) for r, s in scored if s > -999]
                if not scored:
                    continue
                best_run, best_score = max(scored, key=lambda x: x[1])

                card = QFrame()
                card.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
                card.setStyleSheet('QFrame { border: 1px solid #7c3aed; border-radius: 8px; '
                                   'background: #faf8ff; padding: 6px; }')
                card_v = QVBoxLayout(card)
                card_v.setSpacing(3)

                hdr = QLabel(f'<b style="color:#7c3aed;">{crit_label}</b>')
                hdr.setTextFormat(Qt.TextFormat.RichText)
                card_v.addWidget(hdr)

                mt   = best_run.get('model_type', '—')
                name = best_run.get('model', '—')
                ds   = best_run.get('dataset', '—')
                ep   = best_run.get('epochs', '—')
                dev  = best_run.get('device', '—')

                details_html = (
                    f'<b>Μοντέλο:</b> {name} &nbsp;|&nbsp; '
                    f'<b>Τύπος:</b> {mt} &nbsp;|&nbsp; '
                    f'<b>Dataset:</b> {ds} &nbsp;|&nbsp; '
                    f'<b>Εποχές:</b> {ep} &nbsp;|&nbsp; '
                    f'<b>Device:</b> {dev}'
                )
                det_lbl = QLabel(details_html)
                det_lbl.setTextFormat(Qt.TextFormat.RichText)
                card_v.addWidget(det_lbl)

                # Key metrics for this run
                m_parts = []
                for mk, mfmt, mlbl in [
                    ('map50',    '{:.4f}', 'mAP50'),
                    ('map5095',  '{:.4f}', 'mAP50-95'),
                    ('precision','{:.4f}', 'Prec'),
                    ('recall',   '{:.4f}', 'Rec'),
                    ('acc_top1', '{:.4f}', 'Top-1'),
                    ('acc_top5', '{:.4f}', 'Top-5'),
                    ('val_loss', '{:.5f}', 'Val Loss'),
                    ('train_time_min', '{:.1f}', 'Χρόνος (min)'),
                ]:
                    v = best_run.get(mk)
                    if v is not None:
                        try:
                            m_parts.append(f'<b>{mlbl}:</b> {mfmt.format(float(v))}')
                        except Exception:
                            m_parts.append(f'<b>{mlbl}:</b> {v}')
                if m_parts:
                    metrics_lbl = QLabel(' &nbsp;·&nbsp; '.join(m_parts))
                    metrics_lbl.setTextFormat(Qt.TextFormat.RichText)
                    card_v.addWidget(metrics_lbl)

                exp_lbl = QLabel(f'<i style="color:#555;">{explanation}</i>')
                exp_lbl.setTextFormat(Qt.TextFormat.RichText)
                card_v.addWidget(exp_lbl)

                self.best_vbox.addWidget(card)
            except Exception:
                pass

        self.best_vbox.addStretch()
        # Ξανά-εφαρμογή wheel filters
        QTimer.singleShot(50, self._reapply_wheel_filters)

    # ══════════════════════════════════════════════════════════════════════
    #  CSV Export
    # ══════════════════════════════════════════════════════════════════════

    def _export_csv(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, 'Εξαγωγή CSV',
            str(Path.home() / 'training_comparison.csv'), 'CSV (*.csv)')
        if not path:
            return
        try:
            import csv as _csv
            headers = [c[0] for c in self._COLS]
            keys    = [c[1] for c in self._COLS]
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = _csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for r in self._all_runs:
                    row_out = {}
                    for h, k in zip(headers, keys):
                        v = r.get(k, '')
                        row_out[h] = '' if (v is None or v == '—') else v
                    w.writerow(row_out)
            QMessageBox.information(self, 'Εξαγωγή CSV', f'✅ Αποθηκεύτηκε:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Σφάλμα', str(e))


class DetectionPreviewDialog(QDialog):

    def __init__(self, parent: QWidget | None=None):
        super().__init__(parent)
        self.setWindowTitle('Προεπισκόπηση Ανάλυσης Dataset')
        self.setModal(False)
        self.resize(800, 600)
        self._samples: list[tuple[str, str, str, str]] = []
        self._current_index: int = -1
        self._on_close_callback = None
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)
        self.title_label = QLabel('Εικόνα 0: -')
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_title = self.title_label.font()
        font_title.setPointSize(font_title.pointSize() + 2)
        font_title.setBold(True)
        self.title_label.setFont(font_title)
        main_layout.addWidget(self.title_label)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(640, 480))
        self.image_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        main_layout.addWidget(self.image_label, 1)
        info_layout = QVBoxLayout()
        self.pred_label = QLabel('')
        self.pred_label.setWordWrap(True)
        self.truth_label = QLabel('')
        self.truth_label.setWordWrap(True)
        info_layout.addWidget(self.pred_label)
        info_layout.addWidget(self.truth_label)
        main_layout.addLayout(info_layout)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        self.prev_button = QPushButton('<')
        self.next_button = QPushButton('>')
        self.close_button = QPushButton('Έξοδος')
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.close_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.prev_button)
        buttons_layout.addWidget(self.next_button)
        buttons_layout.addSpacing(20)
        buttons_layout.addWidget(self.close_button)
        buttons_layout.addStretch(1)
        main_layout.addLayout(buttons_layout)
        self._update_buttons()

    def set_on_close_callback(self, callback):
        self._on_close_callback = callback

    def add_sample(self, image_path: str, title: str, pred_text: str, truth_text: str):
        self._samples.append((image_path, title, pred_text, truth_text))
        self._current_index = len(self._samples) - 1
        self._update_view()

    def _update_buttons(self):
        has_samples = len(self._samples) > 0
        self.prev_button.setEnabled(has_samples and self._current_index > 0)
        self.next_button.setEnabled(has_samples and self._current_index < len(self._samples) - 1)

    def _update_view(self):
        if not self._samples or self._current_index < 0 or self._current_index >= len(self._samples):
            self.title_label.setText('Εικόνα: -')
            self.image_label.clear()
            self.pred_label.setText('')
            self.truth_label.setText('')
            self._update_buttons()
            return
        image_path, title, pred_text, truth_text = self._samples[self._current_index]
        try:
            index_display = self._current_index + 1
        except Exception:
            index_display = 0
        if title:
            self.title_label.setText(f'Εικόνα {index_display}: {title}')
        else:
            self.title_label.setText(f'Εικόνα {index_display}:')
        self.pred_label.setText(pred_text)
        self.truth_label.setText(truth_text)
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.clear()
        self._update_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_view()

    def show_previous(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._update_view()

    def show_next(self):
        if self._current_index < len(self._samples) - 1:
            self._current_index += 1
            self._update_view()

    def closeEvent(self, event):
        if callable(self._on_close_callback):
            try:
                self._on_close_callback()
            except Exception:
                pass
        try:
            info = perform_smart_memory_cleanup('Κλείσιμο εφαρμογής – καθαρισμός μνήμης')
            print(f'[MEM] {info}')
        except Exception:
            pass
        super().closeEvent(event)
"""Diagnostics dialog (UI).
Παράθυρο διάγνωσης για dependencies, GPU/CUDA, paths, backends και συχνά σφάλματα.
"""
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QPlainTextEdit, QTextBrowser


class DiagnosticsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('🧪 Diagnostics – Models Manager Pro')
        self.setMinimumSize(980, 650)
        self.setModal(True)
        self._data = collect_diagnostics_data()
        self._text = diagnostics_to_text(self._data)
        self._json = diagnostics_to_json(self._data)
        self._html = diagnostics_to_html(self._data)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        title = QLabel('🧪 Diagnostics')
        title.setStyleSheet('font-size: 18px; font-weight: 800;')
        subtitle = QLabel('Όλες οι πληροφορίες συστήματος / περιβάλλοντος σε μία φόρμα, για εύκολο troubleshooting.')
        subtitle.setStyleSheet('color: #555; font-weight: 600;')
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)
        tabs = QTabWidget()
        html_view = QTextBrowser()
        html_view.setOpenExternalLinks(False)
        html_view.setHtml(self._html)
        tabs.addTab(html_view, '🧾 Αναφορά')
        txt_view = QPlainTextEdit()
        txt_view.setReadOnly(True)
        txt_view.setPlainText(self._text)
        txt_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        txt_view.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11.5px;")
        tabs.addTab(txt_view, '📄 Text')
        json_view = QPlainTextEdit()
        json_view.setReadOnly(True)
        json_view.setPlainText(self._json)
        json_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        json_view.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11.5px;")
        tabs.addTab(json_view, '🧩 JSON')
        root.addWidget(tabs)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        copy_btn = QPushButton('📋 Αντιγραφή')
        copy_btn.setToolTip('Αντιγράφει την plain-text αναφορά στο clipboard')
        copy_btn.clicked.connect(self._copy_to_clipboard)
        save_btn = QPushButton('💾 Αποθήκευση...')
        save_btn.setToolTip('Αποθήκευση αναφοράς ως .txt ή .json')
        save_btn.clicked.connect(self._save_report)
        zip_btn = QPushButton('📦 ZIP...')
        zip_btn.setToolTip('Δημιουργία diagnostics ZIP (για αποστολή/αρχειοθέτηση)')
        zip_btn.clicked.connect(self._save_zip)
        close_btn = QPushButton('✅ Κλείσιμο')
        close_btn.clicked.connect(self.accept)
        for b in (copy_btn, save_btn, zip_btn, close_btn):
            b.setMinimumHeight(34)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(zip_btn)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _copy_to_clipboard(self):
        try:
            QApplication.clipboard().setText(self._text)
            QMessageBox.information(self, '✅ Αντιγραφή', 'Η αναφορά αντιγράφηκε στο clipboard.')
        except Exception as e:
            QMessageBox.critical(self, '❌ Αντιγραφή', f'Αποτυχία αντιγραφής:\n{e}')

    def _save_report(self):
        try:
            default_dir = str(Path.home() / 'Desktop')
        except Exception:
            default_dir = ''
        path, selected = QFileDialog.getSaveFileName(self, 'Αποθήκευση Diagnostics', os.path.join(default_dir, 'Models_Manager_Pro_Diagnostics.txt'), 'Text (*.txt);JSON (*.json)')
        if not path:
            return
        try:
            if path.lower().endswith('.json') or 'JSON' in (selected or ''):
                if not path.lower().endswith('.json'):
                    path += '.json'
                Path(path).write_text(self._json, encoding='utf-8')
            else:
                if not path.lower().endswith('.txt'):
                    path += '.txt'
                Path(path).write_text(self._text, encoding='utf-8')
            QMessageBox.information(self, '✅ Αποθήκευση', f'Έτοιμο!\n{path}')
        except Exception as e:
            QMessageBox.critical(self, '❌ Αποθήκευση', f'Αποτυχία αποθήκευσης:\n{e}')

    def _save_zip(self):
        try:
            try:
                default_dir = str(Path.home() / 'Desktop')
            except Exception:
                default_dir = ''
            path, _ = QFileDialog.getSaveFileName(self, 'Αποθήκευση Diagnostics ZIP', os.path.join(default_dir, 'Models_Manager_Pro_Diagnostics.zip'), 'ZIP (*.zip)')
            if not path:
                return
            out = generate_diagnostics_zip(path)
            QMessageBox.information(self, '✅ ZIP', f'Έτοιμο!\n{out}')
        except Exception as e:
            QMessageBox.critical(self, '❌ ZIP', f'Αποτυχία δημιουργίας diagnostics ZIP:\n{e}')
"""Home / Dashboard tab.
Αρχική οθόνη με shortcuts, κατάσταση περιβάλλοντος και βασικές επιλογές.
"""


def _finish_tab_topbar(widget, top_bar_layout, outer_layout):
    top_bar_layout.addStretch()
    widget.dashboard_button, widget.theme_button = add_dashboard_and_theme_buttons( widget, top_bar_layout, widget.go_to_dashboard, widget.toggle_theme)
    outer_layout.addLayout(top_bar_layout)


def _make_tab_layout(widget):
    ol = QVBoxLayout(widget)
    ol.setContentsMargins(15, 15, 15, 15)
    ol.setSpacing(10)
    tbl = QHBoxLayout()
    tbl.setContentsMargins(0, 0, 0, 0)
    return ol, tbl


class HomeTab(QWidget, TabNavigationMixin):
    request_tab_change = Signal(int)

    def __init__(self):
        super().__init__()
        self.current_theme = 'light'
        self.resource_timer = None
        self.init_ui()

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        top_bar_layout.addStretch()
        self.theme_button = QPushButton('Light/Dark')
        self.theme_button.setObjectName('ThemeButton')
        self.theme_button.setFixedHeight(24)
        self.theme_button.clicked.connect(self.toggle_theme)
        top_bar_layout.addWidget(self.theme_button)
        outer_layout.addLayout(top_bar_layout)
        top_separator = add_blue_separator(outer_layout)
        header_frame = QFrame()
        header_frame.setObjectName('HeaderFrame')
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 4, 10, 4)
        header_layout.setSpacing(4)
        title_label = QLabel('🤖 Models Manager Pro (A.I Copilot) Ver 4.0')
        title_label.setObjectName('Title')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label = QLabel('Διαχείριση, Εκπαίδευση και Ανίχνευση με Μοντέλα Αναγνώρισης Αντικειμένων')
        subtitle_label.setObjectName('Subtitle')
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet('font-weight: 600; font-size: 18px;')
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        copyright_label = QLabel('(c) Ευάγγελος Πεφάνης 2026')
        copyright_label.setObjectName('CopyrightLabel')
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet('font-size: 17px; font-weight: 400;')
        header_layout.addWidget(copyright_label)
        outer_layout.addWidget(header_frame)
        card4 = QFrame()
        card4.setObjectName('ResourceCard')
        card4_layout = QVBoxLayout(card4)
        card4_title = QLabel('💻 Live Πόροι Συστήματος')
        card4_title.setObjectName('ResourceTitle')
        self.cpu_mem_label = QLabel('CPU: N/A | Μνήμη: N/A')
        self.gpu_label = QLabel('GPU: N/A (εάν διαθέσιμη)')
        self.cpu_mem_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.gpu_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.cpu_mem_label.setIndent(12)
        self.gpu_label.setIndent(12)
        self.cpu_mem_label.setWordWrap(True)
        self.gpu_label.setWordWrap(True)
        card4_layout.addWidget(card4_title)
        card4_layout.addWidget(self.cpu_mem_label)
        card4_layout.addWidget(self.gpu_label)
        diag_btn = QPushButton('🧪 Diagnostics')
        diag_btn.setToolTip('Άνοιγμα φόρμας Diagnostics με όλες τις πληροφορίες συστήματος/περιβάλλοντος')
        diag_btn.clicked.connect(self.show_diagnostics)
        card4_layout.addWidget(diag_btn)
        nav_group = QGroupBox('🧭 Πλοήγηση καρτελών')
        nav_group.setObjectName('NavigationGroup')
        nav_layout = QGridLayout(nav_group)
        nav_layout.setSpacing(8)
        btn_home = QPushButton('🏠 Πίνακας Ελέγχου')
        btn_train = QPushButton('🎓 Εκπαίδευση Μοντέλου')
        btn_copilot = QPushButton('🤖 A.I Copilot Εκπαίδευσης')
        btn_camera = QPushButton('📷 Live Ανίχνευση')
        btn_benchmark = QPushButton('⚡ Benchmark FPS')
        btn_cam_bench = QPushButton('🎥 Benchmark Κάμερας')
        btn_stats = QPushButton('📊 Στατιστικά Ανίχνευσης')
        btn_exit = QPushButton('⏻ Έξοδος Εφαρμογής')
        btn_home.clicked.connect(lambda: self.request_tab_change.emit(0))
        btn_train.clicked.connect(lambda: self.request_tab_change.emit(1))
        btn_copilot.clicked.connect(lambda: self.request_tab_change.emit(2))
        btn_camera.clicked.connect(lambda: self.request_tab_change.emit(3))
        btn_benchmark.clicked.connect(lambda: self.request_tab_change.emit(4))
        btn_cam_bench.clicked.connect(lambda: self.request_tab_change.emit(5))
        btn_stats.clicked.connect(lambda: self.request_tab_change.emit(6))
        btn_exit.clicked.connect(self.on_exit_app_clicked)
        for btn in [btn_home, btn_train, btn_copilot, btn_camera, btn_benchmark, btn_cam_bench, btn_stats, btn_exit]:
            btn.setMinimumWidth(180)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        nav_layout.addWidget(btn_home, 0, 0)
        nav_layout.addWidget(btn_train, 0, 1)
        nav_layout.addWidget(btn_copilot, 0, 2)
        nav_layout.addWidget(btn_camera, 0, 3)
        nav_layout.addWidget(btn_benchmark, 1, 0)
        nav_layout.addWidget(btn_cam_bench, 1, 1)
        nav_layout.addWidget(btn_stats, 1, 2)
        nav_layout.addWidget(btn_exit, 1, 3)
        outer_layout.addWidget(nav_group)
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)
        left_col = QVBoxLayout()
        left_col.setSpacing(15)
        self.features_group = QGroupBox('📑 Δυνατότητες Εφαρμογής')
        self.features_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.features_group.setObjectName('FeaturesGroup')
        features_layout = QVBoxLayout(self.features_group)
        features_scroll = QScrollArea()
        features_scroll.setWidgetResizable(True)
        features_scroll.setFrameShape(QFrame.Shape.NoFrame)
        features_inner = QWidget()
        features_inner.setObjectName('FeaturesInner')
        features_inner_layout = QVBoxLayout(features_inner)
        features_inner_layout.setContentsMargins(0, 0, 0, 0)
        features_inner_layout.setSpacing(8)
        def _feat(title, desc, layout):
            t = QLabel(title)
            t.setStyleSheet('font-weight: 700; font-size: 12.5pt;')
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setAlignment(Qt.AlignmentFlag.AlignJustify)
            layout.addWidget(t)
            layout.addWidget(d)
            layout.addSpacing(8)

        _feat('🚀 Γενική Επισκόπηση',
              '• Ενοποιημένο περιβάλλον για εκπαίδευση, αξιολόγηση, live ανίχνευση και ανάλυση YOLO / YOLO-CLS / CNN (torchvision) μοντέλων.\n'
              '• Πλήρης pipeline: από την εκπαίδευση και την εξαγωγή, μέχρι τη live χρήση, τη σύγκριση αποτελεσμάτων και την παραγωγή επαγγελματικών PDF αναφορών.\n'
              '• Υποστήριξη CPU / GPU, Triton / TorchInductor, A.I Copilot (Groq LLM), Light/Dark theme, πλήρη καταγραφή log και crash logs.',
              features_inner_layout)

        _feat('🏠 Πίνακας Ελέγχου',
              '• Live παρακολούθηση CPU, RAM και GPU με ανανέωση κάθε 2 δευτερόλεπτα.\n'
              '• Κάρτες κατάστασης για λειτουργικό σύστημα, εκδόσεις βιβλιοθηκών (Python, PyTorch, Ultralytics, TensorRT, Triton, PySide6) και διαθέσιμα backends.\n'
              '• Γρήγορες συντομεύσεις πλοήγησης σε όλα τα tabs, Diagnostics φόρμα και αλλαγή Light/Dark theme.',
              features_inner_layout)

        _feat('🎓 Εκπαίδευση Μοντέλου',
              '• Υποστήριξη YOLO detection (v5–v12), YOLO classification (-cls) και CNN torchvision (MobileNet V2/V3, ResNet-50/101) μοντέλων.\n'
              '• Φίλτρα κατηγορίας μοντέλου, αυτόματη σάρωση datasets, πλήρης ρύθμιση υπερπαραμέτρων (epochs, batch, image size, optimizer, LR, momentum, weight decay, workers, patience).\n'
              '• Υποστήριξη TorchInductor / Triton με 3 compile modes, αυτόματη ανίχνευση CPU/GPU, live progress bar & log, αυτόματη αποθήκευση best/final μοντέλου.\n'
              '• Εξαγωγή σε NCNN, ONNX, TensorRT (.engine) και CNN→ONNX (torchvision). Δημιουργία PDF training report με loss/accuracy curves ανά τύπο μοντέλου.',
              features_inner_layout)

        _feat('📊 Σύγκριση Εκπαιδεύσεων Μοντέλων',
              '• Πλήρης διάλογος σύγκρισης όλων των αποθηκευμένων εκπαιδεύσεων (YOLO detection, YOLO classification, CNN torchvision) σε ένα ενιαίο παράθυρο με 4 tabs.\n'
              '• Tab 📋 Πίνακας: αναλυτικός πίνακας με 18 στήλες (mAP50, mAP50-95, Precision, Recall, Top-1/5 Accuracy, Train/Val Loss, Optimizer, Batch, Χρόνος, Ημερομηνία κ.ά.) — χρωματισμός κελιών (πράσινο ≥0.85 / κεχριμπαρένιο ≥0.70 / κόκκινο <0.70), multi-selection γραμμών, φίλτρο αναζήτησης και εξαγωγή CSV.\n'
              '• Tab 📈 Γραφικές: 7 διαδραστικές γραφικές που ενημερώνονται από την επιλογή γραμμών — (1) mAP50/Top-1 Accuracy bar chart, (2) Precision & Recall grouped bars, (3) Train/Val Loss comparison bars, (4) Precision vs Recall scatter με labels, (5) Χρόνος εκπαίδευσης ανά μοντέλο, (6) Radar chart συνολικού προφίλ (6 axes: mAP50, mAP95, Precision, Recall, Top-1, Top-5), (7) Εποχές vs Καλύτερο Metric scatter.\n'
              '• Tab 📉 Καμπύλες Εκπαίδευσης: φόρτωση results.csv ανά μοντέλο από combo — αυτόματη ανίχνευση και σχεδίαση Loss curves, Accuracy/mAP per epoch, Learning Rate progression.\n'
              '• Tab 🏆 Καλύτερο Μοντέλο: αυτόματη σύσταση νικητή σε 7 κατηγορίες — (1) Υψηλότερο mAP50, (2) Υψηλότερο mAP50-95, (3) Υψηλότερο Top-1 Accuracy, (4) Καλύτερο F1 score (ισορροπία Precision/Recall), (5) Ταχύτερη εκπαίδευση, (6) Χαμηλότερο Val Loss, (7) Συνολική σταθμισμένη βαθμολογία (35% mAP/Top1 + 25% mAP95/Top5 + 20% Precision + 20% Recall). Κάθε σύσταση εμφανίζεται σε κάρτα με όλα τα metrics και εξήγηση του κριτηρίου.\n'
              '• Αυτόματη σάρωση δεδομένων από training_metrics.json, results.csv (Trained_Models/ και Runs_* φάκελοι) με deduplication, ταξινόμηση νεότερα πρώτα και αναγνώριση τύπου μοντέλου (CNN/YOLO-det/YOLO-cls).',
              features_inner_layout)

        _feat('🤖 A.I Copilot Εκπαίδευσης',
              '• Έξυπνος βοηθός εκπαίδευσης μέσω Groq API (LLM) με επιλογή μοντέλου και παραμετροποιήσιμο system prompt.\n'
              '• 3 λειτουργίες: (1) Πρόταση αρχικών ρυθμίσεων, (2) Βελτίωση βάσει τελευταίας εκπαίδευσης, (3) Βελτίωση βάσει Στατιστικών Ανίχνευσης.\n'
              '• Αυτόματη δημιουργία και εφαρμογή YAML config blocks (gui_config & train_hyperparams) στο Training Tab με σήμανση 🤖 στα τροποποιημένα πεδία.\n'
              '• Υποστήριξη streaming απόκρισης, ιστορικό συνομιλίας, ρυθμίσεις API key και επιλογή γλώσσας απάντησης.',
              features_inner_layout)

        _feat('📷 Live Ανίχνευση',
              '• Real-time ανίχνευση από κάμερα ή αρχείο βίντεο με μοντέλα PyTorch, ONNX, NCNN και TensorRT — για detection και classification.\n'
              '• Πλήρης υποστήριξη CNN torchvision μοντέλων με classification overlay (top-5 predictions με confidence bars), επιλογή κλάσεων και classes_filter.\n'
              '• Ρύθμιση confidence threshold, image size, backend, επιλογή κάμερας (index), TensorRT acceleration checkbox.\n'
              '• FPS overlay, live EMA inference time, auto-recover σε camera disconnect, αποθήκευση annotated frames/video.',
              features_inner_layout)

        _feat('🎬 Video Inference',
              '• Επεξεργασία βίντεο αρχείου με εκπαιδευμένο μοντέλο (PyTorch/ONNX/NCNN/TensorRT) — frame-by-frame inference με annotation.\n'
              '• Ρύθμιση confidence threshold, image size, επιλογή κλάσεων, εξαγωγή annotated βίντεο ή individual frames.\n'
              '• Progress bar, live preview, υποστήριξη mp4/avi/mov και άλλων μορφών.',
              features_inner_layout)

        _feat('⚡ Benchmark FPS',
              '• Αυτόματο benchmark PyTorch, ONNX, TensorRT, NCNN backends σε ασφαλές subprocess — αποφυγή native crash από DLL/driver conflicts.\n'
              '• Μετρήσεις FPS, ms/εικόνα, warmup/runs ρυθμιζόμενα, σύγκριση όλων των backends του ίδιου μοντέλου σε έναν πίνακα.\n'
              '• Ειδική διαχείριση CNN μοντέλων (torchvision) μέσω CNNInferenceHelper — αυτόματη ανίχνευση τύπου μοντέλου.',
              features_inner_layout)

        _feat('🎥 Benchmark Κάμερας',
              '• Πραγματική μέτρηση FPS από live κάμερα για κάθε διαθέσιμο backend (PyTorch/ONNX/NCNN/TensorRT/CNN).\n'
              '• Επιλογή image size, duration, conf/iou thresholds, αναλυτικό log ανά backend με EMA FPS, αποτελέσματα σε πίνακα σύγκρισης.\n'
              '• Υποστήριξη YOLO detection/classification και CNN torchvision μοντέλων με classes_filter.',
              features_inner_layout)

        _feat('📊 Στατιστικά Ανίχνευσης',
              '• Batch ανάλυση dataset εικόνων με επιλεγμένο μοντέλο (YOLO-det/cls, CNN torchvision, ONNX) — έως 500 εικόνες.\n'
              '• Per-class counts, mean confidence, κατανομή inference time, confidence histogram, preview annotated εικόνων με ground truth.\n'
              '• Εξαγωγή PDF Detection Report με γραφήματα (detections per class, inference time, confidence distribution) — διαφορετικό cover ανά τύπο μοντέλου.\n'
              '• Για CNN: εμφάνιση ΟΛΩΝ των κλάσεων (ακόμα και με 0 ανιχνεύσεις), chart με έγχρωμες μπάρες, preview για κάθε εικόνα ανεξάρτητα από confidence threshold.\n'
              '• Άμεση τροφοδότηση αποτελεσμάτων στο A.I Copilot για αυτόματη βελτίωση υπερπαραμέτρων εκπαίδευσης.',
              features_inner_layout)

        _feat('🧩 Διαχείριση Μοντέλων CNN (torchvision)',
              '• Εκπαίδευση MobileNet V2, MobileNet V3 Small/Large, ResNet-50, ResNet-101 με classification datasets (train/val/class_dirs).\n'
              '• Φόρτωση .pt checkpoint (PyTorch) ή .onnx (ONNX Runtime) — αυτόματη ανίχνευση τύπου και metadata από class_names.json / _onnx_meta.json.\n'
              '• Εξαγωγή CNN → ONNX μέσω torch.onnx.export (opset 12) με αυτόματη δημιουργία _onnx_meta.json για άμεση χρήση.\n'
              '• Classes filter στο Live Ανίχνευση, top-k predictions overlay, Benchmark, Στατιστικά Ανίχνευσης — πλήρης ενσωμάτωση σε όλο το pipeline.',
              features_inner_layout)

        _feat('📄 PDF Αναφορές & Logging',
              '• Αυτόματη δημιουργία PDF Training Report (Loss curves, Accuracy/mAP, Learning Rate, Ultralytics results.png) — ξεχωριστό format για YOLO-det, YOLO-cls, CNN.\n'
              '• PDF Detection Report με cover, πίνακα metrics, per-class πίνακα, charts confidence/time — ξεχωριστός τίτλος ανά τύπο μοντέλου.\n'
              '• Αρχεία log ανά εκτέλεση, crash logs με full traceback + thread dump, faulthandler για native crashes (SIGSEGV, DLL).',
              features_inner_layout)

        features_inner_layout.addStretch()
        features_scroll.setWidget(features_inner)
        features_layout.addWidget(features_scroll)
        left_col.addWidget(self.features_group, 1)
        info_layout.addLayout(left_col, 3)
        right_col = QVBoxLayout()
        right_col.setSpacing(15)
        sys_group = QGroupBox('💻 Περιβάλλον Συστήματος')
        sys_layout = QVBoxLayout(sys_group)
        import platform as _platform
        import sys as _sys
        os_name = f'{_platform.system()} {_platform.release()}'
        machine = _platform.machine() or 'Άγνωστο'
        cpu_count = os.cpu_count() or 0
        cpu_info_text = _platform.processor() or 'Άγνωστος επεξεργαστής'
        self.os_info_label = QLabel(f'🖥️ Λειτουργικό: {os_name} ({machine})')
        self.cpu_info_label = QLabel(f'🧩 CPU: {cpu_info_text} ({cpu_count} πυρήνες)')
        total_mem_gb = (psutil.virtual_memory().total / 1024 ** 3) if psutil is not None else 0.0
        self.ram_info_label = QLabel(f'📦 RAM: {total_mem_gb:.1f} GB')
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                self.gpu_info_static_label = QLabel(f'🎮 GPU: {gpu_name} ({gpu_total:.1f} GB)')
            else:
                self.gpu_info_static_label = QLabel('🎮 GPU: Μη διαθέσιμη')
        except Exception:
            self.gpu_info_static_label = QLabel('🎮 GPU: N/A')
        for lbl in (self.os_info_label, self.cpu_info_label, self.ram_info_label, self.gpu_info_static_label):
            lbl.setWordWrap(True)
            sys_layout.addWidget(lbl)
        right_col.addWidget(sys_group)
        right_col.addWidget(card4)
        self.env_group = QGroupBox('🧠 Περιβάλλον Κώδικα / Βιβλιοθηκών')
        self.env_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        env_layout = QVBoxLayout(self.env_group)
        python_ver = _sys.version.split()[0]
        torch_ver = getattr(torch, '__version__', 'Άγνωστη')
        cuda_ver = getattr(torch.version, 'cuda', None)
        cuda_text = cuda_ver if cuda_ver else 'Χωρίς CUDA (CPU only)'
        try:
            import ultralytics as _ul
            ul_version = getattr(_ul, '__version__', 'Άγνωστη')
        except Exception:
            ul_version = 'Μη διαθέσιμη'
        device_text = 'CUDA' if torch.cuda.is_available() else 'CPU'
        qt_ver = PYSIDE_VERSION
        try:
            import tensorrt as _trt
            trt_ver = getattr(_trt, '__version__', 'Άγνωστη')
            trt_text = trt_ver
        except Exception:
            trt_text = 'Μη διαθέσιμο'
        try:
            import triton as _triton
            triton_ver = getattr(_triton, '__version__', 'Άγνωστη')
            triton_text = triton_ver
        except Exception:
            triton_text = 'Μη διαθέσιμο'
        try:
            import torch._inductor as _inductor
            inductor_text = 'Διαθέσιμο (TorchInductor backend)'
        except Exception:
            inductor_text = 'Μη διαθέσιμο'
        self.python_label = QLabel(f'🐍 Python: {python_ver}')
        self.torch_label = QLabel(f'🔥 PyTorch: {torch_ver} (Συσκευή: {device_text}, CUDA: {cuda_text})')
        self.ultralytics_label = QLabel(f'🧪 Ultralytics: {ul_version}')
        self.tensorrt_label = QLabel(f'🔥 TensorRT: {trt_text}')
        self.triton_label = QLabel(f'📦 Triton: {triton_text}')
        self.inductor_label = QLabel(f'⚙️ TorchInductor: {inductor_text}')
        self.qt_label = QLabel(f'🪟 PySide6: {qt_ver}')
        for lbl in (self.python_label, self.torch_label, self.ultralytics_label, self.tensorrt_label, self.triton_label, self.inductor_label, self.qt_label):
            lbl.setWordWrap(True)
            env_layout.addWidget(lbl)
        right_col.addWidget(self.env_group)
        right_col.addStretch(1)
        info_layout.addLayout(right_col, 2)
        outer_layout.addLayout(info_layout, 1)
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resources)
        self.resource_timer.start(2000)
        self.update_resources()

    def on_exit_app_clicked(self):
        try:
            win = self.window()
            if win is not None:
                win.close()
        except Exception:
            pass
        from PySide6.QtWidgets import QApplication
        QApplication.quit()

    def update_resources(self) -> None:
        if psutil is not None:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_used = memory.used / 1024 ** 3
                memory_total = memory.total / 1024 ** 3
                self.cpu_mem_label.setText( f"CPU: {cpu_percent:.1f}% | " f"Μνήμη: {memory_used:.1f}/{memory_total:.1f} GB " f"({memory.percent:.1f}%)")
            except Exception as e:
                _MMPRO_LOGGER.debug("update_resources psutil error: %s", e)
                self.cpu_mem_label.setText("CPU: N/A | Μνήμη: N/A")
        else:
            self.cpu_mem_label.setText("CPU: N/A | Μνήμη: N/A (psutil μη εγκατεστημένο)")
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                gpu_name = _torch.cuda.get_device_name(0)
                gpu_total = _torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                gpu_used = _torch.cuda.memory_allocated(0) / 1024 ** 3
                self.gpu_label.setText( f"GPU: {gpu_name} — {gpu_used:.1f}/{gpu_total:.1f} GB")
            else:
                self.gpu_label.setText("GPU: Μη διαθέσιμη")
        except Exception as e:
            _MMPRO_LOGGER.debug("update_resources GPU error: %s", e)
            self.gpu_label.setText("GPU: N/A")

    def show_diagnostics(self):
        try:
            dlg = DiagnosticsDialog(parent=self)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, '❌ Diagnostics', f'Αποτυχία ανοίγματος φόρμας diagnostics:\n{e}')


class HomeTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('🏠 Πίνακας Ελέγχου - Standalone')
        self.setCentralWidget(HomeTab())
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def home_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = HomeTabWindow()
    win.show()
    sys.exit(app.exec())
"""Training tab (UI).
UI για ρύθμιση training, επιλογές dataset/model, και εκκίνηση training worker.
"""
try:
    from PySide6.QtCore import QProcess
except Exception:
    QProcess = None


class TrainingTab(QWidget, TabNavigationMixin):
    training_completed = Signal()

    def __init__(self):
        super().__init__()
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._progress_tick)
        self._progress_value = 0
        self._progress_cap = 95
        self._progress_running = False
        self.available_models = []
        self.training_thread = None
        self.training_worker = None
        self.training_process = None
        self._trainproc_out_buf = ''
        self._trainproc_err_buf = ''
        self._trainproc_last_error = None
        self.export_thread = None
        self.export_worker = None
        self.export_process = None
        self._exportproc_out_buf = ''
        self._exportproc_err_buf = ''
        self._exportproc_last_error = None
        self.progress_poll_timer = QTimer(self)
        self.progress_poll_timer.timeout.connect(self.poll_worker_progress)
        self.progress_poll_timer.setInterval(500)
        self.last_report_path = None
        self.use_triton = False
        self.compile_mode = 'Προεπιλογή'
        self.last_trained_model_name = None
        self.last_trained_dataset_name = None
        self.last_trained_imgsz = None
        self.last_trained_device = None
        self._training_had_error = False
        self.copilot_thread = None
        self.copilot_worker = None
        self.copilot_last_yaml = None
        self.copilot_last_hparams = None
        self._copilot_apply_in_progress = False
        self._copilot_label_map = {}
        self._copilot_label_base = {}
        self._copilot_icon = ' 🤖'
        self.init_ui()
        try:
            self.reset_default_hparams()
            self.refresh_models()
        except Exception:
            pass
        # Restore last-used settings
        try:
            s = _settings()
            if s.get('train_imgsz'):
                s.restore_combo(self.imgsz_combo, 'train_imgsz')
            if s.get('train_device'):
                s.restore_combo(self.device_combo, 'train_device')
            if s.get('train_epochs'):
                try:
                    self.epochs_spin.setValue(int(s.get('train_epochs')))
                except Exception:
                    pass
            if s.get('train_patience'):
                try:
                    self.patience_spin.setValue(int(s.get('train_patience')))
                except Exception:
                    pass
        except Exception:
            pass

    def find_available_models(self):
        models = []
        if MODELS_DIR_TRAINED_PT.exists():
            for d in sorted([d for d in MODELS_DIR_TRAINED_PT.iterdir() if d.is_dir()]):
                pt_candidates = sorted(d.glob('*.pt'))
                if not pt_candidates:
                    continue
                chosen_pt = None
                for p in pt_candidates:
                    if p.stem == d.name:
                        chosen_pt = p
                        break
                if chosen_pt is None:
                    chosen_pt = pt_candidates[0]
                # Detect CNN checkpoints via class_names.json or embedded model_name key
                model_type = 'PyTorch'
                try:
                    cj = d / 'class_names.json'
                    if cj.is_file():
                        import json as _j
                        _d = _j.loads(cj.read_text(encoding='utf-8'))
                        mn = str(_d.get('model_name', '')).lower()
                        if any(c in mn for c in _CNN_MODEL_KEYS):
                            model_type = 'CNN'
                except Exception:
                    pass
                if model_type == 'PyTorch':
                    stem_lower = chosen_pt.stem.lower()
                    if any(c in stem_lower for c in _CNN_MODEL_KEYS):
                        model_type = 'CNN'
                models.append((chosen_pt, model_type))

        def sort_key(item):
            path, mtype = item
            name_lower = path.name.lower()
            is_gpu = 'gpu' in name_lower
            return (0 if is_gpu else 1, name_lower)
        return sorted(models, key=sort_key)

    def refresh_models(self):
        previous_text = None
        previous_index = -1
        try:
            if hasattr(self, 'export_model_combo'):
                previous_index = self.export_model_combo.currentIndex()
                previous_text = self.export_model_combo.currentText()
        except Exception:
            previous_text = None
            previous_index = -1
        self.available_models = self.find_available_models()
        self.export_model_combo.clear()
        if not self.available_models:
            self.export_model_combo.addItems(['Δεν βρέθηκαν μοντέλα'])
            self.export_model_combo.setEnabled(False)
            self.export_button.setEnabled(False)
            self.export_onnx_button.setEnabled(False)
            if hasattr(self, 'export_tensorrt_button'):
                self.export_tensorrt_button.setEnabled(False)
            if hasattr(self, 'export_cnn_onnx_button'):
                self.export_cnn_onnx_button.setEnabled(False)
        else:
            display_names = [f'{path.name}' for path, _ in self.available_models]
            self.export_model_combo.addItems(display_names)
            self.export_model_combo.setEnabled(True)
            # export_button state θα οριστεί από on_export_model_changed (CNN check)
            target_index = 0
            if previous_text and previous_text in display_names:
                target_index = display_names.index(previous_text)
            elif 0 <= previous_index < len(display_names):
                target_index = previous_index
            try:
                self.export_model_combo.setCurrentIndex(target_index)
                self.on_export_model_changed(target_index)
            except Exception:
                pass

    def on_model_filter_checkbox_toggled(self, checked: bool):
        sender = self.sender()
        if not isinstance(sender, QCheckBox):
            self.update_model_list_for_task()
        if checked:
            for cb in (self.yolo_checkbox, self.cls_checkbox, self.cnn_checkbox):
                if cb is sender:
                    continue
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
        elif not (self.yolo_checkbox.isChecked() or self.cls_checkbox.isChecked() or self.cnn_checkbox.isChecked()):
            self.yolo_checkbox.blockSignals(True)
            self.yolo_checkbox.setChecked(True)
            self.yolo_checkbox.blockSignals(False)
        # Enable/disable Triton group – CNN doesn't support Triton
        try:
            is_cnn = getattr(self, 'cnn_checkbox', None) is not None and self.cnn_checkbox.isChecked()
            self.triton_checkbox.setEnabled(not is_cnn and self.device_combo.currentText() == 'GPU' and torch.cuda.is_available())
            if is_cnn:
                self.triton_checkbox.setChecked(False)
        except Exception:
            pass
        self.update_model_list_for_task()

    # ── Ενημέρωση λίστας μοντέλων βάσει επιλεγμένης κατηγορίας ──────────────────
    # Yolo (Detection) checkbox → TRAIN_MODELS (yolov5..yolo12)
    # Yolo (Classification) checkbox → TRAIN_CLS_MODELS (*-cls)
    # CNN (MobileNet/ResNet) checkbox → TRAIN_CNN_MODELS (mobilenet_v2, resnet50...)
    # Αυτόματη απενεργοποίηση Triton για CNN μοντέλα.
    def update_model_list_for_task(self):
        current_text = self.model_combo.currentText().strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        use_cls = False
        use_yolo = False
        use_cnn = False
        try:
            use_cls = self.cls_checkbox.isChecked()
        except Exception:
            use_cls = False
        try:
            use_yolo = self.yolo_checkbox.isChecked()
        except Exception:
            use_yolo = False
        try:
            use_cnn = getattr(self, 'cnn_checkbox', None) is not None and self.cnn_checkbox.isChecked()
        except Exception:
            use_cnn = False
        if use_cnn and not use_yolo and not use_cls:
            self.model_combo.addItems(TRAIN_CNN_MODELS)
        elif use_yolo and not use_cls:
            self.model_combo.addItems(TRAIN_MODELS)
        elif use_cls and not use_yolo:
            self.model_combo.addItems(TRAIN_CLS_MODELS)
        else:
            self.model_combo.addItems(TRAIN_MODELS)
        idx = self.model_combo.findText(current_text)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)
        self.update_dataset_list_for_task()

    # ── Ενημέρωση λίστας datasets βάσει τύπου μοντέλου ──────────────────────────
    # YOLO detection: εμφανίζει coco8, coco128, coco (TRAIN_DATASETS).
    # YOLO-cls / CNN: σαρώνει Data_Sets/ για φακέλους με δομή train/<class>/.
    # Αν δεν βρεθούν classification datasets, fallback στα YOLO datasets.
    def update_dataset_list_for_task(self):
        if not hasattr(self, 'dataset_combo'):
            return
        current_text = self.dataset_combo.currentText().strip()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        use_cls = False
        use_cnn = False
        try:
            use_cls = self.cls_checkbox.isChecked()
        except Exception:
            use_cls = False
        try:
            use_cnn = getattr(self, 'cnn_checkbox', None) is not None and self.cnn_checkbox.isChecked()
        except Exception:
            use_cnn = False
        if use_cls or use_cnn:
            cls_datasets = []
            try:
                if DATASETS_DIR.exists():
                    for d in DATASETS_DIR.iterdir():
                        if not d.is_dir():
                            continue
                        train_dir = d / 'train'
                        val_dir = d / 'val'
                        if not (train_dir.is_dir() and val_dir.is_dir()):
                            continue
                        has_class_subdir = any((ch.is_dir() for ch in train_dir.iterdir()))
                        if not has_class_subdir:
                            continue
                        cls_datasets.append(d.name)
            except Exception:
                cls_datasets = []
            cls_datasets = sorted(set(cls_datasets))
            if cls_datasets:
                self.dataset_combo.addItems(cls_datasets)
            else:
                self.dataset_combo.addItems(list(TRAIN_DATASETS.keys()))
        else:
            self.dataset_combo.addItems(list(TRAIN_DATASETS.keys()))
        idx = self.dataset_combo.findText(current_text)
        if idx >= 0:
            self.dataset_combo.setCurrentIndex(idx)
        elif self.dataset_combo.count() > 0:
            self.dataset_combo.setCurrentIndex(0)
        self.dataset_combo.blockSignals(False)

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        self.reset_hparams_button = QPushButton('Default Ρυθμίσεις')
        self.reset_hparams_button.setToolTip('Επαναφορά όλων των ρυθμίσεων του Tab Εκπαίδευση Μοντέλου στις προεπιλεγμένες τιμές της εφαρμογής.')
        self.reset_hparams_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.reset_hparams_button.clicked.connect(self.reset_default_hparams)
        self.yolo_checkbox = QCheckBox('Yolo (Detection)')
        self.yolo_checkbox.setToolTip('Εμφάνιση μόνο default μοντέλων ανίχνευσης YOLO (TRAIN_MODELS).')
        self.yolo_checkbox.setChecked(True)
        self.yolo_checkbox.toggled.connect(self.on_model_filter_checkbox_toggled)
        self.cls_checkbox = QCheckBox('Yolo (Classification)')
        self.cls_checkbox.setToolTip('Εμφάνιση μόνο μοντέλων ταξινόμησης (YOLO-Cls / classification).')
        self.cls_checkbox.setChecked(False)
        self.cls_checkbox.toggled.connect(self.on_model_filter_checkbox_toggled)
        self.cnn_checkbox = QCheckBox('CNN (MobileNet/ResNet)')
        self.cnn_checkbox.setToolTip(
            'Εμφάνιση CNN classifier μοντέλων (torchvision):\n'
            '  • MobileNet V2\n  • MobileNet V3 Small\n  • MobileNet V3 Large\n'
            '  • ResNet-50\n  • ResNet-101\n\n'
            'Απαιτεί dataset ταξινόμησης (train/<class>/... val/<class>/...).\n'
            'Χρησιμοποιεί PyTorch+torchvision, όχι Ultralytics.')
        self.cnn_checkbox.setChecked(False)
        self.cnn_checkbox.toggled.connect(self.on_model_filter_checkbox_toggled)
        self.model_categories_group = QGroupBox('Κατηγορίες μοντέλων')
        model_categories_layout = QHBoxLayout(self.model_categories_group)
        model_categories_layout.setContentsMargins(8, 4, 8, 4)
        model_categories_layout.setSpacing(10)
        model_categories_layout.addWidget(self.yolo_checkbox)
        model_categories_layout.addWidget(self.cls_checkbox)
        model_categories_layout.addWidget(self.cnn_checkbox)
        self.triton_group = QGroupBox('TorchInductor / Triton')
        self._triton_group_base_title = self.triton_group.title()
        triton_layout = QHBoxLayout(self.triton_group)
        self.triton_checkbox = QCheckBox('🚀 Triton')
        self.triton_checkbox.setToolTip('Ενεργοποίηση TorchInductor/Triton για επιτάχυνση (μόνο GPU)')
        self.triton_checkbox.stateChanged.connect(self.on_triton_changed)
        self.triton_checkbox.setEnabled(False)
        triton_layout.addWidget(self.triton_checkbox)
        self.compile_mode_combo = QComboBox()
        self.compile_mode_combo.addItems(['Προεπιλογή', 'Μείωση επιβάρυνσης', 'Μέγιστος αυτόματος συντονισμός'])
        self.compile_mode_combo.setCurrentText('Προεπιλογή')
        self.compile_mode_combo.currentTextChanged.connect(self.on_compile_mode_changed)
        triton_layout.addWidget(self.compile_mode_combo)
        top_bar_layout.addWidget(self.triton_group)
        top_bar_layout.addWidget(self.reset_hparams_button)
        top_bar_layout.addWidget(self.model_categories_group)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        main_layout = QHBoxLayout()
        training_layout = QVBoxLayout()
        training_layout.setSpacing(15)
        form_layout = QFormLayout()
        try:
            hspace = self.fontMetrics().horizontalAdvance('   ')
        except Exception:
            hspace = 24
        form_layout.setHorizontalSpacing(hspace)
        form_layout.setVerticalSpacing(8)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.model_label = QLabel('Μοντέλο:')
        self.model_combo = QComboBox()
        self.model_combo.setObjectName('TrainModelCombo')
        model_row_widget = QWidget()
        model_row_layout = QHBoxLayout(model_row_widget)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(6)
        model_row_layout.addWidget(self.model_combo)
        form_layout.addRow(self.model_label, model_row_widget)
        self.update_model_list_for_task()
        self.dataset_combo = QComboBox()
        self.dataset_label = QLabel('Dataset:')
        form_layout.addRow(self.dataset_label, self.dataset_combo)
        self.update_dataset_list_for_task()
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems([str(s) for s in TRAIN_IMAGE_SIZES])
        self.imgsz_combo.setCurrentText('640')
        self.imgsz_label = QLabel('Image Size:')
        form_layout.addRow(self.imgsz_label, self.imgsz_combo)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU'])
        self.device_combo.setCurrentText('GPU' if torch.cuda.is_available() else 'CPU')
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        self.device_label = QLabel('Συσκευή:')
        form_layout.addRow(self.device_label, self.device_combo)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        self.epochs_label = QLabel('Εποχές:')
        form_layout.addRow(self.epochs_label, self.epochs_spin)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(5)
        self.patience_label = QLabel('Patience:')
        form_layout.addRow(self.patience_label, self.patience_spin)
        training_layout.addLayout(form_layout)
        self.start_button = QPushButton('🚀 Έναρξη Εκπαίδευσης')
        self.start_button.clicked.connect(self.start_training)
        training_layout.addWidget(self.start_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        try:
            self.progress_bar.setStyleSheet('QProgressBar{color:white;}')
        except Exception:
            pass
        training_layout.addWidget(self.progress_bar)
        self.view_report_button = QPushButton('📄 Προβολή Αναφοράς')
        self.view_report_button.clicked.connect(self.view_report)
        self.view_report_button.setEnabled(False)
        training_layout.addWidget(self.view_report_button)
        self.manage_reports_button = QPushButton('📂 Διαχείριση Αναφορών Εκπαίδευσης')
        self.manage_reports_button.clicked.connect(self.manage_reports)
        training_layout.addWidget(self.manage_reports_button)
        self.compare_runs_button = QPushButton('📊 Σύγκριση Εκπαιδεύσεων')
        self.compare_runs_button.setToolTip('Άνοιγμα διαλόγου σύγκρισης αποτελεσμάτων εκπαίδευσης (mAP, Precision, Recall κ.λπ.)')
        self.compare_runs_button.clicked.connect(self._open_compare_dialog)
        training_layout.addWidget(self.compare_runs_button)
        training_layout.addSpacing(20)
        export_group = QGroupBox('Εκπαιδευμένα Μοντέλα (Επιλογή):')
        export_form_layout = QFormLayout(export_group)
        try:
            hspace_exp = self.fontMetrics().horizontalAdvance('   ')
        except Exception:
            hspace_exp = 24
        export_form_layout.setHorizontalSpacing(hspace_exp)
        export_form_layout.setVerticalSpacing(8)
        export_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        export_form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.export_model_combo = QComboBox()
        self.export_model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.export_model_combo.currentIndexChanged.connect(self.on_export_model_changed)
        refresh_button = QPushButton('🔄')
        refresh_button.setObjectName('RefreshButton')
        refresh_button.setFixedWidth(32)
        refresh_button.setFont(QFont('Segoe UI Emoji'))
        refresh_button.clicked.connect(self.refresh_models)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.export_model_combo)
        model_layout.addWidget(refresh_button)
        export_form_layout.addRow(QLabel('PyTorch:'), model_layout)
        self.export_imgsz_combo = QComboBox()
        self.export_imgsz_combo.addItems([str(s) for s in TRAIN_IMAGE_SIZES])
        self.export_imgsz_combo.setCurrentText('640')
        self.export_imgsz_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        export_form_layout.addRow(QLabel('Image Size:'), self.export_imgsz_combo)
        try:
            self._no_interact_filter = getattr(self, '_no_interact_filter', _NoInteractFilter())
            self.export_imgsz_combo.installEventFilter(self._no_interact_filter)
            self.export_imgsz_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.export_imgsz_combo.setStyleSheet('QComboBox::drop-down { width:0px; border:0; } QComboBox::down-arrow { image: none; }')
        except Exception:
            pass
        training_layout.addWidget(export_group)
        self.export_button = QPushButton('Εξαγωγή PyTorch σε → NCNN')
        self.export_button.clicked.connect(lambda: self.start_export('ncnn'))
        training_layout.addWidget(self.export_button)
        self.export_onnx_button = QPushButton('Εξαγωγή PyTorch σε → ONNX')
        self.export_onnx_button.clicked.connect(lambda: self.start_export('onnx'))
        self.export_onnx_button.setEnabled(False)
        training_layout.addWidget(self.export_onnx_button)
        self.export_tensorrt_button = QPushButton('Εξαγωγή PyTorch σε → 🔥TensorRT Engine')
        self.export_tensorrt_button.clicked.connect(lambda: self.start_export('tensorrt'))
        self.export_tensorrt_button.setEnabled(False)
        training_layout.addWidget(self.export_tensorrt_button)
        self.export_cnn_onnx_button = QPushButton('Εξαγωγή CNN → ONNX  (.onnx)')
        self.export_cnn_onnx_button.setToolTip(
            'Εξαγωγή CNN torchvision μοντέλου (.pt) σε ONNX μέσω torch.onnx.export.\n'
            'Διαθέσιμο μόνο για CNN μοντέλα (MobileNet / ResNet).\n'
            'Παράγει: <όνομα>.onnx + <όνομα>_onnx_meta.json (class names, imgsz κ.λπ.)')
        self.export_cnn_onnx_button.clicked.connect(lambda: self.start_export('cnn_onnx'))
        self.export_cnn_onnx_button.setEnabled(False)
        training_layout.addWidget(self.export_cnn_onnx_button)
        training_layout.addStretch()
        training_widget = QWidget()
        training_widget.setLayout(training_layout)
        training_widget.setMaximumWidth(450)
        log_layout = QVBoxLayout()
        hparams_group = QGroupBox('Υπερ-παράμετροι Εκπαίδευσης')
        hparams_group.setToolTip('Ρυθμίσεις που επηρεάζουν τον τρόπο που εκπαιδεύεται το μοντέλο YOLO (ρυθμός μάθησης, batch size, optimizer κ.λπ.).')
        hparams_layout = QGridLayout(hparams_group)
        try:
            hspace = self.fontMetrics().horizontalAdvance('   ')
        except Exception:
            hspace = 24
        hparams_layout.setHorizontalSpacing(hspace)
        hparams_layout.setVerticalSpacing(8)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip('Batch size: Πόσες εικόνες περνούν ταυτόχρονα από το μοντέλο σε κάθε βήμα της εκπαίδευσης. Μεγαλύτερο batch θέλει περισσότερη μνήμη GPU αλλά δίνει πιο σταθερό gradient.')
        self.batch_label = QLabel('Batch size:')
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['auto', 'SGD', 'Adam', 'AdamW'])
        self.optimizer_combo.setCurrentText('auto')
        self.optimizer_combo.setToolTip("Optimizer: Αλγόριθμος ενημέρωσης των βαρών (SGD, Adam, AdamW). Επηρεάζει την ταχύτητα και τη σταθερότητα σύγκλισης. Η επιλογή 'auto' αφήνει στο YOLO την προεπιλεγμένη τιμή.")
        self.optimizer_combo.currentTextChanged.connect(self.on_optimizer_changed)
        self.optimizer_label = QLabel('Optimizer:')
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        default_workers = 0 if getattr(sys, 'frozen', False) else 2
        self.workers_spin.setValue(default_workers)
        self.workers_spin.setToolTip('Workers: Πλήθος worker διεργασιών για το DataLoader (0–16).\n• Προτεινόμενο για Installer/Exe (Windows): 0 (σταθερότητα)\n• Σε Python (dev) έκδοση μπορείς να βάλεις 2–8 ανάλογα με CPU.')
        self.workers_label = QLabel('Workers:')
        hparam_labels = [self.batch_label, self.optimizer_label, self.workers_label]
        for lbl in hparam_labels:
            sp = lbl.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
            lbl.setSizePolicy(sp)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hparams_layout.addWidget(self.batch_label, 0, 0)
        hparams_layout.addWidget(self.batch_spin, 0, 1)
        hparams_layout.addWidget(self.optimizer_label, 0, 2)
        hparams_layout.addWidget(self.optimizer_combo, 0, 3)
        hparams_layout.addWidget(self.workers_label, 0, 4)
        hparams_layout.addWidget(self.workers_spin, 0, 5)
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(6)
        self.lr0_spin.setRange(1e-06, 1.0)
        self.lr0_spin.setSingleStep(0.0005)
        self.lr0_spin.setValue(0.01)
        self.lr0_spin.setToolTip('Initial LR (lr0): Αρχικός ρυθμός μάθησης. Μεγαλύτερη τιμή οδηγεί σε πιο γρήγορη αλλά πιθανόν πιο ασταθή εκπαίδευση. Πολύ μικρή τιμή μπορεί να κάνει την εκπαίδευση πολύ αργή.')
        self.lr0_label = QLabel('Initial LR (lr0):')
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setDecimals(6)
        self.lrf_spin.setRange(1e-06, 1.0)
        self.lrf_spin.setSingleStep(0.0005)
        self.lrf_spin.setValue(0.01)
        self.lrf_spin.setToolTip('Final LR ratio (lrf): Λόγος του τελικού ρυθμού μάθησης ως ποσοστό του αρχικού lr0 στο τέλος της εκπαίδευσης. Μικρό lrf σημαίνει ότι ο ρυθμός μάθησης μειώνεται πολύ.')
        self.lrf_label = QLabel('Final LR ratio (lrf):')
        hparams_layout.addWidget(self.lr0_label, 1, 0)
        hparams_layout.addWidget(self.lr0_spin, 1, 1)
        hparams_layout.addWidget(self.lrf_label, 1, 2)
        hparams_layout.addWidget(self.lrf_spin, 1, 3)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setRange(0.0, 0.999)
        self.momentum_spin.setSingleStep(0.001)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setToolTip('Momentum: Παράμετρος αδράνειας για τον optimizer. Βοηθά να “φιλτράρονται” οι απότομες αλλαγές στο gradient και να κινείται η εκπαίδευση πιο ομαλά.')
        self.momentum_label = QLabel('Momentum:')
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setToolTip('Weight decay: Παράμετρος L2 regularization. Μεγαλύτερη τιμή τείνει να μειώνει το overfitting αλλά μπορεί να μειώσει και την τελική απόδοση αν είναι υπερβολικά μεγάλη.')
        self.weight_decay_label = QLabel('Weight decay:')
        self.warmup_epochs_spin = QSpinBox()
        self.warmup_epochs_spin.setRange(0, 50)
        self.warmup_epochs_spin.setSingleStep(1)
        self.warmup_epochs_spin.setValue(3)
        self.warmup_epochs_spin.setToolTip('Warmup epochs: Πόσες εποχές χρησιμοποιούνται για warmup του ρυθμού μάθησης. Ο LR ξεκινά πολύ μικρός και αυξάνεται σταδιακά μέχρι το lr0, για πιο σταθερή έναρξη της εκπαίδευσης.')
        self.warmup_label = QLabel('Warmup epochs:')
        hparams_layout.addWidget(self.momentum_label, 2, 0)
        hparams_layout.addWidget(self.momentum_spin, 2, 1)
        hparams_layout.addWidget(self.weight_decay_label, 2, 2)
        hparams_layout.addWidget(self.weight_decay_spin, 2, 3)
        hparams_layout.addWidget(self.warmup_label, 2, 4)
        hparams_layout.addWidget(self.warmup_epochs_spin, 2, 5)
        for w in [self.batch_spin, self.optimizer_combo, getattr(self, 'workers_spin', None), self.lr0_spin, self.lrf_spin, self.momentum_spin, self.weight_decay_spin, self.warmup_epochs_spin]:
            if w is not None:
                try:
                    w.setMinimumWidth(110)
                    w.setMaximumWidth(140)
                except Exception:
                    pass
        try:
            self.model_combo.currentTextChanged.connect(lambda _text: self.on_user_manual_change('model_name'))
            self.dataset_combo.currentTextChanged.connect(lambda _text: self.on_user_manual_change('dataset_name'))
            self.imgsz_combo.currentTextChanged.connect(lambda _text: self.on_user_manual_change('image_size'))
            self.device_combo.currentTextChanged.connect(lambda _text: self.on_user_manual_change('device'))
            self.epochs_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('epochs'))
            self.patience_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('patience'))
            self.batch_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('batch'))
            self.optimizer_combo.currentTextChanged.connect(lambda _text: self.on_user_manual_change('optimizer'))
            self.workers_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('workers'))
            self.lr0_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('lr0'))
            self.lrf_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('lrf'))
            self.momentum_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('momentum'))
            self.weight_decay_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('weight_decay'))
            self.warmup_epochs_spin.valueChanged.connect(lambda _val: self.on_user_manual_change('warmup_epochs'))
        except Exception:
            pass
        self._init_copilot_label_map()
        log_layout.addWidget(hparams_group)
        log_layout.addWidget(QLabel('Log Εκπαίδευσης και Εξαγωγής:'))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName('LogOutput')
        self.log_output.setMinimumHeight(260)
        self.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mono_font = QFont('Consolas')
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        mono_font.setPointSize(11)
        self.log_output.setFont(mono_font)
        self.log_output.setWordWrapMode(QTextOption.NoWrap)
        log_layout.addWidget(self.log_output)
        main_layout.addWidget(training_widget)
        main_layout.addLayout(log_layout)
        outer_layout.addLayout(main_layout, 1)
        try:
            if hasattr(self, 'imgsz_spin'):
                self.imgsz_spin.setReadOnly(True)
                self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        except Exception:
            pass
        try:
            if hasattr(self, 'models_combo'):
                self.models_combo.currentTextChanged.connect(self.on_model_changed)
        except Exception:
            pass
        self.refresh_models()
        self.on_device_changed(self.device_combo.currentText())

    def on_device_changed(self, device_text):
        is_cnn_mode = getattr(self, 'cnn_checkbox', None) is not None and self.cnn_checkbox.isChecked()
        if device_text == 'GPU' and torch.cuda.is_available() and not is_cnn_mode:
            self.triton_checkbox.setEnabled(True)
        else:
            self.triton_checkbox.setEnabled(False)
            self.triton_checkbox.setChecked(False)

    def on_triton_changed(self, state):
        self.use_triton = state == Qt.CheckState.Checked.value
        if getattr(self, '_copilot_apply_in_progress', False):
            return
        try:
            self.set_triton_copilot_mark(False)
        except Exception:
            pass

    def on_compile_mode_changed(self, text):
        self.compile_mode = text
        if getattr(self, '_copilot_apply_in_progress', False):
            return
        try:
            self.set_triton_copilot_mark(False)
        except Exception:
            pass

    def _is_gpu_trained_model(self, model_path: 'Path') -> bool:
        try:
            name = str(getattr(model_path, 'name', model_path)).lower()
        except Exception:
            return False
        try:
            m = re.search(r'_(cpu|gpu)_', name, re.IGNORECASE)
            if m:
                return m.group(1).lower() == 'gpu'
        except Exception:
            pass
        if 'cpu' in name:
            return False
        return ('gpu' in name) or ('cuda' in name)

    def on_export_model_changed(self, index):
        if index >= 0 and index < len(self.available_models):
            model_path, model_type = self.available_models[index]
            try:
                main_window = self.window()
                if hasattr(main_window, 'sync_selected_trained_model'):
                    main_window.sync_selected_trained_model(model_path, model_type)
            except Exception:
                pass
            is_cnn = (model_type == 'CNN') or any(c in model_path.stem.lower() for c in _CNN_MODEL_KEYS)
            is_gpu_trained = (not is_cnn) and (model_type == 'PyTorch') and self._is_gpu_trained_model(model_path)
            # ── CNN μοντέλα: δεν υποστηρίζουν καμία εξαγωγή ─────────────────
            # ONNX/TensorRT/NCNN export δεν ισχύουν για CNN torchvision μοντέλα.
            onnx_ok = bool(is_gpu_trained) and not is_cnn
            self.export_onnx_button.setEnabled(onnx_ok)
            if hasattr(self, 'export_onnx_button'):
                if is_cnn:
                    self.export_onnx_button.setToolTip('⚠️ Export δεν υποστηρίζεται για CNN (torchvision) μοντέλα.')
                elif is_gpu_trained:
                    self.export_onnx_button.setToolTip('✅ ONNX export διαθέσιμο.')
                else:
                    self.export_onnx_button.setToolTip('⚠️ ONNX export: απαιτείται μοντέλο εκπαιδευμένο σε GPU.')
            if hasattr(self, 'export_tensorrt_button'):
                # TensorRT not supported for CNN torchvision models
                self.export_tensorrt_button.setEnabled(bool(is_gpu_trained) and not is_cnn)
                try:
                    if is_cnn:
                        self.export_tensorrt_button.setToolTip('⚠️ Export δεν υποστηρίζεται για CNN (torchvision) μοντέλα.')
                    elif is_gpu_trained:
                        self.export_tensorrt_button.setToolTip('✅ Διαθέσιμο: το μοντέλο είναι εκπαιδευμένο σε GPU.')
                    else:
                        self.export_tensorrt_button.setToolTip('⚠️ Μη διαθέσιμο: TensorRT export επιτρέπεται μόνο για μοντέλα εκπαιδευμένα σε GPU.')
                except Exception:
                    pass
            # NCNN export: not supported for CNN models
            if hasattr(self, 'export_button'):
                self.export_button.setEnabled(not is_cnn and model_type == 'PyTorch')
                if is_cnn:
                    self.export_button.setToolTip('⚠️ NCNN export δεν υποστηρίζεται για CNN (torchvision) μοντέλα.')
                else:
                    self.export_button.setToolTip('')
            # CNN → ONNX export: διαθέσιμο μόνο για CNN .pt μοντέλα
            if hasattr(self, 'export_cnn_onnx_button'):
                cnn_pt = is_cnn and (model_path.suffix.lower() == '.pt')
                self.export_cnn_onnx_button.setEnabled(cnn_pt)
                if cnn_pt:
                    self.export_cnn_onnx_button.setToolTip(
                        '✅ Εξαγωγή CNN → ONNX διαθέσιμη.\n'
                        'Παράγει: .onnx + _onnx_meta.json (class names, imgsz).')
                else:
                    self.export_cnn_onnx_button.setToolTip(
                        '⚠️ Διαθέσιμο μόνο για CNN (.pt) μοντέλα.')
            stem = model_path.stem
            found = None
            m = re.search(r'imgsz(\d+)', model_path.name, re.IGNORECASE)
            if m:
                found = m.group(1)
            if not found:
                m2 = re.search(r'_(\d+)$', stem)
                if m2:
                    found = m2.group(1)
            if hasattr(self, 'export_imgsz_combo'):
                if found and self.export_imgsz_combo.findText(found) != -1:
                    self.export_imgsz_combo.setCurrentText(found)
                self.export_imgsz_combo.setEnabled(True)
        else:
            self.export_onnx_button.setEnabled(False)
            if hasattr(self, 'export_tensorrt_button'):
                self.export_tensorrt_button.setEnabled(False)
            if hasattr(self, 'export_imgsz_combo'):
                self.export_imgsz_combo.setEnabled(True)

    def on_optimizer_changed(self, text: str):
        import platform
        optimizer = (text or '').strip()
        defaults = {'auto': {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0}, 'SGD': {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0}, 'Adam': {'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005, 'warmup_epochs': 3.0}, 'AdamW': {'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9, 'weight_decay': 0.01, 'warmup_epochs': 3.0}}
        cfg = dict(defaults.get(optimizer, defaults['auto']))
        try:
            device_text = self.device_combo.currentText().upper()
        except Exception:
            device_text = 'CPU'
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        gpu_name = ''
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = ''
        os_name = platform.system()
        workers_val = 2
        try:
            if hasattr(self, 'workers_spin'):
                workers_val = int(self.workers_spin.value())
        except Exception:
            workers_val = 2
        effective_workers = max(workers_val, 1)
        if device_text == 'GPU' and cuda_available:
            upper_gpu = gpu_name.upper()
            is_modern_gpu = any((tag in upper_gpu for tag in ['RTX 4', 'RTX 3', 'RTX 2', 'RTX 40', 'RTX 30', 'RTX 20']))
            if optimizer in ('SGD', 'auto'):
                if is_modern_gpu:
                    cfg['lr0'] = min(cfg['lr0'] * 1.3, 0.02)
                else:
                    cfg['lr0'] = min(cfg['lr0'] * 1.15, 0.015)
            elif optimizer in ('Adam', 'AdamW'):
                if is_modern_gpu:
                    cfg['lr0'] = min(cfg['lr0'] * 1.4, 0.005)
                else:
                    cfg['lr0'] = min(cfg['lr0'] * 1.2, 0.003)
            cfg['warmup_epochs'] = max(2.0, cfg['warmup_epochs'] - 0.5)
        else:
            if optimizer in ('SGD', 'auto'):
                cfg['lr0'] = min(cfg['lr0'], 0.005)
            else:
                cfg['lr0'] = min(cfg['lr0'], 0.0015)
            cfg['warmup_epochs'] = max(cfg['warmup_epochs'], 4.0)
            cfg['momentum'] = min(cfg['momentum'], 0.93)
        if effective_workers >= 4:
            cfg['lr0'] = min(cfg['lr0'] * 1.1, 0.02)
            cfg['warmup_epochs'] = max(1.5, cfg['warmup_epochs'] - 0.25)
        elif effective_workers == 1:
            cfg['lr0'] = max(cfg['lr0'] * 0.9, 1e-05)
            cfg['warmup_epochs'] = max(cfg['warmup_epochs'], 4.5)
        if os_name == 'Windows':
            cfg['lrf'] = min(max(cfg['lrf'], 0.005), 0.02)
        else:
            cfg['lrf'] = min(max(cfg['lrf'], 0.005), 0.05)
        try:
            if hasattr(self, 'lr0_spin'):
                self.lr0_spin.setValue(float(cfg['lr0']))
        except Exception:
            pass
        try:
            if hasattr(self, 'lrf_spin'):
                self.lrf_spin.setValue(float(cfg['lrf']))
        except Exception:
            pass
        try:
            if hasattr(self, 'momentum_spin'):
                self.momentum_spin.setValue(float(cfg['momentum']))
        except Exception:
            pass
        try:
            if hasattr(self, 'weight_decay_spin'):
                self.weight_decay_spin.setValue(float(cfg['weight_decay']))
        except Exception:
            pass
        try:
            if hasattr(self, 'warmup_epochs_spin'):
                try:
                    self.warmup_epochs_spin.setValue(int(cfg['warmup_epochs']))
                except Exception:
                    self.warmup_epochs_spin.setValue(int(float(cfg['warmup_epochs'])))
        except Exception:
            pass
        try:
            if hasattr(self, 'batch_spin'):
                batch = self.batch_spin.value()
                if 1 <= batch <= 64:
                    if device_text == 'GPU' and cuda_available:
                        if effective_workers >= 4:
                            suggested = min(max(batch, 16), 64)
                        else:
                            suggested = min(max(batch, 12), 48)
                    else:
                        suggested = min(batch, 16)
                    self.batch_spin.setValue(int(suggested))
        except Exception:
            pass
        try:
            self.clear_copilot_marks()
        except Exception:
            pass

    def _init_copilot_label_map(self):
        try:
            self._copilot_label_map = {'model_name': getattr(self, 'model_label', None), 'dataset_name': getattr(self, 'dataset_label', None), 'image_size': getattr(self, 'imgsz_label', None), 'device': getattr(self, 'device_label', None), 'epochs': getattr(self, 'epochs_label', None), 'patience': getattr(self, 'patience_label', None), 'batch': getattr(self, 'batch_label', None), 'optimizer': getattr(self, 'optimizer_label', None), 'workers': getattr(self, 'workers_label', None), 'lr0': getattr(self, 'lr0_label', None), 'lrf': getattr(self, 'lrf_label', None), 'momentum': getattr(self, 'momentum_label', None), 'weight_decay': getattr(self, 'weight_decay_label', None), 'warmup_epochs': getattr(self, 'warmup_label', None)}
            self._copilot_label_base = {}
            for key, lbl in self._copilot_label_map.items():
                if lbl is not None:
                    self._copilot_label_base[key] = lbl.text()
        except Exception:
            self._copilot_label_map = {}
            self._copilot_label_base = {}

    def mark_field_copilot(self, key: str, enabled: bool=True):
        lbl = self._copilot_label_map.get(key) if hasattr(self, '_copilot_label_map') else None
        base = self._copilot_label_base.get(key) if hasattr(self, '_copilot_label_base') else None
        if not lbl or not base:
            return
        try:
            if enabled:
                if self._copilot_icon.strip() not in lbl.text():
                    lbl.setText(base + self._copilot_icon)
            else:
                lbl.setText(base)
        except Exception:
            pass

    def clear_copilot_marks(self):
        if hasattr(self, '_copilot_label_map'):
            for key in list(self._copilot_label_map.keys()):
                self.mark_field_copilot(key, enabled=False)
        try:
            self.set_triton_copilot_mark(False)
        except Exception:
            pass

    def set_triton_copilot_mark(self, enabled: bool=True):
        try:
            if not hasattr(self, 'triton_group'):
                return
            base = getattr(self, '_triton_group_base_title', self.triton_group.title())
            self._triton_group_base_title = base
            icon = getattr(self, '_copilot_icon', ' 🤖')
            current_title = self.triton_group.title()
            if enabled:
                if icon.strip() not in current_title:
                    self.triton_group.setTitle(base + icon)
            else:
                self.triton_group.setTitle(base)
        except Exception:
            pass

    def on_user_manual_change(self, key: str):
        if getattr(self, '_copilot_apply_in_progress', False):
            return
        try:
            self.mark_field_copilot(key, enabled=False)
        except Exception:
            pass

    def reset_default_hparams(self):
        try:
            if hasattr(self, 'yolo_checkbox'):
                self.yolo_checkbox.blockSignals(True)
                self.yolo_checkbox.setChecked(True)
                self.yolo_checkbox.blockSignals(False)
            if hasattr(self, 'cls_checkbox'):
                self.cls_checkbox.blockSignals(True)
                self.cls_checkbox.setChecked(False)
                self.cls_checkbox.blockSignals(False)
            if hasattr(self, 'update_model_list_for_task'):
                self.update_model_list_for_task()
        except Exception:
            pass
        try:
            if hasattr(self, 'model_combo'):
                idx = self.model_combo.findText('yolov8n')
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            if hasattr(self, 'dataset_combo'):
                idx = self.dataset_combo.findText('coco8')
                if idx >= 0:
                    self.dataset_combo.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            if hasattr(self, 'imgsz_combo'):
                idx = self.imgsz_combo.findText('640')
                if idx >= 0:
                    self.imgsz_combo.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            if hasattr(self, 'device_combo'):
                target_device = 'GPU' if torch.cuda.is_available() else 'CPU'
                idx = self.device_combo.findText(target_device)
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            if hasattr(self, 'epochs_spin'):
                self.epochs_spin.setValue(20)
        except Exception:
            pass
        try:
            if hasattr(self, 'patience_spin'):
                self.patience_spin.setValue(5)
        except Exception:
            pass
        try:
            if hasattr(self, 'triton_checkbox'):
                self.triton_checkbox.setChecked(False)
        except Exception:
            pass
        try:
            if hasattr(self, 'compile_mode_combo'):
                idx = self.compile_mode_combo.findText('Προεπιλογή')
                if idx >= 0:
                    self.compile_mode_combo.setCurrentIndex(idx)
            if hasattr(self, 'log_output'):
                self.log_output.clear()
        except Exception:
            pass
        try:
            if hasattr(self, 'batch_spin'):
                self.batch_spin.setValue(16)
        except Exception:
            pass
        try:
            if hasattr(self, 'workers_spin'):
                self.workers_spin.setValue(2)
        except Exception:
            pass
        try:
            if hasattr(self, 'optimizer_combo'):
                self.optimizer_combo.setCurrentText('auto')
        except Exception:
            pass
        try:
            if hasattr(self, 'lr0_spin'):
                self.lr0_spin.setValue(0.01)
        except Exception:
            pass
        try:
            if hasattr(self, 'lrf_spin'):
                self.lrf_spin.setValue(0.01)
        except Exception:
            pass
        try:
            if hasattr(self, 'momentum_spin'):
                self.momentum_spin.setValue(0.937)
        except Exception:
            pass
        try:
            if hasattr(self, 'weight_decay_spin'):
                self.weight_decay_spin.setValue(0.0005)
        except Exception:
            pass
        try:
            if hasattr(self, 'warmup_epochs_spin'):
                self.warmup_epochs_spin.setValue(3)
        except Exception:
            pass
        try:
            if hasattr(self, 'optimizer_combo'):
                self.on_optimizer_changed(self.optimizer_combo.currentText())
        except Exception:
            pass
        try:
            if hasattr(self, 'clear_copilot_marks'):
                self.clear_copilot_marks()
        except Exception:
            pass

    def manage_reports(self):
        open_folder_externally(str(TRAIN_REPORTS_DIR))

    def start_training(self):
        started = JOB_MANAGER.try_start('Training', lambda: self._start_training_impl(), cancel_cb=getattr(self, 'stop_training', None))
        if not started:
            try:
                self.update_log('<b>🕒 Η εκπαίδευση μπήκε στην ουρά (τρέχει ήδη εργασία).</b>')
            except Exception:
                pass

    # ── Κεντρική υλοποίηση εκκίνησης εκπαίδευσης ────────────────────────────────
    # Αποφασίζει ποιος worker θα χρησιμοποιηθεί:
    #   CNN μοντέλο → CNNTrainingWorker (QThread, no subprocess)
    #   YOLO μοντέλο → subprocess QProcess (ασφαλής από crashes) ή fallback QThread
    # Δημιουργεί job JSON αρχείο και εκκινεί τον κατάλληλο worker.
    # Αποθηκεύει τελευταίες ρυθμίσεις μέσω AppSettings.
    def _start_training_impl(self):
        try:
            warmup_torch_cuda("training_start")
        except Exception:
            pass
        self.progress_start('Training')
        try:
            model = self.model_combo.currentText()
            dataset = self.dataset_combo.currentText()
            imgsz = int(self.imgsz_combo.currentText())
            device_text = self.device_combo.currentText()
            if device_text == 'GPU':
                device = 'cuda:0'
            else:
                device = 'cpu'
            epochs = self.epochs_spin.value()
            patience = self.patience_spin.value()
            extra_hparams = {}
            # Persist last-used settings
            try:
                _settings().set_many({
                    'train_model': self.model_combo.currentText(),
                    'train_dataset': self.dataset_combo.currentText(),
                    'train_imgsz': self.imgsz_combo.currentText(),
                    'train_device': self.device_combo.currentText(),
                    'train_epochs': epochs,
                    'train_patience': patience,
                })
            except Exception:
                pass
            try:
                if hasattr(self, 'batch_spin'):
                    extra_hparams['batch'] = int(self.batch_spin.value())
                if hasattr(self, 'optimizer_combo'):
                    opt = self.optimizer_combo.currentText()
                    if opt and opt.lower() != 'auto':
                        extra_hparams['optimizer'] = opt
                if hasattr(self, 'lr0_spin'):
                    extra_hparams['lr0'] = float(self.lr0_spin.value())
                if hasattr(self, 'lrf_spin'):
                    extra_hparams['lrf'] = float(self.lrf_spin.value())
                if hasattr(self, 'momentum_spin'):
                    extra_hparams['momentum'] = float(self.momentum_spin.value())
                if hasattr(self, 'weight_decay_spin'):
                    extra_hparams['weight_decay'] = float(self.weight_decay_spin.value())
                if hasattr(self, 'warmup_epochs_spin'):
                    extra_hparams['warmup_epochs'] = float(self.warmup_epochs_spin.value())
                if hasattr(self, 'workers_spin'):
                    workers_val = int(self.workers_spin.value())
                    if workers_val > 0:
                        extra_hparams['workers'] = workers_val
            except Exception:
                extra_hparams = extra_hparams or {}
            self.start_button.setEnabled(False)
            self.start_button.setText('⏳ Εκπαίδευση σε εξέλιξη...')
            self.progress_bar.setValue(0)
            self._training_had_error = False
            self._current_train_model_name = model
            self._current_train_dataset_name = dataset
            self._current_train_imgsz = imgsz
            self._current_train_device = device
            self.view_report_button.setEnabled(False)
            self.last_report_path = None

            # ── CNN: always use QThread (no subprocess mode needed) ───────
            if is_cnn_model(model):
                self.update_log(format_html_log(
                    f'🧠 CNN mode: {model} – χρήση CNNTrainingWorker (torchvision)', Colors.CYAN, bold=True))
                self.training_thread = QThread()
                self.training_thread.setObjectName('MMProCNNTrainingThread')
                self.training_worker = CNNTrainingWorker(
                    model, dataset, imgsz, device, epochs, patience,
                    extra_hparams=extra_hparams)
                self.training_worker.current_progress = 0
                self.training_worker.moveToThread(self.training_thread)
                self.training_thread.started.connect(self.training_worker.run)
                self.training_worker.finished.connect(self.on_training_finished)
                self.training_worker.finished.connect(self.training_thread.quit)
                self.training_worker.finished.connect(self.training_worker.deleteLater)
                self.training_thread.finished.connect(self.training_thread.deleteLater)
                self.training_worker.log.connect(self.update_log)
                self.training_worker.error.connect(self.on_error)
                self.training_worker.report_ready.connect(self.on_report_ready)
                self.training_worker.progress.connect(self.progress_apply)
                try:
                    self.progress_bar.setRange(0, 100)
                except Exception:
                    pass
                self.training_thread.start()
                return  # ← early return for CNN path

            if QProcess is not None:
                try:
                    self.progress_bar.setRange(0, 0)
                except Exception:
                    pass
                try:
                    if getattr(sys, 'frozen', False):
                        root_dir = Path(sys.executable).resolve().parent
                        runner = None
                    else:
                        root_dir = Path(__file__).resolve().parent
                        entry = Path(__file__).resolve()
                    job = {'model': model, 'dataset': dataset, 'imgsz': int(imgsz), 'device': device, 'epochs': int(epochs), 'patience': int(patience), 'use_triton': bool(self.use_triton), 'compile_mode': str(self.compile_mode), 'ui_theme': str(getattr(self.window(), 'current_theme', 'light')), 'extra_hparams': extra_hparams or {}}
                    try:
                        CRASH_LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    job_path = CRASH_LOGS_DIR / f'training_job_{int(time.time())}.json'
                    with open(job_path, 'w', encoding='utf-8') as f:
                        json.dump(job, f, ensure_ascii=False, indent=2)
                    self._trainproc_out_buf = ''
                    self._trainproc_err_buf = ''
                    self._trainproc_last_error = None
                    self.training_process = QProcess(self)
                    self.training_process.setWorkingDirectory(str(root_dir))
                    self.training_process.setProgram(sys.executable)
                    if getattr(sys, 'frozen', False):
                        self.training_process.setArguments(['--mmpro-mode=train', str(job_path)])
                    else:
                        self.training_process.setArguments([str(entry), '--mmpro-mode=train', str(job_path)])
                    self.training_process.readyReadStandardOutput.connect(self._on_trainproc_stdout)
                    self.training_process.readyReadStandardError.connect(self._on_trainproc_stderr)
                    self.training_process.finished.connect(self._on_trainproc_finished)
                    self.training_process.start()
                except Exception as e:
                    try:
                        self.update_log(format_html_log(f'⚠️ Αποτυχία εκκίνησης subprocess training: {e}\n➡️ Γίνεται fallback σε QThread.', Colors.ORANGE, bold=True))
                    except Exception:
                        pass
                    self.training_thread = QThread()
                    self.training_thread.setObjectName('MMProTrainingThread')
                    self.training_worker = TrainingWorker(model, dataset, imgsz, device, epochs, patience, self.use_triton, self.compile_mode, extra_hparams=extra_hparams)
                    self.training_worker.current_progress = 0
                    self.training_worker.moveToThread(self.training_thread)
                    self.training_thread.started.connect(self.training_worker.run)
                    self.training_worker.finished.connect(self.on_training_finished)
                    self.training_worker.finished.connect(self.training_thread.quit)
                    self.training_worker.finished.connect(self.training_worker.deleteLater)
                    self.training_thread.finished.connect(self.training_thread.deleteLater)
                    self.training_worker.log.connect(self.update_log)
                    self.training_worker.error.connect(self.on_error)
                    self.training_worker.report_ready.connect(self.on_report_ready)
                    self.training_thread.start()
                    self.progress_poll_timer.start()
            else:
                self.training_thread = QThread()
                self.training_thread.setObjectName('MMProTrainingThread')
                self.training_worker = TrainingWorker(model, dataset, imgsz, device, epochs, patience, self.use_triton, self.compile_mode, extra_hparams=extra_hparams)
                self.training_worker.current_progress = 0
                self.training_worker.moveToThread(self.training_thread)
                self.training_thread.started.connect(self.training_worker.run)
                self.training_worker.finished.connect(self.on_training_finished)
                self.training_worker.finished.connect(self.training_thread.quit)
                self.training_worker.finished.connect(self.training_worker.deleteLater)
                self.training_thread.finished.connect(self.training_thread.deleteLater)
                self.training_worker.log.connect(self.update_log)
                self.training_worker.error.connect(self.on_error)
                self.training_worker.report_ready.connect(self.on_report_ready)
                self.training_thread.start()
                self.progress_poll_timer.start()
        except Exception as e:
            try:
                self.start_button.setEnabled(True)
                self.start_button.setText('🚀 Έναρξη Εκπαίδευσης')
            except Exception:
                pass
            try:
                self.progress_bar.setValue(0)
            except Exception:
                pass
            error_msg = f'Παρουσιάστηκε σφάλμα κατά την προετοιμασία της εκπαίδευσης:\n{e}'
            try:
                self.update_log(format_html_log(error_msg, Colors.RED, bold=True))
                QMessageBox.critical(self, 'Σφάλμα Προετοιμασίας Εκπαίδευσης', error_msg)
            except Exception:
                pass

    def _on_trainproc_stdout(self):
        try:
            if not self.training_process:
                return
            data = bytes(self.training_process.readAllStandardOutput()).decode('utf-8', 'replace')
            if not data:
                return
            self._trainproc_out_buf += data
            self._drain_trainproc_buffer(is_err=False)
        except Exception:
            pass

    def _on_trainproc_stderr(self):
        try:
            if not self.training_process:
                return
            data = bytes(self.training_process.readAllStandardError()).decode('utf-8', 'replace')
            if not data:
                return
            self._trainproc_err_buf += data
            self._drain_trainproc_buffer(is_err=True)
        except Exception:
            pass

    def _drain_trainproc_buffer(self, is_err: bool):
        buf = self._trainproc_err_buf if is_err else self._trainproc_out_buf
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.rstrip('\r')
            self._handle_trainproc_line(line, is_err=is_err)
        if is_err:
            self._trainproc_err_buf = buf
        else:
            self._trainproc_out_buf = buf

    def _handle_trainproc_line(self, line: str, is_err: bool=False):
        if not line:
            return
        if line.startswith('__MM_PROGRESS__'):
            payload = line[len('__MM_PROGRESS__'):].strip()
            try:
                pct_s, msg = payload.split('|', 1)
                self.progress_apply(int(pct_s), msg)
            except Exception:
                pass
        if line.startswith('__MM_LOG__'):
            html = line[len('__MM_LOG__'):].replace('\\n', '\n')
            self.update_log(html)
        if line.startswith('__MM_REPORT__'):
            path_str = line[len('__MM_REPORT__'):].strip()
            if path_str:
                try:
                    self.on_report_ready(path_str)
                except Exception:
                    pass
        if line.startswith('__MM_ERR__'):
            msg = line[len('__MM_ERR__'):].strip()
            self._trainproc_last_error = msg
            try:
                self.update_log(format_html_log(msg, Colors.RED, bold=True))
            except Exception:
                pass
        if line.startswith('__MM_EXCEPTION__'):
            msg = line[len('__MM_EXCEPTION__'):].strip()
            self._trainproc_last_error = msg
            try:
                self.update_log(format_html_log('❌ Exception στο training subprocess:', Colors.RED, bold=True))
                self.update_log(format_html_log(msg, Colors.RED))
            except Exception:
                pass
        if line.startswith('__MM_DONE__'):
            return
        try:
            color = Colors.RED if is_err else Colors.GRAY
            self.update_log(format_html_log(line, color))
        except Exception:
            pass

    def _on_trainproc_finished(self, exit_code: int, exit_status):
        try:
            self.progress_bar.setRange(0, 100)
        except Exception:
            pass
        try:
            self._drain_trainproc_buffer(is_err=False)
            self._drain_trainproc_buffer(is_err=True)
        except Exception:
            pass
        try:
            self.progress_finish(ok=exit_code == 0 and (not self._exportproc_last_error))
            self.progress_finish(ok=exit_code == 0 and (not self._exportproc_last_error))
        except Exception:
            pass
        if exit_code != 0 or self._trainproc_last_error:
            crash_hint = ''
            try:
                crash_hint = f'\n\nΔες επίσης: {CRASH_LOGS_DIR} (faulthandler logs).'
            except Exception:
                crash_hint = ''
            msg = self._trainproc_last_error or f'Το training subprocess τερμάτισε με κωδικό: {exit_code}.{crash_hint}'
            try:
                self.on_error(msg)
                self.on_training_finished(is_error=True)
            except Exception:
                pass
        else:
            try:
                self.on_training_finished(is_error=False)
            except Exception:
                pass

    def _on_exportproc_stdout(self):
        try:
            if not getattr(self, 'export_process', None):
                return
            data = bytes(self.export_process.readAllStandardOutput()).decode('utf-8', 'replace')
            if not data:
                return
            self._exportproc_out_buf += data
            self._drain_exportproc_buffer(is_err=False)
        except Exception:
            pass

    def _on_exportproc_stderr(self):
        try:
            if not getattr(self, 'export_process', None):
                return
            data = bytes(self.export_process.readAllStandardError()).decode('utf-8', 'replace')
            if not data:
                return
            self._exportproc_err_buf += data
            self._drain_exportproc_buffer(is_err=True)
        except Exception:
            pass

    def _drain_exportproc_buffer(self, is_err: bool):
        buf = self._exportproc_err_buf if is_err else self._exportproc_out_buf
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.rstrip('\r')
            self._handle_exportproc_line(line, is_err=is_err)
        if is_err:
            self._exportproc_err_buf = buf
        else:
            self._exportproc_out_buf = buf

    def _handle_exportproc_line(self, line: str, is_err: bool=False):
        if not line:
            return
        if line.startswith('__MM_PROGRESS__'):
            payload = line[len('__MM_PROGRESS__'):].strip()
            try:
                pct_s, msg = payload.split('|', 1)
                self.progress_apply(int(pct_s), msg)
            except Exception:
                pass
        if line.startswith('__MM_LOG__'):
            html = line[len('__MM_LOG__'):].replace('\\n', '\n')
            self.update_log(html)
            try:
                import re as _re
                plain = _re.sub('<[^>]+>', ' ', html)
                plain = plain.replace('&nbsp;', ' ').replace('&amp;', '&')
                parsed = self._parse_progress_from_text(plain)
                if parsed:
                    pct, _ = parsed
                    self.progress_apply(pct, '')
            except Exception:
                pass
        if line.startswith('__MM_ERR__'):
            msg = line[len('__MM_ERR__'):].strip()
            self._exportproc_last_error = msg
            try:
                self.update_log(format_html_log(msg, Colors.RED, bold=True))
            except Exception:
                pass
        if line.startswith('__MM_EXCEPTION__'):
            msg = line[len('__MM_EXCEPTION__'):].strip()
            self._exportproc_last_error = msg
            try:
                self.update_log(format_html_log('❌ Exception στο export subprocess:', Colors.RED, bold=True))
                self.update_log(format_html_log(msg, Colors.RED))
            except Exception:
                pass
        if line.startswith('__MM_DONE__'):
            return
        try:
            color = Colors.RED if is_err else Colors.GRAY
            self.update_log(format_html_log(line, color))
        except Exception:
            pass

    def _on_exportproc_finished(self, exit_code: int, exit_status):
        self._export_running = False
        try:
            self._drain_exportproc_buffer(is_err=False)
            self._drain_exportproc_buffer(is_err=True)
        except Exception:
            pass
        if exit_code != 0 or self._exportproc_last_error:
            crash_hint = ''
            try:
                crash_hint = f'\n\nΔες επίσης: {CRASH_LOGS_DIR} (faulthandler logs).'
            except Exception:
                crash_hint = ''
            msg = self._exportproc_last_error or f'Το export subprocess τερμάτισε με κωδικό: {exit_code}.{crash_hint}'
            try:
                self.on_export_error(msg)
            except Exception:
                pass
        else:
            try:
                self.on_export_finished()
            except Exception:
                pass

    def update_log(self, html_text):
        if isinstance(html_text, str) and html_text.startswith('__INLINE__'):
            html_text = html_text[len('__INLINE__'):]
            cursor = self.log_output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.MoveAnchor)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(html_text)
            self.log_output.setTextCursor(cursor)
        else:
            self.log_output.append(html_text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def on_error(self, error_text):
        self.progress_finish(False)
        self._training_had_error = True
        self.progress_poll_timer.stop()
        self.update_log(format_html_log(error_text, Colors.RED, bold=True))
        QMessageBox.critical(self, 'Σφάλμα Εκπαίδευσης', error_text)

    def on_report_ready(self, path_str):
        self.last_report_path = path_str
        self.view_report_button.setEnabled(True)
        self.update_log(format_html_log(f'Η αναφορά PDF είναι έτοιμη: {Path(path_str).name}', Colors.GREEN, bold=True))

    def view_report(self):
        if self.last_report_path and Path(self.last_report_path).exists():
            if not open_file_externally(self.last_report_path):
                QMessageBox.warning(self, 'Σφάλμα', f'Δεν ήταν δυνατό το άνοιγμα του αρχείου: {self.last_report_path}')
        else:
            QMessageBox.warning(self, 'Σφάλμα', 'Το αρχείο αναφοράς δεν βρέθηκε ή δεν έχει δημιουργηθεί ακόμα.')

    def _open_compare_dialog(self):
        try:
            dlg = TrainingRunsComparisonDialog(self)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, 'Σφάλμα', str(e))

    def on_training_finished(self, is_error=False):
        JOB_MANAGER.done(True)
        self.progress_finish(True)
        self.progress_poll_timer.stop()
        has_error = is_error or getattr(self, '_training_had_error', False)
        if not has_error:
            self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.start_button.setText('🚀 Έναρξη Εκπαίδευσης')
        # Defer ref clear until thread.finished to avoid GC destroying QThread prematurely
        try:
            th = self.training_thread
            if th is not None:
                th.finished.connect(lambda: setattr(self, 'training_thread', None) or setattr(self, 'training_worker', None), Qt.ConnectionType.SingleShotConnection)
            else:
                self.training_thread = None
                self.training_worker = None
        except Exception:
            self.training_thread = None
            self.training_worker = None
        if getattr(self, 'training_process', None):
            try:
                self.training_process.deleteLater()
            except Exception:
                pass
            self.training_process = None
            self._trainproc_out_buf = ''
            self._trainproc_err_buf = ''
            self._trainproc_last_error = None
            try:
                self.progress_bar.setRange(0, 100)
            except Exception:
                pass
        self.training_process = None
        self._trainproc_out_buf = ''
        self._trainproc_err_buf = ''
        self._trainproc_last_error = None
        self._training_had_error = False
        self.refresh_models()
        if not has_error:
            try:
                self.last_trained_model_name = getattr(self, '_current_train_model_name', None)
                self.last_trained_dataset_name = getattr(self, '_current_train_dataset_name', None)
                self.last_trained_imgsz = getattr(self, '_current_train_imgsz', None)
                self.last_trained_device = getattr(self, '_current_train_device', None)
                try:
                    model_name = getattr(self, 'last_trained_model_name', None)
                    dataset_name = getattr(self, 'last_trained_dataset_name', None)
                    imgsz = getattr(self, 'last_trained_imgsz', None)
                    device = getattr(self, 'last_trained_device', None)
                    device_type = 'GPU' if isinstance(device, str) and 'cuda' in device else 'CPU'
                    if model_name and dataset_name and (imgsz is not None):
                        expected_name = f'{model_name}_{device_type}_{dataset_name}_{imgsz}.pt'
                        if hasattr(self, 'export_model_combo'):
                            idx = self.export_model_combo.findText(expected_name)
                            if idx >= 0:
                                self.export_model_combo.setCurrentIndex(idx)
                except Exception:
                    pass
            except Exception:
                pass
        try:
            if not has_error and hasattr(self, 'training_completed'):
                self.training_completed.emit()
        except Exception:
            pass
        try:
            summary = flush_log_once_summary('Training', reset=True, top_n=12, min_total=1)
            if summary:
                try:
                    self.update_log(format_html_summary(summary))
                except Exception:
                    pass
        except Exception:
            pass

    def poll_worker_progress(self):
        if self.training_worker and self.training_thread:
            self.training_worker.progress_mutex.lock()
            current_progress = self.training_worker.current_progress
            self.training_worker.progress_mutex.unlock()
            if self.progress_bar.value() < current_progress:
                self.progress_bar.setValue(current_progress)
        else:
            self.progress_poll_timer.stop()

    def start_export(self, export_format):
        started = JOB_MANAGER.try_start('Export', lambda: self._start_export_impl(export_format), cancel_cb=getattr(self, 'stop_export', None))
        if not started:
            try:
                self.update_log('<b>🕒 Το export μπήκε στην ουρά (τρέχει ήδη εργασία).</b>')
            except Exception:
                pass

    def _start_export_impl(self, export_format):
        self._export_running = False
        try:
            if getattr(self, 'export_process', None) is not None:
                try:
                    if self.export_process.state() != QProcess.ProcessState.NotRunning:
                        QMessageBox.information(self, 'Εξαγωγή μοντέλου', 'Ήδη εκτελείται εξαγωγή μοντέλου. Περίμενε να ολοκληρωθεί πριν ξεκινήσεις νέα.')
                except Exception:
                    pass
            if hasattr(self, 'export_thread') and self.export_thread is not None:
                try:
                    if self.export_thread.isRunning():
                        QMessageBox.information(self, 'Εξαγωγή μοντέλου', 'Ήδη εκτελείται εξαγωγή μοντέλου. Περίμενε να ολοκληρωθεί πριν ξεκινήσεις νέα.')
                except Exception:
                    pass
            if not getattr(self, 'available_models', None):
                QMessageBox.warning(self, 'Σφάλμα', 'Δεν έχετε επιλέξει έγκυρο μοντέλο.')
                return
            current_index = self.export_model_combo.currentIndex()
            if current_index < 0 or current_index >= len(self.available_models):
                QMessageBox.warning(self, 'Σφάλμα', 'Δεν έχετε επιλέξει έγκυρο μοντέλο.')
                return
            selected_model_info = self.available_models[current_index]
            model_path, model_type = selected_model_info
            overwrite = False
            try:
                from pathlib import Path
                target = export_target_for(Path(model_path), export_format)
                if target_exists_and_nonempty(target):
                    if export_format == 'tensorrt':
                        export_path = target.path
                        try:
                            if trt_engine_is_up_to_date(Path(model_path), export_path, int(self.export_imgsz_input.value()), half=True, batch=1, dynamic=False, workspace=4, device='cuda:0'):
                                self.update_log(format_html_log(f'♻️🔥 Το TensorRT engine είναι ήδη ενημερωμένο: {export_path}', Colors.GREEN, bold=False))
                                QMessageBox.information(self, 'Εξαγωγή TensorRT', f'Υπάρχει ήδη ενημερωμένο TensorRT engine:\n\n{export_path.name}\n\nΔεν χρειάζεται νέο export.')
                        except Exception:
                            pass
                    title_map = { 'onnx': 'Εξαγωγή ONNX', 'tensorrt': 'Εξαγωγή TensorRT', 'ncnn': 'Εξαγωγή NCNN',}
                    title = title_map.get(export_format, 'Εξαγωγή')
                    if not ask_overwrite(self, title, target, default_no=True):
                        try:
                            self.update_log(format_html_log(f'⚠️ Παράλειψη εξαγωγής {export_format.upper()}, ο στόχος υπάρχει ήδη: {target.path}', Colors.YELLOW, bold=False))
                        except Exception:
                            pass
                    overwrite = True
            except Exception:
                overwrite = False
            # ── CNN: μόνο cnn_onnx format επιτρέπεται ─────────────────────
            _is_cnn_model = (model_type == 'CNN') or any(
                c in model_path.stem.lower() for c in _CNN_MODEL_KEYS)
            if _is_cnn_model and export_format != 'cnn_onnx':
                QMessageBox.warning(
                    self, 'Εξαγωγή μη διαθέσιμη',
                    'Για CNN (torchvision) μοντέλα χρησιμοποίησε το κουμπί\n'
                    '"Εξαγωγή CNN → ONNX".')
                return
            if export_format == 'ncnn' and model_type != 'PyTorch':
                QMessageBox.warning(self, 'Σφάλμα Εξαγωγής', 'Παρακαλώ επιλέξτε ένα PyTorch (.pt) μοντέλο για εξαγωγή NCNN.')
                return
            if export_format == 'onnx':
                if model_type != 'PyTorch':
                    QMessageBox.warning(self, 'Σφάλμα Εξαγωγής', 'Παρακαλώ επιλέξτε ένα PyTorch (.pt) μοντέλο για εξαγωγή ONNX.')
                    return
                if not self._is_gpu_trained_model(model_path):
                    QMessageBox.warning(self, 'Σφάλμα Εξαγωγής', 'Παρακαλώ επιλέξτε ένα PyTorch μοντέλο GPU για εξαγωγή ONNX.')
                    return
            if export_format == 'tensorrt':
                if model_type != 'PyTorch':
                    QMessageBox.warning(self, 'Σφάλμα Εξαγωγής', 'Παρακαλώ επιλέξτε ένα PyTorch (.pt) μοντέλο για εξαγωγή TensorRT.')
                    return
                if not self._is_gpu_trained_model(model_path):
                    QMessageBox.warning(self, 'Σφάλμα Εξαγωγής', 'Παρακαλώ επιλέξτε ένα PyTorch μοντέλο GPU για εξαγωγή TensorRT.')
                    return
            try:
                imgsz = int(self.export_imgsz_combo.currentText())
            except Exception:
                QMessageBox.warning(self, 'Σφάλμα', 'Μη έγκυρο μέγεθος εικόνας για εξαγωγή.')
                return
            self.start_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.export_onnx_button.setEnabled(False)
            if hasattr(self, 'export_tensorrt_button'):
                self.export_tensorrt_button.setEnabled(False)
            if hasattr(self, 'export_cnn_onnx_button'):
                self.export_cnn_onnx_button.setEnabled(False)
            self.export_model_combo.setEnabled(False)
            if QProcess is not None:
                try:
                    if getattr(sys, 'frozen', False):
                        root_dir = Path(sys.executable).resolve().parent
                        runner = None
                    else:
                        root_dir = Path(__file__).resolve().parent
                        entry = Path(__file__).resolve()
                    job = {'model_path': str(model_path), 'imgsz': int(imgsz), 'export_format': str(export_format), 'overwrite': bool(overwrite), 'ui_theme': str(getattr(self.window(), 'current_theme', 'light'))}
                    try:
                        CRASH_LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    job_path = CRASH_LOGS_DIR / f'export_job_{int(time.time())}.json'
                    with open(job_path, 'w', encoding='utf-8') as f:
                        json.dump(job, f, ensure_ascii=False, indent=2)
                    self._exportproc_out_buf = ''
                    self._exportproc_err_buf = ''
                    self._exportproc_last_error = None
                    self.export_process = QProcess(self)
                    self.export_process.setWorkingDirectory(str(root_dir))
                    self.export_process.setProgram(sys.executable)
                    if getattr(sys, 'frozen', False):
                        self.export_process.setArguments(['--mmpro-mode=export', str(job_path)])
                    else:
                        self.export_process.setArguments([str(entry), '--mmpro-mode=export', str(job_path)])
                    self.export_process.readyReadStandardOutput.connect(self._on_exportproc_stdout)
                    self.export_process.readyReadStandardError.connect(self._on_exportproc_stderr)
                    self.export_process.finished.connect(self._on_exportproc_finished)
                    try:
                        self.export_process.stateChanged.connect(lambda st: self.progress_finish(ok=not self._exportproc_last_error) if getattr(self, '_export_running', False) and st == QProcess.NotRunning else None)
                    except Exception:
                        pass
                    try:
                        self.export_process.errorOccurred.connect(lambda *_: self.progress_finish(ok=False))
                    except Exception:
                        pass
                    self._export_running = True
                    self.progress_start('Export')
                    self.export_process.start()
                except Exception as e:
                    try:
                        self.update_log(format_html_log(f'⚠️ Αποτυχία εκκίνησης subprocess export: {e}\n➡️ Γίνεται fallback σε QThread.', Colors.ORANGE, bold=True))
                    except Exception:
                        pass
                    self.export_thread = QThread()
                    self.export_thread.setObjectName('MMProExportThread')
                    self.export_worker = ExportWorker(model_path, imgsz, export_format, overwrite=bool(overwrite))
                    self.export_worker.moveToThread(self.export_thread)
                    self.export_thread.started.connect(self.export_worker.run)
                    self.export_worker.finished.connect(self.on_export_finished)
                    self.export_worker.finished.connect(self.export_thread.quit)
                    self.export_worker.finished.connect(self.export_worker.deleteLater)
                    self.export_thread.finished.connect(self.export_thread.deleteLater)
                    self.export_worker.log.connect(self.update_log)
                    self.export_worker.error.connect(self.on_export_error)
                    self.export_thread.start()
            else:
                self.export_thread = QThread()
                self.export_thread.setObjectName('MMProExportThread')
                self.export_worker = ExportWorker(model_path, imgsz, export_format, overwrite=bool(overwrite))
                self.export_worker.moveToThread(self.export_thread)
                self.export_thread.started.connect(self.export_worker.run)
                self.export_worker.finished.connect(self.on_export_finished)
                self.export_worker.finished.connect(self.export_thread.quit)
                self.export_worker.finished.connect(self.export_worker.deleteLater)
                self.export_thread.finished.connect(self.export_thread.deleteLater)
                self.export_worker.log.connect(self.update_log)
                self.export_worker.error.connect(self.on_export_error)
                self.export_thread.start()
        except Exception as e:
            err_text = f'Απρόσμενο σφάλμα κατά την εκκίνηση εξαγωγής: {e}'
            try:
                self.update_log(format_html_log(err_text, Colors.RED, bold=True))
            except Exception:
                pass
            QMessageBox.critical(self, 'Σφάλμα Εξαγωγής', err_text)
        self.progress_finish(ok=False)

    def on_export_error(self, error_text):
        JOB_MANAGER.done(False)
        self._export_running = False
        self.progress_finish(False)
        try:
            self.update_log(format_html_log(error_text, Colors.RED, bold=True))
            QMessageBox.critical(self, 'Σφάλμα Εξαγωγής', error_text)
        except Exception:
            pass
        try:
            self.on_export_finished()
        except Exception:
            pass
        self.progress_finish(ok=True)

    def on_export_finished(self):
        JOB_MANAGER.done(True)
        self._export_running = False
        self.update_log(format_html_log('🏁 Η εξαγωγή ολοκληρώθηκε.', Colors.CYAN, bold=True))
        self.start_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.export_model_combo.setEnabled(True)
        try:
            # Επαναφέρει σωστά το state όλων των κουμπιών (incl. cnn_onnx_button)
            self.on_export_model_changed(self.export_model_combo.currentIndex())
        except Exception:
            pass
        try:
            th = self.export_thread
            if th is not None:
                th.finished.connect(lambda: setattr(self, 'export_thread', None) or setattr(self, 'export_worker', None), Qt.ConnectionType.SingleShotConnection)
            else:
                self.export_thread = None
                self.export_worker = None
        except Exception:
            self.export_thread = None
            self.export_worker = None
        if getattr(self, 'export_process', None):
            try:
                self.export_process.deleteLater()
            except Exception:
                pass
            self.export_process = None
            self._exportproc_out_buf = ''
            self._exportproc_err_buf = ''
            self._exportproc_last_error = None
        self.refresh_models()
        try:
            self.log_output.append('')
        except Exception:
            pass
        try:
            summary = flush_log_once_summary('Export', reset=True, top_n=12, min_total=1)
            if summary:
                try:
                    self.update_log(format_html_summary(summary))
                except Exception:
                    pass
        except Exception:
            pass

    def _parse_progress_from_text(self, text: str):
        if not text:
            return None
        t = ' '.join(str(text).split())
        m = re.search(r'(\d{1,3})\s*%', t)
        if m:
            try:
                return (int(m.group(1)), t)
            except Exception:
                return None
        m = re.search(r'epoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)', t, re.IGNORECASE)
        if m:
            cur = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                return (int(cur * 100 / total), t)
        m = re.search(r'(step|iter|iters|batch)\s*[:=]?\s*(\d+)\s*/\s*(\d+)', t, re.IGNORECASE)
        if m:
            cur = int(m.group(2))
            total = int(m.group(3))
            if total > 0:
                return (int(cur * 100 / total), t)

    def progress_apply(self, pct: int, msg: str=''):
        return

    def _get_progress_bar(self):
        return getattr(self, 'progress_bar', None)

    def progress_start(self, msg: str=''):
        pb = self._get_progress_bar()
        if pb is None:
            return
        self._progress_running = True
        try:
            pb.setTextVisible(False)
            pb.setRange(0, 0)
        except Exception:
            pass

    def _progress_tick(self):
        return

    def progress_finish(self, ok: bool=True):
        pb = self._get_progress_bar()
        self._progress_running = False
        try:
            self._progress_timer.stop()
        except Exception:
            pass
        if pb is None:
            return
        try:
            pb.setRange(0, 100)
            pb.setValue(100)
            pb.setTextVisible(True)
            pb.setFormat('%p%')
        except Exception:
            pass
"""Training copilot tab (UI).
Βοηθητικό tab για guidance/auto-checks κατά το training workflow.
"""


def set_copilot_busy(owner, busy: bool) -> None:
    try:
        busy_flag = bool(busy)
    except Exception:
        busy_flag = True
    try:
        if busy_flag:
            owner.copilot_button.setText('⏳ Ζητούνται προτάσεις από LLM API...')
            owner.copilot_button.setEnabled(False)
        else:
            owner.copilot_button.setText('🤖 Πρότεινε ρυθμίσεις εκπαίδευσης')
            owner.copilot_button.setEnabled(True)
    except Exception:
        pass
    try:
        if hasattr(owner, 'copilot_tune_button'):
            owner.copilot_tune_button.setEnabled(not busy_flag)
    except Exception:
        pass
    try:
        if hasattr(owner, 'copilot_detection_button'):
            if busy_flag:
                try:
                    owner._copilot_detection_prev_enabled = owner.copilot_detection_button.isEnabled()
                except Exception:
                    owner._copilot_detection_prev_enabled = None
                owner.copilot_detection_button.setEnabled(False)
            else:
                prev = getattr(owner, '_copilot_detection_prev_enabled', None)
                if prev is not None:
                    try:
                        owner.copilot_detection_button.setEnabled(bool(prev))
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        if hasattr(owner, 'copilot_apply_button') and busy_flag:
            owner.copilot_apply_button.setEnabled(False)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# TrainingCopilotTab – Tab «🤖 Model Training Copilot»
# ════════════════════════════════════════════════════════════════════════════════
# Παρέχει AI-βοηθούμενες προτάσεις βελτιστοποίησης εκπαίδευσης μέσω LLM API.
# Λειτουργίες:
#   - «Πρότεινε ρυθμίσεις»: στέλνει τρέχουσες ρυθμίσεις + hardware info στο LLM
#   - «Βελτίωση με βάση τα αποτελέσματα εκπαίδευσης»: αναλύει training.log
#   - «Βελτίωση με βάση ανίχνευση»: ενεργοποιεί StatisticsTab και αναλύει αποτελέσματα
#   - «Εφαρμογή ρυθμίσεων»: εφαρμόζει YAML blocks απευθείας στο Training Tab
# LLM: Groq API (TRAINING_COPILOT_SYSTEM_PROMPT – γνωρίζει YOLO + CNN μοντέλα)
# ════════════════════════════════════════════════════════════════════════════════
class TrainingCopilotTab(QWidget, TabNavigationMixin):

    def __init__(self, training_tab, statistics_tab=None):
        super().__init__()
        self.training_tab = training_tab
        self.statistics_tab = statistics_tab
        self._pending_detection_from_copilot = False
        self._last_copilot_model_info = ''
        try:
            if hasattr(self.training_tab, 'training_completed'):
                self.training_tab.training_completed.connect(self.on_training_completed)
        except Exception:
            pass
        try:
            if self.statistics_tab is not None and hasattr(self.statistics_tab, 'analysis_completed'):
                self.statistics_tab.analysis_completed.connect(self.on_detection_analysis_completed)
        except Exception:
            pass
        self.copilot_thread = None
        self.copilot_worker = None
        self.copilot_last_gui_cfg = None
        self.copilot_last_hparams = None
        self.init_ui()
        try:
            self._update_copilot_buttons_for_llm(show_message=True)
        except Exception:
            pass

    def init_ui(self):
        layout, top_bar_layout = _make_tab_layout(self)
        self.llm_settings_button = QPushButton('⚙️ Ρυθμίσεις LLM')
        self.llm_settings_button.setToolTip('Άνοιγμα παραθύρου ρυθμίσεων για Groq API key και προεπιλεγμένο LLM μοντέλο.')
        self.llm_settings_button.clicked.connect(self.on_llm_settings_clicked)
        top_bar_layout.addWidget(self.llm_settings_button)
        _finish_tab_topbar(self, top_bar_layout, layout)
        copilot_group = QGroupBox('🧠 Model Training Copilot (Open Source LLM)')
        copilot_layout = QVBoxLayout(copilot_group)
        prompt_group = QGroupBox('✏️ Prompt προς Copilot')
        prompt_layout = QVBoxLayout(prompt_group)
        self.copilot_prompt_edit = QPlainTextEdit()
        self.copilot_prompt_edit.setPlaceholderText('Περιέγραψε με απλά λόγια τι θέλεις να πετύχεις στην εκπαίδευση.\nΠαραδείγματα:\n- Real-time ανίχνευση σε GPU με καλή ακρίβεια.\n- Μέγιστη ακρίβεια χωρίς να μας νοιάζει ο χρόνος.\n- Γρήγορο demo με λίγες εποχές.')
        self.copilot_prompt_edit.setMinimumHeight(140)
        self.copilot_prompt_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.copilot_prompt_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.copilot_prompt_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        prompt_layout.addWidget(self.copilot_prompt_edit)
        copilot_layout.addWidget(prompt_group)
        buttons_layout = QHBoxLayout()
        self.copilot_button = QPushButton('🤖 Πρότεινε ρυθμίσεις εκπαίδευσης')
        self.copilot_button.clicked.connect(self.on_copilot_suggest_clicked)
        buttons_layout.addWidget(self.copilot_button)
        self.copilot_tune_button = QPushButton('📈 Βελτίωση με βάση τα αποτελέσματα εκπαίδευσης')
        self.copilot_tune_button.setToolTip('Διάβασε τα αποτελέσματα της τελευταίας εκπαίδευσης (log / metrics) και πρότεινε βελτιωμένες ρυθμίσεις.')
        self.copilot_tune_button.setEnabled(True)
        self.copilot_tune_button.clicked.connect(self.on_copilot_tune_from_last_run)
        buttons_layout.addWidget(self.copilot_tune_button)
        self.copilot_detection_button = QPushButton('🧪 Βελτίωση με βάση τα αποτελέσματα ανίχνευσης')
        self.copilot_detection_button.setToolTip('Χρησιμοποίησε τα αποτελέσματα ανίχνευσης (καρτέλα Στατιστικά Ανίχνευσης) για να προτείνεις νέες ρυθμίσεις εκπαίδευσης.')
        self.copilot_detection_button.setEnabled(False)
        self.copilot_detection_button.clicked.connect(self.on_copilot_detection_button_clicked)
        buttons_layout.addWidget(self.copilot_detection_button)
        buttons_layout.addStretch(1)
        copilot_layout.addLayout(buttons_layout)
        output_group = QGroupBox('📄 Αποτελέσματα Copilot / Προτεινόμενες Ρυθμίσεις')
        output_layout = QVBoxLayout(output_group)
        self.copilot_output_edit = QPlainTextEdit()
        self.copilot_output_edit.setReadOnly(True)
        self.copilot_output_edit.setPlaceholderText('Εδώ θα εμφανιστούν οι αναλυτικές προτάσεις του LLM API (ανάλυση κειμένου + YAML).')
        self.copilot_output_edit.setMinimumHeight(340)
        self.copilot_output_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.copilot_output_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.copilot_output_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        output_layout.addWidget(self.copilot_output_edit)
        copilot_layout.addWidget(output_group)
        self.copilot_apply_button = QPushButton('⚙️ Επισκόπιση και Εφαρμογή ρυθμίσεων')
        self.copilot_apply_button.setToolTip('Προβολή και εφαρμογή των ρυθμίσεων που πρότεινε το μοντέλο (GUI + hyperparameters).')
        self.copilot_apply_button.setEnabled(False)
        self.copilot_apply_button.clicked.connect(self.on_copilot_apply_clicked)
        copilot_layout.addWidget(self.copilot_apply_button)
        layout.addWidget(copilot_group)
        info_label = QLabel("Το Copilot διαβάζει τις τρέχουσες ρυθμίσεις από το Tab '🎓 Εκπαίδευση Μοντέλου'\nκαι προτείνει νέες τιμές τόσο για τα πεδία του GUI όσο και για τους υπερ-παράμετρους εκπαίδευσης.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        layout.addStretch(1)

    def _show_no_llm_key_message(self) -> None:
        try:
            if hasattr(self, 'copilot_output_edit') and self.copilot_output_edit is not None:
                self.copilot_output_edit.clear()
                self.copilot_output_edit.appendPlainText('❗ Το A.I Copilot είναι προσωρινά απενεργοποιημένο.\nΔεν έχει οριστεί έγκυρο GROQ_API_KEY για το Groq LLM.\n\nΓια να ενεργοποιηθεί το Copilot:\n  1. Πάτησε το κουμπί «⚙️ Ρυθμίσεις LLM» επάνω αριστερά.\n  2. Επικόλλησε το GROQ_API_KEY στο αντίστοιχο πεδίο.\n  3. Επίλεξε προεπιλεγμένο LLM μοντέλο.\n  4. Πάτησε «Αποθήκευση».\n\nΜόλις αποθηκευτούν οι ρυθμίσεις, τα κουμπιά Copilot\nθα ενεργοποιηθούν αυτόματα.')
        except Exception:
            pass

    def _update_copilot_buttons_for_llm(self, show_message: bool=False) -> None:
        has_key = False
        try:
            has_key = has_valid_groq_api_key()
        except Exception:
            has_key = False
        try:
            if hasattr(self, 'copilot_button') and self.copilot_button is not None:
                self.copilot_button.setEnabled(has_key)
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_tune_button') and self.copilot_tune_button is not None:
                self.copilot_tune_button.setEnabled(has_key)
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_detection_button') and self.copilot_detection_button is not None:
                self.copilot_detection_button.setEnabled(has_key)
        except Exception:
            pass
        if show_message and (not has_key):
            self._show_no_llm_key_message()

    def _handle_unexpected_copilot_exception(self, origin: str, exc: Exception) -> None:
        try:
            set_copilot_busy(self, False)
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_output_edit') and self.copilot_output_edit is not None:
                self.copilot_output_edit.appendPlainText(f'\n[Εσωτερικό σφάλμα στο Copilot ({origin})]: {exc}')
        except Exception:
            pass
        try:
            QMessageBox.critical(self, 'Σφάλμα Copilot', f'Παρουσιάστηκε εσωτερικό σφάλμα στο Copilot κατά τη διάρκεια της ενέργειας: {origin}.\nΛεπτομέρειες: {exc}')
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_tune_button') and self.copilot_tune_button is not None:
                self.copilot_tune_button.setText('📈 Βελτίωση με βάση τα αποτελέσματα εκπαίδευσης')
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_detection_button') and self.copilot_detection_button is not None:
                self.copilot_detection_button.setText('🧪 Βελτίωση με βάση τα αποτελέσματα ανίχνευσης')
        except Exception:
            pass
        try:
            self._update_copilot_buttons_for_llm(show_message=False)
        except Exception:
            pass

    def on_training_completed(self):
        try:
            self._update_copilot_buttons_for_llm()
        except Exception:
            pass
        try:
            main_window = self.window()
            if main_window is not None:
                if hasattr(main_window, 'statistics_tab') and main_window.statistics_tab is not None:
                    if hasattr(main_window.statistics_tab, 'refresh_data'):
                        main_window.statistics_tab.refresh_data()
                if hasattr(main_window, 'camera_tab') and main_window.camera_tab is not None:
                    if hasattr(main_window.camera_tab, 'refresh_models'):
                        main_window.camera_tab.refresh_models()
                if hasattr(main_window, 'benchmark_tab') and main_window.benchmark_tab is not None:
                    if hasattr(main_window.benchmark_tab, 'refresh_models'):
                        main_window.benchmark_tab.refresh_models()
                if hasattr(main_window, 'camera_benchmark_tab') and main_window.camera_benchmark_tab is not None:
                    if hasattr(main_window.camera_benchmark_tab, 'refresh_models'):
                        main_window.camera_benchmark_tab.refresh_models()
        except Exception:
            pass

    def on_detection_analysis_completed(self):
        if not getattr(self, '_pending_detection_from_copilot', False):
            return
        self._pending_detection_from_copilot = False
        try:
            self.on_copilot_tune_from_detection()
        except Exception as e:
            QMessageBox.critical(self, 'Model Training Copilot', f'Προέκυψε σφάλμα κατά την επεξεργασία των αποτελεσμάτων ανάλυσης ανίχνευσης:\n{e}')

    # ── Δημιουργία context string για αποστολή στο LLM ─────────────────────────
    # Συλλέγει:
    #   - Τρέχουσες ρυθμίσεις GUI (μοντέλο, dataset, epochs, device κ.λπ.)
    #   - Hyperparameters (batch, lr0, lrf, momentum, weight_decay, optimizer...)
    #   - Hardware info (CPU, RAM, GPU, CUDA, Python, PyTorch versions)
    #   - Διαθέσιμες επιλογές (λίστες από combos)
    #   - Σημείωση για CNN μοντέλα (Triton=false, optimizer=adam κ.λπ.)
    def _build_copilot_context(self) -> str:
        tt = self.training_tab
        try:
            model_name = tt.model_combo.currentText()
        except Exception:
            model_name = 'άγνωστο'
        try:
            dataset_name = tt.dataset_combo.currentText()
        except Exception:
            dataset_name = 'άγνωστο'
        try:
            imgsz = tt.imgsz_combo.currentText()
        except Exception:
            imgsz = '640'
        try:
            device = tt.device_combo.currentText()
        except Exception:
            device = 'CPU'
        try:
            epochs = tt.epochs_spin.value()
        except Exception:
            epochs = 50
        try:
            patience = tt.patience_spin.value()
        except Exception:
            patience = 10
        triton_enabled = bool(getattr(tt, 'use_triton', False))
        compile_mode = getattr(tt, 'compile_mode', 'Προεπιλογή')
        available_models = []
        try:
            for i in range(tt.model_combo.count()):
                available_models.append(tt.model_combo.itemText(i))
        except Exception:
            pass
        available_datasets = []
        try:
            for i in range(tt.dataset_combo.count()):
                available_datasets.append(tt.dataset_combo.itemText(i))
        except Exception:
            pass
        available_imgsz = []
        try:
            for i in range(tt.imgsz_combo.count()):
                available_imgsz.append(tt.imgsz_combo.itemText(i))
        except Exception:
            pass
        available_devices = []
        try:
            for i in range(tt.device_combo.count()):
                available_devices.append(tt.device_combo.itemText(i))
        except Exception:
            pass
        batch = getattr(tt, 'batch_spin', None).value() if hasattr(tt, 'batch_spin') else None
        lr0 = getattr(tt, 'lr0_spin', None).value() if hasattr(tt, 'lr0_spin') else None
        lrf = getattr(tt, 'lrf_spin', None).value() if hasattr(tt, 'lrf_spin') else None
        momentum = getattr(tt, 'momentum_spin', None).value() if hasattr(tt, 'momentum_spin') else None
        weight_decay = getattr(tt, 'weight_decay_spin', None).value() if hasattr(tt, 'weight_decay_spin') else None
        warmup_epochs = getattr(tt, 'warmup_epochs_spin', None).value() if hasattr(tt, 'warmup_epochs_spin') else None
        optimizer = getattr(tt, 'optimizer_combo', None).currentText() if hasattr(tt, 'optimizer_combo') else 'auto'
        workers = getattr(tt, 'workers_spin', None).value() if hasattr(tt, 'workers_spin') else None
        os_name = ''
        machine = ''
        cpu_count = None
        cpu_info_text = ''
        python_ver = ''
        total_mem_gb = None
        cuda_available = False
        gpu_name = ''
        gpu_total_gb = None
        try:
            import platform as _platform
            import sys as _sys
            import psutil as _psutil
            try:
                os_name = f'{_platform.system()} {_platform.release()}'
            except Exception:
                os_name = ''
            try:
                machine = _platform.machine() or ''
            except Exception:
                machine = ''
            try:
                cpu_count = os.cpu_count() or 0
            except Exception:
                cpu_count = None
            try:
                cpu_info_text = _platform.processor() or ''
            except Exception:
                cpu_info_text = ''
            try:
                python_ver = _sys.version.split()[0]
            except Exception:
                python_ver = ''
            try:
                total_mem_gb = _psutil.virtual_memory().total / 1024 ** 3
            except Exception:
                total_mem_gb = None
        except Exception:
            pass
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            except Exception:
                gpu_name = ''
                gpu_total_gb = None
        if total_mem_gb is not None:
            ram_str = f'{total_mem_gb:.2f}'
        else:
            ram_str = 'άγνωστο'
        if gpu_total_gb is not None:
            gpu_ram_str = f'{gpu_total_gb:.2f}'
        else:
            gpu_ram_str = 'άγνωστο'
        lines = ['ΤΡΕΧΟΥΣΕΣ ΡΥΘΜΙΣΕΙΣ:', f'- model_name: {model_name}', f'- dataset_name: {dataset_name}', f'- image_size: {imgsz}', f'- epochs: {epochs}', f'- patience: {patience}', f'- device: {device}', f'- triton_enabled: {triton_enabled}', f'- compile_mode: {compile_mode}', '', 'ΤΡΕΧΟΝΤΕΣ ΥΠΕΡ-ΠΑΡΑΜΕΤΡΟΙ:', f'- batch: {batch}', f'- optimizer: {optimizer}', f'- lr0: {lr0}', f'- lrf: {lrf}', f'- momentum: {momentum}', f'- weight_decay: {weight_decay}', f'- warmup_epochs: {warmup_epochs}', f'- workers: {workers}', '', 'ΠΛΗΡΟΦΟΡΙΕΣ ΣΥΣΤΗΜΑΤΟΣ / ΠΟΡΩΝ:', f'- os_name: {os_name}', f'- machine: {machine}', f'- python_version: {python_ver}', f'- cpu_info: {cpu_info_text}', f'- cpu_cores: {cpu_count}', f'- total_ram_gb: {ram_str}', f'- cuda_available: {cuda_available}', f'- gpu_name: {gpu_name}', f'- gpu_total_gb: {gpu_ram_str}', '', 'ΔΙΑΘΕΣΙΜΕΣ ΕΠΙΛΟΓΕΣ:', f"- διαθέσιμα μοντέλα: {', '.join(available_models)}", f"- διαθέσιμα datasets: {', '.join(available_datasets)}", f"- διαθέσιμα image sizes: {', '.join(available_imgsz)}", f"- διαθέσιμες συσκευές: {', '.join(available_devices)}", '', 'ΣΗΜΕΙΩΣΗ CNN ΜΟΝΤΕΛΑ:', '- CNN μοντέλα (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, resnet50, resnet101) είναι torchvision classifiers.', '- Δεν υποστηρίζουν Triton/TorchCompile και απαιτούν dataset ταξινόμησης (train/<class>/... val/<class>/...).', '- Χρησιμοποίησε optimizer: adam/adamw/sgd. Υποστηρίζουν ONNX export αλλά ΟΧΙ TensorRT/NCNN.']
        return '\n'.join(lines)

    def _require_groq_key(self) -> bool:
        if has_valid_groq_api_key():
            return True
        try:
            self._update_copilot_buttons_for_llm(show_message=True)
        except Exception:
            pass
        return False

    def _launch_copilot_thread(self):
        self.copilot_worker.moveToThread(self.copilot_thread)
        self.copilot_thread.started.connect(self.copilot_worker.run)
        self.copilot_worker.finished.connect(self.on_copilot_result)
        self.copilot_worker.error.connect(self.on_copilot_error)
        self.copilot_worker.error.connect(self.copilot_thread.quit)
        self.copilot_worker.finished.connect(self.copilot_thread.quit)
        self.copilot_worker.finished.connect(self.copilot_worker.deleteLater)
        self.copilot_thread.finished.connect(self.copilot_thread.deleteLater)
        self.copilot_thread.start()

    def on_copilot_suggest_clicked(self):
        try:
            if not self._require_groq_key():
                return
            text_prompt = self.copilot_prompt_edit.toPlainText().strip()
            if not text_prompt:
                QMessageBox.warning(self, 'Model Training Copilot', 'Γράψε πρώτα μια σύντομη περιγραφή του τι θέλεις να πετύχεις.')
            system_prompt = "Είσαι βοηθός εκπαίδευσης YOLO (Ultralytics) μέσα στην εφαρμογή 'Models Manager Pro (A.I Copilot) Ver 4.0'. Λαμβάνεις τις τρέχουσες ρυθμίσεις και τις διαθέσιμες επιλογές (μοντέλα, datasets, image sizes, συσκευές, υπερ-παράμετροι) και μία περιγραφή του στόχου του χρήστη. Προτείνεις βελτιωμένες ρυθμίσεις εκπαίδευσης με σύντομη αλλά πολύ συγκεκριμένη αιτιολόγηση. Απάντησε ΠΑΝΤΑ στα Ελληνικά. ΣΤΟ ΤΕΛΟΣ της απάντησης, ΠΑΝΤΑ πρόσθεσε ΔΥΟ μπλοκ YAML μέσα σε ```yaml ... ``` με: 1) gui_config με κλειδιά model_name, dataset_name, image_size, epochs, patience, device. 2) train_hyperparams με κλειδιά batch, optimizer, lr0, lrf, momentum, weight_decay, warmup_epochs, workers, triton_enabled, compile_mode. Για το 'triton_enabled' χρησιμοποίησε μόνο boolean true/false. Για το 'compile_mode' χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ μία από τις τιμές: 'Προεπιλογή', 'Μείωση επιβάρυνσης', 'Μέγιστος αυτόματος συντονισμός'. ΜΗΝ χρησιμοποιείς άλλες τιμές (π.χ. True, False, 1, 'on', 'auto', 'max'). Χρησιμοποίησε μόνο τιμές που υπάρχουν στις διαθέσιμες επιλογές όπου είναι δυνατό."
            system_prompt = TRAINING_COPILOT_SYSTEM_PROMPT
            self._copilot_expect_yaml = True
            context = self._build_copilot_context()
            user_message = f'Περιγραφή χρήστη:\n{text_prompt}\n\nΤρέχουσες ρυθμίσεις και υπερ-παράμετροι:\n{context}\n'
            self.copilot_button.setEnabled(True)
            self.copilot_button.setText('⏳ Ζητούνται προτάσεις από LLM API...')
            self.copilot_output_edit.clear()
            self.copilot_output_edit.appendPlainText('Ζητώ προτάσεις από το LLM API...\n')
            self.copilot_apply_button.setEnabled(True)
            self.copilot_last_gui_cfg = None
            self.copilot_last_hparams = None
            set_copilot_busy(self, True)
            self.copilot_thread = QThread()
            self.copilot_worker = LLMWorker(system_prompt=system_prompt, user_message=user_message, model=get_current_llm_model())
            self._launch_copilot_thread()
        except Exception as exc:
            self._handle_unexpected_copilot_exception('on_copilot_suggest_clicked', exc)

    def on_copilot_tune_from_last_run(self):
        try:
            if not self._require_groq_key():
                return
            self._last_copilot_model_info = ''
            tt = self.training_tab
            log_text = ''
            try:
                selected_index = -1
                model_path = None
                if hasattr(tt, 'export_model_combo') and hasattr(tt, 'available_models'):
                    selected_index = tt.export_model_combo.currentIndex()
                    if 0 <= selected_index < len(tt.available_models):
                        model_path, model_type = tt.available_models[selected_index]
                if model_path is not None:
                    stem = model_path.stem
                    parts = stem.split('_')
                    if len(parts) >= 4:
                        imgsz_str = parts[-1]
                        dataset_name = parts[-2]
                        device_token = parts[-3]
                        model_name = '_'.join(parts[:-3])
                        try:
                            imgsz = int(imgsz_str)
                            run_device = 'CUDA' if device_token.upper() == 'GPU' else 'CPU'
                            project_prefix = f'Runs_{model_name}_{run_device}'
                            project_name = f'Finetuned_{model_name}_{dataset_name}_imgsz{imgsz}'
                            project_dir = Path(project_prefix) / project_name
                            log_file = project_dir / 'training.log'
                            if log_file.exists():
                                log_text = log_file.read_text(encoding='utf-8', errors='ignore')
                            device_human = 'GPU' if device_token.upper() == 'GPU' else 'CPU'
                            self._last_copilot_model_info = f'📌 Χαρακτηριστικά εκπαιδευμένου μοντέλου (PyTorch):\n  – Βασικό μοντέλο: {model_name}\n  – Dataset εκπαίδευσης: {dataset_name}\n  – Συσκευή εκπαίδευσης: {device_human}\n  – Ανάλυση εικόνας: {imgsz}\n'
                        except Exception:
                            log_text = log_text or ''
            except Exception:
                log_text = log_text or ''
            if not log_text.strip():
                try:
                    log_text = tt.log_output.toPlainText()
                except Exception:
                    log_text = ''
            summary = log_text
            idx = summary.rfind('FINAL METRICS')
            if idx != -1:
                summary = summary[idx:]
            else:
                idx2 = summary.rfind('ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ')
                if idx2 != -1:
                    start_idx = max(0, idx2 - 1200)
                    summary = summary[start_idx:]
            context = self._build_copilot_context()
            extra_user = self.copilot_prompt_edit.toPlainText().strip()
            if extra_user:
                extra_part = f'Επιπλέον επιθυμία / στόχος χρήστη:\n{extra_user}\n\n'
            else:
                extra_part = ''
            system_prompt = "Είσαι εξειδικευμένος βοηθός εκπαίδευσης YOLO (Ultralytics) μέσα στην εφαρμογή 'Models Manager Pro (A.I Copilot) Ver 4.0'. Λαμβάνεις ΠΑΝΤΑ ως input ένα context με τρεις ενότητες: 1) Τρέχουσες ρυθμίσεις και υπερ-παραμέτρους εκπαίδευσης (από το Tab '🎓 Εκπαίδευση Μοντέλου'). 2) Διαθέσιμες επιλογές του GUI (μοντέλα, datasets, image sizes, συσκευές). 3) Πληροφορίες συστήματος / hardware (OS, CPU, RAM, GPU, CUDA διαθεσιμότητα, Triton, compile_mode).    Η παράμετρος 'triton_enabled' αντιστοιχεί στο checkbox 'Triton (TorchInductor)' στο Tab '🎓 Εκπαίδευση Μοντέλου'    (True όταν το checkbox είναι επιλεγμένο, False όταν είναι απενεργοποιημένο).    Η παράμετρος 'compile_mode' αντιστοιχεί στη λίστα 'Compiling' δίπλα από το checkbox με τις επιλογές    'Προεπιλογή', 'Μείωση επιβάρυνσης', 'Μέγιστος αυτόματος συντονισμός'. Επιπλέον, λαμβάνεις λεπτομερές LOG από την εκπαίδευση του επιλεγμένου μοντέλου (αρχείο training.log, με χρόνους, epochs και metrics). Δεν κάνεις αυθαίρετες υποθέσεις έξω από αυτά τα δεδομένα. Στόχος σου είναι να προτείνεις ΡΕΑΛΙΣΤΙΚΕΣ και ΠΡΑΚΤΙΚΕΣ ρυθμίσεις εκπαίδευσης, λαμβάνοντας υπόψη: – τον στόχο του χρήστη, – τους διαθέσιμους πόρους (RAM / GPU / CUDA / workers / CPU cores), – τις επιλογές που πραγματικά υπάρχουν στο GUI, – το αν το Triton / TorchInductor είναι διαθέσιμο και ενεργοποιημένο, – τα πραγματικά αποτελέσματα της τελευταίας εκπαίδευσης (ιδιαίτερα mAP, precision, recall, χρόνους). ΠΡΟΣΕΓΓΙΣΗ: 1) Δώσε πρώτα μια σύντομη περίληψη (2–4 προτάσεις) στα Ελληνικά,    εξηγώντας ΤΙ πήγε καλά και ΤΙ θα ήθελες να βελτιώσεις σε σχέση με την τελευταία εκπαίδευση. 2) Έπειτα εξήγησε με bullets ΠΟΙΕΣ ρυθμίσεις προτείνεις να αλλάξουν (GUI + hyperparameters)    και ΠΩΣ αυτές συνδέονται με τα αποτελέσματα του log (π.χ. overfitting, underfitting, αργή ταχύτητα κ.λπ.). 3) ΜΗΝ αλλάζεις model_name ή dataset_name αν ο χρήστης δεν το ζητάει ρητά, εκτός αν το dataset είναι προφανώς ακατάλληλο. 4) Σε κάθε πρόταση εξήγησε ΣΥΝΟΠΤΙΚΑ γιατί τροποποιείς κάθε υπερπαράμετρο (lr0, lrf, momentum, weight_decay, warmup_epochs, batch, workers). 5) Αν ο χρόνος ανά epoch ήταν μεγάλος ή η RAM/GPU είναι οριακή, πρότεινε πιο συντηρητικές ρυθμίσεις    (μικρότερο batch, προσεκτικά workers, ενδεχομένως λιγότερα epochs ή μεγαλύτερο patience μόνο αν χρειάζεται). 6) Αν τα metrics δείχνουν underfitting ή μικρή ακρίβεια, μπορείς να προτείνεις περισσότερα epochs,    προσαρμογή lr0/lrf και ενδεχομένως διαφορετικό optimizer, πάντα εξηγώντας ξεκάθαρα τον συμβιβασμό. Απάντησε ΠΑΝΤΑ στα Ελληνικά. ΣΤΟ ΤΕΛΟΣ της απάντησης, ΠΑΝΤΑ πρόσθεσε δύο μπλοκ YAML μέσα σε ```yaml ... ``` ως εξής: 1) Ένα μπλοκ με τίτλο 'gui_config' με ΑΚΡΙΒΩΣ τα κλειδιά: model_name, dataset_name, image_size, epochs, patience, device. 2) Ένα μπλοκ με τίτλο 'train_hyperparams' με ΑΚΡΙΒΩΣ τα κλειδιά: batch, optimizer, lr0, lrf, momentum, weight_decay, warmup_epochs, workers, triton_enabled, compile_mode. Χρησιμοποίησε ΜΟΝΟ τιμές που υπάρχουν στις διαθέσιμες επιλογές όπου είναι δυνατό και ΜΗΝ χρησιμοποιείς επιπλέον κλειδιά πέρα από αυτά."
            system_prompt = TRAINING_COPILOT_SYSTEM_PROMPT
            self._copilot_expect_yaml = True
            if not getattr(self, '_last_copilot_model_info', ''):
                try:
                    tt = self.training_tab
                    if hasattr(tt, 'export_model_combo') and tt.export_model_combo.currentText():
                        fname = tt.export_model_combo.currentText()
                        self._last_copilot_model_info = f'📌 Χαρακτηριστικά εκπαιδευμένου μοντέλου (PyTorch):\n  – Αρχείο μοντέλου: {fname}\n'
                except Exception:
                    pass
            header_info = self._last_copilot_model_info + '\n\n' if getattr(self, '_last_copilot_model_info', '') else ''
            user_message = f'Ανάλυσε τα παρακάτω δεδομένα και πρότεινε βελτιωμένες ρυθμίσεις για την επόμενη εκπαίδευση.\n\n{header_info}Τρέχουσες ρυθμίσεις εκπαίδευσης και διαθέσιμες επιλογές:\n{context}\n\nΑποτελέσματα τελευταίας εκπαίδευσης (log):\n{summary}\n\n{extra_part}'
            self.copilot_button.setEnabled(True)
            self.copilot_tune_button.setEnabled(True)
            self.copilot_output_edit.clear()
            self.copilot_last_yaml = None
            self.copilot_last_hparams = None
            set_copilot_busy(self, True)
            self.copilot_thread = QThread()
            self.copilot_worker = LLMWorker(system_prompt=system_prompt, user_message=user_message, model=get_current_llm_model())
            self._launch_copilot_thread()
        except Exception as exc:
            self._handle_unexpected_copilot_exception('on_copilot_tune_from_last_run', exc)

    def on_copilot_detection_button_clicked(self):
        try:
            if not self._require_groq_key():
                return
            try:
                self._last_copilot_model_info = ''
            except Exception:
                pass
            stats_tab = getattr(self, 'statistics_tab', None)
            if stats_tab is None:
                QMessageBox.warning(self, 'Model Training Copilot', "Το Tab 'Στατιστικά Ανίχνευσης' δεν είναι διαθέσιμο.\nΒεβαιώσου ότι η καρτέλα στατιστικών έχει δημιουργηθεί σωστά.")
            try:
                try:
                    self._pending_detection_from_copilot = True
                except Exception:
                    pass
                if hasattr(stats_tab, 'start_analysis'):
                    stats_tab.start_analysis()
                    try:
                        if hasattr(self, 'copilot_output_edit'):
                            self.copilot_output_edit.clear()
                            self.copilot_output_edit.appendPlainText("🧪 Εκκίνηση αυτόματης ανάλυσης ανίχνευσης από το Copilot...\nΜόλις ολοκληρωθεί η ανάλυση στο Tab 'Στατιστικά Ανίχνευσης',\nθα γίνει αυτόματα βελτίωση των ρυθμίσεων εκπαίδευσης με βάση τα αποτελέσματα.")
                    except Exception:
                        pass
                    try:
                        if hasattr(self, 'copilot_detection_button'):
                            self.copilot_detection_button.setEnabled(False)
                    except Exception:
                        pass
                else:
                    QMessageBox.warning(self, 'Model Training Copilot', "Δεν βρέθηκε μέθοδος 'start_analysis' στο Tab 'Στατιστικά Ανίχνευσης'.")
            except Exception as e:
                QMessageBox.critical(self, 'Model Training Copilot', f'Σφάλμα κατά την εκκίνηση της ανάλυσης ανίχνευσης:\n{e}')
        except Exception as exc:
            self._handle_unexpected_copilot_exception('on_copilot_detection_button_clicked', exc)

    def on_copilot_tune_from_detection(self):
        try:
            if not self._require_groq_key():
                return
            self._last_copilot_model_info = ''
            stats_tab = getattr(self, 'statistics_tab', None)
            if stats_tab is None:
                QMessageBox.warning(self, 'Model Training Copilot', "Το Tab 'Στατιστικά Ανίχνευσης' δεν είναι διαθέσιμο.\nΒεβαιώσου ότι η καρτέλα στατιστικών έχει δημιουργηθεί σωστά.")
            if not getattr(stats_tab, 'available_models', []) or len(stats_tab.available_models) == 0:
                QMessageBox.warning(self, 'Model Training Copilot', 'Δεν υπάρχουν εκπαιδευμένα μοντέλα για ανάλυση ανίχνευσης.\nΠαρακαλώ εκπαιδεύστε πρώτα ένα μοντέλο.')
            try:
                if hasattr(self, 'copilot_output_edit'):
                    self.copilot_output_edit.clear()
                    self.copilot_output_edit.appendPlainText("▶️ Έναρξη λειτουργίας 'Βελτίωση με βάση την ανάλυση των αποτελεσμάτων ανίχνευσης' από Copilot...\n")
            except Exception:
                pass
            stats_tab = getattr(self, 'statistics_tab', None)
            if stats_tab is None:
                QMessageBox.warning(self, 'Model Training Copilot', "Το Tab 'Στατιστικά Ανίχνευσης' δεν είναι διαθέσιμο.\nΒεβαιώσου ότι η καρτέλα στατιστικών έχει δημιουργηθεί σωστά.")
            summary_text = ''
            log_text = ''
            training_log_text = ''
            try:
                if hasattr(stats_tab, 'summary_output'):
                    summary_text = stats_tab.summary_output.toPlainText().strip()
            except Exception:
                summary_text = ''
            try:
                if hasattr(stats_tab, 'log_output'):
                    log_text = stats_tab.log_output.toPlainText().strip()
            except Exception:
                log_text = ''
            if not summary_text and (not log_text):
                try:
                    if not hasattr(stats_tab, 'model_combo') or stats_tab.model_combo.count() == 0 or stats_tab.model_combo.currentIndex() < 0:
                        QMessageBox.warning(self, 'Model Training Copilot', "Δεν υπάρχει επιλεγμένο μοντέλο στο Tab 'Στατιστικά Ανίχνευσης'.\nΕπίλεξε πρώτα το μοντέλο με το οποίο θέλεις να γίνει η ανάλυση.")
                    if not hasattr(stats_tab, 'dataset_combo') or stats_tab.dataset_combo.count() == 0 or stats_tab.dataset_combo.currentIndex() < 0:
                        QMessageBox.warning(self, 'Model Training Copilot', "Δεν υπάρχει επιλεγμένο dataset στο Tab 'Στατιστικά Ανίχνευσης'.\nΕπίλεξε πρώτα το κατάλληλο dataset για ανάλυση.")
                except Exception:
                    pass
                self._pending_detection_from_copilot = True
                try:
                    if hasattr(stats_tab, 'start_analysis'):
                        stats_tab.start_analysis()
                        self.copilot_output_edit.clear()
                        self.copilot_output_edit.appendPlainText("🧪 Εκκίνηση αυτόματης ανάλυσης ανίχνευσης από το Copilot...\nΜόλις ολοκληρωθεί η ανάλυση στο Tab 'Στατιστικά Ανίχνευσης',\nθα αναλυθούν τα αποτελέσματα και θα προταθούν ρυθμίσεις εκπαίδευσης.")
                        if hasattr(self, 'copilot_detection_button'):
                            self.copilot_detection_button.setEnabled(False)
                    else:
                        QMessageBox.warning(self, 'Model Training Copilot', "Δεν βρέθηκε μέθοδος 'start_analysis' στο Tab Στατιστικά Ανίχνευσης.")
                except Exception as e:
                    QMessageBox.critical(self, 'Model Training Copilot', f'Σφάλμα κατά την εκκίνηση της ανάλυσης ανίχνευσης:\n{e}')
            analysis_model = ''
            analysis_dataset = ''
            try:
                if hasattr(stats_tab, 'model_combo'):
                    analysis_model = stats_tab.model_combo.currentText()
            except Exception:
                analysis_model = ''
            try:
                if hasattr(stats_tab, 'dataset_combo'):
                    analysis_dataset = stats_tab.dataset_combo.currentText()
            except Exception:
                analysis_dataset = ''
            context = self._build_copilot_context()
            extra_user = self.copilot_prompt_edit.toPlainText().strip()
            if extra_user:
                extra_part = f'Επιπλέον επιθυμία / στόχος χρήστη:\n{extra_user}\n\n'
            else:
                extra_part = ''
            try:
                if hasattr(stats_tab, 'model_combo') and hasattr(stats_tab, 'available_models'):
                    idx = stats_tab.model_combo.currentIndex()
                    if 0 <= idx < len(stats_tab.available_models):
                        model_path, model_type = stats_tab.available_models[idx]
                        stem = model_path.name
                        base = stem.replace('_ncnn_model', '')
                        if model_path.is_file() and '.' in base:
                            base = base.rsplit('.', 1)[0]
                        parts = base.split('_')
                        if len(parts) >= 4:
                            imgsz_str = parts[-1]
                            dataset_name = parts[-2]
                            device_token = parts[-3]
                            model_name = '_'.join(parts[:-3])
                            try:
                                imgsz = int(imgsz_str)
                            except Exception:
                                imgsz = imgsz_str
                            device_human = 'GPU' if device_token.upper() == 'GPU' else 'CPU'
                            self._last_copilot_model_info = f'📌 Χαρακτηριστικά εκπαιδευμένου μοντέλου (PyTorch):\n  – Βασικό μοντέλο: {model_name}\n  – Dataset εκπαίδευσης: {dataset_name}\n  – Συσκευή εκπαίδευσης: {device_human}\n  – Ανάλυση εικόνας: {imgsz}\n'
                            try:
                                run_device = 'CUDA' if device_token.upper() == 'GPU' else 'CPU'
                                project_prefix = f'Runs_{model_name}_{run_device}'
                                project_name = f'Finetuned_{model_name}_{dataset_name}_imgsz{imgsz}'
                                project_dir = Path(project_prefix) / project_name
                                log_file = project_dir / 'training.log'
                                if log_file.exists():
                                    training_log_text = log_file.read_text(encoding='utf-8', errors='ignore')
                            except Exception:
                                training_log_text = training_log_text or ''
            except Exception:
                pass
            self.copilot_output_edit.clear()
            self.copilot_output_edit.appendPlainText('=== ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ ΑΝΙΧΝΕΥΣΗΣ ===\n')
            if analysis_model or analysis_dataset:
                self.copilot_output_edit.appendPlainText(f'Μοντέλο ανάλυσης: {analysis_model}\nDataset ανάλυσης: {analysis_dataset}\n')
            self.copilot_output_edit.appendPlainText('\n=== ΠΡΟΤΑΣΕΙΣ COPILOT (με βάση τα αποτελέσματα ανάλυσης) ===\n\n')
            system_prompt = "Είσαι εξειδικευμένος βοηθός εκπαίδευσης YOLO (Ultralytics) μέσα στην εφαρμογή 'Models Manager Pro A.I Ver 4.0'. Σε αυτό το mode λαμβάνεις επιπλέον αποτελέσματα από ΑΝΑΛΥΣΗ ΑΝΙΧΝΕΥΣΗΣ (Detection Statistics) που παράγονται από το Tab 'Στατιστικά Ανίχνευσης'. Τα δεδομένα περιγράφουν την απόδοση ενός εκπαιδευμένου μοντέλου πάνω σε ένα dataset εικόνων (π.χ. ανά κλάση: TP, FP, FN, precision, recall, F1, confusion patterns, δύσκολες κλάσεις). Επιπλέον, λαμβάνεις και το αρχείο training.log από την εκπαίδευση του ίδιου μοντέλου, όπως προέκυψε από το Tab '🎓 Εκπαίδευση Μοντέλου'. Στόχος σου είναι να προτείνεις νέες ΡΕΑΛΙΣΤΙΚΕΣ ρυθμίσεις εκπαίδευσης που βελτιώνουν την απόδοση σε ανίχνευση, λαμβάνοντας υπόψη:\n– Τις τρέχουσες ρυθμίσεις του GUI (Tab 'Εκπαίδευση Μοντέλου').\n– Το διαθέσιμο hardware (CPU / GPU, RAM, workers, Triton, compile_mode).\n– Το διαθέσιμο hardware (CPU / GPU, RAM, workers, Triton, compile_mode).\n   Η παράμετρος 'triton_enabled' αντιστοιχεί στο checkbox 'Triton (TorchInductor)' στο Tab '🎓 Εκπαίδευση Μοντέλου' \n   (True όταν το checkbox είναι επιλεγμένο, False όταν είναι απενεργοποιημένο). \n   Η παράμετρος 'compile_mode' αντιστοιχεί στη λίστα 'Compiling' δίπλα από το checkbox με τις επιλογές \n   'Προεπιλογή', 'Μείωση επιβάρυνσης', 'Μέγιστος αυτόματος συντονισμός'.\n   Για το κλειδί 'compile_mode' στο YAML χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ μία από αυτές τις τρεις ακριβείς τιμές, χωρίς booleans ή άλλες συμβολοσειρές.\n   Αν ο χρήστης ζητήσει ενεργοποίηση compile για μέγιστη επιτάχυνση, αντιστοίχισέ το στην τιμή 'Μέγιστος αυτόματος συντονισμός'.\n– Τα αποτελέσματα ανάλυσης ανίχνευσης (classes με χαμηλό recall, πολλά false positives, κ.λπ.).\nΠΡΟΣΕΓΓΙΣΗ:\n1) Δώσε μία σύντομη περίληψη (2–4 προτάσεις) στα Ελληνικά για το τι δείχνουν τα αποτελέσματα ανίχνευσης (π.χ. ποιες κλάσεις είναι προβληματικές, over/under-detections).\n2) Στη συνέχεια, με bullets, εξήγησε ποιες ρυθμίσεις εκπαίδευσης (epochs, batch, lr0, lrf, momentum, weight_decay,    warmup_epochs, optimizer, workers) προτείνεις να αλλάξουν και γιατί, σε σχέση με τα detection metrics.\n3) ΜΗΝ αλλάξεις model_name ή dataset_name, εκτός αν τα δεδομένα δείχνουν ξεκάθαρα ότι χρησιμοποιείται    λάθος μοντέλο ή dataset για το πρόβλημα (σε αυτή την περίπτωση εξήγησε πολύ καθαρά γιατί).\n4) Αν υπάρχουν πολλές false positives, σκέψου πιο συντηρητικές ρυθμίσεις (π.χ. διαφορετικό lr0/lrf,    ενδεχομένως περισσότερα epochs ή άλλον optimizer) και εξήγησε τον συμβιβασμό.\n5) Αν υπάρχουν πολλές false negatives ή χαμηλό recall σε συγκεκριμένες κλάσεις, μπορείς να προτείνεις    πιο εντατική εκπαίδευση (περισσότερα epochs, προσεκτική αύξηση lr0 ή διαφορετικό scheduler) αλλά    πάντα μέσα στα όρια των διαθέσιμων πόρων.\n6) Απάντησε ΠΑΝΤΑ στα Ελληνικά.\nΣΤΟ ΤΕΛΟΣ της απάντησης, ΠΑΝΤΑ πρόσθεσε δύο μπλοκ YAML μέσα σε ```yaml ... ``` ως εξής:\n1) Ένα μπλοκ με τίτλο 'gui_config' με ΑΚΡΙΒΩΣ τα κλειδιά: model_name, dataset_name, image_size, epochs, patience, device.\n2) Ένα μπλοκ με τίτλο 'train_hyperparams' με ΑΚΡΙΒΩΣ τα κλειδιά: batch, optimizer, lr0, lrf, momentum, weight_decay, warmup_epochs, workers, triton_enabled, compile_mode.\nΧρησιμοποίησε ΜΟΝΟ τιμές που υπάρχουν στις διαθέσιμες επιλογές όπου είναι δυνατό και ΜΗΝ χρησιμοποιείς επιπλέον κλειδιά πέρα από αυτά."
            system_prompt = TRAINING_COPILOT_SYSTEM_PROMPT
            self._copilot_expect_yaml = True
            training_summary = ''
            if training_log_text:
                training_summary = training_log_text
                idx_ts = training_summary.rfind('FINAL METRICS')
                if idx_ts != -1:
                    training_summary = training_summary[idx_ts:]
                else:
                    idx_ts2 = training_summary.rfind('ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ')
                    if idx_ts2 != -1:
                        start_idx = max(0, idx_ts2 - 1200)
                        training_summary = training_summary[start_idx:]
            detection_context = f'ΠΛΗΡΟΦΟΡΙΕΣ ΑΝΑΛΥΣΗΣ ΑΝΙΧΝΕΥΣΗΣ (Statistics Tab):\nΜοντέλο ανάλυσης: {analysis_model}\nDataset ανάλυσης: {analysis_dataset}\n\nΠερίληψη ανάλυσης (summary):\n{summary_text}\n\nΑναλυτικό log ανάλυσης:\n{log_text}\n'
            if training_summary:
                detection_context += f'\nΑποτελέσματα εκπαίδευσης (training.log) του επιλεγμένου μοντέλου:\n{training_summary}\n'
            if not getattr(self, '_last_copilot_model_info', ''):
                try:
                    stats_tab = getattr(self, 'statistics_tab', None)
                    if stats_tab is not None and hasattr(stats_tab, 'model_combo'):
                        mname = stats_tab.model_combo.currentText()
                        if mname:
                            self._last_copilot_model_info = f'📌 Χαρακτηριστικά εκπαιδευμένου μοντέλου (PyTorch):\n  – Μοντέλο (όνομα λίστας): {mname}\n'
                except Exception:
                    pass
            header_info = self._last_copilot_model_info + '\n\n' if getattr(self, '_last_copilot_model_info', '') else ''
            user_message = f'{header_info}Με βάση τα παρακάτω δεδομένα, θα κάνεις τρία πράγματα:\n1) ΠΡΩΤΑ να σχολιάσεις αναλυτικά τη ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΑΝΙΧΝΕΥΣΗΣ. Η απάντησή σου πρέπει να ξεκινά ΠΑΝΤΑ με ενότητα με τίτλο "Σχολιασμός σύνοψης αποτελεσμάτων ανίχνευσης" και να περιγράφει τι δείχνουν τα metrics.\n2) Έπειτα να προτείνεις βελτιώσεις στις ρυθμίσεις εκπαίδευσης με βάση τόσο τα αποτελέσματα ανίχνευσης όσο και τα στοιχεία εκπαίδευσης.\n3) Τέλος να δώσεις συνοπτικό προτεινόμενο σετ ρυθμίσεων σε YAML, όπως ορίζεται στο system prompt.\n\n### Στοιχεία εκπαίδευσης και περιβάλλοντος (από το Tab Εκπαίδευση Μοντέλου):\n{context}\n\n### Πλαίσιο ανίχνευσης (από το Tab Στατιστικά Ανίχνευσης):\n{detection_context}\n{extra_part}'
            self.copilot_button.setEnabled(True)
            self.copilot_button.setText('⏳ Ζητούνται προτάσεις από LLM API...')
            if hasattr(self, 'copilot_tune_button'):
                self.copilot_tune_button.setEnabled(True)
            if hasattr(self, 'copilot_detection_button'):
                self.copilot_detection_button.setEnabled(True)
            self.copilot_last_yaml = None
            self.copilot_last_hparams = None
            set_copilot_busy(self, True)
            self.copilot_thread = QThread()
            self.copilot_worker = LLMWorker(system_prompt=system_prompt, user_message=user_message, model=get_current_llm_model())
            self._launch_copilot_thread()
        except Exception as exc:
            self._handle_unexpected_copilot_exception('on_copilot_tune_from_detection', exc)

    def on_llm_settings_clicked(self):
        dialog = LLMSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            base_url, api_key, model = dialog.get_values()
            configure_llm(base_url=base_url or BASE_URL, api_key=api_key or API_KEY, model=model or get_current_llm_model())
            info_text = f'Χρησιμοποιείται Groq base URL: {BASE_URL}\nΠροεπιλεγμένο LLM model: {get_current_llm_model()}'
            self._last_copilot_model_info = info_text
            try:
                self._update_copilot_buttons_for_llm(show_message=True)
            except Exception:
                pass
            QMessageBox.information(self, 'Ρυθμίσεις LLM', 'Οι ρυθμίσεις LLM ενημερώθηκαν επιτυχώς.')

    def _extract_yaml_blocks(self, text_value: str):
        if not text_value:
            return []
        lines = text_value.splitlines()
        blocks = []
        inside = False
        buf = []
        for line in lines:
            striped = line.strip()
            if not inside:
                if striped.startswith('```yaml') or striped == '```':
                    inside = True
                    buf = []
                continue
            else:
                if striped.startswith('```'):
                    inside = False
                    if buf:
                        blocks.append('\n'.join(buf).strip())
                        buf = []
                    continue
                buf.append(line)
        return blocks

    def _parse_yaml_like(self, yaml_text: str) -> dict:
        data = {}
        if not yaml_text:
            return data
        for raw_line in yaml_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if val.startswith('"') and val.endswith('"') or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1].strip()
            lower = val.lower()
            if lower in ('true', 'false'):
                cast_val = lower == 'true'
            else:
                try:
                    if '.' in val:
                        cast_val = float(val)
                    else:
                        cast_val = int(val)
                except Exception:
                    cast_val = val
            data[key] = cast_val
        return data

    def on_copilot_result(self, text_result):
        set_copilot_busy(self, False)
        header_info = ''
        try:
            if getattr(self, '_last_copilot_model_info', ''):
                header_info = self._last_copilot_model_info.rstrip() + '\n\n'
                self._last_copilot_model_info = ''
        except Exception:
            header_info = ''
        display_text = header_info + text_result if header_info else text_result
        if self.copilot_output_edit.toPlainText().strip():
            self.copilot_output_edit.appendPlainText('\n--- Απάντηση Copilot ---\n')
            self.copilot_output_edit.appendPlainText(display_text)
        else:
            self.copilot_output_edit.clear()
            self.copilot_output_edit.appendPlainText(display_text)
        expect_yaml = getattr(self, '_copilot_expect_yaml', True)
        gui_cfg = None
        hparams_cfg = None
        if expect_yaml:
            yaml_blocks = self._extract_yaml_blocks(text_result)
            if yaml_blocks:
                if len(yaml_blocks) >= 1:
                    gui_cfg = self._parse_yaml_like(yaml_blocks[0])
                if len(yaml_blocks) >= 2:
                    hparams_cfg = self._parse_yaml_like(yaml_blocks[1])
        self.copilot_last_gui_cfg = gui_cfg
        self.copilot_last_hparams = hparams_cfg
        if gui_cfg or hparams_cfg:
            self.copilot_apply_button.setEnabled(True)
            self.copilot_output_edit.appendPlainText("\n[✅ Βρέθηκαν έγκυρα YAML μπλοκ με προτεινόμενες ρυθμίσεις. Μπορείς να πατήσεις 'Επισκόπιση και Εφαρμογή ρυθμίσεων'.]")
        else:
            self.copilot_apply_button.setEnabled(False)
            if expect_yaml:
                self.copilot_output_edit.appendPlainText('\n[ℹ Δεν βρέθηκαν έγκυρα YAML μπλοκ στην απάντηση. Βεβαιώσου ότι το μοντέλο ακολουθεί τις οδηγίες για τα YAML blocks.]')

    def on_copilot_error(self, error_text):
        set_copilot_busy(self, False)
        try:
            if hasattr(self, 'copilot_output_edit') and self.copilot_output_edit is not None:
                self.copilot_output_edit.appendPlainText(f'\n[Σφάλμα από LLM API]: {error_text}')
        except Exception:
            pass
        try:
            QMessageBox.warning(self, 'Σφάλμα Copilot', f'Παρουσιάστηκε σφάλμα από το LLM API:\n{error_text}')
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_tune_button') and self.copilot_tune_button is not None:
                self.copilot_tune_button.setText('📈 Βελτίωση με βάση τα αποτελέσματα εκπαίδευσης')
        except Exception:
            pass
        try:
            if hasattr(self, 'copilot_detection_button') and self.copilot_detection_button is not None:
                self.copilot_detection_button.setText('🧪 Βελτίωση με βάση τα αποτελέσματα ανίχνευσης')
        except Exception:
            pass
        try:
            self._update_copilot_buttons_for_llm(show_message=False)
        except Exception:
            pass

    def on_copilot_apply_clicked(self):
        if not (self.copilot_last_gui_cfg or self.copilot_last_hparams):
            QMessageBox.warning(self, 'Model Training Copilot', 'Δεν βρέθηκαν έγκυρες YAML ρυθμίσεις για εφαρμογή.')
        tt = self.training_tab
        gui = self.copilot_last_gui_cfg or {}
        hcfg_original = self.copilot_last_hparams or {}
        merged_hcfg = dict(hcfg_original)
        if 'triton_enabled' in gui and 'triton_enabled' not in merged_hcfg:
            merged_hcfg['triton_enabled'] = gui['triton_enabled']
        if 'compile_mode' in gui and 'compile_mode' not in merged_hcfg:
            merged_hcfg['compile_mode'] = gui['compile_mode']
        if 'use_torch_compile' in gui and 'triton_enabled' not in merged_hcfg:
            merged_hcfg['triton_enabled'] = gui['use_torch_compile']
        if 'use_torch_compile' in hcfg_original and 'triton_enabled' not in merged_hcfg:
            merged_hcfg['triton_enabled'] = hcfg_original['use_torch_compile']
        if 'triton_enabled' in merged_hcfg and 'compile_mode' not in merged_hcfg:
            val = merged_hcfg.get('triton_enabled')
            if isinstance(val, str):
                v_bool = val.strip().lower() in ('true', '1', 'yes', 'y', 'on')
            else:
                v_bool = bool(val)
            merged_hcfg['compile_mode'] = 'Μείωση επιβάρυνσης' if v_bool else 'Προεπιλογή'
        hcfg = merged_hcfg

        def _safe_get(attr, default=None):
            try:
                return getattr(tt, attr)
            except Exception:
                return default
        model_combo = _safe_get('model_combo')
        dataset_combo = _safe_get('dataset_combo')
        imgsz_combo = _safe_get('imgsz_combo')
        epochs_spin = _safe_get('epochs_spin')
        patience_spin = _safe_get('patience_spin')
        device_combo = _safe_get('device_combo')
        current_model = model_combo.currentText() if model_combo is not None else ''
        current_dataset = dataset_combo.currentText() if dataset_combo is not None else ''
        current_imgsz = imgsz_combo.currentText() if imgsz_combo is not None else ''
        current_epochs = epochs_spin.value() if epochs_spin is not None else 0
        current_patience = patience_spin.value() if patience_spin is not None else 0
        current_device = device_combo.currentText() if device_combo is not None else ''
        new_model = gui.get('model_name', current_model)
        new_dataset = gui.get('dataset_name', current_dataset)
        new_imgsz = gui.get('image_size', current_imgsz)
        new_epochs = gui.get('epochs', current_epochs)
        new_patience = gui.get('patience', current_patience)
        new_device = gui.get('device', current_device)
        lines: list[str] = []
        lines.append('Οι παρακάτω αλλαγές προτείνονται από το Copilot:')
        if new_model and new_model != current_model:
            lines.append(f'- Μοντέλο: {current_model} → {new_model}')
        if new_dataset and new_dataset != current_dataset:
            lines.append(f'- Dataset: {current_dataset} → {new_dataset}')
        if str(new_imgsz) != str(current_imgsz):
            lines.append(f'- Image Size: {current_imgsz} → {new_imgsz}')
        if new_epochs != current_epochs:
            lines.append(f'- Epochs: {current_epochs} → {new_epochs}')
        if new_patience != current_patience:
            lines.append(f'- Patience: {current_patience} → {new_patience}')
        if str(new_device).upper() != str(current_device).upper():
            lines.append(f'- Συσκευή: {current_device} → {new_device}')
        if 'triton_enabled' in hcfg:
            lines.append(f"- Triton (triton_enabled): {hcfg['triton_enabled']}")
        if 'compile_mode' in hcfg:
            lines.append(f"- Compile mode (compile_mode): {hcfg['compile_mode']}")
        if hcfg:
            important_keys = ['batch', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'workers']
            printed = set()
            for key in important_keys:
                if key in hcfg:
                    lines.append(f'- {key}: {hcfg[key]}')
                    printed.add(key)
            for key, value in hcfg.items():
                if key in printed or key in ('triton_enabled', 'compile_mode'):
                    continue
                lines.append(f'- {key}: {value}')
        if len(lines) == 1:
            lines.append('Δεν εντοπίστηκαν ουσιαστικές αλλαγές στις ρυθμίσεις.')
        preview_text = '\n'.join(lines)
        reply = QMessageBox.question(self, 'Preview ρυθμίσεων Copilot', preview_text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply != QMessageBox.StandardButton.Yes:
            return
        applied_keys: set[str] = set()
        if hasattr(tt, '_copilot_apply_in_progress'):
            tt._copilot_apply_in_progress = True
        try:
            try:
                tt.clear_copilot_marks()
            except Exception:
                pass
            if model_combo is not None and new_model and (new_model != current_model):
                try:
                    idx = model_combo.findText(str(new_model))
                    if idx >= 0:
                        model_combo.setCurrentIndex(idx)
                        applied_keys.add('model_name')
                except Exception:
                    pass
            if dataset_combo is not None and new_dataset and (new_dataset != current_dataset):
                try:
                    idx = dataset_combo.findText(str(new_dataset))
                    if idx >= 0:
                        dataset_combo.setCurrentIndex(idx)
                        applied_keys.add('dataset_name')
                except Exception:
                    pass
            if imgsz_combo is not None and new_imgsz is not None and (str(new_imgsz) != str(current_imgsz)):
                try:
                    new_imgsz_str = str(int(float(new_imgsz)))
                except Exception:
                    new_imgsz_str = str(new_imgsz)
                try:
                    idx = imgsz_combo.findText(new_imgsz_str)
                    if idx < 0:
                        idx = imgsz_combo.findText(str(new_imgsz))
                    if idx >= 0:
                        imgsz_combo.setCurrentIndex(idx)
                        applied_keys.add('image_size')
                except Exception:
                    pass
            if epochs_spin is not None and isinstance(new_epochs, (int, float)) and (new_epochs != current_epochs):
                try:
                    epochs_spin.setValue(int(new_epochs))
                    applied_keys.add('epochs')
                except Exception:
                    pass
            if patience_spin is not None and isinstance(new_patience, (int, float)) and (new_patience != current_patience):
                try:
                    patience_spin.setValue(int(new_patience))
                    applied_keys.add('patience')
                except Exception:
                    pass
            if device_combo is not None and isinstance(new_device, str) and new_device and (str(new_device).upper() != str(current_device).upper()):
                try:
                    dev_str = str(new_device).strip().lower()
                    if 'cuda' in dev_str or 'gpu' in dev_str:
                        target = 'GPU'
                    elif 'cpu' in dev_str:
                        target = 'CPU'
                    else:
                        target = str(new_device).upper()
                    idx = device_combo.findText(target)
                    if idx >= 0:
                        device_combo.setCurrentIndex(idx)
                        applied_keys.add('device')
                except Exception:
                    pass
            if hcfg:
                if 'batch' in hcfg and hasattr(tt, 'batch_spin'):
                    try:
                        tt.batch_spin.setValue(int(hcfg['batch']))
                        applied_keys.add('batch')
                    except Exception:
                        pass
                if 'optimizer' in hcfg and hasattr(tt, 'optimizer_combo'):
                    try:
                        opt = str(hcfg['optimizer']).strip()
                        idx = tt.optimizer_combo.findText(opt)
                        if idx < 0:
                            idx = tt.optimizer_combo.findText(opt.upper())
                        if idx < 0:
                            idx = tt.optimizer_combo.findText(opt.lower())
                        if idx >= 0:
                            tt.optimizer_combo.setCurrentIndex(idx)
                            applied_keys.add('optimizer')
                    except Exception:
                        pass

                def _set_float_spin(attr_name: str, key: str):
                    spin = getattr(tt, attr_name, None)
                    if spin is None or key not in hcfg:
                        return
                    try:
                        spin.setValue(float(hcfg[key]))
                        applied_keys.add(key)
                    except Exception:
                        pass

                def _set_int_spin(attr_name: str, key: str):
                    spin = getattr(tt, attr_name, None)
                    if spin is None or key not in hcfg:
                        return
                    try:
                        spin.setValue(int(hcfg[key]))
                        applied_keys.add(key)
                    except Exception:
                        pass
                _set_float_spin('lr0_spin', 'lr0')
                _set_float_spin('lrf_spin', 'lrf')
                _set_float_spin('momentum_spin', 'momentum')
                _set_float_spin('weight_decay_spin', 'weight_decay')
                _set_int_spin('warmup_epochs_spin', 'warmup_epochs')
                _set_int_spin('workers_spin', 'workers')
                if 'triton_enabled' in hcfg and hasattr(tt, 'triton_checkbox'):
                    try:
                        v = hcfg['triton_enabled']
                        if isinstance(v, str):
                            v_bool = v.strip().lower() in ('true', '1', 'yes', 'y', 'on')
                        else:
                            v_bool = bool(v)
                        if v_bool:
                            try:
                                if hasattr(tt, 'device_combo'):
                                    current_dev = tt.device_combo.currentText()
                                    if str(current_dev).upper() != 'GPU':
                                        idx_gpu = tt.device_combo.findText('GPU')
                                        if idx_gpu >= 0:
                                            tt.device_combo.setCurrentIndex(idx_gpu)
                                            applied_keys.add('device')
                            except Exception:
                                pass
                            try:
                                if hasattr(tt, 'triton_checkbox'):
                                    tt.triton_checkbox.setEnabled(True)
                            except Exception:
                                pass
                        tt.triton_checkbox.setChecked(bool(v_bool))
                        applied_keys.add('triton_enabled')
                    except Exception:
                        pass
                if 'compile_mode' in hcfg and hasattr(tt, 'compile_mode_combo'):
                    try:
                        raw_val = str(hcfg['compile_mode']).strip()
                        key = raw_val.lower()
                        mapping = {'default': 'Προεπιλογή', 'προεπιλογή': 'Προεπιλογή', 'reduce-overhead': 'Μείωση επιβάρυνσης', 'μείωση επιβάρυνσης': 'Μείωση επιβάρυνσης', 'max-autotune': 'Μέγιστος αυτόματος συντονισμός', 'μέγιστος αυτόματος συντονισμός': 'Μέγιστος αυτόματος συντονισμός'}
                        canonical = mapping.get(key, raw_val)
                        candidates = [canonical, canonical.title(), canonical.upper(), canonical.lower()]
                        for c in candidates:
                            idx = tt.compile_mode_combo.findText(c)
                            if idx >= 0:
                                tt.compile_mode_combo.setCurrentIndex(idx)
                                applied_keys.add('compile_mode')
                                try:
                                    if hasattr(tt, 'triton_checkbox'):
                                        current_mode = tt.compile_mode_combo.currentText()
                                        if current_mode != 'Προεπιλογή' and (not tt.triton_checkbox.isChecked()):
                                            tt.triton_checkbox.setEnabled(True)
                                            tt.triton_checkbox.setChecked(True)
                                            applied_keys.add('triton_enabled')
                                            if hasattr(tt, 'device_combo'):
                                                try:
                                                    current_dev = tt.device_combo.currentText()
                                                    if str(current_dev).upper() != 'GPU':
                                                        idx_gpu = tt.device_combo.findText('GPU')
                                                        if idx_gpu >= 0:
                                                            tt.device_combo.setCurrentIndex(idx_gpu)
                                                            applied_keys.add('device')
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                                break
                    except Exception:
                        pass
        finally:
            if hasattr(tt, '_copilot_apply_in_progress'):
                tt._copilot_apply_in_progress = False
        try:
            if 'triton_enabled' in applied_keys or 'compile_mode' in applied_keys:
                tt.set_triton_copilot_mark(True)
        except Exception:
            pass
        for key in applied_keys:
            try:
                tt.mark_field_copilot(key, enabled=True)
            except Exception:
                pass
        QMessageBox.information(self, 'Model Training Copilot', 'Οι προτεινόμενες ρυθμίσεις εφαρμόστηκαν (όπου ήταν δυνατό).')
        set_copilot_busy(self, False)


class TrainingCopilotTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('🤖 A.I Copilot Εκπαίδευσης - Standalone')
        training_tab = TrainingTab()
        stats_tab = StatisticsTab()
        self.setCentralWidget(TrainingCopilotTab(training_tab, stats_tab))
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def training_copilot_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = TrainingCopilotTabWindow()
    win.show()
    sys.exit(app.exec())
"""Benchmark tab (UI).
UI για Auto-Benchmark backends (PyTorch/ONNX/TensorRT/NCNN) και προβολή αποτελεσμάτων.
"""


class _BenchmarkUiBridge(QObject):
    log = Signal(str)
    error = Signal(str)
    results = Signal(list)
    finished = Signal(bool)


class _BenchmarkCore:

    def __init__(self, base_name: str, imgsz: int, models_dir: Path, log_cb, stop_event=None):
        self.base_name = base_name
        self.imgsz = int(imgsz)
        self.models_dir = models_dir
        self.num_warmup = 10
        self.num_runs = 50
        self.conf_threshold = 0.25
        self.iou_threshold = 0.70
        self._log_cb = log_cb
        self._stop_event = stop_event

    def _stop_requested(self) -> bool:
        try:
            return bool(self._stop_event and self._stop_event.is_set())
        except Exception:
            return False

    def _cprint(self, text: str, color: str=Colors.CYAN, bold: bool=False, underline: bool=False):
        try:
            html = format_html_log(text, color, bold, underline)
            self._log_cb(html)
        except Exception:
            pass

    def _benchmark_backend(self, backend: str, path: Path):
        if self._stop_requested():
            return None
        self._cprint(f'🧪 Εκτέλεση {backend_pretty_name(backend)} σε ασφαλές subprocess...', Colors.CYAN)
        job = {
            'backend': (backend or '').lower(),
            'path': str(path),
            'imgsz': int(self.imgsz),
            'num_warmup': int(self.num_warmup),
            'num_runs': int(self.num_runs),
            'conf': float(self.conf_threshold),
            'iou': float(self.iou_threshold),
        }
        job_file = None
        try:
            tmp_dir = Path(tempfile.gettempdir())
            job_file = tmp_dir / f'mmpro_bench_{os.getpid()}_{int(time.time()*1000)}.json'
            job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            self._cprint(f'❌ Αδυναμία δημιουργίας προσωρινού job για benchmark: {e}', Colors.RED, bold=True)
            return (0.0, 0.0)
        try:
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--mmpro-mode=bench', str(job_file)]
            else:
                entry = Path(__file__).resolve()
                cmd = [sys.executable, str(entry), '--mmpro-mode=bench', str(job_file)]
        except Exception:
            cmd = [sys.executable, '--mmpro-mode=bench', str(job_file)]
        env = os.environ.copy()
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')
        env.setdefault('PYTHONUTF8', '1')
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        creationflags = 0
        try:
            if os.name == 'nt':
                creationflags |= subprocess.CREATE_NO_WINDOW
        except Exception:
            pass
        try:
            proc = _run_cmd( cmd, timeout=600, env=env, creationflags=creationflags,)
        except subprocess.TimeoutExpired:
            self._cprint(f'⏳ Timeout στο benchmark για {backend_pretty_name(backend)} (πιθανό hang στο load).', Colors.YELLOW, bold=True)
            return (0.0, 0.0)
        except Exception as e:
            self._cprint(f'❌ Αποτυχία εκτέλεσης subprocess benchmark ({backend}): {e}', Colors.RED, bold=True)
            return (0.0, 0.0)
        finally:
            try:
                if job_file and job_file.exists():
                    job_file.unlink(missing_ok=True)
            except Exception:
                pass

        def _extract_payload(stdout_text: str):
            try:
                out_lines = [ln.strip() for ln in (stdout_text or '').splitlines() if ln.strip()]
                for ln in reversed(out_lines):
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and 'ok' in obj:
                            return obj
                    except Exception:
                        continue
            except Exception:
                return None
        if getattr(proc, 'returncode', 1) != 0:
            payload = _extract_payload(proc.stdout or '')
            if isinstance(payload, dict) and payload.get('ok') is False:
                err = str(payload.get('error', 'Άγνωστο σφάλμα'))
                self._cprint(f'💥 {backend_pretty_name(backend)} απέτυχε: {err}', Colors.RED, bold=True)
                if os.environ.get('MM_DEBUG_TRACEBACK', '').strip() in ('1', 'true', 'TRUE', 'yes', 'YES'):
                    tb = payload.get('traceback')
                    if tb:
                        self._cprint(str(tb), Colors.RED)
                return (0.0, 0.0)
            hint = 'Δες τα logs για λεπτομέρειες.'
            rc = getattr(proc, 'returncode', -1)
            self._cprint(f'💥 Το backend {backend_pretty_name(backend)} απέτυχε (exit={rc}): {hint}', Colors.RED, bold=True)
            return (0.0, 0.0)
        payload = _extract_payload(proc.stdout or '')
        if not payload:
            self._cprint(f'❌ Δεν επέστρεψε έγκυρο αποτέλεσμα το subprocess για {backend_pretty_name(backend)}.', Colors.RED)
            return (0.0, 0.0)
        if not payload.get('ok', False):
            self._cprint(f"❌ Σφάλμα {backend_pretty_name(backend)}: {payload.get('error','Άγνωστο σφάλμα')}", Colors.RED)
            return (0.0, 0.0)
        try:
            fps = float(payload.get('fps', 0.0) or 0.0)
            ms_per_image = float(payload.get('ms', 0.0) or 0.0)
        except Exception:
            fps, ms_per_image = 0.0, 0.0
        self._cprint(f'Αποτέλεσμα {backend_pretty_name(backend)}: {fps:.2f} FPS, {ms_per_image:.2f} ms/εικόνα', Colors.GREEN, bold=True)
        return (fps, ms_per_image)

    def run(self):
        results = []
        self._cprint('🚀 Έναρξη Auto-Benchmark PyTorch / ONNX / TensorRT / NCNN...', Colors.CYAN, bold=True)
        self._cprint(f'🧠 Μοντέλο βάσης: {self.base_name}', Colors.CYAN)
        self._cprint(f'📏 Image size: {self.imgsz}', Colors.CYAN)
        backends = find_available_backends(self.models_dir, self.base_name)
        if not backends:
            self._cprint(f"❌ Δεν βρέθηκε κανένα backend για το '{self.base_name}' στον φάκελο: {self.models_dir}", Colors.RED, bold=True)
            return results
        order = ['pytorch', 'onnx', 'tensorrt', 'ncnn']
        for backend in order:
            if backend not in backends:
                continue
            if self._stop_requested():
                self._cprint('🛑 Διακοπή Benchmark πριν την ολοκλήρωση.', Colors.YELLOW, bold=True)
                break
            self._cprint('─' * 78, Colors.CYAN)
            self._cprint(f'▶ Benchmark για: {backend_pretty_name(backend)}', Colors.MAGENTA, bold=True)
            try:
                res = self._benchmark_backend(backend, backends[backend])
                if res is None:
                    break
                fps, ms = res
                if fps > 0:
                    results.append((backend, fps, ms))
            except Exception as e:
                self._cprint(f'💥 Σφάλμα στο backend {backend_pretty_name(backend)}: {e}', Colors.RED, bold=True)
                self._cprint(traceback.format_exc(), Colors.RED)
        self._cprint('═' * 78, Colors.CYAN, bold=True)
        if results:
            self._cprint('✅ Ολοκλήρωση Benchmark!', Colors.GREEN, bold=True)
        else:
            self._cprint('⚠️ Ολοκλήρωση Benchmark χωρίς έγκυρα αποτελέσματα.', Colors.YELLOW, bold=True)
        return results


def _benchmark_thread_entry(bridge: _BenchmarkUiBridge, stop_event, base_name: str, imgsz: int, model_dir: Path):
    try:
        ensure_cuda_ready_for_thread("_benchmark_thread_entry")
    except Exception:
        pass
    try:
        runner = _BenchmarkCore( base_name=base_name, imgsz=imgsz, models_dir=model_dir, log_cb=lambda html: bridge.log.emit(html), stop_event=stop_event,)
        results = runner.run()
        try:
            bridge.results.emit(results)
            bridge.finished.emit(True)
        except Exception:
            pass
    except Exception as e:
        try:
            bridge.error.emit(f'Σφάλμα στο Benchmark: {e}')
            bridge.finished.emit(False)
        except Exception:
            pass


class BenchmarkTab(QWidget, TabNavigationMixin, BenchmarkUIHelpersMixin):

    def __init__(self, parent: QWidget | None=None):
        super().__init__(parent)
        self.models_dir = MODELS_DIR_TRAINED_PT
        self.benchmark_thread: QThread | None = None
        self.benchmark_worker: BenchmarkWorker | None = None
        self.init_ui()
        self._bench_py_thread = None
        self._bench_stop_event = None
        self._bench_bridge = _BenchmarkUiBridge()
        try:
            self._bench_bridge.log.connect(self.append_log, Qt.ConnectionType.QueuedConnection)
            self._bench_bridge.error.connect(self.on_worker_error, Qt.ConnectionType.QueuedConnection)
            self._bench_bridge.results.connect(self.on_worker_results, Qt.ConnectionType.QueuedConnection)
            self._bench_bridge.finished.connect(self.on_worker_finished, Qt.ConnectionType.QueuedConnection)
        except Exception:
            pass

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        outer_layout.addLayout(main_layout, 1)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        settings_group = QGroupBox('Ρυθμίσεις Benchmark')
        settings_layout = QGridLayout(settings_group)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)
        models_label = QLabel('Εκπαιδευμένο Μοντέλο:')
        self.models_combo = QComboBox()
        self.refresh_button = QPushButton('🔄')
        self.refresh_button.setObjectName('RefreshButton')
        self.refresh_button.setFixedWidth(32)
        self.refresh_button.setFont(QFont('Segoe UI Emoji'))
        self.refresh_button.setToolTip('Σάρωση φακέλου Trained_Models για διαθέσιμα μοντέλα')
        imgsz_label = QLabel('Image size:')
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(160, 1920)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        settings_layout.addWidget(models_label, 0, 0)
        settings_layout.addWidget(self.models_combo, 0, 1)
        settings_layout.addWidget(self.refresh_button, 0, 2)
        settings_layout.addWidget(imgsz_label, 1, 0)
        settings_layout.addWidget(self.imgsz_spin, 1, 1)
        try:
            self.imgsz_spin.setReadOnly(True)
            self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            self.models_combo.currentTextChanged.connect(self.on_model_changed)
        except Exception:
            pass
        left_layout.addWidget(settings_group)
        self.run_button = QPushButton('⚡ Εκτέλεση Benchmark')
        self.run_button.setToolTip('Εκτέλεση Auto-Benchmark PyTorch / ONNX / TensorRT / NCNN για το επιλεγμένο μοντέλο')
        left_layout.addWidget(self.run_button)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(['Backend', 'FPS', 'ms / εικόνα'])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setColumnWidth(0, 224)
        self.results_table.setColumnWidth(1, 112)
        self.results_table.setColumnWidth(2, 80)
        left_layout.addWidget(self.results_table)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(520)
        main_layout.addWidget(left_widget)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        log_group = QGroupBox('Log Benchmark')
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setObjectName('BenchmarkLogOutput')
        log_layout.addWidget(self.log_edit)
        right_layout.addWidget(log_group)
        main_layout.addLayout(right_layout)
        self.refresh_button.clicked.connect(self.refresh_models)
        self.run_button.clicked.connect(self.start_benchmark)
        self.refresh_models()

    def refresh_models(self):
        self.models_combo.clear()
        if not self.models_dir.exists():
            self.models_combo.addItem('<< Ο φάκελος μοντέλων δεν υπάρχει >>')
            self.run_button.setEnabled(False)
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            self.models_combo.addItem('<< Δεν βρέθηκαν μοντέλα >>')
            self.run_button.setEnabled(False)
        model_dirs_sorted = sorted(model_dirs, key=lambda p: p.name)
        base_names_sorted = [d.name for d in model_dirs_sorted]
        target_name = None
        try:
            main_window = self.window()
        except Exception:
            main_window = None
        if main_window is not None and hasattr(main_window, 'current_trained_model_stem'):
            target_stem = getattr(main_window, 'current_trained_model_stem', '') or ''
            if target_stem:
                target_normalized = target_stem.replace('_ncnn_model', '')
                for name in base_names_sorted:
                    if name == target_normalized or name.replace('_ncnn_model', '') == target_normalized:
                        target_name = name
                        break
        if not target_name:
            try:
                last_dir = max(model_dirs_sorted, key=lambda d: d.stat().st_mtime)
                target_name = last_dir.name
            except Exception:
                target_name = base_names_sorted[-1] if base_names_sorted else None
        self.models_combo.addItem('<< Επέλεξε μοντέλο >>')
        for name in base_names_sorted:
            self.models_combo.addItem(name)
        if target_name and target_name in base_names_sorted:
            try:
                idx = base_names_sorted.index(target_name)
                self.models_combo.setCurrentIndex(idx + 1)
            except Exception:
                pass
        self.on_model_changed(self.models_combo.currentText())
        self.run_button.setEnabled(True)

    def on_model_changed(self, _text: str):
        name = self.models_combo.currentText().strip() if hasattr(self, 'models_combo') else ''
        val = self._parse_imgsz_from_name(name)
        if val is not None and hasattr(self, 'imgsz_spin'):
            try:
                self.imgsz_spin.setValue(int(val))
            except Exception:
                pass
        if hasattr(self, 'imgsz_spin'):
            self.imgsz_spin.setReadOnly(True)
        self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def start_benchmark(self):
        try:
            if hasattr(self, '_start_benchmark_impl'):
                return self._start_benchmark_impl()
        except Exception:
            pass
        try:
            if getattr(self, '_bench_py_thread', None) is not None and self._bench_py_thread.is_alive():
                QMessageBox.information(self, 'Benchmark', 'Ήδη εκτελείται benchmark. Περίμενε να ολοκληρωθεί.')
        except Exception:
            pass
        started = JOB_MANAGER.try_start('Benchmark', lambda: self._start_benchmark_impl(), cancel_cb=self.stop_benchmark)
        if not started:
            self.append_log('🕒 Το benchmark μπήκε στην ουρά (τρέχει ήδη εργασία).')

    def _start_benchmark_impl(self):
        current_text = self.models_combo.currentText().strip()
        if not current_text or current_text.startswith('<<'):
            QMessageBox.warning(self, 'Benchmark', 'Δεν έχει επιλεγεί έγκυρο όνομα μοντέλου.')
        base_name = current_text
        imgsz = self.imgsz_spin.value()
        if self.benchmark_thread is not None and self.benchmark_thread.isRunning():
            QMessageBox.information(self, 'Benchmark', 'Ήδη εκτελείται benchmark. Περίμενε να ολοκληρωθεί.')
        try:
            warmup_torch_cuda("benchmark_start")
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            ensure_windows_com_initialized()
        except Exception:
            pass
        self.log_edit.clear()
        header = f'🚀 Ξεκινά Benchmark για μοντέλο: {base_name} (imgsz={imgsz})'
        bar = '═' * 78
        sub_bar = '─' * 78
        self.append_log(format_html_log(bar, Colors.CYAN, bold=True))
        self.append_log(format_html_log(header, Colors.CYAN, bold=True))
        self.append_log(format_html_log('🔁 Εκτέλεση Auto-Benchmark PyTorch / ONNX / TensorRT / NCNN', Colors.CYAN, bold=False))
        self.append_log(format_html_log(sub_bar, Colors.CYAN, bold=False))
        self.run_button.setEnabled(False)
        model_dir = self.models_dir / base_name
        try:
            self._bench_stop_event = threading.Event()
            self._bench_py_thread = threading.Thread( target=_benchmark_thread_entry, args=(self._bench_bridge, self._bench_stop_event, base_name, imgsz, model_dir), daemon=True,)
            self._bench_py_thread.start()
        except Exception as e:
            try:
                self.run_button.setEnabled(True)
            except Exception:
                pass
            QMessageBox.critical(self, 'Benchmark - Σφάλμα', f'Αδυναμία εκκίνησης benchmark thread: {e}')
            try:
                JOB_MANAGER.done(False)
            except Exception:
                pass

    def stop_benchmark(self):
        try:
            if getattr(self, '_bench_stop_event', None) is not None:
                self._bench_stop_event.set()
                self.append_log(format_html_log('🛑 Ζητήθηκε διακοπή benchmark...', Colors.YELLOW, bold=True))
        except Exception:
            pass

    def on_worker_error(self, msg: str):
        QMessageBox.critical(self, 'Benchmark - Σφάλμα', msg)

    def on_worker_finished(self, ok: bool=True):
        try:
            JOB_MANAGER.done(bool(ok))
            self.run_button.setEnabled(True)
        except Exception:
            pass
        try:
            self._bench_stop_event = None
            self._bench_py_thread = None
        except Exception:
            pass
        try:
            if self.benchmark_thread is not None:
                self.benchmark_thread.quit()
                self.benchmark_thread.wait(2000)
                self.benchmark_thread = None
        except Exception:
            pass
        self.benchmark_worker = None
        try:
            summary = flush_log_once_summary('Benchmark', reset=True, top_n=12, min_total=1)
            if summary:
                try:
                    self.append_log(format_html_summary(summary))
                except Exception:
                    try:
                        self.log_edit.append(format_html_summary(summary))
                    except Exception:
                        pass
        except Exception:
            pass


class BenchmarkTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('⚡ Benchmark FPS - Standalone')
        self.setCentralWidget(BenchmarkTab())
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def benchmark_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = BenchmarkTabWindow()
    win.show()
    sys.exit(app.exec())
"""Live Camera tab (UI).
Ο ρόλος του tab είναι να:
- Επιλέγει μοντέλο/backend και να εκκινεί/σταματά το `CameraWorker`.; - Κάνει render frames με σωστό aspect ratio (χωρίς "zoom" / crop artifacts).; - Εμφανίζει overlay (runtime/FPS) και βασικά controls.; Το tab ΔΕΝ χρησιμοποιεί `imgsz` για capture ανάλυση. Η κάμερα έχει δικό της policy (1080p).
"""
try:
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
except Exception:
    QGraphicsDropShadowEffect = None


class AspectRatioLabel(QLabel):

    def __init__(self, *args, aspect_ratio: float = (16.0 / 9.0), **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self._aspect_ratio = float(aspect_ratio) if aspect_ratio else (16.0 / 9.0)
        except Exception:
            self._aspect_ratio = (16.0 / 9.0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        try:
            if self._aspect_ratio <= 0:
                return int(super().heightForWidth(w))
            return int(round(float(w) / float(self._aspect_ratio)))
        except Exception:
            return int(super().heightForWidth(w))

    def sizeHint(self):
        s = super().sizeHint()
        try:
            w = max(16, int(s.width()))
        except Exception:
            w = 640
        return QSize(int(w), int(self.heightForWidth(int(w))))

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        try:
            w = max(16, int(s.width()))
        except Exception:
            w = 320
        return QSize(int(w), int(self.heightForWidth(int(w))))


class CameraTab(QWidget, TabNavigationMixin):

    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.camera_worker = None
        try:
            summary = flush_log_once_summary('Camera', reset=True, top_n=12, min_total=1)
            if summary:
                try:
                    self.append_log(format_html_summary(summary))
                except Exception:
                    try:
                        self.log_edit.append(format_html_summary(summary))
                    except Exception:
                        pass
        except Exception:
            pass
        self._ui_timer = QTimer(self)
        try:
            ui_fps_max = int(os.environ.get('MM_PRO_LIVE_UI_FPS_MAX', '120'))
        except Exception:
            ui_fps_max = 120
        ui_fps_max = max(15, min(240, ui_fps_max))
        interval_ms = max(1, int(round(1000 / ui_fps_max)))
        self._ui_timer.setInterval(interval_ms)
        self._ui_timer.timeout.connect(self._pull_latest_frame)
        self.available_models = []
        self.classes_checkboxes = []
        self.selected_classes = None
        self.use_tensorrt = False
        self._class_names_cache: dict[tuple[str, str], dict] = {}
        self._refreshing_models: bool = False
        self.init_ui()
        # Restore camera settings
        try:
            s = _settings()
            if s.get('camera_index') is not None:
                try:
                    self.camera_index_spin.setValue(int(s.get('camera_index')))
                except Exception:
                    pass
            if s.get('camera_conf') is not None:
                try:
                    self.conf_slider.setValue(int(s.get('camera_conf')))
                except Exception:
                    pass
            if s.get('camera_tensorrt') is not None:
                try:
                    self.tensorrt_checkbox.setChecked(bool(s.get('camera_tensorrt')))
                except Exception:
                    pass
        except Exception:
            pass

    def find_available_models(self):
        models = []
        if MODELS_DIR_TRAINED_PT.exists():
            for d in sorted([d for d in MODELS_DIR_TRAINED_PT.iterdir() if d.is_dir()]):
                base = d.name
                pt_path = d / f'{base}.pt'
                if pt_path.exists():
                    pt_type = 'PyTorch'
                    try:
                        cj = d / 'class_names.json'
                        if cj.is_file():
                            import json as _j2
                            _d2 = _j2.loads(cj.read_text(encoding='utf-8'))
                            mn = str(_d2.get('model_name', '')).lower()
                            if any(c in mn for c in _CNN_MODEL_KEYS):
                                pt_type = 'CNN'
                    except Exception:
                        pass
                    if pt_type == 'PyTorch':
                        sl = base.lower()
                        if any(c in sl for c in _CNN_MODEL_KEYS):
                            pt_type = 'CNN'
                    models.append((pt_path, pt_type))
                # TensorRT .engine
                engine_path = d / f'{base}.engine'
                if engine_path.exists():
                    models.append((engine_path, 'TensorRT'))
                # NCNN
                ncnn_dir = d / f'{base}_ncnn_model'
                if ncnn_dir.exists() and ncnn_dir.is_dir() and (
                    (ncnn_dir / 'model.ncnn.param').exists() or
                    (ncnn_dir / 'model.ncnn.bin').exists() or
                    (ncnn_dir / 'model.param').exists()):
                    models.append((ncnn_dir, 'NCNN'))
                # ONNX (regular or CNN-ONNX)
                onnx_path = d / f'{base}.onnx'
                if onnx_path.exists():
                    onnx_type = 'ONNX'
                    try:
                        # Priority: _onnx_meta.json > class_names.json
                        for cj in [d / (base + '_onnx_meta.json'), d / 'class_names.json']:
                            if cj.is_file():
                                import json as _j3
                                _d3 = _j3.loads(cj.read_text(encoding='utf-8'))
                                mn = str(_d3.get('model_name', '')).lower()
                                if any(c in mn for c in _CNN_MODEL_KEYS):
                                    onnx_type = 'CNN-ONNX'
                                break
                    except Exception:
                        pass
                    # Also check stem name
                    if onnx_type == 'ONNX' and any(c in base.lower() for c in _CNN_MODEL_KEYS):
                        onnx_type = 'CNN-ONNX'
                    models.append((onnx_path, onnx_type))
        return models

    def refresh_models(self):
        if getattr(self, '_refreshing_models', False):
            return
        self._refreshing_models = True
        try:
            selected_type = self.model_type_combo.currentText() if hasattr(self, 'model_type_combo') else 'Όλα'
            selected_device = self.device_combo.currentText() if hasattr(self, 'device_combo') else 'All'
            all_models = self.find_available_models()
            filtered_models: list[tuple[Path, str]] = []

            # Map combo label → internal mtype value(s)
            _type_map = {
                'Όλα':               None,            # None = no filter
                'PyTorch (.pt)':     ('PyTorch',),
                'CNN (.pt)':         ('CNN',),
                'CNN ONNX (.onnx)':  ('CNN-ONNX',),
                'ONNX (.onnx)':      ('ONNX', 'CNN-ONNX'),
                'TensorRT (.engine)':('TensorRT',),
                'NCNN':              ('NCNN',),
            }
            allowed_types = _type_map.get(selected_type, None)

            for path, mtype in all_models:
                # Apply type filter
                if allowed_types is not None and mtype not in allowed_types:
                    continue
                # Apply device filter (only relevant for PyTorch and CNN .pt)
                if mtype in ('PyTorch', 'CNN'):
                    if selected_device != 'All':
                        if 'CPU' in selected_device and 'CPU' not in path.name:
                            continue
                        if 'GPU' in selected_device and 'GPU' not in path.name:
                            continue
                elif mtype == 'NCNN':
                    if selected_device == 'GPU':
                        continue
                    if not list(path.glob('*.param')):
                        continue
                # ONNX, CNN-ONNX, TensorRT: device filter not applicable
                filtered_models.append((path, mtype))
            camera_running = self.camera_thread is not None and self.camera_thread.isRunning()
            try:
                self.model_combo.blockSignals(True)
            except Exception:
                pass
            try:
                self.model_combo.clear()
                if not filtered_models:
                    self.model_combo.addItems(['Δεν βρέθηκαν μοντέλα'])
                    self.model_combo.setEnabled(False)
                    self.start_button.setEnabled(False)
                    if hasattr(self, 'tensorrt_checkbox'):
                        self.tensorrt_checkbox.setEnabled(False)
                        self.tensorrt_checkbox.setChecked(False)
                    self.available_models = []
                else:
                    display_names = [f'[{mtype}] {path.name}' for path, mtype in filtered_models]
                    self.model_combo.addItems(display_names)
                    self.model_combo.setEnabled(True)
                    self.start_button.setEnabled(not camera_running)
                    selected_index = self.model_combo.currentIndex()
                    try:
                        main_window = self.window()
                    except Exception:
                        main_window = None
                    if main_window is not None and hasattr(main_window, 'current_trained_model_path'):
                        target_path = getattr(main_window, 'current_trained_model_path', None)
                        target_stem = getattr(main_window, 'current_trained_model_stem', '') or ''
                        picked = -1
                        try:
                            if target_path is not None:
                                for i, (p, _t) in enumerate(filtered_models):
                                    if p == target_path:
                                        picked = i
                                        break
                        except Exception:
                            picked = -1
                        if picked < 0 and target_stem:
                            try:
                                for i, (p, _t) in enumerate(filtered_models):
                                    p_stem = getattr(p, 'stem', '') or getattr(p, 'name', '')
                                    if p_stem == target_stem or p_stem.replace('_ncnn_model', '') == target_stem:
                                        picked = i
                                        break
                            except Exception:
                                picked = -1
                        if picked >= 0:
                            selected_index = picked
                    if selected_index < 0:
                        selected_index = 0
                    if selected_index >= self.model_combo.count():
                        selected_index = 0
                    try:
                        self.model_combo.setCurrentIndex(selected_index)
                    except Exception:
                        pass
                    self.available_models = filtered_models
            finally:
                try:
                    self.model_combo.blockSignals(False)
                except Exception:
                    pass
        finally:
            self._refreshing_models = False
        try:
            self.on_model_selected()
        except Exception:
            pass

    def on_model_selected(self):
        current_index = self.model_combo.currentIndex()
        if current_index < 0 or current_index >= len(self.available_models):
            try:
                self.tensorrt_checkbox.setEnabled(False)
                self.tensorrt_checkbox.setChecked(False)
            except Exception:
                pass
            return
        model_path, model_type = self.available_models[current_index]

        # ── Detect CNN ──────────────────────────────────────────────────
        model_path_obj = Path(model_path)
        stem_lower = model_path_obj.stem.lower()
        is_cnn = any(c in stem_lower for c in _CNN_MODEL_KEYS)
        if not is_cnn and model_path_obj.suffix.lower() == '.pt':
            try:
                import torch as _tc
                _ck = _tc.load(str(model_path_obj), map_location='cpu', weights_only=False)
                if isinstance(_ck, dict) and 'model_name' in _ck:
                    mn = str(_ck.get('model_name', '')).lower()
                    is_cnn = any(c in mn for c in _CNN_MODEL_KEYS)
            except Exception:
                pass
        # CNN ONNX: check sibling class_names.json
        if not is_cnn and model_path_obj.suffix.lower() == '.onnx':
            cj = model_path_obj.parent / 'class_names.json'
            if cj.is_file():
                try:
                    d = json.loads(cj.read_text(encoding='utf-8'))
                    mn = str(d.get('model_name', '')).lower()
                    is_cnn = any(c in mn for c in _CNN_MODEL_KEYS)
                except Exception:
                    pass

        # ── TensorRT checkbox logic ──────────────────────────────────────
        if model_type == 'TensorRT':
            # Απευθείας .engine — checkbox locked ON, δεν χρειάζεται επιλογή
            self.tensorrt_checkbox.blockSignals(True)
            self.tensorrt_checkbox.setChecked(True)
            self.tensorrt_checkbox.setEnabled(False)
            self.tensorrt_checkbox.blockSignals(False)
            self.use_tensorrt = True
            self.tensorrt_checkbox.setToolTip(
                '🔥 TensorRT Engine επιλεγμένο απευθείας (.engine).\n'
                'Δεν απαιτείται ξεχωριστή ενεργοποίηση.')
        elif model_type == 'ONNX' and not is_cnn:
            # ONNX μοντέλο: έλεγξε αν υπάρχει .engine δίπλα
            engine_path = Path(str(model_path)).with_suffix('.engine')
            if engine_path.exists():
                self.tensorrt_checkbox.setEnabled(True)
                self.tensorrt_checkbox.setToolTip(
                    f'🔥 Ενεργοποίηση TensorRT — βρέθηκε engine:\n{engine_path.name}\n'
                    f'Τσεκάρισε για να χρησιμοποιηθεί αντί για το .onnx.')
            else:
                self.tensorrt_checkbox.setEnabled(False)
                self.tensorrt_checkbox.setChecked(False)
                self.use_tensorrt = False
                self.tensorrt_checkbox.setToolTip(
                    '⚠️ Δεν βρέθηκε .engine αρχείο δίπλα στο .onnx.\n'
                    'Κάνε export σε TensorRT πρώτα για να ενεργοποιηθεί.')
        else:
            # CNN, PyTorch, NCNN: TensorRT δεν υποστηρίζεται
            self.tensorrt_checkbox.blockSignals(True)
            self.tensorrt_checkbox.setChecked(False)
            self.tensorrt_checkbox.setEnabled(False)
            self.tensorrt_checkbox.blockSignals(False)
            self.use_tensorrt = False
            if is_cnn:
                self.tensorrt_checkbox.setToolTip(
                    '⚠️ TensorRT δεν υποστηρίζεται για CNN (torchvision) μοντέλα.')
            elif model_type == 'NCNN':
                self.tensorrt_checkbox.setToolTip(
                    '⚠️ TensorRT δεν υποστηρίζεται για NCNN μοντέλα.')
            else:
                self.tensorrt_checkbox.setToolTip(
                    '⚠️ TensorRT διαθέσιμο μόνο για ONNX μοντέλα με .engine αρχείο.')

        self.update_classes_list(model_path, model_type)

    # ── Ενημέρωση λίστας κλάσεων για το επιλεγμένο μοντέλο ─────────────────────
    # CNN μοντέλα: φορτώνει class_names από JSON ή checkpoint (ΟΧΙ YOLO()).
    #              Αν δεν βρεθούν κλάσεις: εμφανίζει ενημερωτικό widget.
    # YOLO μοντέλα: φορτώνει μέσω YOLO().names dict.
    # Χρησιμοποιεί _class_names_cache για αποφυγή επαναλαμβανόμενης φόρτωσης.
    def update_classes_list(self, model_path, model_type):
        for i in reversed(range(self.classes_layout.count())):
            widget = self.classes_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.classes_checkboxes.clear()
        try:
            cache_key = (str(model_path), str(model_type))
            class_names: dict[int, str] | None = self._class_names_cache.get(cache_key)
            if class_names is None:
                # ── Detect CNN model ─────────────────────────────────────
                model_path_obj = Path(model_path)
                stem_lower     = model_path_obj.stem.lower()
                is_cnn_model_flag = any(c in stem_lower for c in _CNN_MODEL_KEYS)

                # For .pt files: peek at checkpoint to confirm CNN
                if not is_cnn_model_flag and model_path_obj.suffix.lower() == '.pt':
                    try:
                        import torch as _t
                        _ck = _t.load(str(model_path_obj), map_location='cpu',
                                      weights_only=False)
                        if isinstance(_ck, dict) and 'model_name' in _ck:
                            mn = str(_ck.get('model_name', '')).lower()
                            is_cnn_model_flag = any(c in mn for c in _CNN_MODEL_KEYS)
                    except Exception:
                        pass

                # ── CNN: load class names from checkpoint / JSON ──────────
                if is_cnn_model_flag:
                    cnn_classes: list[str] = []

                    # Priority 1: *_onnx_meta.json (παράγεται από CNN→ONNX export)
                    meta_json = model_path_obj.parent / (model_path_obj.stem + '_onnx_meta.json')
                    if meta_json.is_file():
                        try:
                            d = json.loads(meta_json.read_text(encoding='utf-8'))
                            cnn_classes = list(d.get('class_names', []))
                        except Exception:
                            pass

                    # Priority 2: sibling class_names.json
                    if not cnn_classes:
                        for cj_path in [
                            model_path_obj.parent / 'class_names.json',
                            model_path_obj.with_name('class_names.json'),
                        ]:
                            if cj_path.is_file():
                                try:
                                    d = json.loads(cj_path.read_text(encoding='utf-8'))
                                    cnn_classes = list(d.get('class_names', []))
                                    if cnn_classes:
                                        break
                                except Exception:
                                    pass

                    # Priority 3: embedded in .pt checkpoint
                    if not cnn_classes and model_path_obj.suffix.lower() == '.pt':
                        try:
                            import torch as _t2
                            _ck2 = _t2.load(str(model_path_obj), map_location='cpu',
                                            weights_only=False)
                            if isinstance(_ck2, dict):
                                cnn_classes = list(_ck2.get('class_names', []))
                        except Exception:
                            pass

                    if cnn_classes:
                        class_names = {i: str(n) for i, n in enumerate(cnn_classes)}
                    else:
                        # No class info available – show a notice and return
                        notice = QLabel(
                            '🧠 CNN μοντέλο\n'
                            'Δεν βρέθηκε class_names.json.\n'
                            'Η ταξινόμηση λειτουργεί κανονικά.')
                        notice.setWordWrap(True)
                        self.classes_layout.addWidget(notice)
                        return

                else:
                    # ── YOLO: load via Ultralytics ────────────────────────
                    task = guess_ultralytics_task(model_path)
                    try:
                        model = YOLO(str(model_path), task=task)
                    except TypeError:
                        model = YOLO(str(model_path))
                    class_names_raw = getattr(model, 'names', None)
                    parsed: dict[int, str] = {}
                    if isinstance(class_names_raw, dict):
                        for cid, name in class_names_raw.items():
                            try:
                                cid_int = int(cid)
                            except Exception:
                                continue
                            parsed[cid_int] = str(name)
                    elif isinstance(class_names_raw, (list, tuple)):
                        for cid, name in enumerate(class_names_raw):
                            parsed[cid] = str(name)
                    class_names = parsed

                self._class_names_cache[cache_key] = class_names

            if not class_names:
                return
            self.select_all_checkbox = QCheckBox('Επιλογή όλων')
            self.select_all_checkbox.setChecked(True)
            self.select_all_checkbox.toggled.connect(self.toggle_all_classes)
            self.classes_layout.addWidget(self.select_all_checkbox)
            sorted_classes = []
            for class_id, english_name in class_names.items():
                sorted_classes.append((english_name, class_id))
            sorted_classes.sort(key=lambda x: x[0])
            for english_name, class_id in sorted_classes:
                checkbox = QCheckBox(f'{english_name}')
                checkbox.setChecked(True)
                checkbox.class_id = class_id
                checkbox.toggled.connect(self.on_class_toggled)
                self.classes_layout.addWidget(checkbox)
                self.classes_checkboxes.append(checkbox)
            self.selected_classes = [checkbox.class_id for checkbox in self.classes_checkboxes]
        except Exception as e:
            _MMPRO_LOGGER.debug('update_classes_list error: %s', e)

    def toggle_all_classes(self, checked):
        for checkbox in self.classes_checkboxes:
            checkbox.blockSignals(True)
        if checked:
            for checkbox in self.classes_checkboxes:
                checkbox.setChecked(True)
            self.selected_classes = [checkbox.class_id for checkbox in self.classes_checkboxes]
        else:
            for checkbox in self.classes_checkboxes:
                checkbox.setChecked(False)
            self.selected_classes = []
        for checkbox in self.classes_checkboxes:
            checkbox.blockSignals(False)
        if self.camera_worker:
            self.camera_worker.classes_filter = self.selected_classes

    def on_class_toggled(self):
        selected_classes = []
        for checkbox in self.classes_checkboxes:
            if checkbox.isChecked():
                selected_classes.append(checkbox.class_id)
        self.selected_classes = selected_classes
        all_checked = len(selected_classes) == len(self.classes_checkboxes)
        none_checked = len(selected_classes) == 0
        self.select_all_checkbox.blockSignals(True)
        if all_checked:
            self.select_all_checkbox.setChecked(True)
        elif none_checked:
            self.select_all_checkbox.setChecked(False)
        else:
            self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.blockSignals(False)
        if self.camera_worker:
            self.camera_worker.classes_filter = self.selected_classes

    def on_tensorrt_changed(self, state):
        self.use_tensorrt = (state == Qt.CheckState.Checked.value or state == 2)
        # Αν το checkbox ενεργοποιηθεί, ανανέωσε το imgsz hint από το .engine αν υπάρχει
        try:
            if self.use_tensorrt and self.available_models:
                idx = self.model_combo.currentIndex()
                if 0 <= idx < len(self.available_models):
                    model_path, model_type = self.available_models[idx]
                    if model_type == 'ONNX':
                        engine_path = Path(str(model_path)).with_suffix('.engine')
                        if engine_path.exists():
                            self.tensorrt_checkbox.setToolTip(
                                f'🔥 TensorRT ενεργό — engine: {engine_path.name}')
        except Exception:
            pass

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        tensorrt_group = QGroupBox('TensorRT Επιτάχυνση')
        tensorrt_layout = QHBoxLayout(tensorrt_group)
        self.tensorrt_checkbox = QCheckBox('🔥 TensorRT')
        self.tensorrt_checkbox.setToolTip('Ενεργοποίηση TensorRT για επιτάχυνση (μόνο για ONNX μοντέλα)')
        self.tensorrt_checkbox.stateChanged.connect(self.on_tensorrt_changed)
        self.tensorrt_checkbox.setEnabled(False)
        tensorrt_layout.addWidget(self.tensorrt_checkbox)
        top_bar_layout.addWidget(tensorrt_group)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        settings_group = QGroupBox('Ρυθμίσεις Κάμερας')
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(10)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['Όλα', 'PyTorch (.pt)', 'CNN (.pt)', 'CNN ONNX (.onnx)', 'ONNX (.onnx)', 'TensorRT (.engine)', 'NCNN'])
        self.model_type_combo.currentIndexChanged.connect(self.refresh_models)
        settings_layout.addRow(QLabel('Τύπος Μοντέλου:'), self.model_type_combo)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['All', 'CPU', 'GPU'])
        self.device_combo.currentIndexChanged.connect(self.refresh_models)
        settings_layout.addRow(QLabel('Συσκευή (Φίλτρο):'), self.device_combo)
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        self.refresh_models_button = QPushButton('🔄')
        self.refresh_models_button.setObjectName('RefreshButton')
        self.refresh_models_button.setFixedWidth(32)
        self.refresh_models_button.setFont(QFont('Segoe UI Emoji'))
        self.refresh_models_button.clicked.connect(self.refresh_models)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.refresh_models_button)
        settings_layout.addRow(QLabel('Μοντέλο:'), model_layout)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setSingleStep(1)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        self.conf_label = QLabel('0.25')
        self.conf_label.setMinimumWidth(40)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addRow(QLabel('Όριο εμπιστοσύνης:'), conf_layout)
        left_layout.addWidget(settings_group)
        self.start_button = QPushButton('📷 Έναρξη Κάμερας')
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton('⏹️ Στοπ Κάμερας')
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        self.stop_button.setObjectName('StopButton')
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)
        classes_group = QGroupBox('Κλάσεις Ανίχνευσης')
        classes_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        classes_layout = QVBoxLayout(classes_group)
        scroll_area = QScrollArea()
        scroll_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll_widget = QWidget()
        self.classes_layout = QVBoxLayout(scroll_widget)
        self.classes_checkboxes = []
        self.selected_classes = None
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        classes_layout.addWidget(scroll_area, 1)
        left_layout.addWidget(classes_group, 1)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(450)
        video_layout = QVBoxLayout()
        preview_group = QGroupBox('📷 Προεπισκόπηση Κάμερας (Live)')
        preview_group_layout = QVBoxLayout(preview_group)
        preview_group_layout.setContentsMargins(12, 12, 12, 12)
        preview_group_layout.setSpacing(10)
        preview_group.setStyleSheet(
            """
            QGroupBox{border-radius:18px;}; QGroupBox::title{subcontrol-origin: margin;left:12px;top:6px;padding:0 6px;font-weight:600;}
            """
        )
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.preview_status_badge = QLabel('⚫ OFF')
        self.preview_status_badge.setObjectName('PreviewStatusBadgeLive')
        self.preview_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_badge = QLabel('🎯 Έτοιμο για εκκίνηση')
        self.preview_info_badge.setObjectName('PreviewInfoBadgeLive')
        self.preview_info_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(self.preview_status_badge, 0)
        header_row.addStretch(1)
        header_row.addWidget(self.preview_info_badge, 0)
        preview_group_layout.addLayout(header_row)
        self.video_label = RoundedMaskLabel("📷 Η κάμερα είναι ανενεργή.\n👉 Επίλεξε μοντέλο και πάτησε 'Έναρξη'.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            """
            QLabel { background-color: #000000; color: #f2f2f2; border: 2px solid #2f2f2f; border-radius: 18px; padding: 0px;}
            """
        )
        badge_style = _BADGE_STYLE_QSS
        self.preview_status_badge.setStyleSheet(badge_style)
        self.preview_info_badge.setStyleSheet(badge_style)
        try:
            shadow = QGraphicsDropShadowEffect(preview_group)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 6)
            shadow.setColor(QColor(0, 0, 0, 160))
            preview_group.setGraphicsEffect(shadow)
        except Exception:
            pass
        try:
            self.video_aspect_frame = AspectRatioFrame(self.video_label, 16, 9)
            self.video_aspect_frame.setObjectName("LiveVideoAspect16x9")
            try:
                self.video_aspect_frame.setStyleSheet("background-color: #000000; border-radius: 18px;")
            except Exception:
                pass
            preview_group_layout.addWidget(self.video_aspect_frame, 1)
        except Exception:
            preview_group_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(preview_group, 1)
        main_layout.addWidget(left_widget)
        main_layout.addLayout(video_layout)
        try:
            main_layout.setStretch(0, 1)
            main_layout.setStretch(1, 3)
        except Exception:
            pass
        outer_layout.addLayout(main_layout, 1)
        self.refresh_models()

    def on_conf_slider_changed(self, value: int):
        conf = value / 100.0
        self.conf_label.setText(f'{conf:.2f}')

    def start_camera(self):
        try:
            if getattr(self, 'camera_py_thread', None) is not None and self.camera_py_thread.is_alive():
                return
        except Exception:
            pass
        try:
            if self.camera_thread is not None and hasattr(self.camera_thread, 'isRunning') and self.camera_thread.isRunning():
                return
        except Exception:
            pass
        try:
            if getattr(self, "_mmpro_cam_start_deferred", False):
                return
        except Exception:
            pass
        try:
            self._mmpro_cam_start_deferred = True
            QTimer.singleShot(0, self._start_camera_now)
        except Exception:
            try:
                self._mmpro_cam_start_deferred = False
            except Exception:
                pass
            self._start_camera_now()

    def _start_camera_now(self):
        try:
            self._mmpro_cam_start_deferred = False
        except Exception:
            pass
        # Stop camera benchmark if running (same device cannot be opened twice)
        try:
            main_win = self.window()
            cam_bench = getattr(main_win, 'camera_benchmark_tab', None)
            if cam_bench is not None:
                bench_thread = getattr(cam_bench, 'benchmark_thread', None)
                if bench_thread is not None and bench_thread.isRunning():
                    reply = QMessageBox.question(
                        self, 'Live Ανίχνευση',
                        'Το Benchmark Κάμερας τρέχει αυτή τη στιγμή.\n'
                        'Πρέπει να σταματήσει πριν ανοίξει η κάμερα.\nΘέλεις να το σταματήσω αυτόματα;',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    try:
                        cam_bench.stop_benchmark()
                    except Exception:
                        pass
                    stop_thread = getattr(cam_bench, 'benchmark_thread', None)
                    if stop_thread is not None:
                        try:
                            stop_thread.wait(3000)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            warmup_torch_cuda("camera_start")
        except Exception:
            pass
        try:
            ensure_windows_com_initialized()
        except Exception:
            pass
        try:
            if self.camera_thread is not None and hasattr(self.camera_thread, 'isRunning') and self.camera_thread.isRunning():
                return
        except Exception:
            pass
        if not self.available_models or self.model_combo.currentIndex() < 0:
            QMessageBox.warning(self, 'Σφάλμα', 'Δεν έχετε επιλέξει έγκυρο μοντέλο.')
            return
        selected_model_info = self.available_models[self.model_combo.currentIndex()]
        imgsz = 640
        try:
            model_path, model_type = selected_model_info
            # Ανίχνευση imgsz από engine για TensorRT (άμεσο .engine ή ONNX+checkbox)
            engine_path = None
            if model_type == 'TensorRT':
                engine_path = Path(str(model_path))
            elif model_type == 'ONNX' and getattr(self, 'use_tensorrt', False):
                engine_path = Path(str(model_path)).with_suffix('.engine')
                if not engine_path.exists():
                    engine_path = None
            if engine_path and engine_path.exists():
                exp = None
                try:
                    sig_path = trt_signature_path_for_engine(engine_path)
                    if sig_path.exists():
                        sig = json.loads(sig_path.read_text(encoding='utf-8', errors='replace'))
                        exp = int((sig.get('params') or {}).get('imgsz') or 0)
                except Exception:
                    exp = None
                if not exp:
                    exp = (_mmpro_parse_imgsz_from_name(engine_path.stem) or
                           _mmpro_parse_imgsz_from_name(engine_path.parent.name))
                if exp:
                    imgsz = int(exp)
        except Exception:
            pass
        conf_threshold = 0.25
        try:
            if hasattr(self, 'conf_slider'):
                conf_threshold = self.conf_slider.value() / 100.0
        except Exception:
            conf_threshold = 0.25
        try:
            model_path, model_type = selected_model_info
            self.preview_status_badge.setText('🟢 LIVE')
            self.preview_info_badge.setText(f"🧠 {model_type} | 🎥 1080p | 🎯 {conf_threshold:.2f}")
            self.preview_info_badge.setToolTip( f"Μοντέλο: {getattr(model_path, 'name', str(model_path))}\n" f"Type: {model_type}\n" f"Camera: 1920x1080 (1080p)\n" f"conf: {conf_threshold:.2f}")
        except Exception:
            pass
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_combo.setEnabled(False)
        self.refresh_models_button.setEnabled(False)
        self.tensorrt_checkbox.setEnabled(False)
        if hasattr(self, 'conf_slider'):
            self.conf_slider.setEnabled(False)
        try:
            th = getattr(self, 'camera_thread', None)
            if th is not None:
                try:
                    still_running = th.isRunning()
                except Exception:
                    still_running = False
                    self.camera_thread = None
                if still_running:
                    try:
                        self.update_log("⚠️ Η κάμερα τρέχει ήδη. Πάτα STOP πριν την επόμενη εκκίνηση.")
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            py_th = getattr(self, 'camera_py_thread', None)
            if py_th is not None and py_th.is_alive():
                try:
                    self.update_log("⚠️ Η κάμερα τρέχει ήδη (Python thread). Πάτα STOP πριν την επόμενη εκκίνηση.")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            camera_index = int(self.camera_index_spin.value()) if hasattr(self, 'camera_index_spin') else int(os.environ.get('MM_PRO_LIVE_CAMERA_INDEX', '0'))
        except Exception:
            camera_index = 0
        # Persist camera settings
        try:
            _settings().set_many({
                'camera_index': camera_index,
                'camera_conf': self.conf_slider.value() if hasattr(self, 'conf_slider') else 25,
                'camera_tensorrt': getattr(self, 'use_tensorrt', False),
            })
        except Exception:
            pass
        try:
            _mode = str(os.environ.get('MM_PRO_CAMERA_THREAD_MODE', '')).strip().lower()
        except Exception:
            _mode = ''
        use_py_thread = False
        try:
            if _mode in ('py', 'python'):
                use_py_thread = True
            elif _mode in ('qt', 'qthread'):
                use_py_thread = False
            else:
                use_py_thread = bool(getattr(sys, 'frozen', False) and sys.platform.startswith('win'))
        except Exception:
            use_py_thread = False
        if use_py_thread:
            try:
                self._camera_thread_mode = 'py'
                self.camera_thread = None
            except Exception:
                pass
            self.camera_worker = CameraWorker(selected_model_info, imgsz, self.selected_classes, self.use_tensorrt, conf_threshold=conf_threshold, camera_index=camera_index)
            try:
                self.camera_worker.finished.connect(self.on_camera_finished, Qt.ConnectionType.QueuedConnection)
            except Exception:
                try:
                    self.camera_worker.finished.connect(self.on_camera_finished)
                except Exception:
                    pass
            try:
                self.camera_worker.log.connect(self.update_log, Qt.ConnectionType.QueuedConnection)
            except Exception:
                try:
                    self.camera_worker.log.connect(self.update_log)
                except Exception:
                    pass
            try:
                self.camera_worker.error.connect(self.on_error, Qt.ConnectionType.QueuedConnection)
            except Exception:
                try:
                    self.camera_worker.error.connect(self.on_error)
                except Exception:
                    pass
            import threading

            def _mmpro_camera_py_entry():
                try:
                    self.camera_worker.run()
                except Exception as e:
                    try:
                        self.camera_worker.error.emit(str(e))
                    except Exception:
                        pass
                    try:
                        self.camera_worker.finished.emit()
                    except Exception:
                        pass
            try:
                enable_faulthandler(CRASH_LOGS_DIR, 'camera')
            except Exception:
                pass
            self.camera_py_thread = threading.Thread(target=_mmpro_camera_py_entry, name='MMProCameraPyThread', daemon=True)
            self.camera_py_thread.start()
            try:
                self._ui_timer.start()
            except Exception:
                pass
        try:
            enable_faulthandler(CRASH_LOGS_DIR, 'camera')
        except Exception:
            pass
        self.camera_thread = QThread()
        try:
            self.camera_thread.setObjectName('MMProCameraThread')
        except Exception:
            pass
        self.camera_worker = CameraWorker(selected_model_info, imgsz, self.selected_classes, self.use_tensorrt, conf_threshold=conf_threshold, camera_index=camera_index)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run, Qt.ConnectionType.QueuedConnection)
        self.camera_worker.finished.connect(self.on_camera_finished)
        self.camera_worker.log.connect(self.update_log)
        self.camera_worker.error.connect(self.on_error)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()
        try:
            self._ui_timer.start()
        except Exception:
            pass

    def stop_camera(self):
        try:
            if hasattr(self, '_ui_timer'):
                self._ui_timer.stop()
        except Exception:
            pass
        try:
            self.preview_status_badge.setText('🟠 STOP')
            self.preview_info_badge.setText('⏳ Διακοπή...')
        except Exception:
            pass
        try:
            self.video_label.clear()
            self.video_label.setText('⏹️ Σταμάτημα...')
        except Exception:
            pass
        try:
            if getattr(self, 'camera_worker', None) is not None:
                self.camera_worker.stop()
        except Exception:
            pass
        try:
            th = getattr(self, 'camera_thread', None)
            if th is not None:
                try:
                    th.quit()
                except Exception:
                    pass
                try:
                    th.wait(3000)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            py_th = getattr(self, 'camera_py_thread', None)
            if py_th is not None and py_th.is_alive():
                py_th.join(timeout=3.0)
        except Exception:
            pass
        try:
            self.camera_worker = None
            self.camera_thread = None
            self.camera_py_thread = None
        except Exception:
            pass
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass
        try:
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.refresh_models_button.setEnabled(True)
            self.tensorrt_checkbox.setEnabled(True)
            if hasattr(self, 'conf_slider'):
                self.conf_slider.setEnabled(True)
        except Exception:
            pass

    def _pull_latest_frame(self):
        try:
            if self.camera_worker is None:
                return
            q_img = None
            if hasattr(self.camera_worker, 'get_latest_qimage'):
                q_img = self.camera_worker.get_latest_qimage()
            if q_img is None:
                return
            overlay_text = ''
            try:
                if hasattr(self.camera_worker, 'get_latest_overlay_text'):
                    overlay_text = self.camera_worker.get_latest_overlay_text() or ''
            except Exception:
                overlay_text = ''
            self.update_frame(q_img, overlay_text)
        except Exception:
            pass

    def update_frame(self, q_image, overlay_text: str = ''):
        try:
            if q_image is None:
                return
            src_pix = QPixmap.fromImage(q_image)
            rect = self.video_label.contentsRect()
            target = rect.size() if rect is not None else self.video_label.size()
            pixmap = src_pix
            try:
                if target.width() > 2 and target.height() > 2:
                    scaled = src_pix.scaled( target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,)
                    pixmap = QPixmap(int(target.width()), int(target.height()))
                    try:
                        pixmap.fill(Qt.GlobalColor.black)
                    except Exception:
                        pass
                    try:
                        dx = int((target.width() - scaled.width()) / 2)
                        dy = int((target.height() - scaled.height()) / 2)
                        p2 = QPainter(pixmap)
                        p2.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                        p2.drawPixmap(dx, dy, scaled)
                        p2.end()
                    except Exception:
                        pixmap = scaled
            except Exception:
                pixmap = src_pix
            try:
                if overlay_text:
                    p = QPainter(pixmap)
                    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                    try:
                        base_px = max(11, int(pixmap.height() * 0.035))
                    except Exception:
                        base_px = 14
                    f = QFont('Segoe UI')
                    f.setPixelSize(base_px)
                    f.setBold(True)
                    p.setFont(f)
                    fm = p.fontMetrics()
                    pad = max(6, int(base_px * 0.45))
                    tw = fm.horizontalAdvance(overlay_text)
                    th = fm.height()
                    box_w = tw + pad * 2
                    box_h = th + pad * 2
                    x0 = 12
                    y0 = 12
                    p.setPen(QPen(QColor(255, 255, 255, 200), 2))
                    p.setBrush(QBrush(QColor(0, 0, 0, 140)))
                    p.drawRoundedRect(x0, y0, box_w, box_h, 10, 10)
                    p.setPen(QPen(QColor(0, 255, 0, 235), 2))
                    p.drawText(x0 + pad, y0 + pad + fm.ascent(), overlay_text)
                    p.end()
            except Exception:
                pass
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setPixmap(pixmap)
        except Exception:
            pass

    def update_log(self, html_text):
        pass

    def on_error(self, error_text):
        QMessageBox.critical(self, 'Σφάλμα Κάμερας', error_text)
        self.stop_camera()

    def on_camera_finished(self):
        try:
            if hasattr(self, '_ui_timer'):
                self._ui_timer.stop()
        except Exception:
            pass
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self.refresh_models_button.setEnabled(True)
        if hasattr(self, 'conf_slider'):
            self.conf_slider.setEnabled(True)
        try:
            self.preview_status_badge.setText('⚫ OFF')
            self.preview_info_badge.setText('🎯 Έτοιμο για εκκίνηση')
            self.preview_info_badge.setToolTip('')
        except Exception:
            pass
        self.video_label.setText("📷 Η κάμερα είναι ανενεργή.\n👉 Επίλεξε μοντέλο και πάτησε 'Έναρξη'.")
        selected_index = self.model_combo.currentIndex()
        if selected_index >= 0 and selected_index < len(self.available_models):
            model_path, model_type = self.available_models[selected_index]
            if model_type == 'ONNX':
                self.tensorrt_checkbox.setEnabled(True)
        try:
            if getattr(self, 'camera_py_thread', None) is not None:
                try:
                    self.camera_py_thread.join(timeout=2.0)
                except Exception:
                    pass
                self.camera_py_thread = None
        except Exception:
            pass
        if self.camera_thread:
            self.camera_thread.quit()
            self.camera_thread.wait()
            self.camera_thread = None
            self.camera_worker = None
        self.on_model_selected()


class CameraTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('📷 Live Ανίχνευση - Standalone')
        self.setCentralWidget(CameraTab())
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def camera_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = CameraTabWindow()
    win.show()
    sys.exit(app.exec())
"""Camera Benchmark tab (UI).
Παρόμοιο με το Live Camera tab, αλλά:
- Εκκινεί `CameraBenchmarkWorker` που μετρά FPS ανά backend σε πραγματική κάμερα.; - Παρουσιάζει συγκεντρωτικά αποτελέσματα και live preview με σταθερό aspect ratio.
"""
try:
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
except Exception:
    QGraphicsDropShadowEffect = None


class CameraBenchmarkTab(QWidget, TabNavigationMixin, BenchmarkUIHelpersMixin):

    def __init__(self, parent: QWidget | None=None):
        super().__init__(parent)
        self.models_dir = MODELS_DIR_TRAINED_PT
        self.benchmark_thread: QThread | None = None
        self.benchmark_worker: CameraBenchmarkWorker | None = None
        self.init_ui()

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        top_separator = add_blue_separator(outer_layout)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        settings_group = QGroupBox('Ρυθμίσεις Benchmark Κάμερας')
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(10)
        self.models_combo = QComboBox()
        self.models_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        settings_layout.addRow(QLabel('Εκπαιδευμένο Μοντέλο:'), self.models_combo)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(224, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        settings_layout.addRow(QLabel('Image size:'), self.imgsz_spin)
        self.imgsz_spin.setReadOnly(True)
        self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.models_combo.currentTextChanged.connect(self.on_model_changed)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(3, 120)
        self.duration_spin.setSingleStep(1)
        self.duration_spin.setValue(10)
        settings_layout.addRow(QLabel('Διάρκεια ανά backend (sec):'), self.duration_spin)
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 4)
        self.camera_index_spin.setValue(0)
        settings_layout.addRow(QLabel('Δείκτης κάμερας:'), self.camera_index_spin)
        left_layout.addWidget(settings_group)
        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton('🎥 Εκτέλεση Benchmark Κάμερας')
        self.stop_button = QPushButton('⏹️ Διακοπή')
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.stop_button)
        left_layout.addLayout(buttons_layout)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(['Backend', 'FPS', 'ms / εικόνα'])
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setColumnWidth(0, 224)
        self.results_table.setColumnWidth(1, 112)
        self.results_table.setColumnWidth(2, 80)
        log_group = QGroupBox('Log Benchmark Κάμερας')
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setObjectName('CameraBenchmarkLogOutput')
        self.log_edit.setMinimumHeight(320)
        self.log_edit.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_group, 1)
        main_layout.addLayout(left_layout, 1)
        preview_group = QGroupBox('📷 Προεπισκόπηση Κάμερας (Benchmark)')
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(10)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.preview_status_badge = QLabel('⚫ IDLE')
        self.preview_status_badge.setObjectName('PreviewStatusBadgeBenchmark')
        self.preview_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_badge = QLabel('🧪 Έτοιμο για benchmark')
        self.preview_info_badge.setObjectName('PreviewInfoBadgeBenchmark')
        self.preview_info_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(self.preview_status_badge, 0)
        header_row.addStretch(1)
        header_row.addWidget(self.preview_info_badge, 0)
        preview_layout.addLayout(header_row)
        self.video_label = RoundedMaskLabel("📷 Η προεπισκόπηση κάμερας θα εμφανιστεί εδώ.\n👉 Επίλεξε μοντέλο και πάτησε 'Εκτέλεση Benchmark Κάμερας'.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(832, 468)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            """
            QLabel { background-color: #000000; color: #f2f2f2; border: 2px solid #2f2f2f; border-radius: 18px; padding: 0px;}
            """
        )
        badge_style = _BADGE_STYLE_QSS
        self.preview_status_badge.setStyleSheet(badge_style)
        self.preview_info_badge.setStyleSheet(badge_style)
        try:
            shadow = QGraphicsDropShadowEffect(preview_group)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 6)
            shadow.setColor(QColor(0, 0, 0, 160))
            preview_group.setGraphicsEffect(shadow)
        except Exception:
            pass
        self._default_video_label_text = self.video_label.text()
        self._default_video_label_stylesheet = self.video_label.styleSheet()
        preview_layout.addWidget(self.video_label)
        self.results_table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.results_table)
        main_layout.addWidget(preview_group, 1)
        outer_layout.addLayout(main_layout, 1)
        self.run_button.clicked.connect(self.start_benchmark)
        self.stop_button.clicked.connect(self.stop_benchmark)
        self.refresh_models()

    def refresh_models(self):
        self.models_combo.clear()
        benchmark_running = self.benchmark_thread is not None and self.benchmark_thread.isRunning()
        if not self.models_dir.exists():
            self.models_combo.addItem('<< Ο φάκελος μοντέλων δεν υπάρχει >>')
            self.run_button.setEnabled(False)
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            self.models_combo.addItem('<< Δεν βρέθηκαν μοντέλα >>')
            self.run_button.setEnabled(False)
        model_dirs_sorted = sorted(model_dirs, key=lambda p: p.name)
        base_names_sorted = [d.name for d in model_dirs_sorted]
        target_name = None
        try:
            main_window = self.window()
        except Exception:
            main_window = None
        if main_window is not None and hasattr(main_window, 'current_trained_model_stem'):
            target_stem = getattr(main_window, 'current_trained_model_stem', '') or ''
            if target_stem:
                target_normalized = target_stem.replace('_ncnn_model', '')
                for name in base_names_sorted:
                    if name == target_normalized or name.replace('_ncnn_model', '') == target_normalized:
                        target_name = name
                        break
        if not target_name:
            try:
                last_dir = max(model_dirs_sorted, key=lambda d: d.stat().st_mtime)
                target_name = last_dir.name
            except Exception:
                target_name = base_names_sorted[-1] if base_names_sorted else None
        self.models_combo.addItem('<< Επέλεξε μοντέλο >>')
        for name in base_names_sorted:
            self.models_combo.addItem(name)
        if target_name and target_name in base_names_sorted:
            try:
                idx = base_names_sorted.index(target_name)
                self.models_combo.setCurrentIndex(idx + 1)
            except Exception:
                pass
        self.on_model_changed(self.models_combo.currentText())
        self.run_button.setEnabled(not benchmark_running)

    def on_model_changed(self, _text: str):
        try:
            name = self.models_combo.currentText().strip()
        except Exception:
            name = ''
        val = self._parse_imgsz_from_name(name)
        if val is not None:
            try:
                self.imgsz_spin.setValue(int(val))
            except Exception:
                pass
        try:
            if hasattr(self, 'imgsz_spin'):
                self.imgsz_spin.setReadOnly(True)
                self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        except Exception:
            pass

    def start_benchmark(self):
        current_text = self.models_combo.currentText().strip()
        if not current_text or current_text.startswith('<<'):
            QMessageBox.warning(self, 'Benchmark Κάμερας', 'Δεν έχει επιλεγεί έγκυρο όνομα μοντέλου.')
            return
        base_name = current_text
        imgsz = self.imgsz_spin.value()
        duration = self.duration_spin.value()
        camera_index = self.camera_index_spin.value()
        try:
            if getattr(self, '_mmpro_bench_start_deferred', False):
                return
        except Exception:
            pass
        if self.benchmark_thread is not None and self.benchmark_thread.isRunning():
            QMessageBox.information(self, 'Benchmark Κάμερας', 'Ήδη εκτελείται benchmark κάμερας. Περίμενε να ολοκληρωθεί ή πάτησε Διακοπή.')
            return
        # Stop live camera if running (same device cannot be opened twice)
        try:
            main_win = self.window()
            camera_tab = getattr(main_win, 'camera_tab', None)
            if camera_tab is not None:
                cam_thread = getattr(camera_tab, 'camera_thread', None)
                if cam_thread is not None and cam_thread.isRunning():
                    reply = QMessageBox.question(
                        self, 'Benchmark Κάμερας',
                        'Η κάμερα είναι ανοιχτή στο tab "Live Ανίχνευση".\n'
                        'Πρέπει να σταματήσει πριν ξεκινήσει το benchmark.\nΘέλεις να τη σταματήσω αυτόματα;',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    try:
                        camera_tab.stop_camera()
                    except Exception:
                        pass
                    # Give the camera thread time to release the device
                    stop_thread = getattr(camera_tab, 'camera_thread', None)
                    if stop_thread is not None:
                        try:
                            stop_thread.wait(3000)
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            warmup_torch_cuda("camera_benchmark_start")
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            ensure_windows_com_initialized()
        except Exception:
            pass
        try:
            ensure_windows_media_foundation_started()
        except Exception:
            pass
        try:
            self.preview_status_badge.setText('🟡 RUNNING')
            self.preview_info_badge.setText(f"📷 Cam {camera_index} | ⏱️ {duration}s | 🖼️ {imgsz}")
            self.preview_info_badge.setToolTip( f"Μοντέλο: {base_name}\nCam index: {camera_index}\nDuration: {duration}s\nimgsz: {imgsz}")
        except Exception:
            pass
        try:
            enable_faulthandler(CRASH_LOGS_DIR, 'camera_benchmark')
        except Exception:
            pass
        self.log_edit.clear()
        self.results_table.setRowCount(0)
        self.benchmark_thread = QThread()
        self.benchmark_thread.setObjectName('MMProCamBenchThread')
        model_dir = self.models_dir / base_name
        self.benchmark_worker = CameraBenchmarkWorker( base_name=base_name, imgsz=imgsz, duration_sec=duration, camera_index=camera_index, models_dir=model_dir,)
        self.benchmark_worker.moveToThread(self.benchmark_thread)
        self.benchmark_thread.started.connect(self.benchmark_worker.run, Qt.ConnectionType.QueuedConnection)
        self.benchmark_worker.log.connect(self.append_log)
        self.benchmark_worker.frame_ready.connect(self.update_frame)
        self.benchmark_worker.error.connect(self.on_worker_error)
        self.benchmark_worker.results_ready.connect(self.on_worker_results)
        self.benchmark_worker.finished.connect(self.on_worker_finished)
        self.benchmark_worker.finished.connect(self.benchmark_thread.quit)
        self.benchmark_worker.finished.connect(self.benchmark_worker.deleteLater)
        self.benchmark_thread.finished.connect(self.benchmark_thread.deleteLater)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.models_combo.setEnabled(False)
        self.imgsz_spin.setReadOnly(True)
        self.imgsz_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.duration_spin.setEnabled(False)
        self.camera_index_spin.setEnabled(False)
        self._mmpro_bench_start_deferred = True
        QTimer.singleShot(0, self._start_benchmark_thread)

    def _start_benchmark_thread(self) -> None:
        try:
            self._mmpro_bench_start_deferred = False
        except Exception:
            pass
        try:
            if self.benchmark_thread is not None:
                self.benchmark_thread.start()
        except Exception as e:
            _MMPRO_LOGGER.error("_start_benchmark_thread error: %s", e)
            try:
                self.run_button.setEnabled(True)
                self.stop_button.setEnabled(False)
            except Exception:
                pass

    def stop_benchmark(self):
        if self.benchmark_worker is not None:
            self.benchmark_worker.stop()
        self.stop_button.setEnabled(False)
        try:
            self.preview_status_badge.setText('🟠 STOPPING')
            self.preview_info_badge.setText('⏳ Διακοπή...')
        except Exception:
            pass

    def reset_video_preview(self):
        if hasattr(self, 'video_label'):
            self.video_label.clear()
            default_text = getattr(self, '_default_video_label_text', '')
            default_style = getattr(self, '_default_video_label_stylesheet', '')
            if default_text:
                self.video_label.setText(default_text)
            if default_style:
                self.video_label.setStyleSheet(default_style)

    def update_frame(self, image: QImage):
        try:
            if image is None:
                return
            src_pix = QPixmap.fromImage(image)
            rect = self.video_label.contentsRect()
            target = rect.size() if rect is not None else self.video_label.size()
            pixmap = src_pix
            try:
                if target.width() > 2 and target.height() > 2:
                    scaled = src_pix.scaled( target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,)
                    pixmap = QPixmap(int(target.width()), int(target.height()))
                    try:
                        pixmap.fill(Qt.GlobalColor.black)
                    except Exception:
                        pass
                    try:
                        dx = int((target.width() - scaled.width()) / 2)
                        dy = int((target.height() - scaled.height()) / 2)
                        p2 = QPainter(pixmap)
                        p2.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                        p2.drawPixmap(dx, dy, scaled)
                        p2.end()
                    except Exception:
                        pixmap = scaled
            except Exception:
                pixmap = src_pix
            try:
                overlay_text = ''
                if getattr(self, 'benchmark_worker', None) is not None and hasattr(self.benchmark_worker, 'get_latest_overlay_text'):
                    overlay_text = self.benchmark_worker.get_latest_overlay_text() or ''
                if overlay_text:
                    p = QPainter(pixmap)
                    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                    try:
                        base_px = max(11, int(pixmap.height() * 0.035))
                    except Exception:
                        base_px = 14
                    f = QFont('Segoe UI')
                    f.setPixelSize(base_px)
                    f.setBold(True)
                    p.setFont(f)
                    fm = p.fontMetrics()
                    pad = max(6, int(base_px * 0.45))
                    tw = fm.horizontalAdvance(overlay_text)
                    th = fm.height()
                    box_w = tw + pad * 2
                    box_h = th + pad * 2
                    x0 = 12
                    y0 = 12
                    p.setPen(QPen(QColor(255, 255, 255, 200), 2))
                    p.setBrush(QBrush(QColor(0, 0, 0, 140)))
                    p.drawRoundedRect(x0, y0, box_w, box_h, 10, 10)
                    p.setPen(QPen(QColor(0, 255, 0, 235), 2))
                    p.drawText(x0 + pad, y0 + pad + fm.ascent(), overlay_text)
                    p.end()
            except Exception:
                pass
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setPixmap(pixmap)
        except Exception:
            pass

    def on_worker_error(self, msg: str):
        QMessageBox.critical(self, 'Benchmark Κάμερας - Σφάλμα', msg)

    def _clear_bench_refs(self):
        self.benchmark_thread = None
        self.benchmark_worker = None

    def on_worker_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.models_combo.setEnabled(True)
        self.imgsz_spin.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.camera_index_spin.setEnabled(True)
        try:
            self.preview_status_badge.setText('⚫ IDLE')
            self.preview_info_badge.setText('🧪 Έτοιμο για benchmark')
            self.preview_info_badge.setToolTip('')
        except Exception:
            pass
        self.reset_video_preview()
        # Defer nulling refs until AFTER thread.finished fires and deleteLater is queued,
        # otherwise Python GC destroys the QThread object while Qt still holds signals queued to it.
        try:
            th = self.benchmark_thread
            if th is not None:
                th.finished.connect(self._clear_bench_refs, Qt.ConnectionType.SingleShotConnection)
            else:
                self._clear_bench_refs()
        except Exception:
            self.benchmark_thread = None
            self.benchmark_worker = None
        try:
            summary = flush_log_once_summary('Camera Benchmark', reset=True, top_n=12, min_total=1)
            if summary:
                try:
                    self.append_log(format_html_summary(summary))
                except Exception:
                    try:
                        self.log_edit.append(format_html_summary(summary))
                    except Exception:
                        pass
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.stop_benchmark()
        except Exception:
            pass
        try:
            th = self.benchmark_thread
            if th is not None and th.isRunning():
                th.quit()
                th.wait(2000)
        except RuntimeError:
            pass
        except Exception:
            pass
        try:
            self.benchmark_thread = None
            self.benchmark_worker = None
        except Exception:
            pass
        try:
            event.accept()
        except Exception:
            pass


class CameraBenchmarkTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('🎥 Benchmark Κάμερας - Standalone')
        self.setCentralWidget(CameraBenchmarkTab())
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def camera_benchmark_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = CameraBenchmarkTabWindow()
    win.show()
    sys.exit(app.exec())
"""Statistics tab (UI).
UI για στατιστικά ανίχνευσης, previews και εξαγωγή reports.
"""

class VideoTab(QWidget, TabNavigationMixin):
    """Tab για inference σε video αρχεία με αποθήκευση αποτελέσματος."""

    def __init__(self):
        super().__init__()
        self.video_thread: QThread | None = None
        self.video_worker: VideoFileWorker | None = None
        self.available_models: list = []
        self.filtered_models: list = []
        self._current_video: str = ''
        self.init_ui()
        QTimer.singleShot(0, self.refresh_models)
        # Restore settings
        s = _settings()
        try:
            s.restore_combo(self.model_type_combo, 'video_model_type')
            s.restore_check(self.save_output_check, 'video_save_output')
            s.restore_spin(self.imgsz_spin, 'video_imgsz')
            self._on_type_changed()
            s.restore_combo(self.model_combo, 'video_model')
        except Exception:
            pass

    def init_ui(self):
        outer_layout, top_bar_layout = _make_tab_layout(self)
        # Top bar controls
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['PyTorch (.pt)', 'ONNX (.onnx)', 'NCNN (.ncnn)'])
        self.model_type_combo.currentTextChanged.connect(self._on_type_changed)
        top_bar_layout.addWidget(QLabel('Τύπος:'))
        top_bar_layout.addWidget(self.model_type_combo)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        top_bar_layout.addWidget(QLabel('Μοντέλο:'))
        top_bar_layout.addWidget(self.model_combo)
        self.refresh_btn = QPushButton('🔄')
        self.refresh_btn.setFixedWidth(36)
        self.refresh_btn.setToolTip('Ανανέωση λίστας μοντέλων')
        self.refresh_btn.clicked.connect(self.refresh_models)
        top_bar_layout.addWidget(self.refresh_btn)
        top_bar_layout.addStretch()
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1920)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        top_bar_layout.addWidget(QLabel('imgsz:'))
        top_bar_layout.addWidget(self.imgsz_spin)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        top_bar_layout.addWidget(QLabel('conf:'))
        top_bar_layout.addWidget(self.conf_spin)
        self.save_output_check = QCheckBox('💾 Αποθήκευση αποτελέσματος')
        self.save_output_check.setChecked(True)
        top_bar_layout.addWidget(self.save_output_check)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        # Main area
        main = QHBoxLayout()
        # Left panel
        left = QVBoxLayout()
        left.setSpacing(8)
        # File selection
        file_group = QGroupBox('📂 Επιλογή Video')
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel('Δεν έχει επιλεγεί αρχείο.')
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet('color: #888; font-style: italic;')
        self.select_file_btn = QPushButton('📂 Επιλογή Video Αρχείου...')
        self.select_file_btn.clicked.connect(self.select_video_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_btn)
        left.addWidget(file_group)
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left.addWidget(self.progress_bar)
        # Controls
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton('▶ Εκτέλεση Inference')
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn = QPushButton('⏹ Διακοπή')
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        left.addLayout(ctrl)
        # Log
        log_group = QGroupBox('📋 Αρχείο καταγραφής')
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        mono = QFont('Consolas', 10) if os.name == 'nt' else QFont('Monospace', 10)
        self.log_output.setFont(mono)
        log_layout.addWidget(self.log_output)
        left.addWidget(log_group, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(440)
        main.addWidget(left_widget)
        # Right: live preview
        preview_group = QGroupBox('🎥 Προεπισκόπηση')
        preview_layout = QVBoxLayout(preview_group)
        self.video_label = QLabel('Επιλέξτε video αρχείο και μοντέλο για να ξεκινήσετε.')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet('background:#111;color:#aaa;border-radius:8px;padding:16px;font-size:18px;')
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.video_label, 1)
        self.overlay_label = QLabel('')
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet(_BADGE_STYLE_QSS)
        preview_layout.addWidget(self.overlay_label)
        main.addWidget(preview_group, 1)
        outer_layout.addLayout(main, 1)
        # UI timer for pulling frames
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._pull_frame)

    def refresh_models(self):
        self.model_combo.clear()
        mtype = self.model_type_combo.currentText()
        self.filtered_models = []
        if not TRAINED_MODELS_DIR.exists():
            return
        ext_map = {'PyTorch (.pt)': '.pt', 'ONNX (.onnx)': '.onnx', 'NCNN (.ncnn)': ''}
        ext = ext_map.get(mtype, '.pt')
        try:
            for md in sorted(TRAINED_MODELS_DIR.iterdir()):
                if not md.is_dir():
                    continue
                if ext:
                    for f in sorted(md.rglob(f'*{ext}')):
                        self.model_combo.addItem(f'[{md.name}] {f.name}')
                        self.filtered_models.append((f, mtype.split(' ')[0]))
                else:
                    # NCNN dirs
                    for d in sorted(md.rglob('*.ncnn')):
                        if d.is_dir():
                            self.model_combo.addItem(f'[{md.name}] {d.name}')
                            self.filtered_models.append((d, 'NCNN'))
        except Exception:
            pass
        if not self.filtered_models:
            self.model_combo.addItem('<< Δεν βρέθηκαν μοντέλα >>')

    def _on_type_changed(self):
        self.refresh_models()

    def select_video_file(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, 'Επιλογή Video Αρχείου', str(Path.home()),
            'Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;All Files (*)',
        )
        if path:
            self._current_video = path
            self.file_label.setText(Path(path).name)
            self.file_label.setStyleSheet('color: #222; font-style: normal; font-weight: 600;')
            self.file_label.setToolTip(path)

    def start_inference(self):
        if not self._current_video or not Path(self._current_video).exists():
            QMessageBox.warning(self, 'Video Inference', 'Επιλέξτε πρώτα ένα έγκυρο video αρχείο.')
            return
        idx = self.model_combo.currentIndex()
        if idx < 0 or idx >= len(self.filtered_models):
            QMessageBox.warning(self, 'Video Inference', 'Επιλέξτε ένα έγκυρο μοντέλο.')
            return
        if self.video_thread is not None and self.video_thread.isRunning():
            QMessageBox.information(self, 'Video Inference', 'Ήδη εκτελείται inference. Πάτα Διακοπή πρώτα.')
            return
        model_path, model_type = self.filtered_models[idx]
        model_info = (model_path, model_type)
        # Save settings
        _settings().set_many({
            'video_model_type': self.model_type_combo.currentText(),
            'video_model': self.model_combo.currentText(),
            'video_save_output': self.save_output_check.isChecked(),
            'video_imgsz': self.imgsz_spin.value(),
        })
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        try:
            enable_faulthandler(CRASH_LOGS_DIR, 'video_inference')
        except Exception:
            pass
        self.video_worker = VideoFileWorker(
            model_info=model_info,
            video_path=self._current_video,
            imgsz=self.imgsz_spin.value(),
            conf_threshold=self.conf_spin.value(),
            save_output=self.save_output_check.isChecked(),
        )
        self.video_thread = QThread()
        self.video_thread.setObjectName('MMProVideoThread')
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.log.connect(self.update_log)
        self.video_worker.progress.connect(self.progress_bar.setValue)
        self.video_worker.error.connect(self._on_error)
        self.video_worker.finished.connect(self._on_finished)
        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_worker.finished.connect(self.video_worker.deleteLater)
        self.video_thread.finished.connect(self.video_thread.deleteLater)
        self.video_thread.finished.connect(self._clear_thread_refs, Qt.ConnectionType.SingleShotConnection)
        self._ui_timer.start()
        self.video_thread.start()

    def stop_inference(self):
        if self.video_worker is not None:
            self.video_worker.stop()
        self.stop_btn.setEnabled(False)

    def _pull_frame(self):
        w = self.video_worker
        if w is None:
            return
        qi = w.get_latest_qimage()
        if qi is None:
            return
        try:
            pix = QPixmap.fromImage(qi)
            rect = self.video_label.contentsRect()
            target = rect.size() if rect is not None else self.video_label.size()
            if target.width() > 2 and target.height() > 2:
                scaled = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            else:
                scaled = pix
            self.video_label.setPixmap(scaled)
            overlay = w.get_latest_overlay_text()
            self.overlay_label.setText(overlay)
        except Exception:
            pass

    def _on_error(self, msg: str):
        self.update_log(format_html_log(f'❌ Σφάλμα: {msg}', Colors.RED, bold=True))
        QMessageBox.critical(self, 'Video Inference - Σφάλμα', msg)

    def _on_finished(self):
        self._ui_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.overlay_label.setText('')

    def _clear_thread_refs(self):
        self.video_thread = None
        self.video_worker = None

    def update_log(self, html_text: str):
        try:
            self.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.log_output.insertHtml(html_text + '<br>')
            self.log_output.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.stop_inference()
        except Exception:
            pass
        try:
            th = self.video_thread
            if th is not None and th.isRunning():
                th.quit()
                th.wait(2000)
        except RuntimeError:
            pass
        except Exception:
            pass
        try:
            event.accept()
        except Exception:
            pass

class StatisticsTab(QWidget):
    analysis_completed = Signal()

    def __init__(self):
        super().__init__()
        self.stats_thread: QThread | None = None
        self.stats_worker: StatisticsWorker | None = None
        self.last_report_path: str | None = None
        self.preview_dialog: DetectionPreviewDialog | None = None
        self._preview_samples: list[tuple[str, str, str, str]] = []
        self._preview_paths_to_cleanup: list[Path] = []
        self.available_models: list[tuple[Path, str]] = []
        self.filtered_models: list[tuple[Path, str]] = []
        self.available_datasets: list[Path] = self.find_datasets()
        self.init_ui()

    def find_available_models(self) -> list[tuple[Path, str]]:
        models: list[tuple[Path, str]] = []
        if MODELS_DIR_TRAINED_PT.exists():
            for d in sorted([d for d in MODELS_DIR_TRAINED_PT.iterdir() if d.is_dir()]):
                base = d.name
                pt_path = d / f'{base}.pt'
                if pt_path.exists():
                    # Ανίχνευση CNN: ελέγχει class_names.json ή stem name
                    pt_type = 'CNN' if _is_cnn_path(pt_path) else 'PyTorch'
                    models.append((pt_path, pt_type))
                ncnn_dir = d / f'{base}_ncnn_model'
                if ncnn_dir.exists() and ncnn_dir.is_dir() and (
                        (ncnn_dir / 'model.ncnn.param').exists() or
                        (ncnn_dir / 'model.ncnn.bin').exists() or
                        (ncnn_dir / 'model.param').exists()):
                    models.append((ncnn_dir, 'NCNN'))
                onnx_path = d / f'{base}.onnx'
                if onnx_path.exists():
                    onnx_type = 'CNN-ONNX' if _is_cnn_path(onnx_path) else 'ONNX'
                    models.append((onnx_path, onnx_type))
                engine_path = d / f'{base}.engine'
                if engine_path.exists():
                    models.append((engine_path, 'TensorRT'))
        return models

    def find_datasets(self) -> list[Path]:
        if not DATASETS_DIR.exists():
            return []
        return [d for d in DATASETS_DIR.iterdir() if d.is_dir() and d.name.lower() != 'models']

    def refresh_data(self) -> None:
        selected_type = self.model_type_combo.currentText() if hasattr(self, 'model_type_combo') else 'PyTorch'
        self.available_models = self.find_available_models()
        self.model_combo.clear()
        self.filtered_models = []
        filtered_models_with_time: list[tuple[Path, str, float]] = []
        for path, mtype in self.available_models:
            # CNN models always show (regardless of type filter)
            is_cnn_entry = mtype in ('CNN', 'CNN-ONNX')
            if selected_type != 'All' and mtype != selected_type and not is_cnn_entry:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            filtered_models_with_time.append((path, mtype, mtime))
        filtered_models_with_time.sort(key=lambda x: x[2], reverse=True)
        self.filtered_models = [(p, mtype) for p, mtype, _ in filtered_models_with_time]
        if not self.filtered_models:
            self.model_combo.addItems(['Δεν βρέθηκαν μοντέλα'])
            self.model_combo.setEnabled(False)
        else:
            display_names = [f'[{mtype}] {path.name}' for path, mtype in self.filtered_models]
            self.model_combo.addItems(display_names)
            self.model_combo.setEnabled(True)
            selected_index = -1
            main_window = self.window()
            try:
                if main_window is not None and hasattr(main_window, 'current_trained_model_path'):
                    target_path = getattr(main_window, 'current_trained_model_path', None)
                    target_stem = getattr(main_window, 'current_trained_model_stem', '') or ''
                    if target_path is not None:
                        for i, (p, mtype) in enumerate(self.filtered_models):
                            if p == target_path:
                                selected_index = i
                                break
                    if selected_index < 0 and target_stem:
                        for i, (p, mtype) in enumerate(self.filtered_models):
                            try:
                                if p.stem == target_stem:
                                    selected_index = i
                                    break
                            except Exception:
                                continue
            except Exception:
                selected_index = -1
            if selected_index < 0 and self.filtered_models:
                selected_index = 0
            if 0 <= selected_index < len(self.filtered_models):
                self.model_combo.setCurrentIndex(selected_index)
        self.available_datasets = self.find_datasets()
        self.dataset_combo.clear()
        if not self.available_datasets:
            self.dataset_combo.addItems(['Δεν βρέθηκαν datasets'])
            self.dataset_combo.setEnabled(False)
        else:
            self.dataset_combo.addItems([d.name for d in self.available_datasets])
            self.dataset_combo.setEnabled(True)
            try:
                self.auto_select_dataset_from_model()
            except Exception:
                pass
        self.start_button.setEnabled(bool(self.filtered_models and self.available_datasets))
        if self.last_report_path and Path(self.last_report_path).exists():
            self.view_report_button.setEnabled(True)
        else:
            self.view_report_button.setEnabled(False)

    def auto_select_dataset_from_model(self, index: int | None=None) -> None:
        try:
            if index is None:
                index = self.model_combo.currentIndex()
            if not self.filtered_models or index < 0 or index >= len(self.filtered_models):
                return
            model_path, model_type = self.filtered_models[index]
            name = model_path.name
            base = name.replace('_ncnn_model', '')
            if model_path.is_file() and '.' in base:
                base = base.rsplit('.', 1)[0]
            parts = base.split('_')
            if len(parts) < 3:
                return
            dataset_candidate = parts[-2]
            if not dataset_candidate:
                return
            if not hasattr(self, 'dataset_combo'):
                return
            idx = self.dataset_combo.findText(dataset_candidate)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def init_ui(self) -> None:
        outer_layout, top_bar_layout = _make_tab_layout(self)
        _finish_tab_topbar(self, top_bar_layout, outer_layout)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(15)
        form_layout = QFormLayout()
        form_layout.setSpacing(8)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['All', 'PyTorch', 'ONNX', 'TensorRT', 'NCNN'])
        self.model_type_combo.setCurrentText('PyTorch')
        self.model_type_combo.currentIndexChanged.connect(self.refresh_data)
        device_row_widget = QWidget()
        device_row_layout = QHBoxLayout(device_row_widget)
        device_row_layout.setContentsMargins(0, 0, 0, 0)
        device_row_layout.setSpacing(6)
        device_row_layout.addWidget(self.model_type_combo)
        self.refresh_button = QPushButton('🔄')
        self.refresh_button.setObjectName('RefreshButton')
        self.refresh_button.setFixedWidth(32)
        self.refresh_button.setFont(QFont('Segoe UI Emoji'))
        self.refresh_button.setToolTip('Επαναφόρτωση λίστας μοντέλων και datasets.')
        self.refresh_button.clicked.connect(self.refresh_data)
        device_row_layout.addWidget(self.refresh_button)
        form_layout.addRow(QLabel('Τύπος Μοντέλου:'), device_row_widget)
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_combo.currentIndexChanged.connect(self.auto_select_dataset_from_model)
        form_layout.addRow(QLabel('Μοντέλο:'), self.model_combo)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        form_layout.addRow(QLabel('Dataset για Ανάλυση:'), self.dataset_combo)
        self.num_detections_spin = QSpinBox()
        self.num_detections_spin.setRange(1, 1000000)
        self.num_detections_spin.setValue(100)
        self.num_detections_spin.setSingleStep(10)
        self.num_detections_spin.setToolTip('Μέγιστος αριθμός εικόνων του dataset που θα αναλυθούν.')
        form_layout.addRow(QLabel('Αριθμός ανιχνεύσεων:'), self.num_detections_spin)
        settings_layout.addLayout(form_layout)
        self.start_button = QPushButton('📊 Έναρξη Ανάλυσης (εικόνες dataset)')
        self.start_button.clicked.connect(self.start_analysis)
        settings_layout.addWidget(self.start_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        settings_layout.addWidget(self.progress_bar)
        self.view_report_button = QPushButton('📄 Προβολή Αναφοράς')
        self.view_report_button.clicked.connect(self.view_report)
        self.view_report_button.setEnabled(False)
        settings_layout.addWidget(self.view_report_button)
        self.manage_reports_button = QPushButton('📂 Διαχείριση Αναφορών Ανίχνευσης')
        self.manage_reports_button.clicked.connect(self.manage_reports)
        settings_layout.addWidget(self.manage_reports_button)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(QLabel('Log Ανάλυσης Dataset:'))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName('LogOutput')
        self.log_output.setMinimumHeight(260)
        self.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        stats_mono_font = QFont('Consolas')
        stats_mono_font.setStyleHint(QFont.StyleHint.Monospace)
        stats_mono_font.setPointSize(11)
        self.log_output.setFont(stats_mono_font)
        self.log_output.setWordWrapMode(QTextOption.NoWrap)
        settings_layout.addWidget(self.log_output)
        settings_widget = QWidget()
        settings_widget.setLayout(settings_layout)
        settings_widget.setMaximumWidth(450)
        results_layout = QVBoxLayout()
        results_layout.addWidget(QLabel('Σύνοψη Αποτελεσμάτων:'))
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setObjectName('LogOutput')
        self.summary_output.setFont(stats_mono_font)
        results_layout.addWidget(self.summary_output)
        main_layout.addWidget(settings_widget)
        main_layout.addLayout(results_layout)
        outer_layout.addLayout(main_layout, 1)
        self.refresh_data()

    def toggle_theme(self) -> None:
        main_window = self.window()
        if hasattr(main_window, 'toggle_theme_global'):
            main_window.toggle_theme_global()

    def go_to_dashboard(self) -> None:
        main_window = self.window()
        try:
            if main_window is not None and hasattr(main_window, 'tabs') and (main_window.tabs is not None):
                main_window.tabs.setCurrentIndex(0)
        except Exception:
            pass

    def manage_reports(self) -> None:
        open_folder_externally(str(DETECTION_REPORTS_DIR))

    def start_analysis(self) -> None:
        selected_model_index = self.model_combo.currentIndex()
        selected_dataset_index = self.dataset_combo.currentIndex()
        if not self.filtered_models:
            QMessageBox.warning(self, 'Σφάλμα', 'Δεν υπάρχουν εκπαιδευμένα μοντέλα διαθέσιμα για ανάλυση.\nΠαρακαλώ εκπαιδεύστε πρώτα ένα μοντέλο.')
        if not self.available_datasets:
            QMessageBox.warning(self, 'Σφάλμα', 'Δεν υπάρχουν διαθέσιμα datasets για ανάλυση.\nΠαρακαλώ προσθέστε ή επιλέξτε ένα dataset.')
        if selected_model_index < 0 or selected_model_index >= len(self.filtered_models) or selected_dataset_index < 0 or (selected_dataset_index >= len(self.available_datasets)):
            QMessageBox.warning(self, 'Σφάλμα', 'Δεν έχετε επιλέξει έγκυρο μοντέλο ή dataset.')
        model_path, model_type = self.filtered_models[selected_model_index]
        dataset_path = self.available_datasets[selected_dataset_index]
        dataset_name = dataset_path.name
        if not dataset_path.is_dir():
            QMessageBox.warning(self, 'Σφάλμα', f"Δεν βρέθηκε dataset με όνομα '{dataset_name}' μέσα στο Data_Sets.\nΒεβαιωθείτε ότι υπάρχει κατάλογος με αυτό το όνομα.")
        try:
            self.log_output.clear()
            self.summary_output.clear()
        except Exception:
            pass
        self.progress_bar.setValue(0)
        self.last_report_path = None
        self.view_report_button.setEnabled(False)
        self.cleanup_preview_files()
        try:
            if self.stats_worker is not None and hasattr(self.stats_worker, 'stop'):
                self.stats_worker.stop()
        except Exception:
            pass
        try:
            if self.stats_thread is not None and self.stats_thread.isRunning():
                self.stats_thread.quit()
                self.stats_thread.wait(1000)
        except Exception:
            pass
        self.stats_thread = None
        self.stats_worker = None
        self.start_button.setEnabled(False)
        self.start_button.setText('⏳ Ανάλυση σε εξέλιξη...')
        bar = '═' * 72
        sub_bar = '─' * 72
        self.update_log(format_html_log(bar, Colors.CYAN, bold=True))
        self.update_log(format_html_log('🚀 Εκκίνηση ανάλυσης ανίχνευσης dataset...', Colors.CYAN, bold=True))
        self.update_log(format_html_log(sub_bar, Colors.CYAN, bold=False))
        for w in (self.model_type_combo, self.model_combo, self.dataset_combo, self.num_detections_spin):
            try:
                w.setEnabled(False)
            except Exception:
                pass
        try:
            main_window = self.window()
            if hasattr(main_window, 'copilot_detection_button'):
                main_window.copilot_detection_button.setEnabled(False)
                main_window.copilot_detection_button.setText('⏳ Περιμένουμε αποτελέσματα ανίχνευσης...')
        except Exception:
            pass
        self.stats_thread = QThread()
        self.stats_thread.setObjectName('MMProStatsThread')
        self.stats_worker = StatisticsWorker(model_path=model_path, model_type_str=model_type, dataset_path=dataset_path, dataset_name=dataset_name, max_images=int(self.num_detections_spin.value()))
        self.stats_worker.moveToThread(self.stats_thread)
        self.stats_thread.started.connect(self.stats_worker.run)
        self.stats_worker.finished.connect(self.on_analysis_finished)
        self.stats_worker.finished.connect(self.stats_thread.quit)
        self.stats_worker.finished.connect(self.stats_worker.deleteLater)
        self.stats_thread.finished.connect(self.stats_thread.deleteLater)
        self.stats_worker.log.connect(self.update_log)
        self.stats_worker.summary.connect(self.update_summary)
        self.stats_worker.error.connect(self.on_error)
        self.stats_worker.progress.connect(self.progress_bar.setValue)
        self.stats_worker.report_ready.connect(self.on_report_ready)
        try:
            self.stats_worker.preview_sample.connect(self.on_preview_sample)
        except Exception:
            pass
        self.stats_thread.start()

    def on_preview_sample(self, image_path: str, title: str, pred_text: str, truth_text: str) -> None:
        try:
            p = Path(image_path)
        except Exception:
            p = None
        if not hasattr(self, '_preview_samples'):
            self._preview_samples = []
        if not hasattr(self, '_preview_paths_to_cleanup'):
            self._preview_paths_to_cleanup = []
        self._preview_samples.append((image_path, title, pred_text, truth_text))
        if p is not None:
            self._preview_paths_to_cleanup.append(p)
        if not hasattr(self, 'preview_dialog') or self.preview_dialog is None:
            try:
                self.preview_dialog = DetectionPreviewDialog(self)
                self.preview_dialog.set_on_close_callback(self._on_preview_dialog_closed)
                self.preview_dialog.resize(800, 600)
            except Exception:
                self.preview_dialog = None
        if self.preview_dialog is not None:
            if not self.preview_dialog.isVisible():
                self.preview_dialog.show()
            self.preview_dialog.add_sample(image_path, title, pred_text, truth_text)

    def _on_preview_dialog_closed(self) -> None:
        self.cleanup_preview_files()
        self.preview_dialog = None

    def cleanup_preview_files(self) -> None:
        try:
            for p in getattr(self, '_preview_paths_to_cleanup', []):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        self._preview_paths_to_cleanup = []
        self._preview_samples = []
        try:
            if DETECTION_PREVIEW_DIR.exists():
                best_effort_rmtree(DETECTION_PREVIEW_DIR)
        except Exception:
            pass

    def update_log(self, html_text: str) -> None:
        try:
            self.log_output.append(html_text)
            self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        except Exception:
            pass

    def update_summary(self, summary_text: str) -> None:
        try:
            html_text = format_html_summary(summary_text or '')
            self.summary_output.setHtml(html_text)
        except Exception:
            self.summary_output.setPlainText(summary_text or '')

    def on_error(self, error_text: str) -> None:
        self.update_log(format_html_log(error_text, Colors.RED, bold=True))
        QMessageBox.critical(self, 'Σφάλμα Ανάλυσης', error_text)

    def on_report_ready(self, path_str: str) -> None:
        self.last_report_path = path_str
        self.view_report_button.setEnabled(True)
        self.update_log(format_html_log(f'Η αναφορά PDF είναι έτοιμη: {Path(path_str).name}', Colors.GREEN, bold=True))

    def view_report(self) -> None:
        if self.last_report_path and Path(self.last_report_path).exists():
            if not open_file_externally(self.last_report_path):
                QMessageBox.warning(self, 'Σφάλμα', f'Δεν ήταν δυνατό το άνοιγμα του αρχείου: {self.last_report_path}')
        else:
            QMessageBox.warning(self, 'Σφάλμα', 'Το αρχείο αναφοράς δεν βρέθηκε ή δεν έχει δημιουργηθεί ακόμα.')

    def on_analysis_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.start_button.setText('📊 Έναρξη Ανάλυσης (εικόνες dataset)')
        self.update_log(format_html_log('--- Η ανάλυση ολοκληρώθηκε ---', Colors.GREEN, bold=True))
        try:
            th = self.stats_thread
            if th is not None:
                th.finished.connect(lambda: setattr(self, 'stats_thread', None) or setattr(self, 'stats_worker', None), Qt.ConnectionType.SingleShotConnection)
            else:
                self.stats_thread = None
                self.stats_worker = None
        except Exception:
            self.stats_thread = None
            self.stats_worker = None
        for w in (self.model_type_combo, self.model_combo, self.dataset_combo, self.num_detections_spin):
            try:
                w.setEnabled(True)
            except Exception:
                pass
        try:
            main_window = self.window()
            if hasattr(main_window, 'copilot_detection_button'):
                main_window.copilot_detection_button.setText('🧪 Βελτίωση με βάση τα αποτελέσματα ανίχνευσης')
                try:
                    has_key = has_valid_groq_api_key()
                except Exception:
                    has_key = False
                main_window.copilot_detection_button.setEnabled(has_key)
        except Exception:
            pass
        try:
            if hasattr(self, 'analysis_completed'):
                self.analysis_completed.emit()
        except Exception:
            pass


class StatisticsTabWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('📊 Στατιστικά Ανίχνευσης - Standalone')
        self.setCentralWidget(StatisticsTab())
        apply_light_theme_to_window(self)
        self.resize(1400, 900)


def statistics_tab_dev_main() -> None:
    app = QApplication(sys.argv)
    win = StatisticsTabWindow()
    win.show()
    sys.exit(app.exec())
"""App main window.
Κεντρικό παράθυρο της εφαρμογής (tabs, theme, helpers) και glue κώδικας για navigation.
"""


# ════════════════════════════════════════════════════════════════════════════════
# YOLOProManager – Κεντρικό παράθυρο εφαρμογής (QMainWindow)
# ════════════════════════════════════════════════════════════════════════════════
# Ενσωματώνει όλα τα tabs:
#   🏠 Home            – Dashboard και πλοήγηση
#   🎓 Εκπαίδευση      – TrainingTab (YOLO + CNN)
#   🤖 Copilot         – TrainingCopilotTab (AI προτάσεις)
#   🏋️ Benchmark       – BenchmarkTab (offline inference speed)
#   📷 Live Detection  – CameraTab (real-time κάμερα)
#   🎥 Video Inference – VideoTab (inference σε video αρχείο)
#   📊 Στατιστικά      – StatisticsTab (ανάλυση dataset)
#   📸 Benchmark Κάμ.  – CameraBenchmarkTab (FPS benchmark με κάμερα)
#
# Χαρακτηριστικά:
#   - Dark/Light theme toggle
#   - Global job manager (αποτρέπει ταυτόχρονες εκπαιδεύσεις/exports)
#   - Crash log notifications
#   - Sync επιλεγμένου μοντέλου μεταξύ tabs
# ════════════════════════════════════════════════════════════════════════════════
class YOLOProManager(QMainWindow):
    """
    Κύριο παράθυρο της εφαρμογής (QMainWindow).

    Φιλοξενεί tabs: Home, Training, Export, Camera,
    Benchmark, Statistics, AI Copilot.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            warmup_torch_cuda("YOLOProManager_init")
        except Exception:
            pass
        self.setWindowTitle('🤖  Models Manager Pro (A.I Copilot) Ver 4.0')
        self.setWindowIcon(QIcon(resource_path('app_icon.png')))
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_width = min(1760, avail.width())
            target_height = min(980, avail.height())
            min_width = min(1366, avail.width())
            min_height = min(768, avail.height())
            self.setMinimumSize(min_width, min_height)
            self.resize(target_width, target_height)
        else:
            self.setGeometry(100, 100, 1760, 980)
            self.setMinimumSize(1366, 768)
        self._base_size = self.size()
        app = QApplication.instance()
        if app is not None:
            base_font = app.font()
        else:
            base_font = QFont()
        self._base_font_point_size = base_font.pointSizeF() or float(base_font.pointSize() or 10)
        self._last_scale_factor = 1.0
        QApplication.setStyle('Fusion')
        self.current_theme = 'light'
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.home_tab = HomeTab()
        self.training_tab = TrainingTab()
        self.camera_tab = CameraTab()
        self.benchmark_tab = BenchmarkTab()
        self.camera_benchmark_tab = CameraBenchmarkTab()
        self.statistics_tab = StatisticsTab()
        self.video_tab = VideoTab()
        self.training_copilot_tab = TrainingCopilotTab(self.training_tab, self.statistics_tab)
        self.copilot_detection_button = self.training_copilot_tab.copilot_detection_button
        self.tabs.addTab(self._wrap_tab_with_scroll(self.home_tab), '🏠 Πίνακας Ελέγχου')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.training_tab), '🎓 Εκπαίδευση Μοντέλου')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.training_copilot_tab), '🤖 Α.Ι Copilot Εκπαίδευσης')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.camera_tab), '📷 Live Ανίχνευση')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.video_tab), '🎬 Video Inference')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.benchmark_tab), '⚡ Benchmark FPS')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.camera_benchmark_tab), '🎥 Benchmark Κάμερας')
        self.tabs.addTab(self._wrap_tab_with_scroll(self.statistics_tab), '📊 Στατιστικά Ανίχνευσης')
        try:
            sigs = _mmpro_get_signals()
            if sigs is not None:
                sigs.crash_log_created.connect(self._on_crash_log_created)
        except Exception:
            pass
        self.apply_theme('light')
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.home_tab.request_tab_change.connect(self.tabs.setCurrentIndex)
        QTimer.singleShot(0, self.center_on_screen)
        self.update_theme_button_icons()

    def _wrap_tab_with_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        return scroll

    def on_tab_changed(self, index: int) -> None:
        if index == 3:
            camera_running = False
            try:
                thread = getattr(self.camera_tab, 'camera_thread', None)
                camera_running = thread is not None and thread.isRunning()
            except Exception:
                pass
            try:
                self.camera_tab.refresh_models()
            except Exception:
                pass
            if not camera_running:
                try:
                    cb = getattr(self.camera_tab, 'select_all_checkbox', None)
                    if cb is not None:
                        cb.setChecked(True)
                        self.camera_tab.toggle_all_classes(True)
                except Exception:
                    pass
        elif index == 4:
            try:
                self.benchmark_tab.refresh_models()
            except Exception:
                pass
        elif index == 5:
            try:
                self.camera_benchmark_tab.refresh_models()
            except Exception:
                pass
        elif index == 6:
            try:
                self.statistics_tab.refresh_data()
            except Exception:
                pass

    def sync_selected_trained_model(self, model_path: str, model_type: str) -> None:
        self.current_trained_model_path = model_path
        self.current_trained_model_type = model_type
        try:
            p = Path(model_path)
            self.current_trained_model_name = p.name
            self.current_trained_model_stem = p.stem
        except Exception:
            self.current_trained_model_name = str(model_path)
            self.current_trained_model_stem = str(model_path)
        for tab_attr in ("camera_tab", "benchmark_tab", "camera_benchmark_tab", "statistics_tab", "video_tab"):
            tab = getattr(self, tab_attr, None)
            if tab is None:
                continue
            refresh_fn = getattr(tab, "refresh_models", None) or getattr(tab, "refresh_data", None)
            if refresh_fn is not None:
                try:
                    refresh_fn()
                except Exception as e:
                    _MMPRO_LOGGER.debug("sync_selected_trained_model %s error: %s", tab_attr, e)

    def center_on_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        geo = self.frameGeometry()
        new_width = min(geo.width(), avail.width())
        new_height = min(geo.height(), avail.height())
        if new_width != geo.width() or new_height != geo.height():
            self.resize(new_width, new_height)
            geo = self.frameGeometry()
        center_point = avail.center()
        geo.moveCenter(center_point)
        if geo.top() < avail.top():
            geo.moveTop(avail.top())
        self.move(geo.topLeft())

    def toggle_theme_global(self) -> None:
        from PySide6.QtWidgets import QTextEdit
        old_colors = {
            'HEADER': Colors.HEADER,
            'BLUE': Colors.BLUE,
            'CYAN': Colors.CYAN,
            'GREEN': Colors.GREEN,
            'YELLOW': Colors.YELLOW,
            'MAGENTA': Colors.MAGENTA,
            'RED': Colors.RED,
            'LIGHT': Colors.LIGHT,
        }
        current = getattr(self, 'current_theme', 'light')
        new_mode = 'dark' if (current or 'light') == 'light' else 'light'
        self.current_theme = new_mode
        self.apply_theme(new_mode)
        for tab_attr in ('home_tab', 'training_tab', 'training_copilot_tab', 'camera_tab', 'video_tab', 'benchmark_tab', 'camera_benchmark_tab', 'statistics_tab',):
            tab = getattr(self, tab_attr, None)
            if tab is not None:
                try:
                    tab.current_theme = new_mode
                except Exception:
                    pass
        new_colors = {
            'HEADER': Colors.HEADER,
            'BLUE': Colors.BLUE,
            'CYAN': Colors.CYAN,
            'GREEN': Colors.GREEN,
            'YELLOW': Colors.YELLOW,
            'MAGENTA': Colors.MAGENTA,
            'RED': Colors.RED,
            'LIGHT': Colors.LIGHT,
        }
        try:
            from PySide6.QtWidgets import QApplication, QPlainTextEdit
        except Exception:
            QApplication = None
            QPlainTextEdit = None
        roots = [self]
        try:
            app = QApplication.instance() if QApplication is not None else None
            if app is not None:
                for w in app.topLevelWidgets():
                    if w is not None and w not in roots:
                        roots.append(w)
        except Exception:
            pass
        seen = set()
        for root in roots:
            if root is None:
                continue
            try:
                widgets = []
                widgets.extend(root.findChildren(QTextEdit))
                if QPlainTextEdit is not None:
                    widgets.extend(root.findChildren(QPlainTextEdit))
            except Exception:
                continue
            for w in widgets:
                try:
                    if id(w) in seen:
                        continue
                    seen.add(id(w))
                    name = (w.objectName() or '')
                except Exception:
                    name = ''
                try:
                    is_ro = w.isReadOnly()
                except Exception:
                    is_ro = False
                if not (is_ro or ('Log' in name) or ('Output' in name) or ('Console' in name)):
                    continue
                if isinstance(w, QTextEdit):
                    try:
                        html = w.toHtml()
                    except Exception:
                        continue
                    for key in old_colors:
                        try:
                            html = html.replace(old_colors[key], new_colors[key])
                        except Exception:
                            pass
                    try:
                        w.setHtml(html)
                    except Exception:
                        pass
        try:
            self.update_theme_button_icons()
        except Exception:
            pass

    def create_theme_icon(self, mode: str, size: int = 20):
        try:
            from PySide6.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QIcon
            from PySide6.QtCore import Qt
        except Exception:
            try:
                from PySide6.QtGui import QIcon
                return QIcon()
            except Exception:
                return None
        m = (mode or 'light').strip().lower()
        size = int(size) if int(size) > 8 else 20
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if m == 'dark':
            bulb_color = QColor('#EAEAEA')
            base_color = QColor('#BDBDBD')
            outline = QColor('#111111')
            glow = QColor(255, 255, 190, 90)
        else:
            bulb_color = QColor('#2F2F2F')
            base_color = QColor('#5A5A5A')
            outline = QColor('#222222')
            glow = QColor(255, 220, 120, 70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(2, 1, size - 4, size - 6)
        painter.setBrush(QBrush(bulb_color))
        painter.setPen(QPen(outline))
        painter.drawEllipse(5, 3, size - 10, size - 10)
        painter.setBrush(QBrush(base_color))
        painter.setPen(QPen(outline))
        painter.drawRoundedRect(int(size * 0.35), int(size * 0.65), int(size * 0.30), int(size * 0.22), 2, 2)
        pen = QPen(outline)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(size * 0.35), int(size * 0.75), int(size * 0.65), int(size * 0.75))
        painter.drawLine(int(size * 0.35), int(size * 0.82), int(size * 0.65), int(size * 0.82))
        painter.end()
        return QIcon(pixmap)

    def update_theme_button_icons(self) -> None:
        icon = self.create_theme_icon(self.current_theme)
        icon_size = QSize(20, 20)
        for tab in (getattr(self, 'home_tab', None), getattr(self, 'training_tab', None), getattr(self, 'training_copilot_tab', None), getattr(self, 'camera_tab', None), getattr(self, 'benchmark_tab', None), getattr(self, 'camera_benchmark_tab', None), getattr(self, 'statistics_tab', None)):
            if tab is not None and hasattr(tab, 'theme_button'):
                tab.theme_button.setIcon(icon)
                tab.theme_button.setIconSize(icon_size)

    def apply_theme(self, mode: str) -> None:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication, QTextEdit, QPlainTextEdit
        mode = (mode or '').lower()
        if mode == 'dark':
            self.current_theme = 'dark'
            self.setStyleSheet(WINDOWS_11_DARK_STYLE)
            try:
                app = QApplication.instance()
                if app is not None:
                    for w in app.topLevelWidgets():
                        try:
                            w.setStyleSheet(WINDOWS_11_DARK_STYLE)
                        except Exception:
                            pass
            except Exception:
                pass
            Colors.HEADER = '#ffe6ff'
            Colors.BLUE = '#99ccff'
            Colors.CYAN = '#99ffff'
            Colors.GREEN = '#ccff99'
            Colors.YELLOW = '#ffefa3'
            Colors.MAGENTA = '#ffb3dd'
            Colors.RED = '#ff9999'
            Colors.LIGHT = '#ffffff'
        else:
            self.current_theme = 'light'
            self.setStyleSheet(WINDOWS_11_LIGHT_STYLE)
            try:
                app = QApplication.instance()
                if app is not None:
                    for w in app.topLevelWidgets():
                        try:
                            w.setStyleSheet(WINDOWS_11_LIGHT_STYLE)
                        except Exception:
                            pass
            except Exception:
                pass
            Colors.HEADER = '#000080'
            Colors.BLUE = '#003399'
            Colors.CYAN = '#005f73'
            Colors.GREEN = '#004225'
            Colors.YELLOW = '#7a5c00'
            Colors.MAGENTA = '#7b004b'
            Colors.RED = '#8b0000'
            Colors.LIGHT = '#000000'
        self.console_font = QFont('Consolas')
        self.console_font.setStyleHint(QFont.StyleHint.Monospace)
        self._console_base_point_size = 11
        self.console_font.setPointSize(self._console_base_point_size)
        try:
            roots = [self]
            app = QApplication.instance()
            if app is not None:
                for w in app.topLevelWidgets():
                    if w is not None and w not in roots:
                        roots.append(w)
        except Exception:
            roots = [self]
        seen = set()
        for root in roots:
            try:
                widgets = []
                widgets.extend(root.findChildren(QTextEdit))
                widgets.extend(root.findChildren(QPlainTextEdit))
            except Exception:
                continue
            for w in widgets:
                try:
                    if id(w) in seen:
                        continue
                    seen.add(id(w))
                    name = (w.objectName() or '')
                except Exception:
                    name = ''
                try:
                    is_ro = w.isReadOnly()
                except Exception:
                    is_ro = False
                if not (is_ro or ('Log' in name) or ('Output' in name) or ('Console' in name)):
                    continue
                try:
                    w.setFont(self.console_font)
                except Exception:
                    pass

    def update_ui_scaling(self) -> None:
        from PySide6.QtWidgets import QApplication, QTextEdit, QPlainTextEdit
        try:
            app = QApplication.instance()
            if app is None:
                return
            if not hasattr(self, '_base_size') or not hasattr(self, '_base_font_point_size'):
                return
            current_size = self.size()
            if not current_size.isValid():
                return
            base_w = max(self._base_size.width(), 1)
            base_h = max(self._base_size.height(), 1)
            scale_w = current_size.width() / base_w
            scale_h = current_size.height() / base_h
            scale = min(max(min(scale_w, scale_h), 0.8), 1.6)
            last = getattr(self, '_last_scale_factor', 1.0)
            if abs(last - scale) < 0.05:
                return
            self._last_scale_factor = scale
            font = app.font()
            font.setPointSizeF(self._base_font_point_size * scale)
            app.setFont(font)
            if hasattr(self, 'console_font') and hasattr(self, '_console_base_point_size'):
                console_font = QFont(self.console_font)
                console_font.setPointSizeF(self._console_base_point_size * scale)
                try:
                    roots = [self]
                    app2 = QApplication.instance()
                    if app2 is not None:
                        for w in app2.topLevelWidgets():
                            if w is not None and w not in roots:
                                roots.append(w)
                except Exception:
                    roots = [self]
                seen = set()
                for root in roots:
                    try:
                        widgets = []
                        widgets.extend(root.findChildren(QTextEdit))
                        widgets.extend(root.findChildren(QPlainTextEdit))
                    except Exception:
                        continue
                    for w in widgets:
                        try:
                            if id(w) in seen:
                                continue
                            seen.add(id(w))
                            name = (w.objectName() or '')
                        except Exception:
                            name = ''
                        try:
                            is_ro = w.isReadOnly()
                        except Exception:
                            is_ro = False
                        if not (is_ro or ('Log' in name) or ('Output' in name) or ('Console' in name)):
                            continue
                        try:
                            w.setFont(console_font)
                        except Exception:
                            pass
                self.console_font = console_font
        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_ui_scaling()

    def closeEvent(self, event) -> None:
        from PySide6.QtCore import QThread

        def _safe_stop_thread(thread: QThread | None, timeout_ms: int=1000) -> None:
            if not thread:
                return
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(timeout_ms)
            except RuntimeError:
                pass
            except Exception:
                pass

        def _safe_stop_worker(worker) -> None:
            try:
                if worker is not None and hasattr(worker, 'stop'):
                    worker.stop()
            except RuntimeError:
                pass
            except Exception:
                pass
        # Save persistent settings before close
        try:
            _settings().save()
        except Exception:
            pass
        if hasattr(self, 'camera_tab') and self.camera_tab:
            try:
                if getattr(self.camera_tab, 'camera_worker', None):
                    try:
                        self.camera_tab.stop_camera()
                    except Exception:
                        pass
                thread = getattr(self.camera_tab, 'camera_thread', None)
                _safe_stop_thread(thread, 1000)
            except Exception:
                pass
        camera_bench = getattr(self, 'camera_benchmark_tab', None)
        if camera_bench:
            worker = getattr(camera_bench, 'benchmark_worker', None)
            _safe_stop_worker(worker)
            thread = getattr(camera_bench, 'benchmark_thread', None)
            _safe_stop_thread(thread, 1000)
        stats_tab = getattr(self, 'statistics_tab', None)
        if stats_tab:
            worker = getattr(stats_tab, 'stats_worker', None)
            _safe_stop_worker(worker)
            thread = getattr(stats_tab, 'stats_thread', None)
            _safe_stop_thread(thread, 1000)
        train_tab = getattr(self, 'training_tab', None)
        if train_tab:
            try:
                if hasattr(train_tab, 'progress_poll_timer') and train_tab.progress_poll_timer.isActive():
                    train_tab.progress_poll_timer.stop()
            except Exception:
                pass
            worker = getattr(train_tab, 'training_worker', None)
            _safe_stop_worker(worker)
            thread = getattr(train_tab, 'training_thread', None)
            _safe_stop_thread(thread, 2000)
        bench_tab = getattr(self, 'benchmark_tab', None)
        if bench_tab:
            worker = getattr(bench_tab, 'benchmark_worker', None)
            _safe_stop_worker(worker)
            thread = getattr(bench_tab, 'benchmark_thread', None)
            _safe_stop_thread(thread, 1000)
        copilot_tab = getattr(self, 'training_copilot_tab', None)
        if copilot_tab:
            thread = getattr(copilot_tab, 'copilot_thread', None)
            _safe_stop_thread(thread, 1000)
        if hasattr(self, 'home_tab') and self.home_tab:
            try:
                if hasattr(self.home_tab, 'resource_timer') and self.home_tab.resource_timer:
                    self.home_tab.resource_timer.stop()
            except Exception:
                pass
        event.accept()

    def _broadcast_html_to_logs(self, html: str) -> None:
        if not html:
            return
        _TAB_ATTRS = ( "home_tab", "training_tab", "training_copilot_tab", "camera_tab", "benchmark_tab", "camera_benchmark_tab", "statistics_tab",)
        for name in _TAB_ATTRS:
            tab = getattr(self, name, None)
            if tab is None:
                continue
            for method_name in ("update_log", "append_log"):
                method = getattr(tab, method_name, None)
                if method is not None:
                    try:
                        method(html)
                        break
                    except Exception:
                        continue
            else:
                log_edit = getattr(tab, "log_edit", None)
                if log_edit is not None:
                    try:
                        log_edit.append(html)
                    except Exception:
                        pass

    def _on_crash_log_created(self, path_str: str) -> None:
        p = str(path_str or "").strip()
        if not p:
            return
        shown = getattr(self, "_shown_crash_paths", None)
        if shown is None:
            shown = set()
            self._shown_crash_paths = shown
        if p in shown:
            return
        shown.add(p)
        try:
            html = format_html_log(f"🧾 Crash log δημιουργήθηκε: {p}", Colors.RED, bold=True)
            self._broadcast_html_to_logs(html)
        except Exception:
            pass
        show_dialog = _env_bool("MM_PRO_SHOW_CRASH_DIALOG", True)
        if not show_dialog:
            return
        try:
            from PySide6.QtWidgets import QMessageBox, QApplication
            mb = QMessageBox(self)
            mb.setIcon(QMessageBox.Icon.Critical)
            mb.setWindowTitle("⚠️ Crash Log Δημιουργήθηκε")
            mb.setText("Η εφαρμογή κατέγραψε ένα κρίσιμο συμβάν σε crash log.")
            mb.setInformativeText(p)
            btn_open = mb.addButton("📂 Άνοιγμα φακέλου", QMessageBox.ButtonRole.ActionRole)
            btn_copy = mb.addButton("📋 Αντιγραφή path",  QMessageBox.ButtonRole.ActionRole)
            mb.addButton(QMessageBox.StandardButton.Ok)
            mb.exec()
            clicked = mb.clickedButton()
            if clicked == btn_open:
                try:
                    open_folder_externally(str(Path(p).resolve().parent))
                except Exception:
                    pass
            elif clicked == btn_copy:
                try:
                    QApplication.clipboard().setText(p)
                except Exception:
                    pass
        except Exception as e:
            _MMPRO_LOGGER.debug("_on_crash_log_created dialog error: %s", e)


def create_light_palette() -> "QPalette":
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f1f5f9"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#7c3aed"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def qt_global_excepthook(exctype: type, value: BaseException, tb) -> None:
    try:
        msg = "".join(traceback.format_exception(exctype, value, tb))
    except Exception:
        msg = f"{exctype.__name__}: {value}"
    _MMPRO_LOGGER.critical("Uncaught exception:\n%s", msg)
    try:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "⚠️ Απροσδόκητο Σφάλμα", msg[:2000])
    except Exception:
        pass
    print(msg, file=sys.stderr)


def _run_worker_mode(mode: str, job_file: str) -> int:
    try:
        mode = (mode or "").strip().lower()
        job_path = Path(job_file).resolve()
        if not job_path.is_file():
            print_line("__MM_ERR__", f"Job file not found: {job_path}")
            return 2
        sys.argv = [sys.argv[0], str(job_path)]
        if mode in ("train", "training"):
            return training_runner_main()
        if mode in ("export", "exports"):
            return export_runner_main()
        if mode in ("bench", "benchmark", "benchmarking"):
            return benchmark_runner_main()
        print_line("__MM_ERR__", f"Unknown worker mode: {mode!r}")
        return 2
    except Exception as e:
        err_tb = traceback.format_exc()
        print_line("__MM_EXCEPTION__", f"Worker exception: {e}")
        print_line("__MM_EXCEPTION__", err_tb)
        return 5


def _parse_worker_args(argv: list[str]) -> tuple[str | None, str | None]:
    mode: str | None = None
    job_file: str | None = None
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if not a:
            continue
        if a.startswith("--mmpro-mode="):
            mode = a.split("=", 1)[1].strip().lower()
        elif a in ("--mmpro-mode", "--mmpro_mode", "--mmpromode") and i < len(args):
            mode = args[i].strip().lower()
            i += 1
        elif a.startswith("--mmpro-job="):
            job_file = a.split("=", 1)[1].strip()
        elif a.endswith(".json"):
            candidate = Path(a)
            if candidate.is_file():
                job_file = str(candidate)
    return mode, job_file


# ═══════════════════════════════════════════════════════════════════════
# Ενότητα 14 – Entry point: εκκίνηση εφαρμογής
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Κεντρικό entry point της εφαρμογής.

    – Αν παρασχεθούν --mmpro-mode / --mmpro-job: εκτελεί worker mode.
    – Αλλιώς: εκκινεί το PySide6 GUI (QApplication + YOLOProManager).
    """
    try:
        freeze_support()
    except Exception:
        pass
    try:
        _mmpro_install_global_exception_hooks()
    except Exception:
        pass
    mode, job_file = _parse_worker_args(sys.argv)
    if mode and job_file:
        return _run_worker_mode(mode, job_file)
    if getattr(sys, "frozen", False):
        try:
            os.chdir(Path(sys.executable).resolve().parent)
        except Exception as e:
            _MMPRO_LOGGER.warning("Cannot chdir to exe directory: %s", e)
    try:
        perform_smart_memory_cleanup()
    except Exception as e:
        _MMPRO_LOGGER.debug("Pre-GUI memory cleanup error: %s", e)
    try:
        warmup_torch_cuda("main_startup")
    except Exception as e:
        _MMPRO_LOGGER.debug("CUDA warmup skipped: %s", e)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtGui import QPixmap, QPainter, QColor as _QColor, QFont as _QFont
    from PySide6.QtCore import Qt as _Qt
    app = QApplication(sys.argv)
    try:
        app.setPalette(create_light_palette())
    except Exception as e:
        _MMPRO_LOGGER.debug("setPalette error: %s", e)
    # ── Splash screen ──────────────────────────────────────────────────────
    splash = None
    try:
        splash_pix = QPixmap(600, 320)
        splash_pix.fill(_QColor('#0d1117'))
        p = QPainter(splash_pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Background gradient feel
        p.fillRect(0, 0, 600, 6, _QColor('#238636'))
        # Title
        f_title = _QFont('Segoe UI', 26, _QFont.Weight.Bold)
        p.setFont(f_title)
        p.setPen(_QColor('#e6edf3'))
        p.drawText(40, 100, '🤖  Models Manager Pro')
        f_sub = _QFont('Segoe UI', 13)
        p.setFont(f_sub)
        p.setPen(_QColor('#8b949e'))
        p.drawText(40, 140, 'A.I Copilot Edition  ·  Ver 4.0')
        p.setPen(_QColor('#238636'))
        p.drawText(40, 180, 'Φόρτωση...')
        f_tiny = _QFont('Segoe UI', 9)
        p.setFont(f_tiny)
        p.setPen(_QColor('#484f58'))
        p.drawText(40, 300, 'YOLO + CNN (MobileNet/ResNet)  ·  Training  ·  Export  ·  Live Detection  ·  AI Copilot')
        p.end()
        splash = QSplashScreen(splash_pix, _Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
    except Exception as e:
        _MMPRO_LOGGER.debug("Splash screen error: %s", e)
        splash = None
    win = YOLOProManager()
    if splash is not None:
        try:
            splash.finish(win)
        except Exception:
            pass
    win.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
