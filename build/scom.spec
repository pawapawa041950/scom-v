# PyInstaller spec for scom — single-file (onefile) release build.
#
# The heavy ML stack (PyTorch, ComfyUI, the models) is downloaded at first run,
# so the exe bundles only Python + PySide6 + the app. Copying the single
# dist/scom.exe to a fresh machine is enough; on first launch it provisions the
# backend and downloads the chosen models next to the exe.
#
# Build from the repo root:
#   pyinstaller build/scom.spec --clean --noconfirm
# Output: dist/scom.exe
import os

# SPECPATH is the directory holding this spec (…/scom/build); its parent is the
# repo root. Using it keeps the build working regardless of the current dir.
ROOT = os.path.dirname(SPECPATH)

# Qt modules this QtWidgets-only app never touches. Excluding the big ones keeps
# the single exe smaller / faster to unpack. (Unknown names are simply ignored.)
_EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtQuick", "PySide6.QtQuick3D",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
    "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner", "PySide6.QtQuickWidgets", "PySide6.QtQuick3DRuntimeRender",
    "PySide6.QtSensors", "PySide6.QtBluetooth", "PySide6.QtPositioning",
    "PySide6.QtSerialPort", "PySide6.QtNfc", "tkinter",
]

a = Analysis(
    [os.path.join(ROOT, "scom.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[],
    hiddenimports=["websocket", "piexif", "PIL", "tomli"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="scom",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX can corrupt Qt DLLs; keep off for reliability
    runtime_tmpdir=None,
    console=False,        # GUI app: no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
