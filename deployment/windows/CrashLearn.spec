from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH).parents[1]
source = root / "src"
onnx_data, onnx_binaries, onnx_hidden = collect_all("onnxruntime")
analysis = Analysis(
    [str(Path(SPECPATH) / "launcher.py")],
    pathex=[str(source)],
    binaries=onnx_binaries,
    datas=onnx_data + [(str(source / "crashlearn_sim/resources"), "crashlearn_sim/resources")]
          + [(str(path), "crashlearn_sim/simulation")
             for path in (source / "crashlearn_sim/simulation").glob("*.py")]
          + [(str(path), "f110_gym/envs")
             for path in (source / "f110_gym/envs").glob("*.py")],
    hiddenimports=onnx_hidden + ["tkinter", "tkinter.filedialog", "tkinter.colorchooser",
                                "onnxruntime", "numpy", "stable_baselines3", "glcontext"],
    excludes=["IPython", "pytest", "tensorboard", "torch.utils.tensorboard"],
    runtime_hooks=[str(Path(SPECPATH) / "freeze_runtime.py")],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="CrashLearn",
          debug=False, strip=False, upx=False, console=False)
sb3_data = [(str(Path(destination) / Path(source).name), source, "DATA")
            for source, destination in collect_data_files("stable_baselines3")]
coll = COLLECT(exe, analysis.binaries, analysis.datas, sb3_data,
               strip=False, upx=False, name="CrashLearn")
