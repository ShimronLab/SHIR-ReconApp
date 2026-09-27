# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\saard\\OneDrive - Technion\\תואר שני\\MRI_Acquisition_Reconstruction_Toolkit\\AnalyzeMRIDataApp\\Graphics', 'Graphics')]
binaries = []
hiddenimports = ['read_siemens_scans.twix_reading', 'PyQt5.sip', 'matplotlib.backends.backend_qt5agg']
tmp_ret = collect_all('read_siemens_scans')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numba')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('llvmlite')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('finufft')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mri_nufft')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sigpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pygrappa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('twixtools')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('skimage')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\saard\\OneDrive - Technion\\תואר שני\\MRI_Acquisition_Reconstruction_Toolkit\\AnalyzeMRIDataApp\\app.py'],
    pathex=['C:\\Users\\saard\\OneDrive - Technion\\תואר שני\\MRI_Acquisition_Reconstruction_Toolkit\\AnalyzeMRIDataApp', 'C:\\Users\\saard\\OneDrive - Technion\\תואר שני\\MRI_Acquisition_Reconstruction_Toolkit\\read_siemens_scans'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'notebook', 'jupyterlab', 'jupyterlab_server', 'jupyter_server', 'IPython', 'ipykernel', 'ipython', 'nbconvert', 'nbformat', 'nbclient', 'pytest', 'tkinter', 'torch', 'ggrappa'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SHIR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\saard\\OneDrive - Technion\\תואר שני\\MRI_Acquisition_Reconstruction_Toolkit\\AnalyzeMRIDataApp\\Graphics\\logo.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SHIR',
)
