# SHIR — Requirements & Setup for Users

This document explains what you need on your Windows PC to run **SHIR**
and how to set it up. Follow it once; afterwards you just double-click the app.

---

## 1. What you get

You receive a folder (usually a zip) named **`SHIR`** containing
`SHIR.exe` and its support files. **Unzip the whole folder** and run
`SHIR.exe` from inside it. Do **not** move the `.exe` out on its own —
it needs the files next to it.

The app is **fully self-contained** — it reconstructs everything out of the box
using a built-in (sigpy) engine. **BART is optional**: install it only if you
want BART's exact reconstruction numerics. Everything in Sections 3–6 below is
optional.

---

## 2. Requirements at a glance

| Requirement | Needed for | Required? |
|-------------|-----------|-----------|
| Windows 10 / 11, 64-bit | running the app | **Yes** |
| The unzipped `SHIR` folder | running the app | **Yes** |
| **WSL** (Windows Subsystem for Linux) | *exact BART numerics only* | **Optional** |
| **BART** installed inside WSL | *exact BART numerics only* | **Optional** |

**Out of the box (no WSL/BART):** every reconstruction works, computed by the
built-in **sigpy** engine (ESPIRiT + SENSE, CS-SENSE, ESPIRiT ±L1/L2/TV, NUFFT
band/Poisson, GRAPPA, RSS). Results are very close to BART's but not identical.

**With WSL + BART installed:** the app uses BART for those methods (its exact
numerics), and automatically falls back to sigpy if BART is ever unavailable.

### Choosing the engine (in the app)

The **"Recon engine"** dropdown (top of the parameters panel) controls this:
- **Auto** *(default)* — use BART if it's installed, otherwise sigpy.
- **Force BART** — always use BART (for exact numerics; needs WSL + BART).
- **Force sigpy** — always use the built-in engine (no WSL/BART needed).

> The `ggrappa` reconstruction method is **not available** in this build.

---

## 3. (Optional) Install WSL — only for exact BART numerics

Skip this and Sections 4–6 entirely if you're happy with the built-in sigpy
engine. Do them only if you specifically want BART.

1. Open **PowerShell as Administrator**.
2. Run:
   ```powershell
   wsl --install
   ```
   This installs WSL and an Ubuntu Linux distribution.
3. **Reboot** when prompted.
4. After reboot, Ubuntu finishes setup and asks you to create a **username and
   password** — remember these.
5. Verify it works — in a normal PowerShell:
   ```powershell
   wsl echo hello
   ```
   It should print `hello`.

Already have WSL? Make sure it's WSL 2:
```powershell
wsl -l -v
```

---

## 4. (Optional) Install BART inside WSL

BART has no Windows version, so it is installed inside Ubuntu (WSL). Open an
**Ubuntu / WSL terminal** (Start menu → "Ubuntu", or run `wsl` in PowerShell)
and run:

```bash
# 1. Install build tools and libraries BART needs
sudo apt update
sudo apt install -y make gcc git libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev bc

# 2. Download and build BART
cd ~
git clone https://github.com/mrirecon/bart.git
cd bart
make

# 3. Install it system-wide so the app finds it automatically
sudo make install

# 4. Verify
bart version
```

`bart version` should print a version number. Done.

> **Alternative (no system install):** if you skip `sudo make install`, BART
> stays at `~/bart/bart`. In that case you must tell the app where it is — see
> Section 6.

---

## 5. Verify BART is reachable from Windows

The app calls BART through WSL. Confirm this works from **PowerShell**:

```powershell
wsl bart version
```

- Prints a version → **you're done**, the app will find BART automatically.
- `command not found` → BART isn't on WSL's PATH; do the `sudo make install`
  step above, or use Section 6.

---

## 6. If BART is not on the PATH: point the app at it

If `wsl bart version` fails but `bart` exists somewhere (e.g. `~/bart/bart`),
tell the app its absolute WSL path with a Windows environment variable.

1. Find the path — in a WSL terminal:
   ```bash
   ls ~/bart/bart      # or: find $HOME -name bart -type f
   ```
2. Set the variable — in PowerShell (use your real username/path):
   ```powershell
   setx MRI_BART_BIN /home/<your-wsl-username>/bart/bart
   ```
3. **Open a new terminal / restart the app** (the variable only applies to
   newly started programs).

Optional related variables:
- `MRI_WSL_EXE` — set if your WSL launcher isn't the default `wsl.exe`.

---

## 7. Using the app

1. Unzip and open the `SHIR` folder.
2. Double-click **`SHIR.exe`**.
3. Load your Siemens TWIX (`.dat`) data, choose a reconstruction, and run.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| App won't start at all | `.exe` moved out of its folder | Keep `SHIR.exe` inside its unzipped folder. |
| Recon is slower than expected | Using the sigpy engine (CPU, iterative) | Normal — install BART and use "Force BART"/"Auto" for speed, or reduce iterations. |
| "could not launch BART via WSL" (only in **Force BART**) | WSL/BART not installed | Switch the engine to **Auto** or **Force sigpy**, or install BART (Sections 3–4). |
| `bart: command not found` (only when using BART) | BART not on WSL PATH | `sudo make install` (Section 4) or set `MRI_BART_BIN` (Section 6). |
| Results differ slightly from a colleague's | Different engine (sigpy vs BART) | Expected — pick the same "Recon engine" setting to match. |
| Need to see internal errors | — | Check the log file: in PowerShell run `type $env:TEMP\SHIR.log`. |

---

## 9. Summary checklist

- [ ] Windows 10/11 64-bit
- [ ] Unzipped `SHIR` folder (run the `.exe` from inside it)
- [ ] That's it — the app reconstructs everything with the built-in sigpy engine.

Optional, only if you want BART's exact numerics:
- [ ] `wsl --install` done + rebooted
- [ ] BART built and `sudo make install`ed in WSL
- [ ] `wsl bart version` prints a version
- [ ] (Only if needed) `MRI_BART_BIN` set to BART's absolute WSL path
