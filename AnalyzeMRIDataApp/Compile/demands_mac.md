# SHIR — Requirements & Setup for Users (macOS)

This document explains what you need on your **Mac** to run **SHIR**
and how to set it up. Follow it once; afterwards you just open the app.

> Using Windows instead? See [`demands.md`](demands.md).

---

## 1. What you get

You receive a zip containing **`SHIR.app`**. Unzip it, drag the `.app`
to `/Applications` (or anywhere you like), and double-click to run.

The app is **fully self-contained** — it reconstructs everything out of the box
using a built-in (sigpy) engine. **BART is optional**: install it only if you
want BART's exact reconstruction numerics (Sections 4–5). Unlike Windows, BART
runs **natively** on macOS — no WSL or bridge is involved.

---

## 2. Requirements at a glance

| Requirement | Needed for | Required? |
|-------------|-----------|-----------|
| macOS (Apple Silicon or Intel) | running the app | **Yes** |
| A build matching your chip (arm64 vs Intel) | running the app | **Yes** — see §3 |
| **BART** installed natively | *exact BART numerics only* | **Optional** |

**Out of the box (no BART):** every reconstruction works, computed by the
built-in **sigpy** engine (ESPIRiT + SENSE, CS-SENSE, ESPIRiT ±L1/L2/TV, NUFFT
band/Poisson, GRAPPA, RSS). Results are very close to BART's but not identical.

**With BART installed:** the app uses BART for those methods (its exact
numerics), and automatically falls back to sigpy if BART isn't found.

### Choosing the engine (in the app)
The **"Recon engine"** dropdown (top of the parameters panel):
- **Auto** *(default)* — use BART if installed, else sigpy.
- **Force BART** — always use BART (needs BART on the PATH).
- **Force sigpy** — always use the built-in engine (no BART needed).

> The `ggrappa` reconstruction method is **not available** in this build.

---

## 3. First launch — Apple Silicon vs Intel, and Gatekeeper

**Right chip:** an app built for Apple Silicon (arm64) will not run on an Intel
Mac and vice-versa. Make sure you were given the build for your Mac. (Apple menu
→ About This Mac → "Chip"/"Processor".)

**Gatekeeper:** because the app isn't signed with an Apple Developer ID, the
first launch on another Mac may say *"SHIR can't be opened because it
is from an unidentified developer"* or *"is damaged"*. Fix it once:

- **Right-click** the app → **Open** → **Open** in the dialog, **or**
- run this in Terminal from the folder holding the app:
  ```bash
  xattr -dr com.apple.quarantine SHIR.app
  ```

After the first successful open, it launches normally by double-click.

---

## 4. (Optional) Install BART natively — only for exact BART numerics

Skip this and Section 5 if you're happy with the built-in sigpy engine.

BART builds natively on macOS. In Terminal:

```bash
# 1. Install build dependencies (Homebrew — https://brew.sh)
brew install fftw libpng openblas

# 2. Download and build BART
cd ~
git clone https://github.com/mrirecon/bart.git
cd bart
make

# 3. Put it on your PATH (choose one):
sudo make install                      # installs `bart` to /usr/local/bin
#   — or add this line to ~/.zshrc:
#   export PATH="$HOME/bart:$PATH"

# 4. Verify
bart version
```

`bart version` should print a version number. The app (in Auto or Force BART
mode) will then use it automatically.

---

## 5. If BART is not on the PATH: point the app at it

If `bart` lives somewhere not on the PATH (e.g. `~/bart/bart`), tell the app its
absolute path with an environment variable. The simplest persistent way:

```bash
# add to ~/.zshrc (Apple's default shell), then restart the app from a new shell
echo 'export MRI_BART_BIN="$HOME/bart/bart"' >> ~/.zshrc
```

Note: GUI apps launched from Finder don't always inherit `~/.zshrc`. If Force
BART can't find BART when launched from Finder, either use `sudo make install`
(Section 4, puts `bart` in `/usr/local/bin`, which is found), or launch the app
from Terminal:
```bash
open SHIR.app
```

---

## 6. Using the app

1. Open `SHIR.app`.
2. Load your Siemens TWIX (`.dat`) data, pick a reconstruction and the **Recon
   engine**, and run.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "can't be opened / from an unidentified developer" / "is damaged" | Gatekeeper quarantine | Right-click → Open, or `xattr -dr com.apple.quarantine SHIR.app` (§3). |
| "app can't be opened" / crashes immediately | Wrong architecture (Intel vs Apple Silicon) | Get the build matching your chip (§3). |
| Recon slower than expected | Using the sigpy engine (CPU, iterative) | Normal — install BART and use Auto/Force BART, or lower iterations. |
| Force BART can't find `bart` | BART not on PATH (esp. launched from Finder) | `sudo make install` (§4) or set `MRI_BART_BIN` (§5). |
| Results differ slightly from a colleague's | Different engine (sigpy vs BART) | Expected — match the "Recon engine" setting. |
| Need to see internal errors | — | Check the log: in Terminal run `cat "$TMPDIR/SHIR.log"` (or `/tmp/SHIR.log`). |

---

## 8. Summary checklist

- [ ] macOS build matches your chip (Apple Silicon vs Intel)
- [ ] First launch cleared past Gatekeeper (right-click → Open)
- [ ] That's it — the app reconstructs everything with the built-in sigpy engine.

Optional, only if you want BART's exact numerics:
- [ ] `brew install fftw libpng openblas`, build BART, `sudo make install`
- [ ] `bart version` prints a version
- [ ] (Only if needed) `MRI_BART_BIN` set to BART's absolute path
