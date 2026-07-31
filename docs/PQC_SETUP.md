# Setting up liboqs / python-oqs (Priyadharshini + Poojasree — read this first)

`pip install liboqs-python` on its own will likely fail. The Python package
is a wrapper around `liboqs`, a C library, which has to be built first.

## Ubuntu / WSL / Linux lab machines
```bash
sudo apt update
sudo apt install -y cmake gcc ninja-build libssl-dev python3-dev

git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -GNinja ..
ninja
sudo ninja install

# tell the linker where liboqs.so ended up
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

pip install liboqs-python
```

## Windows
Building liboqs natively on Windows is painful (needs Visual Studio Build
Tools + a specific CMake generator). Two options, pick one before you burn
an evening on it:
1. Use **WSL2** (Windows Subsystem for Linux) and follow the Linux steps above.
2. Use **Docker** — build a container from the liboqs Dockerfile and develop inside it.

Do not attempt a bare Windows + PowerShell build unless you've already
confirmed Visual Studio Build Tools + CMake + Ninja are all installed and on PATH.

## Verify it worked
```bash
python -c "import oqs; print(oqs.get_enabled_sig_mechanisms())"
```
You should see `Dilithium2`, `Dilithium3`, `Dilithium5` in the printed list.
If this errors with `ImportError` or `OSError: liboqs.so not found`, the
build step above didn't complete — don't debug the Python side yet, fix the
build first.

## If you're stuck
Post the exact error in the team channel, not "it doesn't work." Include:
- OS (Windows/WSL/Linux/Mac)
- Which command failed
- Full error text
