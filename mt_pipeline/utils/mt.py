# utils/mt.py
import subprocess, pathlib, re, os

def _get_devices() -> list[str]:
    if hasattr(ct2, "available_devices"):
        return ct2.available_devices()
    if hasattr(ct2, "list_devices"):
        return ct2.list_devices()
    if hasattr(ct2, "get_supported_devices"):
        return ct2.get_supported_devices()
    if hasattr(ct2, "Device") and hasattr(ct2.Device, "list_devices"):
        return ct2.Device.list_devices()
    # --- nouveau fallback ---
    so = pathlib.Path(ct2.__file__).with_suffix(".so")
    try:
        out = subprocess.check_output(["ldd", so], text=True)
        if re.search(r"libcu(dart|blas).so", out):
            return ["cpu", "cuda"]
    except Exception:
        pass
    return ["cpu"]
