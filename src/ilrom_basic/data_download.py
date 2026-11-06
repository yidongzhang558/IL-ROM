"""Download the Case 1 temperature fields the first time they are needed."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

T_URL = "https://github.com/yidongzhang558/IL-ROM/releases/download/v1.0/T.npy"
T_BYTES = 655360128
T_SHA256 = "ffc177347795169f7a840a39b8554231617b27a084299f0939c75865abcddf22"
CHUNK = 1024 * 1024


def ensure_temperature_file(directory):
    """Return the path of T.npy, downloading and checking it if it is missing."""
    path = Path(directory) / "T.npy"
    if path.is_file():
        return path
    part = path.with_name("T.npy.part")
    print(f"T.npy not found. Downloading 655 MB from {T_URL}", flush=True)
    digest = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(T_URL, timeout=60) as response, part.open("wb") as stream:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                done += len(block)
                if done % (64 * CHUNK) < len(block):
                    print(f"  {done // CHUNK} / {T_BYTES // CHUNK} MB", flush=True)
        if done != T_BYTES or digest.hexdigest() != T_SHA256:
            raise RuntimeError("the file failed the size or SHA-256 check")
        part.replace(path)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download T.npy: {exc}. Download it from {T_URL} and place it in {path.parent}.") from exc
    print("T.npy downloaded and checked.", flush=True)
    return path
