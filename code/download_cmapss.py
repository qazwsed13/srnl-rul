# -*- coding: utf-8 -*-
"""Download NASA archive, handle nested zip, extract the 12 C-MAPSS files."""
import hashlib, io, json, os, ssl, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "CMAPSSData")
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
FILES = ["train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
         "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
         "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"]
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
os.makedirs(DEST, exist_ok=True)

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
    data = r.read()
print("bytes", len(data))
manifest = {}


def extract_from(zf, label):
    for name in zf.namelist():
        fn = os.path.basename(name)
        if fn in FILES:
            b = zf.read(name)
            with open(os.path.join(DEST, fn), "wb") as f:
                f.write(b)
            manifest[fn] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "member": name, "container": label}
            print("OK", fn, len(b))


with zipfile.ZipFile(io.BytesIO(data)) as zf:
    print("members:", [(n, zf.getinfo(n).file_size) for n in zf.namelist()])
    extract_from(zf, "outer")
    if not manifest:
        for name in zf.namelist():
            if name.lower().endswith(".zip"):
                inner = zf.read(name)
                print("nested", name, len(inner))
                with zipfile.ZipFile(io.BytesIO(inner)) as z2:
                    print("inner members:", len(z2.namelist()))
                    extract_from(z2, name)

with io.open(os.path.join(DEST, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"source_url": URL, "files": manifest}, f, ensure_ascii=False, indent=1)
missing = sorted(set(FILES) - set(manifest))
print("done", len(manifest), "/", len(FILES), "missing", missing)
