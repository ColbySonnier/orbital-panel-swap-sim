"""Download the textures listed in models/assets_manifest.yaml.

    python -m models.fetch_assets

Each asset lands in models/downloads/<dest>/ (git-ignored). An asset whose
`check` file already exists is skipped, so re-running is cheap.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "assets_manifest.yaml"
DL = HERE / "downloads"


def load_manifest(path=MANIFEST):
    return yaml.safe_load(Path(path).read_text())["assets"]


def fetch(asset, root=DL):
    dest = root / asset["dest"]
    if (dest / asset["check"]).exists():
        print(f"skip  {asset['name']} (already in {dest})")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"get   {asset['name']} ({asset['size_mb']} MB, {asset['license']}) from {asset['url']}")
    req = urllib.request.Request(asset["url"], headers={"User-Agent": "asset-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            name = Path(member).name  # flatten; never write outside dest
            if name and name.lower().endswith((".jpg", ".png")):
                (dest / name).write_bytes(z.read(member))
    if not (dest / asset["check"]).exists():
        raise RuntimeError(f"{asset['name']}: {asset['check']} not found in the download")
    return True


def main():
    for a in load_manifest():
        fetch(a)


if __name__ == "__main__":
    main()
