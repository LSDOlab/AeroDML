"""Validate release assets, or explicitly package them with --output-dir."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Create and extraction-test two ZIPs in this directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = [line.strip() for line in (root / "asset_paths.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    manifest = json.loads((root / "asset_manifest.json").read_text())
    records = {item["path"]: item for item in manifest["files"]}
    if len(names) != len(set(names)) or set(names) != set(records):
        raise ValueError("Asset path list and manifest disagree")
    total = 0
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe asset path: {name}")
        path = root / rel
        if any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file():
            raise ValueError(f"Missing asset or symlink: {name}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        size = path.stat().st_size
        if size != records[name]["bytes"] or h.hexdigest() != records[name]["sha256"]:
            raise ValueError(f"Asset differs from manifest: {name}")
        total += size
    if total != manifest["total_bytes"]:
        raise ValueError("Manifest total is incorrect")
    print(f"Validated {len(names)} assets: {total:,} bytes ({total / 1e9:.3f} GB)")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        groups = {
            "aero-dml-assets-mesh.zip": [n for n in names if n.startswith("cases/aircraft_BWB/model/mesh_639k/")],
            "aero-dml-assets-data.zip": [n for n in names if not n.startswith("cases/aircraft_BWB/model/mesh_639k/")],
        }
        outputs = [args.output_dir / name for name in groups]
        checksum_path = args.output_dir / "SHA256SUMS"
        if any(p.exists() for p in [*outputs, checksum_path]):
            raise FileExistsError("Output archives or SHA256SUMS already exist; use a new output directory")
        if shutil.which("unzip") is None:
            raise RuntimeError("unzip is required to test the documented extraction commands")
        for filename, members in groups.items():
            # Leave ample headroom below both 2 decimal GB and 2 GiB.
            if sum(records[n]["bytes"] for n in members) >= 1_990_000_000:
                raise ValueError(f"Asset group too large: {filename}; rebalance before packaging")
        checksums = []
        for output, members in zip(outputs, groups.values()):
            with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for name in members:
                    archive.write(root / name, name)
            size = output.stat().st_size
            if size >= 2_000_000_000:
                raise ValueError(f"Archive exceeds 2 GB: {output}")
            h = hashlib.sha256()
            with output.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
            checksums.append(f"{h.hexdigest()}  {output.name}")
            print(f"Created {output}: {size:,} bytes", flush=True)
        # Exercise the same unzip operations documented for users in a fresh directory.
        with tempfile.TemporaryDirectory(prefix="aerodml-extraction-") as directory:
            clean = Path(directory)
            for output in outputs:
                subprocess.run(["unzip", "-q", str(output.resolve()), "-d", str(clean)], check=True)
            actual = {p.relative_to(clean).as_posix() for p in clean.rglob("*") if p.is_file()}
            if actual != set(names):
                raise ValueError("Extracted inventory differs from asset manifest")
            for name in names:
                path = clean / name
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        h.update(block)
                if path.stat().st_size != records[name]["bytes"] or h.hexdigest() != records[name]["sha256"]:
                    raise ValueError(f"Extracted asset differs: {name}")
            print(f"Verified clean unzip and checksums for all {len(names)} assets", flush=True)
        checksum_path.write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
