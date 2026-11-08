"""Dataset registry, download with checksum verification, licence note and local cache.

Nothing here runs at import time and nothing assumes network access: downloads happen only
when :func:`fetch` is called (CLI ``celldx data fetch <name>``).

Cache directory: ``$CELLDX_DATA`` or ``~/.cache/celldx``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RemoteFile:
    url: str
    filename: str
    sha256: str | None  # None = not yet known: recorded on first fetch (trust on first use, flagged)
    bytes: int | None = None


@dataclass
class DatasetInfo:
    name: str
    title: str
    landing_page: str
    licence: str
    citation: str
    status: str  # tested_against_real_files | untested_against_real_files
    files: list[RemoteFile] = field(default_factory=list)
    doi: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REGISTRY: dict[str, DatasetInfo] = {
    "nasa": DatasetInfo(
        name="nasa",
        title="NASA Ames PCoE Li-ion Battery Aging Datasets (Battery Data Set 5: B0005 to B0056)",
        landing_page="https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
        licence="U.S. Government work (NASA open data; no copyright in the U.S.). Cite: B. Saha and K. Goebel (2007), "
                "'Battery Data Set', NASA Prognostics Data Repository, NASA Ames Research Center.",
        citation="Saha, B. and Goebel, K. (2007). Battery Data Set. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.",
        status="tested_against_real_files",
        files=[RemoteFile(url="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
                          filename="5. Battery Data Set.zip",
                          sha256="82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e", bytes=209708670)],
        notes=[
            "Outer zip holds six inner zips; .mat files B0005-B0056 (B0025-B0028 appear in two inner zips, identical bytes).",
            "Impedance (EIS) cycles for every cell are logged at the cell's test ambient: 24 C for most cells, 43 C for B0029-B0032;"
            " cells cycled at 4 C (B0041-B0056) still have EIS at 24 C.",
            "Re/Rct contain corrupted values in some late cycles (e.g. B0050, B0052: |Re| ~ 1e14); validation masks them.",
        ],
    ),
    "calce": DatasetInfo(
        name="calce",
        title="CALCE Battery Research Group data (University of Maryland): CS2 / CX2 prismatic LCO cells",
        landing_page="https://calce.umd.edu/battery-data",
        licence="CALCE terms of use: free for research with citation of the CALCE publications listed on the page "
                "(redistribution not permitted).",
        citation="He, W., Williard, N., Osterman, M., Pecht, M. (2011). Prognostics of lithium-ion batteries based on "
                 "Dempster-Shafer theory and the Bayesian Monte Carlo method. J. Power Sources 196, 10314-10321.",
        status="untested_against_real_files",
        files=[],
        notes=["Host not reachable from the build sandbox (proxy 403); adapter written against the documented Arbin "
               "export layout (sheet 'Channel_*' with Test_Time(s), Current(A), Voltage(V), Internal_Resistance(Ohm), "
               "Discharge_Capacity(Ah), Cycle_Index). Files must be downloaded manually from the landing page."],
    ),
    "oxford": DatasetInfo(
        name="oxford",
        title="Oxford Battery Degradation Dataset 1 (8 Kokam 740 mAh pouch cells, 40 C, Artemis drive cycle)",
        landing_page="https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac",
        doi="10.5287/bodleian:KO2kdmYGg",
        licence="CC BY 4.0 (University of Oxford Research Archive).",
        citation="Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford.",
        status="untested_against_real_files",
        files=[],
        notes=["Host not reachable from the build sandbox (proxy 403); adapter written against the documented layout: "
               "Oxford_Battery_Degradation_Dataset_1.mat with Cell1..Cell8, each cycNNNN with C1ch/C1dc/OCVch/OCVdc "
               "structs holding t, v, q, T."],
    ),
    "matr": DatasetInfo(
        name="matr",
        title="MIT-Stanford-Toyota fast-charging dataset (Severson et al. 2019, A123 APR18650M1A LFP cells)",
        landing_page="https://data.matr.io/1/",
        doi="10.1038/s41560-019-0356-8",
        licence="CC BY 4.0 per data.matr.io (cite Severson et al., Nature Energy 4, 383-391, 2019).",
        citation="Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 4, 383-391.",
        status="untested_against_real_files",
        files=[],
        notes=["Host not reachable from the build sandbox (proxy 403); adapter written against the documented batch "
               ".mat (HDF5) layout: batch/summary with IR, QD, Tavg, cycle; per-cycle I, V, T, Qc, Qd, t."],
    ),
}


def cache_dir() -> Path:
    p = Path(os.environ.get("CELLDX_DATA", Path.home() / ".cache" / "celldx"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ChecksumError(RuntimeError):
    pass


def _extract_all(zpath: Path, dest: Path) -> list[Path]:
    """Extract a zip and any zips inside it (NASA ships zips of zips)."""
    out: list[Path] = []
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
        names = [dest / n for n in z.namelist()]
    for n in names:
        if n.suffix.lower() == ".zip" and n.exists():
            sub = n.with_suffix("")
            out.extend(_extract_all(n, sub))
        elif n.is_file():
            out.append(n)
    return out


def fetch(name: str, dest: Path | None = None, *, from_file: Path | None = None) -> dict[str, Any]:
    """Download (or copy ``from_file``), verify SHA-256, extract, and write a manifest with the
    checksum of every extracted file. Returns the manifest."""
    info = REGISTRY[name]
    root = (dest or cache_dir()) / name
    root.mkdir(parents=True, exist_ok=True)
    if not info.files and from_file is None:
        raise RuntimeError(f"{name}: no automatic download ({info.status}). Download manually from "
                           f"{info.landing_page} and pass --from-file. Licence: {info.licence}")
    manifest: dict[str, Any] = {"dataset": name, "title": info.title, "licence": info.licence,
                                "landing_page": info.landing_page, "files": [], "extracted": []}
    remotes = info.files or [RemoteFile(url=str(from_file), filename=Path(str(from_file)).name, sha256=None)]
    for rf in remotes:
        target = root / rf.filename
        if from_file is not None:
            shutil.copyfile(from_file, target)
        elif not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(rf.url, timeout=120) as r, open(tmp, "wb") as fh:  # noqa: S310
                shutil.copyfileobj(r, fh)
            tmp.rename(target)
        digest = sha256_file(target)
        if rf.sha256 and digest != rf.sha256:
            raise ChecksumError(f"{target.name}: sha256 {digest} != expected {rf.sha256}")
        manifest["files"].append({"name": rf.filename, "sha256": digest, "verified": bool(rf.sha256),
                                  "bytes": target.stat().st_size})
        if target.suffix.lower() == ".zip":
            for f in _extract_all(target, root / "extracted"):
                manifest["extracted"].append({"path": str(f.relative_to(root)), "sha256": sha256_file(f)})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(name: str, dest: Path | None = None) -> dict[str, Any] | None:
    p = (dest or cache_dir()) / name / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
