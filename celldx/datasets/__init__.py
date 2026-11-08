"""Dataset adapters returning the common schema, plus the download registry.

Status per adapter (see docs/datasets.md):

* ``nasa``   - tested_against_real_files
* ``reon``   - tested against Firebreak's example telemetry (REON format)
* ``calce``, ``oxford``, ``matr`` - untested_against_real_files (hosts unreachable from the build sandbox)
"""

from .registry import REGISTRY, cache_dir, fetch, load_manifest, sha256_file

ADAPTER_STATUS = {
    "nasa": "tested_against_real_files",
    "reon": "tested_against_firebreak_example_files",
    "calce": "untested_against_real_files",
    "oxford": "untested_against_real_files",
    "matr": "untested_against_real_files",
}

__all__ = ["REGISTRY", "ADAPTER_STATUS", "cache_dir", "fetch", "load_manifest", "sha256_file"]
