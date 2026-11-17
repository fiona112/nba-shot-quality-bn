"""Small provenance helpers for reproducible local experiments."""

import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    """Fingerprint file bytes without loading a whole dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root):
    """Allow exported source trees and machines without Git to run normally."""
    unavailable = {"commit": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        # An enclosing course repository is not this project's revision.
        if len(revision) != 2 or Path(revision[0]).resolve() != Path(repo_root).resolve():
            return unavailable
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return unavailable
    return {"commit": revision[1], "dirty": bool(status.strip())}


def collect_provenance():
    """Record actual versions and code fingerprints, including uncommitted code."""
    source_files = sorted((REPO_ROOT / "src").rglob("*.py"))
    source_files.append(REPO_ROOT / "requirements.txt")
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "git": git_metadata(REPO_ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "source_files_sha256": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in source_files
        },
    }
