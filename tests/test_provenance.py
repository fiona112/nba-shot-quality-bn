from types import SimpleNamespace

import pytest

from src import provenance


def test_file_fingerprint_changes_when_input_bytes_change(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_bytes(b"abc")
    assert provenance.file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    path.write_bytes(b"abcd")
    assert provenance.file_sha256(path) != (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_git_metadata_tolerates_missing_git(tmp_path, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", missing_git)
    assert provenance.git_metadata(tmp_path) == {"commit": None, "dirty": None}


@pytest.mark.parametrize("status, dirty", [("", False), (" M src/preprocess.py\n", True)])
def test_git_metadata_records_revision_and_worktree_state(tmp_path, monkeypatch, status, dirty):
    def fake_git(command, **kwargs):
        assert command[:3] == ["git", "-C", str(tmp_path)]
        if "rev-parse" in command:
            return SimpleNamespace(stdout=f"{tmp_path}\n{'a' * 40}\n")
        return SimpleNamespace(stdout=status)

    monkeypatch.setattr(provenance.subprocess, "run", fake_git)
    assert provenance.git_metadata(tmp_path) == {"commit": "a" * 40, "dirty": dirty}


def test_exported_source_does_not_inherit_an_enclosing_git_revision(tmp_path, monkeypatch):
    def enclosing_git(*args, **kwargs):
        return SimpleNamespace(stdout=f"{tmp_path.parent}\n{'b' * 40}\n")

    monkeypatch.setattr(provenance.subprocess, "run", enclosing_git)
    assert provenance.git_metadata(tmp_path) == {"commit": None, "dirty": None}
