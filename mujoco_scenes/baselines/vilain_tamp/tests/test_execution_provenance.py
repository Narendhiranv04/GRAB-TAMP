"""What may and may not stop a ViLaIn run.

The guard used to reject a changed HEAD, a branch other than a configured name,
any tracked change, and any unexpected untracked file -- and it ran only on
episodes where planning had SUCCEEDED, so it destroyed exactly the episodes
worth keeping and left the failures looking like data.  An untracked results
table cost 6 episodes; a branch rename would have cost a 120-episode leg at the
same commit with byte-identical files.

What survives is what can actually make a recorded artifact dishonest: the
commit moved, tracked files changed, or a hashed input changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.artifacts import (
    verify_artifact_manifest,
    verify_clean_worktree,
    verify_repository_provenance,
)


def _provenance(**overrides):
    base = {
        "repository_root": "/repo",
        "head": "abc123",
        "branch": "baseline_executions",
        "tracked_changes": [],
        "untracked_paths": [],
        "dirty": False,
    }
    base.update(overrides)
    return base


def test_a_branch_rename_no_longer_vetoes():
    """Same commit, same contents, different label."""
    verify_repository_provenance(
        _provenance(), _provenance(branch="some_other_branch")
    )


def test_an_untracked_file_no_longer_vetoes():
    """A results table or a scratch note cannot change a PDDL plan."""
    verify_repository_provenance(
        _provenance(),
        _provenance(untracked_paths=["results_table.tex", "notes.md"], dirty=True),
    )


def test_a_moved_head_is_recorded_not_vetoed():
    """A commit landing mid-run used to void every episode in flight.

    It is a fact about the run, so it is recorded.  What binds a plan to the
    code that produced it is the artifact manifest, asserted below.
    """
    recorded = verify_repository_provenance(
        _provenance(), _provenance(head="def456")
    )
    assert recorded["head_at_start"] == "abc123"
    assert recorded["head_at_end"] == "def456"
    assert recorded["head_changed_during_run"] is True


def test_a_tracked_change_during_the_run_is_recorded_not_vetoed():
    recorded = verify_repository_provenance(
        _provenance(), _provenance(tracked_changes=[" M planner.py"])
    )
    assert recorded["tracked_changes_during_run"] == [" M planner.py"]


def test_a_dirty_worktree_may_start_and_says_so():
    recorded = verify_clean_worktree(_provenance(tracked_changes=[" M domain.pddl"]))
    assert recorded["clean_worktree"] is False
    assert recorded["tracked_changes"] == [" M domain.pddl"]


def test_a_clean_worktree_with_untracked_files_may_start():
    recorded = verify_clean_worktree(
        _provenance(untracked_paths=["scratch.txt"], dirty=True)
    )
    assert recorded["clean_worktree"] is True


def test_a_changed_hashed_input_still_vetoes(tmp_path: Path):
    """The guarantee that actually establishes 'this plan came from that domain'."""
    domain = tmp_path / "domain.pddl"
    domain.write_text("(define (domain kitchen))", encoding="utf-8")
    entry = {
        "path": "domain.pddl",
        "sha256": "0" * 64,  # not the file's hash
    }
    with pytest.raises(ValueError):
        verify_artifact_manifest([entry], root=tmp_path)
