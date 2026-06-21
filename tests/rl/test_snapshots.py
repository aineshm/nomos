"""Unit tests for the snapshot helpers in modal_train and export_snapshots.

Written FIRST (TDD / RED) before any implementation exists.  All tests are pure
and offline — no JAX, no filesystem, no network.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers from smoothride.rl.modal_train
# ---------------------------------------------------------------------------

class TestShouldSnapshot:
    """should_snapshot(it, snapshot_every, iters) -> bool"""

    def _fn(self, it: int, snapshot_every: int, iters: int) -> bool:
        from smoothride.rl.modal_train import should_snapshot
        return should_snapshot(it, snapshot_every, iters)

    # --- iter 0 is always True (baseline before meaningful training) ---
    def test_iter_zero_always_true(self):
        assert self._fn(0, 50, 300) is True

    def test_iter_zero_true_even_when_snapshot_disabled(self):
        """snapshot_every<=0 disables mid-run snapshots, but iter 0 is baseline."""
        # The spec says "only the final iter" when snapshot_every<=0.
        # iter 0 == final iter only if iters==1; otherwise iter 0 is NOT the final.
        # For iters=300, iter 0 is NOT the final, so snapshot_every<=0 → False at 0.
        # Clarification: the docstring says "if snapshot_every<=0, only final iter".
        # So iter 0 with snapshot_every=0 and iters=300 → False.
        assert self._fn(0, 0, 300) is False

    # --- multiples of snapshot_every are True ---
    def test_multiple_of_snapshot_every_is_true(self):
        assert self._fn(50, 50, 300) is True

    def test_another_multiple(self):
        assert self._fn(100, 50, 300) is True

    def test_non_multiple_is_false(self):
        assert self._fn(51, 50, 300) is False

    def test_non_multiple_is_false_2(self):
        assert self._fn(1, 50, 300) is False

    # --- final iter is always True ---
    def test_final_iter_always_true(self):
        assert self._fn(299, 50, 300) is True

    def test_final_iter_true_when_disabled(self):
        """snapshot_every<=0 → only final iter."""
        assert self._fn(299, 0, 300) is True

    def test_final_iter_true_negative_snapshot_every(self):
        assert self._fn(299, -1, 300) is True

    # --- snapshot_every<=0: mid-run iters (not 0, not final) are False ---
    def test_mid_run_false_when_disabled(self):
        assert self._fn(50, 0, 300) is False

    def test_mid_run_false_when_negative(self):
        assert self._fn(100, -1, 300) is False

    # --- edge: single-iter training (iters=1, it=0 is also final) ---
    def test_single_iter_is_both_zero_and_final(self):
        assert self._fn(0, 50, 1) is True

    def test_single_iter_disabled_still_true(self):
        """With iters=1 and snapshot_every=0, it==0 is the final iter → True."""
        assert self._fn(0, 0, 1) is True

    # --- snapshot_every=1 snapshots every single iter ---
    def test_every_iter_when_snapshot_every_one(self):
        for it in range(10):
            assert self._fn(it, 1, 10) is True, f"expected True at it={it}"


class TestSnapshotName:
    """snapshot_name(tag, it) -> str"""

    def _fn(self, tag: str, it: int) -> str:
        from smoothride.rl.modal_train import snapshot_name
        return snapshot_name(tag, it)

    def test_basic_formatting(self):
        assert self._fn("", 0) == "trained_it00000.msgpack"

    def test_with_tag(self):
        assert self._fn("_pedtest", 50) == "trained_pedtest_it00050.msgpack"

    def test_five_digit_zero_padded(self):
        assert self._fn("", 1) == "trained_it00001.msgpack"

    def test_large_iter_no_truncation(self):
        # iter 99999 should still fit in 5 digits
        assert self._fn("", 99999) == "trained_it99999.msgpack"

    def test_large_iter_over_five_digits(self):
        # iters beyond 99999 should expand naturally (no truncation)
        assert self._fn("", 100000) == "trained_it100000.msgpack"

    def test_tag_no_underscore(self):
        # tag is inserted verbatim; caller controls the underscore convention
        assert self._fn("foo", 5) == "trainedfoo_it00005.msgpack"


# ---------------------------------------------------------------------------
# Helpers from scripts.export_snapshots
# ---------------------------------------------------------------------------

class TestParseIter:
    """parse_iter(filename) -> int | None"""

    def _fn(self, filename: str):
        from scripts.export_snapshots import parse_iter
        return parse_iter(filename)

    def test_basic_match(self):
        assert self._fn("trained_pedtest_it00050.msgpack") == 50

    def test_zero_iter(self):
        assert self._fn("trained_it00000.msgpack") == 0

    def test_no_tag(self):
        assert self._fn("trained_it00100.msgpack") == 100

    def test_large_iter(self):
        assert self._fn("trained_myreg_it99999.msgpack") == 99999

    def test_no_match_trained_without_iter(self):
        assert self._fn("trained.msgpack") is None

    def test_no_match_untrained(self):
        assert self._fn("untrained.msgpack") is None

    def test_no_match_history(self):
        assert self._fn("history.json") is None

    def test_no_match_plain_msgpack(self):
        assert self._fn("something_else.msgpack") is None

    def test_path_with_dir_component(self):
        # parse_iter operates on filenames; a path with dirs should still work
        # (or return None — either is fine as long as it's consistent).
        # We test the basename convention the script uses.
        import os
        name = os.path.basename("runs/trained_test_it00010.msgpack")
        assert self._fn(name) == 10


class TestBuildManifest:
    """build_manifest(iters) -> dict"""

    def _fn(self, iters: list[int]) -> dict:
        from scripts.export_snapshots import build_manifest
        return build_manifest(iters)

    def test_structure_has_scenes_key(self):
        m = self._fn([0, 50, 100])
        assert "scenes" in m

    def test_scenes_count(self):
        m = self._fn([0, 50, 100])
        assert len(m["scenes"]) == 3

    def test_sorted_by_iter(self):
        m = self._fn([100, 0, 50])
        iters = [s["iter"] for s in m["scenes"]]
        assert iters == [0, 50, 100]

    def test_file_format(self):
        m = self._fn([50])
        assert m["scenes"][0]["file"] == "scene_it00050.json"

    def test_file_format_zero(self):
        m = self._fn([0])
        assert m["scenes"][0]["file"] == "scene_it00000.json"

    def test_label_baseline_at_zero(self):
        m = self._fn([0])
        assert m["scenes"][0]["label"] == "iter 0 (baseline)"

    def test_label_non_zero(self):
        m = self._fn([50])
        assert m["scenes"][0]["label"] == "iter 50"

    def test_label_non_zero_large(self):
        m = self._fn([299])
        assert m["scenes"][0]["label"] == "iter 299"

    def test_empty_iters_returns_empty_scenes(self):
        m = self._fn([])
        assert m["scenes"] == []

    def test_iter_field_is_int(self):
        m = self._fn([0, 50])
        for s in m["scenes"]:
            assert isinstance(s["iter"], int)

    def test_all_required_keys_present(self):
        m = self._fn([0, 50, 100])
        for s in m["scenes"]:
            assert {"iter", "file", "label"} == set(s.keys())
