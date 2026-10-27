import pandas as pd
import pytest
from src import cache


def test_cached_frame_calls_builder_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)
    calls = []

    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})

    first = cache.cached_frame("thing", builder)
    second = cache.cached_frame("thing", builder)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_cached_frame_survives_process_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)
    cache.cached_frame("x", lambda: pd.DataFrame({"a": [1]}))

    def explode():
        raise AssertionError("builder must not be called when cache exists")

    out = cache.cached_frame("x", explode)
    assert list(out["a"]) == [1]


def test_rate_limiter_waits_between_calls():
    clock = {"t": 0.0}
    slept = []
    limiter = cache.RateLimiter(
        min_interval=4.0,
        now=lambda: clock["t"],
        sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)),
    )
    limiter.wait()
    limiter.wait()
    assert slept == [4.0]


def test_rate_limiter_does_not_wait_if_enough_time_passed():
    clock = {"t": 0.0}
    slept = []
    limiter = cache.RateLimiter(
        min_interval=4.0,
        now=lambda: clock["t"],
        sleep=lambda s: slept.append(s),
    )
    limiter.wait()
    clock["t"] = 10.0
    limiter.wait()
    assert slept == []


def test_cached_frame_rebuilds_corrupted_cache(tmp_path, monkeypatch):
    """Verify corrupted parquet file is rebuilt, not read."""
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)
    cache_path = tmp_path / "test_key.parquet"

    # Write a corrupted parquet file
    cache_path.write_bytes(b"corrupted parquet data")

    calls = []
    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [42]})

    # Should rebuild from builder when parquet is corrupted
    result = cache.cached_frame("test_key", builder)
    assert list(result["a"]) == [42]
    assert len(calls) == 1
    # Verify cache was written correctly this time
    assert (tmp_path / "test_key.parquet").exists()


def test_no_tmp_files_after_successful_write(tmp_path, monkeypatch):
    """Verify no .tmp leftover files remain after successful cache writes."""
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)

    # Write frame
    cache.cached_frame("frame_key", lambda: pd.DataFrame({"a": [1]}))

    # Write text
    cache.cached_text("text_key", lambda: "<html></html>")

    # Check no .tmp files exist
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0, f"Found unexpected .tmp files: {tmp_files}"

    # Verify actual cache files exist
    assert (tmp_path / "frame_key.parquet").exists()
    assert (tmp_path / "text_key.html").exists()
