"""Check where the resume state goes and which copy a new chunk picks up.

These are the failures that cost a whole session rather than a step, and none of them are
visible until the next chunk starts, so they are worth testing without waiting eleven hours
to observe one.
"""

from pathlib import Path
from types import SimpleNamespace

from kabr import hf_sync, remote_sync, wandb_sync


def fake_cfg(tmp: Path, **kw):
    cfg = SimpleNamespace(run_name="run-x", run_dir=tmp / "run-x", resume="auto",
                          ckpt_remote="", wandb_mode="online", wandb_project="p")
    cfg.__dict__.update(kw)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_backends_parse_in_order():
    assert remote_sync.backends(SimpleNamespace(ckpt_remote="hf,wandb")) == ["hf", "wandb"]
    assert remote_sync.backends(SimpleNamespace(ckpt_remote=" wandb , hf ")) == ["wandb", "hf"]
    assert remote_sync.backends(SimpleNamespace(ckpt_remote="")) == []
    print("[ok] backends parse in the order given")


def test_typo_is_rejected_not_ignored():
    """A misspelled backend must not parse as "push nowhere".

    That failure trains for a full chunk and looks exactly like success until the next one
    starts from step 0.
    """
    try:
        remote_sync.backends(SimpleNamespace(ckpt_remote="hf,wanb"))
    except SystemExit as exc:
        assert "wanb" in str(exc), exc
        print("[ok] a misspelled backend raises instead of silently pushing nowhere")
        return
    raise AssertionError("typo was accepted")


def test_resume_prefers_first_backend(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hf_sync, "pull_chunk", lambda c: calls.append("hf") or Path("/hf.pt"))
    monkeypatch.setattr(wandb_sync, "pull_checkpoint",
                        lambda c: calls.append("wandb") or Path("/wandb.pt"))
    cfg = fake_cfg(tmp_path, ckpt_remote="hf,wandb")
    assert remote_sync.resolve_resume(cfg) == "/hf.pt"
    assert calls == ["hf"], f"second backend should not be consulted, got {calls}"
    print("[ok] resume takes the first backend that has something")


def test_resume_falls_through_to_the_second(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_sync, "pull_chunk", lambda c: None)
    monkeypatch.setattr(wandb_sync, "pull_checkpoint", lambda c: Path("/wandb.pt"))
    cfg = fake_cfg(tmp_path, ckpt_remote="hf,wandb")
    assert remote_sync.resolve_resume(cfg) == "/wandb.pt"
    print("[ok] an empty first backend falls through to the second")


def test_a_raising_backend_does_not_stop_the_next(tmp_path, monkeypatch):
    """A hub outage must degrade to the other copy, not to starting over."""
    def boom(cfg):
        raise RuntimeError("503")

    monkeypatch.setattr(hf_sync, "pull_chunk", boom)
    monkeypatch.setattr(wandb_sync, "pull_checkpoint", lambda c: Path("/wandb.pt"))
    cfg = fake_cfg(tmp_path, ckpt_remote="hf,wandb")
    assert remote_sync.resolve_resume(cfg) == "/wandb.pt"
    print("[ok] a backend that raises falls through instead of restarting the run")


def test_resume_falls_back_to_local_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_sync, "pull_chunk", lambda c: None)
    cfg = fake_cfg(tmp_path, ckpt_remote="hf")
    (cfg.run_dir / "ckpt-40.pt").write_bytes(b"x")
    (cfg.run_dir / "ckpt-200.pt").write_bytes(b"x")
    assert remote_sync.resolve_resume(cfg).endswith("ckpt-200.pt")
    print("[ok] with no remote copy it resumes from the newest local checkpoint")


def test_local_scan_ignores_the_remote_filename(tmp_path, monkeypatch):
    """`ckpt-latest.pt` must not be parsed as a step number.

    The local scan sorts by `int(name.split('-')[-1])`, so a file named for a word rather
    than a step would raise there if the glob ever matched it.
    """
    monkeypatch.setattr(hf_sync, "pull_chunk", lambda c: None)
    cfg = fake_cfg(tmp_path, ckpt_remote="hf")
    (cfg.run_dir / hf_sync.LATEST).write_bytes(b"x")
    (cfg.run_dir / "ckpt-7.pt").write_bytes(b"x")
    assert wandb_sync.latest_local(cfg.run_dir).name == "ckpt-7.pt"
    print("[ok] the local scan skips ckpt-latest.pt instead of choking on it")


def test_explicit_resume_path_skips_the_remotes(tmp_path, monkeypatch):
    def boom(cfg):
        raise AssertionError("should not have been consulted")

    monkeypatch.setattr(hf_sync, "pull_chunk", boom)
    cfg = fake_cfg(tmp_path, ckpt_remote="hf", resume="/somewhere/ckpt-9.pt")
    assert remote_sync.resolve_resume(cfg) == "/somewhere/ckpt-9.pt"
    print("[ok] an explicit --resume path is taken as given")


def test_push_tries_every_backend_even_after_one_fails(tmp_path, monkeypatch):
    """Writing to two places is pointless if the first failure skips the second."""
    seen = []

    def boom(cfg, ckpt, step, meta):
        seen.append("hf")
        raise RuntimeError("403")

    monkeypatch.setattr(hf_sync, "push_chunk", boom)
    monkeypatch.setattr(wandb_sync, "push_checkpoint",
                        lambda c, r, k, s, m: seen.append("wandb"))
    cfg = fake_cfg(tmp_path, ckpt_remote="hf,wandb")
    remote_sync.push_checkpoint(cfg, None, tmp_path / "c.pt", 5, {})
    assert seen == ["hf", "wandb"], seen
    print("[ok] a failed push does not skip the other backend")


def test_chunk_files_carry_the_best_record(tmp_path):
    """A resume without the best-*.pt record overwrites a better checkpoint on first eval."""
    run = tmp_path / "run-x"
    run.mkdir()
    for name in ("ckpt-9.pt", "best-kvd_val.pt", "config.json", "wandb_id.txt", "noise.log"):
        (run / name).write_bytes(b"x")
    got = {p.name for p in hf_sync._chunk_files(run, run / "ckpt-9.pt")}
    assert got == {"ckpt-9.pt", "best-kvd_val.pt", "config.json", "wandb_id.txt"}, got
    print("[ok] a pushed chunk carries the best record and the wandb id, not stray files")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
