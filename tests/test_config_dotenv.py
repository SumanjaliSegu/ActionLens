"""The .env loader (uses temporary files only; never reads the project's real .env)."""
import os

from robotrace.config import Settings, load_dotenv


def test_loads_values_and_returns_names_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOTRACE_SKIP_DOTENV", raising=False)
    for k in ("OPENCODE_GO_API_KEY", "RT_QUOTED", "RT_EXPORTED"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\n\nOPENCODE_GO_API_KEY=sk-test-value\nRT_QUOTED=\"a b\"\nexport RT_EXPORTED='x'\n"
                   "EMPTY=\nnot a line\n", encoding="utf-8")
    names = load_dotenv(env)
    assert names == ["OPENCODE_GO_API_KEY", "RT_QUOTED", "RT_EXPORTED"]
    assert "sk-test-value" not in repr(names) and "sk-test-value" not in repr(Settings())
    assert Settings().has_opencode_go_key and os.environ["RT_QUOTED"] == "a b" and os.environ["RT_EXPORTED"] == "x"
    for k in names:
        monkeypatch.delenv(k)


def test_existing_environment_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOTRACE_SKIP_DOTENV", raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("OPENCODE_GO_API_KEY=from-file\n", encoding="utf-8")
    assert load_dotenv(env) == [] and os.environ["OPENCODE_GO_API_KEY"] == "from-shell"


def test_skip_flag_and_missing_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("RT_SKIPPED=1\n", encoding="utf-8")
    monkeypatch.setenv("ROBOTRACE_SKIP_DOTENV", "1")
    assert load_dotenv(env) == [] and "RT_SKIPPED" not in os.environ
    monkeypatch.delenv("ROBOTRACE_SKIP_DOTENV")
    assert load_dotenv(tmp_path / "missing.env") == []
