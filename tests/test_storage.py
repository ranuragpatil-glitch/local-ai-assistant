import pytest

from local_ai_assistant import cache, memory, notes


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVKIT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NOTES_DIR", str(tmp_path / "notes"))
    (tmp_path / "notes" / "sub").mkdir(parents=True)
    (tmp_path / "notes" / "a.md").write_text("hello\nfastmcp rocks\n")
    (tmp_path / "notes" / "sub" / "b.txt").write_text("deep python note\n")
    (tmp_path / "notes" / ".hidden.md").write_text("python secret\n")
    (tmp_path / "secret.txt").write_text("top secret")


def test_notes_list_read_search():
    assert notes.list_files() == ["a.md", "sub/b.txt"]
    assert "fastmcp" in notes.read_file("a.md")
    assert notes.search("python") == ["[sub/b.txt] line 1: deep python note"]


def test_notes_cannot_escape_folder():
    for bad in ("../secret.txt", "sub/../../secret.txt"):
        with pytest.raises(ValueError):
            notes.read_file(bad)


def test_notes_empty_env_uses_default_and_creates_it(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DIR", "")          # what an unfilled .env line looks like
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setattr("local_ai_assistant.config.Path.home", lambda: tmp_path / "fakehome")
    assert notes.root() == (tmp_path / "fakehome" / "notes").resolve()
    assert (tmp_path / "fakehome" / "notes" / "welcome.md").exists()


def test_notes_wrong_explicit_folder_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DIR", str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="NOTES_DIR"):
        notes.root()


def test_notes_status_warns_for_project_folder(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text("x")
    monkeypatch.setenv("NOTES_DIR", str(project))
    _, warning = notes.status()
    assert warning and "project" in warning


def test_memory_roundtrip():
    mid = memory.save("User likes chai", "pref")
    assert memory.search("chai")[0][0] == mid
    assert memory.search("chai nope") == []
    assert memory.delete(mid) and not memory.delete(mid)


def test_cache_ttl():
    cache.put("k", "v", 60)
    assert cache.get("k") == "v"
    cache.put("old", "v", -1)
    assert cache.get("old") is None
