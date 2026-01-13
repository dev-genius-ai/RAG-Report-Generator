"""Lightweight tests for the setup verification script."""

from unittest.mock import Mock, patch

from scripts import verify_setup


def test_check_environment_flags_missing_key(capsys):
    """check_environment should report missing API key."""
    with patch("scripts.verify_setup.get_settings") as mock_get_settings:
        settings = Mock()
        settings.openai_api_key = "your_openai_api_key_here"
        settings.openai_model = "gpt-4-turbo-preview"
        settings.openai_embedding_model = "text-embedding-3-large"
        settings.environment = "test"
        mock_get_settings.return_value = settings

        ok = verify_setup.check_environment()

    out = capsys.readouterr().out
    assert "OPENAI_API_KEY not configured" in out
    assert ok is False


def test_check_dependencies_handles_missing_package(capsys, monkeypatch):
    """check_dependencies should surface missing packages cleanly."""

    def fake_import(name):
        if name == "langchain":
            raise ImportError()
        return None

    monkeypatch.setattr("scripts.verify_setup.__import__", fake_import)

    ok = verify_setup.check_dependencies()

    out = capsys.readouterr().out
    assert "Missing packages" in out
    assert ok is False

