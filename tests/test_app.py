from llama4codex import app as app_module


def test_bind_config_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("L4C_HOST", raising=False)
    monkeypatch.delenv("L4C_PORT", raising=False)

    assert app_module.bind_config() == ("0.0.0.0", 8081)


def test_bind_config_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("L4C_HOST", "127.0.0.1")
    monkeypatch.setenv("L4C_PORT", "18080")

    assert app_module.bind_config() == ("127.0.0.1", 18080)
