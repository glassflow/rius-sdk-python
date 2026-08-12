import pytest

from rius.config import DEFAULT_ENDPOINT, GlassflowConfig, resolve_config

_SUFFIXES = [
    "ENDPOINT",
    "API_KEY",
    "SERVICE_NAME",
    "DISABLED",
    "SAMPLE_RATE",
    "CAPTURE_CONTENT",
    "HEARTBEAT",
    "HEARTBEAT_INTERVAL",
    "AGENT_NAME",
    "PARTIAL_SPANS",
    "PARTIAL_SPANS_DELAY",
]
ENV_VARS = [f"{prefix}{suffix}" for prefix in ("RIUS_", "GLASSFLOW_") for suffix in _SUFFIXES]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_sample_rate_default_is_one() -> None:
    assert resolve_config().sample_rate == 1.0


def test_sample_rate_from_argument() -> None:
    assert resolve_config(sample_rate=0.25).sample_rate == 0.25


def test_sample_rate_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_SAMPLE_RATE", "0.5")
    assert resolve_config().sample_rate == 0.5


def test_sample_rate_out_of_range_is_clamped() -> None:
    assert resolve_config(sample_rate=1.5).sample_rate == 1.0
    assert resolve_config(sample_rate=-0.2).sample_rate == 0.0


def test_sample_rate_out_of_range_env_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_SAMPLE_RATE", "50")
    assert resolve_config().sample_rate == 1.0


def test_capture_content_default_is_true() -> None:
    assert resolve_config().capture_content is True


def test_capture_content_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_CAPTURE_CONTENT", "false")
    assert resolve_config().capture_content is False


def test_explicit_arguments_win() -> None:
    config = resolve_config(
        endpoint="https://example.com",
        api_key="secret",
        service_name="my-agent",
    )
    assert config.endpoint == "https://example.com"
    assert config.api_key == "secret"
    assert config.service_name == "my-agent"


def test_environment_variables_used_when_args_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_ENDPOINT", "https://env.example.com")
    monkeypatch.setenv("GLASSFLOW_API_KEY", "env-key")
    monkeypatch.setenv("GLASSFLOW_SERVICE_NAME", "env-agent")

    config = resolve_config()

    assert config.endpoint == "https://env.example.com"
    assert config.api_key == "env-key"
    assert config.service_name == "env-agent"


def test_explicit_arguments_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_ENDPOINT", "https://env.example.com")
    config = resolve_config(endpoint="https://arg.example.com")
    assert config.endpoint == "https://arg.example.com"


def test_defaults_when_nothing_provided() -> None:
    config = resolve_config()
    assert config.endpoint == DEFAULT_ENDPOINT
    assert config.api_key is None
    assert config.service_name == "unknown_service"
    assert config.disabled is False


def test_api_key_injected_as_bearer_header() -> None:
    config = resolve_config(api_key="secret")
    assert config.headers["Authorization"] == "Bearer secret"


def test_explicit_authorization_header_not_overwritten() -> None:
    config = resolve_config(api_key="secret", headers={"Authorization": "Bearer custom"})
    assert config.headers["Authorization"] == "Bearer custom"


def test_traces_endpoint_appends_path_and_strips_trailing_slash() -> None:
    assert resolve_config(endpoint="https://x.dev").traces_endpoint == "https://x.dev/v1/traces"
    assert resolve_config(endpoint="https://x.dev/").traces_endpoint == "https://x.dev/v1/traces"


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
def test_disabled_via_environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GLASSFLOW_DISABLED", value)
    assert resolve_config().disabled is True


def test_config_is_immutable() -> None:
    config = resolve_config()
    with pytest.raises((AttributeError, TypeError)):
        config.endpoint = "mutated"  # type: ignore[misc]


def test_returns_config_instance() -> None:
    assert isinstance(resolve_config(), GlassflowConfig)


# --- RIUS_* env vars with deprecated GLASSFLOW_* fallback ---


def test_rius_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_API_KEY", "gf_new")
    monkeypatch.setenv("RIUS_ENDPOINT", "https://rius.example.com")
    monkeypatch.setenv("RIUS_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("RIUS_HEARTBEAT", "true")
    config = resolve_config()
    assert config.api_key == "gf_new"
    assert config.endpoint == "https://rius.example.com"
    assert config.sample_rate == 0.5
    assert config.heartbeat is True


def test_glassflow_env_vars_still_work_and_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    monkeypatch.setenv("GLASSFLOW_API_KEY", "gf_legacy")
    monkeypatch.setenv("GLASSFLOW_SAMPLE_RATE", "0.25")
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert config.api_key == "gf_legacy"
    assert config.sample_rate == 0.25
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "deprecated" in warnings[0]
    assert "GLASSFLOW_API_KEY" in warnings[0] and "RIUS_API_KEY" in warnings[0]
    assert "GLASSFLOW_SAMPLE_RATE" in warnings[0] and "RIUS_SAMPLE_RATE" in warnings[0]


def test_rius_wins_over_glassflow_without_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    monkeypatch.setenv("RIUS_API_KEY", "gf_new")
    monkeypatch.setenv("GLASSFLOW_API_KEY", "gf_legacy")
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert config.api_key == "gf_new"
    assert not any(
        "deprecated" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


def test_explicit_argument_beats_both_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_API_KEY", "gf_new")
    monkeypatch.setenv("GLASSFLOW_API_KEY", "gf_legacy")
    assert resolve_config(api_key="gf_explicit").api_key == "gf_explicit"
