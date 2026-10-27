import logging
from typing import Any

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
ENV_VARS = [f"RIUS_{suffix}" for suffix in _SUFFIXES]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_sample_rate_default_is_one() -> None:
    assert resolve_config().sample_rate == 1.0


def test_sample_rate_from_argument() -> None:
    assert resolve_config(sample_rate=0.25).sample_rate == 0.25


def test_sample_rate_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_SAMPLE_RATE", "0.5")
    assert resolve_config().sample_rate == 0.5


def test_sample_rate_out_of_range_is_clamped() -> None:
    assert resolve_config(sample_rate=1.5).sample_rate == 1.0
    assert resolve_config(sample_rate=-0.2).sample_rate == 0.0


def test_sample_rate_out_of_range_env_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_SAMPLE_RATE", "50")
    assert resolve_config().sample_rate == 1.0


def test_capture_content_default_is_true() -> None:
    assert resolve_config().capture_content is True


def test_capture_content_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_CAPTURE_CONTENT", "false")
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
    monkeypatch.setenv("RIUS_ENDPOINT", "https://env.example.com")
    monkeypatch.setenv("RIUS_API_KEY", "env-key")
    monkeypatch.setenv("RIUS_SERVICE_NAME", "env-agent")

    config = resolve_config()

    assert config.endpoint == "https://env.example.com"
    assert config.api_key == "env-key"
    assert config.service_name == "env-agent"


def test_explicit_arguments_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_ENDPOINT", "https://env.example.com")
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
    monkeypatch.setenv("RIUS_DISABLED", value)
    assert resolve_config().disabled is True


def test_config_is_immutable() -> None:
    config = resolve_config()
    with pytest.raises((AttributeError, TypeError)):
        config.endpoint = "mutated"  # type: ignore[misc]


def test_returns_config_instance() -> None:
    assert isinstance(resolve_config(), GlassflowConfig)


# --- RIUS_* is the only env prefix read ---


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


def test_pre_rename_glassflow_env_vars_are_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The GLASSFLOW_* fallback was removed in the same major bump that moved
    the wire-visible vendor names under rius.*: the old spellings are neither
    read nor warned about, so a stale value cannot silently configure the SDK."""
    import logging

    monkeypatch.setenv("GLASSFLOW_API_KEY", "gf_legacy")
    monkeypatch.setenv("GLASSFLOW_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("GLASSFLOW_ENDPOINT", "https://legacy.example.com")
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert config.api_key is None
    assert config.sample_rate == 1.0
    assert config.endpoint == DEFAULT_ENDPOINT
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_explicit_argument_beats_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_API_KEY", "gf_new")
    assert resolve_config(api_key="gf_explicit").api_key == "gf_explicit"


def test_service_version_resolves_from_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RIUS_SERVICE_VERSION", "1.0.0-env")
    assert resolve_config(service_version="1.4.2").service_version == "1.4.2"
    assert resolve_config().service_version == "1.0.0-env"


def test_service_version_is_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No placeholder: a fake default merges every deployment into one bucket,
    which is the lesson unknown_service taught. Absence is the honest answer."""
    monkeypatch.delenv("RIUS_SERVICE_VERSION", raising=False)
    assert resolve_config().service_version is None
    assert resolve_config(service_version="").service_version is None


def test_main_agent_identity_resolves_from_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RIUS_MAIN_AGENT_ID", "agent-env")
    monkeypatch.setenv("RIUS_MAIN_AGENT_DESCRIPTION", "desc-env")
    monkeypatch.setenv("RIUS_MAIN_AGENT_VERSION", "7-env")
    assert resolve_config(main_agent_id="agent-arg").main_agent_id == "agent-arg"
    assert resolve_config(main_agent_description="desc-arg").main_agent_description == "desc-arg"
    assert resolve_config(main_agent_version="7-arg").main_agent_version == "7-arg"
    config = resolve_config()
    assert config.main_agent_id == "agent-env"
    assert config.main_agent_description == "desc-env"
    assert config.main_agent_version == "7-env"


def test_main_agent_identity_is_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No placeholder, ever: an absent id, description or version reads as
    "not told", a fabricated one reads as a real (wrong) answer."""
    for name in (
        "RIUS_MAIN_AGENT_ID",
        "RIUS_MAIN_AGENT_DESCRIPTION",
        "RIUS_MAIN_AGENT_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    config = resolve_config()
    assert config.main_agent_id is None
    assert config.main_agent_description is None
    assert config.main_agent_version is None


def test_the_agent_definition_version_is_not_the_service_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two independent facts: a service can sit at 2.3.1 while its prompt,
    tools and policy are at 7. Neither is ever derived from the other."""
    monkeypatch.delenv("RIUS_SERVICE_VERSION", raising=False)
    monkeypatch.delenv("RIUS_MAIN_AGENT_VERSION", raising=False)
    assert resolve_config(service_version="2.3.1").main_agent_version is None
    assert resolve_config(main_agent_version="7").service_version is None


# --- an unrecognised boolean keeps the default, with a warning ---


@pytest.mark.parametrize(
    ("name", "attr", "default"),
    [
        ("RIUS_CAPTURE_CONTENT", "capture_content", True),
        ("RIUS_HEARTBEAT", "heartbeat", True),
        ("RIUS_DISABLED", "disabled", False),
        ("RIUS_PARTIAL_SPANS", "partial_spans", False),
    ],
)
@pytest.mark.parametrize("raw", ["", "  ", "maybe", "enabled"])
def test_an_unrecognised_boolean_env_keeps_the_default_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
    attr: str,
    default: bool,
    raw: str,
) -> None:
    """Reading it as false silently turned off default-on features such as
    content capture and the heartbeat."""
    monkeypatch.setenv(name, raw)
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert getattr(config, attr) is default
    assert any(name in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", " OFF "])
def test_a_recognised_false_boolean_env_turns_the_flag_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("RIUS_CAPTURE_CONTENT", raw)
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        assert resolve_config().capture_content is False
    assert not caplog.records


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("RIUS_SAMPLE_RATE", "0.5abc"),
        ("RIUS_SAMPLE_RATE", ""),
        ("RIUS_HEARTBEAT_INTERVAL", "10s"),
        ("RIUS_PARTIAL_SPANS_DELAY", "5 seconds"),
    ],
)
def test_an_unparseable_numeric_env_keeps_the_default_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: str, raw: str
) -> None:
    """Warn and skip, the SDK-wide policy for bad input, as the TypeScript
    SDK does: a silent fallback hid a typo in the variable."""
    monkeypatch.setenv(name, raw)
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert config == resolve_config_without(monkeypatch, name)
    assert any(name in record.getMessage() for record in caplog.records)


def resolve_config_without(monkeypatch: pytest.MonkeyPatch, name: str) -> Any:
    monkeypatch.delenv(name)
    return resolve_config()


# ---------------------------------------------------------------------------
# Plain-HTTP endpoints
# ---------------------------------------------------------------------------


def _plaintext_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "unencrypted" in r.getMessage()
    ]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://collector.internal:4318",
        "HTTP://ingest.example.com",
        "http://10.0.0.5:4318",
        # Resolvers send subdomains of localhost to DNS, which may answer remotely.
        "http://agent.localhost:4318",
    ],
)
def test_a_remote_http_endpoint_warns_that_traffic_is_unencrypted(
    endpoint: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(endpoint=endpoint, api_key="gf_secret")
    [message] = _plaintext_warnings(caplog)
    assert "https://" in message


def test_an_http_endpoint_from_the_env_warns_too(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RIUS_ENDPOINT", "http://collector.internal:4318")
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config()
    assert len(_plaintext_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        DEFAULT_ENDPOINT,
        "https://collector.internal:4318",
        "http://localhost:4318",
        "http://LOCALHOST:4318",
        "http://127.0.0.1:8787",
        "http://127.5.5.5",
        "http://[::1]:4318",
        "http://localhost.:4318",
        "http://127.0.0.1.:4318",
        "http://127.1:4318",
        "http://2130706433",
        "http://[::ffff:127.0.0.1]:4318",
    ],
)
def test_https_and_loopback_endpoints_do_not_warn(
    endpoint: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(endpoint=endpoint, api_key="gf_secret")
    assert _plaintext_warnings(caplog) == []


def test_the_plaintext_warning_never_logs_credentials(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(
            endpoint="http://user:hunter2@collector.internal/otlp?token=t0k", api_key="gf_secret"
        )
    [message] = _plaintext_warnings(caplog)
    for secret in ("hunter2", "user:", "t0k", "gf_secret"):
        assert secret not in message
    assert "collector.internal" in message


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gf_SECRETKEY:x?y@collector.internal",
        "http://user:ab/cd@collector.internal",
        "http://tok/en@collector.internal",
    ],
)
def test_userinfo_the_parser_splits_early_is_never_logged_as_the_host(
    endpoint: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(endpoint=endpoint)
    [message] = _plaintext_warnings(caplog)
    for secret in ("secretkey", "SECRETKEY", "user", "tok"):
        assert secret not in message


def test_an_endpoint_without_a_host_still_reads_cleanly(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(endpoint="http:/collector")
    [message] = _plaintext_warnings(caplog)
    assert "  " not in message


def test_a_disabled_sdk_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        resolve_config(endpoint="http://collector.internal:4318", disabled=True)
        monkeypatch.setenv("RIUS_DISABLED", "true")
        resolve_config(endpoint="http://collector.internal:4318")
    assert _plaintext_warnings(caplog) == []
