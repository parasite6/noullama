import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:11435"


def read_dotenv(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return {}

    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if not separator or key not in {"NOULLAMA_API_KEY", "NOULLAMA_URL"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        values[key] = value
    return values


def resolve(
    name: str,
    explicit: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv: Path | None = None,
) -> str | None:
    if explicit is not None:
        return explicit
    environ = os.environ if environ is None else environ
    if name in environ:
        return environ[name]
    dotenv = Path.cwd() / ".env" if dotenv is None else dotenv
    return read_dotenv(dotenv).get(name)


def normalize_base_url(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")
