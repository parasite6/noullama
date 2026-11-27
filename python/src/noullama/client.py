import errno
import json
import time
from collections.abc import Mapping
from http.client import HTTPException
from os import PathLike
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .answers import Decision
from .config import DEFAULT_URL, normalize_base_url, resolve
from .errors import ModelUnavailable, NotRunning, NoullamaError, error_for
from .images import to_data_url
from .questions import Choice, Noul, Score, question_dict


class _NoRedirects(HTTPRedirectHandler):
    def http_error_302(self, request, response, code, message, headers):
        # Let urllib's default error handler raise HTTPError for the original response.
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


# No proxies: the gateway listens on 127.0.0.1 only, so a proxy can't reach it
# and would see the API key.
_OPENER = build_opener(ProxyHandler({}), _NoRedirects())


class _RequestTimeout(NoullamaError):
    pass


_UNSET = object()
_ADDRESS_FIELDS = {"url", "api_key", "timeout"}

Address = str | Mapping[str, object]


class Client:
    def __init__(
        self,
        addresses: Mapping[str, Address] | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 900.0,
    ):
        self.timeout = timeout
        self._name = None
        self._clients: dict[str, Client] = {}
        if addresses is None:
            url = resolve("NOULLAMA_URL", base_url)
            self.base_url = normalize_base_url(DEFAULT_URL if url is None else url)
            self._api_key = resolve("NOULLAMA_API_KEY", api_key)
            return

        if not isinstance(addresses, Mapping):
            raise TypeError(
                "Client() takes a dict of names and addresses, such as "
                'Client({"gpu": "http://127.0.0.1:11440/v1"}); for one address use base_url=...'
            )
        if base_url is not None:
            raise TypeError("Pass either a dict of addresses or base_url, not both.")
        if not addresses:
            raise ValueError("The dict of addresses needs at least one name.")
        self.base_url = None
        self._api_key = None
        shared_key = resolve("NOULLAMA_API_KEY", api_key)
        for name, address in addresses.items():
            self._clients[name] = type(self)._named(name, address, shared_key, timeout)

    @classmethod
    def _named(cls, name, address, shared_key, timeout) -> "Client":
        if not isinstance(name, str):
            raise TypeError(f"Address names must be strings, not {type(name).__name__}.")
        if not name:
            raise ValueError("An address name can't be empty.")
        key = timeout_value = _UNSET
        if isinstance(address, Mapping):
            unknown = set(address) - _ADDRESS_FIELDS
            if unknown:
                fields = ", ".join(sorted(map(str, unknown)))
                raise ValueError(f'Address "{name}" has unknown field(s): {fields}. Use url, api_key and timeout.')
            if "url" not in address:
                raise ValueError(f'Address "{name}" needs a "url".')
            url = address["url"]
            key = address.get("api_key", _UNSET)
            timeout_value = address.get("timeout", _UNSET)
        else:
            url = address
        if not isinstance(url, str):
            raise TypeError(f'The address for "{name}" must be a URL string, not {type(url).__name__}.')
        if not url.strip():
            raise ValueError(f'The address for "{name}" is empty.')

        client = cls.__new__(cls)
        client._name = name
        client._clients = {}
        client.base_url = normalize_base_url(url.strip())
        client._api_key = shared_key if key is _UNSET else key
        client.timeout = timeout if timeout_value is _UNSET else timeout_value
        return client

    @property
    def names(self) -> list[str]:
        return list(self._clients)

    def items(self):
        return self._clients.items()

    def __contains__(self, name: object) -> bool:
        return name in self._clients

    def __getitem__(self, name: str) -> "Client":
        if not self._clients:
            raise KeyError(
                f"{name!r}: this client has no named addresses. "
                'Make one with Client({"gpu": "http://127.0.0.1:11440/v1", ...}).'
            )
        try:
            return self._clients[name]
        except KeyError:
            raise KeyError(f"{name!r} is not one of this client's addresses: {', '.join(self._clients)}.") from None

    def __repr__(self) -> str:
        if self._clients:
            return f"Client({self.names!r}, timeout={self.timeout!r})"
        key = "***" if self._api_key else None
        name = "" if self._name is None else f"name={self._name!r}, "
        return f"Client({name}base_url={self._redact(self.base_url)!r}, api_key={key!r}, timeout={self.timeout!r})"

    def _need_address(self) -> None:
        if self._clients:
            first = next(iter(self._clients))
            raise TypeError(
                f'This client has several addresses; pick one first, such as nl["{first}"].decide(...).'
            )

    def _url_hint(self, verb: str) -> str:
        return f"{verb} NOULLAMA_URL" if self._name is None else f'check the address for "{self._name}"'

    def health(self) -> dict:
        return self._request("GET", "/health")

    def models(self) -> list[str]:
        return self._model_ids()

    def _model_ids(self, *, timeout: float | None = None) -> list[str]:
        response = self._request("GET", "/v1/models", timeout=timeout)
        return [model["id"] for model in response["data"]]

    def decide(
        self,
        *,
        model: str,
        state: str | dict | list,
        questions: Mapping[str, Choice | Score | Noul | dict],
        images: list[str | bytes | PathLike] | None = None,
        winnow_temperature: float | None = None,
    ) -> Decision:
        self._need_address()
        body = {
            "model": model,
            "state": state,
            "questions": {name: question_dict(question) for name, question in questions.items()},
        }
        if images is not None:
            body["images"] = [to_data_url(image) for image in images]
        if winnow_temperature is not None:
            body["winnow"] = {"temperature": winnow_temperature}
        started = time.perf_counter()
        raw = self._request("POST", "/v1/systemone", body)
        return Decision.from_raw(raw, nl_time=time.perf_counter() - started)

    def wait_until_ready(
        self,
        *,
        model: str | None = None,
        timeout: float = 60.0,
        interval: float = 0.5,
    ) -> None:
        self._need_address()
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            request_timeout = min(self.timeout, remaining)
            ready = False
            try:
                if model is None:
                    ready = self._request("GET", "/health", timeout=request_timeout)["model_ready"]
                else:
                    ready = model in self._model_ids(timeout=request_timeout)
            except (NotRunning, ModelUnavailable):
                pass
            except _RequestTimeout:
                if request_timeout < remaining and time.monotonic() < deadline:
                    raise
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if ready:
                return
            time.sleep(min(interval, remaining))

        raise TimeoutError(
            self._redact(f"{model or 'A model'} was not ready after {timeout} seconds.")
        )

    def _redact(self, message: str) -> str:
        return message.replace(self._api_key, "***") if self._api_key else message

    def _open(self, method: str, path: str, body: dict | None, timeout: float):
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        return _OPENER.open(request, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        self._need_address()
        timeout = self.timeout if timeout is None else timeout
        timeout_message = self._redact(
            f"No response from {self.base_url} within {timeout} seconds."
        )
        try:
            try:
                with self._open(method, path, body, timeout) as response:
                    payload = response.read()
                    status = response.getcode()
            except HTTPError as error:
                with error:
                    if 300 <= error.code < 400:
                        raise NoullamaError(
                            f"Noullama answered with a redirect to another address; {self._url_hint("check")}.",
                            error.code,
                        ) from None
                    text = error.read().decode("utf-8", errors="replace")
                try:
                    message = json.loads(text)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = None
                if isinstance(message, str):
                    raise error_for(error.code, self._redact(message)) from None
                raise NoullamaError(self._redact(text), error.code) from None
        except URLError as error:
            reason = error.reason
            if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
                where = "" if self._name is None else f' ("{self._name}")'
                message = (
                    f"Noullama is not running at {self.base_url}{where}. "
                    f"Open the app, or {self._url_hint('set')}."
                )
                raise NotRunning(self._redact(message)) from None
            if isinstance(reason, TimeoutError):
                raise _RequestTimeout(timeout_message) from None
            raise NoullamaError(self._redact(str(error))) from None
        except TimeoutError:
            raise _RequestTimeout(timeout_message) from None
        except (HTTPException, OSError, ValueError) as error:
            raise NoullamaError(self._redact(str(error))) from None
        try:
            return json.loads(payload)
        except ValueError:
            message = f"Noullama sent a response that isn't JSON (status {status})."
            raise NoullamaError(self._redact(message), status) from None
