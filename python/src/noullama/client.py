import errno
import json
import time
from collections.abc import Mapping
from os import PathLike
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .answers import Decision
from .config import DEFAULT_URL, normalize_base_url, resolve
from .errors import ModelUnavailable, NotRunning, NoullamaError, error_for
from .images import to_data_url
from .questions import Choice, Noul, Score, question_dict


class _RequestTimeout(NoullamaError):
    pass


class Client:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 900.0,
    ):
        url = resolve("NOULLAMA_URL", base_url)
        self.base_url = normalize_base_url(DEFAULT_URL if url is None else url)
        self._api_key = resolve("NOULLAMA_API_KEY", api_key)
        self.timeout = timeout

    def __repr__(self) -> str:
        key = "***" if self._api_key else None
        return f"Client(base_url={self._redact(self.base_url)!r}, api_key={key!r}, timeout={self.timeout!r})"

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
        body = {
            "model": model,
            "state": state,
            "questions": {name: question_dict(question) for name, question in questions.items()},
        }
        if images is not None:
            body["images"] = [to_data_url(image) for image in images]
        if winnow_temperature is not None:
            body["winnow"] = {"temperature": winnow_temperature}
        return Decision.from_raw(self._request("POST", "/v1/systemone", body))

    def wait_until_ready(
        self,
        *,
        model: str | None = None,
        timeout: float = 60.0,
        interval: float = 0.5,
    ) -> None:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            request_timeout = min(self.timeout, remaining)
            try:
                if model is None:
                    ready = self._request("GET", "/health", timeout=request_timeout)["model_ready"]
                else:
                    ready = model in self._model_ids(timeout=request_timeout)
                if ready and time.monotonic() <= deadline:
                    return
            except (NotRunning, ModelUnavailable):
                pass
            except _RequestTimeout:
                if request_timeout < remaining:
                    raise
                break

            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(interval, remaining))

        raise TimeoutError(
            self._redact(f"{model or 'A model'} was not ready after {timeout} seconds.")
        )

    def _redact(self, message: str) -> str:
        return message.replace(self._api_key, "***") if self._api_key else message

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout: float | None = None,
    ) -> dict:
        timeout = self.timeout if timeout is None else timeout
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        timeout_message = self._redact(
            f"No response from {self.base_url} within {timeout} seconds."
        )
        try:
            request = Request(self.base_url + path, data=data, headers=headers, method=method)
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
            except HTTPError as error:
                with error:
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
                message = (
                    f"Noullama is not running at {self.base_url}. "
                    "Open the app, or set NOULLAMA_URL."
                )
                raise NotRunning(self._redact(message)) from None
            if isinstance(reason, TimeoutError):
                raise _RequestTimeout(timeout_message) from None
            raise NoullamaError(self._redact(str(error))) from None
        except TimeoutError:
            raise _RequestTimeout(timeout_message) from None
        except ValueError as error:
            raise NoullamaError(self._redact(str(error))) from None
        return json.loads(payload)
