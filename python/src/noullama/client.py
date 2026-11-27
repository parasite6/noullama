import errno
import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .answers import Decision
from .config import DEFAULT_URL, normalize_base_url, resolve
from .errors import NotRunning, NoullamaError, error_for
from .questions import Choice, Noul, Score, question_dict


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
        return [model["id"] for model in self._request("GET", "/v1/models")["data"]]

    def decide(
        self,
        *,
        model: str,
        state: str | dict | list,
        questions: Mapping[str, Choice | Score | Noul | dict],
        images: list | None = None,
        winnow_temperature: float | None = None,
    ) -> Decision:
        body = {
            "model": model,
            "state": state,
            "questions": {name: question_dict(question) for name, question in questions.items()},
        }
        if images is not None:
            body["images"] = images
        if winnow_temperature is not None:
            body["winnow"] = {"temperature": winnow_temperature}
        return Decision.from_raw(self._request("POST", "/v1/systemone", body))

    def _redact(self, message: str) -> str:
        return message.replace(self._api_key, "***") if self._api_key else message

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        timeout_message = self._redact(
            f"No response from {self.base_url} within {self.timeout} seconds."
        )
        try:
            request = Request(self.base_url + path, data=data, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout) as response:
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
                raise NoullamaError(timeout_message) from None
            raise NoullamaError(self._redact(str(error))) from None
        except TimeoutError:
            raise NoullamaError(timeout_message) from None
        except ValueError as error:
            raise NoullamaError(self._redact(str(error))) from None
        return json.loads(payload)
