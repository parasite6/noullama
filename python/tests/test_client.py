import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fake_gateway import FakeGateway
from noullama import (
    AmbiguousModel, AuthError, BadRequest, Choice, Client, Forbidden,
    InferenceFailed, ModelUnavailable, NotRunning, Noul, NoullamaError, Score, UnknownModel,
)
from test_answers import NATIVE


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ClientTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(temp.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_decide_wire_format(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            d = Client(base_url=gw.url, api_key="da-noul-secret").decide(model="d1-2B-Q8_0", state="s",
                  questions={"route": Choice(instructions="Team?", billing=None, technical=None)},
                  winnow_temperature=1.2)
            r = gw.requests[-1]
            self.assertEqual(r["body"], {"model": "d1-2B-Q8_0", "state": "s", "winnow": {"temperature": 1.2},
                "questions": {"route": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "technical": None}}}})
            self.assertEqual(r["headers"]["Authorization"], "Bearer da-noul-secret")
            self.assertEqual(d["route"].value, "billing")

    def test_no_auth_header_without_key(self):
        with FakeGateway() as gw:
            Client(base_url=gw.url).health()
            self.assertNotIn("Authorization", gw.requests[-1]["headers"])

    def test_decide_is_keyword_only(self):
        with self.assertRaises(TypeError):
            Client(base_url="http://x").decide("m", "s", {})

    def test_models_lists_ids(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/v1/models", 200, {"object": "list", "data": [{"id": "d1"}, {"id": "any-available"}]})
            self.assertEqual(Client(base_url=gw.url).models(), ["d1", "any-available"])

    def test_error_statuses(self):
        for status, cls in [(400, BadRequest), (401, AuthError), (403, Forbidden), (404, UnknownModel),
                            (409, AmbiguousModel), (502, InferenceFailed), (503, ModelUnavailable)]:
            with FakeGateway() as gw:
                gw.reply("POST", "/v1/systemone", status, {"error": {"message": f"m{status}", "type": "invalid_request_error", "code": status}})
                with self.assertRaises(cls) as cm:
                    Client(base_url=gw.url).decide(model="m", state="s", questions={"q": Noul(instructions="x")})
                self.assertEqual((cm.exception.status, str(cm.exception)), (status, f"m{status}"))

    def test_plain_text_error_body(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 413, "Failed to buffer the request body: length limit exceeded")
            with self.assertRaises(NoullamaError) as cm:
                Client(base_url=gw.url).decide(model="m", state="s", questions={})
            self.assertEqual(cm.exception.status, 413)
        self.assertIn("length limit", str(cm.exception))

    def test_not_running(self):
        port = free_port()
        with self.assertRaises(NotRunning):
            Client(base_url=f"http://127.0.0.1:{port}").health()

    def test_key_never_leaks(self):
        c = Client(base_url="http://127.0.0.1:1", api_key="da-noul-secret")
        self.assertNotIn("da-noul-secret", repr(c))
        with self.assertRaises(NotRunning) as cm:
            c.health()
        self.assertNotIn("da-noul-secret", str(cm.exception))

    def test_non_ascii_state_round_trips(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 200, {"model": "m", "answers": {}, "usage": {}})
            Client(base_url=gw.url).decide(model="m", state="Rückerstattung bitte 💳", questions={"q": Noul(instructions="x")})
            self.assertEqual(gw.requests[-1]["body"]["state"], "Rückerstattung bitte 💳")

    def test_env_and_dotenv_used(self):
        with FakeGateway() as gw:
            Path(".env").write_text(f"NOULLAMA_URL={gw.url}/v1\nNOULLAMA_API_KEY=k\n")
            client = Client()
            self.assertEqual(client.base_url, gw.url)
            self.assertEqual(gw.requests, [])
            self.assertTrue(client.health()["model_ready"])
            self.assertEqual(gw.requests[-1]["headers"]["Authorization"], "Bearer k")

    def test_constructor_defaults_and_keyword_only(self):
        client = Client()
        self.assertEqual(client.base_url, "http://127.0.0.1:11435")
        self.assertEqual(client.timeout, 900.0)
        with self.assertRaises(TypeError):
            Client("http://127.0.0.1:11435")

    def test_echoed_key_is_redacted(self):
        with FakeGateway() as gw:
            key = "da-noul-secret"
            gw.reply("GET", "/health", 401, {"error": {"message": f"Rejected {key}."}})
            with self.assertRaises(AuthError) as cm:
                Client(base_url=gw.url, api_key=key).health()
            self.assertEqual(cm.exception.message, "Rejected ***.")
            self.assertNotIn(key, str(cm.exception))

    def test_score_and_noul_wire_format(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            Client(base_url=gw.url).decide(
                model="m",
                state={"case": 7},
                questions={
                    "urgency": Score(instructions=["How urgent?"], levels=["Can wait", "Now"]),
                    "refund": Noul(instructions="Refund?", true="Requested"),
                },
            )
            self.assertEqual(gw.requests[-1]["body"]["questions"], {
                "urgency": {"type": "score", "instructions": ["How urgent?"], "criteria": ["Can wait", "Now"]},
                "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "Requested"}},
            })

    def test_socket_timeout(self):
        url = "http://127.0.0.1:11435"
        with patch("noullama.client.urlopen", side_effect=TimeoutError):
            with self.assertRaises(NoullamaError) as cm:
                Client(base_url=url, timeout=0.1).health()
        self.assertEqual(str(cm.exception), f"No response from {url} within 0.1 seconds.")

    def test_decide_encodes_images(self):
        from noullama.images import to_data_url
        from test_images import PNG

        with FakeGateway() as gw:
            client = Client(base_url=gw.url)
            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            client.decide(model="m", state="s", questions={}, images=[PNG])
            self.assertEqual(gw.requests[-1]["body"]["images"], [to_data_url(PNG)])

            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            client.decide(model="m", state="s", questions={})
            self.assertNotIn("images", gw.requests[-1]["body"])

    def test_waits_for_named_model(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/v1/models", 200, {"data": []})
            gw.reply("GET", "/v1/models", 200, {"data": [{"id": "d1"}]})
            Client(base_url=gw.url).wait_until_ready(model="d1", timeout=5, interval=0.01)
            self.assertEqual(sum(r["path"] == "/v1/models" for r in gw.requests), 2)

    def test_waits_for_any_model(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/health", 200, {"model_ready": False})
            gw.reply("GET", "/health", 200, {"model_ready": True})
            Client(base_url=gw.url).wait_until_ready(timeout=5, interval=0.01)

    def test_times_out(self):
        with FakeGateway() as gw:
            for _ in range(100):
                gw.reply("GET", "/v1/models", 200, {"data": []})
            with self.assertRaises(TimeoutError):
                Client(base_url=gw.url).wait_until_ready(model="d1", timeout=0.1, interval=0.01)

    def test_not_running_is_retried_then_times_out(self):
        with self.assertRaises(TimeoutError):
            Client(base_url=f"http://127.0.0.1:{free_port()}").wait_until_ready(timeout=0.1, interval=0.01)

    def test_auth_error_raised_at_once(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/health", 401, {"error": {"message": "Unknown API key.", "code": 401}})
            with self.assertRaises(AuthError):
                Client(base_url=gw.url).wait_until_ready(timeout=5)

    def test_model_unavailable_is_retried(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/health", 503, {"error": {"message": "Still loading."}})
            gw.reply("GET", "/health", 200, {"model_ready": True})
            Client(base_url=gw.url).wait_until_ready(timeout=5, interval=0.01)
            self.assertEqual(len(gw.requests), 2)

    def test_poll_request_is_capped_by_remaining_time(self):
        client = Client(timeout=900.0)
        with patch("noullama.client.urlopen", side_effect=TimeoutError) as request:
            with self.assertRaisesRegex(TimeoutError, r"d1 was not ready after 0\.1 seconds\."):
                client.wait_until_ready(model="d1", timeout=0.1)
        self.assertGreater(request.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(request.call_args.kwargs["timeout"], 0.1)
        self.assertEqual(client.timeout, 900.0)
