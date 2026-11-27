import os
import socket
import tempfile
import traceback
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

    def test_decide_gives_score_labels_from_the_sent_levels(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 200, {"answers": {"t": {"type": "score", "score": 1.0}}})
            d = Client(base_url=gw.url).decide(model="m", state="s", questions={
                "t": Score(instructions="How bad?", levels=["Can wait", "Keep watching", "Notify seniors"]),
            })
            self.assertEqual(d["t"].label, "Keep watching")

    def test_decide_records_nl_time(self):
        with FakeGateway() as gw:
            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            gw.reply("POST", "/v1/systemone", 200, NATIVE)
            client = Client(base_url=gw.url)
            with patch("noullama.client.time.perf_counter", side_effect=[10.0, 10.25]):
                decision = client.decide(model="m", state="s", questions={"q": Noul(instructions="x")})
            self.assertEqual(decision.nl_time, 0.25)
            real = client.decide(model="m", state="s", questions={"q": Noul(instructions="x")})
            self.assertIsInstance(real.nl_time, float)
            self.assertGreaterEqual(real.nl_time, 0)

    def test_proxy_settings_are_ignored(self):
        # The gateway listens on 127.0.0.1 only, so a proxy can never reach it
        # and would see the API key. Reload so import-time proxy lookup is covered too.
        import importlib
        import noullama.client as client_module
        with FakeGateway() as gw, FakeGateway() as proxy:
            with patch.dict(os.environ, {"HTTP_PROXY": proxy.url, "http_proxy": proxy.url}):
                reloaded = importlib.reload(client_module)
                self.addCleanup(importlib.reload, client_module)
                reloaded.Client(base_url=gw.url, api_key="da-noul-secret").health()
            self.assertEqual(proxy.requests, [])
            self.assertEqual(gw.requests[-1]["path"], "/health")

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

    def test_non_json_success_is_a_noullama_error(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/health", 200, "<html>da-noul-test</html>")
            with self.assertRaises(NoullamaError) as error:
                Client(base_url=gw.url, api_key="da-noul-test").health()
            self.assertEqual(error.exception.status, 200)
            self.assertEqual(str(error.exception), "Noullama sent a response that isn't JSON (status 200).")
            self.assertTrue(error.exception.__suppress_context__)

    def test_truncated_response_is_a_noullama_error(self):
        with FakeGateway() as gw:
            gw.reply("GET", "/health", 200, "short", headers={"Content-Length": "100"})
            with self.assertRaises(NoullamaError) as error:
                Client(base_url=gw.url).health()
            self.assertTrue(error.exception.__suppress_context__)

    def test_response_read_os_error_is_redacted(self):
        key = "da-noul-read-error"
        with patch("noullama.client._OPENER.open") as opening:
            response = opening.return_value.__enter__.return_value
            response.read.side_effect = OSError(f"Failed while reading {key}")
            with self.assertRaises(NoullamaError) as error:
                Client(api_key=key).health()
            self.assertEqual(str(error.exception), "Failed while reading ***")
            self.assertTrue(error.exception.__suppress_context__)

    def test_redirect_does_not_reach_target(self):
        with FakeGateway() as gateway, FakeGateway() as target:
            gateway.reply("GET", "/health", 302, "", headers={"Location": target.url + "/health"})
            client = Client(base_url=gateway.url, api_key="da-noul-test")
            with self.assertRaises(NoullamaError) as error:
                client.health()
            self.assertEqual(target.requests, [])
            self.assertEqual(error.exception.status, 302)
            self.assertEqual(
                str(error.exception),
                "Noullama answered with a redirect to another address; check NOULLAMA_URL.",
            )

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

    def test_invalid_url_redacts_key_from_error_and_traceback(self):
        key = "da-noul-invalid-url"
        client = Client(base_url=f"http://127.0.0.1:1/{key} path", api_key=key)
        try:
            client.health()
        except NoullamaError as error:
            self.assertIn("***", str(error))
            self.assertNotIn(key, str(error))
            self.assertNotIn(key, "".join(traceback.format_exception(error)))
            self.assertTrue(error.__suppress_context__)
        else:
            self.fail("The invalid URL should raise NoullamaError.")

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
        with patch("noullama.client._OPENER.open", side_effect=TimeoutError):
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
            self.assertEqual([request["path"] for request in gw.requests], ["/health", "/health"])

    def test_times_out(self):
        with FakeGateway() as gw:
            for _ in range(100):
                gw.reply("GET", "/v1/models", 200, {"data": []})
            with self.assertRaises(TimeoutError):
                Client(base_url=gw.url).wait_until_ready(model="d1", timeout=0.1, interval=0.01)

    def test_not_running_is_retried_then_times_out(self):
        client = Client()
        with patch("noullama.client.time") as clock:
            clock.monotonic.side_effect = [0, 0, 0, 0.5, 1]
            with patch.object(client, "_request", side_effect=NotRunning("Offline")) as request:
                with self.assertRaises(TimeoutError):
                    client.wait_until_ready(timeout=1, interval=0.5)
            self.assertEqual(request.call_count, 2)
            clock.sleep.assert_called_once_with(0.5)

    def test_ready_at_deadline_times_out_without_another_poll(self):
        with FakeGateway() as gw, patch("noullama.client.time") as clock:
            clock.monotonic.side_effect = [0, 0, 1]
            with self.assertRaises(TimeoutError):
                Client(base_url=gw.url).wait_until_ready(timeout=1)
            self.assertEqual([request["path"] for request in gw.requests], ["/health"])
            clock.sleep.assert_not_called()

    def test_short_socket_timeout_after_deadline_raises_readiness_timeout(self):
        client = Client(timeout=0.5)
        with patch("noullama.client.time") as clock:
            clock.monotonic.side_effect = [0, 0, 2]
            with patch("noullama.client._OPENER.open", side_effect=TimeoutError) as request:
                with self.assertRaises(TimeoutError):
                    client.wait_until_ready(timeout=1)
            request.assert_called_once()
            clock.sleep.assert_not_called()

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
        with patch("noullama.client._OPENER.open", side_effect=TimeoutError) as request:
            with self.assertRaisesRegex(TimeoutError, r"d1 was not ready after 0\.1 seconds\."):
                client.wait_until_ready(model="d1", timeout=0.1)
        self.assertGreater(request.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(request.call_args.kwargs["timeout"], 0.1)
        self.assertEqual(client.timeout, 900.0)


class NamedAddressTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(temp.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_each_name_reaches_its_own_address(self):
        with FakeGateway() as gpu, FakeGateway() as vision:
            gpu.reply("POST", "/v1/systemone", 200, NATIVE)
            vision.reply("POST", "/v1/systemone", 200, NATIVE)
            nl = Client({"gpu": gpu.url + "/v1", "vision": vision.url})
            questions = {"q": Noul(instructions="x")}
            nl["gpu"].decide(model="a", state="s", questions=questions)
            nl["vision"].decide(model="b", state="s", questions=questions)
            self.assertEqual([r["body"]["model"] for r in gpu.requests], ["a"])
            self.assertEqual([r["body"]["model"] for r in vision.requests], ["b"])
            self.assertEqual(nl["gpu"].base_url, gpu.url)

    def test_same_client_is_returned_for_a_name(self):
        nl = Client({"gpu": "http://127.0.0.1:11440"})
        self.assertIs(nl["gpu"], nl["gpu"])
        self.assertIs(type(nl["gpu"]), type(nl))

    def test_names_and_items(self):
        nl = Client({"gpu": "http://127.0.0.1:11440", "cpu": "http://127.0.0.1:11441"})
        self.assertEqual(nl.names, ["gpu", "cpu"])
        self.assertEqual([(name, c.base_url) for name, c in nl.items()],
                         [("gpu", "http://127.0.0.1:11440"), ("cpu", "http://127.0.0.1:11441")])
        self.assertIn("gpu", nl)
        self.assertNotIn("gpuu", nl)

    def test_unknown_name_lists_the_names(self):
        nl = Client({"gpu": "http://127.0.0.1:11440", "cpu": "http://127.0.0.1:11441"})
        with self.assertRaises(KeyError) as cm:
            nl["gpuu"]
        self.assertIn("gpuu", str(cm.exception))
        self.assertIn("gpu, cpu", str(cm.exception))

    def test_single_address_client_has_no_names(self):
        nl = Client(base_url="http://127.0.0.1:11440")
        self.assertEqual(nl.names, [])
        with self.assertRaisesRegex(KeyError, "no named addresses"):
            nl["gpu"]

    def test_top_level_calls_ask_for_a_name(self):
        nl = Client({"gpu": "http://127.0.0.1:1"})
        with patch("noullama.client._OPENER.open") as request:
            for call in (nl.health, nl.models, lambda: nl.wait_until_ready(timeout=1),
                         lambda: nl.decide(model="m", state="s", questions={"q": Noul(instructions="x")})):
                with self.assertRaisesRegex(TypeError, r'nl\["gpu"\]'):
                    call()
            request.assert_not_called()

    def test_shared_key_from_argument_env_and_dotenv(self):
        with FakeGateway() as gw:
            Client({"a": gw.url}, api_key="arg-key")["a"].health()
            os.environ["NOULLAMA_API_KEY"] = "env-key"
            Client({"a": gw.url})["a"].health()
            del os.environ["NOULLAMA_API_KEY"]
            Path(".env").write_text("NOULLAMA_API_KEY=dotenv-key\n")
            Client({"a": gw.url})["a"].health()
            self.assertEqual([r["headers"]["Authorization"] for r in gw.requests],
                             ["Bearer arg-key", "Bearer env-key", "Bearer dotenv-key"])

    def test_per_address_key_and_timeout(self):
        with FakeGateway() as gw:
            os.environ["NOULLAMA_API_KEY"] = "shared"
            nl = Client({
                "own": {"url": gw.url, "api_key": "own-key", "timeout": 5},
                "none": {"url": gw.url, "api_key": None},
                "shared": gw.url,
            }, timeout=30)
            for name in ("own", "none", "shared"):
                nl[name].health()
            headers = [r["headers"].get("Authorization") for r in gw.requests]
            self.assertEqual(headers, ["Bearer own-key", None, "Bearer shared"])
            self.assertEqual((nl["own"].timeout, nl["shared"].timeout), (5, 30))

    def test_named_addresses_ignore_noullama_url(self):
        os.environ["NOULLAMA_URL"] = "http://127.0.0.1:9999"
        self.assertEqual(Client({"a": "http://127.0.0.1:11440"})["a"].base_url, "http://127.0.0.1:11440")

    def test_bad_address_tables(self):
        cases = [
            ({}, ValueError, "at least one"),
            ({"": "http://x"}, ValueError, "name"),
            ({1: "http://x"}, TypeError, "name"),
            ({"a": ""}, ValueError, '"a"'),
            ({"a": 11440}, TypeError, '"a"'),
            ({"a": {"api_key": "k"}}, ValueError, '"url"'),
            ({"a": {"url": "http://x", "key": "k"}}, ValueError, "key"),
            ({"a": {"url": "http://x", "api_key": 123}}, TypeError, 'api_key for "a"'),
            ({"a": {"url": "http://x", "timeout": "60"}}, TypeError, 'timeout for "a"'),
            ({"a": {"url": "http://x", "timeout": True}}, TypeError, 'timeout for "a"'),
            ({"a": {"url": "http://x", "timeout": 0}}, ValueError, 'timeout for "a"'),
        ]
        for servers, error, text in cases:
            with self.subTest(servers=servers), self.assertRaisesRegex(error, text):
                Client(servers)
        with self.assertRaisesRegex(TypeError, "base_url"):
            Client({"a": "http://x"}, base_url="http://y")
        with self.assertRaisesRegex(TypeError, "dict"):
            Client("http://127.0.0.1:11435")

    def test_not_running_names_the_address(self):
        port = free_port()
        with self.assertRaises(NotRunning) as cm:
            Client({"gpu": f"http://127.0.0.1:{port}"})["gpu"].health()
        self.assertIn('"gpu"', str(cm.exception))
        self.assertNotIn("NOULLAMA_URL", str(cm.exception))

    def test_keys_never_leak_in_repr(self):
        nl = Client({"a": {"url": "http://127.0.0.1:1", "api_key": "da-noul-secret"}})
        self.assertNotIn("da-noul-secret", repr(nl))
        self.assertNotIn("da-noul-secret", repr(nl["a"]))
        self.assertIn("'a'", repr(nl))
