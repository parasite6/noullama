import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from noullama.config import normalize_base_url, read_dotenv, resolve
from noullama.errors import (
    AmbiguousModel, AuthError, BadRequest, Forbidden, InferenceFailed,
    ModelUnavailable, NoullamaError, UnknownModel, error_for,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write_env(self, content):
        path = self.tmp / ".env"
        path.write_bytes(content.encode("utf-8"))
        return path

    def test_lookup_order(self):
        env_file = self.write_env("NOULLAMA_API_KEY=from-file\n")
        self.assertEqual(resolve("NOULLAMA_API_KEY", "arg", environ={"NOULLAMA_API_KEY": "env"}, dotenv=env_file), "arg")
        self.assertEqual(resolve("NOULLAMA_API_KEY", None, environ={"NOULLAMA_API_KEY": "env"}, dotenv=env_file), "env")
        self.assertEqual(resolve("NOULLAMA_API_KEY", None, environ={}, dotenv=env_file), "from-file")
        self.assertIsNone(resolve("NOULLAMA_API_KEY", None, environ={}, dotenv=self.tmp / "missing"))

    def test_dotenv_edge_cases(self):
        p = self.write_env('# c\r\nexport NOULLAMA_API_KEY="da-noul-a=b#c"\r\n\r\nNOULLAMA_URL=\'http://x:1\'\r\nOTHER=1\r\n')
        self.assertEqual(read_dotenv(p), {"NOULLAMA_API_KEY": "da-noul-a=b#c", "NOULLAMA_URL": "http://x:1"})
        self.assertEqual(read_dotenv(self.tmp / "nope"), {})

    def test_dotenv_keeps_first_key_after_bom(self):
        path = self.write_env("\ufeffNOULLAMA_API_KEY=da-noul-test\n")
        self.assertEqual(read_dotenv(path), {"NOULLAMA_API_KEY": "da-noul-test"})

    def test_dotenv_not_exported(self):
        resolve("NOULLAMA_API_KEY", None, environ={}, dotenv=self.write_env("NOULLAMA_API_KEY=k\n"))
        self.assertNotIn("NOULLAMA_API_KEY", os.environ)

    def test_base_url_strips_v1_and_slash(self):
        for u in ["http://127.0.0.1:11440", "http://127.0.0.1:11440/", "http://127.0.0.1:11440/v1", "http://127.0.0.1:11440/v1/"]:
            self.assertEqual(normalize_base_url(u), "http://127.0.0.1:11440")

    def test_error_for_maps_statuses(self):
        for status, cls in [(400, BadRequest), (401, AuthError), (403, Forbidden), (404, UnknownModel),
                            (409, AmbiguousModel), (502, InferenceFailed), (503, ModelUnavailable)]:
            e = error_for(status, "msg")
            self.assertIs(type(e), cls)
            self.assertEqual((e.status, str(e)), (status, "msg"))
        self.assertIs(type(error_for(418, "t")), NoullamaError)
