import base64
import tempfile
import unittest
from pathlib import Path

from noullama.images import to_data_url

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 8


class ImageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def test_bytes_png(self):
        self.assertEqual(to_data_url(PNG), "data:image/png;base64," + base64.b64encode(PNG).decode())

    def test_path_jpeg(self):
        path = self.tmp / "r.bin"
        path.write_bytes(b"\xff\xd8\xff\xe0rest")
        self.assertTrue(to_data_url(path).startswith("data:image/jpeg;base64,"))

    def test_str_path(self):
        path = self.tmp / "r.webp"
        path.write_bytes(b"RIFF\0\0\0\0WEBPVP8 ")
        self.assertTrue(to_data_url(str(path)).startswith("data:image/webp;"))

    def test_extension_fallback(self):
        path = self.tmp / "x.png"
        path.write_bytes(b"???")
        self.assertTrue(to_data_url(path).startswith("data:image/png;"))

    def test_data_url_passthrough(self):
        self.assertEqual(to_data_url("data:image/png;base64,AAAA"), "data:image/png;base64,AAAA")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            to_data_url(b"???")
