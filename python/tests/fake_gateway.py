import json
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread


class FakeGateway:
    def __init__(self):
        self.requests = []
        self._replies = defaultdict(deque)

    def reply(self, method: str, path: str, status: int, body: dict | str):
        self._replies[method, path].append((status, body))

    def __enter__(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                content = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                valid_type = self.headers.get("Content-Type") == "application/json"
                gateway.requests.append({
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(content) if content and valid_type else None,
                })

                replies = gateway._replies[self.command, self.path]
                if self.command == "POST" and not valid_type:
                    status, body = 415, "Expected application/json"
                elif replies:
                    status, body = replies.popleft()
                elif self.command == "GET" and self.path == "/health":
                    status, body = 200, {
                        "status": "ok", "service": "noullama", "api_port": 0, "model_ready": True,
                    }
                else:
                    status, body = 404, "No queued response"

                content_type = "application/json" if isinstance(body, dict) else "text/plain"
                payload = (json.dumps(body) if isinstance(body, dict) else body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01})
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()
