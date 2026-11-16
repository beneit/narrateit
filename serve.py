"""Local test server for the NarrateIt demo.

`python -m http.server` ignores HTTP Range requests, so the browser cannot seek
in the MP3 (the seek bar and click-to-seek snap back to 0:00). This is the same
server with Range support added. Standard library only; for local testing, not
for facing the internet.

    python serve.py            # http://localhost:8000
    python serve.py 9000       # another port
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mp3": "audio/mpeg", ".json": "application/json"}

    def send_head(self):
        self._range = None
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not m or os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, end = m.groups()
        if start == "":                       # suffix range: last N bytes
            start, end = max(0, size - int(end or 0)), size - 1
        else:
            start, end = int(start), min(int(end) if end else size - 1, size - 1)
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self._range = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if getattr(self, "_range", None) is None:
            return super().copyfile(source, outputfile)
        left = self._range
        while left > 0:
            chunk = source.read(min(64 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def end_headers(self):
        if getattr(self, "_range", None) is None:
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"NarrateIt demo on http://localhost:{port}  (Ctrl-C to stop)")
    try:
        ThreadingHTTPServer(("", port), RangeHandler).serve_forever()
    except KeyboardInterrupt:
        pass
