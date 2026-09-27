"""Seekable read-only file over HTTP range requests, with retries.

Lets zipfile read single members of a large remote archive, and lets tarfile
stream a remote tar that survives dropped connections, without storing the
archive on disk.
"""
import io
import time

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=16 << 20, retries=8):
        self.url, self.block, self.retries = url, block, retries
        self.s = requests.Session()
        h = self._get(0, 0)
        self.size = int(h.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self._buf_start, self._buf = 0, b""

    def _get(self, a, b):
        for k in range(self.retries):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
                if r.status_code == 206:
                    return r
                raise IOError(f"HTTP {r.status_code}")
            except (requests.RequestException, IOError):
                if k == self.retries - 1:
                    raise
                time.sleep(2 ** k)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            rel = self.pos - self._buf_start
            if 0 <= rel < len(self._buf):
                chunk = self._buf[rel:rel + n]
            else:
                end = min(self.pos + max(self.block, n) - 1, self.size - 1)
                self._buf_start, self._buf = self.pos, self._get(self.pos, end).content
                continue
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
