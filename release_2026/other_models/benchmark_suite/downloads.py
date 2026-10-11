"""Bounded streaming and HTTP range reads. Never extract an entire archive."""

import io
import urllib.request
from contextlib import contextmanager
from pathlib import Path


class Budget:
    def __init__(self, max_bytes=64 << 20):
        if max_bytes < 0:
            raise ValueError("max-download-mb must be nonnegative; 0 means unlimited")
        self.maximum = max_bytes or None
        self.used = 0

    def consume(self, n):
        self.used += n
        if self.maximum is not None and self.used > self.maximum:
            raise ValueError(
                "Download budget exceeded; use local data or explicitly raise --max-download-mb"
            )


class BoundedStream:
    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def read(self, size=-1):
        if self.budget.maximum is not None:
            remaining = self.budget.maximum - self.budget.used + 1
            size = remaining if size < 0 else min(size, remaining)
        data = self.stream.read(size)
        self.budget.consume(len(data))
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@contextmanager
def stream(source, budget):
    if str(source).startswith(("https://", "http://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            yield BoundedStream(response, budget)
    else:
        with open(source, "rb") as response:
            yield response


class HTTPRangeReader(io.RawIOBase):
    """Seekable ZIP/Parquet input; servers must honor ranges (no full fallback)."""

    def __init__(self, url, budget):
        self.url, self.budget, self.position = url, budget, 0
        self.blocks = {}
        req = urllib.request.Request(
            url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206 or not response.headers.get("Content-Range"):
                raise ValueError(
                    "Server does not support HTTP ranges; supply a local archive instead"
                )
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.budget.consume(len(response.read(1)))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else (self.position if whence == 1 else self.size) + offset
        )
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b""
        # Cache small requests such as the central directory/footer. Large columns
        # are returned directly, without a second in-memory copy in the cache.
        key = (self.position, size)
        if key in self.blocks:
            data = self.blocks[key]
        else:
            if (
                self.budget.maximum is not None
                and size > self.budget.maximum - self.budget.used
            ):
                raise ValueError(
                    "Download budget exceeded; use local data or explicitly raise --max-download-mb"
                )
            end = self.position + size - 1
            req = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={self.position}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                expected = f"bytes {self.position}-{end}/"
                if response.status != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(expected):
                    raise ValueError("Invalid HTTP range response")
                data = response.read(size)
            self.budget.consume(len(data))
            if len(data) != size:
                raise IOError("Incomplete HTTP range response")
            if size <= 65536 and len(self.blocks) < 128:
                self.blocks[key] = data
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@contextmanager
def seekable(source, budget):
    if str(source).startswith(("https://", "http://")):
        with HTTPRangeReader(source, budget) as reader:
            yield reader
    else:
        with Path(source).open("rb") as reader:
            yield reader


def json_items(handle, prefix="item"):
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "Install benchmark readers: pip install -r release_2026/other_models/requirements.txt"
        ) from exc
    yield from ijson.items(handle, prefix)


def parquet_rows(url, budget, start=0, stop=None, columns=None):
    import pyarrow.parquet as pq

    with seekable(url, budget) as handle:
        file = pq.ParquetFile(handle)
        offset = 0
        for group in range(file.metadata.num_row_groups):
            end = offset + file.metadata.row_group(group).num_rows
            if stop is not None and offset >= stop:
                break
            if end > start:
                position = offset
                for batch in file.iter_batches(
                    batch_size=128, row_groups=[group], columns=columns
                ):
                    for row in batch.to_pylist():
                        if position >= start and (stop is None or position < stop):
                            yield row
                        position += 1
            offset = end
