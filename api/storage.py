"""Private local-disk and S3-compatible object storage."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from api.config import Settings


def _validated_key(key: str) -> str:
    path = PurePosixPath(key)
    if (
        not key
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in key
        or len(key) > 255
    ):
        raise ValueError("Invalid storage key")
    return str(path)


class StorageBackend(ABC):
    """Server-generated keys address private objects; clients never select a path."""

    @abstractmethod
    def put_file(self, source: Path, key: str, content_type: str) -> None:
        """Store the already size-checked local staging file."""

    @abstractmethod
    def download_file(self, key: str, destination: Path) -> None:
        """Materialize an object for subtitle parsing or model inference."""

    @abstractmethod
    def iter_range(self, key: str, start: int, end: int) -> Iterator[bytes]:
        """Stream an inclusive byte range without buffering the whole object."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object that was allocated by the application."""


class LocalStorage(StorageBackend):
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, key: str) -> Path:
        result = (self.root / _validated_key(key)).resolve()
        if not result.is_relative_to(self.root) or result == self.root:
            raise ValueError("Storage key escapes its root")
        return result

    def put_file(self, source: Path, key: str, content_type: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Exclusive creation avoids overwriting an existing object even on a race.
        created = False
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise

    def download_file(self, key: str, destination: Path) -> None:
        with self._path(key).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)

    def iter_range(self, key: str, start: int, end: int) -> Iterator[bytes]:
        with self._path(key).open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("Stored media ended before the requested range")
                remaining -= len(chunk)
                yield chunk

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, endpoint_url: str | None, region: str):
        import boto3

        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def put_file(self, source: Path, key: str, content_type: str) -> None:
        with source.open("rb") as body:
            self.client.put_object(
                Bucket=self.bucket,
                Key=_validated_key(key),
                Body=body,
                ContentType=content_type,
                IfNoneMatch="*",
            )

    def download_file(self, key: str, destination: Path) -> None:
        # boto3's download_file opens the destination itself. Reject any existing file.
        if destination.exists():
            raise FileExistsError(destination)
        self.client.download_file(self.bucket, _validated_key(key), str(destination))

    def iter_range(self, key: str, start: int, end: int) -> Iterator[bytes]:
        obj = self.client.get_object(
            Bucket=self.bucket, Key=_validated_key(key), Range=f"bytes={start}-{end}"
        )
        body = obj["Body"]
        try:
            yield from body.iter_chunks(chunk_size=1024 * 1024)
        finally:
            body.close()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_validated_key(key))


def create_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_root)
    if not settings.s3_bucket:
        raise ValueError("An S3 bucket is required")
    return S3Storage(settings.s3_bucket, settings.s3_endpoint_url, settings.s3_region)
