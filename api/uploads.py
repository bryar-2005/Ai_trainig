"""Stream and bound each uploaded multipart file before it is spooled to disk."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser


class UploadLimitExceeded(MultiPartException):
    """A subtitle or video exceeded its configured per-file limit."""


class ProjectUploadParser(MultiPartParser):
    def __init__(self, request: Request):
        super().__init__(request.headers, request.stream(), max_files=2, max_fields=0)
        self.limits = {
            "subtitle": request.app.state.settings.max_subtitle_bytes,
            "video": request.app.state.settings.max_video_bytes,
        }
        self.seen: set[str] = set()
        self.part_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        field = self._current_part.field_name
        if field not in self.limits or self._current_part.file is None:
            raise MultiPartException("Only subtitle and video files may be uploaded")
        if field in self.seen:
            raise MultiPartException(f"Only one {field} file may be uploaded")
        self.seen.add(field)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.part_bytes += end - start
        field = self._current_part.field_name
        if field in self.limits and self.part_bytes > self.limits[field]:
            raise UploadLimitExceeded(f"The {field} file exceeds its configured size limit")
        super().on_part_data(data, start, end)


async def parse_project_upload(
    request: Request,
) -> AsyncIterator[tuple[UploadFile, UploadFile | None]]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "multipart/form-data":
        raise HTTPException(status_code=415, detail="Expected multipart/form-data")
    parser = ProjectUploadParser(request)
    try:
        form = await parser.parse()
    except UploadLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=exc.message) from exc
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    try:
        subtitle, video = form.get("subtitle"), form.get("video")
        if not isinstance(subtitle, UploadFile) or (
            video is not None and not isinstance(video, UploadFile)
        ):
            raise HTTPException(status_code=422, detail="One subtitle file is required")
        yield subtitle, video
    finally:
        await form.close()
