"""Subtitle export without automatic retiming, plus project statistics."""

from __future__ import annotations

import json
import uuid
from typing import Literal

import pysubs2
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import project_for
from api.db.models import Cue, User
from api.db.session import get_session
from api.security import current_user
from api.serializers import cue_json
from ml.inference.text import LRI, PDI, isolate_ltr, normalize_target

router = APIRouter(prefix="/api", tags=["export"])
_METADATA = (
    "Machine-assisted translation; subtitles may contain human revisions and require review."
)


def _timestamp(ms: int, separator: str) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}"


def _display(cue: Cue, direction: str, bilingual: bool) -> str:
    target = cue.human_text if cue.human_text is not None else cue.machine_text
    target = normalize_target(target, direction)
    if direction == "en-ckb":
        target = isolate_ltr(target)
    if not bilingual or not cue.source_text:
        return target
    source = cue.source_text
    if direction == "en-ckb":
        source = LRI + source + PDI
    else:
        source = isolate_ltr(normalize_target(source, "en-ckb"))
    return f"{target}\n{source}".strip()


def render_export(
    cues: list[Cue],
    direction: str,
    format: str,
    bilingual: bool,
    *,
    ass_font: str | None = None,
    ass_font_size: int = 20,
    ass_color: str = "#FFFFFF",
    ass_margin: int = 10,
) -> str:
    if format == "json":
        return (
            json.dumps(
                jsonable_encoder(
                    {
                        "metadata": {"origin": _METADATA, "direction": direction},
                        "cues": [cue_json(cue) for cue in cues],
                    }
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    if format == "txt":
        return (
            "# "
            + _METADATA
            + "\n\n"
            + "\n".join(_display(cue, direction, bilingual) for cue in cues)
            + "\n"
        )
    if format in {"srt", "vtt"}:
        header = (
            f"<!-- {_METADATA} -->\n\n" if format == "srt" else f"WEBVTT\n\nNOTE {_METADATA}\n\n"
        )
        separator = "," if format == "srt" else "."
        blocks = []
        for index, cue in enumerate(cues, 1):
            timing = (
                f"{_timestamp(cue.start_ms, separator)} --> {_timestamp(cue.end_ms, separator)}"
            )
            blocks.append(f"{index}\n{timing}\n{_display(cue, direction, bilingual)}")
        return header + "\n\n".join(blocks) + "\n"
    if format == "ass":
        # ASS represents time in centiseconds. Reject values it cannot express
        # exactly instead of silently shifting a cue to a different timestamp.
        if any(cue.start_ms % 10 or cue.end_ms % 10 for cue in cues):
            raise ValueError("ASS cannot represent one or more timestamps exactly; use SRT or VTT")
        subtitle = pysubs2.SSAFile()
        subtitle.info["Title"] = _METADATA
        subtitle.info["ScriptType"] = "v4.00+"
        style = subtitle.styles["Default"]
        style.fontname = ass_font or ("Noto Naskh Arabic" if direction == "en-ckb" else "Arial")
        style.fontsize = ass_font_size
        style.primarycolor = pysubs2.Color(
            int(ass_color[1:3], 16), int(ass_color[3:5], 16), int(ass_color[5:7], 16)
        )
        style.marginv = ass_margin
        for cue in cues:
            subtitle.append(
                pysubs2.SSAEvent(
                    start=cue.start_ms,
                    end=cue.end_ms,
                    text=_display(cue, direction, bilingual).replace("\n", r"\N"),
                )
            )
        # ASS has centisecond timestamp precision; pysubs2 rounds milliseconds.
        return subtitle.to_string(format_="ass")
    raise ValueError("Unsupported export format")


@router.get("/projects/{project_id}/export")
async def export_subtitles(
    project_id: uuid.UUID,
    format: Literal["srt", "vtt", "ass", "json", "txt"] = "srt",
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8",
    bilingual: bool = False,
    approved_only: bool = False,
    ass_font: str | None = Query(
        default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9 _-]+$"
    ),
    ass_font_size: int = Query(default=20, ge=12, le=100),
    ass_color: str = Query(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$"),
    ass_margin: int = Query(default=10, ge=0, le=500),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    query = select(Cue).where(Cue.project_id == project.id).order_by(Cue.cue_index)
    if approved_only:
        query = query.where(Cue.status == "approved")
    cues = (await db.scalars(query)).all()
    if not cues:
        raise HTTPException(status_code=422, detail="There are no cues matching this export")
    try:
        body = render_export(
            cues,
            project.direction,
            format,
            bilingual,
            ass_font=ass_font,
            ass_font_size=ass_font_size,
            ass_color=ass_color,
            ass_margin=ass_margin,
        ).encode(encoding)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    types = {
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "ass": "text/plain",
        "json": "application/json",
        "txt": "text/plain",
    }
    return Response(
        content=body,
        media_type=types[format],
        headers={
            "Content-Disposition": (
                f'attachment; filename="subtitles-{str(project.id)[:8]}.{format}"'
            ),
            "Cache-Control": "private, no-store",
            "X-Subtitle-Origin": "machine-assisted; human-review-may-apply",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/projects/{project_id}/stats")
async def project_stats(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    cues = (await db.scalars(select(Cue).where(Cue.project_id == project.id))).all()
    count = len(cues)
    scored = [cue.confidence_score for cue in cues if cue.confidence_score is not None]
    edited = sum(cue.human_text is not None and cue.human_text != cue.machine_text for cue in cues)
    return {
        "cue_count": count,
        "edited_count": edited,
        "edit_rate": edited / count if count else 0,
        "approved_count": sum(cue.status == "approved" for cue in cues),
        "translated_count": sum(
            bool((cue.human_text if cue.human_text is not None else cue.machine_text).strip())
            for cue in cues
        ),
        "average_confidence": sum(scored) / len(scored) if scored else None,
        "confidence_histogram": [
            sum(
                lower <= score < upper if upper < 100 else lower <= score <= upper
                for score in scored
            )
            for lower, upper in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
        ],
        "unscored_count": count - len(scored),
        "cps_violations": sum(cue.cps > project.target_cps for cue in cues),
        "hard_cps_violations": sum(cue.cps > project.hard_cps for cue in cues),
        "flagged_count": sum(cue.flagged for cue in cues),
    }
