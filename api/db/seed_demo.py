"""Explicitly create a synthetic, untranslated demo in a non-production workspace."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import sys
import tempfile
import uuid
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from api.config import Settings, get_settings
from api.db.models import Cue, MediaFile, Project, User
from api.db.session import create_engine, create_session_factory
from api.schemas import RegisterIn
from api.security import hash_password
from api.storage import create_storage
from ml.data.loaders import load_subtitles

# Original short fixture text, not extracted from a movie or a copyrighted corpus.
DEMO_SRT = (
    b"1\n00:00:01,000 --> 00:00:03,000\nThe lantern is lit.\n\n"
    b"2\n00:00:03,500 --> 00:00:06,500\nWe can start the journey.\n\n"
    b"3\n00:00:07,000 --> 00:00:09,500\nSee you at sunrise.\n"
)


async def create_demo(
    email: str, display_name: str, password: str, settings: Settings
) -> uuid.UUID:
    """Create a new ordinary user and one untranslated project, never modify an account."""
    if settings.environment == "production":
        raise ValueError("Demo seeding is disabled in production")
    data = RegisterIn(email=email, display_name=display_name, password=password)
    engine = create_engine(settings.database_url)
    storage = create_storage(settings)
    root = settings.storage_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    project_id, user_id = uuid.uuid4(), uuid.uuid4()
    key = f"{project_id}/{uuid.uuid4().hex}.srt"
    stored = False
    try:
        sessions = create_session_factory(engine)
        async with sessions() as db:
            if await db.scalar(select(User.id).where(User.email == str(data.email).casefold())):
                raise ValueError("Account already exists; nothing was changed")
            db.add(
                User(
                    id=user_id,
                    email=str(data.email).casefold(),
                    display_name=data.display_name,
                    password_hash=await hash_password(data.password),
                    role="user",
                    preferences={},
                )
            )
            await db.flush()
            db.add(
                Project(
                    id=project_id,
                    owner_id=user_id,
                    name="Synthetic demo project",
                    description=(
                        "Original synthetic example; untranslated and not a quality benchmark."
                    ),
                    direction="en-ckb",
                    status="draft",
                )
            )
            await db.flush()
            with tempfile.TemporaryDirectory(prefix=".demo-", dir=root) as temp:
                path = Path(temp) / "synthetic-demo.srt"
                path.write_bytes(DEMO_SRT)
                parsed = load_subtitles(path)
                await asyncio.to_thread(storage.put_file, path, key, "application/x-subrip")
                stored = True
                db.add(
                    MediaFile(
                        id=uuid.uuid4(),
                        project_id=project_id,
                        kind="subtitle",
                        storage_key=key,
                        original_name="synthetic-demo.srt",
                        content_type="application/x-subrip",
                        size_bytes=len(DEMO_SRT),
                        sha256=hashlib.sha256(DEMO_SRT).hexdigest(),
                    )
                )
                for cue in parsed:
                    db.add(
                        Cue(
                            id=uuid.uuid4(),
                            project_id=project_id,
                            cue_index=cue.index,
                            original_start_ms=cue.start_ms,
                            original_end_ms=cue.end_ms,
                            start_ms=cue.start_ms,
                            end_ms=cue.end_ms,
                            source_cue_ids=[cue.index],
                            sentence_source_cue_ids=[cue.index],
                            source_text=cue.text,
                            normalized_source_text=cue.text,
                            speaker=cue.speaker,
                            speaker_labels=list(cue.speaker_labels),
                            sound_effects=list(cue.sound_effects),
                            style=cue.style,
                            machine_text="",
                            human_text=None,
                            qa_warnings=["untranslated"],
                            flagged=True,
                            status="unreviewed",
                        )
                    )
                await db.commit()
                stored = False
        return project_id
    except IntegrityError as exc:
        raise ValueError(
            "Account already exists or demo creation conflicted; nothing was changed"
        ) from exc
    finally:
        if stored:
            await asyncio.to_thread(storage.delete, key)
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synthetic untranslated demo (non-production only)"
    )
    parser.add_argument(
        "--create-synthetic-demo",
        action="store_true",
        help="Explicitly opt in to creating the demo",
    )
    parser.add_argument("--email", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--password-stdin", action="store_true", help="Read one password line from standard input"
    )
    args = parser.parse_args()
    if not args.create_synthetic_demo:
        parser.error("Pass --create-synthetic-demo to create this optional account and project")
    if args.password_stdin:
        password = sys.stdin.readline().removesuffix("\n").removesuffix("\r")
    else:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Confirm password: "):
            parser.error("Passwords did not match")
    try:
        project_id = asyncio.run(
            create_demo(args.email, args.display_name, password, get_settings())
        )
    except ValidationError:
        parser.exit(2, "Invalid email, display name or password (12–128 characters required).\n")
    except ValueError as exc:
        parser.exit(2, f"Unable to create synthetic demo: {exc}\n")
    print(f"Created an untranslated synthetic demo project ({project_id}); no model was installed.")


if __name__ == "__main__":
    main()
