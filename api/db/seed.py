"""Administrative account bootstrap, intended for a trusted server operator."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
import uuid

from pydantic import ValidationError
from sqlalchemy import select

from api.config import get_settings
from api.db.models import User
from api.db.session import create_engine, create_session_factory
from api.schemas import RegisterIn
from api.security import hash_password


async def create_admin(email: str, display_name: str, password: str) -> None:
    data = RegisterIn(email=email, display_name=display_name, password=password)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        sessions = create_session_factory(engine)
        async with sessions() as db:
            if await db.scalar(select(User.id).where(User.email == str(data.email).casefold())):
                raise ValueError("An account with this email already exists; no role was changed")
            user = User(
                id=uuid.uuid4(),
                email=str(data.email).casefold(),
                display_name=data.display_name,
                password_hash=await hash_password(data.password),
                role="admin",
            )
            db.add(user)
            await db.commit()
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the first server administrator")
    parser.add_argument("--email", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read one password line from standard input instead of an interactive prompt",
    )
    args = parser.parse_args()
    if args.password_stdin:
        password = sys.stdin.readline().removesuffix("\n").removesuffix("\r")
    else:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Confirm password: "):
            parser.error("Passwords did not match")
    try:
        asyncio.run(create_admin(args.email, args.display_name, password))
    except ValidationError:
        parser.exit(2, "Invalid email, display name or password (12–128 characters required).\n")
    except ValueError as exc:
        parser.exit(2, f"Unable to create administrator: {exc}\n")
    print("Administrator account created.")


if __name__ == "__main__":
    main()
