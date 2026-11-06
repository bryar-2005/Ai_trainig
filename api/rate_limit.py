"""Fixed-window rate limits using atomic Redis counters."""

from __future__ import annotations

import asyncio
import hashlib
import time

import redis.asyncio as redis
from redis.exceptions import RedisError

from api.config import Settings

_SCRIPT = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return n
"""


class RateLimiter:
    def __init__(self, settings: Settings):
        self.window = settings.rate_window_seconds
        self.client = (
            None if settings.redis_url == "memory://" else redis.from_url(settings.redis_url)
        )
        if self.client is None and settings.environment != "test":
            raise ValueError("The in-memory limiter is allowed only in tests")
        self._local: dict[str, tuple[int, int]] = {}
        self._lock = asyncio.Lock()

    async def check(self, identity: str, category: str, limit: int) -> bool:
        bucket = int(time.time() // self.window)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        key = f"sorani:rate:{category}:{digest}:{bucket}"
        if self.client is not None:
            count = await self.client.eval(_SCRIPT, 1, key, self.window + 1)
            return int(count) <= limit
        async with self._lock:
            # Expire old entries opportunistically to bound memory.
            self._local = {k: value for k, value in self._local.items() if value[0] >= bucket - 1}
            _, count = self._local.get(key, (bucket, 0))
            count += 1
            self._local[key] = (bucket, count)
            return count <= limit

    async def healthy(self) -> bool:
        if self.client is None:
            return True
        try:
            return bool(await self.client.ping())
        except RedisError:
            return False

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
