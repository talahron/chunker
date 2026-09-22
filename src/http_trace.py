"""
Process-wide logging of outbound aiohttp requests (used by google-genai for
async Gemini calls). Injects an aiohttp TraceConfig into every ClientSession
at construction time; nothing on the request/SSL path is modified.

Call install() once from main.py before any Gemini client is created.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger("src.http_trace")

_installed = False


def install(only_hosts: tuple[str, ...] = ("googleapis.com",)) -> None:
    global _installed
    if _installed:
        return
    _installed = True

    try:
        import aiohttp
    except ImportError:
        logger.info("aiohttp not installed; nothing to trace")
        return

    async def on_request_start(session, ctx, params):
        url = str(params.url)
        if any(h in url for h in only_hosts):
            logger.info("GEMINI REQUEST %s %s", params.method, url)

    async def on_request_end(session, ctx, params):
        url = str(params.url)
        if any(h in url for h in only_hosts):
            logger.info("GEMINI RESPONSE %s %s -> %s",
                        params.method, url, params.response.status)

    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(on_request_start)
    trace.on_request_end.append(on_request_end)

    original_init = aiohttp.ClientSession.__init__

    @functools.wraps(original_init)
    def traced_init(self, *args, **kwargs):
        configs = list(kwargs.get("trace_configs") or [])
        configs.append(trace)
        kwargs["trace_configs"] = configs
        original_init(self, *args, **kwargs)

    aiohttp.ClientSession.__init__ = traced_init
    logger.info("aiohttp trace_configs injection installed (aiohttp %s)", aiohttp.__version__)
