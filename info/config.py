from __future__ import annotations

import os

_ENV_BASE_URL_KEY = "XPYQ_LA_BASE_URL"
_DEFAULT_BASE_URL = "https://yihcbkt55p.us-west-2.awsapprunner.com"


def get_base_url() -> str:
    return (os.getenv(_ENV_BASE_URL_KEY) or _DEFAULT_BASE_URL).rstrip("/")
