"""Shared xpyq client helpers for forecast and optimize layers."""

from __future__ import annotations

import json
import os
import time

_XPYQ_BASE = os.environ.get("XPYQ_BASE", "https://xpyq-lib-production.up.railway.app")


def run_xpyq_code(
    api_key: str,
    code: str,
    *,
    name: str = "quantinel",
    timeout: float = 60.0,
    use_execute: bool = True,
    poll_secs: float = 0.4,
) -> dict:
    """
    Run Python on xpyq hardware.

    Prefers the synchronous /api/v1/execute endpoint (~1-3s on hardware).
    Falls back to /api/v1/compute/runs polling when execute is unavailable.
    """
    import requests

    if not api_key:
        return {"status": "disabled", "stdout": "", "stderr": "missing api key"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if use_execute:
        try:
            resp = requests.post(
                f"{_XPYQ_BASE}/api/v1/execute",
                headers=headers,
                json={"code": code, "backend": "hardware"},
                timeout=timeout,
            )
            if resp.status_code in (401, 403):
                return {"status": "auth_failed", "stdout": "", "stderr": resp.text}
            data = resp.json()
            out = data.get("output") or {}
            stdout = (out.get("stdout") or "").strip()
            stderr = (out.get("stderr") or "").strip()
            if data.get("success") and stdout:
                return {"status": "completed", "stdout": stdout, "stderr": stderr}
            return {
                "status": "failed",
                "stdout": stdout,
                "stderr": stderr or data.get("error") or resp.text,
            }
        except requests.RequestException as exc:
            # Network blip — try queued path below.
            last_error = str(exc)
        else:
            last_error = ""
    else:
        last_error = ""

    try:
        run = requests.post(
            f"{_XPYQ_BASE}/api/v1/compute/runs",
            headers=headers,
            json={"code": code, "name": name},
            timeout=10,
        ).json()
    except requests.RequestException as exc:
        return {"status": "failed", "stdout": "", "stderr": last_error or str(exc)}

    run_id = run.get("run_id") or run.get("id")
    if not run_id:
        return {"status": "failed", "stdout": "", "stderr": json.dumps(run)[:500]}

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = requests.get(
                f"{_XPYQ_BASE}/api/v1/compute/runs/{run_id}",
                headers=headers,
                timeout=10,
            ).json()
        except requests.RequestException as exc:
            return {"status": "failed", "stdout": "", "stderr": str(exc)}

        status = result.get("status", "unknown")
        if status in ("completed", "failed", "timed_out", "cancelled"):
            return {
                "status": status,
                "stdout": (result.get("stdout") or "").strip(),
                "stderr": (result.get("stderr") or "").strip(),
            }
        time.sleep(poll_secs)

    return {"status": "timed_out", "stdout": "", "stderr": "queue wait exceeded timeout"}


def parse_json_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("xpyq stdout did not contain a JSON object")
