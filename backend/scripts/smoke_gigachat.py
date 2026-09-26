"""Opt-in single real GigaChat planning call; does not contact Parser or download PDFs."""

import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic

from backend.app.config import get_settings
from backend.app.llm.base import LlmError
from backend.app.llm.gigachat import GigaChatClient, GigaChatQueryPlanner


async def run_smoke() -> dict:
    settings = get_settings()
    result = {"ok": False, "provider": "gigachat", "model": settings.gigachat_plan_model,
              "key_present": bool(settings.gigachat_auth_key)}
    if not settings.gigachat_auth_key:
        result["error"] = "GIGACHAT_AUTH_KEY is not configured"
        return result
    planner = GigaChatQueryPlanner(
        GigaChatClient(settings.gigachat_auth_key, scope=settings.gigachat_scope,
                       ca_bundle=settings.gigachat_ca_bundle),
        model=settings.gigachat_plan_model,
    )
    started = monotonic()
    try:
        plan = await asyncio.wait_for(planner.plan(
            "Поставщик передал товар покупателю, но покупатель не оплатил его. "
            "Поставщик требует взыскать долг по договору поставки."),
            settings.plan_timeout_seconds)
        result["ok"] = bool(plan.queries)
        result["query_count"] = len(plan.queries)
    except TimeoutError:
        result["error"] = "Planning timed out"
    except LlmError as exc:
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = type(exc).__name__
    result["planning_seconds"] = round(monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_smoke())
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
