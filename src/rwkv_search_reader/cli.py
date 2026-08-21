from __future__ import annotations

import argparse
import json
import sys

from .models import SearchReaderError
from .pipeline import SearchReaderPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rwkv-search-reader",
        description="Always search first, then let RWKV read the evidence.",
    )
    parser.add_argument("question", nargs="*", help="question to search and answer")
    parser.add_argument(
        "--json", action="store_true", help="print the complete JSON result"
    )
    parser.add_argument(
        "--show-evidence", action="store_true", help="print evidence after the answer"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("问题> ").strip()
        except EOFError:
            question = ""
    if not question:
        raise SystemExit("问题不能为空")
    try:
        result = SearchReaderPipeline().answer(question)
    except SearchReaderError as exc:
        print(f"[{exc.stage}] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print(result.answer)
    if args.show_evidence:
        print("\n来源：")
        for source in result.sources:
            print(f"- {source.title}: {source.url}")


if __name__ == "__main__":
    main()
