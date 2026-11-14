import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastAPI OpenAPI 3.1 schema to JSON.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("openapi.json"),
        help="Output path for openapi.json",
    )
    args = parser.parse_args()

    schema = app.openapi()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Exported OpenAPI schema ({len(schema.get('paths', {}))} paths) to {args.output}")


if __name__ == "__main__":
    main()
