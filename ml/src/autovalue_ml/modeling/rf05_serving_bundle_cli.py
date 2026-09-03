"""CLI for the governed RF05 deployment-only reconstruction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .rf05_serving_bundle import RF05ServingBundleError, reconstruct_rf05_serving_bundle


def main(argv: Sequence[str] | None = None) -> int:
    """Reconstruct one authenticated private serving bundle or fail closed."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    root = Path(cast(str, arguments.project_root))
    try:
        result = reconstruct_rf05_serving_bundle(
            project_root=root,
            bundle_dir=Path("models/retail-rf05-v1"),
            report_path=Path("docs/experiments/retail-rf05-serving-reconstruction-v1.json"),
            golden_fixture_path=Path("tests/fixtures/retail-rf05-v1.golden.json"),
            force=cast(bool, arguments.force),
        )
    except (OSError, RF05ServingBundleError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        "RF05 serving reconstruction passed | "
        f"model_sha256={result.model_sha256} | "
        f"manifest_sha256={result.manifest_sha256} | "
        f"bundle_sha256={result.bundle_sha256} | "
        "publication=deployment-private",
        flush=True,
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autovalue_ml.modeling.rf05_serving_bundle_cli",
        description=(
            "Deterministically reconstruct and authenticate the already-frozen retail RF05 "
            "estimator for private serving. This performs no model selection or tuning."
        ),
    )
    parser.add_argument("--project-root", required=True, metavar="PATH")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the known bundle and reconstruction outputs",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
