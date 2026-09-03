"""Distribution and coarse-overlap comparison against the current retail corpus."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autovalue_ml.acquisition.contracts import PriceKind
from autovalue_ml.acquisition.huggingface_dataset import VerifiedHuggingFaceArtifact
from autovalue_ml.acquisition.huggingface_quality import load_candidate_frame
from autovalue_ml.acquisition.scalar_parsing import parse_price_text_cents
from autovalue_ml.acquisition.sources.huggingface_candidates import parse_mileage_text
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars import (
    verify_kaggle_us_sales_cars_artifact_set,
)


def build_retail_candidate_comparison(
    *,
    current_candidate_path: Path,
    current_manifest_path: Path,
    current_review_path: Path,
    yoad_artifact: VerifiedHuggingFaceArtifact,
    carson_artifact: VerifiedHuggingFaceArtifact,
    generated_at: datetime | None = None,
    today: date | None = None,
) -> dict[str, object]:
    """Compare aggregate distributions after every artifact passes its own gate."""
    verify_kaggle_us_sales_cars_artifact_set(
        current_manifest_path,
        current_review_path,
        today=today,
    )
    current = pd.read_csv(
        current_candidate_path,
        usecols=["year", "make", "model", "mileage", "vehicle_status", "price_cents"],
    )
    yoad = load_candidate_frame(yoad_artifact)
    carson = load_candidate_frame(carson_artifact)
    timestamp = datetime.now(UTC) if generated_at is None else generated_at

    current_view = _view(
        year=current["year"],
        make=current["make"],
        model=current["model"],
        mileage=current["mileage"],
        price=current["price_cents"] / 100,
    )
    yoad_view = _view(
        year=yoad["year"],
        make=yoad["manufacturer"],
        model=None,
        mileage=yoad["odometer"],
        price=yoad["price"],
    )
    carson_view = _view(
        year=carson["model_year"],
        make=carson["brand"],
        model=carson["model"],
        mileage=carson["milage"].map(_parse_mileage),
        price=carson["price"].map(_parse_price),
    )

    current_makes = _category_set(current_view["make"])
    current_models = _pair_set(current_view["make"], current_view["model"])
    yoad_makes = _category_set(yoad_view["make"])
    carson_makes = _category_set(carson_view["make"])
    carson_models = _pair_set(carson_view["make"], carson_view["model"])
    return {
        "report_schema_version": 1,
        "report_type": "hugging_face_candidates_vs_current_retail",
        "generated_at": timestamp.isoformat(),
        "current_retail_artifact": {
            "source_id": "kaggle_us_sales_cars_v2",
            "upstream_origin": "historical Cars.com extraction",
            "rows": len(current),
            "manifest_path": current_manifest_path.as_posix(),
            "review_path": current_review_path.as_posix(),
        },
        "distribution_summary": {
            "current_retail": _distribution_summary(current_view),
            "yoad22_craigslist": _distribution_summary(yoad_view),
            "carson_shively": _distribution_summary(carson_view),
        },
        "coverage_comparison": {
            "yoad22_craigslist": {
                "makes_not_present_in_current_retail": sorted(yoad_makes - current_makes),
                "existing_make_overlap_count": len(yoad_makes & current_makes),
                "model_coverage": "unavailable",
                "richer_fields": [
                    "condition",
                    "cylinders",
                    "fuel",
                    "title_status",
                    "transmission",
                    "drive",
                    "vehicle_type",
                    "state",
                ],
            },
            "carson_shively": {
                "makes_not_present_in_current_retail": sorted(carson_makes - current_makes),
                "existing_make_overlap_count": len(carson_makes & current_makes),
                "make_model_pairs_not_present_in_current_retail": len(
                    carson_models - current_models
                ),
                "existing_make_model_pair_overlap_count": len(carson_models & current_models),
                "richer_fields": [
                    "engine",
                    "transmission",
                    "fuel_type",
                    "accident",
                    "clean_title",
                ],
            },
        },
        "coarse_cross_source_key_collisions": {
            "yoad22_craigslist": _collision_report(
                current_view,
                yoad_view,
                include_model=False,
                warning=(
                    "Year/make/mileage/price collisions are only possible-duplicate signals; "
                    "Yoad omits model, VIN, listing ID, and URL."
                ),
            ),
            "carson_shively": _collision_report(
                current_view,
                carson_view,
                include_model=True,
                warning=(
                    "Year/make/model/mileage/price collisions are possible-duplicate signals, "
                    "not proof, because neither comparison exposes a common stable listing ID."
                ),
            ),
        },
        "domain_shift_findings": [
            "Current retail is dominated by 2023 New/Used/Certified Cars.com snapshots and has "
            "substantial structurally missing mileage for New vehicles.",
            "Yoad is an older, lower-priced Craigslist used-vehicle population with complete "
            "odometer/state coverage but no model field; it changes the target domain rather than "
            "simply filling current-retail gaps.",
            "Carson is smaller and more luxury-heavy, with detailed model/engine/accident/title "
            "fields, but its geography, USD semantics, timestamp, and upstream origin are "
            "unresolved.",
        ],
        "feature_strategy": {
            "broad_coverage_model": ["year", "make", "mileage"],
            "rich_feature_candidate": [
                "model",
                "engine",
                "transmission",
                "fuel_type",
                "accident_status",
                "title_status",
            ],
            "decision": (
                "Do not force rich fields into the current model. If permissions are resolved, "
                "evaluate a separate rich-feature experiment against the broad model."
            ),
        },
        "merge_decision": {
            "merged": False,
            "experiment_b_carson": "blocked_pending_provenance_and_usd_us_scope",
            "experiment_c_yoad": "approved_controlled_batch_experiment_only",
            "experiment_d_combined": "blocked_until_each_source_is_independently_approved",
        },
    }


def write_comparison_report(report: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _view(*, year: Any, make: Any, model: Any | None, mileage: Any, price: Any) -> Any:
    frame = pd.DataFrame(
        {
            "year": pd.to_numeric(year, errors="coerce"),
            "make": make.astype("string").str.strip().str.casefold(),
            "mileage": pd.to_numeric(mileage, errors="coerce"),
            "price": pd.to_numeric(price, errors="coerce"),
        }
    )
    frame["model"] = (
        model.astype("string").str.strip().str.casefold()
        if model is not None
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    return frame


def _distribution_summary(frame: Any) -> dict[str, object]:
    return {
        "rows": len(frame),
        "price_usd": _quantiles(frame["price"]),
        "year": _quantiles(frame["year"]),
        "mileage_miles": _quantiles(frame["mileage"]),
        "mileage_present_percentage": _percentage(int(frame["mileage"].notna().sum()), len(frame)),
        "distinct_makes": int(frame["make"].nunique()),
        "distinct_make_model_pairs": len(_pair_set(frame["make"], frame["model"])),
        "above_75000_usd_percentage": _percentage(int((frame["price"] > 75_000).sum()), len(frame)),
        "top_makes": [
            {"make": str(value), "count": int(count)}
            for value, count in frame["make"].value_counts().head(15).items()
        ],
    }


def _collision_report(
    current: Any,
    candidate: Any,
    *,
    include_model: bool,
    warning: str,
) -> dict[str, object]:
    columns = ["year", "make", "mileage", "price"]
    if include_model:
        columns.insert(2, "model")
    current_keys = _complete_key_set(current, columns)
    candidate_keys = _complete_key_set(candidate, columns)
    shared = current_keys & candidate_keys
    return {
        "key_fields": columns,
        "current_unique_complete_keys": len(current_keys),
        "candidate_unique_complete_keys": len(candidate_keys),
        "shared_unique_keys": len(shared),
        "warning": warning,
        "training_action": "group_or_deduplicate_before_any_cross-source_split",
    }


def _complete_key_set(frame: Any, columns: list[str]) -> set[tuple[object, ...]]:
    complete = frame[columns].dropna()
    return {
        tuple(_key_value(value) for value in row)
        for row in complete.itertuples(index=False, name=None)
    }


def _category_set(series: Any) -> set[str]:
    return {str(value) for value in series.dropna().unique() if str(value).strip()}


def _pair_set(make: Any, model: Any) -> set[tuple[str, str]]:
    frame = pd.DataFrame({"make": make, "model": model}).dropna()
    return {
        (str(row.make), str(row.model))
        for row in frame.itertuples(index=False)
        if str(row.make).strip() and str(row.model).strip()
    }


def _quantiles(series: Any) -> dict[str, float | int | None]:
    valid = series.dropna()
    if valid.empty:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    values = valid.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(valid.size),
        "min": float(valid.min()),
        "p25": float(values.loc[0.25]),
        "median": float(values.loc[0.5]),
        "p75": float(values.loc[0.75]),
        "max": float(valid.max()),
    }


def _parse_price(value: object) -> float:
    try:
        return (
            parse_price_text_cents(str(value), expected_currency="USD", price_kind=PriceKind.ASKING)
            / 100
        )
    except ValueError:
        return float("nan")


def _parse_mileage(value: object) -> float:
    try:
        parsed = parse_mileage_text(value)
    except ValueError:
        return float("nan")
    return float("nan") if parsed is None else float(parsed)


def _key_value(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _percentage(count: int, total: int) -> float:
    return round(100 * count / total, 6) if total else 0.0


__all__ = ["build_retail_candidate_comparison", "write_comparison_report"]
