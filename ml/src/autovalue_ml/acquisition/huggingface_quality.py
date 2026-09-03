"""Aggregate-only quality reports for governed Hugging Face candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from autovalue_ml.acquisition.contracts import PriceKind
from autovalue_ml.acquisition.huggingface_dataset import (
    ApprovalStatus,
    HuggingFaceDatasetError,
    SourceOrigin,
    VerifiedHuggingFaceArtifact,
    assess_source_overlap,
    build_huggingface_provenance,
)
from autovalue_ml.acquisition.scalar_parsing import parse_price_text_cents
from autovalue_ml.acquisition.sources.huggingface_candidates import (
    CARSON_SHIVELY_SPEC,
    YOAD22_CRAIGSLIST_SPEC,
    parse_mileage_text,
)
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import US_50_PLUS_DC

QUALITY_REPORT_SCHEMA_VERSION: Final = 1
CURRENT_RETAIL_ORIGIN: Final = SourceOrigin(
    source_id="kaggle_us_sales_cars_v2",
    upstream_families=frozenset({"cars_com_2023_historical_listings"}),
    provenance_known=True,
)
YOAD_ORIGIN: Final = SourceOrigin(
    source_id=YOAD22_CRAIGSLIST_SPEC.source_id,
    upstream_families=frozenset({"austin_reese_craigslist_vehicles_v10"}),
    provenance_known=True,
)
CARSON_ORIGIN: Final = SourceOrigin(
    source_id=CARSON_SHIVELY_SPEC.source_id,
    upstream_families=frozenset(),
    provenance_known=False,
)

_YOAD_COLUMNS: Final = (
    "price",
    "year",
    "manufacturer",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
    "car_age",
)
_CARSON_COLUMNS: Final = (
    "brand",
    "model",
    "model_year",
    "milage",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
    "clean_title",
    "price",
)


def load_candidate_frame(artifact: VerifiedHuggingFaceArtifact) -> Any:
    """Load only a checksum-verified reviewed artifact using a bounded format path."""
    suffix = artifact.path.suffix.casefold()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(artifact.path)
        elif suffix == ".parquet":
            frame = pd.read_parquet(artifact.path)
        else:
            raise HuggingFaceDatasetError("unsupported candidate artifact format")
    except (OSError, UnicodeError, ValueError) as error:
        raise HuggingFaceDatasetError("candidate artifact cannot be parsed") from error
    if len(frame) != artifact.spec.expected_row_count:
        raise HuggingFaceDatasetError("candidate row count does not match the reviewed revision")
    return frame


def profile_huggingface_candidate(
    artifact: VerifiedHuggingFaceArtifact,
    *,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Produce a deterministic, aggregate-only quality and compatibility report."""
    frame = load_candidate_frame(artifact)
    spec = artifact.spec
    expected_columns: tuple[str, ...]
    if spec.source_id == YOAD22_CRAIGSLIST_SPEC.source_id:
        report = _profile_yoad(frame)
        expected_columns = _YOAD_COLUMNS
        origin = YOAD_ORIGIN
        mapping = _yoad_mapping()
    elif spec.source_id == CARSON_SHIVELY_SPEC.source_id:
        report = _profile_carson(frame)
        expected_columns = _CARSON_COLUMNS
        origin = CARSON_ORIGIN
        mapping = _carson_mapping()
    else:
        raise HuggingFaceDatasetError("candidate source is not reviewed")
    if tuple(str(column) for column in frame.columns) != expected_columns:
        raise HuggingFaceDatasetError("candidate schema does not match the reviewed mapping")

    overlap = assess_source_overlap(origin, CURRENT_RETAIL_ORIGIN)
    accepted = int(report["row_accounting"]["accepted_rows"])  # type: ignore[index]
    rejected = int(report["row_accounting"]["rejected_rows"])  # type: ignore[index]
    duplicates = int(report["row_accounting"]["exact_duplicate_rows"])  # type: ignore[index]
    provenance = build_huggingface_provenance(
        artifact,
        raw_row_count=len(frame),
        accepted_row_count=accepted,
        rejected_row_count=rejected,
        duplicate_row_count=duplicates,
    )
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    return {
        "report_schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "report_type": "hugging_face_candidate_quality_and_compatibility",
        "generated_at": timestamp.isoformat(),
        "candidate": {
            "source_id": spec.source_id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "file_path": spec.file_path.as_posix(),
            "config": spec.config,
            "split": spec.split,
            "required_product_market_country": spec.market_country,
            "required_product_currency": spec.currency,
            "source_market_scope_status": (
                "reviewed_us_craigslist_source"
                if spec.source_id == YOAD22_CRAIGSLIST_SPEC.source_id
                else "unverified"
            ),
            "source_currency_status": (
                "reviewed_us_craigslist_asking_price_semantics"
                if spec.source_id == YOAD22_CRAIGSLIST_SPEC.source_id
                else "dollar_strings_present_but_usd_semantics_unverified"
            ),
            "target_semantics": "historical_used_vehicle_advertised_asking_price",
        },
        "artifact": {
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "cache_hit_at_profile_time": artifact.cache_hit,
        },
        "license_and_permissions": {
            "declared_license": spec.declared_license,
            "license_url": spec.license_url,
            "attribution": spec.attribution,
            "usage_restrictions": list(spec.usage_restrictions),
            "acquisition": spec.approvals.acquisition.value,
            "batch_training": spec.approvals.batch_training.value,
            "online_learning": spec.approvals.online_learning.value,
            "acquisition_evidence": spec.approvals.acquisition_evidence,
            "batch_training_evidence": spec.approvals.batch_training_evidence,
            "online_learning_evidence": spec.approvals.online_learning_evidence,
        },
        "source_provenance": {
            "upstream_source": spec.upstream_source,
            "known_upstream_families": sorted(origin.upstream_families),
            "provenance_known": origin.provenance_known,
        },
        "overlap_with_current_retail": {
            "current_retail_source_id": CURRENT_RETAIL_ORIGIN.source_id,
            "risk": overlap.risk.value,
            "shared_upstream_families": list(overlap.shared_upstream_families),
            "merge_blocked_by_overlap_gate": overlap.merge_blocked,
            "rationale": overlap.rationale,
            "current_retail_origin": "historical Cars.com extraction, not Craigslist",
        },
        "schema": {
            "source_columns": list(expected_columns),
            "mapping_version": spec.schema_mapping_version,
            "canonical_mapping": mapping,
            "source_id_available_for_audit_only": True,
            "source_id_is_predictive_feature": False,
        },
        "provenance": {
            "repo_id": provenance.repo_id,
            "revision": provenance.revision,
            "file_path": provenance.file_path,
            "config": provenance.config,
            "split": provenance.split,
            "acquired_at": provenance.acquired_at.isoformat(),
            "raw_row_count": provenance.raw_row_count,
            "accepted_row_count": provenance.accepted_row_count,
            "rejected_row_count": provenance.rejected_row_count,
            "duplicate_row_count": provenance.duplicate_row_count,
            "schema_mapping_version": provenance.schema_mapping_version,
            "acquisition_approval": provenance.acquisition_approval.value,
            "batch_training_approval": provenance.batch_training_approval.value,
            "online_learning_approval": provenance.online_learning_approval.value,
        },
        "quality": report,
        "promotion_decision": {
            "merged_into_existing_training_data": False,
            "training_experiment_started": (
                spec.source_id == YOAD22_CRAIGSLIST_SPEC.source_id
                and spec.approvals.batch_training is ApprovalStatus.APPROVED
            ),
            "reason": spec.approvals.batch_training_evidence,
        },
    }


def write_quality_report(report: Mapping[str, object], destination: Path) -> None:
    """Persist aggregate JSON without row-level source values."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")


def _profile_yoad(frame: Any) -> dict[str, object]:
    price = pd.to_numeric(frame["price"], errors="coerce")
    year = pd.to_numeric(frame["year"], errors="coerce")
    mileage = pd.to_numeric(frame["odometer"], errors="coerce")
    make_present = ~_missing_mask(frame["manufacturer"])
    state_text = frame["state"].astype("string").str.upper()
    state_valid = _missing_mask(frame["state"]) | state_text.isin(US_50_PLUS_DC)
    valid = (
        price.notna()
        & (price > 0)
        & year.between(1886, 2028)
        & make_present
        & (mileage.isna() | (mileage >= 0))
        & state_valid
    )
    return _profile_common(
        frame,
        price=price,
        year=year,
        mileage=mileage,
        valid=valid,
        make_column="manufacturer",
        model_column=None,
        geography_column="state",
        accident_column=None,
        title_column="title_status",
        condition_column="condition",
        source_notes=(
            "The cleaned artifact has no model, VIN, stable listing ID, URL, or row timestamp.",
            "The source card documents prior price/year/odometer filtering and IQR outlier "
            "removal; "
            "this report does not silently apply another target filter.",
            "Model coverage is zero, limiting exact-model leakage detection and rich-model use.",
        ),
    )


def _profile_carson(frame: Any) -> dict[str, object]:
    price = frame["price"].map(_safe_price_dollars)
    year = pd.to_numeric(frame["model_year"], errors="coerce")
    mileage = frame["milage"].map(_safe_mileage)
    valid = (
        price.notna()
        & (price > 0)
        & year.between(1886, 2028)
        & ~_missing_mask(frame["brand"])
        & ~_missing_mask(frame["model"])
        & (mileage.isna() | (mileage >= 0))
    )
    result = _profile_common(
        frame,
        price=price,
        year=year,
        mileage=mileage,
        valid=valid,
        make_column="brand",
        model_column="model",
        geography_column=None,
        accident_column="accident",
        title_column="clean_title",
        condition_column=None,
        source_notes=(
            "The bronze artifact contains 4,009 original-format rows, not 7,970 independent rows.",
            "The repository's 3,961-row silver file is a transformed/filtered layer of the bronze "
            "data and must not be concatenated with bronze as new observations.",
            "No row geography or observation timestamp is present; U.S. scope and upstream origin "
            "remain unverified, so batch and online training are blocked.",
        ),
    )
    result["repository_layer_audit"] = {
        "bronze_rows": 4_009,
        "silver_rows": 3_961,
        "silver_file_path": "data/silver/silver.parquet",
        "silver_size_bytes": 82_060,
        "silver_sha256": ("26ed9d0d159ece7ab68b152e1355503ecd6bba46604523d21fe59fe506a7ffa7"),
        "naive_combined_rows": 7_970,
        "independent_observation_interpretation": "not_supported",
    }
    return result


def _profile_common(
    frame: Any,
    *,
    price: Any,
    year: Any,
    mileage: Any,
    valid: Any,
    make_column: str,
    model_column: str | None,
    geography_column: str | None,
    accident_column: str | None,
    title_column: str | None,
    condition_column: str | None,
    source_notes: tuple[str, ...],
) -> dict[str, object]:
    duplicate_mask = frame.duplicated(keep="first")
    accepted_mask = valid & ~duplicate_mask
    rejected_mask = ~valid & ~duplicate_mask
    row_count = len(frame)
    accepted = int(accepted_mask.sum())
    rejected = int(rejected_mask.sum())
    duplicates = int(duplicate_mask.sum())
    if accepted + rejected + duplicates != row_count:
        raise HuggingFaceDatasetError("candidate row accounting failed")

    price_valid = price[price.notna() & (price > 0)]
    year_valid = year[year.notna()]
    mileage_valid = mileage[mileage.notna() & (mileage >= 0)]
    return {
        "row_accounting": {
            "raw_rows": row_count,
            "accepted_rows": accepted,
            "rejected_rows": rejected,
            "exact_duplicate_rows": duplicates,
            "acceptance_is_training_approval": False,
        },
        "missing_value_percentages": {
            str(column): _percent(int(_missing_mask(frame[column]).sum()), row_count)
            for column in frame.columns
        },
        "price_distribution_usd": _numeric_distribution(price_valid),
        "price_quality_flags": {
            "nonpositive_or_unparseable": int((price.isna() | (price <= 0)).sum()),
            "below_500": int(((price > 0) & (price < 500)).sum()),
            "below_2000": int(((price > 0) & (price < 2_000)).sum()),
            "above_250000": int((price > 250_000).sum()),
            "above_1000000": int((price > 1_000_000).sum()),
            "most_repeated_prices": _top_numeric_counts(price_valid, limit=10),
            "filtering_decision": "No additional price rows removed by the quality report.",
        },
        "year_coverage": {
            **_numeric_distribution(year_valid),
            "distinct_years": int(year_valid.nunique()),
            "impossible_or_future_after_2028": int(((year < 1886) | (year > 2028)).sum()),
        },
        "make_coverage": _categorical_coverage(frame, make_column),
        "model_coverage": (
            _categorical_coverage(frame, model_column)
            if model_column is not None
            else _unsupported_coverage(row_count, "source artifact has no model column")
        ),
        "mileage_coverage": {
            **_numeric_distribution(mileage_valid),
            "present_rows": int(mileage_valid.size),
            "present_percentage": _percent(int(mileage_valid.size), row_count),
            "negative_or_unparseable_nonmissing": int(
                ((~_missing_mask_from_values(frame, mileage)) & mileage.isna()).sum()
            ),
            "above_500000": int((mileage > 500_000).sum()),
        },
        "geographic_coverage": (
            _categorical_coverage(frame, geography_column)
            if geography_column is not None
            else _unsupported_coverage(row_count, "no row-level geography")
        ),
        "accident_history_coverage": (
            _categorical_coverage(frame, accident_column)
            if accident_column is not None
            else _unsupported_coverage(row_count, "no accident-history field")
        ),
        "title_status_coverage": (
            _categorical_coverage(frame, title_column)
            if title_column is not None
            else _unsupported_coverage(row_count, "no title-status field")
        ),
        "condition_coverage": (
            _categorical_coverage(frame, condition_column)
            if condition_column is not None
            else _unsupported_coverage(row_count, "no condition field")
        ),
        "distribution_indicators": {
            "vehicles_above_75000_usd": int((price > 75_000).sum()),
            "vehicles_above_75000_percentage": _percent(int((price > 75_000).sum()), row_count),
            "records_with_price_and_mileage": int((price.notna() & mileage.notna()).sum()),
        },
        "schema_compatibility": {
            "broad_coverage_fields": _field_support(
                frame,
                {
                    "year": "year" if "year" in frame else "model_year",
                    "make": make_column,
                    "model": model_column,
                    "mileage": "odometer" if "odometer" in frame else "milage",
                    "transmission": "transmission",
                    "fuel_type": "fuel" if "fuel" in frame else "fuel_type",
                    "drivetrain": "drive" if "drive" in frame else None,
                    "vehicle_type": "type" if "type" in frame else None,
                    "geography": geography_column,
                },
            ),
            "rich_fields": _field_support(
                frame,
                {
                    "engine": "engine" if "engine" in frame else "cylinders",
                    "accident_status": accident_column,
                    "title_status": title_column,
                    "condition": condition_column,
                },
            ),
        },
        "notes": list(source_notes),
    }


def _missing_mask(series: Any) -> Any:
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)


def _missing_mask_from_values(frame: Any, parsed: Any) -> Any:
    if "milage" in frame:
        return _missing_mask(frame["milage"])
    return _missing_mask(frame["odometer"]) | parsed.isna()


def _numeric_distribution(series: Any) -> dict[str, float | int | None]:
    if int(series.size) == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    quantiles = series.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(series.size),
        "min": _finite_float(series.min()),
        "p01": _finite_float(quantiles.loc[0.01]),
        "p25": _finite_float(quantiles.loc[0.25]),
        "median": _finite_float(quantiles.loc[0.5]),
        "p75": _finite_float(quantiles.loc[0.75]),
        "p95": _finite_float(quantiles.loc[0.95]),
        "p99": _finite_float(quantiles.loc[0.99]),
        "max": _finite_float(series.max()),
    }


def _categorical_coverage(frame: Any, column: str) -> dict[str, object]:
    missing = _missing_mask(frame[column])
    present = frame.loc[~missing, column].astype("string").str.strip()
    counts = present.value_counts().head(15)
    return {
        "present_rows": int(present.size),
        "present_percentage": _percent(int(present.size), len(frame)),
        "distinct_values": int(present.nunique()),
        "top_values": [
            {"value": str(value), "count": int(count)} for value, count in counts.items()
        ],
    }


def _unsupported_coverage(row_count: int, reason: str) -> dict[str, object]:
    return {
        "present_rows": 0,
        "present_percentage": 0.0,
        "distinct_values": 0,
        "top_values": [],
        "reason": reason,
        "raw_rows": row_count,
    }


def _field_support(frame: Any, mapping: Mapping[str, str | None]) -> dict[str, object]:
    return {
        canonical: {
            "source_column": source,
            "supported": source is not None and source in frame,
            "present_percentage": (
                _percent(int((~_missing_mask(frame[source])).sum()), len(frame))
                if source is not None and source in frame
                else 0.0
            ),
        }
        for canonical, source in mapping.items()
    }


def _top_numeric_counts(series: Any, *, limit: int) -> list[dict[str, float | int]]:
    counts = series.value_counts().head(limit)
    return [
        {"price_usd": _finite_float(value), "count": int(count)} for value, count in counts.items()
    ]


def _safe_price_dollars(value: object) -> float:
    try:
        cents = parse_price_text_cents(
            str(value), expected_currency="USD", price_kind=PriceKind.ASKING
        )
    except ValueError:
        return math.nan
    return cents / 100


def _safe_mileage(value: object) -> float:
    try:
        parsed = parse_mileage_text(value)
    except ValueError:
        return math.nan
    return math.nan if parsed is None else float(parsed)


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise HuggingFaceDatasetError("quality metric is not finite")
    return parsed


def _percent(count: int, total: int) -> float:
    return round(100 * count / total, 6) if total else 0.0


def _yoad_mapping() -> dict[str, str | None]:
    return {
        "price_cents": "price (numeric USD asking price multiplied by 100)",
        "year": "year",
        "make": "manufacturer",
        "model": None,
        "mileage": "odometer",
        "condition": "condition",
        "engine": "cylinders (limited engine proxy)",
        "drivetrain": "drive",
        "accident_status": None,
        "title_status": "title_status",
        "transmission": "transmission",
        "fuel_type": "fuel",
        "vehicle_type": "type",
        "state": "state",
    }


def _carson_mapping() -> dict[str, str | None]:
    return {
        "price_cents": "price (strict dollar-string parser)",
        "year": "model_year",
        "make": "brand",
        "model": "model",
        "mileage": "milage (strict miles-string parser)",
        "condition": None,
        "engine": "engine",
        "drivetrain": None,
        "accident_status": "accident",
        "title_status": "clean_title",
        "transmission": "transmission",
        "fuel_type": "fuel_type",
        "vehicle_type": None,
        "state": None,
    }


__all__ = [
    "CARSON_ORIGIN",
    "CURRENT_RETAIL_ORIGIN",
    "QUALITY_REPORT_SCHEMA_VERSION",
    "YOAD_ORIGIN",
    "load_candidate_frame",
    "profile_huggingface_candidate",
    "write_quality_report",
]
