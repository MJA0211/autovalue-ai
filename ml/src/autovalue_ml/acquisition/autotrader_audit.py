"""Governed acquisition and aggregate-only audit for Rebrowser AutoTrader data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np
import pandas as pd

from autovalue_ml.acquisition.huggingface_dataset import (
    ApprovalStatus,
    DatasetUseApprovals,
    HuggingFaceArtifactSpec,
    VerifiedHuggingFaceArtifact,
    acquire_huggingface_artifact,
)
from autovalue_ml.acquisition.sources.kaggle_vehicle_sales import US_50_PLUS_DC

AUTOTRADER_REPO_ID: Final = "rebrowser/autotrader-dataset"
AUTOTRADER_REVISION: Final = "a6cd0c8addded3591ccdfcd6ee4249b454f99792"
AUTOTRADER_SOURCE_ID: Final = "hf_rebrowser_autotrader_preview"
AUTOTRADER_SCHEMA_VERSION: Final = "rebrowser-autotrader-audit/1.0.0"
PREMIUM_SENTINEL: Final = "[PREMIUM]"


@dataclass(frozen=True, slots=True)
class PreviewFile:
    """Expected identity of one Parquet file in the immutable preview."""

    path: str
    size_bytes: int
    sha256: str
    row_count: int


AUTOTRADER_PREVIEW_FILES: Final = (
    PreviewFile(
        "car-listings/data/2026-07-20.parquet",
        841689,
        "df4ffb1d7943122131fb6d38cea8b85d23f3199aa646524c96c71bc9a738d155",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-21.parquet",
        875935,
        "58f9d42ddd64bb1b74c40186bc0a1c96593c49eff9ae4626497f5db7e81368d5",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-22.parquet",
        893008,
        "155a979381920e12a9d6e15ce11f369fb023cf996ac878365d93248a136dd7d1",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-23.parquet",
        838246,
        "9ea1b5e1ba15bb85825440d61eb49dfa8ee50c8f65d1348409060d1274457796",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-24.parquet",
        720226,
        "45cc204a09fd512e6c1f5d184190c52111a6d877dd812c850642ce6ca5554c4d",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-25.parquet",
        874820,
        "486397ef43900c8886aa9a0c38a65a3982ce410cbcf89f0caa424db1a7d56db4",
        1000,
    ),
    PreviewFile(
        "car-listings/data/2026-07-26.parquet",
        79737,
        "2ac9d6f62b8c7719f14b20b5b40af7bb29ac678b5bbd7fa619e8c0af30bb8638",
        8,
    ),
    PreviewFile(
        "car-listings/data/2026-07-27.parquet",
        68369,
        "7b4d04fd7ff347da5a7f2deb34eb6e2dd4a2b0e252a9ae9edd1a06ef9635451d",
        12,
    ),
    PreviewFile(
        "car-listings/data/2026-07-28.parquet",
        246709,
        "41d3bab653d9017e13d1a919743545a71b46e5ffed4e203b5b20113c49ac9f3a",
        220,
    ),
    PreviewFile(
        "car-listings/data/2026-07-29.parquet",
        573993,
        "5b802c4ccdce59572482271d4191af366b223defca2755e917e5c5b644712382",
        561,
    ),
    PreviewFile(
        "car-listings/data/2026-07-30.parquet",
        84969,
        "b14e4eed57e286b78e278d90fab70a1939ec53515805aea683374ce2e7237e36",
        27,
    ),
    PreviewFile(
        "car-listings/data/2026-07-31.parquet",
        93019,
        "c7034611734100cf1534516bbb05a03688b91bc0dc37664d1f554f62d9ce213e",
        24,
    ),
    PreviewFile(
        "car-listings/data/2026-08-01.parquet",
        61134,
        "928347cdcd572ea8bba101652142e5b1ff722a0875f2d07365eb9088f2e23afd",
        2,
    ),
    PreviewFile(
        "car-listings/data/2026-08-02.parquet",
        86610,
        "3281f318ef10175e3826c571218cb6f871e3bf27db2dcb70eaff60365ea789e5",
        14,
    ),
    PreviewFile(
        "car-listings/data/2026-08-03.parquet",
        80557,
        "28cf96944466a5933bf011866cc3eddb9b822ee596531fad470614cafe27c0b5",
        11,
    ),
    PreviewFile(
        "car-listings/data/2026-08-04.parquet",
        458355,
        "02ca18bd32121b26bc1bf6a822512d0653d0673908ee7dfc0cf0d82435a9d247",
        428,
    ),
    PreviewFile(
        "car-listings/data/2026-08-05.parquet",
        119021,
        "383aef656d7ccfd02c8428d01064e6d274eaa3803f875ec504c766ce13dd1635",
        62,
    ),
    PreviewFile(
        "car-listings/data/2026-08-06.parquet",
        53205,
        "dcd535eb408537bdb40b0d15f5eb4059562f3b9a9d5fa0fa9ae3de20b9a571db",
        17,
    ),
    PreviewFile(
        "car-listings/data/2026-08-07.parquet",
        54378,
        "27ceb478d2ff952320c8038c8ff7567ed37e5f5339284732278bd25a06e8e0e1",
        18,
    ),
    PreviewFile(
        "car-listings/data/2026-08-08.parquet",
        75723,
        "8d9c3b7e7e968940d2be9cde2c1e3f33f93dc14acb18890acb2af6adbee77c22",
        14,
    ),
    PreviewFile(
        "car-listings/data/2026-08-09.parquet",
        60296,
        "3f9f88200356e10cb79aaccea086c25c931a922383f0f0a5bcc9f35ef1884d0e",
        9,
    ),
    PreviewFile(
        "car-listings/data/2026-08-10.parquet",
        54986,
        "33744fbaa6a4dcbf812e95466281b907d23163520ef09c5c27f5004d63be2723",
        9,
    ),
    PreviewFile(
        "car-listings/data/2026-08-11.parquet",
        360649,
        "57a3c434d7423e4319c3f371f0cdc72efab26441568f043528954d9deb17005d",
        496,
    ),
    PreviewFile(
        "car-listings/data/2026-08-12.parquet",
        81651,
        "3d4421a62c217948542bc45fb4c2c8c29f96e52877d739eebbcf26d5805ad39e",
        14,
    ),
    PreviewFile(
        "car-listings/data/2026-08-13.parquet",
        65139,
        "393c3c1251547ad79b1239683042495680c3a518d8a3fb112aecf84599843634",
        13,
    ),
    PreviewFile(
        "car-listings/data/2026-08-14.parquet",
        93378,
        "d9af68faafaca60f3bdaf900d3fa7a80a9a11be462026baf23ba7cd64b686543",
        13,
    ),
    PreviewFile(
        "car-listings/data/2026-08-15.parquet",
        59930,
        "29be10e57b523e22d477287ef97688df95a146c522cf779dbb2c0e401fbd6399",
        6,
    ),
    PreviewFile(
        "car-listings/data/2026-08-16.parquet",
        76381,
        "5e18e6844010a8711f672720d081659e4fbe8cb99ab5b6b9a054c9d10707c4db",
        9,
    ),
    PreviewFile(
        "car-listings/data/2026-08-17.parquet",
        74622,
        "db8d471e6f0230b6495019109e5e133a7eafa099531fb9b0f2d1e614e2e2913b",
        11,
    ),
    PreviewFile(
        "car-listings/data/2026-08-18.parquet",
        91374,
        "3f23552f2dde9cdc4f0fd8a65b20aff73148e93c5d48d5d22612e71c25081ee3",
        21,
    ),
)

EXPECTED_COLUMNS: Final = (
    "_primaryKey",
    "_firstSeenAt",
    "_lastSeenAt",
    "listingId",
    "vin",
    "stockNumber",
    "year",
    "makeCode",
    "makeName",
    "modelCode",
    "modelName",
    "trim",
    "bodyStyle",
    "listingType",
    "listingTitle",
    "mileage",
    "salePrice",
    "msrp",
    "dealIndicator",
    "kbbFairPurchasePrice",
    "kbbFairPriceLow",
    "kbbFairPriceHigh",
    "exteriorColor",
    "exteriorColorSimple",
    "interiorColor",
    "interiorColorSimple",
    "engineCode",
    "engine",
    "drivetrain",
    "transmissionCode",
    "transmission",
    "transmissionGroup",
    "fuelTypeCode",
    "fuelType",
    "fuelTypeGroup",
    "mpgCity",
    "mpgHighway",
    "displacementUOM",
    "hasLeatherSeats",
    "daysOnMarket",
    "isHot",
    "isNewlyListed",
    "isReducedPrice",
    "isNoHagglePrice",
    "hasSpecialOffer",
    "moneyBackGuarantee",
    "mainImageIsStock",
    "priority",
    "sellerId",
    "sellerName",
    "sellerPhone",
    "sellerRating",
    "sellerReviewCount",
    "sellerWebsite",
    "isPrivateSeller",
    "sellerContractLevel",
    "kbbVehicleId",
    "kbbConsumerRatings",
    "kbbConsumerReviewCount",
    "safetyRecallCount",
    "vhrPreview",
    "options",
    "optionsCount",
    "description",
    "images",
    "imagesCount",
    "sellerAddress",
    "sellerCity",
    "sellerState",
    "sellerZip",
    "listingUrl",
)

PREMIUM_FIELDS: Final = (
    "vin",
    "salePrice",
    "kbbFairPurchasePrice",
    "sellerName",
    "sellerPhone",
    "sellerRating",
    "sellerWebsite",
    "images",
    "listingUrl",
)

_APPROVALS: Final = DatasetUseApprovals(
    acquisition=ApprovalStatus.APPROVED,
    batch_training=ApprovalStatus.BLOCKED,
    online_learning=ApprovalStatus.BLOCKED,
    acquisition_evidence=(
        "Public ungated preview may be privately acquired at the pinned revision for an "
        "aggregate-only noncommercial audit under the repository's CC BY-NC 4.0 label."
    ),
    batch_training_evidence=(
        "Blocked: Rebrowser disclaims any grant of third-party source intellectual-property "
        "rights, and permission to train on and publish derivatives of KBB valuation targets "
        "has not been established."
    ),
    online_learning_evidence=(
        "Blocked: batch reuse is not approved and this rotating preview is not an authorized, "
        "append-only, replay-safe online feed."
    ),
)

_FORBIDDEN_PREDICTORS: Final = {
    "kbbFairPriceLow": "direct target",
    "kbbFairPriceHigh": "direct target",
    "kbbMidpoint": "KBB-derived target",
    "kbbRangeWidth": "KBB-derived target",
    "kbbFairPurchasePrice": "premium KBB point estimate and direct target proxy",
    "kbbVehicleId": "KBB lookup identifier and target-proxy risk",
    "kbbConsumerRatings": "KBB-derived metadata",
    "kbbConsumerReviewCount": "KBB-derived metadata",
    "dealIndicator": "KBB deal score derived from valuation and listing price",
    "salePrice": "listing-price field",
    "msrp": "price field",
    "isReducedPrice": "price-history derivative",
    "isNoHagglePrice": "price-policy derivative",
    "daysOnMarket": "post-listing outcome and temporal leakage risk",
    "isHot": "post-listing market-response signal",
    "priority": "paid marketplace-position signal",
    "listingTitle": "unsanitized text can contain price or KBB language",
    "description": "unsanitized text can contain price or KBB language",
    "_primaryKey": "record identifier; grouping only",
    "listingId": "listing identifier; grouping only",
    "stockNumber": "dealer inventory identifier; grouping only",
    "sellerId": "seller identifier; grouping/audit only",
    "_firstSeenAt": "split/audit timestamp only",
    "_lastSeenAt": "split/audit timestamp only",
}


def preview_artifact_specs() -> tuple[HuggingFaceArtifactSpec, ...]:
    """Build independently checksum-pinned specs for every free Parquet file."""
    return tuple(
        HuggingFaceArtifactSpec(
            source_id=AUTOTRADER_SOURCE_ID,
            repo_id=AUTOTRADER_REPO_ID,
            revision=AUTOTRADER_REVISION,
            file_path=PurePosixPath(item.path),
            expected_size_bytes=item.size_bytes,
            expected_sha256=item.sha256,
            expected_row_count=item.row_count,
            declared_license=(
                "CC-BY-NC-4.0 repository label; third-party KBB/AutoTrader rights unresolved"
            ),
            license_url=(
                "https://huggingface.co/datasets/rebrowser/autotrader-dataset/blob/"
                f"{AUTOTRADER_REVISION}/README.md"
            ),
            upstream_source="Rebrowser collection of public AutoTrader listing pages",
            schema_mapping_version=AUTOTRADER_SCHEMA_VERSION,
            approvals=_APPROVALS,
            usage_restrictions=(
                "Noncommercial aggregate audit only; retain Rebrowser attribution.",
                "Do not redistribute raw rows or premium/restricted fields.",
                "Do not train on KBB targets until third-party ML reuse rights are documented.",
                "Do not merge these observations with Cars.com or Yoad data.",
            ),
            attribution=(
                "Rebrowser, AutoTrader Vehicle Listings Dataset (2026), "
                "https://rebrowser.net/products/datasets/autotrader"
            ),
            config="car-listings",
            split="train",
        )
        for item in AUTOTRADER_PREVIEW_FILES
    )


def acquire_autotrader_preview(raw_root: Path) -> tuple[VerifiedHuggingFaceArtifact, ...]:
    """Acquire only the free revision-pinned Parquet preview; never premium exports."""
    return tuple(acquire_huggingface_artifact(spec, raw_root) for spec in preview_artifact_specs())


def audit_autotrader_preview(
    artifacts: Sequence[VerifiedHuggingFaceArtifact], *, generated_at: datetime | None = None
) -> dict[str, object]:
    """Verify and profile an acquired snapshot without emitting row-level content."""
    expected = preview_artifact_specs()
    if len(artifacts) != len(expected):
        raise ValueError("AutoTrader audit requires the complete 30-file manifest")
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for artifact, spec in zip(artifacts, expected, strict=True):
        if artifact.spec != spec:
            raise ValueError("AutoTrader artifacts are not in the reviewed manifest order")
        frame = pd.read_parquet(artifact.path)
        if tuple(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected AutoTrader schema in {spec.file_path.name}")
        if len(frame) != spec.expected_row_count:
            raise ValueError(f"unexpected AutoTrader row count in {spec.file_path.name}")
        frame = frame.copy()
        frame["_snapshotFile"] = spec.file_path.stem
        frames.append(frame)
        manifest.append(
            {
                "path": spec.file_path.as_posix(),
                "rows": len(frame),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    return profile_autotrader_frame(
        combined,
        manifest=manifest,
        generated_at=datetime.now(UTC) if generated_at is None else generated_at,
    )


def profile_autotrader_frame(
    frame: pd.DataFrame,
    *,
    manifest: Sequence[Mapping[str, object]],
    generated_at: datetime,
) -> dict[str, object]:
    """Build the deterministic aggregate report from schema-validated rows."""
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if tuple(column for column in frame.columns if column != "_snapshotFile") != EXPECTED_COLUMNS:
        raise ValueError("unexpected AutoTrader aggregate schema")
    if "_snapshotFile" not in frame:
        raise ValueError("snapshot provenance column is required")
    total_rows = len(frame)
    low = pd.to_numeric(frame["kbbFairPriceLow"], errors="coerce")
    high = pd.to_numeric(frame["kbbFairPriceHigh"], errors="coerce")
    finite = pd.Series(np.isfinite(low) & np.isfinite(high), index=frame.index)
    valid = finite & low.gt(0) & high.gt(0) & low.le(high)
    midpoint = (low[valid] + high[valid]) / 2
    width = high[valid] - low[valid]

    listing_counts = _nonblank(frame["listingId"]).value_counts()
    primary_counts = _nonblank(frame["_primaryKey"]).value_counts()
    listing_snapshot_counts = frame.groupby("listingId", dropna=False)["_snapshotFile"].nunique()
    repeated_listing_ids = listing_counts[listing_counts > 1].index
    repeated_kbb = frame.loc[
        frame["listingId"].isin(repeated_listing_ids),
        ["listingId", "kbbFairPriceLow", "kbbFairPriceHigh"],
    ]
    kbb_variants = repeated_kbb.groupby("listingId")[
        ["kbbFairPriceLow", "kbbFairPriceHigh"]
    ].nunique(dropna=False)
    repeated_kbb_changes = int(kbb_variants.max(axis=1).gt(1).sum())
    stock_group = frame.loc[
        _present_mask(frame["stockNumber"]) & frame["sellerId"].notna(),
        ["sellerId", "stockNumber", "listingId"],
    ]
    stock_counts = stock_group.groupby(["sellerId", "stockNumber"], dropna=False).size()

    first_seen = pd.to_datetime(frame["_firstSeenAt"], errors="coerce", utc=True)
    last_seen = pd.to_datetime(frame["_lastSeenAt"], errors="coerce", utc=True)
    file_date = frame["_snapshotFile"].astype("string")
    first_seen_date = first_seen.dt.strftime("%Y-%m-%d")
    state = frame["sellerState"].astype("string").str.strip().str.upper()
    valid_state = state.isin(US_50_PLUS_DC)

    premium = {
        column: {
            "premium_sentinel_rows": int(frame[column].eq(PREMIUM_SENTINEL).sum()),
            "missing_rows": int(frame[column].isna().sum()),
            "unexpected_exposed_rows": int(
                (frame[column].notna() & ~frame[column].eq(PREMIUM_SENTINEL)).sum()
            ),
        }
        for column in PREMIUM_FIELDS
    }
    coverage_fields = (
        "year",
        "makeName",
        "modelName",
        "trim",
        "mileage",
        "engine",
        "transmission",
        "drivetrain",
        "listingType",
        "vhrPreview",
        "sellerCity",
        "sellerState",
        "sellerZip",
    )
    feature_coverage = {column: _coverage(frame[column]) for column in coverage_fields}
    feature_coverage["physicalCondition"] = {
        "present_rows": 0,
        "present_percentage": 0.0,
        "note": "listingType is inventory status, not physical vehicle condition",
    }

    exact_duplicate_mask = frame.loc[:, EXPECTED_COLUMNS].duplicated(keep=False)
    exact_duplicate_excess = frame.loc[:, EXPECTED_COLUMNS].duplicated().sum()
    file_payload = "".join(
        f"{item['path']}|{item['size_bytes']}|{item['sha256']}|{item['rows']}\n"
        for item in manifest
    ).encode("utf-8")
    report: dict[str, object] = {
        "report_schema_version": 1,
        "report_type": "rebrowser_autotrader_free_preview_acquisition_audit",
        "generated_at": generated_at.astimezone(UTC).isoformat(),
        "source": {
            "repo_id": AUTOTRADER_REPO_ID,
            "revision": AUTOTRADER_REVISION,
            "bundle_sha256": hashlib.sha256(file_payload).hexdigest(),
            "file_count": len(manifest),
            "manifest": list(manifest),
            "declared_license": "CC-BY-NC-4.0",
            "attribution_required": True,
            "attribution": (
                "Rebrowser, AutoTrader Vehicle Listings Dataset (2026), "
                "https://rebrowser.net/products/datasets/autotrader"
            ),
            "provenance": "Rebrowser collection from publicly accessible AutoTrader pages",
            "upstream_affiliation": "none claimed",
        },
        "permissions": {
            "acquisition": "approved_for_private_aggregate_audit_only",
            "batch_training": "blocked",
            "online_learning": "blocked",
            "raw_redistribution": "blocked",
            "premium_access": "not_attempted",
            "blocking_reason": (
                "Rebrowser's terms grant no third-party source-IP rights; permission to use "
                "KBB-derived targets for ML training and model publication is unresolved."
            ),
        },
        "artifact": {
            "rows": total_rows,
            "columns": len(EXPECTED_COLUMNS),
            "column_names": list(EXPECTED_COLUMNS),
            "schema_consistent_across_files": True,
            "advertised_maximum_rows": 30_000,
            "advertised_field_count": 68,
            "observed_card_discrepancies": {
                "row_shortfall_from_advertised_maximum": 30_000 - total_rows,
                "additional_columns_vs_advertised": len(EXPECTED_COLUMNS) - 68,
            },
            "listing_type_counts": {
                str(key): int(value)
                for key, value in frame["listingType"].value_counts(dropna=False).items()
            },
        },
        "field_access": {
            "free_field_count": len(EXPECTED_COLUMNS) - len(PREMIUM_FIELDS),
            "free_fields": [item for item in EXPECTED_COLUMNS if item not in PREMIUM_FIELDS],
            "premium_field_count": len(PREMIUM_FIELDS),
            "premium_fields": list(PREMIUM_FIELDS),
            "premium_field_audit": premium,
        },
        "kbb_targets": {
            "meaning": (
                "KBB Fair Purchase Price low/high range in USD as reproduced in AutoTrader "
                "listings; not an observed sale price and not an empirical prediction interval"
            ),
            "low": _numeric_audit(frame["kbbFairPriceLow"], low),
            "high": _numeric_audit(frame["kbbFairPriceHigh"], high),
            "complete_valid_ranges": int(valid.sum()),
            "complete_valid_percentage": _percentage(int(valid.sum()), total_rows),
            "both_missing": int((low.isna() & high.isna()).sum()),
            "exactly_one_missing": int((low.isna() ^ high.isna()).sum()),
            "low_greater_than_high": int((finite & low.gt(high)).sum()),
            "zero_low": int(low.eq(0).sum()),
            "zero_high": int(high.eq(0).sum()),
            "negative_low": int(low.lt(0).sum()),
            "negative_high": int(high.lt(0).sum()),
            "equal_low_high": int((finite & low.eq(high)).sum()),
            "midpoint_usd": _distribution(midpoint),
            "range_width_usd": _distribution(width),
            "hard_extremes": {
                "low_above_500000": int(low.gt(500_000).sum()),
                "high_above_500000": int(high.gt(500_000).sum()),
                "midpoint_above_500000": int(midpoint.gt(500_000).sum()),
                "range_width_above_100000": int(width.gt(100_000).sum()),
            },
            "tukey_extreme_outliers_3_iqr": {
                "low": _extreme_outlier_count(low[low.notna()]),
                "high": _extreme_outlier_count(high[high.notna()]),
                "midpoint": _extreme_outlier_count(midpoint),
                "range_width": _extreme_outlier_count(width),
            },
            "valid_ranges_by_listing_type": {
                str(listing_type): {
                    "rows": int(mask.sum()),
                    "valid_ranges": int((valid & mask).sum()),
                    "valid_percentage": _percentage(int((valid & mask).sum()), int(mask.sum())),
                }
                for listing_type, mask in (
                    (value, frame["listingType"].eq(value))
                    for value in sorted(frame["listingType"].dropna().unique())
                )
            },
        },
        "scope": {
            "market": "United States",
            "currency": "USD",
            "currency_evidence": "pinned schema descriptions; no row-level currency column",
            "state_present_rows": int(state.notna().sum()),
            "valid_50_states_plus_dc_rows": int(valid_state.sum()),
            "invalid_or_non_us_state_rows": int((state.notna() & ~valid_state).sum()),
            "missing_state_rows": int(state.isna().sum()),
            "distinct_valid_states": int(state[valid_state].nunique()),
            "five_digit_zip_rows": int(
                frame["sellerZip"].astype("string").str.fullmatch(r"\d{5}", na=False).sum()
            ),
        },
        "timestamps": {
            "first_seen_min": _iso_min(first_seen),
            "first_seen_max": _iso_max(first_seen),
            "last_seen_min": _iso_min(last_seen),
            "last_seen_max": _iso_max(last_seen),
            "invalid_or_missing_first_seen": int(first_seen.isna().sum()),
            "invalid_or_missing_last_seen": int(last_seen.isna().sum()),
            "first_after_last": int((first_seen > last_seen).sum()),
            "snapshot_file_min": str(frame["_snapshotFile"].min()),
            "snapshot_file_max": str(frame["_snapshotFile"].max()),
            "snapshot_file_count": int(frame["_snapshotFile"].nunique()),
            "rows_whose_first_seen_date_matches_file_name": int(
                first_seen_date.eq(file_date).sum()
            ),
            "file_semantics_inference": (
                "files behave as first-seen date cohorts, not repeated full daily snapshots"
            ),
        },
        "identifiers_and_repetition": {
            "listing_id_present": int(_present_mask(frame["listingId"]).sum()),
            "unique_listing_ids": int(listing_counts.size),
            "listing_ids_repeated": int((listing_counts > 1).sum()),
            "listing_rows_beyond_first": int((listing_counts - 1).clip(lower=0).sum()),
            "max_rows_per_listing_id": int(listing_counts.max()) if not listing_counts.empty else 0,
            "listing_ids_across_multiple_snapshot_files": int((listing_snapshot_counts > 1).sum()),
            "primary_key_present": int(_present_mask(frame["_primaryKey"]).sum()),
            "unique_primary_keys": int(primary_counts.size),
            "primary_keys_repeated": int((primary_counts > 1).sum()),
            "exact_duplicate_rows_total": int(exact_duplicate_mask.sum()),
            "exact_duplicate_rows_beyond_first": int(exact_duplicate_excess),
            "seller_stock_groups": int(stock_counts.size),
            "seller_stock_groups_repeated": int((stock_counts > 1).sum()),
            "vin_available": False,
            "vin_status": "premium sentinel only",
            "repeated_vins": "not_evaluable_premium_sentinel_only",
            "repeated_listing_ids_with_kbb_changes": repeated_kbb_changes,
            "kbb_change_assessment": (
                "not_evaluable_no_repeated_listing_ids"
                if repeated_listing_ids.empty
                else "evaluated_on_repeated_listing_ids"
            ),
            "future_grouping_rule": (
                "group by listingId; additionally group stable sellerId+stockNumber where "
                "available, and keep all snapshots of a group in one validation partition"
            ),
        },
        "feature_coverage": feature_coverage,
        "vehicle_history": _vehicle_history_audit(frame["vhrPreview"]),
        "leakage_audit": {
            "forbidden_predictors": dict(_FORBIDDEN_PREDICTORS),
            "target_derivatives_created_for_audit_only": ["kbbMidpoint", "kbbRangeWidth"],
            "source_identity_as_predictor": "forbidden",
            "identifiers_as_predictors": "forbidden",
            "free_text_as_predictor": "forbidden_until_price_and_kbb_language_is_sanitized",
        },
        "comparison": _feature_comparison(feature_coverage),
        "decision": {
            "classification": "reference/analytics only",
            "model_training_run": False,
            "merge_with_cars_or_yoad": False,
            "requirements_before_reconsideration": [
                "document permission covering ML reuse of KBB-derived valuation targets",
                "document permission to publish aggregate results and any trained derivative",
                "freeze the exact approved artifact and attribution/redistribution terms",
                "preregister grouped temporal validation and the leakage denylist",
            ],
        },
    }
    return report


def write_autotrader_audit(report: Mapping[str, object], output: Path) -> None:
    """Atomically write a stable aggregate-only JSON report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output)


def _feature_comparison(auto: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "basis": "current governed development/audit artifacts; no datasets are merged",
        "rows": {"cars_com_development": 98_552, "yoad_approved": 242_666, "autotrader_raw": 8_019},
        "coverage_percentage": {
            "exact_model": {
                "cars_com": 100.0,
                "yoad": 0.0,
                "autotrader": auto["modelName"]["present_percentage"],
            },
            "trim": {
                "cars_com": 0.0,
                "yoad": 0.0,
                "autotrader": auto["trim"]["present_percentage"],
            },
            "engine": {
                "cars_com": 0.0,
                "yoad": "cylinders proxy only",
                "autotrader": auto["engine"]["present_percentage"],
            },
            "timestamps": {"cars_com": 0.0, "yoad": 0.0, "autotrader": 100.0},
            "vehicle_history": {
                "cars_com": 0.0,
                "yoad": "title status only",
                "autotrader": auto["vhrPreview"]["present_percentage"],
            },
            "row_geography": {
                "cars_com": 0.0,
                "yoad": 100.0,
                "autotrader": auto["sellerState"]["present_percentage"],
            },
            "mileage": {
                "cars_com": 40.780501664096114,
                "yoad": 100.0,
                "autotrader": auto["mileage"]["present_percentage"],
            },
        },
    }


def _coverage(series: pd.Series) -> dict[str, object]:
    present = _present_mask(series)
    count = int(present.sum())
    return {
        "present_rows": count,
        "present_percentage": _percentage(count, len(series)),
        "distinct_values": int(series[present].nunique()),
    }


def _numeric_audit(raw: pd.Series, numeric: pd.Series) -> dict[str, object]:
    raw_present = _present_mask(raw)
    return {
        "raw_present_rows": int(raw_present.sum()),
        "numeric_rows": int(numeric.notna().sum()),
        "nonnumeric_nonmissing_rows": int((raw_present & numeric.isna()).sum()),
        "missing_rows": int((~raw_present).sum()),
        "distribution_usd": _distribution(numeric[numeric.notna()]),
    }


def _vehicle_history_audit(series: pd.Series) -> dict[str, object]:
    present = series[_present_mask(series)].astype("string")
    flag_counts: Counter[str] = Counter()
    invalid_json = 0
    non_list_json = 0
    for value in present:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            invalid_json += 1
            continue
        if not isinstance(parsed, list):
            non_list_json += 1
            continue
        flag_counts.update(str(flag) for flag in parsed)
    return {
        "meaning": "preview flags, not a complete vehicle-history report",
        "present_rows": int(present.size),
        "missing_rows": int(len(series) - present.size),
        "valid_json_array_rows": int(present.size - invalid_json - non_list_json),
        "invalid_json_rows": invalid_json,
        "non_list_json_rows": non_list_json,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def _distribution(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(clean.size),
        "min": float(clean.min()),
        "p01": float(clean.quantile(0.01)),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def _extreme_outlier_count(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return 0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((clean < q1 - 3 * iqr) | (clean > q3 + 3 * iqr)).sum())


def _present_mask(series: pd.Series) -> pd.Series:
    present = series.notna()
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        present &= series.astype("string").str.strip().ne("").fillna(False)
    return present


def _nonblank(series: pd.Series) -> pd.Series:
    return series[_present_mask(series)].astype("string").str.strip()


def _percentage(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total * 100


def _iso_min(series: pd.Series) -> str | None:
    return None if series.dropna().empty else series.min().isoformat()


def _iso_max(series: pd.Series) -> str | None:
    return None if series.dropna().empty else series.max().isoformat()


__all__ = [
    "AUTOTRADER_PREVIEW_FILES",
    "AUTOTRADER_REPO_ID",
    "AUTOTRADER_REVISION",
    "AUTOTRADER_SOURCE_ID",
    "EXPECTED_COLUMNS",
    "PREMIUM_FIELDS",
    "PreviewFile",
    "acquire_autotrader_preview",
    "audit_autotrader_preview",
    "preview_artifact_specs",
    "profile_autotrader_frame",
    "write_autotrader_audit",
]
