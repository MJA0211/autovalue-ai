"""Fail-closed permissions dedicated to River updates."""

from __future__ import annotations

from dataclasses import dataclass

from autovalue_ml.online.errors import SourcePermissionError

SYNTHETIC_SHADOW_SOURCE_ID = "autovalue.synthetic.shadow.v1"
CARS_RETAIL_SOURCE_ID = "kaggle_us_sales_cars_v2"
WHOLESALE_SOURCE_ID = "kaggle_vehicle_sales_v1"
YOAD22_SOURCE_ID = "hf_yoad22_craigslist_used_cars"
AUTOTRADER_SOURCE_ID = "hf_rebrowser_autotrader_preview"
CARSON_SHIVELY_SOURCE_ID = "hf_carson_shively_used_car_price"
PERMISSION_POLICY_VERSION = "river-source-permissions-v1"


@dataclass(frozen=True, slots=True)
class OnlineSourcePermission:
    """One explicit online-learning decision, independent of batch approval."""

    source_id: str
    approved: bool
    data_class: str
    reason: str


class OnlineSourcePermissionRegistry:
    """Allow only enumerated sources; unknown sources are denied."""

    version = PERMISSION_POLICY_VERSION

    def __init__(self) -> None:
        decisions = (
            OnlineSourcePermission(
                source_id=SYNTHETIC_SHADOW_SOURCE_ID,
                approved=True,
                data_class="synthetic_simulation",
                reason="project-owned synthetic events approved for architecture testing only",
            ),
            OnlineSourcePermission(
                source_id=CARS_RETAIL_SOURCE_ID,
                approved=False,
                data_class="historical_retail",
                reason="batch permission does not grant online-learning permission",
            ),
            OnlineSourcePermission(
                source_id=WHOLESALE_SOURCE_ID,
                approved=False,
                data_class="historical_wholesale",
                reason="batch permission does not grant online-learning permission",
            ),
            OnlineSourcePermission(
                source_id=YOAD22_SOURCE_ID,
                approved=False,
                data_class="historical_retail",
                reason="controlled batch experimentation only; River use is blocked",
            ),
            OnlineSourcePermission(
                source_id=AUTOTRADER_SOURCE_ID,
                approved=False,
                data_class="third_party_reference",
                reason="controlled non-commercial research only; River use is blocked",
            ),
            OnlineSourcePermission(
                source_id=CARSON_SHIVELY_SOURCE_ID,
                approved=False,
                data_class="unresolved_provenance",
                reason="upstream provenance and U.S./USD scope remain unresolved",
            ),
        )
        self._decisions = {decision.source_id: decision for decision in decisions}

    def decision_for(self, source_id: str) -> OnlineSourcePermission:
        """Return an explicit decision, synthesizing a denied unknown decision."""
        return self._decisions.get(
            source_id,
            OnlineSourcePermission(
                source_id=source_id,
                approved=False,
                data_class="unknown",
                reason="source is absent from the online-learning allowlist",
            ),
        )

    def require_learning_approval(self, source_id: str) -> OnlineSourcePermission:
        """Raise unless the exact source is approved for River updates."""
        decision = self.decision_for(source_id)
        if not decision.approved:
            raise SourcePermissionError(f"River source blocked: {source_id}: {decision.reason}")
        return decision

    def public_summary(self) -> list[dict[str, str | bool]]:
        """Return deterministic, non-row-level permission evidence."""
        return [
            {
                "source_id": decision.source_id,
                "approved": decision.approved,
                "data_class": decision.data_class,
                "reason": decision.reason,
            }
            for decision in sorted(self._decisions.values(), key=lambda item: item.source_id)
        ]
