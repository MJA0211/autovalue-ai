"""Acquisition errors with safe, user-facing failure categories."""


class AcquisitionError(RuntimeError):
    """Base class for expected scraper failures."""


class PolicyViolationError(AcquisitionError):
    """The requested run is outside its reviewed source policy."""


class RobotsDeniedError(AcquisitionError):
    """The source's robots policy disallows the requested page."""


class FetchError(AcquisitionError):
    """A page could not be fetched safely within the configured limits."""


class ContentValidationError(AcquisitionError):
    """A response did not satisfy the expected content contract."""


class ListingParseError(AcquisitionError):
    """A source page no longer matches its reviewed parser contract."""


class CrawlBudgetExceededError(AcquisitionError):
    """A hard request, page, record, byte, or runtime budget was reached."""


class PaginationLoopError(AcquisitionError):
    """Pagination revisited a page instead of terminating."""


class DuplicateListingConflictError(AcquisitionError):
    """One source listing ID produced conflicting records in a single run."""
