"""Experiment identifiers and UTC timestamps."""

from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid4().hex[:16]
