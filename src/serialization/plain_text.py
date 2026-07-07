from __future__ import annotations

from .base import BaseSerializer, SerializedItem, real_token_items
from ..preprocessing.schema import CanonicalDocument


class PlainTextSerializer(BaseSerializer):
    """Baseline serializer: OCR tokens in canonical reading order only."""

    name = "plain_text"

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        return real_token_items(doc)
