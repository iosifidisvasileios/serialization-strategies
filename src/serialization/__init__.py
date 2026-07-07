from .base import (
    IGNORE_LABEL,
    BaseSerializer,
    SerializedItem,
    collapse_serialized_labels_to_original,
    iter_real_serialized_positions,
)
from .plain_text import PlainTextSerializer
from .page_aware import PageAwareSerializer
from .block_aware import BlockAwareSerializer, infer_token_block_ids
from .line_aware import LineAwareSerializer
from .rowcol_bucket import RowColBucketSerializer
from .bbox_token import BBoxTokenSerializer
from .t5_json import T5JsonSerializer

TOKEN_CLASSIFICATION_SERIALIZERS = {
    "plain_text": PlainTextSerializer,
    "page_aware": PageAwareSerializer,
    "block_aware": BlockAwareSerializer,
    "line_aware": LineAwareSerializer,
    "rowcol_bucket": RowColBucketSerializer,
    "bbox_token": BBoxTokenSerializer,
}

SEQ2SEQ_SERIALIZERS = {
    "t5_json": T5JsonSerializer,
}

ALL_SERIALIZERS = {
    **TOKEN_CLASSIFICATION_SERIALIZERS,
    **SEQ2SEQ_SERIALIZERS,
}


def get_serializer_class(name: str):
    try:
        return ALL_SERIALIZERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ALL_SERIALIZERS))
        raise KeyError(f"Unknown serializer {name!r}. Available: {available}") from exc


__all__ = [
    "IGNORE_LABEL",
    "BaseSerializer",
    "SerializedItem",
    "PlainTextSerializer",
    "PageAwareSerializer",
    "BlockAwareSerializer",
    "LineAwareSerializer",
    "RowColBucketSerializer",
    "BBoxTokenSerializer",
    "T5JsonSerializer",
    "TOKEN_CLASSIFICATION_SERIALIZERS",
    "SEQ2SEQ_SERIALIZERS",
    "ALL_SERIALIZERS",
    "get_serializer_class",
    "collapse_serialized_labels_to_original",
    "iter_real_serialized_positions",
    "infer_token_block_ids",
]
