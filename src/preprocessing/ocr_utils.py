from __future__ import annotations
import ast, gzip, json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from .schema import OCRBlock, OCRToken

def parse_json_like(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)): return value
    if isinstance(value, bytes): value = value.decode('utf-8')
    if not isinstance(value, str): return value
    s = value.strip()
    if not s: return None
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return value

def read_json_or_gzip(path: str|Path) -> Any:
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f: return json.load(f)
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def iter_ocr_units(ocr_payload: Any) -> Iterable[Mapping[str, Any]]:
    if ocr_payload is None: return []
    if isinstance(ocr_payload, Mapping): return [ocr_payload]
    if isinstance(ocr_payload, list): return [x for x in ocr_payload if isinstance(x, Mapping)]
    return []

def position_to_bbox(position: Mapping[str, Any]) -> Optional[list[int]]:
    def get_int(*names: str):
        for n in names:
            if n in position and position[n] is not None: return int(position[n])
        return None
    vals = [get_int('left','bbLeft','x0'), get_int('top','bbTop','y0'), get_int('right','bbRight','x1'), get_int('bottom','bbBot','y1')]
    return None if any(v is None for v in vals) else [int(v) for v in vals]

def normalize_bbox(bbox: list[int], page_width: int, page_height: int, scale: int = 1000, clamp: bool = True) -> list[int]:
    x0,y0,x1,y1 = bbox
    vals = [round(scale*x0/page_width), round(scale*y0/page_height), round(scale*x1/page_width), round(scale*y1/page_height)]
    if clamp: vals = [max(0, min(scale, int(v))) for v in vals]
    return [int(v) for v in vals]

def denormalize_bbox(bbox: list[int], page_width: int, page_height: int, scale: int = 1000) -> list[int]:
    x0,y0,x1,y1 = bbox
    return [round(page_width*x0/scale), round(page_height*y0/scale), round(page_width*x1/scale), round(page_height*y1/scale)]

def bbox_center(bbox: list[int]) -> tuple[float,float]:
    return ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)

def bbox_union(bboxes: Iterable[list[int]]) -> Optional[list[int]]:
    b = list(bboxes)
    return None if not b else [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]

def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1])

def bbox_intersection(a: list[int], b: list[int]) -> Optional[list[int]]:
    x0,y0,x1,y1 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    return None if x1 <= x0 or y1 <= y0 else [x0,y0,x1,y1]

def bbox_iou(a: list[int], b: list[int]) -> float:
    inter = bbox_intersection(a,b)
    if inter is None: return 0.0
    union = bbox_area(a) + bbox_area(b) - bbox_area(inter)
    return 0.0 if union <= 0 else bbox_area(inter) / union

def extract_page_sizes(ocr_payload: Any) -> dict[int, dict[str,int]]:
    sizes = {}
    for unit in iter_ocr_units(ocr_payload):
        for page in unit.get('pages', []) or []:
            if not isinstance(page, Mapping): continue
            size = page.get('size', {}) or {}
            if size.get('width') is None or size.get('height') is None: continue
            p = int(page.get('page_num', len(sizes)))
            sizes[p] = {'width': int(size['width']), 'height': int(size['height'])}
    return sizes

def extract_text_from_ocr(ocr_payload: Any) -> str:
    pages = []
    for unit in iter_ocr_units(ocr_payload):
        for page in unit.get('pages', []) or []:
            if isinstance(page, Mapping): pages.append((int(page.get('page_num', len(pages))), str(page.get('text',''))))
    return '\n\n'.join(t for _,t in sorted(pages))

def raw_token_to_ocr_token(raw: Mapping[str, Any], page_sizes: Optional[Mapping[int, Mapping[str,int]]] = None) -> Optional[OCRToken]:
    page_sizes = page_sizes or {}; off = raw.get('doc_offset',{}) or {}; pos = raw.get('position',{}) or {}
    if 'start' not in off or 'end' not in off: return None
    bbox = position_to_bbox(pos)
    if bbox is None: return None
    page = int(raw.get('page_num',0)); size = page_sizes.get(page,{})
    po = raw.get('page_offset',{}) or {}; bo = raw.get('block_offset',{}) or {}
    return OCRToken(text=str(raw.get('text','')), start=int(off['start']), end=int(off['end']), page=page, bbox=bbox, page_width=size.get('width'), page_height=size.get('height'), page_start=_maybe_int(po.get('start')), page_end=_maybe_int(po.get('end')), block_start=_maybe_int(bo.get('start')), block_end=_maybe_int(bo.get('end')), style=dict(raw.get('style',{}) or {}), extra={k:v for k,v in dict(raw).items() if k not in {'text','doc_offset','page_offset','block_offset','position','style','page_num'}})

def extract_tokens_from_ocr(ocr_payload: Any) -> list[OCRToken]:
    sizes = extract_page_sizes(ocr_payload); toks = []
    for unit in iter_ocr_units(ocr_payload):
        for raw in unit.get('tokens', []) or []:
            if isinstance(raw, Mapping):
                tok = raw_token_to_ocr_token(raw, sizes)
                if tok is not None: toks.append(tok)
    return sort_tokens_reading_order(toks)

def raw_block_to_ocr_block(raw: Mapping[str, Any], block_id: int) -> Optional[OCRBlock]:
    off = raw.get('doc_offset',{}) or {}; pos = raw.get('position',{}) or {}
    if 'start' not in off or 'end' not in off: return None
    bbox = position_to_bbox(pos)
    if bbox is None: return None
    po = raw.get('page_offset',{}) or {}
    return OCRBlock(text=str(raw.get('text','')), start=int(off['start']), end=int(off['end']), page=int(raw.get('page_num',0)), bbox=bbox, block_id=block_id, block_type=raw.get('block_type'), page_start=_maybe_int(po.get('start')), page_end=_maybe_int(po.get('end')), extra={k:v for k,v in dict(raw).items() if k not in {'text','doc_offset','page_offset','position','page_num','block_type'}})

def extract_blocks_from_ocr(ocr_payload: Any) -> list[OCRBlock]:
    blocks=[]; bid=0
    for unit in iter_ocr_units(ocr_payload):
        for raw in unit.get('blocks', []) or []:
            if isinstance(raw, Mapping):
                blk = raw_block_to_ocr_block(raw, bid)
                if blk is not None: blocks.append(blk); bid += 1
    return sorted(blocks, key=lambda b: (b.page, b.start, b.y0 if hasattr(b,'y0') else b.bbox[1], b.bbox[0]))

def group_tokens_by_page(tokens: Iterable[OCRToken]) -> dict[int, list[OCRToken]]:
    d = defaultdict(list)
    for t in tokens: d[t.page].append(t)
    return {p: sort_tokens_reading_order(v) for p,v in d.items()}

def sort_tokens_reading_order(tokens: Iterable[OCRToken]) -> list[OCRToken]:
    return sorted(tokens, key=lambda t: (t.page, t.start, t.y0, t.x0))

def sort_tokens_layout_order(tokens: Iterable[OCRToken]) -> list[OCRToken]:
    return sorted(tokens, key=lambda t: (t.page, t.y0, t.x0, t.start))

def reconstruct_text_from_tokens(tokens: Iterable[OCRToken], joiner: str = ' ') -> str:
    return joiner.join(t.text for t in tokens)

def token_context_window(tokens: list[OCRToken], token_index: int, window: int = 5) -> list[OCRToken]:
    return tokens[max(0, token_index-window): min(len(tokens), token_index+window+1)]

def bucket_value(value: int|float, n_buckets: int = 100, scale: int = 1000) -> int:
    return max(0, min(n_buckets-1, int(value * n_buckets / scale)))

def token_row_col_buckets(token: OCRToken, n_buckets: int = 100, scale: int = 1000, use_normalized: bool = True) -> tuple[int,int]:
    if use_normalized:
        x0,y0,x1,y1 = token.normalized_bbox(scale=scale); cx,cy = (x0+x1)//2, (y0+y1)//2
        return bucket_value(cy, n_buckets, scale), bucket_value(cx, n_buckets, scale)
    if token.page_width is None or token.page_height is None: raise ValueError('Raw buckets require page size.')
    return int(token.center_y*n_buckets/token.page_height), int(token.center_x*n_buckets/token.page_width)

def line_group_tokens(tokens: Iterable[OCRToken], y_threshold: Optional[float] = None) -> list[list[OCRToken]]:
    lines=[]
    for tok in sort_tokens_layout_order(tokens):
        thr = y_threshold if y_threshold is not None else max(6.0, tok.height*0.6)
        for line in lines:
            if line[0].page == tok.page and abs((sum(t.center_y for t in line)/len(line)) - tok.center_y) <= thr:
                line.append(tok); line.sort(key=lambda t: (t.x0,t.start)); break
        else:
            lines.append([tok])
    return sorted(lines, key=lambda line: (line[0].page, sum(t.center_y for t in line)/len(line), line[0].x0))

def _maybe_int(value: Any) -> Optional[int]:
    try: return None if value is None else int(value)
    except Exception: return None
