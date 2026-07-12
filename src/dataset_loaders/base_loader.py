from __future__ import annotations
import ast, gzip, json, math
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from ..preprocessing.schema import Annotation, CanonicalDocument, OCRBlock, OCRToken

@dataclass(frozen=True)
class LoaderColumns:
    doc_id: tuple[str,...] = ('doc_id','id','original_filename','filename','file_name','pdf_name')
    text: tuple[str,...] = ('text','raw_text','ocr_text','document_text')
    labels: tuple[str,...] = ('labels','Labels','annotations','Annotations')
    ocr_json: tuple[str,...] = ('OCR','ocr_json','ocr_output','ocr_data')
    ocr_path: tuple[str,...] = ('ocr','ocr_path','ocr_file','ocr_filename')
    image_files: tuple[str,...] = ('image_files','images','page_images')

class BaseDatasetLoader(ABC):
    dataset_name = 'base'
    default_label_map: dict[str,str] = {}
    global_label_map = {'Line Item - Start Data':'Line Item - Start Date','line item - start data':'Line Item - Start Date'}
    def __init__(self, data_root: Optional[str|Path]=None, labels_to_keep: Optional[Iterable[str]]=None, label_map: Optional[Mapping[str,str]]=None, columns: Optional[LoaderColumns]=None, strict: bool=False, keep_raw_ocr: bool=False):
        self.data_root = Path(data_root) if data_root is not None else None
        self.columns = columns or LoaderColumns(); self.strict = strict; self.keep_raw_ocr = keep_raw_ocr
        self.label_map = {**self.global_label_map, **self.default_label_map, **(dict(label_map) if label_map else {})}
        self.labels_to_keep = {self.normalize_label_name(x) for x in labels_to_keep} if labels_to_keep is not None else None
    def load_row(self, row: Mapping[str,Any]|Any, training: bool=True) -> CanonicalDocument:
        rec = self._as_mapping(row)
        doc_id = self._get_first_value(rec, self.columns.doc_id) or self._get_first_value(rec, self.columns.ocr_path) or f'{self.dataset_name}_unknown'
        ocr = self._load_ocr_payload(rec)
        text = self._get_first_value(rec, self.columns.text)
        text = '' if self._is_missing(text) else str(text)
        if not text: text = self.extract_text_from_ocr(ocr)
        annotations = self.parse_annotations(self._get_first_value(rec, self.columns.labels)) if training else None
        md = self._metadata_from_row(rec)
        if self.keep_raw_ocr: md['raw_ocr'] = ocr
        return CanonicalDocument(doc_id=str(doc_id), dataset_name=self.dataset_name, text=text, tokens=self.extract_tokens(ocr), blocks=self.extract_blocks(ocr), annotations=annotations, metadata=md)
    def iter_documents(self, rows: Iterable[Mapping[str,Any]|Any], training: bool=True):
        for row in rows: yield self.load_row(row, training)
    def normalize_label_name(self, label: str) -> str:
        label = ' '.join(str(label).strip().split())
        return self.label_map.get(label, label)
    def parse_annotations(self, value: Any) -> list[Annotation]:
        if self._is_missing(value): return []
        parsed = self.parse_json_like(value)
        if parsed is None: return []
        if isinstance(parsed, dict):
            for k in ('labels','annotations','entities'):
                if isinstance(parsed.get(k), list): parsed = parsed[k]; break
        if not isinstance(parsed, list): raise ValueError(f'Expected labels list, got {type(parsed)!r}')
        anns=[]
        for item in parsed:
            if not isinstance(item, Mapping): continue
            raw = str(item.get('label','')).strip()
            if not raw: continue
            label = self.normalize_label_name(raw)
            if self.labels_to_keep is not None and label not in self.labels_to_keep: continue
            try: start,end = int(item['start']), int(item['end'])
            except Exception:
                if self.strict: raise
                continue
            anns.append(Annotation(label=label, raw_label=raw, start=start, end=end, text=str(item.get('text','')), extra={k:v for k,v in dict(item).items() if k not in {'label','start','end','text'}}))
        return sorted(anns, key=lambda a:(a.start,a.end,a.label))
    def extract_text_from_ocr(self, ocr: Any) -> str:
        pages=[]
        for unit in self._iter_ocr_units(ocr):
            for page in unit.get('pages',[]) or []:
                if isinstance(page, Mapping): pages.append((int(page.get('page_num',len(pages))), str(page.get('text',''))))
        return '\n\n'.join(t for _,t in sorted(pages))
    def extract_page_sizes(self, ocr: Any) -> dict[int,dict[str,int]]:
        sizes={}
        for unit in self._iter_ocr_units(ocr):
            for page in unit.get('pages',[]) or []:
                if not isinstance(page, Mapping): continue
                size = page.get('size',{}) or {}
                if size.get('width') is not None and size.get('height') is not None:
                    sizes[int(page.get('page_num',len(sizes)))] = {'width':int(size['width']), 'height':int(size['height'])}
        return sizes
    def extract_tokens(self, ocr: Any) -> list[OCRToken]:
        sizes = self.extract_page_sizes(ocr); toks=[]
        for unit in self._iter_ocr_units(ocr):
            for raw in unit.get('tokens',[]) or []:
                if isinstance(raw, Mapping):
                    tok = self._parse_token(raw, sizes)
                    if tok is not None: toks.append(tok)
        return sorted(toks, key=lambda t:(t.page,t.start,t.y0,t.x0))
    def extract_blocks(self, ocr: Any) -> list[OCRBlock]:
        blocks=[]; bid=0
        for unit in self._iter_ocr_units(ocr):
            for raw in unit.get('blocks',[]) or []:
                if isinstance(raw, Mapping):
                    blk = self._parse_block(raw, bid)
                    if blk is not None: blocks.append(blk); bid += 1
        return sorted(blocks, key=lambda b:(b.page,b.start,b.bbox[1],b.bbox[0]))
    def parse_json_like(self, value: Any) -> Any:
        if self._is_missing(value): return None
        if isinstance(value,(dict,list)): return value
        if isinstance(value, bytes): value = value.decode('utf-8')
        if not isinstance(value, str): return value
        s=value.strip()
        if not s: return None
        for fn in (json.loads, ast.literal_eval):
            try: return fn(s)
            except Exception: pass
        if self.strict: raise ValueError('Could not parse JSON-like value.')
        return s
    def load_json_file(self, path: str|Path) -> Any:
        p = Path(path)
        if not p.is_absolute() and self.data_root is not None: p = self.data_root / p
        if not p.exists(): raise FileNotFoundError(f'OCR file not found: {p}')
        if p.suffix == '.gz':
            with gzip.open(p,'rt',encoding='utf-8') as f: return json.load(f)
        with open(p,'r',encoding='utf-8') as f: return json.load(f)
    def _load_ocr_payload(self, rec: Mapping[str,Any]) -> Any:
        value = self._get_first_value(rec, self.columns.ocr_json)
        if not self._is_missing(value):
            parsed = self.parse_json_like(value)
            return self.load_json_file(parsed) if isinstance(parsed,str) else parsed
        path = self._get_first_value(rec, self.columns.ocr_path)
        if self._is_missing(path):
            if self.strict: raise ValueError('No OCR JSON or OCR path found.')
            return []
        return self.load_json_file(str(path))
    def _parse_token(self, raw: Mapping[str,Any], page_sizes: Mapping[int,Mapping[str,int]]) -> Optional[OCRToken]:
        off = raw.get('doc_offset',{}) or {}; bbox = self._position_to_bbox(raw.get('position',{}) or {})
        try: start,end = int(off.get('start')), int(off.get('end'))
        except Exception: return None
        if bbox is None: return None
        page = int(raw.get('page_num',0)); size = page_sizes.get(page,{})
        po = raw.get('page_offset',{}) or {}; bo = raw.get('block_offset',{}) or {}
        return OCRToken(text=str(raw.get('text','')), start=start, end=end, page=page, bbox=bbox, page_width=size.get('width'), page_height=size.get('height'), page_start=self._maybe_int(po.get('start')), page_end=self._maybe_int(po.get('end')), block_start=self._maybe_int(bo.get('start')), block_end=self._maybe_int(bo.get('end')), style=dict(raw.get('style',{}) or {}), extra={k:v for k,v in dict(raw).items() if k not in {'text','doc_offset','page_offset','block_offset','position','style','page_num'}})
    def _parse_block(self, raw: Mapping[str,Any], block_id: int) -> Optional[OCRBlock]:
        off = raw.get('doc_offset',{}) or {}; bbox = self._position_to_bbox(raw.get('position',{}) or {})
        try: start,end = int(off.get('start')), int(off.get('end'))
        except Exception: return None
        if bbox is None: return None
        po = raw.get('page_offset',{}) or {}
        return OCRBlock(text=str(raw.get('text','')), start=start, end=end, page=int(raw.get('page_num',0)), bbox=bbox, block_id=block_id, block_type=raw.get('block_type'), page_start=self._maybe_int(po.get('start')), page_end=self._maybe_int(po.get('end')), extra={k:v for k,v in dict(raw).items() if k not in {'text','doc_offset','page_offset','position','page_num','block_type'}})
    @staticmethod
    def _position_to_bbox(pos: Mapping[str,Any]) -> Optional[list[int]]:
        def get(*names):
            for n in names:
                if n in pos and pos[n] is not None: return int(pos[n])
            return None
        vals=[get('left','bbLeft','x0'), get('top','bbTop','y0'), get('right','bbRight','x1'), get('bottom','bbBot','y1')]
        return None if any(v is None for v in vals) else vals
    def _metadata_from_row(self, rec: Mapping[str,Any]) -> dict[str,Any]:
        imgs = self.parse_json_like(self._get_first_value(rec, self.columns.image_files))
        return {'source_columns': list(rec.keys()), 'ocr_path': self._none_if_missing(self._get_first_value(rec, self.columns.ocr_path)), 'image_files': imgs if imgs is not None else []}
    @staticmethod
    def _iter_ocr_units(ocr: Any):
        if ocr is None: return []
        if isinstance(ocr, Mapping): return [ocr]
        if isinstance(ocr, list): return ocr
        return []
    @staticmethod
    def _as_mapping(row):
        if isinstance(row, Mapping): return row
        if hasattr(row,'to_dict'): return row.to_dict()
        raise TypeError(f'Expected mapping-like row, got {type(row)!r}')
    @classmethod
    def _get_first_value(cls, rec: Mapping[str,Any], candidates: Sequence[str]) -> Any:
        for k in candidates:
            if k in rec and not cls._is_missing(rec[k]): return rec[k]
        return None
    @staticmethod
    def _is_missing(v: Any) -> bool:
        if v is None: return True
        if isinstance(v,float) and math.isnan(v): return True
        try:
            if v is not None and v != v: return True
        except Exception: pass
        return False
    @classmethod
    def _none_if_missing(cls, v: Any) -> Any: return None if cls._is_missing(v) else v
    @staticmethod
    def _maybe_int(v: Any) -> Optional[int]:
        try: return None if v is None else int(v)
        except Exception: return None
