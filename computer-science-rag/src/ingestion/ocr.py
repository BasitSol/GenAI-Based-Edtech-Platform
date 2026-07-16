"""Selective OCR with PaddleOCR as primary and Tesseract as fallback."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Any

import fitz

from .text_quality import quality_score


@dataclass
class OCRResult:
    text: str = ""
    engine: str | None = None
    confidence: float | None = None
    error: str | None = None


def _page_image(page: fitz.Page, scale: float = 2.0):
    from PIL import Image
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
    return Image.frombytes("RGB",[pix.width,pix.height],pix.samples)


@lru_cache(maxsize=1)
def _paddle_engine():
    from paddleocr import PaddleOCR
    # PaddleOCR 3.x names. A compatibility fallback supports deployed 2.x images.
    try:
        return PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            # Source PDFs are already upright. Skipping the two orientation
            # classifiers keeps PP-OCRv6 medium practical on CPU without
            # weakening text detection or recognition.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(
            lang="en",
            use_angle_cls=False,
            show_log=False,
            use_gpu=False,
            enable_mkldnn=False,
        )


def _collect_paddle(value: Any) -> tuple[list[str],list[float]]:
    texts:list[str]=[]; scores:list[float]=[]
    if value is None:
        return texts,scores
    if hasattr(value,"json"):
        value=value.json
    if isinstance(value,dict):
        payload=value.get("res",value)
        rec_texts=payload.get("rec_texts") or payload.get("texts") or []
        rec_scores=payload.get("rec_scores") or payload.get("scores") or []
        texts.extend(str(item) for item in rec_texts if str(item).strip())
        scores.extend(float(item) for item in rec_scores if item is not None)
        if texts:
            return texts,scores
        for child in value.values():
            child_texts,child_scores=_collect_paddle(child); texts.extend(child_texts); scores.extend(child_scores)
        return texts,scores
    if isinstance(value,(list,tuple)):
        # PaddleOCR 2.x leaf: [box, (text, confidence)].
        if len(value)>=2 and isinstance(value[1],(list,tuple)) and value[1] and isinstance(value[1][0],str):
            texts.append(value[1][0])
            if len(value[1])>1 and isinstance(value[1][1],(int,float)): scores.append(float(value[1][1]))
            return texts,scores
        for child in value:
            child_texts,child_scores=_collect_paddle(child); texts.extend(child_texts); scores.extend(child_scores)
    return texts,scores


def paddle_ocr(page: fitz.Page) -> OCRResult:
    try:
        import numpy as np
        image=_page_image(page)
        engine=_paddle_engine()
        raw=engine.predict(np.asarray(image)) if hasattr(engine,"predict") else engine.ocr(np.asarray(image),cls=True)
        texts,scores=_collect_paddle(raw)
        return OCRResult("\n".join(texts),"paddleocr",sum(scores)/len(scores) if scores else None)
    except Exception as exc:
        return OCRResult(error=f"{type(exc).__name__}: {exc}")


def tesseract_ocr(page: fitz.Page) -> OCRResult:
    try:
        import pytesseract
        executable=shutil.which("tesseract")
        if not executable:
            for candidate in (
                Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            ):
                if candidate.exists():
                    executable=str(candidate); break
        if executable:
            pytesseract.pytesseract.tesseract_cmd=executable
        image=_page_image(page)
        data=pytesseract.image_to_data(image,output_type=pytesseract.Output.DICT,config="--oem 3 --psm 6")
        words=[]; scores=[]
        for text,confidence in zip(data.get("text",[]),data.get("conf",[])):
            if not str(text).strip(): continue
            words.append(str(text))
            try:
                value=float(confidence)
                if value>=0: scores.append(value/100)
            except (TypeError,ValueError): pass
        return OCRResult(" ".join(words),"tesseract",sum(scores)/len(scores) if scores else None)
    except Exception as exc:
        return OCRResult(error=f"{type(exc).__name__}: {exc}")


def extract_with_fallback(page: fitz.Page, native_text: str="") -> OCRResult:
    """Use the best readable result, preferring PaddleOCR on equal quality."""
    paddle=paddle_ocr(page)
    if paddle.text.strip() and quality_score(paddle.text)>=max(.75,quality_score(native_text)):
        return paddle
    tesseract=tesseract_ocr(page)
    candidates=[item for item in (paddle,tesseract) if item.text.strip()]
    if candidates:
        return max(candidates,key=lambda item:quality_score(item.text))
    errors="; ".join(filter(None,[paddle.error,tesseract.error]))
    return OCRResult(error=errors or "No OCR text returned")
