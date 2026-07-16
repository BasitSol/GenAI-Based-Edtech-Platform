from __future__ import annotations
import json, re
from pathlib import Path
import fitz
from src.core import AUTHORITY, ROOT, load_yaml, sha256, write_csv, write_jsonl
from .cleaner import blocks, clean_text
from .document_classifier import classify_document
from .metadata_parser import extract_metadata
from .text_quality import needs_ocr, quality_score
from .ocr import extract_with_fallback

MANIFEST_FIELDS=["document_id","source_path","source_filename","checksum_sha256","level","qualification","subject_code","document_type","year","session","paper_number","component","variant","syllabus_valid_from","syllabus_valid_to","page_count","authority_level","exact_pair_id","native_text_available","ocr_required_pages","processing_status"]
def _configured_sources(config: dict):
    for group in config["sources"].values():
        for entry in group.values(): yield entry
def _source_for(path: Path, config: dict):
    normalized=path.as_posix().lower()
    for source in _configured_sources(config):
        if source["folder"].lower() in normalized:
            contains=source.get("filename_contains", [])
            if not contains or any(value.lower() in path.name.lower() for value in contains): return source
    return {"level":"UNKNOWN","subject_code":"","expected_document_type":"UNKNOWN"}
def _secondary_text(pdf: Path,index: int) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as document:
            return document.pages[index].extract_text(layout=True) or ""
    except Exception:
        return ""

def _save_figure_page(page,document_id:str,page_number:int,processed:Path) -> str:
    target=processed/"figures"/document_id/f"page_{page_number:04d}.png"
    target.parent.mkdir(parents=True,exist_ok=True)
    pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
    pix.save(target)
    return str(target.relative_to(ROOT)).replace("\\","/")
def build_corpus(config_path: Path = ROOT / "configs/data_sources.yaml", force: bool=False) -> dict:
    config=load_yaml(config_path); raw_root=(ROOT/config["raw_data_root"]).resolve(); processed=(ROOT/config["processed_data_root"]).resolve()
    manifests=[]; unknown=[]; ocr_rows=[]
    for pdf in sorted(raw_root.rglob("*.pdf")):
        source=_source_for(pdf.relative_to(raw_root),config)
        document=fitz.open(pdf); preview="\n".join(document[index].get_text() for index in range(min(2, len(document))))
        doc_type=classify_document(pdf.name,preview,str(pdf.parent))
        if doc_type=="UNKNOWN": doc_type=source.get("expected_document_type","UNKNOWN")
        meta=extract_metadata(pdf,preview,source["level"],source["subject_code"],doc_type)
        page_records=[]; ocr_required=[]
        for number,page in enumerate(document,1):
            raw=page.get_text("text") or ""; native=bool(raw.strip()); used=False; attempted=False; engine=None; confidence=None; ocr_error=None; secondary=False
            raster_images=page.get_images(full=True)
            vector_drawings=page.get_drawings()
            if needs_ocr(raw):
                alternate=_secondary_text(pdf,number-1)
                if quality_score(alternate)>quality_score(raw): raw=alternate; secondary=True
            # OCR can recover text from raster content, but it cannot improve
            # already-extracted vector text or blank/vector-only pages.
            if needs_ocr(raw) and raster_images:
                attempted=True; result=extract_with_fallback(page,raw)
                engine=result.engine; confidence=result.confidence; ocr_error=result.error
                if result.text.strip() and quality_score(result.text)>=quality_score(raw): raw=result.text; used=True
                ocr_required.append(number)
            cleaned=clean_text(raw); score=quality_score(cleaned)
            # `Page.find_tables()` can crash on malformed scanned PDFs. Keep page
            # extraction robust and flag likely tables for later structured parsing.
            likely_table = bool(re.search(r"(?:\S+\s{3,}){2,}|\bTable\s+\d+", cleaned))
            # Cambridge diagrams are often PDF vector paths rather than images.
            contains_diagram=bool(raster_images) or (doc_type=="QUESTION_PAPER" and len(vector_drawings)>=15)
            figure_path=_save_figure_page(page,meta["document_id"],number,processed) if contains_diagram else None
            page_records.append({"document_id":meta["document_id"],"page_number":number,"printed_page_number":None,"document_type":doc_type,"raw_text":raw,"clean_text":cleaned,"blocks":blocks(cleaned),"native_text_available":native,"secondary_parser_used":secondary,"ocr_attempted":attempted,"ocr_used":used,"ocr_engine":engine,"ocr_confidence":confidence,"ocr_error":ocr_error,"quality_score":score,"contains_table":likely_table,"contains_code":bool(re.search(r"\b(DECLARE|PROCEDURE|FUNCTION|WHILE|FOR)\b",cleaned,re.I)),"contains_diagram":contains_diagram,"figure_path":figure_path})
            if attempted or score<.75: ocr_rows.append({"document_id":meta["document_id"],"page_number":number,"ocr_attempted":attempted,"ocr_used":used,"quality_score":score,"ocr_engine":engine,"ocr_confidence":confidence,"ocr_error":ocr_error})
        document.close(); write_jsonl(processed/"pages"/meta["document_id"]/"pages.jsonl",page_records)
        manifests.append({**meta,"source_path":str(pdf.relative_to(ROOT)).replace("\\","/"),"source_filename":pdf.name,"checksum_sha256":sha256(pdf),"page_count":len(page_records),"authority_level":AUTHORITY.get(doc_type,0),"native_text_available":any(x["native_text_available"] for x in page_records),"ocr_required_pages":";".join(map(str,ocr_required)),"processing_status":"COMPLETE"})
        if doc_type=="UNKNOWN": unknown.append({"source_path":str(pdf),"reason":"Could not classify"})
    write_csv(processed/"manifests"/"documents_manifest.csv",manifests,MANIFEST_FIELDS)
    write_csv(processed/"manifests"/"documents_checksums.csv",manifests,["document_id","source_path","checksum_sha256"])
    write_csv(processed/"manifests"/"unknown_documents.csv",unknown,["source_path","reason"])
    write_csv(processed/"manifests"/"ocr_report.csv",ocr_rows,["document_id","page_number","ocr_attempted","ocr_used","quality_score","ocr_engine","ocr_confidence","ocr_error"])
    return {"documents":len(manifests),"pages":sum(x["page_count"] for x in manifests),"unknown":len(unknown),"ocr_candidates":len(ocr_rows)}
