"""Refresh diagram flags/images without repeating extraction or OCR."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import fitz

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ROOT,read_jsonl,write_jsonl


def main()->None:
    manifest=ROOT/"data_processed/manifests/documents_manifest.csv"
    with manifest.open(encoding="utf-8") as handle:
        documents=list(csv.DictReader(handle))
    diagrams=0
    for document in documents:
        page_path=ROOT/"data_processed/pages"/document["document_id"]/"pages.jsonl"
        records=read_jsonl(page_path)
        pdf=fitz.open(ROOT/document["source_path"])
        for index,(page,record) in enumerate(zip(pdf,records),1):
            contains=bool(page.get_images(full=True)) or (
                document["document_type"]=="QUESTION_PAPER" and len(page.get_drawings())>=15
            )
            record["contains_diagram"]=contains
            if contains:
                target=ROOT/"data_processed/figures"/document["document_id"]/f"page_{index:04d}.png"
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(target)
                record["figure_path"]=str(target.relative_to(ROOT)).replace("\\","/")
                diagrams+=1
            else:
                record["figure_path"]=None
        pdf.close(); write_jsonl(page_path,records)
    print({"documents":len(documents),"diagram_pages":diagrams})


if __name__=="__main__": main()
