from src.ingestion.document_classifier import classify_document
from src.ingestion.text_quality import needs_ocr
def test_document_classifier():
 assert classify_document('x.pdf','Cambridge International AS & A Level Computer Science Mark Scheme')=='MARK_SCHEME'
 assert classify_document('x.pdf','This question paper has 10 questions')=='QUESTION_PAPER'
def test_ocr_trigger(): assert needs_ocr('')
