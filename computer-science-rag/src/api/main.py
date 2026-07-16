from fastapi import FastAPI
from src.api.schemas import AskRequest
from src.generation.answer_generator import answer_question
from src.retrieval.hybrid_retriever import HybridRetriever
app=FastAPI(title='Computer Science RAG',version='0.1.0')
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/ask')
def ask(request:AskRequest): return answer_question(**request.model_dump())
@app.post('/retrieve')
def retrieve(request:AskRequest): return HybridRetriever().retrieve(request.query,request.level,request.exam_year)
@app.get('/documents')
def documents():
 import csv
 from src.core import ROOT
 with (ROOT/'data_processed/manifests/documents_manifest.csv').open(encoding='utf-8') as f:return list(csv.DictReader(f))
@app.get('/evaluation/status')
def evaluation_status(): return {'status':'not_run','datasets_frozen':False}
