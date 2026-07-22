from fastapi import FastAPI
from src.api.schemas import AskRequest
from src.generation.answer_generator import answer_question
from src.workflows.rag_graph import retrieve as retrieve_workflow
from src.observability.telemetry import TelemetryStore
from src.observability.tracing import langsmith_status
app=FastAPI(title='Enterprise Educational Computer Science RAG',version='1.0.0')
@app.get('/health')
def health():
 from src.core import current_build_path
 try:
  build=current_build_path()
  return {'status':'ready','build_id':build.name}
 except RuntimeError as exc:
  return {'status':'not_ready','reason':str(exc)}
@app.post('/ask')
def ask(request:AskRequest): return answer_question(**request.model_dump())
@app.post('/retrieve')
def retrieve(request:AskRequest): return retrieve_workflow(request.query,request.level,request.exam_year)
@app.get('/documents')
def documents():
 import csv
 from src.core import current_build_path
 with (current_build_path()/'manifests/documents.csv').open(encoding='utf-8') as f:return list(csv.DictReader(f))
@app.get('/evaluation/status')
def evaluation_status():
 from pathlib import Path
 from src.core import ROOT
 results=sorted((ROOT/'evaluation/results').glob('*.json'),key=lambda path:path.stat().st_mtime,reverse=True)
 return {'status':'available' if results else 'not_run','latest_result':str(results[0]) if results else None,'langsmith':langsmith_status()}
@app.get('/observability/summary')
def observability_summary(days:int=30): return TelemetryStore().summary(days)
