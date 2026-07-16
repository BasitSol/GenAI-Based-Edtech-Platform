from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
from src.core import tokens
class BM25Index:
    def __init__(self,chunks):
        self.chunks=chunks; self.docs=[tokens(c['text']) for c in chunks]; self.df=Counter(t for d in self.docs for t in set(d)); self.avgdl=sum(map(len,self.docs))/max(1,len(self.docs))
    def search(self,query,k=15):
        q=tokens(query); n=len(self.docs); ranked=[]
        for i,d in enumerate(self.docs):
            tf=Counter(d); score=sum(math.log(1+(n-self.df[t]+.5)/(self.df[t]+.5))*tf[t]*2.2/(tf[t]+1.2*(1-.75+.75*len(d)/self.avgdl)) for t in q if t in tf)
            if score: ranked.append((score,self.chunks[i]))
        return [x[1] for x in sorted(ranked,key=lambda x:x[0],reverse=True)[:k]]
    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(self.chunks,ensure_ascii=False),encoding='utf-8')
    @classmethod
    def load(cls,path): return cls(json.loads(path.read_text(encoding='utf-8')))
