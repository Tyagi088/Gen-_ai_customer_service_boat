from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import retrieve, generate
from app.scheduler import start_scheduler

app = FastAPI()

start_scheduler()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    context = retrieve(q.question)
    answer = generate(q.question, str(context))
    return {"answer": answer}
