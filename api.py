from fastapi import FastAPI,Form
from pydantic import BaseModel
from helper_functions import get_qa_chain

app = FastAPI()

class ResponseText(BaseModel):
    response: str


@app.get("/")
def home():
    return {"welcome":"Question and Answer System Based on Google Palm LLM and Langchain"}

@app.post("/qa")
def palm(prompt: str = Form(...)):
    try:
        chain = get_qa_chain()
        out = chain.invoke(prompt)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")