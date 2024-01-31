import os
from constant import PALM_MODEL,FAQ_FILE,INSTRUCTOR_EMBEDDING,VECTORDB_PATH
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model=PALM_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0)

PaLM_embeddings = GooglePalmEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))
'''
if you want you can try instructor embeddings also. Below is thge code :

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name=INSTRUCTOR_EMBEDDING,query_instruction="Represent the query for retrieval: "
)
'''
def create_vector_db():
    loader = CSVLoader(file_path=FAQ_FILE)
    data = loader.load()
    vectordb = FAISS.from_documents(documents = data,embedding=PaLM_embeddings)
    vectordb.save_local(VECTORDB_PATH)

def get_qa_chain():
    vectordb = FAISS.load_local(VECTORDB_PATH,PaLM_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))
