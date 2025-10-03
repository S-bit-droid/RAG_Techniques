from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatGoogle
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from typing import List
import fitz
import textwrap
import asyncio
import random
import numpy as np
from enum import Enum
from rank_bm25 import BM25Okapi

# --------------------------
# Helper Functions
# --------------------------

def replace_t_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents

def text_wrap(text, width=120):
    return textwrap.fill(text, width=width)

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    embeddings = GoogleEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore

def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.create_documents([content])
    for chunk in chunks:
        chunk.metadata['relevance_score'] = 1.0
    embeddings = GoogleEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def retrieve_context_per_question(question, chunks_query_retriever):
    docs = chunks_query_retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]

def show_context(context):
    for i, c in enumerate(context):
        print(f"Context {i+1}:\n{c}\n")

def read_pdf_to_string(path):
    doc = fitz.open(path)
    content = ""
    for page in doc:
        content += page.get_text()
    return content

# --------------------------
# Structured Output Model
# --------------------------
class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )

def create_question_answer_from_context_chain(llm):
    question_answer_prompt_template = """ 
    For the question below, provide a concise answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """
    prompt = PromptTemplate(template=question_answer_prompt_template, input_variables=["context", "question"])
    chain = prompt | llm.with_structured_output(QuestionAnswerFromContext)
    return chain

def answer_question_from_context(question, context, chain):
    input_data = {"question": question, "context": context}
    output = chain.invoke(input_data)
    return {"answer": output.answer_based_on_content, "context": context, "question": question}

# --------------------------
# Async Retry Utilities
# --------------------------
async def exponential_backoff(attempt):
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    await asyncio.sleep(wait_time)

async def retry_with_exponential_backoff(coroutine, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await coroutine
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await exponential_backoff(attempt)

# --------------------------
# Enums & Embedding Provider
# --------------------------
class EmbeddingProvider(Enum):
    GOOGLE = "google"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"

class ModelProvider(Enum):
    GOOGLE = "google"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"

def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    if provider == EmbeddingProvider.GOOGLE:
        return GoogleEmbeddings(model=model_id) if model_id else GoogleEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
