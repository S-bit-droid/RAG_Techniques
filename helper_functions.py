from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from google.api_core.exceptions import ResourceExhausted  # instead of openai.RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum


def replace_t_with_space(list_of_documents):
    """Replaces all tab characters ('\t') with spaces in document page content."""
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def text_wrap(text, width=120):
    """Wraps the input text to the specified width."""
    return textwrap.fill(text, width=width)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """Encodes a PDF into a FAISS vectorstore with Gemini embeddings."""
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """Encodes a string into FAISS vectorstore using Gemini embeddings."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"Encoding error: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """Retrieves relevant context for a given question."""
    docs = chunks_query_retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]


class QuestionAnswerFromContext(BaseModel):
    """Model to generate an answer from given context."""
    answer_based_on_content: str = Field(
        description="Answer to a query based only on the provided context."
    )


def create_question_answer_from_context_chain():
    """Creates a chain for answering questions using Gemini."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    question_answer_prompt_template = """ 
    For the question below, provide a concise answer based ONLY on the provided context:
    {context}
    Question:
    {question}
    """

    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    return (
        question_answer_from_context_prompt
        | llm.with_structured_output(QuestionAnswerFromContext)
    )


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """Answer a question using context by invoking the chain."""
    input_data = {"question": question, "context": context}
    print("Answering question from context...")
    output = question_answer_from_context_chain.invoke(input_data)
    return {"answer": output.answer_based_on_content, "context": context, "question": question}


def show_context(context):
    """Prints retrieved context chunks."""
    for i, c in enumerate(context):
        print(f"Context {i + 1}:\n{c}\n")


def read_pdf_to_string(path):
    """Reads a PDF file and returns its text as a string."""
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """Perform BM25 retrieval for top-k relevant chunks."""
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    return [cleaned_texts_]()
