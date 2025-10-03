import json
from typing import List, Dict, Any
from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question,
    get_langchain_embedding_provider
)
from langchain.chat_models import ChatGoogle
from langchain import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    llm = ChatGoogle(temperature=0, model="chat-bison-001")

    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.
    
    Question: {question}
    Retrieved Context: {context}
    
    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance
    2. Completeness
    3. Conciseness
    
    Provide ratings in JSON format:
    """)

    eval_chain = eval_prompt | llm | StrOutputParser()

    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")
    
    results = []
    for question in questions:
        context_docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context_docs])
        eval_result = eval_chain.invoke({"question": question, "context": context_text})
        results.append(eval_result)
    
    return {"questions": questions, "results": results}

if __name__ == "__main__":
    pass
