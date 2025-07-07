from typing import Any
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.base import VectorStoreRetriever
from src.agentic_prompts.interest_conversation_prompt import interest_conversation_prompt
from src.agentic_prompts.interest_extraction_prompt import extract_interest_prompt
from src.agentic_prompts.course_recommendation_prompt import recommendation_prompt

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def format_docs(docs: list[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_interest_conversation_chain():
    return interest_conversation_prompt | llm | StrOutputParser()

def build_extract_interest_chain():
    return extract_interest_prompt | llm | StrOutputParser()

def build_recommendation_chain(retriever: VectorStoreRetriever):
    def retrieve_docs(x: dict[str, Any]) -> str:
        return format_docs(retriever.invoke(x["question"]))

    input_mapper = RunnableParallel({
        "context": RunnableLambda(retrieve_docs),
        "grade": lambda x: str(x["grade"]),
        "interests": lambda x: x["interests"],
        "credit_type": lambda x: x.get("credit_type", "any"), 
        "question": lambda x: x["question"]
    })

    return input_mapper | recommendation_prompt | llm | StrOutputParser()
