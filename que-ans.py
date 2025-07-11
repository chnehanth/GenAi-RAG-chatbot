# Question Answering module
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq  # or any LLM

def query_vectordb(question, persist_directory="chroma_db"):
    # Load vector store
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Get relevant docs
    docs = vectordb.similarity_search(question, k=3)

    # Combine content
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt template
    prompt = PromptTemplate.from_template(
        "Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
    )

    # Use a local or hosted LLM (Groq or OpenAI etc.)
    llm = ChatGroq(model="llama3-8b-8192")

    chain: Runnable = (
        {"context": lambda x: context, "question": lambda x: question}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({})
