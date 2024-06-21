import os
from langchain_openai import OpenAIEmbeddings
os.environ['USER_AGENT'] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAI


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_documents_from_web(docs):
    loader = WebBaseLoader(docs)
    data= loader.load()
    # print(data)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
                 )
    splitDocs = text_splitter.split_documents(data)
    # print(splitDocs)
    return splitDocs

def embed(docs):
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding=embedding)
    return db

def chaine(x):
    # model= genai.GenerativeModel("gemini-pro")
    model= OpenAI()
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}                                       
    """)

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = x.as_retriever(search_kwargs={"k": 1})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
x=embed(docs)
print(x)
chain= chaine(x)
response= chain.invoke(
    {
    "input": "What is LCEL?",
    }
)
print(response["context"])