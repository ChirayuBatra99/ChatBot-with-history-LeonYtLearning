import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
os.environ['USER_AGENT'] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
import os
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
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
    prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Answer the user's questions based on the context: {context}"),
        ("human", "{input}"),
    ]
)

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = x.as_retriever(search_kwargs={"k": 1})
    retriever_prompt=  ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, msj, chat_history):
    response= chain.invoke({
         "input": msj,
        "chat_history": chat_history
    })
    return response
    print("1", response)


docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
docs= get_documents_from_web('https://www.bankbazaar.com/gold-rate-hoshiarpur.html')
x=embed(docs)
print(x)
chain= chaine(x)

chat_history=[]

while(True):
    msj= input("Enter message")
    if(msj.lower() == 'exit'):
        break
    resp= process_chat(chain, msj, chat_history)
    chat_history.append(HumanMessage(content=msj))
    chat_history.append(AIMessage(content=resp["answer"]))

    print(resp["answer"])