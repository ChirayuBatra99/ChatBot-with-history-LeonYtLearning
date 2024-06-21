import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model= genai.GenerativeModel("gemini-pro")
model= ChatGoogleGenerativeAI(model="gemini-pro")
# template = ChatPromptTemplate.from_messages([
#     ("system", "You are a chef, you have to tell the color of the vegetable:{input} "),
#     ("human", "{input}"),
# ])
# model= genai.GenerativeModel("gemini-pro")

template = ChatPromptTemplate.from_messages([
     "You are a chef, you have to tell the color of the vegetable:{input} "
])
# ("system", "You are a chef, you have to tell the color of the vegetable:{input} "),
    # ("human", "{input}"),
# template = ChatPromptTemplate.from_template(
#    [
#         (
#            ""
#         ),
#         ("human", "{input}"),
#     ]
# )

# print(model.generate_content("hi"))
chain = LLMChain(llm=model, prompt=template, verbose= True)
input=  "beans"
resp= chain(input)
print(resp)
