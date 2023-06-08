import argparse
import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, openai
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma

from ChatGLM import ChatGLM

load_dotenv("config.env")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
# openai.api_key = os.getenv("OPENAI_API_KEY")
from constants import CHROMA_SETTINGS



if __name__ == '__main__':
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
    llm = ChatGLM()

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
    已知内容:
    {context}
    问题:
    {question}"""

    promptA = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": promptA}
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    while True:
        query = input("\n请输入问题: ")
        if query == "exit":
            break

        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        print("\n\n> 问题:")
        print(query)
        print("\n> 回答:")
        print(answer)

        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
