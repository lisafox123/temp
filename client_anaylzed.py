from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
import jieba
import numpy as np
import collections
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from QA_chat import initialize_models,load_vector_store,chat_logic


# --------------------------
#  建立檢索器函式
# --------------------------
def create_retriever(vectorstore):
    if vectorstore:
        retriever = vectorstore.as_retriever(
    metadata_filters={"type": {"$in": ["insurance", "client", "customer_type"]}}
)

        print("Retriever created successfully.")
        return retriever
    return None

def create_qa_chain(llm, retriever):
    # 動態生成格式化的提示詞模板
    prompt_template = ChatPromptTemplate.from_messages([
    ("user", "我的問題：{input}")
    ,("system", (
        "你是一位知識型保險客戶，正在向業務員諮詢問題。請只以顧客的身份回答問題，不要提供專業建議，也不要扮演業務員的角色。回答時表現出作為顧客的疑惑、關切或需求。\n"
        "知識型客戶類型：{context}\n"
        "業務：{input}\n"
        "知識型客戶："
    ))
    ])

    # create_stuff_documents_chain常會要求一個 ChatPromptTemplate 類型的輸入，而非字串格式的輸入
    document_chain = create_stuff_documents_chain(llm, prompt_template)  # 動態傳入知識型詞彙)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user", "{input}"),
        ("system",  "根據以上對話內容，請根據歷史紀錄和上下文來提供連貫、相關的回答。"
        "請務必保持前後文一致，並結合過去對話中的重點進行回應。"
        "若有已討論過的主題或相關資訊，請加以參考，以產生更具關聯性的回答。")
    ])
    # 對話紀錄做成可以被檢索的向量
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    qa_chain = create_retrieval_chain(retriever_chain, document_chain)
    return qa_chain

# --------------------------
#  主程式
# --------------------------
def main():
    
    llm,embedings = initialize_models()  # 初始化模型
    persist_directory = r"C:\Users\IMproject\Desktop\chromadb"
    vectorstore = load_vector_store(persist_directory,embedings)
    retriever = create_retriever(vectorstore)  # 建立檢索器
    qa_chain = create_qa_chain(llm, retriever)  # 建立問答鏈
    chat_logic(qa_chain)  # 開始聊天

if __name__ == "__main__":
    main()