from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

# --------------------------
#  初始化模型函式
# --------------------------
def initialize_models():
    # 初始化 Ollama 模型
    llm = OllamaLLM(model='jcai/llama3-taide-lx-8b-chat-alpha1:q6_k', temperature=0.2)
    embedding_model = OllamaEmbeddings(model='jeffh/intfloat-multilingual-e5-large:f16')
    return llm,embedding_model

# --------------------------
#  載入 Chroma 向量儲存函式
# --------------------------
def load_vector_store(persist_directory,embedding_model):
    try:
        vectorstore = Chroma(persist_directory=persist_directory,embedding_function=embedding_model)
        print("Vectorstore loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        return None

# --------------------------
#  建立檢索器函式
# --------------------------
def create_retriever(vectorstore):
    if vectorstore:
        retriever = vectorstore.as_retriever(metadata_filters={"category": "QA問答"})
        print("Retriever created successfully.")
        return retriever
    return None


# --------------------------
#  使用 create_retrieval_chain 建立問答鏈函式
# --------------------------
def create_qa_chain(llm, retriever):
    
    prompt = ChatPromptTemplate.from_messages([
    ("system",  "你是一個富邦保險專家，叫做叫做：Fubo，請用清楚簡明的方式回答以下問題：\n問題：{input}\n前後文：{context}\n回答："),
    # "你是一個富邦保險專家，叫做叫做：Fubo，請用清楚簡明的方式回答以下問題：\n問題：{input}\n前後文：{context}\n回答："
    ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user", "{input}"),
        ("system",  "根據以上對話內容，產生一個搜尋查詢，以便搜尋與對話相關的資訊")
    ])
    # 對話紀錄做成可以被檢索的向量
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    qa_chain = create_retrieval_chain(retriever_chain, document_chain)
    return qa_chain

# --------------------------
#  聊天邏輯函式
# --------------------------
def chat_logic(qa_chain):
    chat_history=[]
    input_text = input('>>> ')
    # 依據對話內容建立向量
    chat_history.append(HumanMessage(content=input_text))
    while input_text.lower() != 'bye':
        try:
            response = qa_chain.invoke({
                "input": input_text,
                "chat_history": chat_history,
                })  # 使用 `invoke` 方法
            
            # 檢查回應結構
            if isinstance(response, dict) and 'answer' in response:
                print("Response:", response['answer'])  # 回應結果
                chat_history.append(AIMessage(content=response['answer']))

            # 檢查檢索結果
            if 'context' in response and response['context']:
                print("找到的相關文件資訊:")
                for doc in response['context']:
                    print(f"文件內容: {doc.page_content}")  # 顯示文件內容
                    print("-" * 50)  # 分隔線
            else:
                print("未找到相關文件。")
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        input_text = input('>>> ')
        chat_history.append(HumanMessage(content=input_text))
    for message in chat_history:
        message.pretty_print()

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
