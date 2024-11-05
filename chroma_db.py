from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import json
from langchain_community.vectorstores.utils import filter_complex_metadata

# --------------------------
#  初始化模型函式
# --------------------------
def initialize_models():
    embedding_model = OllamaEmbeddings(model='jeffh/intfloat-multilingual-e5-large:f16')
    return embedding_model

# --------------------------
#  讀取和處理資料函式
# --------------------------
def read_and_process_data(file_path):
    # 讀取 Excel 文件
    df = pd.read_excel(file_path, header=None)  # 使用 pandas 讀取 Excel 文件
    # 將每一行的內容轉換為 Document
    docs = [
        Document(page_content=f"輸入: {row[0]} 輸出: {row[1]}",metadata={"category": "QA問答"})
        for index, row in df.iterrows()
    ]
    return docs

# --------------------------
#  讀取和處理資料函式(pdf)
# --------------------------
def read_and_process_pdf(file_path):
    reader = PdfReader("條款.pdf")
    docs=[]
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        # 將文本包裝成 Chroma 需要的格式
        if text:  # 確保頁面有內文內容
            docs.append({
                "content": text,
                "metadata": {
                    "source": "富邦人壽星美富足外幣分紅終身壽險",
                    "page": page_num + 1  # 包含頁碼以便追溯
                }
            })
     
    return docs
# --------------------------
#  讀取和處理資料函式(json)
# --------------------------
def read_and_process_json(file_path):
    # # 讀取 JSON 檔案
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    # docs = []  # 初始化 docs 列表

    # # 處理每個物件
    # for item in data:
    #     if item['type'] == 'insurance':
    #         insurance_data = item['data']
    #         doc = Document(
    #             page_content=(
    #                 f"Product ID: {insurance_data['product_id']}, "
    #                 f"Product Name: {insurance_data['product_name']}, "
    #                 f"Insurance Type: {insurance_data['Ins_type']}, "
    #                 f"Age Limit: {insurance_data['insurance_details']['age_limit']['min']} - "
    #                 f"{insurance_data['insurance_details']['age_limit']['max']}, "
    #                 f"Features: {', '.join(insurance_data['insurance_details']['features'])}, "
    #                 f"Payment Terms: {', '.join(insurance_data['insurance_details']['payment_terms']['payment_periods'])}, "
    #                 f"Coverage Term: {insurance_data['insurance_details']['coverage']['coverage_term']}, "
    #                 f"Currency: {insurance_data['insurance_details']['coverage']['currency']}"
    #             ),
    #             metadata={
    #                 "category": "insurance",
    #                 "product_id": insurance_data['product_id'],
    #                 "product_name": insurance_data['product_name'],
    #                 "Ins_type": insurance_data['Ins_type'],
    #                 "age_limit_min": insurance_data['insurance_details']['age_limit']['min'],
    #                 "age_limit_max": insurance_data['insurance_details']['age_limit']['max'],
    #                 "features": ', '.join(insurance_data['insurance_details']['features']),
    #                 "payment_terms": ', '.join(insurance_data['insurance_details']['payment_terms']['payment_periods']),
    #                 "coverage_term": insurance_data['insurance_details']['coverage']['coverage_term'],
    #                 "currency": insurance_data['insurance_details']['coverage']['currency']
    #             }
    #         )
    #         docs.append(doc)  # 將文件添加到 docs 列表

    #     elif item['type'] == 'client':
    #         client_data = item['data']
    #         doc = Document(
    #             page_content=(
    #                 f"Customer ID: {client_data['customer_id']}, "
    #                 f"Name: {client_data['name']}, "
    #                 f"Age: {client_data['age']}, "
    #                 f"Occupation: {client_data['occupation']}, "
    #                 f"Interests: {', '.join(client_data['interests'])}, "
    #                 f"Financial Goals: {', '.join(client_data['financial_goals'])}, "
    #                 f"Product Preference: {client_data['product_preferences']['product_name']}, "
    #                 f"Preferences: {', '.join(client_data['product_preferences']['preferences'])}"
    #             ),
    #             metadata={
    #                 "category": "client",
    #                 "client_type": client_data['client_type'],
    #                 "customer_id": client_data['customer_id'],
    #                 "name": client_data['name'],
    #                 "age": client_data['age'],
    #                 "occupation": client_data['occupation'],
    #                 "interests": ', '.join(client_data['interests']),
    #                 "financial_goals": ', '.join(client_data['financial_goals']),
    #                 "product_name": client_data['product_preferences']['product_name'],
    #                 "preferences": ', '.join(client_data['product_preferences']['preferences'])
    #             }
    #         )
    #         docs.append(doc)  # 將文件添加到 docs 列表

    # return docs  # 返回 docs 列表
    # 讀取 JSON 檔案
    with open(file_path, 'r', encoding='utf-8') as file:
        customer_types_data = json.load(file)
    docs = [] 
    # 現在添加客戶類型資料
    for customer_type in customer_types_data:
        characteristics = customer_type['characteristics']
        doc = Document(
            page_content=(
                f"客戶類型: {customer_type['customer_type']}, "
                f"財務目標: {', '.join(characteristics['financial_goals'])}, "
                f"保險偏好: {', '.join([pref['preference'] for pref in characteristics['insurance_preferences']])}, "
                f"資訊需求: {', '.join(characteristics['information_needs'])}, "
                f"溝通風格: {', '.join(characteristics['communication_style'])}, "
                f"決策因素: {', '.join(characteristics['decision_factors'])}"
            ),
             metadata={
            "category": "customer_type",
            "customer_type": customer_type['customer_type'],
            "financial_goals": ', '.join(characteristics['financial_goals']),
            # 使用 '; '.join(...) 將每個偏好的 preference 和 description 組合成一個字元串
            "insurance_preferences": '; '.join([f"{pref['preference']}: {pref['description']}" for pref in characteristics['insurance_preferences']]),
            "information_needs": ', '.join(characteristics['information_needs']),
            "communication_style": ', '.join(characteristics['communication_style']),
            "decision_factors": ', '.join(characteristics['decision_factors'])
        }
    )
        docs.append(doc)

    return docs
# --------------------------
#  建立 Chroma 向量儲存函式
# --------------------------
def create_vector_store(docs, embedding_model):
    print("Creating Chroma vector store...")
    # 設定 ChromaDB 的持久化儲存
    try:
        # 初始化 Chroma 並傳入嵌入模型
        vectorstore = Chroma(
            persist_directory=r"C:\Users\IMproject\Desktop\chromadb",
            embedding_function=embedding_model  # 提供嵌入函數
        )
        # 將文檔添加到向量儲存中
        vectorstore.add_documents(documents=docs)  # 這裡不需要提供嵌入，因為 Chroma 會自動處理
        print("Vectorstore created successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        return None

# --------------------------
#  主程式
# --------------------------
def main():
    embedding_model = initialize_models()  # 初始化模型
    file_path = 'client_type.json'  # 指定 Excel 文件的路徑
    docs = read_and_process_json(file_path)
    create_vector_store(docs, embedding_model)  # 傳入 embedding_model

if __name__ == "__main__":
    main()
