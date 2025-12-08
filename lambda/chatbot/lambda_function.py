import os
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import boto3
from boto3.dynamodb.conditions import Key
from pinecone import Pinecone


# =========================
# I. CONFIGURATION (FROM ENVIRONMENT VARIABLES)
# =========================

# 1. DynamoDB
DDB_MESSAGE_TABLE = os.environ.get("DDB_MESSAGE_TABLE", "sorcererxstreme-chatMessages")

# 2. Bedrock Models
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-micro-v1:0")
BEDROCK_EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")

# 3. Pinecone Config
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

# =========================
# II. GLOBAL CLIENTS
# =========================

dynamodb = boto3.resource("dynamodb")
ddb_table = dynamodb.Table(DDB_MESSAGE_TABLE)
bedrock = boto3.client("bedrock-runtime")

pc_index = None
if PINECONE_API_KEY and PINECONE_HOST:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pc.Index(host=PINECONE_HOST)
    except Exception as e:
        print(f"INIT ERROR: Không thể kết nối Pinecone. {e}")
else:
    print("WARNING: Thiếu PINECONE_API_KEY hoặc PINECONE_HOST trong Env Vars.")

# =========================
# III. HELPER FUNCTIONS
# =========================

def load_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Lấy lịch sử chat từ DynamoDB"""
    try:
        resp = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            ScanIndexForward=False, 
            Limit=limit,
        )
        return resp.get("Items", [])[::-1] 
    except Exception as e:
        print(f"DB Load Error: {e}")
        return []

def append_message(session_id: str, role: str, content: str) -> None:
    """Ghi tin nhắn vào DynamoDB"""
    try:
        item = {
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "content": content,
        }
        ddb_table.put_item(Item=item)
    except Exception as e:
        print(f"DB Save Error: {e}")

def embed_query(text: str) -> List[float]:
    """Vector hóa câu hỏi bằng Cohere"""
    if not text: return []
    text = text[:2000] 
    
    body = json.dumps({
        "texts": [text], 
        "input_type": "search_query"
    })

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="*/*"
        )
        resp_body = json.loads(resp["body"].read())
        return resp_body["embeddings"][0]
    except Exception as e:
        print(f"Embed Error: {e}")
        return []

def query_pinecone_rag(question: str, top_k: int = 3, min_score: float = 0.35) -> List[Dict[str, Any]]:
    """Tìm kiếm vector tương đồng trên Pinecone"""
    if not pc_index:
        return []

    vector_values = embed_query(question)
    if not vector_values: return []

    try:
        results = pc_index.query(
            vector=vector_values,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        print(f"Pinecone Query Error: {e}")
        return []

    docs = []
    for match in results.get('matches', []):
        if match['score'] < min_score: continue
        
        md = match.get('metadata', {})
        content = md.get('context_str') or md.get('text') or md.get('content') or str(md)
        title = md.get('entity_name') or md.get('title') or 'Unknown'
        
        docs.append({
            "id": match['id'],
            "score": match['score'],
            "title": title,
            "content": content
        })
    return docs

def build_prompt(
    user_context: Dict, 
    partner_context: Dict, 
    rag_docs: List[Dict], 
    history: List[Dict], 
    question: str
) -> Tuple[str, str]:
    
    # 1. Chuẩn bị dữ liệu Input (Vẫn nạp đủ để AI tính toán, nhưng cấm AI nói ra)
    u_name = user_context.get("name", "Bạn")
    u_info = f"Name: {u_name}, {user_context.get('birth_date')}, {user_context.get('birth_time')}, {user_context.get('birth_place')}"
    
    # Dù có tên Partner, ta vẫn format string này để AI hiểu ngữ cảnh
    p_name = partner_context.get("name", "Người ấy") if partner_context else "Không có"
    p_info = ""
    if partner_context:
        p_info = f"Partner Name: {p_name}, {partner_context.get('birth_date')}, {partner_context.get('birth_time')}, {partner_context.get('birth_place')}"

    rag_text = "\n".join([f"- {d['content']}" for d in rag_docs]) if rag_docs else "Không có dữ liệu RAG."
    
    # Lấy lịch sử, user name trong lịch sử cũng sẽ được filter bởi rule bên dưới
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history[-5:]])

    # ---------------------------------------------------------
    # 2. SYSTEM PROMPT (CẬP NHẬT: LUẬT GIẤU TÊN PARTNER)
    # ---------------------------------------------------------
    system_prompt = """
# ROLE
Bạn là chuyên gia tổng hợp Huyền học (Tử Vi, Thần Số Học, Chiêm Tinh). 

# CRITICAL RULES (TUÂN THỦ TUYỆT ĐỐI)
1. **PRIVACY & NAMING:** - KHÔNG nhắc lại ngày/giờ/nơi sinh.
   - Với User: Gọi bằng Tên riêng (VD: "Chào Lâm Anh").
   - Với Partner: **TUYỆT ĐỐI KHÔNG DÙNG TÊN THẬT** trong câu trả lời (dù input có cung cấp). Hãy thay thế bằng: **"Người ấy"**, **"Đối phương"**, hoặc **"Bạn mình"**.
2. **MULTI-DISCIPLINARY:** Phân tích kết hợp 3 góc độ:
   - Thần số học (Số chủ đạo).
   - Chiêm tinh (Cung hoàng đạo).
   - Tử Vi (Tuổi/năm hạn).
3. **DIRECTNESS:** Đi thẳng vào vấn đề, ngắn gọn, súc tích.

# RESPONSE FORMAT (MẪU CÂU TRẢ LỜI)
Hãy cấu trúc câu trả lời theo dạng sau:

"Chào [Tên User], về câu hỏi của bạn đối với [Người ấy]:

1. **Góc nhìn Thần số học:** - Số chủ đạo của bạn là [X] (tính cách...), còn của [Người ấy] là [Y] (tính cách...). 
   - [Đánh giá độ hợp/xung khắc].

2. **Góc nhìn Tử Vi & Chiêm Tinh:**
   - [Phân tích cung hoàng đạo và vận hạn].
   - [Nhận định vấn đề].

3. **Lời khuyên tổng kết:**
   - [Hành động nên làm]."
"""

    # ---------------------------------------------------------
    # 3. USER PROMPT
    # ---------------------------------------------------------
    user_prompt = f"""
[USER DATA]
{u_info}

[PARTNER DATA]
{p_info}

[CONTEXT/RAG]
{rag_text}

[HISTORY]
{history_text}

[QUESTION]
"{question}"
"""
    return system_prompt.strip(), user_prompt.strip()

def call_bedrock_nova(system_prompt: str, user_prompt: str) -> str:
    """Gọi Amazon Nova (Lite/Pro/Micro)"""
    
    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 1500,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "system": [{"text": system_prompt}],
        "messages": [
            {"role": "user", "content": [{"text": user_prompt}]}
        ]
    })

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_LLM_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        resp_body = json.loads(resp["body"].read())
        return resp_body["output"]["message"]["content"][0]["text"]

    except Exception as e:
        print(f"Nova Error ({BEDROCK_LLM_MODEL_ID}): {e}")
        return "Xin lỗi, kết nối với vũ trụ Nova đang bị gián đoạn. Vui lòng thử lại sau."

# =========================
# IV. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    print("DEBUG Event:", json.dumps(event))
    
    # 1. Parse Body
    if isinstance(event, dict) and "body" in event and isinstance(event["body"], str):
        try:
            payload = json.loads(event["body"])
        except: 
            return {"statusCode": 400, "body": "Invalid JSON Body"}
    else:
        payload = event

    # 2. Extract Data
    try:
        user_ctx = payload.get("user_context", {})
        partner_ctx = payload.get("partner_context", {})
        data_block = payload.get("data", {})
        
        session_id = data_block.get("sessionId")
        question = data_block.get("question")

        if not session_id or not question:
            raise ValueError("Missing sessionId or question")
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

    # 3. Process Logic
    try:
        # Load History
        history = load_history(session_id, limit=6)
        
        # Save User Question
        append_message(session_id, "user", question)
        
        # RAG Search
        rag_docs = query_pinecone_rag(question)
        
        # Build Prompt (New Logic)
        sys_prompt, user_prompt = build_prompt(user_ctx, partner_ctx, rag_docs, history, question)
        
        # Call AI
        reply = call_bedrock_nova(sys_prompt, user_prompt)
        
        # Save AI Reply
        append_message(session_id, "assistant", reply)

        # 4. Return
        result = {
            "sessionId": session_id,
            "reply": reply
        }
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result, ensure_ascii=False)
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "Internal Server Error"})}  