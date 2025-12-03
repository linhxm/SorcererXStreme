import os
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import boto3
from boto3.dynamodb.conditions import Key
from pinecone import Pinecone


# =========================
# I. CONFIGURATION (FROM ENVIRONMENT VARIABLES)
# =========================

# 1. DynamoDB
# Default: sorcererxstreme-chatMessages
DDB_MESSAGE_TABLE = os.environ.get("DDB_MESSAGE_TABLE", "sorcererxstreme-chatMessages")

# 2. Bedrock Models
# Default: Nova Micro. Có thể đổi thành "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"
# Default: Cohere Embed Multilingual
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-micro-v1:0")
BEDROCK_EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")

# 3. Pinecone Config
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

# =========================
# II. GLOBAL CLIENTS
# =========================

# DynamoDB & Bedrock
dynamodb = boto3.resource("dynamodb")
ddb_table = dynamodb.Table(DDB_MESSAGE_TABLE)
bedrock = boto3.client("bedrock-runtime")

# Pinecone Init
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
    """Lấy lịch sử chat từ DynamoDB (Mới nhất -> Cũ nhất, sau đó đảo ngược)"""
    try:
        resp = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            ScanIndexForward=False, 
            Limit=limit,
        )
        return resp.get("Items", [])[::-1] # Đảo ngược để xếp theo trình tự thời gian
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
    text = text[:2000] # Truncate để tránh lỗi
    
    body = json.dumps({
        "texts": [text], 
        "input_type": "search_query" # Quan trọng cho Cohere v3
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
        print("Pinecone chưa được khởi tạo.")
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
        # Fallback các trường metadata nếu tên không khớp
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
    
    system_prompt = """Bạn là chuyên gia luận giải Tử Vi của ứng dụng Lasotuvi.
Nhiệm vụ: Trả lời câu hỏi người dùng dựa trên Context và RAG.
Quy tắc:
- Giọng văn: Ấm áp, sâu sắc, ngắn gọn.
- Dữ liệu: Ưu tiên dùng thông tin trong RAG. Không bịa đặt.
- Luôn kết hợp ngày giờ sinh của User/Partner để cá nhân hóa câu trả lời."""

    # Format Data
    user_str = json.dumps(user_context, ensure_ascii=False)
    partner_str = json.dumps(partner_context, ensure_ascii=False)
    
    if rag_docs:
        rag_text = "\n".join([f"- [{d['score']:.2f}] {d['title']}: {d['content']}" for d in rag_docs])
    else:
        rag_text = "Không có tài liệu tham khảo cụ thể."

    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])

    user_prompt = f"""
[USER INFO]
Me: {user_str}
Partner: {partner_str}

[KNOWLEDGE BASE]
{rag_text}

[HISTORY]
{history_text}

[QUESTION]
"{question}"
"""
    return system_prompt, user_prompt

def call_bedrock_nova(system_prompt: str, user_prompt: str) -> str:
    """Gọi Amazon Nova (Lite/Pro)"""
    
    # Cấu trúc Body đặc thù của Amazon Nova
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
        
        # Parse Response Amazon Nova
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
        
        # Build Prompt
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