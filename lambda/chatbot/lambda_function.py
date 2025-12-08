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
    """
    Tạo Prompt theo cấu trúc Role & Objective chuẩn (Updated Logic)
    Trả về (System Prompt, User Prompt)
    """

    # 1. Parse User Data
    birth_date = user_context.get("birth_date", "Chưa rõ")
    birth_time = user_context.get("birth_time", "Chưa rõ")
    birth_place = user_context.get("birth_place", "Chưa rõ")
    
    # 2. Parse Partner Data
    # Kiểm tra nếu partner_context có dữ liệu thực sự
    has_partner = partner_context and (partner_context.get("name") or partner_context.get("birth_date"))
    
    if has_partner:
        p_name = partner_context.get("name", "Người ấy")
        p_date = partner_context.get("birth_date", "?")
        p_time = partner_context.get("birth_time", "?")
        partner_info_str = f"Tên: {p_name}, Ngày sinh: {p_date}, Giờ: {p_time}"
    else:
        partner_info_str = "Người dùng chưa cung cấp thông tin đối phương (Partner)."

    # 3. Format RAG Data
    if rag_docs:
        rag_text = "\n".join([f"- [Tài liệu độ tin cậy {d['score']:.2f}]: {d['content']}" for d in rag_docs])
    else:
        rag_text = "Không có dữ liệu tham khảo (RAG) phù hợp hoặc độ tin cậy thấp."

    # 4. Format History (Lấy 5 tin gần nhất)
    recent_history = history[-5:] if len(history) > 5 else history
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in recent_history])

    # ----------------------------------------
    # SYSTEM PROMPT: Role, Rules & Logic
    # ----------------------------------------
    system_prompt = """
# ROLE & OBJECTIVE
Bạn là chuyên gia tư vấn Tử Vi - Chiêm Tinh - Tarot AI của ứng dụng "Lasotuvi". Sứ mệnh của bạn là cung cấp lời khuyên thấu đáo, dựa trên dữ liệu cá nhân hóa, với giọng văn ấm áp, chữa lành nhưng súc tích.

# CRITICAL RULES (BẮT BUỘC TUÂN THỦ)
1. **Phạm vi trả lời:** TUYỆT ĐỐI KHÔNG tư vấn y tế (bệnh lý cụ thể), đầu tư tài chính (mua mã nào, con số cụ thể), hoặc pháp luật. Nếu gặp, hãy từ chối khéo léo và hướng người dùng đến chuyên gia thực tế.
2. **Cá nhân hóa:** Mọi phân tích PHẢI dựa trên thông tin sinh (Ngày/Giờ/Nơi sinh). Không đưa ra lời khuyên chung chung (kiểu "người tuổi Tý thường...").
3. **Ưu tiên Tình cảm:** Nếu câu hỏi liên quan đến tình yêu/hôn nhân, BẮT BUỘC kiểm tra thông tin Partner (nếu có) để phân tích độ tương hợp trước khi đưa ra lời khuyên.

# DATA HANDLING LOGIC (QUY TRÌNH XỬ LÝ)
**Bước 1: Đánh giá RAG Data**
- Đọc phần Context/RAG Data được cung cấp.
- Nếu RAG Data chứa thông tin khớp và hữu ích cho câu hỏi: Ưu tiên sử dụng 80% nội dung từ RAG, 20% diễn giải thêm.
- Nếu RAG Data rỗng, không liên quan, hoặc quá sơ sài: BỎ QUA hoàn toàn RAG. Tự động kích hoạt kiến thức chuyên sâu của bạn về Tử Vi/Chiêm tinh để lập lá số (trong tư duy) và trả lời dựa trên Birth Info.

**Bước 2: Soạn thảo câu trả lời**
- Tone: Ấm áp, thấu hiểu, như một người bạn tri kỷ nhưng có kiến thức uyên thâm.
- Format: Đi thẳng vào vấn đề. Không chào hỏi rườm rà. Dùng bullet points nếu liệt kê ý.
- Độ dài: Giữ câu trả lời CONCISE (ngắn gọn, súc tích). Tối đa 150-200 từ trừ khi người dùng yêu cầu chi tiết.

# RESPONSE TEMPLATE
(Không cần tiêu đề, trả lời trực tiếp vào nội dung)
[Lời khuyên/Dự đoán dựa trên dữ liệu sao/lá số]
[Hành động cụ thể/Lời khuyên thực tế tiếp theo]
"""

    # ----------------------------------------
    # USER PROMPT: Data, Context & Question
    # ----------------------------------------
    user_prompt = f"""
# INPUT DATA
- User Birth Info: Ngày {birth_date}, Giờ {birth_time}, Nơi sinh {birth_place}.
- Current Partner (nếu có): {partner_info_str}

# KNOWLEDGE BASE (RAG DATA)
{rag_text}

# HISTORY (CONTEXT)
{history_text}

# USER QUESTION
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