import os
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

import boto3
from boto3.dynamodb.conditions import Key
from pinecone import Pinecone

# Import thư viện Tử Vi (Giả định folder lasotuvi nằm cùng cấp)
try:
    from lasotuvi import App, DiaBan
    from lasotuvi.AmDuong import diaChi
except ImportError:
    print("WARNING: Không tìm thấy thư viện lasotuvi.")

# =========================
# I. CONFIGURATION
# =========================
DDB_MESSAGE_TABLE = os.environ.get("DDB_MESSAGE_TABLE", "sorcererxstreme-chatMessages")
BEDROCK_LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-micro-v1:0")
BEDROCK_EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "cohere.embed-multilingual-v3")
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
        print(f"INIT ERROR: Pinecone {e}")

# =========================
# III. CALCULATION ENGINES (HUYỀN HỌC)
# =========================

def get_current_date_vn():
    return datetime.now(timezone(timedelta(hours=7)))

def calculate_numerology(date_str: str) -> dict:
    """Tính số chủ đạo"""
    try:
        if "-" in date_str:
            parts = date_str.split("-")
            d, m, y = int(parts[2]), int(parts[1]), int(parts[0])
        else:
            parts = date_str.split("/")
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        
        def sum_digits(n):
            s = sum(int(digit) for digit in str(n))
            if s == 11 or s == 22 or s == 33: return s
            return s if s < 10 else sum_digits(s)

        total = sum_digits(d) + sum_digits(m) + sum_digits(y)
        life_path = sum_digits(total)
        if life_path == 4 and total == 22: life_path = 22

        return {"life_path": str(life_path), "details": f"{d}/{m}/{y}"}
    except:
        return {"life_path": None, "details": date_str}

def calculate_zodiac(date_str: str) -> str:
    """Tính cung hoàng đạo"""
    try:
        if "-" in date_str:
            parts = date_str.split("-")
            d, m = int(parts[2]), int(parts[1])
        else:
            parts = date_str.split("/")
            d, m = int(parts[0]), int(parts[1])
            
        zodiacs = [
            (1, 20, "Ma Kết"), (2, 19, "Bảo Bình"), (3, 21, "Song Ngư"),
            (4, 20, "Bạch Dương"), (5, 21, "Kim Ngưu"), (6, 22, "Song Tử"),
            (7, 23, "Cự Giải"), (8, 23, "Sư Tử"), (9, 23, "Xử Nữ"),
            (10, 24, "Thiên Bình"), (11, 23, "Bọ Cạp"), (12, 22, "Nhân Mã")
        ]
        
        for month, day, sign in zodiacs:
            if m == month:
                return sign if d < day else zodiacs[(zodiacs.index((month, day, sign)) + 1) % 12][2]
        return "Ma Kết"
    except:
        return "Không xác định"

def get_tuvi_summary(birth_date: str, birth_time: str, gender: int) -> dict:
    """Tính tử vi cơ bản"""
    try:
        if "-" in birth_date:
            parts = birth_date.split("-")
            d, m, y = int(parts[2]), int(parts[1]), int(parts[0])
        else:
            parts = birth_date.split("/")
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        
        h = int(birth_time.split(":")[0])
        gio_chi = int((h + 1) / 2) % 12
        if gio_chi == 0: gio_chi = 12
        
        db = App.lapDiaBan(DiaBan.diaBan, d, m, y, gio_chi, gender, True, 7)
        cung_menh = db.thapNhiCung[db.cungMenh]
        
        chinh_tinh = [s['saoTen'] for s in cung_menh.cungSao if s['saoLoai'] == 1]
        
        return {
            "menh_tai": diaChi[cung_menh.cungSo]['tenChi'],
            "chinh_tinh": ", ".join(chinh_tinh) if chinh_tinh else "Vô Chính Diệu"
        }
    except Exception as e:
        print(f"TuVi Error: {e}")
        return {}

# =========================
# IV. HELPER FUNCTIONS
# =========================

def load_history(session_id: str, limit: int = 6) -> List[Dict]:
    try:
        resp = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            ScanIndexForward=False, Limit=limit,
        )
        return resp.get("Items", [])[::-1] 
    except: return []

def append_message(session_id: str, role: str, content: str):
    try:
        ddb_table.put_item(Item={
            "sessionId": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "content": content,
        })
    except: pass

def embed_query(text: str) -> List[float]:
    if not text: return []
    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=json.dumps({"texts": [text[:2000]], "input_type": "search_query"}),
            contentType="application/json", accept="*/*"
        )
        return json.loads(resp["body"].read())["embeddings"][0]
    except: return []

def query_pinecone_rag(keywords: List[str], top_k: int = 2) -> List[str]:
    if not pc_index or not keywords: return []
    search_text = " ".join(keywords)
    vector = embed_query(search_text)
    if not vector: return []
    try:
        results = pc_index.query(vector=vector, top_k=top_k, include_metadata=True)
        docs = []
        for match in results.get('matches', []):
            if match['score'] < 0.35: continue
            md = match.get('metadata', {})
            content = md.get('context_str') or md.get('content') or ""
            entity = md.get('entity_name') or ""
            docs.append(f"[{entity}]: {content}")
        return docs
    except: return []

def analyze_input(question: str) -> dict:
    result = {"is_specific_date": False, "date": None, "tarot_cards": []}
    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', question)
    if date_match:
        result["is_specific_date"] = True
        result["date"] = date_match.group(0)
    
    tarot_keywords = ["Fool", "Magician", "Empress", "Emperor", "Lover", "Chariot", "Strength", "Hermit", "Wheel", "Justice", "Hanged", "Death", "Temperance", "Devil", "Tower", "Star", "Moon", "Sun", "Judgement", "World", "Cup", "Wand", "Sword", "Pentacle"]
    found_cards = [word for word in tarot_keywords if word.lower() in question.lower()]
    if found_cards:
        result["tarot_cards"] = found_cards
    return result

def build_dynamic_prompt(question: str, user_ctx: Dict, partner_ctx: Dict, calculated_data: Dict, rag_content: List[str], history: List[Dict]) -> Tuple[str, str]:
    current_date = get_current_date_vn().strftime("%d/%m/%Y")
    user_name = user_ctx.get("name", "Bạn")
    
    partner_str = ""
    if partner_ctx:
        p_calc = calculated_data.get("partner", {})
        partner_str = f"- Đối phương (Gọi là 'Người ấy'): Số chủ đạo {p_calc.get('numerology', {}).get('life_path', 'N/A')}, Cung {p_calc.get('zodiac', 'N/A')}"

    u_calc = calculated_data.get("user", {})
    user_str = f"- User ({user_name}): Số chủ đạo {u_calc.get('numerology', {}).get('life_path', 'N/A')}, Cung {u_calc.get('zodiac', 'N/A')}, Mệnh {u_calc.get('tuvi', {}).get('menh_tai', 'N/A')}"

    spec_str = ""
    if calculated_data.get("specific_date"):
        sd = calculated_data["specific_date"]
        spec_str = f"[THÔNG TIN NGÀY {sd['details']}]: Số chủ đạo {sd['numerology']['life_path']}, Cung {sd['zodiac']}"

    tarot_str = f"[BÀI TAROT]: {calculated_data['tarot_context']}" if calculated_data.get("tarot_context") else ""
    rag_text = "\n".join(rag_content) if rag_content else "Không có."
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])

    system_prompt = f"""
# ROLE: AI Huyền Học (SorcererXstreme). Hôm nay: {current_date}.
# RULES:
1. PRIVACY: KHÔNG nhắc ngày/nơi sinh. Gọi Partner là "Người ấy".
2. CONTEXT: Ưu tiên [THÔNG TIN NGÀY] nếu có. Nếu có [BÀI TAROT], tập trung giải bài.
3. OUTPUT: Ngắn gọn, sâu sắc, chia mục rõ ràng.
"""
    user_prompt = f"""
[DATA]
{user_str}
{partner_str}
{spec_str}
{tarot_str}

[RAG]
{rag_text}

[HISTORY]
{history_text}

[QUESTION]
"{question}"
"""
    return system_prompt, user_prompt

# Hàm này được tách ra để pass 'test_success_flow' (test đang mock hàm này)
def call_bedrock_nova(system_prompt: str, user_prompt: str) -> str:
    """Gọi Amazon Nova"""
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.6},
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": [{"text": user_prompt}]}]
    })
    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_LLM_MODEL_ID,
            body=body, contentType="application/json", accept="application/json"
        )
        return json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"Nova Error: {e}")
        return "Vũ trụ đang tắc nghẽn, vui lòng thử lại."

# =========================
# V. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    # 1. Parse Input & Validate Strict (Để pass test_invalid_json_body)
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    except:
        return {"statusCode": 400, "body": "Invalid JSON Body"} # Test expect chuỗi này

    # Extract Data
    user_ctx = body.get("user_context", {})
    partner_ctx = body.get("partner_context", {})
    data_block = body.get("data", {})
    session_id = data_block.get("sessionId")
    question = data_block.get("question")
    input_cards = data_block.get("tarot_cards", [])

    # Validate Required Fields (Để pass test_missing_session_id & test_missing_question)
    # Test expect 400 nếu thiếu session hoặc question
    if not session_id or (not question and not input_cards):
         return {"statusCode": 400, "body": json.dumps({"error": "Missing sessionId or question"})}

    # 2. Process Logic
    intent = analyze_input(question or "")
    calculated_data = {"user": {}, "partner": {}}
    rag_keywords = []

    # Calculate User
    if user_ctx.get("birth_date"):
        lp = calculate_numerology(user_ctx["birth_date"])
        zd = calculate_zodiac(user_ctx["birth_date"])
        tv = {}
        if user_ctx.get("birth_time"):
            gender = 1 if user_ctx.get("gender") == "Nam" else -1
            tv = get_tuvi_summary(user_ctx["birth_date"], user_ctx["birth_time"], gender)
        calculated_data["user"] = {"numerology": lp, "zodiac": zd, "tuvi": tv}
        if not intent["is_specific_date"]:
            rag_keywords.extend([f"Số chủ đạo {lp['life_path']}", f"Cung {zd}"])

    # Calculate Partner
    if partner_ctx.get("birth_date"):
        lp_p = calculate_numerology(partner_ctx["birth_date"])
        zd_p = calculate_zodiac(partner_ctx["birth_date"])
        calculated_data["partner"] = {"numerology": lp_p, "zodiac": zd_p}

    # Calculate Specific Date
    if intent["is_specific_date"]:
        lp_s = calculate_numerology(intent["date"])
        zd_s = calculate_zodiac(intent["date"])
        calculated_data["specific_date"] = {"details": intent["date"], "numerology": lp_s, "zodiac": zd_s}
        rag_keywords = [f"Số chủ đạo {lp_s['life_path']}", f"Cung {zd_s}"]

    # Handle Tarot
    final_cards = input_cards + intent["tarot_cards"]
    if final_cards:
        rag_keywords = final_cards
        calculated_data["tarot_context"] = ", ".join(final_cards)

    # RAG
    rag_content = query_pinecone_rag(rag_keywords) if rag_keywords else []

    # Build Prompt
    history = load_history(session_id)
    sys_prompt, user_prompt = build_dynamic_prompt(question or "Giải bài", user_ctx, partner_ctx, calculated_data, rag_content, history)

    # Call AI (Sử dụng hàm tách riêng để pass mock test)
    reply = call_bedrock_nova(sys_prompt, user_prompt)
    
    append_message(session_id, "user", question or "Tarot Reading")
    append_message(session_id, "assistant", reply)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)
    }