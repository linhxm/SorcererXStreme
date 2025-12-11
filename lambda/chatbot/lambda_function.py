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
    HAS_TUVI = True
except ImportError:
    HAS_TUVI = False
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
# III. CALCULATION ENGINES (CORE LOGIC)
# =========================

def get_current_date_vn():
    return datetime.now(timezone(timedelta(hours=7)))

def normalize_date(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Chuyển đổi các định dạng ngày về d, m, y"""
    try:
        # Hỗ trợ DD/MM/YYYY hoặc YYYY-MM-DD
        if "-" in date_str:
            parts = date_str.split("-")
            return int(parts[2]), int(parts[1]), int(parts[0])
        elif "/" in date_str:
            parts = date_str.split("/")
            return int(parts[0]), int(parts[1]), int(parts[2])
    except:
        return None
    return None

def calculate_numerology(d: int, m: int, y: int) -> str:
    """Tính số chủ đạo (Life Path Number)"""
    def sum_digits(n):
        s = sum(int(digit) for digit in str(n))
        if s == 11 or s == 22 or s == 33: return s
        return s if s < 10 else sum_digits(s)
    
    total = sum_digits(d) + sum_digits(m) + sum_digits(y)
    lp = sum_digits(total)
    if lp == 4 and total == 22: lp = 22
    return str(lp)

def calculate_zodiac(d: int, m: int) -> str:
    """Tính cung hoàng đạo"""
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

def calculate_tuvi(d: int, m: int, y: int, h_str: str, gender: int) -> dict:
    """Tính tử vi: Mệnh, Chính tinh. Gender: 1 (Nam), -1 (Nữ)"""
    if not HAS_TUVI or not h_str: return {}
    try:
        # Parse giờ (HH:MM) ra chi giờ (1=Tý... 12=Hợi)
        hour_val = int(h_str.split(":")[0])
        gio_chi = int((hour_val + 1) / 2) % 12
        if gio_chi == 0: gio_chi = 12
        
        # Gọi thư viện lasotuvi
        db = App.lapDiaBan(DiaBan.diaBan, d, m, y, gio_chi, gender, True, 7)
        cung_menh = db.thapNhiCung[db.cungMenh]
        chinh_tinh = [s['saoTen'] for s in cung_menh.cungSao if s['saoLoai'] == 1]
        
        return {
            "menh_tai": diaChi[cung_menh.cungSo]['tenChi'],
            "chinh_tinh": ", ".join(chinh_tinh) if chinh_tinh else "Vô Chính Diệu"
        }
    except Exception as e:
        print(f"TuVi Logic Error: {e}")
        return {}

# =========================
# IV. INTELLIGENT HANDLERS
# =========================

def analyze_intent_and_extract(question: str, input_tarot: List[str]) -> dict:
    """
    Phân tích câu hỏi để tìm: Ngày tháng cụ thể, Tarot, hoặc câu hỏi chung.
    """
    intent = {
        "explicit_date": None, # Ngày được nhắc đến trong câu hỏi
        "has_tarot": False,
        "tarot_cards": list(input_cards) if input_cards else [], # input_cards từ global scope (fix later) -> Pass as arg
        "needs_calculation": True
    }
    
    # 1. Tìm ngày tháng trong câu hỏi (Explicit Date)
    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', question)
    if date_match:
        d, m, y = map(int, [date_match.group(1), date_match.group(2), date_match.group(3)])
        intent["explicit_date"] = (d, m, y)

    # 2. Tìm bài Tarot trong câu hỏi (nếu chưa có trong input payload)
    if not intent["tarot_cards"]:
        tarot_keywords = ["Fool", "Magician", "Empress", "Emperor", "Lover", "Chariot", "Strength", "Hermit", "Wheel", "Justice", "Hanged", "Death", "Temperance", "Devil", "Tower", "Star", "Moon", "Sun", "Judgement", "World", "Cup", "Wand", "Sword", "Pentacle"]
        found = [w for w in tarot_keywords if w.lower() in question.lower()]
        if found:
            intent["tarot_cards"] = found
    
    if intent["tarot_cards"]:
        intent["has_tarot"] = True

    # 3. Detect Chit-chat (Không cần tính toán)
    # Nếu câu hỏi quá ngắn và không có keyword huyền học -> Chit chat
    meta_keywords = ["tử vi", "chiêm tinh", "thần số", "bói", "tình cảm", "sự nghiệp", "ngày mai", "hôm nay", "tháng này", "năm nay", "hợp", "kỵ", "số", "cung"]
    is_meta = any(k in question.lower() for k in meta_keywords)
    if not is_meta and not intent["explicit_date"] and not intent["has_tarot"] and len(question.split()) < 4:
        intent["needs_calculation"] = False

    return intent

def process_subject_data(intent: dict, user_ctx: dict, partner_ctx: dict) -> dict:
    """
    Xử lý logic ưu tiên: Explicit Query > Payload Data
    """
    result = {
        "rag_keywords": [],
        "prompt_context": "",
        "user_calculated": {},
        "partner_calculated": {}
    }

    # 1. Xử lý Explicit Date (User hỏi về ngày cụ thể: "10/10/2000 là số mấy?")
    # Yêu cầu số 3 & 4: Nếu hỏi ngày cụ thể, hiển thị thông tin ngày đó.
    if intent["explicit_date"]:
        d, m, y = intent["explicit_date"]
        lp = calculate_numerology(d, m, y)
        zd = calculate_zodiac(d, m)
        result["prompt_context"] += f"- [THÔNG TIN ĐƯỢC HỎI - NGÀY {d}/{m}/{y}]: Số chủ đạo {lp}, Cung {zd}.\n"
        result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])
        # Khi hỏi ngày cụ thể, ta ít quan tâm payload user trừ khi câu hỏi liên kết (vd: "Ngày X có hợp tôi không")
        # Nhưng để an toàn, vẫn tính toán ngầm user payload bên dưới nhưng không cho vào RAG chính.

    # 2. Xử lý Tarot (Yêu cầu số 6 & 7)
    if intent["has_tarot"]:
        cards_str = ", ".join(intent["tarot_cards"])
        result["prompt_context"] += f"- [BÀI TAROT RÚT ĐƯỢC]: {cards_str} (Hãy giải nghĩa dựa trên các lá này).\n"
        result["rag_keywords"].extend(intent["tarot_cards"])

    # 3. Xử lý User Payload (Yêu cầu số 3: Privacy - Chỉ hiển thị metadata)
    # Chỉ tính toán nếu cần thiết (không phải chit-chat)
    if intent["needs_calculation"] and user_ctx.get("birth_date"):
        dmy = normalize_date(user_ctx["birth_date"])
        if dmy:
            d, m, y = dmy
            lp = calculate_numerology(d, m, y)
            zd = calculate_zodiac(d, m)
            
            # Tử vi (Chỉ tính nếu có giờ)
            tv = {}
            if user_ctx.get("birth_time"):
                gender = 1 if user_ctx.get("gender") == "Nam" else -1
                tv = calculate_tuvi(d, m, y, user_ctx["birth_time"], gender)
            
            result["user_calculated"] = {"lp": lp, "zd": zd, "tv": tv}
            
            # Thêm vào context prompt (Ẩn ngày sinh, chỉ hiện kết quả)
            tv_str = f", Mệnh {tv.get('menh_tai')}" if tv else ""
            result["prompt_context"] += f"- [USER METADATA - {user_ctx.get('name', 'Bạn')}]: Số chủ đạo {lp}, Cung {zd}{tv_str}.\n"
            
            # Chỉ thêm vào RAG keywords nếu câu hỏi KHÔNG phải là hỏi ngày cụ thể
            # (Tránh nhiễu: Hỏi ngày 10/10 thì đừng search số chủ đạo của user)
            if not intent["explicit_date"] and not intent["has_tarot"]:
                result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])
                if tv: result["rag_keywords"].append(f"Sao {tv.get('chinh_tinh', '')}")

    # 4. Xử lý Partner Payload (Yêu cầu số 5: Privacy - Xưng hô 'Người ấy')
    if intent["needs_calculation"] and partner_ctx.get("birth_date"):
        dmy = normalize_date(partner_ctx["birth_date"])
        if dmy:
            d, m, y = dmy
            lp = calculate_numerology(d, m, y)
            zd = calculate_zodiac(d, m)
            result["partner_calculated"] = {"lp": lp, "zd": zd}
            
            result["prompt_context"] += f"- [PARTNER METADATA - Người ấy]: Số chủ đạo {lp}, Cung {zd}.\n"
            
            # Thêm RAG nếu câu hỏi có nhắc đến partner
            if not intent["explicit_date"] and not intent["has_tarot"]:
                result["rag_keywords"].extend([f"Số chủ đạo {lp}", f"Cung {zd}"])

    return result

# =========================
# V. RAG & LLM
# =========================

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

def query_pinecone_rag(keywords: List[str]) -> List[str]:
    # Yêu cầu số 1: Chỉ search khi có keywords
    if not pc_index or not keywords: return []
    
    # Lọc trùng và search
    unique_kw = list(set(keywords))
    search_text = " ".join(unique_kw)
    vector = embed_query(search_text)
    if not vector: return []
    
    try:
        results = pc_index.query(vector=vector, top_k=3, include_metadata=True)
        docs = []
        for match in results.get('matches', []):
            if match['score'] < 0.35: continue
            md = match.get('metadata', {})
            content = md.get('context_str') or md.get('content') or ""
            entity = md.get('entity_name') or ""
            docs.append(f"[{entity}]: {content}")
        return docs
    except: return []

def load_history(session_id: str) -> str:
    try:
        items = ddb_table.query(KeyConditionExpression=Key("sessionId").eq(session_id), ScanIndexForward=False, Limit=5).get("Items", [])
        return "\n".join([f"{h['role'].upper()}: {h['content']}" for h in items[::-1]])
    except: return ""

def call_bedrock_nova(system: str, user: str) -> str:
    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.6},
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}]
    })
    try:
        resp = bedrock.invoke_model(modelId=BEDROCK_LLM_MODEL_ID, body=body, contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return "Vũ trụ đang tắc nghẽn, vui lòng thử lại."

# =========================
# VI. MAIN HANDLER
# =========================

def lambda_handler(event, context):
    # 1. Validate Input (Strict)
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
    except:
        return {"statusCode": 400, "body": "Invalid JSON Body"}

    data = body.get("data", {})
    user_ctx = body.get("user_context", {})
    partner_ctx = body.get("partner_context", {})
    session_id = data.get("sessionId")
    question = data.get("question")
    input_cards = data.get("tarot_cards", [])

    if not session_id or (not question and not input_cards):
         return {"statusCode": 400, "body": json.dumps({"error": "Missing sessionId or question"})}

    # 2. Analyze & Calculate
    # Truyền input_cards vào hàm analyze
    intent = analyze_intent_and_extract(question or "", input_cards) 
    
    # Nếu là chit-chat (không cần tính toán, không RAG)
    if not intent["needs_calculation"]:
        reply = "Chào bạn, tôi là trợ lý huyền học. Bạn muốn hỏi về Tử vi, Thần số hay Tarot hôm nay?"
        # Save & Return nhanh
        try: ddb_table.put_item(Item={"sessionId": session_id, "timestamp": datetime.utcnow().isoformat()+"Z", "role": "assistant", "content": reply})
        except: pass
        return {"statusCode": 200, "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)}

    # Tính toán số liệu & Chuẩn bị RAG keywords
    # Hàm này đã handle logic Explicit > Payload và Privacy
    processed_data = process_subject_data(intent, user_ctx, partner_ctx)
    
    # 3. RAG Search (Chỉ search những keyword cần thiết)
    rag_docs = query_pinecone_rag(processed_data["rag_keywords"])
    
    # 4. Build Prompt (Strict Requirements)
    current_date = get_current_date_vn().strftime("%d/%m/%Y")
    rag_text = "\n".join(rag_docs) if rag_docs else "Không có dữ liệu tra cứu."
    history_text = load_history(session_id)
    
    system_prompt = f"""
# ROLE: AI Huyền Học (SorcererXstreme). Hôm nay: {current_date}.

# STRICT RULES (BẮT BUỘC):
1. **PRIVACY:**
   - KHÔNG nhắc lại ngày tháng năm sinh, giờ sinh của User/Partner trong câu trả lời (trừ khi câu hỏi của User chứa ngày đó).
   - Chỉ sử dụng các thông số đã tính toán (Số chủ đạo, Cung, Sao) để luận giải.
   - Gọi Partner là "Người ấy", "Đối phương", tuyệt đối KHÔNG dùng tên thật nếu có trong dữ liệu.

2. **LOGIC TRẢ LỜI:**
   - Nếu có [BÀI TAROT]: Chỉ tập trung giải bài, kết nối với câu chuyện của user. KHÔNG tự bịa lá bài nếu không có dữ liệu.
   - Nếu hỏi về Tương Hợp/Mối quan hệ: **PHẢI TỔNG HỢP** (Combine) thành 1-2 câu nhận định chung, lồng ghép đặc điểm cả hai. KHÔNG liệt kê kiểu "Bạn là A, người ấy là B".
   - Nếu câu hỏi chung chung (Vd: "Tôi thế nào?"): Kết hợp đa chiều (Thần số + Chiêm tinh + Tử vi).

3. **TONE:** Ngắn gọn, súc tích, huyền bí nhưng thực tế.
"""

    user_prompt = f"""
[CALCULATED CONTEXT]
{processed_data['prompt_context']}

[KNOWLEDGE BASE]
{rag_text}

[HISTORY]
{history_text}

[USER QUESTION]
"{question}"
"""

    # 5. Call AI
    reply = call_bedrock_nova(system_prompt, user_prompt)

    # 6. Save History
    try:
        ddb_table.put_item(Item={"sessionId": session_id, "timestamp": datetime.utcnow().isoformat()+"Z", "role": "user", "content": question})
        ddb_table.put_item(Item={"sessionId": session_id, "timestamp": datetime.utcnow().isoformat()+"Z", "role": "assistant", "content": reply})
    except: pass

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)
    }