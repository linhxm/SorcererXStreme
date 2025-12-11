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
    from lasotuvi import App, DiaBan, ThienBan
    from lasotuvi.AmDuong import diaChi
except ImportError:
    print("WARNING: Không tìm thấy thư viện lasotuvi. Chức năng tử vi sẽ bị hạn chế.")

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
    """Lấy ngày giờ hiện tại theo giờ VN (UTC+7)"""
    return datetime.now(timezone(timedelta(hours=7)))

def calculate_numerology(date_str: str) -> dict:
    """
    Tính số chủ đạo và năm cá nhân.
    Format input: DD/MM/YYYY hoặc YYYY-MM-DD
    """
    try:
        # Chuẩn hóa ngày tháng
        if "-" in date_str:
            parts = date_str.split("-") # YYYY-MM-DD
            d, m, y = int(parts[2]), int(parts[1]), int(parts[0])
        else:
            parts = date_str.split("/") # DD/MM/YYYY
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Hàm đệ quy tính tổng các chữ số
        def sum_digits(n):
            s = sum(int(digit) for digit in str(n))
            if s == 11 or s == 22 or s == 33: return s # Số master
            return s if s < 10 else sum_digits(s)

        # Tính Life Path (Cách tính phổ biến: Tổng ngày + Tổng tháng + Tổng năm)
        # Lưu ý: Có nhiều cách tính, đây là cách Pythagoras phổ biến
        total = sum_digits(d) + sum_digits(m) + sum_digits(y)
        life_path = sum_digits(total)
        if life_path == 4 and total == 22: life_path = 22 # Trường hợp đặc biệt

        return {
            "life_path": str(life_path),
            "details": f"{d}/{m}/{y}"
        }
    except:
        return {"life_path": None, "details": date_str}

def calculate_zodiac(date_str: str) -> str:
    """Xác định cung hoàng đạo"""
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
                if d < day:
                    return sign
                else:
                    # Trả về cung tiếp theo
                    idx = zodiacs.index((month, day, sign))
                    return zodiacs[(idx + 1) % 12][2]
        return "Ma Kết" # Default fallback
    except:
        return "Không xác định"

def get_tuvi_summary(birth_date: str, birth_time: str, gender: int) -> dict:
    """
    Wrapper gọi thư viện lasotuvi để lấy thông tin cơ bản.
    gender: 1 (Nam), -1 (Nữ)
    """
    try:
        # Parse ngày sinh
        if "-" in birth_date:
            parts = birth_date.split("-")
            d, m, y = int(parts[2]), int(parts[1]), int(parts[0])
        else:
            parts = birth_date.split("/")
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Parse giờ sinh (Lấy chi giờ: 1=Tý... 12=Hợi)
        # Đơn giản hóa: Giả sử birth_time là "HH:MM"
        h = int(birth_time.split(":")[0])
        # Chuyển giờ dương lịch sang chi giờ (Tý: 23h-1h, Sửu: 1h-3h...)
        gio_chi = int((h + 1) / 2) % 12
        if gio_chi == 0: gio_chi = 12 # 12 là Hợi, nhưng logic python có thể khác, map theo thư viện
        # Mapping tạm thời cho thư viện lasotuvi (Cần check kỹ App.py expect gì)
        # App.py expect gioSinh là int index trong diaChi (1=Tý)
        # Code lasotuvi: diaChi[1] = Tý.
        
        gio_sinh_idx = gio_chi + 1 # Hack nhẹ để khớp index nếu cần
        if gio_sinh_idx > 12: gio_sinh_idx = 1
        
        # Gọi thư viện
        # Lưu ý: Class diaBan trong DiaBan.py cần khởi tạo
        # Hàm lapDiaBan trong App.py: lapDiaBan(diaBan, nn, tt, nnnn, gioSinh, gioiTinh, duongLich, timeZone)
        
        db = App.lapDiaBan(DiaBan.diaBan, d, m, y, gio_chi, gender, True, 7)
        
        # Trích xuất thông tin Mệnh
        cung_menh_idx = db.cungMenh
        cung_menh = db.thapNhiCung[cung_menh_idx]
        
        chinh_tinh = [s['saoTen'] for s in cung_menh.cungSao if s['saoLoai'] == 1]
        phu_tinh_tot = [s['saoTen'] for s in cung_menh.cungSao if s['saoLoai'] in [3,4,5,6]]
        
        return {
            "menh_tai": diaChi[cung_menh.cungSo]['tenChi'],
            "chinh_tinh": ", ".join(chinh_tinh) if chinh_tinh else "Vô Chính Diệu",
            "phu_tinh": ", ".join(phu_tinh_tot[:3]) # Lấy vài sao tiêu biểu
        }
    except Exception as e:
        print(f"TuVi Error: {e}")
        return {}

# =========================
# IV. HELPER FUNCTIONS (RAG & MEMORY)
# =========================

def load_history(session_id: str, limit: int = 6) -> List[Dict]:
    try:
        resp = ddb_table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            ScanIndexForward=False, 
            Limit=limit,
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
    """Search Pinecone based on extracted keywords, NOT the full question"""
    if not pc_index or not keywords: return []
    
    docs = []
    # Gộp keywords thành 1 query string hoặc query từng cái
    search_text = " ".join(keywords)
    
    vector = embed_query(search_text)
    if not vector: return []

    try:
        results = pc_index.query(vector=vector, top_k=top_k, include_metadata=True)
        for match in results.get('matches', []):
            if match['score'] < 0.35: continue
            md = match.get('metadata', {})
            content = md.get('context_str') or md.get('content') or ""
            entity = md.get('entity_name') or ""
            docs.append(f"[{entity}]: {content}")
    except Exception as e:
        print(f"RAG Error: {e}")
        
    return docs

# =========================
# V. MAIN LOGIC CONTROLLER
# =========================

def analyze_input(question: str) -> dict:
    """Phân tích câu hỏi để tìm Entity ngày tháng hoặc Tarot"""
    result = {"is_specific_date": False, "date": None, "tarot_cards": []}
    
    # Regex tìm ngày (DD/MM/YYYY)
    date_match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', question)
    if date_match:
        result["is_specific_date"] = True
        result["date"] = date_match.group(0)
    
    # Đơn giản hóa check tarot (Nếu frontend gửi kèm list bài thì tốt hơn)
    # Ở đây giả sử check keyword tên các lá bài phổ biến nếu user gõ
    tarot_keywords = ["Cup", "Wand", "Sword", "Pentacle", "Fool", "Magician", "Empress", "Emperor", "Lover", "Chariot", "Strength", "Hermit", "Wheel", "Justice", "Hanged", "Death", "Temperance", "Devil", "Tower", "Star", "Moon", "Sun", "Judgement", "World"]
    found_cards = [word for word in tarot_keywords if word.lower() in question.lower()]
    if found_cards:
        result["tarot_cards"] = found_cards # Đây chỉ là basic detection
        
    return result

def build_dynamic_prompt(
    question: str, 
    user_ctx: Dict, 
    partner_ctx: Dict,
    calculated_data: Dict,
    rag_content: List[str],
    history: List[Dict]
) -> Tuple[str, str]:
    
    current_date = get_current_date_vn().strftime("%d/%m/%Y")
    
    # 1. Xử lý thông tin hiển thị (Privacy Mode)
    # Không đưa raw date vào user prompt để tránh AI lặp lại
    user_name = user_ctx.get("name", "Bạn")
    
    # Partner info logic
    has_partner = bool(partner_ctx)
    partner_info_str = ""
    if has_partner:
        # Chúng ta chỉ đưa thông tin ĐÃ TÍNH TOÁN vào, không đưa ngày sinh
        p_calc = calculated_data.get("partner", {})
        partner_info_str = f"""
        - Đối phương (Gọi là 'Người ấy'/'Đối phương'):
          + Số chủ đạo: {p_calc.get('numerology', {}).get('life_path', 'N/A')}
          + Cung Hoàng Đạo: {p_calc.get('zodiac', 'N/A')}
        """

    # User info logic
    u_calc = calculated_data.get("user", {})
    user_info_str = f"""
    - User (Gọi là {user_name}):
      + Số chủ đạo: {u_calc.get('numerology', {}).get('life_path', 'N/A')}
      + Cung Hoàng Đạo: {u_calc.get('zodiac', 'N/A')}
      + Tử Vi (Mệnh): {u_calc.get('tuvi', {}).get('menh_tai', 'N/A')} có sao {u_calc.get('tuvi', {}).get('chinh_tinh', '')}
    """

    # Specific Query Info (Nếu user hỏi về ngày khác)
    specific_info_str = ""
    if calculated_data.get("specific_date"):
        sd = calculated_data["specific_date"]
        specific_info_str = f"""
        [THÔNG TIN NGÀY ĐƯỢC HỎI: {sd['details']}]
        - Thần số học ngày này: {sd['numerology']['life_path']}
        - Cung hoàng đạo ngày này: {sd['zodiac']}
        - Lưu ý: User đang hỏi về ngày này, hãy ưu tiên phân tích năng lượng của nó.
        """

    # Tarot Info
    tarot_str = ""
    if calculated_data.get("tarot_context"):
        tarot_str = f"[LÁ BÀI RÚT ĐƯỢC/ĐƯỢC HỎI]\n{calculated_data['tarot_context']}\n(Hãy giải bài dựa trên nội dung này)"

    # RAG Context
    rag_text = "\n".join(rag_content) if rag_content else "Không có thông tin tra cứu thêm."

    # History
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])

    # 2. System Prompt (Cực kỳ quan trọng)
    system_prompt = f"""
# ROLE
Bạn là AI Huyền Học (SorcererXstreme), chuyên gia Tử Vi, Thần Số, Chiêm Tinh và Tarot.
Ngày hiện tại: {current_date}.

# STRICT RULES (BẮT BUỘC)
1. **PRIVACY FIRST:** - KHÔNG BAO GIỜ nhắc lại ngày/tháng/năm sinh, giờ sinh, nơi sinh của User hoặc Partner trong câu trả lời.
   - Nếu cần nhắc đến thông tin tính toán, chỉ nói kết quả (VD: "Bạn có số chủ đạo 10", "Bạn thuộc cung Thiên Bình").
   - Gọi Partner là "Người ấy", "Đối phương", hoặc "Bạn đời", KHÔNG dùng tên thật nếu có.

2. **CONTEXT AWARENESS:**
   - Nếu User hỏi về một ngày cụ thể (VD: 10/10/2025), hãy dùng dữ liệu [THÔNG TIN NGÀY ĐƯỢC HỎI] để trả lời.
   - Nếu User hỏi "Hôm nay tôi thế nào", hãy dùng dữ liệu ngày hiện tại ({current_date}) kết hợp với thông số của User.
   - Nếu câu hỏi có bài Tarot, hãy tập trung giải nghĩa lá bài.
   - Nếu câu hỏi chung chung (VD: "Tính cách của tôi"), hãy tổng hợp từ Thần số học + Chiêm tinh + Tử vi (nếu có dữ liệu).

3. **NO HALLUCINATION ON TAROT:**
   - Chỉ giải bài Tarot khi người dùng cung cấp tên lá bài hoặc hệ thống cung cấp dữ liệu bài. KHÔNG tự bịa ra lá bài.

4. **TONE:**
   - Huyền bí nhưng khoa học, khách quan, sâu sắc, đồng cảm.
   - Cấu trúc câu trả lời rõ ràng (Bullet points).

# OUTPUT FORMAT
"Chào [Tên User], [Câu dẫn nhập dựa trên câu hỏi]...

1. **Góc nhìn [Lĩnh vực chính liên quan]:**
   - [Nội dung phân tích]

2. **Kết hợp [Lĩnh vực bổ trợ]:**
   - [Nội dung bổ trợ]

3. **Lời khuyên/Thông điệp:**
   - [Hành động cụ thể]"
"""

    user_prompt = f"""
[USER DATA CALCULATED]
{user_info_str}

[PARTNER DATA CALCULATED]
{partner_info_str}

{specific_info_str}

{tarot_str}

[KNOWLEDGE BASE / RAG]
{rag_text}

[CHAT HISTORY]
{history_text}

[QUESTION]
"{question}"
"""
    return system_prompt, user_prompt

# =========================
# VI. LAMBDA HANDLER
# =========================

def lambda_handler(event, context):
    # 1. Parse Input
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
        user_ctx = body.get("user_context", {})
        partner_ctx = body.get("partner_context", {})
        data_block = body.get("data", {})
        session_id = data_block.get("sessionId", "guest")
        question = data_block.get("question", "")
        
        # New: Nhận bài Tarot từ Frontend (nếu có tính năng bốc bài trên UI)
        input_cards = data_block.get("tarot_cards", []) 
    except:
        return {"statusCode": 400, "body": "Invalid Request"}

    if not question and not input_cards:
        return {"statusCode": 200, "body": json.dumps({"reply": "Xin chào, tôi có thể giúp gì về vận mệnh của bạn?"})}

    # 2. Analyze Intent & Entities
    intent = analyze_input(question)
    
    # 3. Perform Calculations (The "Smarter" Part)
    calculated_data = {"user": {}, "partner": {}}
    rag_keywords = []

    # 3a. Calculate for User
    if user_ctx.get("birth_date"):
        lp = calculate_numerology(user_ctx["birth_date"])
        zd = calculate_zodiac(user_ctx["birth_date"])
        tv = {}
        # Chỉ tính tử vi nếu có giờ sinh
        if user_ctx.get("birth_time"):
            gender = 1 if user_ctx.get("gender") == "Nam" else -1
            tv = get_tuvi_summary(user_ctx["birth_date"], user_ctx["birth_time"], gender)
        
        calculated_data["user"] = {"numerology": lp, "zodiac": zd, "tuvi": tv}
        
        # Nếu câu hỏi về bản thân ("Tôi", "Mình"), thêm keywords RAG
        if not intent["is_specific_date"]:
            rag_keywords.append(f"Số chủ đạo {lp['life_path']}")
            rag_keywords.append(f"Cung {zd}")
            if tv: rag_keywords.append(f"Sao {tv.get('chinh_tinh')}")

    # 3b. Calculate for Partner (if needed)
    if partner_ctx.get("birth_date"):
        lp_p = calculate_numerology(partner_ctx["birth_date"])
        zd_p = calculate_zodiac(partner_ctx["birth_date"])
        calculated_data["partner"] = {"numerology": lp_p, "zodiac": zd_p}
        # Nếu câu hỏi nhắc đến người ấy, thêm keywords
        if "người ấy" in question.lower() or "anh ấy" in question.lower() or "cô ấy" in question.lower():
            rag_keywords.append(f"Số chủ đạo {lp_p['life_path']}")
            rag_keywords.append(f"Cung {zd_p}")

    # 3c. Calculate for Specific Date (from Question)
    if intent["is_specific_date"]:
        lp_s = calculate_numerology(intent["date"])
        zd_s = calculate_zodiac(intent["date"])
        calculated_data["specific_date"] = {
            "details": intent["date"],
            "numerology": lp_s,
            "zodiac": zd_s
        }
        rag_keywords = [f"Số chủ đạo {lp_s['life_path']}", f"Cung {zd_s}"] # Override keywords để tập trung vào ngày đó

    # 3d. Handle Tarot
    # Kết hợp bài từ Frontend gửi lên HOẶC bài detect trong câu hỏi
    final_cards = input_cards + intent["tarot_cards"]
    if final_cards:
        # Nếu có bài, RAG ưu tiên tìm ý nghĩa bài
        rag_keywords = final_cards 
        calculated_data["tarot_context"] = ", ".join(final_cards)

    # 4. RAG Execution (Contextual Search)
    # Chỉ search nếu có keywords đã tính toán hoặc từ bài Tarot. 
    # Nếu hỏi "Hôm nay ngày mấy", rag_keywords rỗng -> skip RAG -> tiết kiệm & chính xác.
    rag_content = []
    if rag_keywords:
        rag_content = query_pinecone_rag(rag_keywords)

    # 5. Build Final Prompt & Call LLM
    history = load_history(session_id)
    system_prompt, user_prompt = build_dynamic_prompt(
        question, user_ctx, partner_ctx, calculated_data, rag_content, history
    )

    try:
        # Gọi Nova-Micro (nhanh, rẻ) hoặc Nova-Lite/Pro tùy config
        body = json.dumps({
            "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.6},
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}]
        })
        
        resp = bedrock.invoke_model(
            modelId=BEDROCK_LLM_MODEL_ID,
            body=body, contentType="application/json", accept="application/json"
        )
        reply = json.loads(resp["body"].read())["output"]["message"]["content"][0]["text"]
        
        # Save History
        append_message(session_id, "user", question)
        append_message(session_id, "assistant", reply)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"sessionId": session_id, "reply": reply}, ensure_ascii=False)
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": "Vũ trụ đang tắc nghẽn, vui lòng thử lại."})}