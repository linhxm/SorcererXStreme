import json
import boto3
import os
import sys
import traceback
import re
from datetime import datetime, timedelta

# --- CẤU HÌNH ĐƯỜNG DẪN THƯ VIỆN ---
# Thêm thư mục hiện tại vào path để import được folder 'lasotuvi' nằm cùng cấp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORT THƯ VIỆN TỬ VI ---
HAS_TUVI_LIB = False
try:
    from lasotuvi import App, DiaBan, ThienBan
    from lasotuvi.AmDuong import diaChi
    HAS_TUVI_LIB = True
except ImportError as e:
    print(f"WARNING: Không load được thư viện Tử Vi. Lỗi: {e}")

# ==========================================
# 0. CONFIGURATION & CLIENTS
# ==========================================
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-1")
LLM_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-pro-v1:0") 
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "SorcererXStreme_KnowledgeBase")

try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    dynamodb = boto3.resource('dynamodb', region_name=BEDROCK_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
except Exception as e:
    print(f"INIT ERROR: {e}")
    bedrock_client = None
    table = None

# ==========================================
# 1. HELPER FUNCTIONS (UTILITIES)
# ==========================================

def get_current_date():
    """Lấy ngày hiện tại (UTC+7 cho Việt Nam)"""
    return datetime.utcnow() + timedelta(hours=7)

def parse_date_input(date_input):
    """
    Xử lý input linh hoạt:
    1. Nếu input là text chứa ngày (VD: 'xem ngày 10/10/2025'), trích xuất ngày đó.
    2. Nếu input là chuỗi ngày (YYYY-MM-DD), parse ra date object.
    """
    if not date_input: return None
    
    # Regex tìm ngày trong chuỗi văn bản (DD/MM/YYYY hoặc DD-MM-YYYY)
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', str(date_input))
    if match:
        try:
            d, m, y = map(int, match.groups())
            return datetime(y, m, d)
        except ValueError:
            pass
            
    # Fallback cho định dạng chuẩn ISO
    try:
        return datetime.strptime(str(date_input), "%Y-%m-%d")
    except:
        return None

def get_knowledge_context(category, entity_name):
    """
    Lấy dữ liệu ý nghĩa từ Database (RAG).
    Thay thế cho việc hardcode dataset_v4.jsonl.
    """
    if not table: return ""
    try:
        # Chuẩn hóa key
        entity_key = str(entity_name).strip()
        response = table.get_item(Key={'category': category, 'entity_name': entity_key})
        item = response.get('Item')
        
        if not item:
            # Fallback tìm kiếm gần đúng hoặc mapping (nếu cần)
            return ""
            
        contexts = item.get('contexts', {})
        # Nếu lưu dạng JSON string
        if isinstance(contexts, str):
            try: contexts = json.loads(contexts)
            except: pass
            
        # Flatten context thành text để đưa vào Prompt
        context_text = ""
        if isinstance(contexts, dict):
            for k, v in contexts.items():
                context_text += f"- {k}: {v}\n"
        else:
            context_text = str(contexts)
            
        return context_text
    except Exception as e:
        print(f"DB Error: {e}")
        return ""

def call_bedrock_llm(system_instruction, user_input, temperature=0.6):
    """Gửi request tới Bedrock"""
    if not bedrock_client: return "Lỗi kết nối AI."

    body = json.dumps({
        "inferenceConfig": {"max_new_tokens": 2000, "temperature": temperature},
        "system": [{"text": system_instruction}],
        "messages": [{"role": "user", "content": [{"text": user_input}]}]
    })

    try:
        response = bedrock_client.invoke_model(modelId=LLM_MODEL_ID, body=body)
        response_body = json.loads(response.get('body').read())
        return response_body['output']['message']['content'][0]['text']
    except Exception as e:
        print(f"Bedrock Error: {e}")
        return "Vũ trụ đang tắc nghẽn, vui lòng thử lại sau."

# ==========================================
# 2. CALCULATION ENGINES (TÍNH TOÁN LOGIC)
# ==========================================

# --- Thần Số Học ---
def calc_numerology_number(dt):
    """Tính số chủ đạo"""
    if not dt: return None
    s = str(dt.day) + str(dt.month) + str(dt.year)
    total = sum(int(c) for c in s)
    
    # Rút gọn (giữ lại 11, 22, 33 nếu muốn, ở đây giữ logic cơ bản 2-11 và 22/4)
    while total > 11 and total != 22:
        total = sum(int(c) for c in str(total))
    
    if total == 22: return "22"
    if total == 11: return "11"
    if total == 10: return "1" # Một số trường phái coi 10 là 1
    return str(total)

# --- Chiêm Tinh ---
def calc_zodiac_sign(day, month):
    zodiacs = [
        (1, 20, "Ma Kết"), (2, 19, "Bảo Bình"), (3, 21, "Song Ngư"),
        (4, 20, "Bạch Dương"), (5, 21, "Kim Ngưu"), (6, 22, "Song Tử"),
        (7, 23, "Cự Giải"), (8, 23, "Sư Tử"), (9, 23, "Xử Nữ"),
        (10, 24, "Thiên Bình"), (11, 23, "Thiên Yết"), (12, 22, "Nhân Mã")
    ]
    for m, d, sign in zodiacs:
        if month == m:
            if day < d: return sign
            else:
                # Lấy cung kế tiếp
                idx = zodiacs.index((m, d, sign)) + 1
                return zodiacs[idx % 12][2]
    return "Ma Kết"

# --- Tử Vi (Dùng thư viện lasotuvi) ---
def calc_tuvi_summary(dob, time_str, gender_str, name):
    if not HAS_TUVI_LIB or not time_str:
        return None
    
    try:
        # Parse giờ sinh (HH:MM) thành Chi (1-12)
        h = int(time_str.split(':')[0])
        chi_gio = int((h + 1) / 2) % 12
        if chi_gio == 0: chi_gio = 12
        
        # Gender: 1 Nam, -1 Nữ
        gender = 1 if gender_str in ['male', 'nam'] else -1
        
        # Gọi thư viện tính toán
        db = App.lapDiaBan(DiaBan.diaBan, dob.day, dob.month, dob.year, chi_gio, gender, True, 7)
        tb = ThienBan.lapThienBan(dob.day, dob.month, dob.year, chi_gio, gender, name, db, True, 7)
        
        # Trích xuất dữ liệu quan trọng để gửi cho AI
        cung_menh_idx = db.cungMenh
        cung_menh_data = db.thapNhiCung[cung_menh_idx]
        chinh_tinh = [s['saoTen'] for s in cung_menh_data.cungSao if s['saoLoai'] == 1]
        
        return {
            "menh": tb.banMenh,
            "cuc": tb.tenCuc,
            "menh_tai_cung": diaChi[cung_menh_data.cungSo]['tenChi'],
            "chinh_tinh": ", ".join(chinh_tinh) if chinh_tinh else "Vô Chính Diệu"
        }
    except Exception as e:
        print(f"TuVi Calc Error: {e}")
        return None

# ==========================================
# 3. LOGIC CONTROLLER
# ==========================================

def process_request(body):
    """Xử lý logic trung tâm: Payload -> Calculate -> Context -> Prompt"""
    
    # 1. Trích xuất Payload
    data = body.get('data', {})
    user_ctx = body.get('user_context', {})
    partner_ctx = body.get('partner_context', {})
    
    session_id = data.get('sessionId')
    question = data.get('question', '')
    tarot_cards = data.get('tarot_cards', []) # Danh sách bài Tarot nếu có
    
    # 2. Phân tích Intent & Time
    # Ưu tiên ngày trong câu hỏi (Explicit) hơn ngày sinh trong profile (Implicit)
    explicit_date = parse_date_input(question)
    current_date = get_current_date()
    
    # Xác định chủ đề (nếu người dùng không chọn feature_type cụ thể)
    domain_data = {} # Chứa data đã tính toán
    knowledge_context = [] # Chứa text ý nghĩa từ DB
    
    # --- LOGIC NGƯỜI DÙNG (USER) ---
    user_dob = parse_date_input(user_ctx.get('birth_date'))
    if user_dob:
        # Thần số
        lp = calc_numerology_number(user_dob)
        domain_data['user_numerology'] = lp
        knowledge_context.append(f"Kiến thức số chủ đạo {lp}: {get_knowledge_context('numerology_number', f'Số {lp}')}")
        
        # Chiêm tinh
        zodiac = calc_zodiac_sign(user_dob.day, user_dob.month)
        domain_data['user_zodiac'] = zodiac
        knowledge_context.append(f"Kiến thức cung {zodiac}: {get_knowledge_context('cung-hoang-dao', zodiac)}")
        
        # Tử vi (Chỉ khi có giờ sinh)
        tv = calc_tuvi_summary(user_dob, user_ctx.get('birth_time'), user_ctx.get('gender'), user_ctx.get('name'))
        if tv:
            domain_data['user_tuvi'] = tv
            # Tử vi phức tạp, context lấy từ kết quả tính toán trực tiếp là chính
    
    # --- LOGIC ĐỐI PHƯƠNG (PARTNER - Nếu có) ---
    partner_dob = parse_date_input(partner_ctx.get('birth_date'))
    if partner_dob:
        p_lp = calc_numerology_number(partner_dob)
        p_zodiac = calc_zodiac_sign(partner_dob.day, partner_dob.month)
        domain_data['partner_numerology'] = p_lp
        domain_data['partner_zodiac'] = p_zodiac
        # Lấy thêm context nếu cần so sánh
        knowledge_context.append(f"Kiến thức cung đối phương {p_zodiac}: {get_knowledge_context('cung-hoang-dao', p_zodiac)}")

    # --- LOGIC NGÀY CỤ THỂ (EXPLICIT DATE) ---
    # Nếu câu hỏi hỏi về ngày cụ thể (VD: "Ngày 10/10/2025 thế nào?")
    if explicit_date:
        ex_lp = calc_numerology_number(explicit_date)
        domain_data['explicit_date'] = {
            "date": explicit_date.strftime("%d/%m/%Y"),
            "numerology": ex_lp
        }
        # Clear bớt context cá nhân nếu câu hỏi chỉ tập trung vào ngày này? 
        # Tùy logic, nhưng ở đây giữ lại để so sánh độ hợp ngày.
        knowledge_context.append(f"Kiến thức thần số ngày {ex_lp}: {get_knowledge_context('numerology_number', f'Số {ex_lp}')}")

    # --- LOGIC TAROT ---
    if tarot_cards:
        # Chỉ giải khi có bài
        card_meanings = []
        for card in tarot_cards:
            # Giả sử card là string tên bài. Nếu là dict thì cần parse.
            meaning = get_knowledge_context('tarot_card', card)
            card_meanings.append(f"Lá bài: {card}\nÝ nghĩa: {meaning}")
        domain_data['tarot_reading'] = card_meanings

    # ==========================================
    # 4. PROMPT ENGINEERING (PRIVACY & LOGIC)
    # ==========================================
    
    # Xây dựng Context String an toàn (Không lộ PII)
    safe_context_str = f"Thời điểm hiện tại: {current_date.strftime('%d/%m/%Y')}\n"
    
    # Thông tin User (Đã ẩn ngày sinh)
    if 'user_numerology' in domain_data:
        safe_context_str += f"- USER (Người hỏi): Số chủ đạo {domain_data['user_numerology']}, Cung {domain_data['user_zodiac']}"
        if 'user_tuvi' in domain_data:
            tv = domain_data['user_tuvi']
            safe_context_str += f", Mệnh {tv['menh']}, Cục {tv['cuc']}, Chính tinh {tv['chinh_tinh']} tại {tv['menh_tai_cung']}"
        safe_context_str += ".\n"
        
    # Thông tin Partner (Đã ẩn ngày sinh & Tên)
    if 'partner_numerology' in domain_data:
        safe_context_str += f"- PARTNER (Người ấy/Đối phương): Số chủ đạo {domain_data['partner_numerology']}, Cung {domain_data['partner_zodiac']}.\n"
        
    # Thông tin ngày được hỏi
    if 'explicit_date' in domain_data:
        ed = domain_data['explicit_date']
        safe_context_str += f"- NGÀY ĐƯỢC HỎI ({ed['date']}): Mang năng lượng số {ed['numerology']}.\n"
        
    # Thông tin Tarot
    if 'tarot_reading' in domain_data:
        safe_context_str += "\n--- TRẢI BÀI TAROT ---\n" + "\n".join(domain_data['tarot_reading']) + "\n"

    # Kiến thức bổ trợ (RAG)
    rag_str = "\n".join(knowledge_context)

    # SYSTEM PROMPT
    system_instruction = """
    Bạn là AI Huyền Học (SorcererXstreme), chuyên gia về Tử Vi, Thần Số, Chiêm Tinh và Tarot.
    
    QUY TẮC BẢO MẬT & XỬ LÝ (TUÂN THỦ TUYỆT ĐỐI):
    1. **Bảo mật PII:** KHÔNG BAO GIỜ hiển thị ngày sinh, giờ sinh, nơi sinh cụ thể của User hoặc Partner trong câu trả lời, trừ khi User hỏi đích danh (VD: "Ngày sinh của tôi là số mấy?"). Hãy dùng các dữ liệu đã tính toán (Cung, Số chủ đạo...) để luận giải.
    2. **Xưng hô:** Gọi Partner là "Người ấy", "Đối phương" hoặc "Bạn đời". Không dùng tên riêng nếu không cần thiết.
    3. **Ưu tiên dữ liệu:** - Nếu có bài Tarot: Ưu tiên giải bài dựa trên câu hỏi.
       - Nếu câu hỏi về ngày cụ thể: Phân tích năng lượng ngày đó dựa trên tính toán đã cung cấp.
       - Nếu câu hỏi chung (VD: "Tôi thế nào?"): Tổng hợp từ Tử vi, Chiêm tinh, Thần số (nếu có dữ liệu) để đưa ra bức tranh đa chiều.
    4. **Phong cách:** Huyền bí, sâu sắc, nhưng thực tế và đưa ra lời khuyên cụ thể.
    """

    user_prompt = f"""
    DỮ LIỆU TÍNH TOÁN & TRA CỨU:
    {safe_context_str}
    
    THÔNG TIN BỔ TRỢ TỪ SÁCH (RAG):
    {rag_str}
    
    CÂU HỎI CỦA NGƯỜI DÙNG:
    "{question}"
    
    Hãy trả lời câu hỏi trên dựa vào các dữ liệu đã cung cấp.
    """

    # Gọi AI
    ai_response = call_bedrock_llm(system_instruction, user_prompt)
    
    return ai_response

# ==========================================
# 5. LAMBDA HANDLER
# ==========================================

def lambda_handler(event, context):
    try:
        # Parse Body
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)
            
        print("DEBUG Payload:", json.dumps(body, ensure_ascii=False))

        # Xử lý chính
        answer = process_request(body)
        
        # Format Response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'sessionId': body.get('data', {}).get('sessionId'),
                'reply': answer
            }, ensure_ascii=False)
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Lỗi hệ thống', 'details': str(e)})
        }