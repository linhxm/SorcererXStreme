import json
import pytest
import io
import sys
import os
from unittest.mock import MagicMock, patch

# =============================================================================
# 1. SETUP MÔI TRƯỜNG & IMPORT (QUAN TRỌNG)
# =============================================================================
# Thiết lập biến môi trường GIẢ trước khi import code chính để tránh lỗi KeyError/ImportError
os.environ['DYNAMODB_TABLE_NAME'] = 'Test_Metaphysical_Table'
os.environ['BEDROCK_MODEL_ID'] = 'amazon.nova-pro-v1:0'
os.environ['BEDROCK_REGION'] = 'us-east-1'

# Thêm đường dẫn thư mục cha để tìm thấy file lambda_function.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock boto3 TRƯỚC khi import để ngăn code chính kết nối AWS thật
with patch('boto3.client'), patch('boto3.resource'):
    try:
        import lambda_function
    except ImportError:
        # Fallback xử lý đường dẫn nếu chạy sai thư mục
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import lambda_function

# =============================================================================
# 2. HELPER FUNCTIONS & FIXTURES
# =============================================================================

def create_bedrock_stream(text_content):
    """
    Tạo giả lập StreamingBody của AWS bằng io.BytesIO.
    Đây là fix cho lỗi: 'Object of type MagicMock is not JSON serializable'
    """
    mock_response_data = {
        "output": {
            "message": {
                "content": [{"text": text_content}]
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 10}
    }
    # Chuyển Dict -> JSON String -> Bytes -> BytesIO Stream
    body_bytes = json.dumps(mock_response_data).encode('utf-8')
    return io.BytesIO(body_bytes)

@pytest.fixture
def mock_clients():
    """Fixture để kiểm soát Bedrock Client và DynamoDB Table"""
    # 1. Mock Bedrock
    mock_bedrock = MagicMock()
    lambda_function.bedrock_client = mock_bedrock
    
    # 2. Mock DynamoDB Table
    mock_table = MagicMock()
    lambda_function.table = mock_table
    
    return {"bedrock": mock_bedrock, "table": mock_table}

@pytest.fixture
def mock_lasotuvi_lib():
    """Fixture giả lập thư viện Tử Vi để không cần cài đặt thư viện thật"""
    # Mock Thiên Bàn
    mock_tb = MagicMock()
    mock_tb.ten = "Test User"
    mock_tb.banMenh = "Lộ Bàng Thổ"
    mock_tb.tenCuc = "Hỏa Lục Cục"
    mock_tb.canNamTen = "Giáp"
    mock_tb.chiNamTen = "Thìn"
    mock_tb.menhChu = "Tham Lang"
    mock_tb.thanChu = "Linh Tinh"
    mock_tb.ngayAm = 15
    mock_tb.thangAm = 8
    mock_tb.namAm = 2024

    # Mock Địa Bàn và các Cung
    mock_db = MagicMock()
    mock_db.cungMenh = 1
    mock_db.cungThan = 2
    
    ds_cung = {}
    for i in range(1, 14):
        cung = MagicMock()
        cung.cungTen = f"Cung {i}"
        cung.cungChu = "Mệnh" if i == 1 else "Tài Bạch"
        cung.cungSao = [{'saoTen': 'Tử Vi', 'saoLoai': 1}]
        ds_cung[i] = cung
    
    mock_db.thapNhiCung = ds_cung

    # Gán vào lambda_function
    lambda_function.lapDiaBan = MagicMock(return_value=mock_db)
    lambda_function.DiaBanClass = MagicMock()
    lambda_function.lapThienBan = MagicMock(return_value=mock_tb)
    
    return lambda_function.lapThienBan

# =============================================================================
# 3. TEST CASES
# =============================================================================

def test_handle_horoscope_tuvi(mock_clients, mock_lasotuvi_lib):
    """Test logic Tử Vi (Horoscope)"""
    bedrock = mock_clients['bedrock']
    
    # Setup Bedrock trả về stream bytes chuẩn
    bedrock.invoke_model.return_value = {'body': create_bedrock_stream("Luận giải: Lá số rất tốt.")}

    body = {
        "domain": "horoscope",
        "user_context": {
            "name": "Nam",
            "birth_date": "01-01-1990",
            "birth_time": "10:00",
            "gender": "male"
        }
    }

    response = lambda_function.lambda_handler(body, None)

    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    
    # Kiểm tra cấu trúc trả về
    assert res_body['domain'] == 'horoscope'
    assert 'answer' in res_body
    # Kiểm tra dữ liệu từ Mock Tử Vi có được dùng không
    assert res_body['answer']['summary']['ban_menh'] == "Lộ Bàng Thổ"
    assert "Luận giải" in res_body['answer']['analysis']

def test_handle_tarot_reading(mock_clients):
    """Test logic Tarot"""
    bedrock = mock_clients['bedrock']
    table = mock_clients['table']

    bedrock.invoke_model.return_value = {'body': create_bedrock_stream("The Sun là lá bài tích cực.")}
    # Mock DynamoDB trả về thông tin lá bài (Contexts lưu dạng String JSON)
    table.get_item.return_value = {
        'Item': {'contexts': json.dumps({'general_upright': 'Thành công, niềm vui'})}
    }

    body = {
        "domain": "tarot",
        "data": {
            "question": "Công việc thế nào?",
            "cards_drawn": [{"card_name": "The Sun", "is_upright": True, "position": "future"}]
        }
    }

    response = lambda_function.lambda_handler(body, None)

    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    assert "The Sun" in res_body['answer']

def test_handle_astrology(mock_clients):
    """Test logic Chiêm tinh (Astrology)"""
    bedrock = mock_clients['bedrock']
    table = mock_clients['table']

    bedrock.invoke_model.return_value = {'body': create_bedrock_stream("Ma Kết rất kiên trì.")}
    table.get_item.return_value = {
        'Item': {'contexts': json.dumps({'tinh-cach': 'Nghiêm túc'})}
    }

    body = {
        "domain": "astrology",
        "feature_type": "overview",
        "user_context": {"birth_date": "01-01-1990"} # 01/01 -> Ma Kết
    }

    response = lambda_function.lambda_handler(body, None)

    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    # Assert này đảm bảo code chạy hết flow
    assert res_body['domain'] == 'astrology'

def test_handle_numerology(mock_clients):
    """Test logic Thần số học"""
    bedrock = mock_clients['bedrock']
    table = mock_clients['table']

    bedrock.invoke_model.return_value = {'body': create_bedrock_stream("Số 10 là người lãnh đạo.")}
    table.get_item.return_value = {
        'Item': {'contexts': json.dumps({'tong-quan': 'Độc lập, mạnh mẽ'})}
    }

    body = {
        "domain": "numerology",
        "user_context": {"birth_date": "01-01-1990"}
    }

    response = lambda_function.lambda_handler(body, None)

    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    assert res_body['domain'] == 'numerology'

def test_missing_domain():
    """Test validation khi thiếu domain"""
    body = {"user_context": {}} # Thiếu key domain
    
    response = lambda_function.lambda_handler(body, None)
    
    # Code sẽ trả về 400 và lỗi Invalid domain
    assert response['statusCode'] == 400
    res_body = json.loads(response['body'])
    assert "error" in res_body

def test_invalid_domain_value():
    """Test domain không hỗ trợ"""
    body = {"domain": "unknown_magic"}
    
    response = lambda_function.lambda_handler(body, None)
    
    assert response['statusCode'] == 400
    assert "Invalid domain" in response['body']