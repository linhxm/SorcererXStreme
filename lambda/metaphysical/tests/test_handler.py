import sys
import json
import os
import pytest
from unittest.mock import MagicMock, patch

# ==========================================
# PHẦN 1: MOCK MÔI TRƯỜNG & MODULE CON
# ==========================================
os.environ["BEDROCK_REGION"] = "ap-southeast-1"
os.environ["LLM_MODEL_ID"] = "amazon.nova-pro-v1:0"
# Set biến môi trường này để dự phòng
os.environ["DYNAMODB_TABLE_NAME"] = "TestTable"

# --- 1. MOCK FILE 'constants' (QUAN TRỌNG: FIX LỖI IMPORT ERROR) ---
# Tạo một module giả
mock_constants = MagicMock()
# Gán các giá trị giả định mà lambda_function cần import
mock_constants.BEDROCK_REGION = "ap-southeast-1"
mock_constants.LLM_MODEL_ID = "amazon.nova-pro-v1:0"
mock_constants.DYNAMODB_TABLE_NAME = "TestTable" # <--- Khắc phục lỗi thiếu biến này

# Đăng ký module giả vào hệ thống
sys.modules["constants"] = mock_constants

# --- 2. MOCK FILE 'prompts' ---
mock_prompts = MagicMock()
# Giả lập các hàm get_prompt trả về chuỗi rỗng
mock_prompts.get_tarot_prompt.return_value = "Tarot Prompt"
mock_prompts.get_astrology_prompt.return_value = "Astro Prompt"
mock_prompts.get_numerology_prompt.return_value = "Numero Prompt"
mock_prompts.get_horoscope_prompt.return_value = "Horoscope Prompt"

sys.modules["prompts"] = mock_prompts

# --- 3. Mock thư viện 'lasotuvi' ---
mock_lasotuvi = MagicMock()
mock_app = MagicMock()
mock_diaban = MagicMock()
mock_thienban = MagicMock()

# Setup giả lập class ThienBan trả về object có thuộc tính
mock_tb_instance = MagicMock()
mock_tb_instance.ten = "Test User"
mock_tb_instance.banMenh = "Lộ Bàng Thổ"
mock_tb_instance.tenCuc = "Thủy Nhị Cục"
mock_tb_instance.ngayAm = 1
mock_tb_instance.thangAm = 1
mock_tb_instance.namAm = 1990
mock_thienban.lapThienBan.return_value = mock_tb_instance 

# Setup giả lập class DiaBan
mock_db_instance = MagicMock()
mock_db_instance.cungMenh = 1
mock_db_instance.cungThan = 1
# Giả lập thapNhiCung là list chứa các mock object
mock_cung = MagicMock()
mock_cung.cungTen = "Tý"
mock_cung.cungChu = "Mệnh"
mock_cung.cungSao = [{'saoTen': 'Tử Vi', 'saoLoai': 1}]
mock_db_instance.thapNhiCung = {i: mock_cung for i in range(1, 14)} # Dict giả lập cung 1-13

mock_app.lapDiaBan.return_value = mock_db_instance

sys.modules["lasotuvi"] = mock_lasotuvi
sys.modules["lasotuvi.App"] = mock_app
sys.modules["lasotuvi.DiaBan"] = mock_diaban
sys.modules["lasotuvi.ThienBan"] = mock_thienban

# --- 4. Mock Boto3 ---
mock_boto3 = MagicMock()
mock_bedrock = MagicMock()
mock_dynamo = MagicMock()
mock_table = MagicMock()

mock_boto3.client.return_value = mock_bedrock
mock_boto3.resource.return_value = mock_dynamo
mock_dynamo.Table.return_value = mock_table

sys.modules["boto3"] = mock_boto3

# ==========================================
# PHẦN 2: IMPORT CODE CHÍNH
# ==========================================
# Lúc này import an toàn vì constants và prompts đã được mock
import lambda_function

# ==========================================
# PHẦN 3: TEST CASES
# ==========================================

def test_handle_astrology_overview_nova():
    """Test logic Cung Hoàng Đạo với Mock Amazon Nova Pro"""
    mock_table.get_item.return_value = {
        'Item': {'contexts': '{"tinh-cach": "Tốt"}'}
    }
    
    # Mock response Nova Pro
    mock_response_body = json.dumps({
        "output": {
            "message": {
                "content": [
                    {"text": "Dự đoán Nova Pro: Bạn là người tuyệt vời."}
                ]
            }
        }
    })
    
    mock_bedrock.invoke_model.return_value = {
        'body': MagicMock(read=lambda: mock_response_body)
    }

    body = {
        "domain": "astrology",
        "feature_type": "overview",
        "user_context": {"birth_date": "20-01-1995"}
    }
    
    response = lambda_function.lambda_handler(body, None)
    
    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    assert "Dự đoán Nova Pro" in res_body['answer']

def test_handle_horoscope_tuvi():
    """Test logic Tử Vi (Horoscope)"""
    mock_response_body = json.dumps({
        "output": {"message": {"content": [{"text": "Luận giải lá số..."}]}}
    })
    mock_bedrock.invoke_model.return_value = {
        'body': MagicMock(read=lambda: mock_response_body)
    }

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
    
    # Kiểm tra cấu trúc JSON trả về
    assert "summary" in res_body['answer']
    assert res_body['answer']['analysis'] == "Luận giải lá số..."

def test_error_handling():
    body = {"domain": "unknown_domain"}
    response = lambda_function.lambda_handler(body, None)
    assert response['statusCode'] == 400
    assert "Invalid domain" in response['body']