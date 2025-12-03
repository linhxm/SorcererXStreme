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
os.environ["DYNAMODB_TABLE_NAME"] = "TestTable"

# --- Mock thư viện 'lasotuvi' ---
# Vì module này nằm local, ta mock nó để test chạy được mà không cần quan tâm logic bên trong
mock_lasotuvi = MagicMock()
mock_app = MagicMock()
mock_diaban = MagicMock()
mock_thienban = MagicMock()

# Setup giả lập cấu trúc class trả về cho Tử Vi
mock_tb_instance = MagicMock()
mock_tb_instance.ten = "Test User"
mock_tb_instance.banMenh = "Lộ Bàng Thổ"
mock_tb_instance.tenCuc = "Thủy Nhị Cục"
mock_thienban.lapThienBan.return_value = mock_tb_instance # Khi gọi hàm thì trả về object giả này

sys.modules["lasotuvi"] = mock_lasotuvi
sys.modules["lasotuvi.App"] = mock_app
sys.modules["lasotuvi.DiaBan"] = mock_diaban
sys.modules["lasotuvi.ThienBan"] = mock_thienban

# --- Mock Boto3 ---
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
# Thêm đường dẫn hiện tại vào sys.path để import được constants và prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lambda_function

# ==========================================
# PHẦN 3: TEST CASES (Cập nhật cho Nova Pro)
# ==========================================

def test_handle_astrology_overview_nova():
    """Test logic Cung Hoàng Đạo với Mock Amazon Nova Pro"""
    # 1. Mock DB trả về thông tin cung
    mock_table.get_item.return_value = {
        'Item': {'contexts': '{"tinh-cach": "Tốt"}'}
    }
    
    # 2. Mock Bedrock trả về văn bản theo cấu trúc NOVA PRO
    # Cấu trúc: output -> message -> content -> [ {text: ...} ]
    mock_response_body = json.dumps({
        "output": {
            "message": {
                "content": [
                    {"text": "Dự đoán Nova Pro: Bạn là người tuyệt vời."}
                ]
            }
        }
    })
    
    # Setup mock để khi gọi read() sẽ trả về json trên
    mock_bedrock.invoke_model.return_value = {
        'body': MagicMock(read=lambda: mock_response_body)
    }

    # 3. Input
    body = {
        "domain": "astrology",
        "feature_type": "overview",
        "user_context": {"birth_date": "20-01-1995"} # Bảo Bình
    }
    
    # 4. Chạy hàm handler
    response = lambda_function.lambda_handler(body, None)
    
    # 5. Kiểm tra kết quả
    assert response['statusCode'] == 200
    res_body = json.loads(response['body'])
    
    # Kiểm tra xem code có lấy đúng text từ Nova mock không
    assert "Dự đoán Nova Pro" in res_body['answer']

def test_handle_horoscope_tuvi():
    """Test logic Tử Vi (Horoscope)"""
    # Test xem nó có gọi thư viện lasotuvi (đã mock) và trả về JSON không
    
    # Mock Bedrock cho phần luận giải
    mock_response_body = json.dumps({
        "output": {
            "message": {
                "content": [{"text": "Luận giải lá số..."}]
            }
        }
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
    
    # Kiểm tra cấu trúc trả về đặc thù của Tử Vi (JSON chứ không phải string đơn)
    assert "result" in res_body or "answer" in res_body
    # Code handler của bạn trả về 'answer' chứa dict
    answer_data = res_body['answer']
    assert "summary" in answer_data
    assert answer_data['analysis'] == "Luận giải lá số..."

def test_error_handling():
    """Test trường hợp domain không hợp lệ"""
    body = {"domain": "unknown_domain"}
    response = lambda_function.lambda_handler(body, None)
    assert response['statusCode'] == 400
    assert "Invalid domain" in response['body']

def test_nova_pro_structure_check():
    """Test kỹ xem code có gọi đúng Model ID của Nova không"""
    body = {
        "domain": "numerology", # Test qua thần số học cho đơn giản
        "user_context": {"birth_date": "01-01-1990"}
    }
    
    # Mock return để không bị lỗi
    mock_table.get_item.return_value = {'Item': {'contexts': '{}'}}
    mock_response_body = json.dumps({
        "output": {"message": {"content": [{"text": "OK"}]}}
    })
    mock_bedrock.invoke_model.return_value = {'body': MagicMock(read=lambda: mock_response_body)}
    
    lambda_function.lambda_handler(body, None)
    
    # Kiểm tra tham số gọi đi
    call_args = mock_bedrock.invoke_model.call_args
    # call_args[1] là kwargs, kiểm tra modelId
    assert call_args[1]['modelId'] == "amazon.nova-pro-v1:0"
    
    # Kiểm tra body gửi đi có đúng format Nova không (inferenceConfig)
    request_body = json.loads(call_args[1]['body'])
    assert "inferenceConfig" in request_body
    assert "messages" in request_body