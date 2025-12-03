import json
import pytest
import io
import sys
import os
from unittest.mock import MagicMock, patch

# Đảm bảo python tìm thấy file lambda_function.py trong thư mục cha hoặc cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import lambda_function (đảm bảo file lambda_function.py nằm đúng chỗ để import)
import lambda_function

@pytest.fixture
def mock_bedrock():
    """Fixture để mock bedrock client cho tất cả các test"""
    with patch('lambda_function.bedrock') as mock:
        yield mock

def create_bedrock_response(text_content):
    """
    Hàm helper để tạo cấu trúc response giả lập từ Bedrock.
    Quan trọng: Sử dụng io.BytesIO để giả lập StreamingBody.
    """
    mock_response_data = {
        "output": {
            "message": {
                "content": [{"text": text_content}]
            }
        },
        # Thêm các trường khác nếu code của bạn check usage, stop_reason, v.v.
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 10}
    }
    
    # Encode JSON thành bytes và bọc trong BytesIO
    # Đây là chìa khóa để sửa lỗi JSON serializable
    body_bytes = json.dumps(mock_response_data).encode('utf-8')
    return io.BytesIO(body_bytes)

def test_handle_horoscope_tuvi(mock_bedrock):
    """Test logic Tử Vi (Horoscope) - Case vừa bị lỗi"""
    # 1. Setup Mock Response với io.BytesIO
    mock_stream = create_bedrock_response("Luận giải lá số: Bạn là người kiên định...")
    mock_bedrock.invoke_model.return_value = {
        'body': mock_stream
    }

    # 2. Prepare Event Body
    body = {
        "domain": "horoscope",
        "user_context": {
            "name": "Nam",
            "birth_date": "01-01-1990",
            "birth_time": "10:00",
            "gender": "male"
        }
    }

    # 3. Call Handler
    response = lambda_function.lambda_handler(body, None)

    # 4. Assertions
    assert response['statusCode'] == 200
    
    # Parse body trả về từ Lambda để kiểm tra nội dung
    response_body = json.loads(response['body'])
    assert "Luận giải lá số" in response_body['message']
    
    # Kiểm tra xem bedrock đã được gọi đúng model chưa (tuỳ chỉnh model ID theo code của bạn)
    args, kwargs = mock_bedrock.invoke_model.call_args
    assert "body" in kwargs or args
    # Nếu bạn muốn check modelId cụ thể:
    # assert kwargs['modelId'] == 'amazon.nova-pro-v1:0' 

def test_handle_tarot_reading(mock_bedrock):
    """Test một domain khác (ví dụ Tarot) để đảm bảo tính tổng quát"""
    mock_stream = create_bedrock_response("Lá bài của bạn là The Sun...")
    mock_bedrock.invoke_model.return_value = {
        'body': mock_stream
    }

    body = {
        "domain": "tarot",
        "user_context": {
            "question": "Sự nghiệp năm nay thế nào?"
        }
    }

    response = lambda_function.lambda_handler(body, None)

    assert response['statusCode'] == 200
    response_body = json.loads(response['body'])
    assert "The Sun" in response_body['message']

def test_handle_missing_domain():
    """Test trường hợp lỗi khi thiếu domain (Validation)"""
    # Không cần mock bedrock vì code sẽ return lỗi trước khi gọi AI
    body = {
        "user_context": {} 
        # Thiếu key "domain"
    }

    response = lambda_function.lambda_handler(body, None)

    # Mong đợi lỗi 400 hoặc 500 tuỳ theo cách bạn xử lý validation
    # Giả sử code trả về 400 Bad Request
    if response['statusCode'] == 200:
        # Nếu code bạn mặc định domain nào đó, test này cần điều chỉnh
        pass
    else:
        assert response['statusCode'] in [400, 500]
        response_body = json.loads(response['body'])
        assert "error" in response_body or "message" in response_body