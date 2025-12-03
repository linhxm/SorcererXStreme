import sys
import json
import os
import pytest
from unittest.mock import MagicMock, patch

# ==========================================
# PHẦN 1: MOCK (GIẢ LẬP) CÁC THƯ VIỆN NGOÀI
# ==========================================
# Mục đích: Ngăn code tự động kết nối AWS/Pinecone thật khi chạy test.

# 1. Mock biến môi trường
os.environ["DDB_MESSAGE_TABLE"] = "TestTable"
os.environ["BEDROCK_LLM_MODEL_ID"] = "test-model"

# 2. Mock boto3 (AWS)
mock_boto3 = MagicMock()

# Mock DynamoDB Table resource
mock_ddb_resource = MagicMock()
mock_table = MagicMock()
mock_ddb_resource.Table.return_value = mock_table
mock_boto3.resource.return_value = mock_ddb_resource

# Mock Bedrock Client
mock_bedrock_client = MagicMock()
mock_boto3.client.return_value = mock_bedrock_client

# 3. Mock Pinecone
mock_pinecone_module = MagicMock()

# 4. Gán Mock vào sys.modules để Python nhận diện thư viện giả
sys.modules["boto3"] = mock_boto3
sys.modules["pinecone"] = mock_pinecone_module

# ==========================================
# PHẦN 2: IMPORT CODE CỦA BẠN
# ==========================================
import lambda_function

# ==========================================
# PHẦN 3: CÁC TEST CASE (KỊCH BẢN KIỂM TRA)
# ==========================================

def test_missing_session_id():
    """
    Case 1: Gửi input thiếu sessionId -> Phải lỗi 400
    """
    event = {
        "body": json.dumps({
            "data": {
                "question": "Xin chào"
                # Thiếu sessionId
            }
        })
    }
    
    response = lambda_function.lambda_handler(event, None)
    
    assert response["statusCode"] == 400
    assert "Missing sessionId" in response["body"]

def test_missing_question():
    """
    Case 2: Gửi input có sessionId nhưng thiếu question -> Phải lỗi 400
    """
    event = {
        "body": json.dumps({
            "data": {
                "sessionId": "session-test-123"
                # Thiếu question
            }
        })
    }
    
    response = lambda_function.lambda_handler(event, None)
    
    assert response["statusCode"] == 400
    # Kiểm tra xem lỗi có liên quan đến việc thiếu dữ liệu không
    assert "question" in response["body"] or "Missing" in response["body"]

def test_invalid_json_body():
    """
    Case 3: Body gửi lên bị lỗi format (không phải JSON chuẩn) -> Phải lỗi 400
    """
    event = {
        "body": "{ day la chuoi json bi loi"
    }
    
    response = lambda_function.lambda_handler(event, None)
    
    assert response["statusCode"] == 400
    assert "Invalid JSON Body" in response["body"]

# Test luồng xử lý thành công (Happy Path)
# Dùng @patch để giả lập kết quả trả về của các hàm nội bộ
@patch('lambda_function.load_history')
@patch('lambda_function.append_message')
@patch('lambda_function.query_pinecone_rag')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow(mock_call_ai, mock_rag, mock_append, mock_load_history):
    """
    Case 4: Mọi thứ đều đúng -> Phải trả về 200 và câu trả lời AI
    """
    # 1. Setup kết quả giả định (Mock Return Values)
    mock_load_history.return_value = [] # Lịch sử chat rỗng
    mock_rag.return_value = [           # Tìm thấy 1 tài liệu
        {"score": 0.9, "title": "Test Doc", "content": "Nội dung test"}
    ]
    mock_call_ai.return_value = "Xin chào, đây là câu trả lời từ Test."

    # 2. Tạo input hợp lệ đầy đủ
    event = {
        "body": json.dumps({
            "user_context": {"name": "User A"},
            "partner_context": {"name": "Partner B"},
            "data": {
                "sessionId": "session-test-ok",
                "question": "Bạn là ai?"
            }
        })
    }

    # 3. Chạy hàm chính
    response = lambda_function.lambda_handler(event, None)

    # 4. Kiểm tra (Assert)
    # Kiểm tra Status Code
    assert response["statusCode"] == 200
    
    # Kiểm tra Body trả về
    body_data = json.loads(response["body"])
    assert body_data["sessionId"] == "session-test-ok"
    assert body_data["reply"] == "Xin chào, đây là câu trả lời từ Test."
    
    # Kiểm tra logic: Code phải gọi các hàm con
    mock_load_history.assert_called_once() # Phải gọi lấy lịch sử
    mock_rag.assert_called_once()          # Phải gọi tìm kiếm Pinecone
    mock_call_ai.assert_called_once()      # Phải gọi AI trả lời
    mock_append.assert_called()            # Phải gọi lưu tin nhắn mới