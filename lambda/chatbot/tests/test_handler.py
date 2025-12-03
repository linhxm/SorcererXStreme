import sys
import json
import os
import pytest
from unittest.mock import MagicMock, patch

# ==========================================
# PHẦN 1: MOCK (GIẢ LẬP) CÁC THƯ VIỆN NGOÀI
# ==========================================
# QUAN TRỌNG: Phải mock đầy đủ cấu trúc thư viện trước khi import code chính

# 1. Mock biến môi trường
os.environ["DDB_MESSAGE_TABLE"] = "TestTable"
os.environ["BEDROCK_LLM_MODEL_ID"] = "test-model"

# 2. Mock cấu trúc boto3 phức tạp
mock_boto3 = MagicMock()
mock_dynamodb_module = MagicMock()
mock_conditions_module = MagicMock()

# Giả lập class 'Key' bên trong conditions
mock_conditions_module.Key = MagicMock()

# Gán vào sys.modules để Python tìm thấy khi chạy lệnh: 
# "from boto3.dynamodb.conditions import Key"
sys.modules["boto3"] = mock_boto3
sys.modules["boto3.dynamodb"] = mock_dynamodb_module
sys.modules["boto3.dynamodb.conditions"] = mock_conditions_module

# 3. Mock Pinecone
mock_pinecone_module = MagicMock()
sys.modules["pinecone"] = mock_pinecone_module

# 4. Setup hành vi cho boto3 resource/client (để test logic bên dưới)
mock_ddb_resource = MagicMock()
mock_table = MagicMock()
mock_ddb_resource.Table.return_value = mock_table
mock_boto3.resource.return_value = mock_ddb_resource

mock_bedrock_client = MagicMock()
mock_boto3.client.return_value = mock_bedrock_client

# ==========================================
# PHẦN 2: IMPORT CODE CỦA BẠN
# ==========================================
import lambda_function

# ==========================================
# PHẦN 3: CÁC TEST CASE
# ==========================================

def test_missing_session_id():
    """Case 1: Gửi input thiếu sessionId -> Phải lỗi 400"""
    event = {
        "body": json.dumps({
            "data": {
                "question": "Xin chào"
            }
        })
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "Missing sessionId" in response["body"]

def test_missing_question():
    """Case 2: Gửi input có sessionId nhưng thiếu question -> Phải lỗi 400"""
    event = {
        "body": json.dumps({
            "data": {
                "sessionId": "session-test-123"
            }
        })
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400
    # Kiểm tra lỗi có nhắc đến question hoặc missing
    assert "question" in response["body"] or "Missing" in response["body"]

def test_invalid_json_body():
    """Case 3: Body gửi lên bị lỗi format -> Phải lỗi 400"""
    event = {
        "body": "{ day la chuoi json bi loi"
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "Invalid JSON Body" in response["body"]

@patch('lambda_function.load_history')
@patch('lambda_function.append_message')
@patch('lambda_function.query_pinecone_rag')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow(mock_call_ai, mock_rag, mock_append, mock_load_history):
    """Case 4: Mọi thứ đều đúng -> Phải trả về 200"""
    # Setup kết quả giả định
    mock_load_history.return_value = []
    mock_rag.return_value = [{"score": 0.9, "title": "Doc", "content": "Content"}]
    mock_call_ai.return_value = "Xin chào AI."

    event = {
        "body": json.dumps({
            "user_context": {},
            "partner_context": {},
            "data": {
                "sessionId": "session-ok",
                "question": "Hi"
            }
        })
    }

    response = lambda_function.lambda_handler(event, None)

    assert response["statusCode"] == 200
    mock_call_ai.assert_called_once()