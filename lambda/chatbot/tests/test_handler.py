import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# ==========================================
# PHẦN 1: MOCK THƯ VIỆN (CHẠY TRƯỚC KHI IMPORT CODE)
# ==========================================
# Mục đích: Giúp test chạy được mà không cần cài boto3/pinecone thật

# 1. Mock boto3 và các module con
mock_boto3 = MagicMock()
mock_dynamodb = MagicMock()
mock_conditions = MagicMock()
# Giả lập class Key
mock_conditions.Key = MagicMock()

sys.modules["boto3"] = mock_boto3
sys.modules["boto3.dynamodb"] = mock_dynamodb
sys.modules["boto3.dynamodb.conditions"] = mock_conditions

# 2. Mock Pinecone
mock_pinecone = MagicMock()
sys.modules["pinecone"] = mock_pinecone

# ==========================================
# PHẦN 2: IMPORT CODE CHÍNH
# ==========================================
# Bây giờ mới được import, sau khi đã mock xong ở trên
import lambda_function

# ==========================================
# PHẦN 3: FIXTURES & TEST CASES
# ==========================================

@pytest.fixture
def valid_payload():
    return {
        "user_context": {
            "name": "User Test",
            "birth_date": "10/10/1995",
            "gender": "Nam"
        },
        "partner_context": {},
        "data": {
            "sessionId": "session-123",
            "question": "Tử vi năm nay của tôi thế nào?",  
            "tarot_cards": []
        }
    }

@patch('lambda_function.load_history')
@patch('lambda_function.append_message')
@patch('lambda_function.query_pinecone_rag')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow_deep_dive(mock_call_ai, mock_rag, mock_append, mock_history, valid_payload):
    """
    Case 1: Câu hỏi Huyền học (Deep Dive)
    Mong đợi: Phải tính toán, RAG và GỌI BEDROCK AI.
    """
    # Setup Mock
    mock_history.return_value = ""
    mock_rag.return_value = ["[Tử Vi]: Mệnh VCD..."]
    mock_call_ai.return_value = "Năm nay bạn có sao Thiên Việt..."

    # Thực thi
    event = {"body": json.dumps(valid_payload)}
    response = lambda_function.lambda_handler(event, None)

    # Assertions
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["reply"] == "Năm nay bạn có sao Thiên Việt..."
    
    # AI phải được gọi
    mock_call_ai.assert_called_once()
    # DynamoDB ghi 2 lần
    assert mock_append.call_count == 2


@patch('lambda_function.append_message')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow_chit_chat(mock_call_ai, mock_append):
    """
    Case 2: Câu hỏi Xã giao (Chit-chat) - Ví dụ: "Hi"
    Mong đợi: Trả lời nhanh, KHÔNG GỌI BEDROCK AI.
    """
    payload = {
        "data": {
            "sessionId": "session-chit-chat",
            "question": "Hi" 
        }
    }
    event = {"body": json.dumps(payload)}
    
    response = lambda_function.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    
    # Kiểm tra câu trả lời mặc định
    assert "trợ lý huyền học" in body["reply"]
    
    # QUAN TRỌNG: AI KHÔNG ĐƯỢC GỌI
    mock_call_ai.assert_not_called()
    # Ghi log DB ít nhất 1 lần
    assert mock_append.call_count >= 1


def test_missing_session_id():
    """Case 3: Thiếu Session ID -> Lỗi 400"""
    event = {
        "body": json.dumps({
            "data": {
                "question": "Test"
            }
        })
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "Missing sessionId" in response["body"]


def test_missing_question_and_tarot():
    """Case 4: Thiếu cả Question và Tarot -> Lỗi 400"""
    event = {
        "body": json.dumps({
            "data": {
                "sessionId": "123"
            }
        })
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400


def test_invalid_json_body():
    """Case 5: JSON lỗi format -> Lỗi 400"""
    event = {
        "body": "{ json nay bi thieu dau ngoac"
    }
    response = lambda_function.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "Invalid JSON Body" in response["body"]