import json
import pytest
from unittest.mock import patch, MagicMock
import lambda_function

# ==========================================
# FIXTURES & MOCK DATA
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
            "question": "Tử vi năm nay của tôi thế nào?",  # Câu hỏi đủ dài để kích hoạt AI
            "tarot_cards": []
        }
    }

# ==========================================
# TEST CASES
# ==========================================

@patch('lambda_function.load_history')
@patch('lambda_function.append_message')
@patch('lambda_function.query_pinecone_rag')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow_deep_dive(mock_call_ai, mock_rag, mock_append, mock_history, valid_payload):
    """
    Case 1: Câu hỏi Huyền học (Deep Dive)
    Mong đợi: Phải tính toán, RAG và GỌI BEDROCK AI.
    """
    # 1. Setup Mock
    mock_history.return_value = ""
    mock_rag.return_value = ["[Tử Vi]: Mệnh VCD..."]
    mock_call_ai.return_value = "Năm nay bạn có sao Thiên Việt..."

    # 2. Thực thi
    event = {"body": json.dumps(valid_payload)}
    response = lambda_function.lambda_handler(event, None)

    # 3. Assertions
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["reply"] == "Năm nay bạn có sao Thiên Việt..."
    
    # QUAN TRỌNG: Kiểm tra AI đã được gọi vì đây là câu hỏi tử vi
    mock_call_ai.assert_called_once()
    # Kiểm tra DynamoDB ghi 2 lần (Hỏi + Trả lời)
    assert mock_append.call_count == 2


@patch('lambda_function.append_message')
@patch('lambda_function.call_bedrock_nova')
def test_success_flow_chit_chat(mock_call_ai, mock_append):
    """
    Case 2: Câu hỏi Xã giao (Chit-chat optimization)
    Mong đợi: Trả lời nhanh, KHÔNG GỌI BEDROCK AI.
    """
    payload = {
        "data": {
            "sessionId": "session-chit-chat",
            "question": "Hi" # Câu hỏi ngắn, không keyword
        }
    }
    event = {"body": json.dumps(payload)}
    
    response = lambda_function.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    
    # Kiểm tra câu trả lời mặc định
    assert "trợ lý huyền học" in body["reply"]
    
    # QUAN TRỌNG: AI KHÔNG ĐƯỢC GỌI (Tiết kiệm chi phí)
    mock_call_ai.assert_not_called()
    # Vẫn phải ghi log vào DB 1 lần (câu trả lời của bot) - hoặc tùy logic code
    # Trong code mới của bạn, chit-chat gọi append_message 1 lần cho câu reply
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