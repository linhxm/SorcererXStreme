import sys
import json
import os
import pytest
from unittest.mock import MagicMock, patch

# ==========================================
# PHẦN 1: MOCK MÔI TRƯỜNG & THƯ VIỆN
# ==========================================
# Phải set biến môi trường giả TRƯỚC khi import code
os.environ['S3_BUCKET_NAME'] = 'test-bucket'
os.environ['S3_FILE_KEY'] = 'test-data.jsonl'
os.environ['DYNAMODB_TABLE'] = 'TestTable'
os.environ['PINECONE_API_KEY'] = 'test-key'
os.environ['PINECONE_HOST'] = 'test-host'
os.environ['BEDROCK_REGION'] = 'us-east-1'

# Mock Boto3 (AWS Services)
mock_boto3 = MagicMock()

# 1. Mock S3 Client
mock_s3_client = MagicMock()
# 2. Mock Bedrock Client
mock_bedrock_client = MagicMock()
# 3. Mock DynamoDB Resource & Table
mock_ddb_resource = MagicMock()
mock_table = MagicMock()
mock_ddb_resource.Table.return_value = mock_table

# Setup return cho boto3.client và boto3.resource
def side_effect_client(service_name, **kwargs):
    if service_name == 's3': return mock_s3_client
    if service_name == 'bedrock-runtime': return mock_bedrock_client
    return MagicMock()

mock_boto3.client.side_effect = side_effect_client
mock_boto3.resource.return_value = mock_ddb_resource

# Mock Pinecone
mock_pinecone_module = MagicMock()
mock_pinecone_instance = MagicMock()
mock_index = MagicMock()
mock_pinecone_instance.Index.return_value = mock_index
mock_pinecone_module.Pinecone.return_value = mock_pinecone_instance

# Gán vào sys.modules
sys.modules["boto3"] = mock_boto3
sys.modules["pinecone"] = mock_pinecone_module

# ==========================================
# PHẦN 2: IMPORT CODE CHÍNH
# ==========================================
import lambda_function

# ==========================================
# PHẦN 3: TEST CASES
# ==========================================

def test_flatten_contexts():
    """Test hàm tiện ích làm phẳng dữ liệu context"""
    input_ctx = {
        "description": "Là một người máy",
        "hobbies": ["đọc sách", "coding"]
    }
    result = lambda_function.flatten_contexts(input_ctx)
    assert "description: Là một người máy" in result
    assert "hobbies: đọc sách, coding" in result

def test_get_embedding_success():
    """Test gọi Bedrock thành công"""
    # Giả lập phản hồi từ Bedrock
    mock_response = {
        'body': MagicMock()
    }
    mock_response['body'].read.return_value = json.dumps({
        "embeddings": [[0.1, 0.2, 0.3]]
    }).encode('utf-8')
    
    mock_bedrock_client.invoke_model.return_value = mock_response
    
    vector = lambda_function.get_embedding("Test text")
    assert vector == [0.1, 0.2, 0.3]

def test_get_embedding_fail():
    """Test gọi Bedrock bị lỗi"""
    mock_bedrock_client.invoke_model.side_effect = Exception("AWS Error")
    vector = lambda_function.get_embedding("Test text")
    assert vector is None

def test_lambda_handler_success():
    """Test luồng chạy chính thành công (Happy Flow)"""
    # ====================================================
    # BƯỚC QUAN TRỌNG: RESET TRẠNG THÁI MOCK
    # ====================================================
    # 1. Xóa bộ đếm số lần gọi (đưa về 0)
    mock_bedrock_client.reset_mock()
    
    # 2. Xóa lệnh báo lỗi (Exception) từ bài test trước đó
    mock_bedrock_client.invoke_model.side_effect = None 
    # ====================================================

    # 1. Giả lập S3 trả về file JSONL nội dung giả
    fake_content = (
        '{"category": "test", "entity_name": "A", "keywords": ["k1"], "contexts": {"desc": "val"}}\n'
        '{"category": "test", "entity_name": "B", "keywords": ["k2"], "contexts": {"desc": "val"}}'
    )
    mock_s3_body = MagicMock()
    mock_s3_body.read.return_value.decode.return_value = fake_content
    mock_s3_client.get_object.return_value = {'Body': mock_s3_body}

    # 2. Giả lập Bedrock trả về vector
    mock_bedrock_resp = {'body': MagicMock()}
    mock_bedrock_resp['body'].read.return_value = json.dumps({"embeddings": [[0.1]]}).encode('utf-8')
    mock_bedrock_client.invoke_model.return_value = mock_bedrock_resp

    # 3. Giả lập DynamoDB Batch Writer (Context Manager)
    mock_batch = MagicMock()
    mock_table.batch_writer.return_value.__enter__.return_value = mock_batch

    # --- CHẠY HÀM ---
    response = lambda_function.lambda_handler({}, None)

    # --- KIỂM TRA ---
    assert response['statusCode'] == 200
    
    # Kiểm tra S3 đã được gọi
    mock_s3_client.get_object.assert_called_once()
    
    # Kiểm tra DynamoDB đã put 2 item
    assert mock_batch.put_item.call_count == 2
    
    # Kiểm tra Bedrock được gọi 2 lần (cho 2 item)
    # Lúc này call_count đã được reset về 0 ở đầu hàm, nên assert 2 sẽ đúng
    assert mock_bedrock_client.invoke_model.call_count == 2