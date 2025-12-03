# === CẤU HÌNH MÔI TRƯỜNG & MODEL ===
BEDROCK_REGION = 'ap-southeast-1'
EMBEDDING_MODEL_ID = 'cohere.embed-multilingual-v3'
LLM_MODEL_ID = 'apac.anthropic.claude-3-sonnet-20240229-v1:0'
DB_PORT = 5432

# Định dạng: (Tháng, Ngày bắt đầu)
ZODIAC_DATE_RANGES = [
    (1, 20, "Bảo Bình"),   # Từ 20/1
    (2, 19, "Song Ngư"),   # Từ 19/2
    (3, 21, "Bạch Dương"), # Từ 21/3
    (4, 20, "Kim Ngưu"),
    (5, 21, "Song Tử"),
    (6, 22, "Cự Giải"),
    (7, 23, "Sư Tử"),
    (8, 23, "Xử Nữ"),
    (9, 23, "Thiên Bình"),
    (10, 23, "Thiên Yết"),
    (11, 22, "Nhân Mã"),
    (12, 22, "Ma Kết")
]
# Lưu ý: Ma Kết là trường hợp đặc biệt vắt qua năm mới (22/12 - 19/01) => xử lý logic.
