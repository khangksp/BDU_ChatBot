FROM python:3.10-slim

# -------------------------------------------------------
# Environment variables
# -------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    TRANSFORMERS_CACHE=/app/ai_models_cache \
    HF_HOME=/app/ai_models_cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# -------------------------------------------------------
# 1. System dependencies
# -------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    postgresql-client \
    build-essential \
    libpq-dev \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 2. Python dependencies - tách thành các layer riêng
#    để Docker cache tái sử dụng tốt hơn khi chỉ đổi code
# -------------------------------------------------------
COPY requirements.txt .

# Cài uv - pip resolver thế hệ mới: nhanh hơn pip 10-100x,
# xử lý tốt dependency graph phức tạp (langchain, etc.)
RUN pip install uv

# Layer B: Các gói nặng nhất - ít thay đổi → cache lâu dài
RUN uv pip install --system "numpy<2.0.0"
RUN uv pip install --system torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system sentence-transformers==2.7.0

# Layer C: Toàn bộ requirements còn lại
RUN uv pip install --system -r requirements.txt

# -------------------------------------------------------
# 3. Copy code (sau khi install deps để tận dụng cache)
# -------------------------------------------------------
COPY . .

# -------------------------------------------------------
# 4. Tạo thư mục cần thiết
# -------------------------------------------------------
RUN mkdir -p logs media staticfiles ai_models_cache \
    && chmod 777 ai_models_cache

# -------------------------------------------------------
# 5. Copy entrypoint và cấp quyền thực thi
# -------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# -------------------------------------------------------
# 6. Expose port
# -------------------------------------------------------
EXPOSE 3019

# -------------------------------------------------------
# 7. Entrypoint
# -------------------------------------------------------
CMD ["/entrypoint.sh"]