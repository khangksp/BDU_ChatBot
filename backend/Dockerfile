# Backend Dockerfile - FIXED VERSION
FROM python:3.10-slim

# Set environment variable

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# ✅ FIXED: Install Python dependencies with proper order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch first (stable)
RUN pip install --no-cache-dir torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Install compatible versions of problematic packages
RUN pip install --no-cache-dir \
    huggingface-hub==0.19.4 \
    tokenizers>=0.14.0 \
    transformers==4.35.2

# Install sentence-transformers after fixing dependencies
RUN pip install --no-cache-dir sentence-transformers==2.3.1

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for PostgreSQL
RUN pip install --no-cache-dir psycopg2-binary

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs media static

# Make port 8000 available
EXPOSE 8000

# Create optimized entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== BDU ChatBot Docker Setup (Fixed Version) ==="\n\
\n\
# Wait for PostgreSQL\n\
echo "⏳ Waiting for PostgreSQL..."\n\
while ! pg_isready -h postgres -p 5432 -U postgres; do\n\
    sleep 1\n\
done\n\
echo "✅ PostgreSQL is ready"\n\
\n\
# Configure Django to use PostgreSQL\n\
export DATABASE_URL="postgresql://postgres:postgres123@postgres:5432/bdu_chatbot"\n\
\n\
# Show Python and package versions for debugging\n\
echo "🐍 Python version: $(python --version)"\n\
echo "📦 Checking key packages..."\n\
python -c "import sentence_transformers; print(f\"✅ sentence-transformers: {sentence_transformers.__version__}\")" || echo "❌ sentence-transformers failed"\n\
python -c "import transformers; print(f\"✅ transformers: {transformers.__version__}\")" || echo "❌ transformers failed"\n\
python -c "import huggingface_hub; print(f\"✅ huggingface-hub: {huggingface_hub.__version__}\")" || echo "❌ huggingface-hub failed"\n\
\n\
# Run migrations\n\
echo "📊 Running database migrations..."\n\
python manage.py migrate --noinput\n\
\n\
# Create superuser only if not exists\n\
echo "👤 Setting up admin user..."\n\
python manage.py shell -c "\n\
from django.contrib.auth import get_user_model;\n\
User = get_user_model();\n\
if not User.objects.filter(username='\''admin'\'').exists():\n\
    User.objects.create_superuser('\''admin'\'', '\''admin@bdu.edu.vn'\'', '\''admin123456'\'', faculty_code='\''ADMIN001'\'', full_name='\''Administrator'\'');\n\
    print('\''✅ Admin user created'\'');\n\
else:\n\
    print('\''ℹ️ Admin user already exists'\'');\n\
" 2>/dev/null || echo "⚠️ Admin user creation skipped"\n\
\n\
# Test AI models loading\n\
echo "🤖 Testing AI models..."\n\
python -c "\n\
try:\n\
    from sentence_transformers import SentenceTransformer;\n\
    print('\''✅ SentenceTransformer import successful'\'');\n\
    # model = SentenceTransformer('\''keepitreal/vietnamese-sbert'\'');\n\
    # print('\''✅ Vietnamese SBERT model loaded'\'');\n\
except Exception as e:\n\
    print(f'\''⚠️ SentenceTransformer test failed: {e}'\'');\n\
" || echo "⚠️ AI model test skipped"\n\
\n\
# Start Django server\n\
echo "🚀 Starting Django server..."\n\
python manage.py runserver 0.0.0.0:8000\n' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Run entrypoint script
CMD ["/entrypoint.sh"]