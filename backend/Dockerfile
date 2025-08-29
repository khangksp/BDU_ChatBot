FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies, INCLUDING OCR TOOLS
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
        git \
        curl \
        tesseract-ocr \
        tesseract-ocr-vie \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# SỬA Ở ĐÂY: COPY requirements.txt từ gốc của context
COPY requirements.txt .

# Install Python dependencies in a stable order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers==2.7.0
RUN pip install --no-cache-dir -r requirements.txt

# SỬA Ở ĐÂY: COPY tất cả file từ context (thư mục backend) vào WORKDIR
COPY . .

# Create necessary directories
RUN mkdir -p logs media staticfiles

# Expose port 8000
EXPOSE 3019

# Create the entrypoint script
# Script này đã rất tốt, không cần sửa
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== BDU ChatBot Docker Setup (Final Version) ==="\n\
echo "⏳ Waiting for PostgreSQL..."\n\
while ! pg_isready -h postgres -p 5432 -U postgres; do sleep 1; done\n\
echo "✅ PostgreSQL is ready"\n\
echo "📊 Running database migrations..."\n\
python manage.py migrate --noinput\n\
echo "📦 Collecting static files..."\n\
python manage.py collectstatic --noinput\n\
echo "👤 Setting up admin user..."\n\
python manage.py shell -c "\n\
from django.contrib.auth import get_user_model; User = get_user_model();\n\
if not User.objects.filter(username='\''admin'\'').exists():\n\
    User.objects.create_superuser('\''admin'\'', '\''admin@bdu.edu.vn'\'', '\''admin123456'\'', faculty_code='\''ADMIN001'\'', full_name='\''Administrator'\'');\n\
    print('\''✅ Admin user created'\'');\n\
else:\n\
    print('\''ℹ️ Admin user already exists'\'');\n\
" 2>/dev/null || echo "⚠️ Admin user creation skipped"\n\
echo "🚀 Starting Django server..."\n\
python manage.py runserver 0.0.0.0:3019\n' > /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the default command
CMD ["/entrypoint.sh"]