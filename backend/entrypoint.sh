#!/bin/bash
set -e

echo "=== BDU ChatBot GV - Docker Setup ==="

# -------------------------------------------------------
# 1. Chờ Database sẵn sàng
# -------------------------------------------------------
echo "⏳ Waiting for Database ($DB_HOST)..."
while ! pg_isready -h "$DB_HOST" -p 5432 -U postgres; do
    sleep 1
done
echo "✅ Database connected!"

# -------------------------------------------------------
# 2. Migrations
# -------------------------------------------------------
echo "📊 Running database migrations..."
python manage.py migrate --noinput

# -------------------------------------------------------
# 3. Collect static files
# -------------------------------------------------------
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput

# -------------------------------------------------------
# 4. Tạo SuperAdmin nếu chưa có
# -------------------------------------------------------
echo "👤 Setting up SuperAdmin user..."
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
try:
    if not User.objects.filter(username='admin').exists():
        print('Creating new admin user...')
        u = User.objects.create_superuser('admin', 'admin@bdu.edu.vn', 'Fira@2024')
        u.faculty_code = 'admin'
        u.full_name = 'Administrator'
        u.is_staff = True
        u.is_superuser = True
        u.save()
        print('✅ Admin user created: admin / Fira@2024')
    else:
        print('ℹ️ Admin user already exists. Checking faculty_code...')
        u = User.objects.get(username='admin')
        if u.faculty_code != 'admin':
            u.faculty_code = 'admin'
            u.save()
            print('🔧 Fixed faculty_code for existing admin')
        else:
            print('✅ Admin user OK, no changes needed.')
except Exception as e:
    print(f'❌ Failed to create/update admin: {e}')
"

# -------------------------------------------------------
# 5. Khởi động Gunicorn
# -------------------------------------------------------
echo "🚀 Starting Gunicorn on 0.0.0.0:3019..."
exec gunicorn backend.wsgi:application \
    --config /app/gunicorn.conf.py
