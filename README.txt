# 🤖 Chatbot Đại học Bình Dương - Hướng dẫn Setup & Test

## 📋 Tổng quan hệ thống
- **Backend**: Django + PhoBERT + Whisper + FAISS
- **Frontend**: React + Speech-to-Text
- **AI Features**: Personalization, RAG system, Voice chat
- **Authentication**: Token-based với personalized context

---

## 🚀 Cài đặt nhanh

### 1. Backend Setup
```bash
# Clone và setup environment
cd backend/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database setup
python manage.py makemigrations
python manage.py migrate

# Load initial data (FAQ, knowledge base)
python manage.py loaddata initial_data.json  # Nếu có

# Run server
python manage.py runserver
```

### 2. Frontend Setup
```bash
cd frontend/
npm install
npm start
```

---

## 👥 Tài khoản test

### Cách tạo tài khoản (Django Shell)
```bash
# Vào Django shell
python manage.py shell
```

**Copy-paste các block code sau:**

### 🔧 BLOCK 1: Tạo tài khoản TEST cơ bản
```python
from authentication.models import Faculty
from django.contrib.auth.hashers import make_password

Faculty.objects.create(
    faculty_code='TEST',
    username='TEST',
    full_name='Tài Khoản Test',
    email='test@bdu.edu.vn',
    department='test',
    position='Test User',
    password=make_password('123456'),
    is_active=True,
    is_active_faculty=True
)
print("✅ Tạo user TEST/123456")
```

### 👑 BLOCK 2: Tạo tài khoản ADMIN
```python
Faculty.objects.create(
    faculty_code='ADMIN001',
    username='ADMIN001',
    full_name='Nguyễn Văn Admin',
    email='admin@bdu.edu.vn',
    department='ban_giam_hieu',
    position='Quản trị viên',
    password=make_password('admin123456'),
    is_active=True,
    is_active_faculty=True,
    is_staff=True,
    is_superuser=True
)
print("✅ Tạo admin ADMIN001/admin123456")
```

### 🎓 BLOCK 3: Tạo tài khoản giảng viên chuyên ngành
```python
# Giảng viên CNTT
Faculty.objects.create(
    faculty_code='GV_CNTT_001',
    username='GV_CNTT_001',
    full_name='Nguyễn Văn Khoa',
    email='khoa.nv@bdu.edu.vn',
    department='cntt',
    position='Giảng viên',
    password=make_password('gv001@2024'),
    is_active=True,
    is_active_faculty=True
)

# Giảng viên Dược
Faculty.objects.create(
    faculty_code='GV_DUOC_001',
    username='GV_DUOC_001',
    full_name='Trần Thị Dược',
    email='duoc.tt@bdu.edu.vn',
    department='duoc',
    position='Giảng viên',
    password=make_password('gv002@2024'),
    is_active=True,
    is_active_faculty=True
)

# Giảng viên Điện tử
Faculty.objects.create(
    faculty_code='GV_DIEN_001',
    username='GV_DIEN_001',
    full_name='Lê Văn Điện',
    email='dien.lv@bdu.edu.vn',
    department='dien_tu',
    position='Giảng viên',
    password=make_password('gv003@2024'),
    is_active=True,
    is_active_faculty=True
)

print("✅ Tạo các giảng viên chuyên ngành")
```

### 📊 BLOCK 4: Kiểm tra tài khoản đã tạo
```python
# Liệt kê tất cả tài khoản
users = Faculty.objects.all()
for user in users:
    print(f"👤 {user.faculty_code} | {user.full_name} | {user.department}")
```

---

## 🧪 Test Scenarios

### 1. **Basic Authentication Test**
| Tài khoản | Mật khẩu | Vai trò |
|-----------|----------|---------|
| `TEST` | `123456` | User cơ bản |
| `ADMIN001` | `admin123456` | Quản trị viên |
| `GV_CNTT_001` | `gv001@2024` | Giảng viên CNTT |
| `GV_DUOC_001` | `gv002@2024` | Giảng viên Dược |
| `GV_DIEN_001` | `gv003@2024` | Giảng viên Điện tử |

### 2. **Personalization Test Flow**
1. **Login**: Dùng `GV_CNTT_001/gv001@2024`
2. **Check Settings**: Nút "🎯 Settings" xuất hiện
3. **Open Modal**: Config personalization:
   - Response Style: `Technical`
   - Focus Areas: `Công nghệ thông tin`, `Lập trình`
   - Department Priority: ✅ Enabled
4. **Save** và test chat: "Chương trình CNTT như thế nào?"
5. **Expected**: Response có personalization với technical style

### 3. **Voice Test (Nếu có Whisper)**
1. **Check Voice Icon**: 🎤 hiện khi speech supported
2. **Record**: Click micro, nói "Học phí CNTT bao nhiêu?"
3. **Transcription**: Text xuất hiện trong input
4. **Send**: Chat với personalized response

### 4. **Department Context Test**
```python
# Test các câu hỏi khác nhau với user khác nhau
test_cases = [
    ("GV_CNTT_001", "Môn học lập trình có gì?"),      # → Tech focus
    ("GV_DUOC_001", "Thực hành dược như thế nào?"),   # → Medical focus  
    ("GV_DIEN_001", "Thiết bị điện tử gì?"),          # → Electronics focus
    ("ADMIN001", "Báo cáo tuyển sinh 2024"),          # → Admin context
]
```

---

## 🛠️ Debugging

### Backend Issues
```bash
# Check server logs
python manage.py runserver --verbosity=2

# Test API endpoints
curl http://127.0.0.1:8000/api/health/
curl http://127.0.0.1:8000/api/auth/status/
```

### Database Issues
```bash
# Reset database
python manage.py flush
python manage.py migrate
# Tạo lại users với code trên
```

### Frontend Issues
```bash
# Check console for errors
# Verify API_BASE_URL in config.js
# Check browser network tab for failed requests
```

---

## 🔍 API Endpoints

### Authentication
- `POST /api/auth/login/` - Đăng nhập
- `GET /api/auth/status/` - Kiểm tra trạng thái
- `POST /api/auth/logout/` - Đăng xuất

### Personalization  
- `GET /api/auth/chatbot/preferences/` - Lấy preferences
- `POST /api/auth/chatbot/preferences/update/` - Cập nhật
- `GET /api/auth/chatbot/suggestions/` - Lấy suggestions

### Chat
- `POST /api/chat/` - Chat thường
- `POST /api/personalized-chat/` - Chat với personalization
- `POST /api/speech-to-text/` - Voice input

---

## 🎯 Production Checklist
- [ ] Environment variables setup
- [ ] Database optimization
- [ ] Static files serving
- [ ] CORS configuration
- [ ] SSL/HTTPS setup
- [ ] Error logging
- [ ] Performance monitoring
- [ ] Backup strategy

---

## 📞 Support
- **Email**: support@bdu.edu.vn
- **Documentation**: [Link to detailed docs]
- **Issues**: [Link to issue tracker]

---

## 🏆 Features Implemented
✅ **Authentication System**
✅ **Personalized Responses** 
✅ **Voice Input (Whisper)**
✅ **RAG with FAISS**
✅ **Department Context**
✅ **Response Styling**
✅ **Session Management**
✅ **Feedback System**
✅ **Real-time Chat**
✅ **Mobile Responsive**

---

**Made with ❤️ for Đại học Bình Dương**