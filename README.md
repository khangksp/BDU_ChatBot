Chatbot BDU - Đại học Bình Dương
📋 Tổng quan hệ thống
Đây là một hệ thống Trợ lý AI Hybrid, được thiết kế chuyên biệt cho giảng viên và sinh viên Đại học Bình Dương, kết hợp nhiều công nghệ tiên tiến để mang lại trải nghiệm hội thoại thông minh và cá nhân hóa.

✨ Kiến trúc cốt lõi
Hybrid Retrieval-Augmented Generation (RAG): Hệ thống không chỉ dựa vào AI tạo sinh. Nó sử dụng mô hình SBERT và FAISS để tìm kiếm thông tin chính xác từ cơ sở tri thức (lấy từ Google Drive & Database), sau đó sử dụng Google Gemini để tạo ra câu trả lời tự nhiên, mạch lạc và đúng ngữ cảnh.

API-First Priority: Đối với các thông tin cá nhân và nhạy cảm (như lịch giảng dạy), hệ thống ưu tiên gọi API đến hệ thống của trường thông qua JWT Token, đảm bảo dữ liệu luôn chính xác và theo thời gian thực.

Hỏi-Đáp theo Ngữ cảnh Tài liệu (Document QA): Đây là một tính năng đột phá, cho phép người dùng tải lên các file tài liệu (PDF, DOCX). Hệ thống sẽ sử dụng Tesseract OCR để trích xuất văn bản và ưu tiên ngữ cảnh từ tài liệu này để trả lời câu hỏi, giúp giải quyết các vấn đề tức thời và chuyên biệt.

Multi-Modal Interaction: Hỗ trợ tương tác đa phương thức, cho phép người dùng nhập liệu bằng văn bản hoặc giọng nói (sử dụng Whisper để nhận diện) và nhận lại phản hồi bằng văn bản kèm âm thanh (sử dụng gTTS).

Contextual NLU: Sử dụng PhoBERT để phân loại ý định (Intent) của người dùng một cách chính xác, kết hợp với các thuật toán Re-ranking để sắp xếp và lựa chọn câu trả lời phù hợp nhất.

🛠️ Công nghệ sử dụng
Lĩnh vực	Công nghệ / Thư viện	Mục đích	File chính
Backend	Django, Django Rest Framework	Xây dựng API server, quản lý database.	backend/, ai_models/
AI - Tạo sinh (NLG)	Google Gemini	Tạo câu trả lời, tóm tắt, diễn giải.	ai_models/gemini_service.py
AI - Hiểu ngôn ngữ (NLU)	PhoBERT, SBERT, FAISS	Phân loại ý định, tìm kiếm ngữ nghĩa.	ai_models/phobert_service.py, ai_models/services.py
AI - Giọng nói	faster-whisper, gTTS	Chuyển giọng nói thành văn bản và ngược lại.	ai_models/speech_service.py
AI - Xử lý Tài liệu (OCR)	Tesseract, pdf2image, python-docx	Trích xuất văn bản từ file PDF, DOCX.	ai_models/ocr_service.py
Database & Cache	SQLite, PostgreSQL, Pandas	Lưu trữ dữ liệu, cache thông tin.	db.sqlite3, google_drive_service.py
API Ngoài	JWT, Requests	Giao tiếp với hệ thống của trường.	ai_models/external_api_service.py

Xuất sang Trang tính
🚀 Cài đặt
1. Backend Setup
Bash

# Clone a project
git clone https://github.com/khangksp/BDU_ChatBot.git
cd BDU_ChatBot/backend/

# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Tạo file .env ở thư mục root backend/ và thêm các biến môi trường
# Xem mục "Biến môi trường (.env)" bên dưới

# Setup database
python manage.py migrate

# (Tùy chọn) Chạy lệnh để build FAISS index lần đầu
python manage.py rebuild_faiss_index

# Chạy server
python manage.py runserver
2. Frontend Setup
Bash

# Đi tới thư mục frontend
cd ../frontend/ # Giả sử thư mục frontend ngang cấp với backend

# Cài đặt packages
npm install

# Chạy app
npm start
3. Yêu cầu Hệ thống cho OCR (Quan trọng)
Để tính năng xử lý tài liệu (PDF, DOCX) hoạt động, bạn cần cài đặt các phần mềm sau trên hệ thống của mình:

Tesseract OCR:

Truy cập trang chủ Tesseract để xem hướng dẫn cài đặt.

Trên Windows: Tải bộ cài đặt từ UB-Mannheim. Trong quá trình cài đặt, hãy chọn thêm gói ngôn ngữ Tiếng Việt (Vietnamese).

Trên macOS: brew install tesseract tesseract-lang

Trên Linux (Ubuntu): sudo apt install tesseract-ocr tesseract-ocr-vie

Poppler (dành cho xử lý PDF):

Trên Windows: Tải file binary mới nhất từ trang release của Poppler, giải nén và trỏ đường dẫn đến thư mục bin của nó.

Trên macOS: brew install poppler

Trên Linux (Ubuntu): sudo apt install poppler-utils

⚙️ Biến môi trường (.env)
Tạo một file tên là .env trong thư mục backend/ với nội dung sau:

Đoạn mã

# Chìa khóa bí mật của Django, có thể tạo ngẫu nhiên
SECRET_KEY='your-django-secret-key'

# API Key cho Google Gemini (có thể có nhiều key để xoay vòng)
GEMINI_API_KEY='your-gemini-api-key-1'
GEMINI_API_KEY2='your-gemini-api-key-2'

# Chìa khóa bí mật để giải mã JWT token từ hệ thống của trường
JWT_SECRET_KEY='your-jwt-secret-key'

# (Tùy chọn) URL của hệ thống API trường
SCHOOL_API_BASE_URL='https://cds.bdu.edu.vn'

# === Đường dẫn cho OCR (QUAN TRỌNG) ===
# Đường dẫn tuyệt đối hoặc tương đối đến file thực thi Tesseract
# Ví dụ trên Windows: TESSERACT_CMD_PATH="C:/Program Files/Tesseract-OCR/tesseract.exe"
# Ví dụ trên Linux/macOS: TESSERACT_CMD_PATH="/usr/local/bin/tesseract"
TESSERACT_CMD_PATH='path-to-your-tesseract-executable'

# Đường dẫn tuyệt đối hoặc tương đối đến thư mục bin của Poppler (chỉ cần cho Windows)
# Ví dụ trên Windows: POPPLER_PATH_BIN="C:/path/to/poppler-23.11.0/Library/bin"
POPPLER_PATH_BIN='path-to-your-poppler-bin-directory'
🧪 Tài khoản Test
Đây là các tài khoản mẫu bạn có thể dùng để đăng nhập và kiểm tra các tính năng.

Username	Password	Vai trò	Ghi chú
ADMIN001	admin123456	Quản trị	Có quyền cao nhất
TK_CNTT_001	khangksp456	Giảng viên	Khoa Công nghệ thông tin
GV_CNTT_001	khangksp789	Giảng viên	Khoa Công nghệ thông tin
TEST	Nothingthere456	Test	Tài khoản test cơ bản

Xuất sang Trang tính
🔬 Kịch bản Test
1. Test kiến thức chung (Hybrid RAG)
Hành động: Không cần đăng nhập.

Hỏi: "tạp chí khoa học của trường có những yêu cầu gì?"

Kết quả mong đợi: Bot trả lời các thông tin về việc gửi bài, thể lệ của tạp chí khoa học, lấy từ cơ sở tri thức. Dưới câu trả lời sẽ có "Nguồn tham khảo".

2. Test cá nhân hóa (Personalization)
Hành động: Đăng nhập bằng tài khoản GV_CNTT_001.

Hỏi: "Tôi cần chuẩn bị gì cho việc nghiên cứu khoa học?"

Kết quả mong đợi: Bot đưa ra câu trả lời có xu hướng tập trung vào lĩnh vực CNTT. Câu chào sẽ là "Dạ thầy/cô Khoa,..." (nếu full_name là "Nguyễn Văn Khoa").

3. Test API-First (Thông tin cá nhân)
Hành động: Đăng nhập bằng tài khoản TK_CNTT_001.

Hỏi: "Lịch dạy của tôi hôm nay là gì?"

Kết quả mong đợi: Bot trả về chính xác lịch dạy của giảng viên có mã TK_CNTT_001. Dưới câu trả lời sẽ có tag "🌐 Thông tin cá nhân từ hệ thống". Đây là minh chứng cho thấy luồng gọi API qua JWT Token đã thành công.

4. Test Tương tác giọng nói (STT + TTS)
Hành động: Bật "Chế độ giọng nói" 🎤🔊.

Click vào icon micro và nói: "Học phí ngành công nghệ thông tin là bao nhiêu?".

Kết quả mong đợi:

Văn bản "Học phí ngành công nghệ thông tin là bao nhiêu?" xuất hiện trong ô chat.

Bot gửi câu hỏi đi và xử lý.

Bot trả lời bằng văn bản, đồng thời tự động phát câu trả lời đó bằng giọng nói.

5. Test Hỏi-Đáp Tài liệu (OCR)
Hành động: Không cần đăng nhập. Kéo thả một file PDF (ví dụ: một thông báo của trường) vào ô chat và đặt câu hỏi: "ai là chủ tịch hội đồng thi đua?".

Kết quả mong đợi: Bot đọc nội dung file PDF, tìm và trả lời chính xác tên của người giữ chức vụ Chủ tịch trong Hội đồng Thi đua Khen thưởng dựa trên nội dung của file. Đây là minh chứng cho luồng xử lý OCR đã thành công.

📡 API Endpoints
Đây là danh sách các API quan trọng mà frontend cần gọi. Logic được xử lý chính trong ai_models/views.py và authentication/views.py.

1. Chat & Tương tác chính
Method	Endpoint	Quyền	Chức năng
POST	/api/chat/	Public	Endpoint chính để gửi tin nhắn và nhận phản hồi. Hỗ trợ multipart/form-data để đính kèm file document.
POST	/api/speech-to-text/	Public	Gửi file audio, nhận lại văn bản.
GET	/api/health/	Public	Kiểm tra trạng thái hoạt động của hệ thống.

Xuất sang Trang tính
2. Quản lý phiên Chat (Yêu cầu đăng nhập)
Method	Endpoint	Quyền	Chức năng
GET	/api/chat-sessions/	Authenticated	Lấy danh sách các đoạn chat gần đây.
POST	/api/chat-sessions/	Authenticated	Tạo một đoạn chat mới.
GET	/api/chat-sessions/<id>/	Authenticated	Tải toàn bộ tin nhắn của một đoạn chat.
PATCH	/api/chat-sessions/<id>/	Authenticated	Đổi tên một đoạn chat.
DELETE	/api/chat-sessions/<id>/	Authentated	Xóa một đoạn chat và toàn bộ tin nhắn.

Xuất sang Trang tính
3. Xác thực & Cá nhân hóa
Method	Endpoint	Quyền	Chức năng
POST	/api/auth/login/	Public	Đăng nhập để nhận token.
POST	/api/auth/logout/	Authenticated	Đăng xuất và vô hiệu hóa token.
GET	/api/auth/profile/	Authenticated	Lấy thông tin chi tiết của người dùng hiện tại.
GET	/api/personalized-context/	Authenticated	Lấy thông tin cá nhân hóa (tên, khoa, gợi ý).
POST	/api/auth/chatbot/preferences/	Authenticated	Cập nhật/lưu cài đặt chatbot (ghi nhớ, phong cách).
POST	/api/auth/password/change/	Authenticated	Đổi mật khẩu khi đã đăng nhập.

Xuất sang Trang tính
4. Quản trị & Tiện ích (Dành cho Admin)
Method	Endpoint	Quyền	Chức năng
GET	/api/qa/status/	Admin	Lấy trạng thái của hệ thống Quản lý QA.
POST	/api/knowledge/upload-csv/	Admin	Upload file CSV để thêm vào cơ sở tri thức.
GET	/api/knowledge/categories/	Admin	Lấy danh sách các danh mục kiến thức.

Xuất sang Trang tính
🏆 Tính năng nổi bật
✅ Hệ thống xác thực & phân quyền qua JWT Token.
✅ Kiến trúc API-First ưu tiên dữ liệu cá nhân từ hệ thống trường.
✅ Hệ thống RAG lai (Hybrid RAG) kết hợp tìm kiếm ngữ nghĩa (SBERT, FAISS) và tái xếp hạng (Re-ranking) để tăng độ chính xác.
✅ Tích hợp AI tạo sinh Google Gemini để tạo ra các câu trả lời tự nhiên, linh hoạt.
✅ Phân loại ý định theo ngữ cảnh bằng PhoBERT.
✅ Hỏi-Đáp trực tiếp trên tài liệu (PDF, DOCX) nhờ tích hợp Tesseract OCR.
✅ Xử lý thông minh văn bản OCR "bẩn", đặc biệt là các cấu trúc bảng.
✅ Tương tác giọng nói hai chiều (STT với Whisper & TTS với gTTS).
✅ Cá nhân hóa sâu sắc cho phép người dùng "dạy" chatbot với các chỉ dẫn riêng (User Memory Prompt).
✅ Quản lý phiên trò chuyện đầy đủ (tạo, xem lại, đổi tên, xóa).
✅ Cơ chế tự phục hồi (xoay vòng API key, cache dữ liệu) để tăng tính ổn định.
✅ Giao diện quản trị (Admin) để quản lý câu hỏi và đồng bộ dữ liệu.