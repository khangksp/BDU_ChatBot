# test_ocr.py
import os
import re
import sys
from typing import List, Dict, Optional

# --- SECTION 1: DYNAMIC PATH CONFIGURATION & IMPORTS ---
# Import a
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
import docx

# Tự động xác định các đường dẫn gốc dựa trên vị trí của file script này
# Giả sử file này nằm trong D:\Github\BDU_ChatBot\backend\
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Thêm thư mục gốc vào sys.path để có thể import các module Django sau này khi cần
sys.path.append(PROJECT_ROOT)

# Tải các biến môi trường từ file .env nằm cùng cấp với script này
load_dotenv()
print(f"✅ Đã tải file .env từ thư mục: {BACKEND_DIR}")

# Lấy đường dẫn tương đối đã cấu hình trong file .env
tesseract_relative_path = os.getenv("TESSERACT_PATH_RELATIVE")
poppler_relative_path = os.getenv("POPPLER_PATH_RELATIVE")

# Xây dựng đường dẫn tuyệt đối, linh hoạt để sử dụng
TESSERACT_CMD_PATH = os.path.join(PROJECT_ROOT, tesseract_relative_path) if tesseract_relative_path else None
POPPLER_PATH_BIN = os.path.join(PROJECT_ROOT, poppler_relative_path) if poppler_relative_path else None

# Kiểm tra và gán Tesseract command
if TESSERACT_CMD_PATH and os.path.exists(TESSERACT_CMD_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
    print(f"✅ Tesseract được cấu hình tại: {TESSERACT_CMD_PATH}")
else:
    print(f"LỖI: Không tìm thấy Tesseract tại '{TESSERACT_CMD_PATH}'. Vui lòng kiểm tra biến TESSERACT_PATH_RELATIVE trong file .env")
    exit()

# Kiểm tra cấu hình Poppler
if not POPPLER_PATH_BIN or not os.path.exists(POPPLER_PATH_BIN):
    print(f"LỖI: Đường dẫn Poppler không hợp lệ tại '{POPPLER_PATH_BIN}'. Vui lòng kiểm tra biến POPPLER_PATH_RELATIVE trong file .env")
else:
    print(f"✅ Poppler được cấu hình tại: {POPPLER_PATH_BIN}")

# Đường dẫn đến các file văn bản test (tính từ thư mục gốc của project)
PDF_FILE = os.path.join(PROJECT_ROOT, "QD1863_thanh_lap_Hoi_dong_thi_dua_khen_thuong_va_ky_luat_sinh_vien.pdf")
DOCX_FILE = os.path.join(PROJECT_ROOT, "444-KH tu danh gia 5 CTĐT.docx")


# --- SECTION 2: FILE READING FUNCTIONS ---

def extract_text_from_pdf(pdf_path: str) -> Optional[List[Dict]]:
    """Xử lý file PDF với đường dẫn Poppler động."""
    if not POPPLER_PATH_BIN or not os.path.exists(POPPLER_PATH_BIN):
        print("-> Bỏ qua xử lý PDF vì Poppler chưa được cấu hình đúng.")
        return None
    try:
        pages_as_images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH_BIN)
        extracted_data = []
        for i, page_image in enumerate(pages_as_images):
            page_num = i + 1
            print(f"  -> Đang OCR trang {page_num}/{len(pages_as_images)} của file PDF...")
            text = pytesseract.image_to_string(page_image, lang='vie')
            extracted_data.append({"page": page_num, "text": text})
        return extracted_data
    except Exception as e:
        print(f"LỖI khi xử lý PDF: {e}")
        return None

def extract_text_from_docx(docx_path: str) -> Optional[List[Dict]]:
    """Trích xuất văn bản từ file .docx."""
    try:
        print(f"  -> Đang đọc file DOCX...")
        doc = docx.Document(docx_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)
        return [{"page": 1, "text": full_text}]
    except Exception as e:
        print(f"LỖI khi xử lý DOCX: {e}")
        return None

def read_document(file_path: str) -> Optional[List[Dict]]:
    """Hàm điều phối: Tự động nhận diện định dạng file và gọi hàm xử lý phù hợp."""
    if not os.path.exists(file_path):
        print(f"LỖI: File không tồn tại tại '{file_path}'")
        return None
    print(f"\n🚀 Bắt đầu xử lý file: {os.path.basename(file_path)}")
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"LỖI: Định dạng file '{file_extension}' chưa được hỗ trợ.")
        return None


# --- SECTION 3: CITATION FUNCTION ---

def find_precise_quote(pages_data: List[Dict], search_phrase: str) -> Optional[Dict]:
    """Hàm trích dẫn: Chỉ lấy chính xác dòng chứa từ khóa."""
    print(f"🔍 Đang tìm trích dẫn chính xác cho: '{search_phrase}'")
    for page_info in pages_data:
        lines = page_info["text"].split('\n')
        for line in lines:
            if search_phrase.lower() in line.lower():
                precise_quote = line.strip()
                if len(precise_quote) > 3:
                    print(f"  -> ✅ Tìm thấy tại trang {page_info['page']}!")
                    return {"quote": precise_quote, "location_text": f"Trang {page_info['page']}"}
    print("  -> ❌ Không tìm thấy trích dẫn phù hợp.")
    return None


# --- SECTION 4: MAIN EXECUTION ---

if __name__ == "__main__":
    print("\n--- CHƯƠNG TRÌNH TEST OCR (PHIÊN BẢN TÍCH HỢP) ---")
    
    # Test file DOCX
    docx_data = read_document(DOCX_FILE)
    if docx_data:
        citation_docx = find_precise_quote(docx_data, "Phân công thực hiện nhiệm vụ")
        print("\n--- KẾT QUẢ TRÍCH DẪN (DOCX) ---")
        if citation_docx:
            print(f"📍 Vị trí: {citation_docx['location_text']}")
            print(f"💬 Trích dẫn: \"{citation_docx['quote']}\"")
        else:
            print("Không tìm thấy trích dẫn.")
            
    # Test file PDF
    pdf_data = read_document(PDF_FILE)
    if pdf_data:
        citation_pdf = find_precise_quote(pdf_data, "Điều 4")
        print("\n--- KẾT QUẢ TRÍCH DẪN (PDF) ---")
        if citation_pdf:
            print(f"📍 Vị trí: {citation_pdf['location_text']}")
            print(f"💬 Trích dẫn: \"{citation_pdf['quote']}\"")
        else:
            print("Không tìm thấy trích dẫn.")

    print("\n--- TEST KẾT THÚC ---")