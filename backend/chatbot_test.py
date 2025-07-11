#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUGGING SCRIPT FOR PERSONALIZED ADDRESSING
Mục đích: Giả lập phiên đăng nhập của giảng viên và truy vết tại sao
cách xưng hô không được cá nhân hóa (thầy/cô).
"""

import os
import sys
import django
import json
import time

# --- Setup Django Environment ---
# Đảm bảo đường dẫn này trỏ đúng đến thư mục chứa file manage.py của bạn
# Ví dụ: /path/to/your/project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()
# ---------------------------------

# Import các thành phần cần thiết
from ai_models.services import HybridChatbotAI
from authentication.models import Faculty

class AddressingDebugger:
    """Công cụ chuyên dụng để debug vấn đề xưng hô"""

    def __init__(self):
        # Khởi tạo Chatbot và tạo một session ID duy nhất cho mỗi lần chạy
        self.chatbot = HybridChatbotAI()
        self.session_id = f"debug_session_{int(time.time())}"
        print("🚀 Addressing Debugger initialized!")
        print(f"🔬 Session ID for this run: {self.session_id}")

    def trace_addressing(self, faculty_code: str, test_query: str):
        """
        Hàm chính để giả lập, thực thi và truy vết.
        """
        print("\n" + "="*80)
        print(f"🕵️  STARTING TRACE for Faculty Code: '{faculty_code}'")
        print(f"❓ Query: '{test_query}'")
        print("="*80)

        # BƯỚC 1: LẤY THÔNG TIN GIẢNG VIÊN TỪ DATABASE
        # ----------------------------------------------------
        try:
            print("\n[DEBUG] BƯỚC 1: Fetching faculty data from database...")
            faculty = Faculty.objects.get(faculty_code=faculty_code)
            print(f"✅ Found: {faculty.full_name} ({faculty.get_gender_display()})")
        except Faculty.DoesNotExist:
            print(f"❌ ERROR: Không tìm thấy giảng viên với mã '{faculty_code}' trong database.")
            print("="*80)
            return
        except Exception as e:
            print(f"❌ ERROR: Lỗi khi truy vấn database: {e}")
            print("="*80)
            return

        # BƯỚC 2: TẠO NGỮ CẢNH NGƯỜI DÙNG (USER CONTEXT)
        # ----------------------------------------------------
        # Đây chính là bước mà app điện thoại làm sau khi đăng nhập thành công.
        print("\n[DEBUG] BƯỚC 2: Generating user context...")
        user_context = faculty.get_chatbot_context()
        print("✅ Generated Context:")
        # In ra context dưới dạng JSON để dễ đọc
        print(json.dumps(user_context, indent=2, ensure_ascii=False))
        print("---")
        print("👉 KIỂM TRA: Trường 'gender' và 'full_name' có chính xác không?")

        # BƯỚC 3: "ĐĂNG NHẬP GIẢ LẬP" - SET CONTEXT CHO CHATBOT
        # ----------------------------------------------------
        # Ta nói cho chatbot biết "Ai đang hỏi?" bằng cách gán context vào session
        try:
            print("\n[DEBUG] BƯỚC 3: Injecting user context into the chatbot's memory...")
            # Đây là hàm quan trọng nhất để giả lập phiên đăng nhập
            self.chatbot.response_generator.set_user_context(self.session_id, user_context)
            print("✅ Context injected successfully for this session.")
        except Exception as e:
             print(f"❌ ERROR: Không thể set user context. Lỗi: {e}")
             print("Đây có thể là lỗi nghiêm trọng trong cấu trúc code. Cần kiểm tra lại các class service.")
             print("="*80)
             return


        # BƯỚC 4: KIỂM TRA LOGIC XƯNG HÔ (BÊN NGOÀI)
        # ----------------------------------------------------
        # Tái tạo lại logic của hàm _get_personal_address để xem nó nên trả về gì
        print("\n[DEBUG] BƯỚC 4: Simulating the expected addressing logic...")
        salutation_map = {'male': 'thầy', 'female': 'cô'}
        salutation = salutation_map.get(user_context.get('gender'), 'thầy/cô')
        name_parts = user_context.get('full_name', '').split()
        name_suffix = name_parts[-1] if name_parts else ''
        expected_address = f"{salutation} {name_suffix}" if name_suffix else salutation
        print(f"✅ Expected Addressing: '{expected_address}'")


        # BƯỚC 5: GỬI CÂU HỎI VÀ NHẬN PHẢN HỒI
        # ----------------------------------------------------
        print("\n[DEBUG] BƯỚC 5: Processing the query with the injected context...")
        start_time = time.time()
        response_data = self.chatbot.process_query(test_query, session_id=self.session_id)
        processing_time = time.time() - start_time
        print(f"✅ Query processed in {processing_time:.3f}s.")
        print("\n" + "-"*35 + " RESPONSE " + "-"*35)
        print(response_data.get('response', '!!! NO RESPONSE TEXT !!!'))
        print("-"*80)


        # BƯỚC 6: PHÂN TÍCH KẾT QUẢ
        # ----------------------------------------------------
        print("\n[ANALYZE] KẾT QUẢ PHÂN TÍCH:")
        final_response_text = response_data.get('response', '')
        
        if expected_address in final_response_text:
             print(f"✅ SUCCESS: Phản hồi chứa cách xưng hô đúng: '{expected_address}'")
        elif "thầy/cô" in final_response_text:
             print(f"❌ FAILURE: Phản hồi vẫn dùng 'thầy/cô' chung chung.")
             print(f"   - Expected: Chứa '{expected_address}'")
             print(f"   - Actual:   Chứa 'thầy/cô'")
             print("   - GỢI Ý: Vấn đề có thể nằm ở các hàm fallback, hàm xử lý lỗi, hoặc các hàm prompt bị hardcode 'thầy/cô' bên trong `gemini_service.py` mà không lấy từ context.")
        else:
             print(f"🟡 WARNING: Không tìm thấy cách xưng hô đúng '{expected_address}' và cũng không thấy 'thầy/cô'.")
             print("   - Cần kiểm tra lại cấu trúc câu trả lời của Gemini.")

        print("\n" + "="*80)
        print("🕵️  TRACE COMPLETE")
        print("="*80)

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Advanced Debugging Script for ChatBot Addressing.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--code',
        required=True,
        help='Mã giảng viên cần test (ví dụ: "GV001", "BDU12345").'
    )
    parser.add_argument(
        '--query',
        required=True,
        help='Câu hỏi test (ví dụ: "tạp chí khoa học là gì?").\nLưu ý: Đặt câu hỏi trong dấu ngoặc kép nếu có khoảng trắng.'
    )

    args = parser.parse_args()

    debugger = AddressingDebugger()
    debugger.trace_addressing(faculty_code=args.code, test_query=args.query)


if __name__ == "__main__":
    main()