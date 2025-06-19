import requests
import json
import time
from datetime import datetime

# Configuration - SỬA THÔNG TIN NÀY
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "faculty_code": "TEST",  # Thay bằng mã giảng viên của bạn
    "password": "Nothingthere456"  # Thay bằng password của bạn
}

class PersonalizationTester:
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.session_id = None
        self.test_results = []
        
    def log(self, message, data=None):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        print("-" * 50)
        
    def test_login(self):
        """Test 1: Login và kiểm tra auto-setup"""
        self.log("🔐 TEST 1: Login và kiểm tra auto-setup preferences")
        
        try:
            response = self.session.post(
                f"{BASE_URL}/api/auth/login/",
                json=TEST_USER
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data['data']['token']
                self.session_id = data['data']['session_id']
                
                # Set auth header
                self.session.headers.update({
                    'Authorization': f'Token {self.token}'
                })
                
                self.log("✅ Login thành công", {
                    "user": data['data']['user'],
                    "chatbot_setup": data['data'].get('chatbot_setup', {})
                })
                
                self.test_results.append({
                    "test": "Login",
                    "status": "PASS",
                    "chatbot_setup": data['data'].get('chatbot_setup', {})
                })
                return True
            else:
                self.log("❌ Login thất bại", response.json())
                self.test_results.append({
                    "test": "Login", 
                    "status": "FAIL",
                    "error": response.json()
                })
                return False
        except requests.exceptions.ConnectionError:
            self.log("❌ Không thể kết nối đến server. Hãy chắc chắn server đang chạy!")
            self.test_results.append({
                "test": "Login",
                "status": "FAIL", 
                "error": "Server not running"
            })
            return False
            
    def test_get_preferences(self):
        """Test 2: Lấy preferences hiện tại"""
        self.log("📋 TEST 2: Lấy chatbot preferences")
        
        response = self.session.get(f"{BASE_URL}/api/auth/chatbot/preferences/")
        
        if response.status_code == 200:
            data = response.json()
            self.log("✅ Preferences hiện tại:", data['data'])
            
            self.test_results.append({
                "test": "Get Preferences",
                "status": "PASS",
                "preferences": data['data']['preferences']
            })
            return data['data']
        else:
            self.log("❌ Không lấy được preferences", response.json())
            self.test_results.append({
                "test": "Get Preferences",
                "status": "FAIL"
            })
            return None
            
    def test_update_preferences(self, memory_prompt, department_priority):
        """Test 3: Update preferences"""
        self.log(f"🔧 TEST 3: Update preferences")
        self.log(f"- Memory prompt: {memory_prompt[:50]}...")
        self.log(f"- Department priority: {department_priority}")
        
        update_data = {
            "preferences": {
                "user_memory_prompt": memory_prompt,
                "response_style": "professional",
                "department_priority": department_priority
            }
        }
        
        response = self.session.post(
            f"{BASE_URL}/api/auth/chatbot/preferences/update/",
            json=update_data
        )
        
        if response.status_code == 200:
            data = response.json()
            self.log("✅ Update thành công", data['data']['changes'])
            
            self.test_results.append({
                "test": "Update Preferences",
                "status": "PASS",
                "changes": data['data']['changes']
            })
            return True
        else:
            self.log("❌ Update thất bại", response.json())
            self.test_results.append({
                "test": "Update Preferences",
                "status": "FAIL"
            })
            return False
            
    def test_chat_with_context(self, message, test_name):
        """Test 4: Chat và kiểm tra personalization"""
        self.log(f"💬 TEST: {test_name}")
        self.log(f"Message: {message}")
        
        chat_data = {
            "message": message,
            "session_id": self.session_id  # QUAN TRỌNG: phải có session_id
        }
        
        response = self.session.post(
            f"{BASE_URL}/api/chat/",
            json=chat_data
        )
        
        if response.status_code == 200:
            data = response.json()
            self.log("✅ Chat response:", {
                "response": data['response'],
                "personalized": data.get('personalized', False),
                "user_context": data.get('user_context', {})
            })
            
            self.test_results.append({
                "test": test_name,
                "status": "PASS",
                "personalized": data.get('personalized', False),
                "response": data['response'][:100] + "..."
            })
            return data
        else:
            self.log("❌ Chat thất bại", response.json())
            self.test_results.append({
                "test": test_name,
                "status": "FAIL"
            })
            return None
            
    def test_system_prompt(self):
        """Test 5: Kiểm tra system prompt"""
        self.log("🤖 TEST 5: Kiểm tra personalized system prompt")
        
        response = self.session.get(f"{BASE_URL}/api/auth/chatbot/system-prompt/")
        
        if response.status_code == 200:
            data = response.json()
            prompt = data['data']['system_prompt']
            
            # Kiểm tra các thành phần trong prompt
            checks = {
                "has_faculty_code": TEST_USER['faculty_code'] in prompt,
                "has_user_memory": "THÔNG TIN CÁ NHÂN:" in prompt,
                "has_department_info": "CHUYÊN MÔN NGÀNH" in prompt,
                "department_priority_enabled": data['data']['department_info']['department_priority_enabled']
            }
            
            self.log("✅ System prompt checks:", checks)
            self.log("Full prompt preview:", prompt[:500] + "...")
            
            self.test_results.append({
                "test": "System Prompt",
                "status": "PASS" if all(checks.values()) else "PARTIAL",
                "checks": checks
            })
            return checks
        else:
            self.log("❌ Không lấy được system prompt")
            self.test_results.append({
                "test": "System Prompt",
                "status": "FAIL"
            })
            return None
            
    def test_department_priority_toggle(self):
        """Test 6: Test bật/tắt department priority"""
        self.log("🔄 TEST 6: Test toggle department priority")
        
        # Lấy preferences hiện tại
        current = self.test_get_preferences()
        if not current:
            return
            
        current_dept_priority = current['preferences'].get('department_priority', True)
        
        # Test với department_priority = True
        self.log("📍 Test với department_priority = TRUE")
        self.test_update_preferences(
            memory_prompt="Tôi thích câu trả lời chi tiết với ví dụ cụ thể",
            department_priority=True
        )
        time.sleep(1)
        
        response1 = self.test_chat_with_context(
            "Cho tôi biết về chương trình đào tạo",
            "Chat với department_priority=True"
        )
        
        # Test với department_priority = False
        self.log("📍 Test với department_priority = FALSE")
        self.test_update_preferences(
            memory_prompt="Tôi thích câu trả lời chi tiết với ví dụ cụ thể",
            department_priority=False
        )
        time.sleep(1)
        
        response2 = self.test_chat_with_context(
            "Cho tôi biết về chương trình đào tạo",
            "Chat với department_priority=False"
        )
        
        # So sánh kết quả
        if response1 and response2:
            comparison = {
                "response1_length": len(response1['response']),
                "response2_length": len(response2['response']),
                "both_personalized": response1.get('personalized') and response2.get('personalized'),
                "responses_different": response1['response'] != response2['response']
            }
            
            self.log("📊 So sánh kết quả:", comparison)
            self.test_results.append({
                "test": "Department Priority Toggle",
                "status": "PASS" if comparison['responses_different'] else "FAIL",
                "comparison": comparison
            })
            
    def test_memory_prompt_effect(self):
        """Test 7: Test memory prompt effect"""
        self.log("🧠 TEST 7: Test user memory prompt effect")
        
        # Test 1: Với memory prompt cụ thể
        self.log("📝 Test với memory prompt CHI TIẾT")
        self.test_update_preferences(
            memory_prompt="Tôi là giảng viên CNTT, chuyên về AI và Machine Learning. Tôi thích câu trả lời có code Python và ví dụ thực tế. Tôi đang nghiên cứu về NLP và chatbot.",
            department_priority=True
        )
        time.sleep(1)
        
        response1 = self.test_chat_with_context(
            "Làm sao để cải thiện chatbot?",
            "Chat với memory prompt chi tiết"
        )
        
        # Test 2: Với memory prompt rỗng
        self.log("📝 Test với memory prompt RỖNG")
        self.test_update_preferences(
            memory_prompt="",
            department_priority=True
        )
        time.sleep(1)
        
        response2 = self.test_chat_with_context(
            "Làm sao để cải thiện chatbot?",
            "Chat với memory prompt rỗng"
        )
        
        # So sánh
        if response1 and response2:
            self.log("📊 Kiểm tra sự khác biệt trong response")
            has_python = "Python" in response1['response'] or "python" in response1['response']
            has_code = "```" in response1['response'] or "code" in response1['response']
            
            self.test_results.append({
                "test": "Memory Prompt Effect",
                "status": "PASS" if has_python or has_code else "PARTIAL",
                "has_python_mention": has_python,
                "has_code_example": has_code
            })
            
    def test_conversation_memory(self):
        """Test 8: Test conversation memory"""
        self.log("💭 TEST 8: Test conversation memory")
        
        # Câu hỏi đầu tiên
        response1 = self.test_chat_with_context(
            "Tôi muốn tìm hiểu về ngân hàng đề thi",
            "Memory Test - Câu 1"
        )
        
        time.sleep(1)
        
        # Câu hỏi tiếp theo (follow-up)
        response2 = self.test_chat_with_context(
            "Còn thời hạn nộp thì sao?",
            "Memory Test - Câu 2 (follow-up)"
        )
        
        # Kiểm tra xem có nhớ context không
        if response2:
            has_context = (
                "ngân hàng" in response2['response'].lower() or
                "đề thi" in response2['response'].lower() or
                "nộp" in response2['response'].lower()
            )
            
            self.test_results.append({
                "test": "Conversation Memory",
                "status": "PASS" if has_context else "FAIL",
                "has_context_reference": has_context
            })
            
    def generate_report(self):
        """Tạo báo cáo test"""
        self.log("\n📊 ===== BÁO CÁO TEST PERSONALIZATION =====")
        
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        partial = sum(1 for r in self.test_results if r['status'] == 'PARTIAL')
        
        print(f"\n📈 Tổng kết:")
        print(f"- Tổng số test: {total_tests}")
        print(f"- PASS: {passed} ✅")
        print(f"- FAIL: {failed} ❌")
        print(f"- PARTIAL: {partial} ⚠️")
        print(f"- Tỷ lệ thành công: {(passed/total_tests)*100:.1f}%")
        
        print(f"\n📝 Chi tiết từng test:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
            print(f"{i}. {result['test']}: {status_icon} {result['status']}")
            
            # In thêm thông tin chi tiết
            for key, value in result.items():
                if key not in ['test', 'status']:
                    print(f"   - {key}: {value}")
                    
        # Lưu report
        with open('personalization_test_report.json', 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total_tests,
                    'passed': passed,
                    'failed': failed,
                    'partial': partial
                },
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)
            
        print(f"\n💾 Report đã lưu tại: personalization_test_report.json")
        
    def run_all_tests(self):
        """Chạy tất cả các test"""
        print("🚀 BẮT ĐẦU TEST PERSONALIZATION CHATBOT")
        print("=" * 60)
        
        # Test 1: Login
        if not self.test_login():
            print("❌ Không thể login, dừng test")
            print("\n⚠️ KIỂM TRA:")
            print("1. Server đang chạy không? (python manage.py runserver)")
            print("2. Thông tin đăng nhập đúng không?")
            print(f"3. URL: {BASE_URL}")
            return
            
        time.sleep(1)
        
        # Test 2: Get preferences
        self.test_get_preferences()
        time.sleep(1)
        
        # Test 3: Update preferences với memory prompt
        self.test_update_preferences(
            memory_prompt="Tôi là giảng viên thích câu trả lời ngắn gọn, súc tích. Tôi quan tâm đến công nghệ và đổi mới trong giáo dục.",
            department_priority=True
        )
        time.sleep(1)
        
        # Test 4: Chat với context
        self.test_chat_with_context(
            "Xin chào, tôi cần hỗ trợ",
            "Chat test với personalization"
        )
        time.sleep(1)
        
        # Test 5: System prompt
        self.test_system_prompt()
        time.sleep(1)
        
        # Test 6: Department priority toggle
        self.test_department_priority_toggle()
        time.sleep(1)
        
        # Test 7: Memory prompt effect
        self.test_memory_prompt_effect()
        time.sleep(1)
        
        # Test 8: Conversation memory
        self.test_conversation_memory()
        
        # Generate report
        self.generate_report()

# Hướng dẫn sử dụng
if __name__ == "__main__":
    print("""
    ⚙️ HƯỚNG DẪN SỬ DỤNG:
    1. Sửa TEST_USER với thông tin đăng nhập của bạn
    2. Đảm bảo server đang chạy ở localhost:8000
    3. Chạy script này
    
    Script sẽ test:
    - Auto-setup khi login
    - Update preferences (memory prompt & department priority)
    - Chat với personalization
    - Toggle department priority
    - Memory prompt effect
    - Conversation memory
    
    ⚠️ LƯU Ý: Cần chạy server trước:
    python manage.py runserver
    """)
    
    # Kiểm tra kết nối trước
    try:
        test_conn = requests.get(f"{BASE_URL}/api/", timeout=2)
        print("✅ Server đang chạy!")
    except:
        print("❌ KHÔNG THỂ KẾT NỐI ĐẾN SERVER!")
        print("Vui lòng chạy: python manage.py runserver")
        print("Rồi chạy lại script này.")
        exit(1)
    
    confirm = input("\nBạn đã sẵn sàng chạy test? (y/n): ")
    if confirm.lower() == 'y':
        tester = PersonalizationTester()
        tester.run_all_tests()
    else:
        print("❌ Đã hủy test")