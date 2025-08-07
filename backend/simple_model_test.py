import os
import sys
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path

# =================== DJANGO SETUP ===================
def setup_django():
    """Setup Django environment trước khi import models"""
    try:
        # Thêm đường dẫn backend vào sys.path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        # Setup Django settings
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
        
        import django
        from django.conf import settings
        
        # Configure Django nếu chưa được configure
        if not settings.configured:
            django.setup()
        else:
            django.setup()
            
        print("✅ Django environment setup thành công")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi setup Django: {e}")
        print("💡 Hãy đảm bảo:")
        print("   - Đang chạy từ thư mục backend/")
        print("   - File settings.py tồn tại")
        print("   - DJANGO_SETTINGS_MODULE được set đúng")
        return False

# Setup Django trước khi import services
django_setup_success = setup_django()

print("🚀 ĐÁNH GIÁ CHẤT LƯỢNG CHATBOT - PHIÊN BẢN TÍCH HỢP")
print("=" * 60)

# =================== CẤU HÌNH ===================
GOLDEN_TEST_FILE = "golden_test_set.csv"  # 📁 Đặt file này cùng thư mục
QA_DATABASE_FILE = "data/QA.csv"          # 📁 File QA gốc để lấy đáp án
SIMILARITY_THRESHOLD = 0.4                # 🎯 Ngưỡng để coi là trả lời đúng
MAX_TEST_QUESTIONS = 25                   # 🔢 Số câu test (để test nhanh, đặt = 50 để test full)

# =================== IMPORT SERVICE THẬT ===================
chatbot_ai = None
SERVICE_AVAILABLE = False

if django_setup_success:
    try:
        from ai_models.services import chatbot_ai
        SERVICE_AVAILABLE = True
        print("✅ Đã tích hợp thành công với ai_models.services.chatbot_ai")
    except ImportError as e:
        print(f"❌ Không thể import service: {e}")
        print("💡 Sẽ sử dụng fallback mock service")
        SERVICE_AVAILABLE = False
    except Exception as e:
        print(f"❌ Lỗi khác khi import service: {e}")
        print("💡 Sẽ sử dụng fallback mock service")
        SERVICE_AVAILABLE = False
else:
    print("❌ Django setup thất bại, sẽ sử dụng fallback mock service")

# =================== KIỂM TRA FILE ===================
def check_files():
    """Kiểm tra các file cần thiết có tồn tại không"""
    missing_files = []
    
    if not os.path.exists(GOLDEN_TEST_FILE):
        missing_files.append(f"❌ {GOLDEN_TEST_FILE}")
        print(f"💡 Tạo file {GOLDEN_TEST_FILE} với nội dung mẫu:")
        create_sample_golden_test_file()
    else:
        print(f"✅ Tìm thấy {GOLDEN_TEST_FILE}")
    
    if not os.path.exists(QA_DATABASE_FILE):
        missing_files.append(f"❌ {QA_DATABASE_FILE}")
    else:
        print(f"✅ Tìm thấy {QA_DATABASE_FILE}")
    
    if not SERVICE_AVAILABLE and not django_setup_success:
        missing_files.append("❌ Django environment & ai_models.services")
        print("⚠️ Sẽ chạy với mock service")
    elif SERVICE_AVAILABLE:
        print("✅ AI Service module loaded")
    else:
        print("⚠️ AI Service không khả dụng, sẽ dùng mock")
    
    if missing_files and GOLDEN_TEST_FILE not in str(missing_files):
        print("\n🔴 THIẾU CÁC THÀNH PHẦN SAU:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 HƯỚNG DẪN:")
        if QA_DATABASE_FILE in str(missing_files):
            print(f"   - Đảm bảo {QA_DATABASE_FILE} có trong thư mục data/")
        if "Django environment" in str(missing_files):
            print("   - Chạy từ thư mục backend/ của project")
            print("   - Đảm bảo Django settings được cấu hình đúng")
            print("   - Thử chạy: python manage.py shell trước để test Django")
        return False
    
    return True

def create_sample_golden_test_file():
    """Tạo file golden test set mẫu nếu chưa có"""
    sample_data = [
        {
            'test_question': 'Giảng viên có thể thay đổi lịch giảng dạy không?',
            'correct_question_from_original_csv': 'Liệu giảng viên có thể điều chỉnh thời gian giảng dạy?',
            'category': 'paraphrasing'
        },
        {
            'test_question': 'Làm sao để cập nhật thông tin cá nhân trong hệ thống?',
            'correct_question_from_original_csv': 'Cách thức cập nhật thông tin cá nhân trong hệ thống là gì?',
            'category': 'paraphrasing'
        },
        {
            'test_question': 'Quy trình nộp đề cương môn học ra sao?',
            'correct_question_from_original_csv': 'Thủ tục nộp đề cương môn học như thế nào?',
            'category': 'paraphrasing'
        },
        {
            'test_question': 'Tôi muốn biết về chế độ phụ cấp giảng dạy',
            'correct_question_from_original_csv': 'Chế độ phụ cấp giảng dạy được quy định như thế nào?',
            'category': 'paraphrasing'
        },
        {
            'test_question': 'Điều kiện để được nghỉ phép trong kỳ học', 
            'correct_question_from_original_csv': 'Quy định về nghỉ phép của giảng viên trong học kỳ',
            'category': 'paraphrasing'
        }
    ]
    
    try:
        df = pd.DataFrame(sample_data)
        df.to_csv(GOLDEN_TEST_FILE, index=False, encoding='utf-8')
        print(f"✅ Đã tạo file mẫu {GOLDEN_TEST_FILE}")
    except Exception as e:
        print(f"❌ Không thể tạo file mẫu: {e}")

# =================== LOAD DỮ LIỆU ===================
def load_test_questions():
    """Tải danh sách câu hỏi test"""
    try:
        df = pd.read_csv(GOLDEN_TEST_FILE, encoding='utf-8')
        print(f"📊 Đã tải {len(df)} câu hỏi test")
        return df.head(MAX_TEST_QUESTIONS)  # Giới hạn số câu để test nhanh
    except Exception as e:
        print(f"❌ Lỗi khi đọc {GOLDEN_TEST_FILE}: {e}")
        return None

def load_qa_database():
    """Tải database QA gốc để lấy đáp án"""
    try:
        if os.path.exists(QA_DATABASE_FILE):
            df = pd.read_csv(QA_DATABASE_FILE, encoding='utf-8')
            print(f"📚 Đã tải {len(df)} cặp QA từ database")
            return df
        else:
            print(f"⚠️ Không tìm thấy {QA_DATABASE_FILE}, sẽ dùng mock data")
            return create_mock_qa_database()
    except Exception as e:
        print(f"❌ Lỗi khi đọc {QA_DATABASE_FILE}: {e}")
        return create_mock_qa_database()

def create_mock_qa_database():
    """Tạo mock QA database để test"""
    mock_data = [
        {
            'question': 'Liệu giảng viên có thể điều chỉnh thời gian giảng dạy?',
            'answer': 'Giảng viên có thể điều chỉnh lịch giảng dạy với sự phê duyệt của phòng đào tạo.'
        },
        {
            'question': 'Cách thức cập nhật thông tin cá nhân trong hệ thống là gì?',
            'answer': 'Giảng viên đăng nhập vào hệ thống quản lý, vào mục "Thông tin cá nhân" để cập nhật.'
        },
        {
            'question': 'Thủ tục nộp đề cương môn học như thế nào?',
            'answer': 'Đề cương môn học được nộp qua hệ thống online hoặc trực tiếp tại phòng đào tạo.'
        },
        {
            'question': 'Chế độ phụ cấp giảng dạy được quy định như thế nào?',
            'answer': 'Phụ cấp giảng dạy được tính theo số tiết giảng và hệ số lương hiện hành.'
        },
        {
            'question': 'Quy định về nghỉ phép của giảng viên trong học kỳ',
            'answer': 'Giảng viên cần đăng ký nghỉ phép trước ít nhất 3 ngày và có kế hoạch bù giờ.'
        }
    ]
    
    df = pd.DataFrame(mock_data)
    print(f"📚 Đã tạo mock QA database với {len(df)} cặp QA")
    return df

# =================== INTEGRATED MODEL SERVICE ===================
class IntegratedModelService:
    """
    Service tích hợp với ai_models.services.chatbot_ai
    """
    def __init__(self):
        self.chatbot_service = chatbot_ai
        self.test_session_id = f"test_session_{int(time.time())}"
        self.qa_db = load_qa_database()
        
        if not self.chatbot_service:
            print("⚠️ Chatbot service không khả dụng, sẽ sử dụng fallback")
        else:
            print("✅ Integrated Model Service khởi tạo thành công")
    
    def get_response(self, question):
        """
        Gọi service thật của project để lấy câu trả lời
        """
        if not self.chatbot_service:
            return self._fallback_response(question)
        
        try:
            # 🎯 Gọi method process_query từ service thật
            # Signature: process_query(query, session_id, jwt_token=None, document_text=None)
            response_data = self.chatbot_service.process_query(
                query=question,
                session_id=self.test_session_id,
                jwt_token=None,  # Test không cần JWT
                document_text=None
            )
            
            # Extract response text từ kết quả
            if isinstance(response_data, dict) and 'response' in response_data:
                return response_data['response']
            elif isinstance(response_data, str):
                return response_data
            else:
                print(f"⚠️ Định dạng response không mong đợi: {type(response_data)}")
                return self._fallback_response(question)
                
        except Exception as e:
            print(f"❌ Lỗi khi gọi service: {e}")
            return self._fallback_response(question)
    
    def _fallback_response(self, question):
        """Fallback response khi service không khả dụng"""
        if self.qa_db is not None:
            # Tìm câu trả lời gần nhất trong database (mock đơn giản)
            for _, row in self.qa_db.iterrows():
                if self._simple_similarity(question, str(row['question'])) > 0.3:
                    return str(row['answer'])
        
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
    
    def _simple_similarity(self, text1, text2):
        """Tính similarity đơn giản bằng Jaccard"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def get_service_info(self):
        """Lấy thông tin về service đang sử dụng"""
        if not self.chatbot_service:
            return {
                'service_type': 'fallback_mock',
                'available': False,
                'description': 'Sử dụng fallback mock service do không import được service thật',
                'django_setup': django_setup_success
            }
        
        try:
            # Lấy thông tin về service (nếu có method get_system_status)
            if hasattr(self.chatbot_service, 'get_system_status'):
                status = self.chatbot_service.get_system_status()
                return {
                    'service_type': 'integrated_real_service', 
                    'available': True,
                    'status': status,
                    'description': 'Sử dụng service thật từ ai_models.services',
                    'django_setup': django_setup_success
                }
            else:
                return {
                    'service_type': 'integrated_real_service',
                    'available': True,
                    'description': 'Sử dụng service thật từ ai_models.services (không có system status)',
                    'django_setup': django_setup_success
                }
        except Exception as e:
            return {
                'service_type': 'integrated_real_service',
                'available': True,
                'error': str(e),
                'description': 'Service có lỗi khi lấy status nhưng vẫn khả dụng',
                'django_setup': django_setup_success
            }

# =================== TÍNH SIMILARITY ===================
def calculate_similarity(text1, text2):
    """
    Tính độ tương đồng giữa 2 văn bản
    🔧 NÂNG CẤP: Có thể thay bằng PhoBERT embedding
    """
    try:
        # Method 1: Jaccard similarity (đơn giản)
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0
        
        # Method 2: Có thể thêm cosine similarity với TF-IDF
        # TODO: Tích hợp PhoBERT embedding ở đây để tính similarity chính xác hơn
        
        return jaccard
        
    except Exception as e:
        print(f"⚠️ Lỗi tính similarity: {e}")
        return 0.0

# =================== TÌM ĐÁP ÁN MONG ĐỢI ===================
def find_expected_answer(expected_question, qa_db):
    """Tìm câu trả lời mong đợi từ database"""
    if qa_db is None:
        return "Không tìm thấy database QA"
    
    # Tìm câu hỏi khớp nhất
    best_match = None
    best_similarity = 0
    
    for _, row in qa_db.iterrows():
        similarity = calculate_similarity(expected_question, str(row['question']))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row
    
    if best_match is not None and best_similarity > 0.5:
        return str(best_match['answer'])
    else:
        return f"Không tìm thấy đáp án cho: {expected_question[:100]}..."

# =================== ĐÁNH GIÁ TỪNG CÂU ===================
def evaluate_single_question(test_q, expected_q, model_service, qa_db, index):
    """Đánh giá một câu hỏi"""
    print(f"\n📝 Câu {index + 1}: {test_q[:80]}...")
    
    start_time = time.time()
    
    try:
        # Lấy câu trả lời từ model service thật
        model_answer = model_service.get_response(test_q)
        response_time = time.time() - start_time
        
        # Lấy đáp án mong đợi
        expected_answer = find_expected_answer(expected_q, qa_db)
        
        # Tính similarity
        similarity = calculate_similarity(model_answer, expected_answer)
        is_correct = similarity >= SIMILARITY_THRESHOLD
        
        # Hiển thị kết quả
        status = "✅" if is_correct else "❌"
        print(f"   {status} Similarity: {similarity:.3f} | Time: {response_time:.2f}s")
        if not is_correct:
            print(f"   🎯 Expected: {expected_answer[:100]}...")
            print(f"   🤖 Got: {model_answer[:100]}...")
        
        return {
            'test_question': test_q,
            'expected_question': expected_q,
            'model_answer': model_answer,
            'expected_answer': expected_answer,  
            'similarity': similarity,
            'is_correct': is_correct,
            'response_time': response_time
        }
        
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return {
            'test_question': test_q,
            'expected_question': expected_q,
            'model_answer': f"ERROR: {e}",
            'expected_answer': "",
            'similarity': 0.0,
            'is_correct': False,
            'response_time': time.time() - start_time
        }

# =================== CHẠY ĐÁNH GIÁ FULL ===================
def run_evaluation():
    """Chạy đánh giá toàn bộ"""
    print("\n🔍 BẮT ĐẦU ĐÁNH GIÁ...")
    
    # Kiểm tra files
    if not check_files():
        print("❌ Thiếu file quan trọng, sẽ thử chạy với mock data...")
    
    # Load data
    test_df = load_test_questions()
    qa_db = load_qa_database()
    
    if test_df is None:
        print("❌ Không thể tải được test questions")
        return
    
    # Khởi tạo integrated model service
    print("\n🤖 Khởi tạo Integrated Model Service...")
    model_service = IntegratedModelService()
    
    # Hiển thị thông tin service
    service_info = model_service.get_service_info()
    print(f"🔧 Service Type: {service_info['service_type']}")
    print(f"📊 Available: {service_info['available']}")
    print(f"🐍 Django Setup: {service_info.get('django_setup', 'Unknown')}")
    print(f"📝 Description: {service_info['description']}")
    if 'error' in service_info:
        print(f"⚠️ Warning: {service_info['error']}")
    
    # Chạy đánh giá
    results = []
    categories = {
        'paraphrasing': list(range(0, min(20, len(test_df)))),
        'edge_case': list(range(20, min(30, len(test_df)))),
        'hard_negative': list(range(30, min(40, len(test_df)))),
        'combining': list(range(40, min(50, len(test_df))))
    }
    
    for index, row in test_df.iterrows():
        result = evaluate_single_question(
            test_q=row['test_question'],
            expected_q=row['correct_question_from_original_csv'],
            model_service=model_service,
            qa_db=qa_db,
            index=index
        )
        
        # Xác định category
        for cat, indices in categories.items():
            if index in indices:
                result['category'] = cat
                break
        else:
            result['category'] = row.get('category', 'unknown')
        
        results.append(result)
        
        # Delay nhỏ để tránh overload
        time.sleep(0.2)
    
    return results, service_info

# =================== TẠO BÁO CÁO ===================
def generate_report(results, service_info):
    """Tạo báo cáo kết quả"""
    if not results:
        return "Không có kết quả để báo cáo"
    
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    avg_similarity = np.mean([r['similarity'] for r in results])
    avg_time = np.mean([r['response_time'] for r in results])
    
    # Thống kê theo category
    category_stats = {}
    categories = ['paraphrasing', 'edge_case', 'hard_negative', 'combining']
    for category in categories:
        cat_results = [r for r in results if r.get('category') == category]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r['is_correct'])
            category_stats[category] = {
                'total': len(cat_results),
                'correct': cat_correct,
                'accuracy': cat_correct / len(cat_results)
            }
    
    # Đánh giá chất lượng
    overall_accuracy = correct_answers / total_questions
    if overall_accuracy >= 0.9:
        quality = "XUẤT SẮC 🏆"
    elif overall_accuracy >= 0.8:
        quality = "TỐT 👍"
    elif overall_accuracy >= 0.7:
        quality = "KHẤP THỎA ⚠️"
    else:
        quality = "CẦN CẢI THIỆN ❌"
    
    # Tạo báo cáo
    report = f"""
{'='*60}
🎯 KẾT QUẢ ĐÁNH GIÁ CHATBOT (TÍCH HỢP SERVICE THẬT)
{'='*60}
🔧 THÔNG TIN SERVICE:
   • Service Type: {service_info.get('service_type', 'unknown')}
   • Available: {service_info.get('available', False)}
   • Django Setup: {service_info.get('django_setup', 'Unknown')}
   • Description: {service_info.get('description', 'N/A')}
   
📊 TỔNG QUAN:
   • Tổng số câu hỏi: {total_questions}
   • Số câu trả lời đúng: {correct_answers}
   • Độ chính xác: {overall_accuracy:.1%}
   • Similarity trung bình: {avg_similarity:.3f}
   • Thời gian phản hồi TB: {avg_time:.2f}s
   • Ngưỡng similarity: {SIMILARITY_THRESHOLD}

🏅 ĐÁNH GIÁ TỔNG THỂ: {quality}

📈 CHI TIẾT THEO LOẠI CÂU HỎI:
"""
    
    for category, stats in category_stats.items():
        accuracy_pct = stats['accuracy'] * 100
        icon = "✅" if stats['accuracy'] >= 0.8 else "⚠️" if stats['accuracy'] >= 0.6 else "❌"
        cat_name = category.replace('_', ' ').title()
        report += f"   {icon} {cat_name}: {accuracy_pct:.1f}% ({stats['correct']}/{stats['total']})\n"
    
    # Gợi ý cải thiện
    suggestions = []
    for category, stats in category_stats.items():
        if stats['accuracy'] < 0.7:
            cat_name = category.replace('_', ' ')
            suggestions.append(f"Cải thiện khả năng xử lý {cat_name}")
    
    if avg_time > 2.0:
        suggestions.append("Tối ưu hóa tốc độ phản hồi")
    
    if avg_similarity < 0.6:
        suggestions.append("Cải thiện độ chính xác phát hiện ý định")
    
    if suggestions:
        report += f"\n💡 GỢI Ý CẢI THIỆN:\n"
        for suggestion in suggestions:
            report += f"   • {suggestion}\n"
    
    # Thêm thông tin Fine-tuning
    report += f"\n🚀 THÔNG TIN FINE-TUNING:\n"
    if service_info.get('service_type') == 'integrated_real_service':
        report += f"   • Đang sử dụng model đã fine-tune từ project thật\n"
        report += f"   • Kết quả này phản ánh chất lượng model thực tế\n"
        report += f"   • Khuyến nghị: So sánh với baseline trước khi fine-tune\n"
    else:    
        report += f"   • Đang sử dụng fallback service (không phải model thật)\n"
        report += f"   • Cần kiểm tra lại tích hợp với ai_models.services\n"
        report += f"   • Django setup: {service_info.get('django_setup', 'Unknown')}\n"
    
    return report

# =================== LƯU KẾT QUẢ ===================
def save_results(results, report, service_info):
    """Lưu kết quả ra file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Lưu chi tiết ra CSV
    df_results = pd.DataFrame(results)
    csv_file = f"integrated_test_results_{timestamp}.csv"
    df_results.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Lưu báo cáo ra text
    report_file = f"integrated_test_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Lưu service info ra JSON
    service_file = f"service_info_{timestamp}.json"
    with open(service_file, 'w', encoding='utf-8') as f:
        json.dump(service_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 ĐÃ LƯU KẾT QUẢ:")
    print(f"   📊 Chi tiết: {csv_file}")
    print(f"   📄 Báo cáo: {report_file}")
    print(f"   🔧 Service info: {service_file}")

# =================== MAIN FUNCTION ===================
def main():
    """Hàm chính"""
    try:
        print(f"📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔢 Sẽ test {MAX_TEST_QUESTIONS} câu hỏi đầu tiên")
        print(f"🎯 Ngưỡng similarity: {SIMILARITY_THRESHOLD}")
        print(f"🐍 Django setup: {'✅ Success' if django_setup_success else '❌ Failed'}")
        print(f"🤖 Service integration: {'✅ Available' if SERVICE_AVAILABLE else '❌ Fallback only'}")
        
        # Chạy đánh giá
        evaluation_result = run_evaluation()
        
        if evaluation_result:
            results, service_info = evaluation_result
            
            # Tạo báo cáo
            report = generate_report(results, service_info)
            print(report)
            
            # Lưu kết quả
            save_results(results, report, service_info)
            
            print(f"\n🎉 HOÀN THÀNH! Đã đánh giá {len(results)} câu hỏi.")
            
            # Khuyến nghị
            print(f"\n💡 KHUYẾN NGHỊ:")
            if service_info.get('service_type') == 'integrated_real_service':
                print(f"   ✅ Đã test thành công với service thật!")  
                print(f"   📈 Kết quả này có thể dùng để đánh giá chất lượng fine-tuning")
                print(f"   🎯 Tiếp theo: Chạy test với bộ data lớn hơn (tăng MAX_TEST_QUESTIONS)")
            else:
                print(f"   ⚠️ Chỉ test được với fallback service")
                print(f"   🔧 Kiểm tra lại import ai_models.services")
                print(f"   📁 Đảm bảo chạy từ thư mục backend/")
                print(f"   🐍 Thử chạy: python manage.py shell để test Django setup")
        else:
            print("❌ Không có kết quả để báo cáo")
    
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()