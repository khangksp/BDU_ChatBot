import torch
import numpy as np
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import time
from typing import Dict, List, Tuple, Optional

# Try to import transformers, fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. PhoBERT will use fallback mode.")

logger = logging.getLogger(__name__)

class PhoBERTIntentClassifier:
    """
    🚀 Enhanced PhoBERT-based Intent Classification for BDU Lecturers - PRODUCTION VERSION
    
    Features:
    - Ensemble methods (PhoBERT + Keyword + Pattern + Context)
    - 15 intent categories (7 existing + 8 new from QA analysis)
    - Multi-intent detection
    - Context-aware classification
    - Confidence calibration
    - Vietnamese normalization support
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TRANSFORMERS_AVAILABLE else None
        self.tokenizer = None
        self.model = None
        
        # ESSENTIAL: Set fallback_mode FIRST
        self.fallback_mode = True  # Default to fallback mode
        
        # 🆕 Enhanced components
        self.ensemble_weights = {
            'phobert_semantic': 0.4,
            'keyword_matching': 0.3,
            'pattern_matching': 0.2,
            'context_analysis': 0.1
        }
        
        # 🆕 Multi-intent detection patterns
        self.multi_intent_patterns = self._initialize_multi_intent_patterns()
        
        # 🆕 Context-aware intent boosting
        self.context_boosting_rules = self._initialize_context_boosting()
        
        # 🆕 Intent confidence calibration
        self.confidence_calibration = self._initialize_confidence_calibration()
        
        # Initialize components
        self.intent_categories = self._initialize_enhanced_lecturer_intents()
        self.entity_patterns = self._initialize_enhanced_lecturer_entities()
        
        # ✅ Initialize normalizer
        self.normalizer = None
        try:
            from .vietnamese_normalizer import VietnameseNormalizer
            self.normalizer = VietnameseNormalizer()
            logger.info("✅ Vietnamese Normalizer initialized successfully")
        except ImportError as e:
            logger.warning(f"❌ Failed to import VietnameseNormalizer: {e}")
            # Create dummy normalizer
            self.normalizer = self._create_dummy_normalizer()
        
        # Try to load model only if transformers available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.load_model()
            except Exception as e:
                logger.warning(f"PhoBERT model failed to load: {str(e)}")
                self.fallback_mode = True
        else:
            logger.warning("PhoBERT running in enhanced fallback mode (keyword-based) for lecturers")

    def _create_dummy_normalizer(self):
        """Create dummy normalizer if import fails"""
        class DummyNormalizer:
            def normalize_query(self, query):
                return query.lower().strip()
            
            def create_search_variants(self, query):
                return [query, query.lower()]
        
        return DummyNormalizer()
    
    def _initialize_multi_intent_patterns(self):
        """🆕 Patterns để detect multi-intent queries"""
        return {
            'sequential_intents': [
                # "Tôi muốn xem lịch dạy và nộp báo cáo đề thi"
                r'(?P<intent1>\w+)\s+(?:và|rồi|sau đó)\s+(?P<intent2>\w+)',
                r'(?P<intent1>\w+)\s*,\s*(?P<intent2>\w+)',
            ],
            'conditional_intents': [
                # "Nếu tôi nộp báo cáo đề thi thì có được khen thưởng không?"
                r'(?:nếu|khi)\s+(?P<condition>\w+).*(?:thì|sẽ)\s+(?P<result>\w+)',
            ],
            'comparison_intents': [
                # "So sánh quy trình báo cáo đề thi và báo cáo nghiên cứu"
                r'(?:so sánh|khác biệt)\s+(?P<intent1>\w+)\s+(?:và|với)\s+(?P<intent2>\w+)',
            ]
        }
    
    def _initialize_context_boosting(self):
        """🆕 Context-aware intent boosting rules"""
        return {
            'time_sensitive_boost': {
                'deadline_keywords': ['hạn cuối', 'deadline', 'trước', 'sau'],
                'boost_intents': ['deadline_temporal', 'bank_exam_questions'],
                'boost_factor': 1.3
            },
            'personal_context_boost': {
                'personal_keywords': ['tôi', 'của tôi', 'cho tôi', 'với tôi'],
                'boost_intents': ['personal_schedule', 'personal_info'],
                'boost_factor': 1.5
            },
            'urgency_boost': {
                'urgency_keywords': ['gấp', 'khẩn cấp', 'urgent', 'cần ngay'],
                'boost_intents': ['deadline_temporal', 'clarification_needed'],
                'boost_factor': 1.4
            },
            'recent_conversation_boost': {
                # Boost intents that appeared in recent conversation
                'decay_factor': 0.8,  # Each turn back reduces boost by 20%
                'max_history': 3
            }
        }
    
    def _initialize_confidence_calibration(self):
        """🆕 Confidence calibration parameters"""
        return {
            'base_thresholds': {
                'very_high': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3,
                'very_low': 0.1
            },
            'calibration_factors': {
                'single_keyword_match': 0.8,     # Reduce confidence for single keyword
                'multiple_keyword_match': 1.2,   # Boost for multiple keywords
                'exact_phrase_match': 1.5,       # Strong boost for exact phrases
                'semantic_similarity_high': 1.3, # Boost for high semantic similarity
                'context_continuity': 1.2,       # Boost for context continuity
                'multi_intent_detected': 0.9     # Slight reduction for multi-intent
            }
        }
    
    def _initialize_enhanced_lecturer_intents(self):
        """🚀 Enhanced intent categories với tất cả 15 intents (7 + 8 new)"""
        return {
            # ===== EXISTING INTENTS (7) =====
            'greeting': {
                'keywords': ['xin chào', 'hello', 'hi', 'chào thầy', 'chào cô', 'halo', 'chào', 'hey'],
                'confidence_threshold': 0.6,
                'description': 'Chào hỏi',
                'patterns': [r'(?:xin\s+)?chào\s+(?:thầy|cô)', r'hello|hi|hey'],
                'context_indicators': {'greeting_markers': ['chào', 'hello']},
                'boosters': {'polite_terms': ['xin chào', 'chào thầy'], 'boost_factor': 1.2}
            },
            
            'bank_exam_questions': {
                'keywords': ['ngân hàng đề thi', 'ngan hang de thi', 'đề thi', 'de thi', 'báo cáo đề thi', 'file mềm'],
                'confidence_threshold': 0.4,
                'description': 'Ngân hàng đề thi',
                'patterns': [r'(?:báo cáo|nộp|gửi).*(?:ngân hàng|đề thi)', r'file\s+mềm.*đề\s+thi'],
                'context_indicators': {'document_markers': ['báo cáo', 'file'], 'exam_markers': ['đề thi', 'ngân hàng']},
                'boosters': {'document_refs': ['TB_1252'], 'specific_terms': ['ldkham@bdu.edu.vn'], 'boost_factor': 1.4}
            },
            
            'annual_task_declaration': {
                'keywords': ['kê khai nhiệm vụ', 'ke khai nhiem vu', 'nhiệm vụ năm học', 'giờ chuẩn', 'gio chuan', 'thỉnh giảng'],
                'confidence_threshold': 0.4,
                'description': 'Kê khai nhiệm vụ năm học',
                'patterns': [r'kê\s+khai.*nhiệm\s+vụ', r'giờ\s+chuẩn.*năm\s+học'],
                'context_indicators': {'task_markers': ['kê khai', 'nhiệm vụ'], 'time_markers': ['năm học']},
                'boosters': {'document_refs': ['TB_746'], 'specific_terms': ['daotao@bdu.edu.vn'], 'boost_factor': 1.4}
            },
            
            'academic_journal': {
                'keywords': ['tạp chí', 'tap chi', 'tạp chí khoa học', 'bài viết', 'bai viet', 'nghiên cứu', 'gửi bài'],
                'confidence_threshold': 0.4,
                'description': 'Tạp chí khoa học',
                'patterns': [r'(?:gửi|nộp).*(?:bài viết|tạp chí)', r'tạp\s+chí.*khoa\s+học'],
                'context_indicators': {'research_markers': ['nghiên cứu', 'bài viết'], 'journal_markers': ['tạp chí']},
                'boosters': {'document_refs': ['TB_676'], 'specific_terms': ['jst@bdu.edu.vn'], 'boost_factor': 1.4}
            },
            'academic_regulations': {
                'keywords': [
                    'điểm', 'học lại', 'nâng điểm', 'điểm trung bình', 'dtb', 'tính điểm',
                    'đạt', 'không đạt', 'tín chỉ', 'chuyển đổi', 'công nhận', 
                    'phần trăm', 'khối lượng', 'quy định học tập', 'xử lý học vụ', 
                    'cảnh báo học tập', 'đình chỉ', 'buộc thôi học', 'vi phạm quy định'
                ],
                'confidence_threshold': 0.3,
                'description': 'Quy định học tập, điểm số và xử lý học vụ',
                'patterns': [r'(?:điểm|học lại|quy định).*(?:như thế nào|quy định)', r'(?:tín chỉ|chuyển đổi).*(?:phần trăm|tối thiểu)'],
            },

            'graduation_ceremony': {
                'keywords': [
                    'tốt nghiệp', 'lễ tốt nghiệp', 'tham dự', 'ai tham dự', 'được phép',
                    'cử nhân', 'bằng cấp', 'thành phần', 'danh sách'
                ],
                'confidence_threshold': 0.3,
                'description': 'Lễ tốt nghiệp và cấp bằng',
                'patterns': [r'(?:ai|những ai).*(?:tham dự|được phép)', r'lễ\s+tốt\s+nghiệp'],
            },
            
            'competition_awards': {
                'keywords': ['thi đua', 'thi dua', 'khen thưởng', 'khen thuong', 'danh hiệu', 'bằng khen', 'lễ khen thưởng', 'le khen thuong', 'hội đồng', 'thường trực', 'kỷ luật'],
                'confidence_threshold': 0.4,
                'description': 'Thi đua khen thưởng và kỷ luật',
                'patterns': [r'thi\s+đua.*khen\s+thưởng', r'(?:danh hiệu|bằng khen|kỷ luật)'],
                'context_indicators': {'award_markers': ['thi đua', 'khen thưởng'], 'recognition_markers': ['danh hiệu']},
                'boosters': {'award_terms': ['chiến sĩ thi đua', 'lao động tiên tiến'], 'boost_factor': 1.3}
            },
            
            'personal_schedule': {
                'keywords': ['lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi', 'tôi giảng', 'tôi dạy'],
                'confidence_threshold': 0.3,
                'description': 'Lịch giảng dạy cá nhân',
                'patterns': [r'(?:lịch|tkb|thời khóa biểu).*(?:tôi|của tôi)', r'(?:tôi|mình).*(?:dạy|giảng)'],
                'context_indicators': {'personal_markers': ['tôi', 'của tôi'], 'schedule_markers': ['lịch', 'tkb']},
                'boosters': {'time_contexts': ['hôm nay', 'ngày mai'], 'boost_factor': 1.5}
            },
            
            'personal_info': {
                'keywords': ['tôi là ai', 'toi la ai', 'thông tin của tôi', 'email của tôi', 'chức danh của tôi'],
                'confidence_threshold': 0.4,
                'description': 'Thông tin cá nhân giảng viên',
                'patterns': [r'(?:tôi|mình)\s+là\s+ai', r'(?:thông tin|email).*của\s+tôi'],
                'context_indicators': {'identity_markers': ['tôi là', 'thông tin'], 'info_markers': ['email', 'chức danh']},
                'boosters': {'identity_terms': ['hồ sơ', 'profile'], 'boost_factor': 1.2}
            },
            
            # ===== NEW INTENTS FROM QA ANALYSIS (8) =====
            'document_reference': {
                'keywords': ['thông báo số', 'TB_', 'quyết định số', 'QĐ_', 'theo thông báo', 'TB_1252', 'TB_746', 'TB_676'],
                'confidence_threshold': 0.3,
                'description': 'Hỏi về văn bản, thông báo cụ thể',
                'patterns': [r'(?:theo|căn cứ)\s+(?:thông báo|TB|quyết định|QĐ)\s+số\s*\d+', r'TB_\d+'],
                'context_indicators': {'authority_markers': ['theo', 'căn cứ'], 'document_numbers': ['số', 'TB_']},
                'boosters': {'document_refs': ['TB_1252', 'TB_746'], 'boost_factor': 1.6}
            },
            
            'deadline_temporal': {
                'keywords': ['hạn cuối', 'han cuoi', 'deadline', 'trước ngày', 'hết ngày', 'chậm nhất', '15/01/2024'],
                'confidence_threshold': 0.4,
                'description': 'Hỏi về thời hạn, deadline',
                'patterns': [r'(?:hạn|deadline)\s*(?:cuối|chót|nào)', r'\d{1,2}[/]\d{1,2}[/]\d{4}'],
                'context_indicators': {'time_markers': ['ngày', 'tháng'], 'deadline_urgency': ['hạn', 'deadline']},
                'boosters': {'urgent_terms': ['gấp', 'khẩn cấp'], 'boost_factor': 1.4}
            },
            
            'contact_responsibility': {
                'keywords': ['gửi cho ai', 'phụ trách', 'địa chỉ email', '@bdu.edu.vn', 'ldkham@bdu.edu.vn', 'ai ký'],
                'confidence_threshold': 0.4,
                'description': 'Hỏi về liên hệ, người phụ trách',
                'patterns': [r'(?:gửi|liên hệ)\s+(?:cho\s+)?ai', r'\w+@bdu\.edu\.vn'],
                'context_indicators': {'contact_markers': ['email', 'liên hệ'], 'responsibility_markers': ['phụ trách', 'ký']},
                'boosters': {'specific_emails': ['ldkham@bdu.edu.vn'], 'boost_factor': 1.3}
            },
            
            'technical_specification': {
                'keywords': ['định dạng', 'file mềm', 'cách gửi', 'thể lệ', 'bản điện tử', 'yêu cầu kỹ thuật'],
                'confidence_threshold': 0.5,
                'description': 'Hỏi về yêu cầu kỹ thuật, định dạng',
                'patterns': [r'(?:định dạng|format)\s+(?:file|gì|nào)', r'(?:cách|thế nào)\s+(?:gửi|nộp)'],
                'context_indicators': {'format_markers': ['định dạng', 'file'], 'submission_markers': ['gửi', 'nộp']},
                'boosters': {'technical_terms': ['định dạng', 'yêu cầu kỹ thuật'], 'boost_factor': 1.2}
            },
            
            'compliance_consequence': {
                'keywords': ['nếu không', 'xử lý như thế nào', 'hậu quả', 'vi phạm', 'bị xem là', 'thiếu giờ nghĩa vụ'],
                'confidence_threshold': 0.5,
                'description': 'Hỏi về tuân thủ và hậu quả vi phạm',
                'patterns': [r'(?:nếu|nếu như)\s+(?:không|chậm|vi phạm)', r'(?:xử lý|hậu quả)\s+(?:như thế nào|gì)'],
                'context_indicators': {'condition_markers': ['nếu'], 'consequence_markers': ['bị', 'hậu quả']},
                'boosters': {'violation_terms': ['vi phạm quy định'], 'boost_factor': 1.3}
            },
            
            'process_sequence': {
                'keywords': ['quy trình', 'thủ tục', 'các bước', 'thứ tự', 'từng bước', 'làm thế nào', 'cách thức'],
                'confidence_threshold': 0.4,
                'description': 'Hỏi về quy trình, thủ tục',
                'patterns': [r'(?:quy trình|thủ tục)\s+(?:gì|nào|như thế nào)', r'(?:các bước|thứ tự)\s+(?:thực hiện|làm)'],
                'context_indicators': {'process_markers': ['quy trình', 'thủ tục'], 'sequence_markers': ['bước', 'thứ tự']},
                'boosters': {'process_terms': ['quy trình chi tiết'], 'boost_factor': 1.2}
            },
            
            'authorization_approval': {
                'keywords': ['phê duyệt', 'ai duyệt', 'cho phép', 'thẩm quyền', 'hiệu trưởng', 'phó hiệu trưởng'],
                'confidence_threshold': 0.4,
                'description': 'Hỏi về phê duyệt, ủy quyền',
                'patterns': [r'(?:ai|đơn vị nào)\s+(?:duyệt|phê duyệt)', r'(?:có được|được)\s+(?:phép|quyền)'],
                'context_indicators': {'authority_markers': ['duyệt', 'phê duyệt'], 'hierarchy_markers': ['hiệu trưởng']},
                'boosters': {'authority_titles': ['hiệu trưởng'], 'boost_factor': 1.3}
            },
            
            'document_comparison': {
                'keywords': ['so sánh', 'khác biệt', 'TB_1252 và TB_746', 'các thông báo', 'giống nhau', 'khác với'],
                'confidence_threshold': 0.5,
                'description': 'So sánh giữa các văn bản',
                'patterns': [r'(?:so sánh|khác biệt)\s+(?:giữa|với)', r'TB_\d+\s+(?:và|với)\s+TB_\d+'],
                'context_indicators': {'comparison_markers': ['so sánh', 'khác'], 'conjunction_markers': ['và', 'với']},
                'boosters': {'multi_doc_refs': ['TB_1252 và TB_746'], 'boost_factor': 1.4}
            },
            
            # ===== UTILITY INTENTS =====
            'clarification_needed': {
                'keywords': ['gì', 'gi', 'sao', 'nào', 'nao', 'như thế nào', 'làm sao', 'cách nào'],
                'confidence_threshold': 0.2,
                'description': 'Cần làm rõ',
                'patterns': [r'(?:gì|sao|nào|như thế nào)\s*\?*$'],
                'context_indicators': {'vague_markers': ['gì', 'sao', 'nào']},
                'boosters': {'question_markers': ['?'], 'boost_factor': 1.1}
            },
            
            'general': {
                'keywords': ['thông tin', 'hỗ trợ', 'giúp', 'hướng dẫn', 'bdu', 'đại học bình dương'],
                'confidence_threshold': 0.15,
                'description': 'Câu hỏi chung',
                'patterns': [r'(?:thông tin|hỗ trợ|giúp).*(?:bdu|đại học)'],
                'context_indicators': {'general_markers': ['thông tin', 'hỗ trợ']},
                'boosters': {'school_terms': ['bdu', 'đại học bình dương'], 'boost_factor': 1.0}
            },
            
            'university_general_info': {
                'keywords': [
                    # Thông tin trường
                    'trường đại học bình dương', 'bdu', 'đại học bình dương',
                    'thành lập', 'lịch sử', 'phát triển', 'đặc điểm',
                    
                    # Cơ cấu tổ chức  
                    'phòng', 'khoa', 'bộ môn', 'cơ cấu tổ chức', 'đơn vị',
                    'phòng quản lý đào tạo', 'phòng tổng hợp', 'phòng tài chính',
                    
                    # Đào tạo chung
                    'ngành', 'chuyên ngành', 'đào tạo', 'chương trình', 'cấp bậc',
                    'phương pháp đào tạo', 'cộng học', 'người thầy',
                    
                    # Đội ngũ
                    'giảng viên', 'cán bộ', 'nhân sự', 'đội ngũ', 'giáo viên',
                    
                    # Sinh viên  
                    'sinh viên', 'học sinh', 'dịch vụ sinh viên', 'hỗ trợ sinh viên'
                ],
                'confidence_threshold': 0.3,
                'description': 'Thông tin chung về Đại học Bình Dương',
                'patterns': [
                    r'trường đại học bình dương.*(?:gì|nào|như thế nào)',
                    r'(?:phòng|khoa|bộ môn).*(?:chức năng|nhiệm vụ)',
                    r'bdu.*(?:có|là|được|thành lập)',
                    r'(?:ngành|chuyên ngành).*(?:nào|gì|như thế nào)'
                ],
                'context_indicators': {
                    'university_markers': ['trường', 'bdu', 'đại học'],
                    'org_structure': ['phòng', 'khoa', 'đơn vị'],
                    'info_requests': ['là gì', 'như thế nào', 'bao gồm']
                },
                'boosters': {
                    'specific_terms': ['cộng học', 'người thầy', 'cơ cấu tổ chức'],
                    'boost_factor': 1.3
                }
            },
            
            'quality_assessment_accreditation': {
                'keywords': ['chất lượng', 'đánh giá', 'kiểm định', 'akc', 'chuẩn đầu ra', 'mục tiêu', 'ctđt', 'clo', 'plo'],
                'confidence_threshold': 0.4,
                'description': 'Đánh giá chất lượng và kiểm định',
                'patterns': [r'(?:đánh giá|chất lượng).*(?:chương trình|môn học)', r'kiểm định.*chất lượng'],
            },

            'financial_tuition_fees': {
                'keywords': ['học phí', 'chi phí', 'tiền', 'thanh toán', 'miễn giảm', 'phí dịch vụ'],
                'confidence_threshold': 0.4,
                'description': 'Học phí và tài chính',
                'patterns': [r'(?:học phí|chi phí).*(?:bao nhiêu|như thế nào)', r'thanh toán.*học phí'],
            },

            'registration_admission': {
                'keywords': ['đăng ký', 'tuyển sinh', 'xét tuyển', 'nộp hồ sơ', 'thủ tục', 'nhập học'],
                'confidence_threshold': 0.4, 
                'description': 'Đăng ký và tuyển sinh',
                'patterns': [r'(?:đăng ký|tuyển sinh).*(?:như thế nào|thủ tục)', r'hồ sơ.*(?:gì|nào)'],
            },

            'scholarships_financial_support': {
                'keywords': ['học bổng', 'hỗ trợ', 'miễn giảm', 'khó khăn', 'ưu đãi'],
                'confidence_threshold': 0.4,
                'description': 'Học bổng và hỗ trợ tài chính',
                'patterns': [r'học bổng.*(?:như thế nào|điều kiện)', r'hỗ trợ.*(?:tài chính|học phí)'],
            },

            'facilities_infrastructure': {
                'keywords': ['cơ sở vật chất', 'phòng học', 'thiết bị', 'thư viện', 'ký túc xá', 'wifi', 'phòng thí nghiệm', 'phòng máy tính', 'phòng CAD/CAM', 'sân thể thao', 'bãi xe', 'căn tin', 'khu vực nghỉ ngơi', 'máy chiếu', 'điều hòa', 'camera an ninh'],
                'confidence_threshold': 0.4,
                'description': 'Cơ sở vật chất và tiện ích', 
                'patterns': [r'(?:phòng học|thiết bị|cơ sở vật chất).*(?:như thế nào|ra sao|trường)', r'ký túc xá'],
            },
            
            'lecturer_compensation': {
                'keywords': [
                    # Thù lao giảng dạy
                    'thù lao', 'thù lao giảng dạy', 'thu lao', 'tiền dạy', 'tiền giảng dạy',
                    'hệ số giảng dạy', 'hệ số thực hành', 'hệ số lý thuyết', 'hệ số tiếng nước ngoài',
                    'công thức tính thù lao', 'cách tính thù lao', 'tính thù lao',
                    
                    # Định mức và giờ chuẩn
                    'định mức giờ dạy', 'định mức giờ chuẩn', 'giờ chuẩn giảng dạy',
                    'khối lượng giảng dạy', 'số giờ dạy', 'tiết dạy', 'buổi dạy',
                    'giờ lý thuyết', 'giờ thực hành', 'giờ thí nghiệm',
                    
                    # Lương và chế độ
                    'lương giảng viên', 'mức lương', 'bậc lương', 'hạng lương',
                    'lương cơ bản', 'hệ số lương', 'ngạch lương', 'bảng lương',
                    'nâng lương', 'tăng lương', 'điều chỉnh lương',
                    
                    # Phụ cấp và trợ cấp
                    'phụ cấp', 'phụ cấp trách nhiệm', 'phụ cấp thâm niên', 'phụ cấp độc hại',
                    'phụ cấp khu vực', 'phụ cấp đi xa', 'phụ cấp giảng dạy xa',
                    'trợ cấp', 'trợ cấp ăn trưa', 'trợ cấp xăng xe', 'trợ cấp điện thoại',
                    
                    # Thưởng và khen thưởng
                    'thưởng giảng viên', 'thưởng cuối năm', 'thưởng thành tích',
                    'thưởng nghiên cứu khoa học', 'thưởng xuất sắc', 'tiền thưởng',
                    'khen thưởng tài chính', 'bonus', 'incentive',
                    
                    # Hợp đồng và tuyển dụng
                    'thù lao hợp đồng', 'lương hợp đồng', 'thù lao thỉnh giảng',
                    'mức thù lao tuyển dụng', 'thù lao cộng tác viên',
                    'lương thử việc', 'lương kí kết hợp đồng',
                    
                    # Thanh toán và chi trả
                    'chi trả lương', 'thanh toán thù lao', 'ngày trả lương',
                    'chuyển khoản lương', 'nhận lương', 'rút lương',
                    'bảng thanh toán', 'phiếu lương', 'slip lương',
                    
                    # Quyết định liên quan
                    'QĐ 442', 'QĐ 895', 'QĐ 1733', 'QĐ 101', 'QĐ 1153', 'QĐ 2004',
                    'quyết định 442', 'quyết định thù lao', 'quyết định lương',
                    
                    # Từ đặc biệt từ phân tích
                    'giáo sư 15 triệu', 'phó giáo sư 13 triệu', 'tiến sĩ 10 triệu', 'thạc sĩ 5 triệu',
                    'dạy 1 giờ bao nhiêu', 'giờ dạy được bao nhiêu', 'mức giá giờ dạy'
                ],
                'confidence_threshold': 0.3,
                'description': 'Lương, thù lao và chế độ đãi ngộ giảng viên',
                'patterns': [
                    r'(?:thù lao|lương).*(?:giảng viên|dạy|giảng dạy)',
                    r'(?:hệ số|định mức).*(?:giảng dạy|dạy|lương)',
                    r'(?:dạy|giảng)\s+\d+\s+(?:giờ|tiết).*(?:bao nhiêu|được|nhận)',
                    r'(?:giáo sư|phó giáo sư|tiến sĩ|thạc sĩ).*(?:\d+\s*triệu|thù lao|lương)',
                    r'(?:phụ cấp|trợ cấp).*(?:giảng viên|dạy)',
                    r'(?:thanh toán|chi trả).*(?:thù lao|lương)',
                    r'(?:QĐ|quyết định)\s*(?:442|895|1733|101|1153|2004)',
                    r'(?:bao nhiêu|mức|giá).*(?:tiền|đồng).*(?:dạy|giảng)'
                ],
                'context_indicators': {
                    'compensation_markers': ['thù lao', 'lương', 'tiền', 'đồng', 'triệu'],
                    'teaching_markers': ['giảng dạy', 'dạy', 'giảng', 'giờ', 'tiết'],
                    'calculation_markers': ['hệ số', 'định mức', 'tính', 'công thức'],
                    'position_markers': ['giáo sư', 'phó giáo sư', 'tiến sĩ', 'thạc sĩ', 'giảng viên'],
                    'money_markers': ['triệu', 'nghìn', 'đồng', 'k', 'tr', 'VNĐ']
                },
                'boosters': {
                    'specific_amounts': ['15 triệu', '13 triệu', '10 triệu', '5 triệu', '2 triệu'],
                    'official_docs': ['QĐ 442', 'QĐ 895', 'QĐ 1733', 'QĐ 101'], 
                    'calculation_terms': ['công thức tính thù lao', 'hệ số giảng dạy'],
                    'boost_factor': 1.5
                }
            },

            'salary_benefits': {
                'keywords': [
                    # Chế độ làm việc
                    'chế độ làm việc', 'chế độ công tác', 'chế độ nghỉ phép', 'nghỉ phép năm',
                    'thời gian làm việc', 'giờ làm việc', 'ca làm việc', 'lịch làm việc',
                    'nghỉ lễ', 'nghỉ tết', 'nghỉ hè', 'nghỉ thai sản', 'nghỉ ốm',
                    
                    # Phúc lợi và đãi ngộ  
                    'phúc lợi', 'đãi ngộ', 'quyền lợi', 'chế độ ưu đãi',
                    'bảo hiểm xã hội', 'bảo hiểm y tế', 'bảo hiểm thất nghiệp',
                    'bảo hiểm tai nạn', 'chăm sóc sức khỏe', 'khám sức khỏe định kỳ',
                    
                    # Nghỉ dưỡng và du lịch
                    'nghỉ dưỡng', 'du lịch công tác', 'tham quan học tập',
                    'nghỉ mát', 'team building', 'hoạt động tập thể',
                    
                    # Đào tạo và phát triển
                    'đào tạo nâng cao', 'học tập nâng cao trình độ', 'tự đào tạo',
                    'học thêm', 'học cao học', 'học tiến sĩ', 'nghiên cứu sinh',
                    'hỗ trợ học phí', 'học bổng nâng cao trình độ',
                    
                    # Trang thiết bị và cơ sở vật chất
                    'laptop công tác', 'máy tính làm việc', 'điện thoại công việc',
                    'phòng làm việc', 'bàn ghế làm việc', 'trang thiết bị giảng dạy',
                    'hỗ trợ internet', 'wifi miễn phí', 'văn phòng phẩm',
                    
                    # Hỗ trợ khác
                    'hỗ trợ nhà ở', 'hỗ trợ đi lại', 'hỗ trợ con em', 'trợ cấp gia đình',
                    'hỗ trợ tang chế', 'hỗ trợ hiếu hỷ', 'quà tết', 'quà sinh nhật',
                    'tiệc cuối năm', 'gala dinner', 'sự kiện công ty'
                ],
                'confidence_threshold': 0.4,
                'description': 'Chế độ đãi ngộ và phúc lợi giảng viên',
                'patterns': [
                    r'(?:chế độ|quyền lợi|phúc lợi).*(?:giảng viên|cán bộ)',
                    r'(?:bảo hiểm|hỗ trợ).*(?:y tế|xã hội|học phí)',
                    r'(?:nghỉ|ngày nghỉ).*(?:phép|lễ|tết)',
                    r'(?:đào tạo|học).*(?:nâng cao|thêm|tiếp tục)',
                    r'(?:trang thiết bị|hỗ trợ).*(?:làm việc|giảng dạy)'
                ],
                'context_indicators': {
                    'benefit_markers': ['chế độ', 'quyền lợi', 'phúc lợi', 'đãi ngộ'],
                    'support_markers': ['hỗ trợ', 'trợ cấp', 'bảo hiểm', 'chăm sóc'],
                    'development_markers': ['đào tạo', 'nâng cao', 'học tập', 'phát triển'],
                    'welfare_markers': ['nghỉ dưỡng', 'du lịch', 'team building']
                },
                'boosters': {
                    'welfare_terms': ['đãi ngộ tốt', 'phúc lợi tốt', 'chế độ ưu đãi'],
                    'boost_factor': 1.2
                }
            },

            'university_programs': {
                'keywords': [
                    'chương trình đào tạo', 'chương trình học', 'khóa học', 'học chế', 
                    'tín chỉ tích lũy', 'chuẩn đầu ra', 'mục tiêu đào tạo', 'phương pháp giảng dạy',
                    'giáo trình', 'tài liệu học tập', 'đào tạo đại học', 'đào tạo sau đại học',
                    'liên kết đào tạo', 'hợp tác quốc tế', 'chương trình tiên tiến',
                    'chất lượng đào tạo', 'cải tiến chương trình', 'phát triển chương trình'
                ],
                'confidence_threshold': 0.4,
                'description': 'Chương trình và phương pháp đào tạo',
                'patterns': [
                    r'chương trình.*(?:đào tạo|học|giảng dạy)',
                    r'(?:phương pháp|cách thức).*(?:giảng dạy|học tập)',
                    r'(?:chuẩn|mục tiêu).*đầu ra'
                ],
                'context_indicators': {
                    'program_markers': ['chương trình', 'khóa học', 'đào tạo'],
                    'quality_markers': ['chất lượng', 'chuẩn đầu ra', 'mục tiêu']
                },
                'boosters': {
                    'program_terms': ['chương trình tiên tiến', 'liên kết quốc tế'], 
                    'boost_factor': 1.3
                }
            },

            'academic_assessment': {
                'keywords': [
                    'đánh giá học tập', 'đánh giá kết quả', 'kiểm tra đánh giá', 'phương pháp đánh giá',
                    'thang điểm', 'tiêu chí đánh giá', 'feedback sinh viên', 'phản hồi học tập',
                    'đánh giá giảng viên', 'chất lượng giảng dạy', 'khảo sát sinh viên',
                    'cải thiện chất lượng', 'giám sát chất lượng', 'đảm bảo chất lượng',
                    'rà soát chương trình', 'điều chỉnh chương trình', 'nâng cao chất lượng'
                ],
                'confidence_threshold': 0.4,
                'description': 'Đánh giá và kiểm định chất lượng học tập',
                'patterns': [
                    r'đánh giá.*(?:chất lượng|kết quả|học tập)',
                    r'(?:khảo sát|feedback).*sinh viên',
                    r'(?:kiểm tra|giám sát).*chất lượng'
                ],
                'context_indicators': {
                    'assessment_markers': ['đánh giá', 'kiểm tra', 'khảo sát'],
                    'quality_markers': ['chất lượng', 'tiêu chuẩn', 'chuẩn']
                },
                'boosters': {
                    'assessment_terms': ['đánh giá toàn diện', 'kiểm định chất lượng'], 
                    'boost_factor': 1.4
                }
            },

            'student_services': {
                'keywords': [
                    'dịch vụ sinh viên', 'hỗ trợ sinh viên', 'cố vấn học tập', 'tư vấn học tập',
                    'ký túc xá', 'phòng ở', 'căn tin', 'phương tiện ăn uống', 'y tế sinh viên',
                    'bảo hiểm y tế', 'hoạt động ngoại khóa', 'câu lạc bộ sinh viên',
                    'sinh hoạt cộng đồng', 'đoàn thanh niên', 'hội sinh viên',
                    'tuyên truyền giáo dục', 'định hướng nghề nghiệp', 'tìm việc làm',
                    'thực tập sinh viên', 'kết nối doanh nghiệp'
                ],
                'confidence_threshold': 0.4,
                'description': 'Dịch vụ và hỗ trợ sinh viên',
                'patterns': [
                    r'(?:dịch vụ|hỗ trợ).*sinh viên',
                    r'(?:cố vấn|tư vấn).*học tập',
                    r'(?:ký túc xá|phòng ở|căn tin)'
                ],
                'context_indicators': {
                    'service_markers': ['dịch vụ', 'hỗ trợ', 'tư vấn'],
                    'student_markers': ['sinh viên', 'học sinh', 'cố vấn']
                },
                'boosters': {
                    'service_terms': ['dịch vụ toàn diện', 'hỗ trợ đặc biệt'], 
                    'boost_factor': 1.2
                }
            },

            'administrative_procedures': {
                'keywords': [
                    'thủ tục hành chính', 'quy trình hành chính', 'giấy tờ hành chính',
                    'đơn từ', 'giấy xác nhận', 'bản sao bằng cấp', 'chứng minh học tập',
                    'giấy chuyển trường', 'bảng điểm tích lũy', 'khai học', 'thôi học',
                    'chuyển ngành', 'chuyển lớp', 'nghỉ học tạm thời', 'bảo lưu kết quả',
                    'phúc khảo bài thi', 'khiếu nại điểm số', 'giải quyết thắc mắc',
                    'hồ sơ sinh viên', 'cập nhật thông tin', 'thay đổi thông tin cá nhân'
                ],
                'confidence_threshold': 0.4,
                'description': 'Thủ tục hành chính và giấy tờ',
                'patterns': [
                    r'(?:thủ tục|quy trình).*hành chính',
                    r'(?:đơn|giấy).*(?:xác nhận|chứng minh)',
                    r'(?:chuyển|thôi|nghỉ).*(?:trường|học|ngành)'
                ],
                'context_indicators': {
                    'procedure_markers': ['thủ tục', 'quy trình', 'giấy tờ'],
                    'document_markers': ['đơn', 'giấy', 'bản sao']
                },
                'boosters': {
                    'procedure_terms': ['quy trình đơn giản', 'thủ tục nhanh chóng'], 
                    'boost_factor': 1.3
                }
            },

            'academic_calendar': {
                'keywords': [
                    'lịch học tập', 'thời khóa biểu', 'lịch thi', 'lịch giảng dạy',
                    'học kỳ', 'niên khóa', 'năm học', 'khai giảng', 'bế giảng',
                    'nghỉ lễ', 'nghỉ tết', 'nghỉ hè', 'học phần', 'tín chỉ',
                    'đăng ký môn học', 'rút môn', 'thêm môn', 'thay đổi lịch học',
                    'lịch tập trung', 'lịch học bù', 'lịch thi lại', 'thi vấn đáp',
                    'bảo vệ khóa luận', 'thực tập tốt nghiệp', 'đồ án tốt nghiệp'
                ],
                'confidence_threshold': 0.4,
                'description': 'Lịch học tập và tổ chức giảng dạy',
                'patterns': [
                    r'(?:lịch|thời gian).*(?:học|thi|giảng)',
                    r'(?:học kỳ|năm học|niên khóa)',
                    r'(?:đăng ký|rút|thêm).*môn'
                ],
                'context_indicators': {
                    'time_markers': ['lịch', 'thời gian', 'học kỳ'],
                    'academic_markers': ['học', 'thi', 'giảng dạy']
                },
                'boosters': {
                    'calendar_terms': ['lịch học linh hoạt', 'thời khóa biểu tối ưu'], 
                    'boost_factor': 1.2
                }
            },

            'lecturer_affairs': {
                'keywords': [
                    'công việc giảng viên', 'nhiệm vụ giảng viên', 'trách nhiệm giảng viên',
                    'định mức giờ dạy', 'khối lượng công việc', 'đánh giá giảng viên',
                    'thăng hạng', 'bổ nhiệm', 'thi đua giảng viên', 'khen thưởng giảng viên',
                    'nghiên cứu khoa học', 'công trình nghiên cứu', 'đề tài nghiên cứu',
                    'hội nghị khoa học', 'seminar', 'workshop', 'tập huấn giảng viên',
                    'phát triển năng lực', 'nâng cao trình độ', 'học tập nâng cao',
                    'sabbatical', 'nghỉ phép nghiên cứu', 'trao đổi học thuật'
                ],
                'confidence_threshold': 0.4,
                'description': 'Công việc và phát triển giảng viên',
                'patterns': [
                    r'(?:công việc|nhiệm vụ|trách nhiệm).*giảng viên',
                    r'(?:nghiên cứu|đề tài).*khoa học',
                    r'(?:đánh giá|thăng hạng|khen thưởng).*giảng viên'
                ],
                'context_indicators': {
                    'lecturer_markers': ['giảng viên', 'thầy cô', 'cán bộ'],
                    'work_markers': ['công việc', 'nhiệm vụ', 'nghiên cứu']
                },
                'boosters': {
                    'lecturer_terms': ['giảng viên ưu tú', 'nghiên cứu nổi bật'], 
                    'boost_factor': 1.4
                }
            },

            'international_cooperation': {
                'keywords': [
                    'hợp tác quốc tế', 'trao đổi sinh viên', 'chương trình liên kết',
                    'du học', 'học bổng quốc tế', 'đối tác nước ngoài',
                    'trao đổi giảng viên', 'nghiên cứu hợp tác', 'dự án quốc tế',
                    'hội nghị quốc tế', 'chứng chỉ quốc tế', 'chuẩn quốc tế',
                    'tiếng anh', 'ngoại ngữ', 'IELTS', 'TOEFL', 'TOEIC',
                    'văn hóa quốc tế', 'giao lưu văn hóa', 'sự kiện quốc tế'
                ],
                'confidence_threshold': 0.4,
                'description': 'Hợp tác và giao lưu quốc tế',
                'patterns': [
                    r'(?:hợp tác|trao đổi).*quốc tế',
                    r'(?:du học|học bổng).*(?:nước ngoài|quốc tế)',
                    r'(?:chương trình|dự án).*liên kết'
                ],
                'context_indicators': {
                    'international_markers': ['quốc tế', 'nước ngoài', 'liên kết'],
                    'cooperation_markers': ['hợp tác', 'trao đổi', 'giao lưu']
                },
                'boosters': {
                    'international_terms': ['hợp tác chiến lược', 'đối tác uy tín'], 
                    'boost_factor': 1.3
                }
            },

            'library_resources': {
                'keywords': [
                    'thư viện', 'tài liệu tham khảo', 'sách giáo khoa', 'giáo trình',
                    'cơ sở dữ liệu', 'tài nguyên điện tử', 'sách điện tử', 'tạp chí điện tử',
                    'mượn sách', 'gia hạn sách', 'đặt chỗ sách', 'tìm kiếm tài liệu',
                    'phòng đọc', 'khu vực học tập', 'máy tính tra cứu',
                    'wifi thư viện', 'dịch vụ thư viện', 'hướng dẫn sử dụng',
                    'đào tạo kỹ năng', 'tra cứu thông tin', 'nghiên cứu tài liệu'
                ],
                'confidence_threshold': 0.4,
                'description': 'Thư viện và tài nguyên học tập',
                'patterns': [
                    r'thư viện.*(?:sách|tài liệu|dịch vụ)',
                    r'(?:mượn|gia hạn|tìm).*sách',
                    r'(?:cơ sở dữ liệu|tài nguyên).*điện tử'
                ],
                'context_indicators': {
                    'library_markers': ['thư viện', 'sách', 'tài liệu'],
                    'resource_markers': ['cơ sở dữ liệu', 'điện tử', 'tham khảo']
                },
                'boosters': {
                    'library_terms': ['thư viện hiện đại', 'tài nguyên phong phú'], 
                    'boost_factor': 1.2
                }
            },
        }
    
    def _initialize_enhanced_lecturer_entities(self):
        """Enhanced entity patterns for lecturers"""
        return {
            'lecturer_departments': [
                'phòng đảm bảo chất lượng', 'phòng khảo thí', 'phòng tổ chức cán bộ',
                'phòng nghiên cứu hợp tác', 'phòng đào tạo', 'phòng công tác sinh viên'
            ],
            'lecturer_positions': [
                'giảng viên', 'phó giáo sư', 'tiến sĩ', 'thạc sĩ', 'trưởng khoa', 'phó khoa'
            ],
            'document_types': [
                'báo cáo', 'kế hoạch', 'thông báo', 'quyết định', 'file mềm', 'văn bản'
            ],
            'personal_pronouns': [
                'tôi', 'toi', 'của tôi', 'cua toi', 'cho tôi', 'cho toi'
            ],
            'schedule_contexts': [
                'hôm nay', 'hom nay', 'ngày mai', 'ngay mai', 'tuần này', 'tuan nay'
            ],
            'time_expressions': [
                'năm học', 'học kỳ I', 'học kỳ II', 'học kỳ III', 'trước ngày', 'hạn cuối'
            ],
            'lecturer_activities': [
                'giảng dạy', 'nghiên cứu khoa học', 'phục vụ cộng đồng', 'thi đua', 'khen thưởng'
            ],
            'emotions': [
                'cần gấp', 'khẩn cấp', 'urgent', 'quan trọng', 'lo lắng', 'khó khăn'
            ],
            'university_departments_extended': [
                'phòng đảm bảo chất lượng', 'phòng khảo thí', 'phòng tổ chức cán bộ',
                'phòng nghiên cứu hợp tác', 'phòng đào tạo', 'phòng công tác sinh viên',
                'phòng quản lý đào tạo', 'phòng tổng hợp', 'phòng tài chính',  # 🆕
                'khoa công nghệ thông tin', 'khoa kinh tế', 'khoa ngoại ngữ'    # 🆕
            ],
            'academic_programs': [  # 🆕
                'công nghệ thông tin', 'kinh tế', 'quản trị kinh doanh', 'ngôn ngữ anh',
                'kế toán', 'tài chính ngân hàng', 'du lịch', 'logistics'
            ],
            'quality_terms': [  # 🆕
                'akc', 'kiểm định chất lượng', 'chuẩn đầu ra', 'mục tiêu học tập',
                'ctđt', 'clo', 'plo', 'đánh giá môn học', 'feedback sinh viên'
            ],
            'financial_terms': [  # 🆕
                'học phí', 'chi phí đào tạo', 'miễn giảm học phí', 'học bổng',
                'hỗ trợ tài chính', 'phí dịch vụ', 'thanh toán trực tuyến'
            ],
            'facility_terms': [  # 🆕
                'thư viện', 'phòng thí nghiệm', 'phòng máy tính', 'wifi',
                'ký túc xá', 'căn tin', 'bãi xe', 'sân thể thao'
            ],
            'program_types': [
                'chương trình cử nhân', 'chương trình thạc sĩ', 'chương trình tiến sĩ',
                'chương trình liên kết', 'chương trình tiên tiến', 'chương trình quốc tế',
                'đào tạo chính quy', 'đào tạo tại chức', 'đào tạo từ xa',
                'học chế tín chỉ', 'học chế niên chế', 'tín chỉ tích lũy'
            ],

            'assessment_methods': [
                'kiểm tra thường xuyên', 'kiểm tra giữa kỳ', 'thi cuối kỳ',
                'bài tập lớn', 'tiểu luận', 'thực hành', 'thí nghiệm',
                'thuyết trình', 'báo cáo', 'dự án', 'khóa luận',
                'đồ án tốt nghiệp', 'thực tập tốt nghiệp', 'vấn đáp'
            ],

            'student_support_services': [
                'tư vấn tâm lý', 'hỗ trợ học tập', 'định hướng nghề nghiệp',
                'kết nối việc làm', 'hỗ trợ khởi nghiệp', 'tư vấn du học',
                'câu lạc bộ sinh viên', 'hoạt động ngoại khóa', 'thể thao',
                'văn nghệ', 'tình nguyện', 'cộng đồng'
            ],

            'administrative_documents': [
                'đơn xin nghỉ học', 'đơn chuyển ngành', 'đơn chuyển lớp',
                'đơn xin thôi học', 'đơn xin bảo lưu', 'đơn phúc khảo',
                'giấy xác nhận sinh viên', 'bảng điểm tạm thời', 'bản sao bằng cấp',
                'giấy chuyển trường', 'giấy giới thiệu thực tập', 'giấy nghỉ học'
            ],

            'academic_periods': [
                'học kỳ I', 'học kỳ II', 'học kỳ hè', 'học kỳ phụ',
                'năm học 2023-2024', 'năm học 2024-2025', 'khóa học',
                'khai giảng', 'bế giảng', 'nghỉ lễ', 'nghỉ tết',
                'thời khóa biểu', 'lịch thi', 'lịch học', 'lịch học bù'
            ],

            'lecturer_positions_extended': [
                'giảng viên chính', 'giảng viên cao cấp', 'giảng viên',
                'phó giáo sư', 'giáo sư', 'tiến sĩ', 'thạc sĩ',
                'trưởng khoa', 'phó khoa', 'trưởng bộ môn', 'phó bộ môn',
                'cố vấn học tập', 'giảng viên hướng dẫn', 'giảng viên thỉnh giảng',
                'nghiên cứu viên', 'chuyên viên', 'cán bộ giảng dạy'
            ],

            'international_terms': [
                'chương trình liên kết quốc tế', 'trao đổi sinh viên',
                'du học', 'học bổng quốc tế', 'đối tác nước ngoài',
                'hợp tác song phương', 'memorandum', 'MOU',
                'IELTS', 'TOEFL', 'TOEIC', 'chứng chỉ ngoại ngữ',
                'tiếng Anh', 'tiếng Nhật', 'tiếng Hàn', 'tiếng Trung'
            ],

            'library_services': [
                'mượn sách', 'trả sách', 'gia hạn', 'đặt chỗ',
                'tìm kiếm tài liệu', 'cơ sở dữ liệu', 'sách điện tử',
                'tạp chí điện tử', 'luận văn', 'luận án', 'khóa luận',
                'phòng đọc', 'khu vực học nhóm', 'máy tính tra cứu',
                'wifi thư viện', 'in ấn', 'photocopy'
            ],

            'quality_standards': [
                'chuẩn đầu ra chương trình', 'chuẩn đầu ra học phần',
                'mục tiêu giáo dục', 'chuẩn AUN-QA', 'chuẩn ABET',
                'kiểm định chất lượng', 'đánh giá ngoài', 'tự đánh giá',
                'báo cáo tự đánh giá', 'ma trận chuẩn đầu ra',
                'rubric đánh giá', 'thang điểm', 'tiêu chí đánh giá'
            ],

            'financial_support_types': [
                'học bổng khuyến khích học tập', 'học bổng xã hội',
                'học bổng tài trợ', 'hỗ trợ sinh viên khó khăn',
                'miễn giảm học phí', 'trả góp học phí', 'hoãn nộp học phí',
                'hỗ trợ đột xuất', 'quỹ hỗ trợ sinh viên', 'vay vốn sinh viên'
            ],

            'facility_types': [
                'phòng học lý thuyết', 'phòng thí nghiệm', 'phòng thực hành',
                'phòng máy tính', 'phòng CAD/CAM', 'phòng thiết kế',
                'xưởng cơ khí', 'phòng mô phỏng', 'studio',
                'hội trường', 'phòng hội nghị', 'sân thể thao',
                'sân bóng đá', 'sân bóng chuyền', 'phòng gym'
            ],

            'graduation_requirements': [
                'điều kiện tốt nghiệp', 'số tín chỉ tối thiểu',
                'điểm trung bình tích lũy', 'không nợ môn',
                'hoàn thành khóa luận', 'hoàn thành thực tập',
                'chứng chỉ ngoại ngữ', 'chứng chỉ tin học',
                'điểm rèn luyện', 'không bị kỷ luật'
            ],

            'academic_violations': [
                'vi phạm quy chế thi', 'gian lận trong thi cử',
                'đạo văn', 'sao chép bài làm', 'mang tài liệu vào phòng thi',
                'cảnh báo học tập', 'đình chỉ học tập', 'buộc thôi học',
                'kỷ luật khiển trách', 'kỷ luật cảnh cáo', 'kỷ luật đình chỉ'
            ],

            # 🚨 ENTITY PATTERNS MỚI CHO LƯƠNG THÙ LAO
            'salary_scales': [
                'bậc 1', 'bậc 2', 'bậc 3', 'bậc 4', 'bậc 5', 'bậc 6', 'bậc 7',
                'hạng I', 'hạng II', 'hạng III', 'hạng IV',
                'ngạch giảng viên', 'ngạch giảng viên chính', 'ngạch giảng viên cao cấp',
                'mã số 01.001', 'mã số 01.002', 'mã số 01.003',
                'hệ số 2.34', 'hệ số 3.0', 'hệ số 4.0', 'hệ số 6.2'
            ],

            'compensation_types': [
                'thù lao giảng dạy', 'thù lao nghiên cứu', 'thù lao hướng dẫn',
                'thù lao chấm thi', 'thù lao ra đề', 'thù lao thẩm định',
                'thù lao biên soạn', 'thù lao dịch thuật', 'thù lao tư vấn',
                'lương cơ bản', 'lương theo ngạch', 'lương theo vị trí việc làm',
                'phụ cấp trách nhiệm', 'phụ cấp thâm niên', 'phụ cấp khu vực',
                'phụ cấp độc hại', 'phụ cấp đặc thù', 'phụ cấp giảng dạy xa'
            ],

            'teaching_rates': [
                'hệ số 1.0', 'hệ số 1.2', 'hệ số 1.5', 'hệ số 2.0', 'hệ số 4.5',
                'giờ chuẩn', 'giờ vượt chuẩn', 'giờ thiếu chuẩn',
                'tiết lý thuyết', 'tiết thực hành', 'tiết thí nghiệm',
                'định mức 270 giờ', 'định mức 300 giờ', 'định mức 360 giờ',
                'hệ số thực hành 1.0', 'hệ số tiếng nước ngoài 4.5'
            ],

            'salary_amounts': [
                '15 triệu đồng', '13 triệu đồng', '10 triệu đồng', '5 triệu đồng',
                '2 triệu đồng', '2.4 triệu đồng', '500.000 đồng', '300.000 đồng',
                '15,000,000 VNĐ', '13,000,000 VNĐ', '10,000,000 VNĐ', '5,000,000 VNĐ',
                'giáo sư 15 triệu', 'phó giáo sư 13 triệu', 'tiến sĩ 10 triệu', 'thạc sĩ 5 triệu'
            ],

            'payment_methods': [
                'chuyển khoản', 'tiền mặt', 'thẻ ATM', 'tài khoản ngân hàng',
                'thanh toán qua ngân hàng', 'chuyển khoản tự động',
                'trả lương cuối tháng', 'trả lương đầu tháng', 'trả lương ngày 15',
                'phiếu lương', 'bảng thanh toán', 'slip lương'
            ],

            'benefit_types': [
                'bảo hiểm xã hội', 'bảo hiểm y tế', 'bảo hiểm thất nghiệp',
                'bảo hiểm tai nạn lao động', 'kinh phí công đoàn',
                'nghỉ phép năm', 'nghỉ lễ', 'nghỉ tết', 'nghỉ thai sản',
                'khám sức khỏe định kỳ', 'chăm sóc y tế', 'điều trị bệnh',
                'nghỉ dưỡng', 'du lịch', 'team building', 'gala dinner'
            ],

            'work_contracts': [
                'hợp đồng lao động', 'hợp đồng có thời hạn', 'hợp đồng không thời hạn',
                'hợp đồng thử việc', 'hợp đồng thỉnh giảng', 'hợp đồng cộng tác',
                'hợp đồng 1 năm', 'hợp đồng 2 năm', 'hợp đồng 3 năm',
                'giảng viên cơ hữu', 'giảng viên thỉnh giảng', 'cộng tác viên',
                'viên chức', 'công chức', 'người lao động'
            ],

            'salary_regulations': [
                'Quyết định 442/QĐ-ĐHBD', 'Quyết định 895/QĐ-ĐHBD', 
                'Quyết định 1733/QĐ-ĐHBD', 'Quyết định 101/QĐ-ĐHBD',
                'Quyết định 1153/QĐ-ĐHBD', 'Quyết định 2004/QĐ-ĐHBD',
                'QĐ 442', 'QĐ 895', 'QĐ 1733', 'QĐ 101', 'QĐ 1153', 'QĐ 2004',
                'Thông báo 746', 'TB 746', 'quy định thù lao', 'quy định lương'
            ],

            'academic_positions_salary': [
                'giáo sư - 15 triệu', 'phó giáo sư - 13 triệu',
                'tiến sĩ - 10 triệu', 'thạc sĩ - 5 triệu',
                'nhân sự nước ngoài - 10 triệu',
                'chủ tịch hội đồng - 500k', 'ủy viên hội đồng - 300k',
                'giảng viên thỉnh giảng', 'giảng viên cộng tác'
            ],

            'salary_calculation_terms': [
                'hệ số nhân', 'lương cơ sở', 'phụ cấp chức vụ', 'phụ cấp kiêm nhiệm',
                'thưởng theo tháng', 'thưởng theo quý', 'thưởng cuối năm',
                'tăng ca', 'làm thêm giờ', 'làm việc ngoài giờ',
                'khấu trừ thuế', 'khấu trừ bảo hiểm', 'thu nhập trước thuế',
                'thu nhập sau thuế', 'thu nhập thực nhận'
            ],
        }
    
    def load_model(self):
        """Load PhoBERT model with enhanced error handling"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available")
                
            model_name = "vinai/phobert-base"
            logger.info(f"Loading PhoBERT model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Only set fallback_mode to False if everything loaded successfully
            self.fallback_mode = False
            logger.info("✅ PhoBERT model loaded successfully for lecturers")
            
        except Exception as e:
            logger.warning(f"⚠️ PhoBERT not available, using enhanced fallback for lecturers: {str(e)}")
            self.tokenizer = None
            self.model = None
            self.fallback_mode = True  # Ensure fallback mode is set
    
    def classify_intent(self, query):
        """🚀 Main classify method - Enhanced intent classification với ensemble methods"""
        return self.enhanced_classify_intent(query)
    
    def enhanced_classify_intent(self, query, conversation_context=None):
        """🚀 Enhanced intent classification với ensemble methods"""
        if not query or not query.strip():
            return self._get_default_intent()
        
        # Normalize query
        if not self.normalizer:
            query_variants = [query, query.lower()]
            normalized_query = query.lower()
        else:
            normalized_query = self.normalizer.normalize_query(query)
            query_variants = self.normalizer.create_search_variants(query)
        
        logger.info(f"🧠 Enhanced Intent Classification: '{query}' -> normalized: '{normalized_query}'")
        
        # 🎯 Ensemble Classification
        ensemble_results = {}
        
        # Method 1: Enhanced Keyword Matching
        keyword_results = self._enhanced_keyword_classification(query_variants)
        ensemble_results['keyword_matching'] = keyword_results
        
        # Method 2: Pattern Matching
        pattern_results = self._pattern_based_classification(normalized_query)
        ensemble_results['pattern_matching'] = pattern_results
        
        # Method 3: Context Analysis
        context_results = self._context_aware_classification(normalized_query, conversation_context)
        ensemble_results['context_analysis'] = context_results
        
        # Method 4: PhoBERT Semantic (if available)
        if not self.fallback_mode and self.model:
            semantic_results = self._semantic_classification(normalized_query)
            ensemble_results['phobert_semantic'] = semantic_results
        
        # 🎯 Ensemble Fusion
        final_result = self._fuse_ensemble_results(ensemble_results, query, conversation_context)
        
        # 🎯 Multi-Intent Detection
        multi_intent_result = self._detect_multi_intent(query, final_result)
        
        # 🎯 Confidence Calibration
        calibrated_result = self._calibrate_intent_confidence(final_result, query, conversation_context)
        
        logger.info(f"🎯 Final Intent: {calibrated_result['intent']} (confidence: {calibrated_result['confidence']:.3f})")
        
        return calibrated_result
    
    def _enhanced_keyword_classification(self, query_variants):
        """🚀 Enhanced keyword matching với multiple variants"""
        intent_scores = defaultdict(float)
        
        for variant in query_variants:
            variant_lower = variant.lower().strip()
            
            for intent_name, config in self.intent_categories.items():
                # Regular keywords
                keyword_score = self._calculate_keyword_score(variant_lower, config['keywords'])
                
                # Boosters
                booster_score = self._calculate_booster_score(variant_lower, config.get('boosters', {}))
                
                total_score = keyword_score + booster_score
                intent_scores[intent_name] = max(intent_scores[intent_name], total_score)
        
        return dict(intent_scores)
    
    def _pattern_based_classification(self, query):
        """🆕 Pattern-based classification"""
        intent_scores = defaultdict(float)
        
        for intent_name, config in self.intent_categories.items():
            patterns = config.get('patterns', [])
            
            for pattern in patterns:
                try:
                    if re.search(pattern, query, re.IGNORECASE):
                        intent_scores[intent_name] += 0.8  # High score for pattern match
                        logger.debug(f"🎯 Pattern match: {intent_name} -> '{pattern}'")
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {pattern} - {e}")
        
        return dict(intent_scores)
    
    def _context_aware_classification(self, query, conversation_context):
        """🆕 Context-aware classification"""
        intent_scores = defaultdict(float)
        
        if not conversation_context:
            return dict(intent_scores)
        
        # Recent conversation boost
        recent_intents = [
            item.get('intent_info', {}).get('intent', '') 
            for item in conversation_context[-3:]
        ]
        
        intent_counter = Counter(recent_intents)
        for intent, count in intent_counter.items():
            if intent in self.intent_categories:
                # Boost score based on frequency and recency
                boost_score = count * 0.3 * (0.8 ** (len(recent_intents) - recent_intents[::-1].index(intent)))
                intent_scores[intent] += boost_score
        
        # Context indicators boost
        for intent_name, config in self.intent_categories.items():
            context_indicators = config.get('context_indicators', {})
            context_score = self._calculate_context_score(query, context_indicators)
            intent_scores[intent_name] += context_score
        
        return dict(intent_scores)
    
    def _semantic_classification(self, query):
        """🆕 PhoBERT semantic classification"""
        intent_scores = defaultdict(float)
        
        try:
            query_embedding = self.encode_text(query)
            if query_embedding is None:
                return dict(intent_scores)
            
            for intent_name, config in self.intent_categories.items():
                # Create intent representation from description and keywords
                intent_text = f"{config['description']} {' '.join(config['keywords'][:5])}"
                intent_embedding = self.encode_text(intent_text)
                
                if intent_embedding is not None:
                    similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
                    intent_scores[intent_name] = float(similarity)
        
        except Exception as e:
            logger.warning(f"Semantic classification failed: {e}")
        
        return dict(intent_scores)
    
    def _fuse_ensemble_results(self, ensemble_results, query, conversation_context):
        """🚀 Fuse ensemble results với weighted voting"""
        final_scores = defaultdict(float)
        
        # Weighted combination
        for method, weight in self.ensemble_weights.items():
            if method in ensemble_results:
                method_scores = ensemble_results[method]
                for intent, score in method_scores.items():
                    final_scores[intent] += score * weight
        
        # Find best intent
        if final_scores:
            best_intent = max(final_scores.items(), key=lambda x: x[1])
            intent_name, confidence = best_intent
            
            # Get intent config
            intent_config = self.intent_categories.get(intent_name, {})
            
            return {
                'intent': intent_name,
                'confidence': confidence,
                'description': intent_config.get('description', 'Unknown'),
                'normalized_query': query,
                'ensemble_scores': dict(final_scores),
                'method_breakdown': ensemble_results,
                'lecturer_optimized': True
            }
        
        return self._get_default_intent()
    
    def _detect_multi_intent(self, query, primary_intent_result):
        """🆕 Detect multiple intents in single query"""
        multi_intents = []
        
        for pattern_type, patterns in self.multi_intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    multi_intents.append({
                        'type': pattern_type,
                        'pattern': pattern,
                        'groups': match.groupdict()
                    })
        
        if multi_intents:
            primary_intent_result['multi_intent_detected'] = True
            primary_intent_result['multi_intent_details'] = multi_intents
            # Slightly reduce confidence for multi-intent queries
            primary_intent_result['confidence'] *= 0.9
        
        return primary_intent_result
    
    def _calibrate_intent_confidence(self, intent_result, query, conversation_context):
        """🆕 Calibrate confidence based on various factors"""
        base_confidence = intent_result['confidence']
        calibration_factor = 1.0
        
        # Apply calibration rules
        for factor_name, factor_value in self.confidence_calibration['calibration_factors'].items():
            if self._check_calibration_condition(factor_name, intent_result, query, conversation_context):
                calibration_factor *= factor_value
                logger.debug(f"🎯 Confidence calibration: {factor_name} -> {factor_value}")
        
        calibrated_confidence = min(1.0, base_confidence * calibration_factor)
        intent_result['confidence'] = calibrated_confidence
        intent_result['calibration_factor'] = calibration_factor
        
        return intent_result
    
    def _check_calibration_condition(self, factor_name, intent_result, query, conversation_context):
        """Check if calibration condition is met"""
        if factor_name == 'multiple_keyword_match':
            # Check if multiple keywords matched
            return len([kw for kw in self.intent_categories[intent_result['intent']]['keywords'] 
                       if kw in query.lower()]) > 1
        
        elif factor_name == 'exact_phrase_match':
            # Check for exact phrase matches
            return any(kw == query.lower().strip() 
                      for kw in self.intent_categories[intent_result['intent']]['keywords'])
        
        elif factor_name == 'context_continuity':
            # Check if intent continues from recent conversation
            if conversation_context:
                recent_intents = [item.get('intent_info', {}).get('intent', '') 
                                for item in conversation_context[-2:]]
                return intent_result['intent'] in recent_intents
        
        return False
    
    def _calculate_keyword_score(self, query, keywords):
        """Enhanced keyword scoring"""
        if not keywords:
            return 0
        
        score = 0
        matched_keywords = 0
        
        for keyword in keywords:
            if keyword in query:
                matched_keywords += 1
                if keyword == query:  # Exact match
                    score += 2.0
                elif query.startswith(keyword) or query.endswith(keyword):
                    score += 1.5
                else:
                    score += 1.0
        
        # Normalize score and apply bonus for multiple matches
        normalized_score = score / len(keywords)
        if matched_keywords > 1:
            normalized_score *= 1.2  # Bonus for multiple keyword matches
        
        return min(1.0, normalized_score)
    
    def _calculate_booster_score(self, query, boosters):
        """Calculate booster score"""
        if not boosters:
            return 0
        
        booster_score = 0
        boost_factor = boosters.get('boost_factor', 1.0)
        
        for booster_type, booster_values in boosters.items():
            if booster_type == 'boost_factor':
                continue
            
            if isinstance(booster_values, list):
                for booster_value in booster_values:
                    if booster_value in query:
                        booster_score += 0.2 * boost_factor
        
        return min(0.5, booster_score)  # Cap booster contribution
    
    def _calculate_context_score(self, query, context_indicators):
        """Calculate context score from indicators"""
        if not context_indicators:
            return 0
        
        context_score = 0
        
        for indicator_type, indicators in context_indicators.items():
            if isinstance(indicators, list):
                matched_indicators = sum(1 for indicator in indicators if indicator in query)
                if matched_indicators > 0:
                    context_score += matched_indicators * 0.1
        
        return min(0.3, context_score)
    
    def encode_text(self, text):
        """Encode text using PhoBERT with error handling"""
        if self.fallback_mode or not self.model or not self.tokenizer:
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
            
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return None
    
    def extract_entities(self, query):
        """Enhanced entity extraction for lecturers WITH PERSONAL CONTEXT"""
        if not query:
            return {}
            
        query_lower = query.lower()
        entities = {}
        
        # Extract personal context indicators
        personal_pronouns_found = []
        for pronoun in self.entity_patterns['personal_pronouns']:
            if pronoun in query_lower:
                personal_pronouns_found.append(pronoun)
        
        if personal_pronouns_found:
            entities['personal_context'] = personal_pronouns_found
            entities['has_personal_context'] = True
            entities['personal_context_confidence'] = 0.9
        
        # Extract schedule context
        schedule_contexts_found = []
        for context in self.entity_patterns['schedule_contexts']:
            if context in query_lower:
                schedule_contexts_found.append(context)
        
        if schedule_contexts_found:
            entities['schedule_context'] = schedule_contexts_found
            entities['schedule_context_confidence'] = 0.8
        
        # Extract departments with confidence
        for dept in self.entity_patterns['lecturer_departments']:
            if dept in query_lower:
                entities['department'] = dept
                entities['department_confidence'] = 1.0 if dept == query_lower else 0.9
                break
        
        # Extract emotions with lecturer-specific intensity
        emotion_intensity = 0
        detected_emotion = None
        for emotion in self.entity_patterns['emotions']:
            if emotion in query_lower:
                if emotion in ['cần gấp', 'khẩn cấp', 'urgent']:
                    emotion_intensity = 0.9
                elif emotion in ['quan trọng', 'ưu tiên']:
                    emotion_intensity = 0.8
                else:
                    emotion_intensity = 0.6
                detected_emotion = emotion
                break
        
        if detected_emotion:
            entities['emotion'] = detected_emotion
            entities['emotion_intensity'] = emotion_intensity
        
        return entities
    
    def analyze_query(self, query):
        """Comprehensive query analysis with safe fallbacks for lecturers"""
        start_time = time.time()
        
        try:
            intent_result = self.classify_intent(query)
            entities = self.extract_entities(query)
            
            analysis = {
                'intent': intent_result,
                'entities': entities,
                'query_length': len(query) if query else 0,
                'word_count': len(query.split()) if query else 0,
                'is_question': '?' in query if query else False,
                'urgency': self._detect_lecturer_urgency(query),
                'complexity': self._assess_lecturer_complexity(query),
                'sentiment': self._detect_lecturer_sentiment(query, entities),
                'processing_time': time.time() - start_time,
                'fallback_mode': self.fallback_mode,
                'lecturer_optimized': True
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            return {
                'intent': self._get_default_intent(),
                'entities': {},
                'query_length': len(query) if query else 0,
                'word_count': len(query.split()) if query else 0,
                'is_question': False,
                'urgency': 'normal',
                'complexity': 'simple',
                'sentiment': 'neutral',
                'processing_time': time.time() - start_time,
                'fallback_mode': True,
                'lecturer_optimized': True
            }
    
    def _detect_lecturer_urgency(self, query):
        """Detect urgency level specifically for lecturers"""
        if not query:
            return 'normal'
            
        urgent_words = ['gấp', 'urgent', 'khẩn cấp', 'cần ngay', 'hạn cuối', 'deadline']
        medium_urgent_words = ['sớm', 'nhanh chóng', 'ưu tiên', 'quan trọng']
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in urgent_words):
            return 'high'
        elif any(word in query_lower for word in medium_urgent_words):
            return 'medium'
        else:
            return 'normal'
    
    def _assess_lecturer_complexity(self, query):
        """Assess query complexity for lecturers"""
        if not query:
            return 'simple'
            
        word_count = len(query.split())
        question_marks = query.count('?')
        
        # Lecturer-specific: Consider technical terms
        technical_terms = ['ngân hàng đề thi', 'kê khai nhiệm vụ', 'tạp chí khoa học']
        has_technical = any(term in query.lower() for term in technical_terms)
        
        if (word_count > 20 or question_marks > 1) or has_technical:
            return 'complex'
        elif word_count > 10:
            return 'medium'
        else:
            return 'simple'
    
    def _detect_lecturer_sentiment(self, query, entities):
        """Detect overall sentiment for lecturers"""
        if not query:
            return 'neutral'
            
        positive_words = ['tốt', 'hay', 'thích', 'muốn', 'quan tâm', 'hỗ trợ']
        negative_words = ['khó khăn', 'lo lắng', 'không', 'chán', 'tệ', 'vấn đề']
        urgent_words = ['gấp', 'khẩn cấp', 'cần ngay']
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        urgent_count = sum(1 for word in urgent_words if word in query_lower)
        
        # Factor in emotional entities
        if entities and 'emotion' in entities:
            emotion = entities['emotion']
            if emotion in ['quan trọng', 'ưu tiên']:
                positive_count += 1
            elif emotion in ['lo lắng', 'khó khăn']:
                negative_count += 2
            elif emotion in ['cần gấp', 'khẩn cấp']:
                urgent_count += 2
        
        if urgent_count > 0:
            return 'urgent'
        elif positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_default_intent(self):
        """Default intent when no match found"""
        return {
            'intent': 'general',
            'confidence': 0.3,
            'description': 'Câu hỏi chung',
            'lecturer_optimized': True
        }
    
    def get_system_status(self):
        """Get PhoBERT system status for lecturers WITH enhanced features"""
        return {
            'model_loaded': bool(self.model),
            'fallback_mode': self.fallback_mode,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'device': str(self.device) if self.device else 'cpu',
            'intents_available': len(self.intent_categories),
            'enhanced_intents': [
                'document_reference', 'deadline_temporal', 'contact_responsibility',
                'technical_specification', 'compliance_consequence', 'process_sequence',
                'authorization_approval', 'document_comparison'
            ],
            'original_intents': [
                'greeting', 'bank_exam_questions', 'annual_task_declaration',
                'academic_journal', 'competition_awards', 'personal_schedule', 'personal_info'
            ],
            'lecturer_optimized': True,
            'ensemble_methods': list(self.ensemble_weights.keys()),
            'features': [
                'ensemble_classification',
                'multi_intent_detection',
                'context_aware_boosting',
                'confidence_calibration',
                'pattern_matching',
                'enhanced_keyword_matching',
                'vietnamese_normalization',
                'personal_context_detection',
                'semantic_similarity',
                'lecturer_specific_intents',
                'document_reference_detection',
                'deadline_temporal_analysis',
                'compliance_consequence_analysis'
            ]
        }