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
                    'phần trăm', 'khối lượng', 'quy định học tập'
                ],
                'confidence_threshold': 0.3,
                'description': 'Quy định học tập và điểm số',
                'patterns': [r'(?:điểm|học lại).*(?:như thế nào|quy định)', r'(?:tín chỉ|chuyển đổi).*(?:phần trăm|tối thiểu)'],
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
                'description': 'Thi đua khen thưởng',
                'patterns': [r'thi\s+đua.*khen\s+thưởng', r'(?:danh hiệu|bằng khen)'],
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
                'keywords': ['cơ sở vật chất', 'phòng học', 'thiết bị', 'thư viện', 'ký túc xá', 'wifi'],
                'confidence_threshold': 0.4,
                'description': 'Cơ sở vật chất và tiện ích', 
                'patterns': [r'(?:phòng học|thiết bị).*(?:như thế nào|ra sao)', r'cơ sở vật chất.*trường'],
            }
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
                'năm học 2023-2024', 'học kỳ I', 'học kỳ II', 'trước ngày', 'hạn cuối'
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
            ]    
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