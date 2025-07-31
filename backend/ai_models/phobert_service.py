import torch
import numpy as np
import logging
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import time
from typing import Dict, List, Tuple, Optional
from django.conf import settings

# Try to import transformers, fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. PhoBERT will use fallback mode.")

# Try to import sentence-transformers for fine-tuned model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Fine-tuned model will not be available.")

logger = logging.getLogger(__name__)

class PhoBERTIntentClassifier:
    """
    🚀 Enhanced PhoBERT-based Intent Classification with Fine-tuned Model Support
    
    Features:
    - Fine-tuned model integration with fallback to base model
    - Simplified to 6-7 mega-intents instead of 25+ specific intents
    - Keywords now loaded automatically from CSV data
    - Ensemble methods (PhoBERT + Keyword + Pattern + Context)
    - Multi-intent detection
    - Context-aware classification
    - Confidence calibration
    - Vietnamese normalization support
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TRANSFORMERS_AVAILABLE else None
        self.tokenizer = None
        self.model = None
        self.fine_tuned_model = None  # 🚀 NEW: Fine-tuned sentence transformer model
        
        # ESSENTIAL: Set fallback_mode FIRST
        self.fallback_mode = True  # Default to fallback mode
        
        # Fine-tuned model path
        self.fine_tuned_model_path = os.path.join(settings.BASE_DIR, 'fine_tuned_phobert')
        
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
        
        # 🆕 Intent confidence calibration - 🚀 FIXED: Max confidence is 1.0
        self.confidence_calibration = self._initialize_confidence_calibration()
        
        # Initialize simplified mega-intent categories
        self.intent_categories = self._initialize_mega_intents()
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
        
        # 🚀 NEW: Load fine-tuned model first, then fallback to base model
        self._load_models_with_priority()

    def _create_dummy_normalizer(self):
        """Create dummy normalizer if import fails"""
        class DummyNormalizer:
            def normalize_query(self, query):
                return query.lower().strip()
            
            def create_search_variants(self, query):
                return [query, query.lower()]
        
        return DummyNormalizer()
    
    def _load_models_with_priority(self):
        """🚀 NEW: Load models with priority: Fine-tuned > Base model > Fallback"""
        
        # Priority 1: Try to load fine-tuned sentence transformer model
        if self._load_fine_tuned_model():
            logger.info("✅ Using fine-tuned PhoBERT model for enhanced retrieval")
            self.fallback_mode = False
            return
        
        # Priority 2: Try to load base PhoBERT model
        if self._load_base_model():
            logger.info("✅ Using base PhoBERT model")
            self.fallback_mode = False
            return
        
        # Priority 3: Fallback mode
        logger.warning("⚠️ Using enhanced fallback mode (keyword-based) for lecturers")
        self.fallback_mode = True

    def _load_fine_tuned_model(self):
        """🚀 NEW: Load fine-tuned sentence transformer model"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("📦 sentence-transformers not available, skipping fine-tuned model")
                return False
            
            if not os.path.exists(self.fine_tuned_model_path):
                logger.info(f"📁 Fine-tuned model not found at {self.fine_tuned_model_path}")
                return False
            
            # Check if it's a valid sentence-transformers model
            config_path = os.path.join(self.fine_tuned_model_path, 'config.json')
            model_bin_path = os.path.join(self.fine_tuned_model_path, 'pytorch_model.bin')
            model_safetensors_path = os.path.join(self.fine_tuned_model_path, 'model.safetensors')

            if not os.path.exists(config_path) or not (os.path.exists(model_bin_path) or os.path.exists(model_safetensors_path)):
                logger.warning(f"⚠️ Fine-tuned model directory incomplete at {self.fine_tuned_model_path}. Missing config.json or model weights (.bin/.safetensors).")
                return False
            
            logger.info(f"📥 Loading fine-tuned sentence transformer model from: {self.fine_tuned_model_path}")
            self.fine_tuned_model = SentenceTransformer(self.fine_tuned_model_path)
            
            # Test the model with a simple encoding
            test_text = "Test encoding for PhoBERT"
            test_embedding = self.fine_tuned_model.encode([test_text])
            
            if test_embedding is not None and len(test_embedding) > 0:
                logger.info("✅ Fine-tuned PhoBERT model loaded and tested successfully")
                return True
            else:
                logger.error("❌ Fine-tuned model test failed")
                self.fine_tuned_model = None
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load fine-tuned model: {str(e)}")
            self.fine_tuned_model = None
            return False

    def _load_base_model(self):
        """Load base PhoBERT model as fallback"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                return False
                
            model_name = "vinai/phobert-base"
            logger.info(f"📥 Loading base PhoBERT model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Base PhoBERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Base PhoBERT not available: {str(e)}")
            self.tokenizer = None
            self.model = None
            return False

    def load_model(self):
        """Legacy method for backward compatibility"""
        return self._load_models_with_priority()
    
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
                'boost_intents': ['cong_viec_giang_vien', 'tra_cuu_thong_tin_ca_nhan'],
                'boost_factor': 1.3
            },
            'personal_context_boost': {
                'personal_keywords': ['tôi', 'của tôi', 'cho tôi', 'với tôi'],
                'boost_intents': ['tra_cuu_thong_tin_ca_nhan'],
                'boost_factor': 1.5
            },
            'urgency_boost': {
                'urgency_keywords': ['gấp', 'khẩn cấp', 'urgent', 'cần ngay'],
                'boost_intents': ['cong_viec_giang_vien', 'hoi_dap_chung'],
                'boost_factor': 1.4
            },
            'recent_conversation_boost': {
                # Boost intents that appeared in recent conversation
                'decay_factor': 0.8,  # Each turn back reduces boost by 20%
                'max_history': 3
            }
        }
    
    def _initialize_confidence_calibration(self):
        """🆕 Confidence calibration parameters - 🚀 FIXED: Ensure max confidence is 1.0"""
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
                'multiple_keyword_match': 1.0,   # 🚀 FIXED: Reduced from 1.2 to 1.0
                'exact_phrase_match': 1.0,       # 🚀 FIXED: Reduced from 1.5 to 1.0  
                'semantic_similarity_high': 1.0, # 🚀 FIXED: Reduced from 1.3 to 1.0
                'context_continuity': 1.0,       # 🚀 FIXED: Reduced from 1.2 to 1.0
                'multi_intent_detected': 0.9     # Slight reduction for multi-intent
            },
            'max_confidence': 1.0  # 🚀 NEW: Hard cap at 1.0
        }
    
    def _initialize_mega_intents(self):
        """🚀 SIMPLIFIED: Only 6-7 mega-intents for broad categorization"""
        return {
            # ===== MEGA-INTENT 1: Công việc giảng viên =====
            'cong_viec_giang_vien': {
                'keywords': [
                    'ngân hàng đề thi', 'ngan hang de thi', 'đề thi', 'de thi', 'báo cáo đề thi',
                    'kê khai nhiệm vụ', 'ke khai nhiem vu', 'nhiệm vụ năm học', 'giờ chuẩn', 'gio chuan',
                    'tạp chí', 'tap chi', 'tạp chí khoa học', 'bài viết', 'nghiên cứu',
                    'thi đua', 'thi dua', 'khen thưởng', 'khen thuong', 'danh hiệu', 'bằng khen',
                    'thù lao', 'thu lao', 'lương', 'phụ cấp', 'hệ số', 'chế độ',
                    'phúc lợi', 'đãi ngộ', 'bảo hiểm', 'nghỉ phép', 'chấm thi',
                    'giảng viên', 'giang vien', 'thầy', 'cô', 'giảng dạy', 'dạy',
                    'nghiên cứu khoa học', 'hội nghị', 'seminar', 'workshop'
                ],
                'confidence_threshold': 0.4,
                'description': 'Công việc và quy định dành cho giảng viên',
                'patterns': [
                    r'(?:báo cáo|nộp|gửi).*(?:đề thi|ngân hàng)',
                    r'(?:kê khai|nhiệm vụ).*năm học',
                    r'(?:thi đua|khen thưởng).*giảng viên',
                    r'(?:thù lao|lương|phụ cấp).*(?:giảng viên|dạy)',
                    r'(?:nghiên cứu|tạp chí).*khoa học'
                ]
            },
            
            # ===== MEGA-INTENT 2: Quy định học vụ =====
            'quy_dinh_hoc_vu': {
                'keywords': [
                    'điểm', 'học lại', 'nâng điểm', 'điểm trung bình', 'dtb', 'tính điểm',
                    'đạt', 'không đạt', 'tín chỉ', 'chuyển đổi', 'công nhận',
                    'phần trăm', 'khối lượng', 'quy định học tập', 'xử lý học vụ',
                    'cảnh báo học tập', 'đình chỉ', 'buộc thôi học', 'vi phạm quy định',
                    'tốt nghiệp', 'lễ tốt nghiệp', 'điều kiện tốt nghiệp', 'bằng cấp',
                    'phúc khảo', 'chấm lại', 'khiếu nại điểm', 'bài thi'
                ],
                'confidence_threshold': 0.3,
                'description': 'Quy định học tập, điểm số và tốt nghiệp',
                'patterns': [
                    r'(?:điểm|học lại|quy định).*(?:như thế nào|quy định)',
                    r'(?:tín chỉ|chuyển đổi).*(?:phần trăm|tối thiểu)',
                    r'(?:tốt nghiệp|bằng cấp|văn bằng)',
                    r'(?:vi phạm|kỷ luật).*học tập'
                ]
            },
            
            # ===== MEGA-INTENT 3: Tuyển sinh và nhập học =====
            'tuyen_sinh_nhap_hoc': {
                'keywords': [
                    'đăng ký', 'dang ky', 'tuyển sinh', 'tuyen sinh', 'xét tuyển', 'xet tuyen',
                    'nộp hồ sơ', 'nop ho so', 'thủ tục', 'thu tuc', 'nhập học', 'nhap hoc',
                    'hồ sơ tuyển sinh', 'điều kiện tuyển sinh', 'phương thức tuyển sinh',
                    'chỉ tiêu', 'chi tieu', 'ngành học', 'nganh hoc', 'chuyên ngành',
                    'đăng ký môn học', 'rút môn', 'thêm môn', 'thay đổi đăng ký',
                    'khai học', 'thôi học', 'chuyển ngành', 'chuyển lớp'
                ],
                'confidence_threshold': 0.4,
                'description': 'Tuyển sinh, nhập học và đăng ký môn học',
                'patterns': [
                    r'(?:đăng ký|tuyển sinh).*(?:như thế nào|thủ tục)',
                    r'(?:hồ sơ|thủ tục).*(?:tuyển sinh|nhập học)',
                    r'(?:chuyển|rút|thêm).*môn',
                    r'(?:ngành|chuyên ngành).*(?:nào|gì)'
                ]
            },
            
            # ===== MEGA-INTENT 4: Tài chính và học phí =====
            'tai_chinh_hoc_phi': {
                'keywords': [
                    'học phí', 'hoc phi', 'chi phí', 'chi phi', 'tiền', 'thanh toán', 'miễn giảm',
                    'phí dịch vụ', 'phi dich vu', 'học bổng', 'hoc bong', 'hỗ trợ', 'ho tro',
                    'khó khăn', 'kho khan', 'ưu đãi', 'uu dai', 'vay vốn', 'vay von',
                    'trả góp', 'tra gop', 'hoãn nộp', 'hoan nop', 'quỹ hỗ trợ',
                    'tài chính', 'tai chinh', 'ngân hàng', 'ngan hang', 'chuyển khoản'
                ],
                'confidence_threshold': 0.4,
                'description': 'Học phí, tài chính và hỗ trợ học bổng',
                'patterns': [
                    r'(?:học phí|chi phí).*(?:bao nhiêu|như thế nào)',
                    r'(?:thanh toán|nộp).*học phí',
                    r'(?:học bổng|hỗ trợ).*(?:tài chính|học phí)',
                    r'(?:miễn giảm|ưu đãi).*(?:học phí|chi phí)'
                ]
            },
            
            # ===== MEGA-INTENT 5: Hỏi đáp chung =====
            'hoi_dap_chung': {
                'keywords': [
                    'trường đại học bình dương', 'bdu', 'đại học bình dương',
                    'phòng', 'khoa', 'bộ môn', 'cơ cấu tổ chức', 'đơn vị',
                    'cơ sở vật chất', 'co so vat chat', 'phòng học', 'thiết bị', 'thiet bi',
                    'thư viện', 'thu vien', 'ký túc xá', 'ky tuc xa', 'căn tin', 'can tin',
                    'wifi', 'phòng thí nghiệm', 'sân thể thao', 'bãi xe',
                    'chương trình đào tạo', 'chuong trinh dao tao', 'chất lượng', 'chat luong',
                    'đánh giá', 'danh gia', 'kiểm định', 'kiem dinh', 'akc',
                    'thông tin chung', 'thong tin chung', 'giới thiệu', 'gioi thieu'
                ],
                'confidence_threshold': 0.3,
                'description': 'Thông tin chung về trường và cơ sở vật chất',
                'patterns': [
                    r'(?:trường|bdu|đại học bình dương).*(?:gì|như thế nào|ở đâu)',
                    r'(?:phòng|khoa|bộ môn).*(?:chức năng|nhiệm vụ)',
                    r'(?:cơ sở vật chất|phòng học|thiết bị)',
                    r'(?:chương trình|chất lượng|đánh giá).*đào tạo'
                ]
            },
            
            # ===== MEGA-INTENT 6: Tra cứu thông tin cá nhân =====
            'tra_cuu_thong_tin_ca_nhan': {
                'keywords': [
                    'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
                    'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'tôi là ai', 'toi la ai',
                    'thông tin của tôi', 'thong tin cua toi', 'email của tôi', 'chức danh của tôi',
                    'lịch giảng dạy', 'lich giang day', 'lịch học', 'lich hoc', 'schedule',
                    'hôm nay', 'hom nay', 'ngày mai', 'ngay mai', 'tuần này', 'tuan nay',
                    'lớp của tôi', 'lop cua toi', 'môn của tôi', 'mon cua toi'
                ],
                'confidence_threshold': 0.3,
                'description': 'Tra cứu thông tin cá nhân và lịch giảng dạy',
                'patterns': [
                    r'(?:lịch|tkb|thời khóa biểu).*(?:tôi|của tôi)',
                    r'(?:tôi|mình).*(?:dạy|giảng|làm)',
                    r'(?:tôi|mình)\s+là\s+ai',
                    r'(?:thông tin|email).*của\s+tôi'
                ]
            },
            
            # ===== MEGA-INTENT 7: Chào hỏi =====
            'greeting': {
                'keywords': ['xin chào', 'hello', 'hi', 'chào thầy', 'chào cô', 'halo', 'chào', 'hey'],
                'confidence_threshold': 0.6,
                'description': 'Chào hỏi',
                'patterns': [r'(?:xin\s+)?chào\s+(?:thầy|cô)', r'hello|hi|hey'],
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
                'năm học', 'học kỳ I', 'học kỳ II', 'học kỳ III', 'trước ngày', 'hạn cuối'
            ],
            'lecturer_activities': [
                'giảng dạy', 'nghiên cứu khoa học', 'phục vụ cộng đồng', 'thi đua', 'khen thưởng'
            ],
            'emotions': [
                'cần gấp', 'khẩn cấp', 'urgent', 'quan trọng', 'lo lắng', 'khó khăn'
            ]
        }
    
    def classify_intent(self, query):
        """🚀 Main classify method - Enhanced intent classification với ensemble methods"""
        return self.enhanced_classify_intent(query)
    
    def enhanced_classify_intent(self, query, conversation_context=None):
        """🚀 Enhanced intent classification với ensemble methods và fine-tuned model"""
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
        
        # Method 4: Semantic Classification (Fine-tuned > Base model > Skip)
        if self.fine_tuned_model:
            # 🚀 NEW: Use fine-tuned model for semantic classification
            semantic_results = self._fine_tuned_semantic_classification(normalized_query)
            ensemble_results['phobert_semantic'] = semantic_results
        elif not self.fallback_mode and self.model:
            # Use base model
            semantic_results = self._semantic_classification(normalized_query)
            ensemble_results['phobert_semantic'] = semantic_results
        
        # 🎯 Ensemble Fusion
        final_result = self._fuse_ensemble_results(ensemble_results, query, conversation_context)
        
        # 🎯 Multi-Intent Detection
        multi_intent_result = self._detect_multi_intent(query, final_result)
        
        # 🎯 Confidence Calibration - 🚀 FIXED: Ensure confidence <= 1.0
        calibrated_result = self._calibrate_intent_confidence(final_result, query, conversation_context)
        
        logger.info(f"🎯 Final Intent: {calibrated_result['intent']} (confidence: {calibrated_result['confidence']:.3f})")
        
        return calibrated_result

    def _fine_tuned_semantic_classification(self, query):
        """🚀 NEW: Use fine-tuned sentence transformer for semantic classification"""
        intent_scores = defaultdict(float)
        
        try:
            if not self.fine_tuned_model:
                return dict(intent_scores)
            
            # Encode the query using fine-tuned model
            query_embedding = self.fine_tuned_model.encode([query])
            
            if query_embedding is None or len(query_embedding) == 0:
                return dict(intent_scores)
            
            # Compare with intent descriptions and keywords
            for intent_name, config in self.intent_categories.items():
                # Create intent representation from description and keywords
                intent_text = f"{config['description']} {' '.join(config['keywords'][:5])}"
                intent_embedding = self.fine_tuned_model.encode([intent_text])
                
                if intent_embedding is not None and len(intent_embedding) > 0:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
                    intent_scores[intent_name] = float(similarity)
            
            logger.debug(f"🚀 Fine-tuned semantic scores: {dict(intent_scores)}")
            
        except Exception as e:
            logger.warning(f"Fine-tuned semantic classification failed: {e}")
        
        return dict(intent_scores)
    
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
        
        return dict(intent_scores)
    
    def _semantic_classification(self, query):
        """🆕 Base PhoBERT semantic classification"""
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
            logger.warning(f"Base semantic classification failed: {e}")
        
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
        """🆕 Confidence calibration - 🚀 FIXED: Ensure confidence never exceeds 1.0"""
        base_confidence = intent_result['confidence']
        calibration_factor = 1.0
        
        # Apply calibration rules
        for factor_name, factor_value in self.confidence_calibration['calibration_factors'].items():
            if self._check_calibration_condition(factor_name, intent_result, query, conversation_context):
                calibration_factor *= factor_value
                logger.debug(f"🎯 Confidence calibration: {factor_name} -> {factor_value}")
        
        # 🚀 CRITICAL FIX: Ensure confidence never exceeds max_confidence (1.0)
        max_confidence = self.confidence_calibration['max_confidence']
        calibrated_confidence = min(max_confidence, base_confidence * calibration_factor)
        
        intent_result['confidence'] = calibrated_confidence
        intent_result['calibration_factor'] = calibration_factor
        intent_result['confidence_capped'] = calibrated_confidence == max_confidence
        
        # 🚀 ADDITIONAL: Log if confidence was capped
        if calibrated_confidence == max_confidence and base_confidence * calibration_factor > max_confidence:
            logger.info(f"🛡️ Confidence capped: {base_confidence * calibration_factor:.3f} -> {max_confidence}")
        
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
    
    def encode_text(self, text):
        """Encode text using available model (fine-tuned > base > None)"""
        
        # Priority 1: Use fine-tuned model
        if self.fine_tuned_model:
            try:
                embeddings = self.fine_tuned_model.encode([text])
                return embeddings.reshape(1, -1) if embeddings is not None else None
            except Exception as e:
                logger.error(f"Error encoding with fine-tuned model: {str(e)}")
        
        # Priority 2: Use base model
        if not self.fallback_mode and self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      padding=True, truncation=True, max_length=256)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.pooler_output
                
                return embeddings.cpu().numpy()
            except Exception as e:
                logger.error(f"Error encoding text with base model: {str(e)}")
        
        # Priority 3: No encoding available
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
                'lecturer_optimized': True,
                'fine_tuned_model_used': bool(self.fine_tuned_model)  # 🚀 NEW
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
                'lecturer_optimized': True,
                'fine_tuned_model_used': False  # 🚀 NEW
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
        
        # Lecturer-specific: Consider technical terms from mega-intents
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
            'intent': 'hoi_dap_chung',
            'confidence': 0.3,
            'description': 'Hỏi đáp chung',
            'lecturer_optimized': True,
            'fine_tuned_model_used': bool(self.fine_tuned_model)  # 🚀 NEW
        }
    
    def get_system_status(self):
        """Get PhoBERT system status for lecturers WITH fine-tuned model info"""
        return {
            'model_loaded': bool(self.model),
            'fine_tuned_model_loaded': bool(self.fine_tuned_model),  # 🚀 NEW
            'fine_tuned_model_path': self.fine_tuned_model_path,     # 🚀 NEW
            'model_priority': 'fine_tuned' if self.fine_tuned_model else 'base' if self.model else 'fallback',  # 🚀 NEW
            'fallback_mode': self.fallback_mode,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,  # 🚀 NEW
            'device': str(self.device) if self.device else 'cpu',
            'intents_available': len(self.intent_categories),
            'mega_intents': list(self.intent_categories.keys()),
            'lecturer_optimized': True,
            'ensemble_methods': list(self.ensemble_weights.keys()),
            'confidence_calibration': {  # 🚀 NEW: Show confidence limits
                'max_confidence': self.confidence_calibration['max_confidence'],
                'calibration_factors': self.confidence_calibration['calibration_factors']
            },
            'features': [
                'simplified_mega_intent_classification',
                'csv_auto_keyword_loading',
                'ensemble_classification',
                'multi_intent_detection',
                'context_aware_boosting',
                'confidence_calibration_with_caps',  # 🚀 UPDATED
                'pattern_matching',
                'enhanced_keyword_matching',
                'vietnamese_normalization',
                'personal_context_detection',
                'semantic_similarity',
                'lecturer_specific_intents',
                'fine_tuned_model_support',  # 🚀 NEW
                'model_priority_system',     # 🚀 NEW
                'confidence_overflow_protection'  # 🚀 NEW
            ]
        }