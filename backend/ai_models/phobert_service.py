import torch
import numpy as np
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity
import time

# Try to import transformers, fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. PhoBERT will use fallback mode.")

logger = logging.getLogger(__name__)

class PhoBERTIntentClassifier:
    """Enhanced PhoBERT-based Intent Classification for BDU Lecturers - COMPLETE VERSION"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TRANSFORMERS_AVAILABLE else None
        self.tokenizer = None
        self.model = None
        
        # ESSENTIAL: Set fallback_mode FIRST
        self.fallback_mode = True  # Default to fallback mode
        
        # Initialize components
        self.intent_categories = self._initialize_lecturer_intents()
        self.entity_patterns = self._initialize_lecturer_entities()
        
        # ✅ THÊM: Initialize normalizer BEFORE model loading
        self.normalizer = None
        try:
            from .vietnamese_normalizer import VietnameseNormalizer
            self.normalizer = VietnameseNormalizer()
            print("✅ Vietnamese Normalizer initialized successfully")
        except ImportError as e:
            print(f"❌ Failed to import VietnameseNormalizer: {e}")
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
    
    def _initialize_lecturer_intents(self):
        """Comprehensive intent categories specifically for BDU lecturers"""
        return {
            'greeting': {
                'keywords': ['xin chào', 'hello', 'hi', 'chào thầy', 'chào cô', 'halo', 'chào', 'hey'],
                'confidence_threshold': 0.6,
                'description': 'Chào hỏi',
                'response_style': 'friendly'
            },
            
            # ✅ LECTURER-SPECIFIC INTENTS based on QA.csv analysis
            'bank_exam_questions': {
                'keywords': ['ngân hàng đề thi', 'ngan hang de thi', 'đề thi', 'de thi', 'báo cáo đề thi', 'bao cao de thi', 'kế hoạch đề thi', 'ke hoach de thi', 'file mềm', 'file mem'],
                'confidence_threshold': 0.4,
                'description': 'Ngân hàng đề thi',
                'response_style': 'detailed'
            },
            
            'annual_task_declaration': {
                'keywords': ['kê khai nhiệm vụ', 'ke khai nhiem vu', 'nhiệm vụ năm học', 'nhiem vu nam hoc', 'kê khai', 'ke khai', 'giờ chuẩn', 'gio chuan', 'giảng viên cơ hữu', 'giang vien co huu', 'thỉnh giảng', 'thinh giang'],
                'confidence_threshold': 0.4,
                'description': 'Kê khai nhiệm vụ năm học',
                'response_style': 'informative'
            },
            
            'academic_journal': {
                'keywords': ['tạp chí', 'tap chi', 'tạp chí khoa học', 'tap chi khoa hoc', 'bài viết', 'bai viet', 'nghiên cứu', 'nghien cuu', 'khoa học công nghệ', 'khoa hoc cong nghe', 'gửi bài', 'gui bai'],
                'confidence_threshold': 0.4,
                'description': 'Tạp chí khoa học',
                'response_style': 'detailed'
            },
            
            'competition_awards': {
                'keywords': ['thi đua', 'thi dua', 'khen thưởng', 'khen thuong', 'danh hiệu', 'danh hieu', 'bằng khen', 'bang khen', 'lễ khen thưởng', 'le khen thuong', 'chiến sĩ thi đua', 'chien si thi dua', 'lao động tiên tiến', 'lao dong tien tien'],
                'confidence_threshold': 0.4,
                'description': 'Thi đua khen thưởng',
                'response_style': 'encouraging'
            },
            
            'reports_deadlines': {
                'keywords': ['báo cáo', 'bao cao', 'nộp', 'nop', 'hạn cuối', 'han cuoi', 'deadline', 'thời hạn', 'thoi han', 'gửi về', 'gui ve', 'phòng đảm bảo chất lượng', 'phong dam bao chat luong'],
                'confidence_threshold': 0.4,
                'description': 'Báo cáo và thủ tục',
                'response_style': 'urgent'
            },
            
            'teaching_schedule': {
                'keywords': ['lịch giảng dạy', 'lich giang day', 'thời khóa biểu', 'thoi khoa bieu', 'lịch học', 'lich hoc', 'cập nhật dữ liệu', 'cap nhat du lieu', 'phần mềm quản lý', 'phan mem quan ly', 'đào tạo', 'dao tao'],
                'confidence_threshold': 0.4,
                'description': 'Lịch giảng dạy',
                'response_style': 'informative'
            },
            
            'quality_assurance': {
                'keywords': ['đảm bảo chất lượng', 'dam bao chat luong', 'kiểm tra', 'kiem tra', 'giám sát', 'giam sat', 'đánh giá', 'danh gia', 'chuẩn đầu ra', 'chuan dau ra', 'tiêu chuẩn', 'tieu chuan'],
                'confidence_threshold': 0.5,
                'description': 'Đảm bảo chất lượng',
                'response_style': 'detailed'
            },
            
            'departments_contacts': {
                'keywords': ['phòng ban', 'phong ban', 'liên hệ', 'lien he', 'email', 'phone', 'contact', 'phòng tổ chức', 'phong to chuc', 'phòng nghiên cứu', 'phong nghien cuu', 'phòng khảo thí', 'phong khao thi'],
                'confidence_threshold': 0.4,
                'description': 'Thông tin phòng ban',
                'response_style': 'helpful'
            },
            
            # ✅ GENERAL EDUCATION INTENTS (kept from original)
            'admission_general': {
                'keywords': ['tuyển sinh', 'tuyen sinh', 'nhập học', 'đăng ký học', 'vào trường', 'điều kiện tuyển sinh', 'xét tuyển'],
                'confidence_threshold': 0.5,
                'description': 'Thông tin tuyển sinh chung',
                'response_style': 'informative'
            },
            'tuition_general': {
                'keywords': ['học phí', 'hoc phi', 'hp', 'chi phí', 'chi phi', 'tiền học', 'tien hoc', 'mức phí', 'muc phi', 'phí học tập', 'phi hoc tap'],
                'confidence_threshold': 0.4,
                'description': 'Học phí chung',
                'response_style': 'informative'
            },
            'programs': {
                'keywords': ['ngành', 'nganh', 'chuyên ngành', 'chuyen nganh', 'đào tạo', 'dao tao', 'chương trình học', 'chuong trinh hoc'],
                'confidence_threshold': 0.5,
                'description': 'Chương trình đào tạo',
                'response_style': 'informative'
            },
            'facilities': {
                'keywords': ['cơ sở vật chất', 'co so vat chat', 'phòng học', 'phong hoc', 'thư viện', 'thu vien', 'lab', 'ký túc xá', 'ky tuc xa', 'tiện ích', 'tien ich'],
                'confidence_threshold': 0.6,
                'description': 'Cơ sở vật chất',
                'response_style': 'descriptive'
            },
            
            # ✅ CLARIFICATION AND VAGUE INTENTS
            'clarification_needed': {
                'keywords': ['gì', 'gi', 'sao', 'nào', 'nao', 'như thế nào', 'nhu the nao', 'làm sao', 'lam sao', 'cách nào', 'cach nao', 'thế nào', 'the nao'],
                'confidence_threshold': 0.2,
                'description': 'Cần làm rõ',
                'response_style': 'clarifying'
            },
            
            'general': {
                'keywords': ['thông tin', 'thong tin', 'hỗ trợ', 'ho tro', 'giúp', 'giup', 'hướng dẫn', 'huong dan', 'bdu', 'đại học bình dương', 'dai hoc binh duong'],
                'confidence_threshold': 0.2,
                'description': 'Câu hỏi chung',
                'response_style': 'neutral'
            }
        }
    
    def _initialize_lecturer_entities(self):
        """Enhanced entity patterns for lecturers"""
        return {
            'lecturer_departments': [
                'phòng đảm bảo chất lượng', 'phòng khảo thí', 'phòng tổ chức cán bộ',
                'phòng nghiên cứu hợp tác', 'phòng đào tạo', 'phòng công tác sinh viên',
                'phong dam bao chat luong', 'phong khao thi', 'phong to chuc can bo',
                'phong nghien cuu hop tac', 'phong dao tao', 'phong cong tac sinh vien'
            ],
            'lecturer_positions': [
                'giảng viên', 'giảng viên cơ hữu', 'giảng viên thỉnh giảng',
                'phó giáo sư', 'tiến sĩ', 'thạc sĩ', 'trưởng khoa', 'phó khoa',
                'giang vien', 'giang vien co huu', 'giang vien thinh giang',
                'pho giao su', 'tien si', 'thac si', 'truong khoa', 'pho khoa'
            ],
            'document_types': [
                'báo cáo', 'kế hoạch', 'thông báo', 'quyết định', 'file mềm',
                'bao cao', 'ke hoach', 'thong bao', 'quyet dinh', 'file mem',
                'văn bản', 'hồ sơ', 'tài liệu', 'van ban', 'ho so', 'tai lieu'
            ],
            'time_expressions': [
                'năm học 2023-2024', 'học kỳ I', 'học kỳ II', 'kỳ hè',
                'nam hoc 2023-2024', 'hoc ky I', 'hoc ky II', 'ky he',
                'trước ngày', 'hạn cuối', 'deadline', 'truoc ngay', 'han cuoi'
            ],
            'lecturer_activities': [
                'giảng dạy', 'nghiên cứu khoa học', 'phục vụ cộng đồng',
                'giang day', 'nghien cuu khoa hoc', 'phuc vu cong dong',
                'thi đua', 'khen thưởng', 'đánh giá', 'thi dua', 'khen thuong', 'danh gia'
            ],
            'majors': [
                'công nghệ thông tin', 'cntt', 'it', 'khoa học máy tính', 'tin học',
                'kinh tế', 'quản trị kinh doanh', 'marketing', 'tài chính ngân hàng', 'kế toán',
                'luật', 'luật học', 'pháp lý', 'luật kinh tế',
                'y khoa', 'y học', 'bác sĩ', 'điều dưỡng', 'dược', 'y tế',
                'kỹ thuật', 'xây dựng', 'cơ khí', 'điện', 'điện tử', 'oto'
            ],
            'emotions': [
                'cần gấp', 'khẩn cấp', 'urgent', 'quan trọng', 'ưu tiên',
                'can gap', 'khan cap', 'quan trong', 'uu tien',
                'lo lắng', 'khó khăn', 'căng thẳng', 'stress',
                'lo lang', 'kho khan', 'cang thang'
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
        """Enhanced intent classification with normalization for lecturers"""
        if not query or not query.strip():
            return {
                'intent': 'general',
                'confidence': 0.3,
                'description': 'Câu hỏi chung',
                'response_style': 'neutral'
            }
        
        # ✅ CRITICAL: Check if normalizer exists
        if not self.normalizer:
            print("❌ NORMALIZER ERROR: Normalizer not available")
            query_variants = [query, query.lower()]
            normalized_query = query.lower()
        else:
            # NORMALIZE QUERY FIRST
            normalized_query = self.normalizer.normalize_query(query)
            query_variants = self.normalizer.create_search_variants(query)
        
        print(f"🔍 LECTURER INTENT DEBUG: Original = '{query}'")
        print(f"🔍 LECTURER INTENT DEBUG: Normalized = '{normalized_query}'")
        print(f"🔍 LECTURER INTENT DEBUG: Variants = {query_variants}")
        
        intent_scores = {}
        
        # Method 1: Enhanced keyword matching with variants for lecturers
        for variant in query_variants:
            variant_lower = variant.lower().strip()
            
            for intent, config in self.intent_categories.items():
                score = 0
                keyword_matches = 0
                
                for keyword in config['keywords']:
                    if keyword in variant_lower:
                        # ✅ ENHANCED: Boost score for lecturer-specific terms
                        if intent.startswith(('bank_exam', 'annual_task', 'academic_journal', 'competition_awards')):
                            score += 2.5  # Higher weight for lecturer-specific intents
                        elif keyword == variant_lower:
                            score += 2
                        elif variant_lower.startswith(keyword) or variant_lower.endswith(keyword):
                            score += 1.5
                        else:
                            score += 1
                        keyword_matches += 1
                
                # Normalize and boost for multiple keyword matches
                if len(config['keywords']) > 0:
                    base_score = score / len(config['keywords'])
                    # Bonus for multiple keyword matches
                    if keyword_matches > 1:
                        base_score *= (1 + (keyword_matches - 1) * 0.3)  # Increased bonus
                    
                    # Update intent score with max from all variants
                    intent_scores[intent] = max(intent_scores.get(intent, 0), min(base_score, 1.0))
        
        print(f"🔍 LECTURER INTENT DEBUG: Intent scores = {intent_scores}")
        
        # Method 2: Context-based boosting with normalized query for lecturers
        self._boost_lecturer_contextual_intents(normalized_query.lower(), intent_scores)
        
        print(f"🔍 LECTURER INTENT DEBUG: After lecturer boosting = {intent_scores}")
        
        # Method 3: PhoBERT similarity (if available)
        if not self.fallback_mode and self.model and self.tokenizer:
            try:
                # Use normalized query for semantic similarity
                self._add_semantic_similarity(normalized_query, intent_scores)
            except Exception as e:
                logger.warning(f"Semantic similarity failed, using fallback: {str(e)}")
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_name, confidence = best_intent
            
            print(f"🔍 LECTURER INTENT DEBUG: Best intent = {intent_name}, confidence = {confidence}")
            
            # ✅ Dynamic threshold based on query complexity for lecturers
            base_threshold = self.intent_categories[intent_name]['confidence_threshold']
            if self.fallback_mode:
                threshold = base_threshold * 0.3  # ✅ VERY LOW for lecturer fallback
            else:
                threshold = base_threshold * 0.5  # ✅ LOWER with normalization
            
            print(f"🔍 LECTURER INTENT DEBUG: Threshold = {threshold}, base = {base_threshold}")
            
            if confidence >= threshold:
                print(f"🔍 LECTURER INTENT DEBUG: INTENT MATCHED!")
                return {
                    'intent': intent_name,
                    'confidence': confidence,
                    'description': self.intent_categories[intent_name]['description'],
                    'response_style': self.intent_categories[intent_name]['response_style'],
                    'normalized_query': normalized_query,
                    'lecturer_optimized': True
                }
        
        print(f"🔍 LECTURER INTENT DEBUG: NO INTENT MATCHED - using general")
        return {
            'intent': 'general',
            'confidence': 0.3,
            'description': 'Câu hỏi chung',
            'response_style': 'neutral',
            'normalized_query': normalized_query,
            'lecturer_optimized': True
        }
    
    def _boost_lecturer_contextual_intents(self, query_lower, intent_scores):
        """Boost intent scores based on lecturer-specific context"""
        
        # ✅ LECTURER-SPECIFIC: Department context
        if any(phrase in query_lower for phrase in ['phòng đảm bảo', 'phòng khảo thí', 'phong dam bao', 'phong khao thi']):
            intent_scores['bank_exam_questions'] = intent_scores.get('bank_exam_questions', 0) + 0.4
            intent_scores['quality_assurance'] = intent_scores.get('quality_assurance', 0) + 0.3
        
        if any(phrase in query_lower for phrase in ['phòng tổ chức', 'phòng cán bộ', 'phong to chuc', 'phong can bo']):
            intent_scores['annual_task_declaration'] = intent_scores.get('annual_task_declaration', 0) + 0.4
            intent_scores['competition_awards'] = intent_scores.get('competition_awards', 0) + 0.3
        
        # ✅ LECTURER-SPECIFIC: Urgency context
        if any(word in query_lower for word in ['hạn cuối', 'deadline', 'gấp', 'khẩn cấp', 'han cuoi', 'gap', 'khan cap']):
            intent_scores['reports_deadlines'] = intent_scores.get('reports_deadlines', 0) + 0.5
        
        # ✅ LECTURER-SPECIFIC: Academic context
        if any(word in query_lower for word in ['nghiên cứu', 'bài viết', 'tạp chí', 'nghien cuu', 'bai viet', 'tap chi']):
            intent_scores['academic_journal'] = intent_scores.get('academic_journal', 0) + 0.4
        
        # ✅ LECTURER-SPECIFIC: Teaching context
        if any(word in query_lower for word in ['giảng dạy', 'lịch học', 'thời khóa biểu', 'giang day', 'lich hoc', 'thoi khoa bieu']):
            intent_scores['teaching_schedule'] = intent_scores.get('teaching_schedule', 0) + 0.4
        
        # ✅ LECTURER-SPECIFIC: Awards context
        if any(word in query_lower for word in ['thi đua', 'khen thưởng', 'danh hiệu', 'thi dua', 'khen thuong', 'danh hieu']):
            intent_scores['competition_awards'] = intent_scores.get('competition_awards', 0) + 0.4
        
        # Question patterns (enhanced for lecturers)
        if query_lower.endswith('?'):
            # Questions from lecturers tend to be more specific
            for intent in ['bank_exam_questions', 'annual_task_declaration', 'academic_journal', 'reports_deadlines']:
                if intent in intent_scores:
                    intent_scores[intent] += 0.2
        
        # Vague questions that need clarification
        vague_indicators = ['gì', 'sao', 'nào', 'như thế nào', 'gi', 'nao', 'nhu the nao']
        if any(word in query_lower for word in vague_indicators) and len(query_lower.split()) <= 5:
            intent_scores['clarification_needed'] = intent_scores.get('clarification_needed', 0) + 0.3
    
    def _add_semantic_similarity(self, query, intent_scores):
        """Add PhoBERT semantic similarity scores"""
        try:
            if self.fallback_mode or not self.model or not self.tokenizer:
                return
                
            query_embedding = self.encode_text(query)
            if query_embedding is not None:
                for intent, config in self.intent_categories.items():
                    # Create comprehensive intent representation
                    intent_text = f"{config['description']} {' '.join(config['keywords'][:5])}"
                    intent_embedding = self.encode_text(intent_text)
                    
                    if intent_embedding is not None:
                        similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
                        # Blend with keyword score
                        current_score = intent_scores.get(intent, 0)
                        blended_score = (current_score * 0.7) + (similarity * 0.3)  # Favor keywords more
                        intent_scores[intent] = blended_score
                        
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {str(e)}")
    
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
        """Enhanced entity extraction for lecturers"""
        if not query:
            return {}
            
        query_lower = query.lower()
        entities = {}
        
        # ✅ LECTURER-SPECIFIC: Extract departments with confidence
        for dept in self.entity_patterns['lecturer_departments']:
            if dept in query_lower:
                entities['department'] = dept
                entities['department_confidence'] = 1.0 if dept == query_lower else 0.9
                break
        
        # ✅ LECTURER-SPECIFIC: Extract positions
        for position in self.entity_patterns['lecturer_positions']:
            if position in query_lower:
                entities['position'] = position
                entities['position_confidence'] = 1.0 if position == query_lower else 0.8
                break
        
        # ✅ LECTURER-SPECIFIC: Extract document types
        for doc_type in self.entity_patterns['document_types']:
            if doc_type in query_lower:
                entities['document_type'] = doc_type
                break
        
        # ✅ LECTURER-SPECIFIC: Extract lecturer activities
        for activity in self.entity_patterns['lecturer_activities']:
            if activity in query_lower:
                entities['activity'] = activity
                break
        
        # Extract majors with confidence
        for major in self.entity_patterns['majors']:
            if major in query_lower:
                entities['major'] = major
                entities['major_confidence'] = 1.0 if major == query_lower else 0.8
                break
        
        # Extract time expressions
        for time_expr in self.entity_patterns['time_expressions']:
            if time_expr in query_lower:
                entities['time'] = time_expr
                break
        
        # ✅ ENHANCED: Extract emotions with lecturer-specific intensity
        emotion_intensity = 0
        detected_emotion = None
        for emotion in self.entity_patterns['emotions']:
            if emotion in query_lower:
                # Lecturer-specific emotions get different intensity
                if emotion in ['cần gấp', 'khẩn cấp', 'urgent', 'can gap', 'khan cap']:
                    emotion_intensity = 0.9  # High urgency for lecturers
                elif emotion in ['quan trọng', 'ưu tiên', 'quan trong', 'uu tien']:
                    emotion_intensity = 0.8
                elif emotion in ['lo lắng', 'khó khăn', 'lo lang', 'kho khan']:
                    emotion_intensity = 0.7
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
            
            # ✅ ENHANCED: Additional analysis for lecturers
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
            # Safe fallback for lecturers
            return {
                'intent': {
                    'intent': 'general',
                    'confidence': 0.3,
                    'description': 'Câu hỏi chung',
                    'response_style': 'neutral'
                },
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
            
        # ✅ LECTURER-SPECIFIC urgent terms
        urgent_words = ['gấp', 'urgent', 'khẩn cấp', 'cần ngay', 'hạn cuối', 'deadline', 
                       'gap', 'khan cap', 'can ngay', 'han cuoi']
        medium_urgent_words = ['sớm', 'nhanh chóng', 'ưu tiên', 'quan trọng',
                              'som', 'nhanh chong', 'uu tien', 'quan trong']
        
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
        
        # ✅ LECTURER-SPECIFIC: Consider technical terms
        technical_terms = ['ngân hàng đề thi', 'kê khai nhiệm vụ', 'tạp chí khoa học', 
                          'đảm bảo chất lượng', 'phần mềm quản lý']
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
            
        # ✅ LECTURER-SPECIFIC sentiment words
        positive_words = ['tốt', 'hay', 'thích', 'muốn', 'quan tâm', 'hào hứng', 'hỗ trợ',
                         'tot', 'thich', 'quan tam', 'hao hung', 'ho tro']
        negative_words = ['khó khăn', 'lo lắng', 'không', 'chán', 'tệ', 'vấn đề', 'lỗi',
                         'kho khan', 'lo lang', 'khong', 'chan', 'te', 'van de', 'loi']
        urgent_words = ['gấp', 'khẩn cấp', 'cần ngay', 'gap', 'khan cap', 'can ngay']
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        urgent_count = sum(1 for word in urgent_words if word in query_lower)
        
        # Factor in emotional entities
        if entities and 'emotion' in entities:
            emotion = entities['emotion']
            if emotion in ['quan trọng', 'ưu tiên', 'quan trong', 'uu tien']:
                positive_count += 1
            elif emotion in ['lo lắng', 'khó khăn', 'lo lang', 'kho khan']:
                negative_count += 2
            elif emotion in ['cần gấp', 'khẩn cấp', 'can gap', 'khan cap']:
                urgent_count += 2
        
        # ✅ LECTURER-SPECIFIC: Urgency is often neutral/professional
        if urgent_count > 0:
            return 'urgent'
        elif positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    # COMPATIBILITY METHODS - Enhanced for lecturers
    def get_system_status(self):
        """Get PhoBERT system status for lecturers"""
        return {
            'model_loaded': bool(self.model),
            'fallback_mode': self.fallback_mode,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'device': str(self.device) if self.device else 'cpu',
            'intents_available': len(self.intent_categories),
            'lecturer_intents': [
                'bank_exam_questions', 'annual_task_declaration', 'academic_journal',
                'competition_awards', 'reports_deadlines', 'teaching_schedule',
                'quality_assurance', 'departments_contacts'
            ],
            'lecturer_optimized': True,
            'features': [
                'lecturer_specific_intents',
                'department_entity_extraction',
                'urgency_detection',
                'clarification_detection',
                'vietnamese_normalization'
            ]
        }