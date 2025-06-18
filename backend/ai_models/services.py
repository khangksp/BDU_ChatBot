import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import pickle
import os
import re
from django.conf import settings
from knowledge.models import KnowledgeBase
import logging
from .phobert_service import PhoBERTIntentClassifier
from .gemini_service import GeminiResponseGenerator, SimpleVietnameseRestorer
import pandas as pd

logger = logging.getLogger(__name__)

class LecturerDecisionEngine:
    """
    Enhanced Decision Engine specifically for BDU Lecturers
    """
    
    def __init__(self):
        # ✅ ENHANCED: Confidence thresholds for lecturers
        self.confidence_thresholds = {
            'high_trust': 0.7,      # Very specific lecturer info - use direct
            'medium_trust': 0.5,    # Related info - enhance with context
            'low_trust': 0.25,      # Uncertain - ask for clarification
            'no_trust': 0.1         # No match - say don't know
        }
        
        # ✅ EXPANDED: Education keywords CHO GIẢNG VIÊN BDU
        self.education_keywords = [
            # Từ khóa cơ bản về giáo dục
            'học', 'trường', 'sinh viên', 'tuyển sinh', 'học phí', 'ngành', 
            'đại học', 'bdu', 'gv', 'giảng viên', 'dạy', 'quy định', 'khoa',
            'chương trình', 'đào tạo', 'lịch', 'thời khóa biểu', 'phòng', 'lớp',
            
            # ✅ CRITICAL: Thêm từ khóa cho SINH VIÊN (vì giảng viên cũng hỏi về sinh viên)
            'lệ phí', 'phí', 'tiền', 'bằng', 'văn bằng', 'tốt nghiệp', 'nhận bằng',
            'chuyển khoản', 'thanh toán', 'nộp tiền', 'đóng phí', 'thu ngân',
            'kế toán', 'tài chính', 'điểm', 'transcript', 'bảng điểm',
            'thủ tục', 'giấy tờ', 'hồ sơ', 'đăng ký', 'xin cấp',
            'le phi', 'phi', 'tien', 'bang', 'van bang', 'tot nghiep', 'nhan bang',
            'chuyen khoan', 'thanh toan', 'nop tien', 'dong phi', 'thu ngan',
            'ke toan', 'tai chinh', 'diem', 'bang diem',
            'thu tuc', 'giay to', 'ho so', 'dang ky', 'xin cap',
            
            # ✅ THÊM: Từ khóa QUAN TRỌNG cho GIẢNG VIÊN (extracted from QA.csv analysis)
            'hội đồng', 'nghiên cứu', 'công tác', 'báo cáo', 'đánh giá',
            'thi đua', 'thành tích', 'khen thưởng', 'xét', 'xét thi đua',
            'cá nhân', 'tập thể', 'hoàn thành', 'nhiệm vụ', 'chức năng',
            'tiêu chuẩn', 'tiêu chí', 'định mức', 'chất lượng',
            'kiểm tra', 'giám sát', 'quản lý', 'vận hành',
            'kết quả', 'hiệu quả', 'thực hiện', 'hoạt động',
            'phân công', 'giao nhiệm vụ', 'trách nhiệm',
            'chuẩn đầu ra', 'mục tiêu', 'chỉ tiêu', 'kế hoạch',
            'học kỳ', 'năm học', 'kỳ thi', 'thi cử', 'điểm số',
            'bài giảng', 'giáo án', 'tài liệu', 'giáo trình',
            'lớp học', 'môn học', 'học phần', 'tín chỉ',
            'cố vấn', 'hướng dẫn', 'tư vấn', 'hỗ trợ',
            'ảnh hưởng', 'mất lòng', 'xuất sắc', 'đồng nghiệp',
            
            # ✅ CRITICAL: Từ khóa cụ thể từ QA.csv cho giảng viên
            'ngân hàng đề thi', 'file mềm', 'báo cáo', 'nộp', 'hạn cuối',
            'email', 'phòng ban', 'đơn vị', 'khoa', 'bộ môn',
            'kê khai', 'nhiệm vụ năm học', 'giờ chuẩn', 'thỉnh giảng', 'cơ hữu',
            'tạp chí', 'khoa học công nghệ', 'bài viết', 'nghiên cứu',
            'lễ khen thưởng', 'bằng khen', 'danh hiệu', 'công nhận',
            'phòng đảm bảo chất lượng', 'khảo thí', 'phòng tổ chức cán bộ',
            'quyết định', 'thông báo', 'văn bản', 'triển khai',
            'cập nhật', 'dữ liệu', 'phần mềm', 'quản lý đào tạo',
            'hoạt động giảng dạy', 'công tác giảng dạy', 'đảm bảo chất lượng',
            
            # Từ khóa không dấu (QUAN TRỌNG cho search)
            'hoc', 'truong', 'sinh vien', 'tuyen sinh', 'hoc phi', 'nganh',
            'dai hoc', 'giang vien', 'day', 'quy dinh', 'chuong trinh', 'dao tao',
            'thi dua', 'thanh tich', 'khen thuong', 'xet', 'xet thi dua',
            'ca nhan', 'tap the', 'hoan thanh', 'nhiem vu', 'chuc nang',
            'tieu chuan', 'tieu chi', 'dinh muc', 'chat luong',
            'kiem tra', 'giam sat', 'quan ly', 'van hanh',
            'ket qua', 'hieu qua', 'thuc hien', 'hoat dong',
            'phan cong', 'giao nhiem vu', 'trach nhiem',
            'chuan dau ra', 'muc tieu', 'chi tieu', 'ke hoach',
            'hoc ky', 'nam hoc', 'ky thi', 'thi cu', 'diem so',
            'bai giang', 'giao an', 'tai lieu', 'giao trinh',
            'lop hoc', 'mon hoc', 'hoc phan', 'tin chi',
            'co van', 'huong dan', 'tu van', 'ho tro',
            'anh huong', 'mat long', 'xuat sac', 'dong nghiep',
            'ngan hang de thi', 'file mem', 'bao cao', 'nop', 'han cuoi',
            'ke khai', 'nhiem vu nam hoc', 'gio chuan', 'thinh giang', 'co huu',
            'tap chi', 'khoa hoc cong nghe', 'bai viet', 'nghien cuu',
            'le khen thuong', 'bang khen', 'danh hieu', 'cong nhan'
        ]
        
        # ✅ THÊM: Giảng viên specific keywords
        self.lecturer_keywords = [
            'giảng viên', 'gv', 'thầy', 'cô', 'phụ trách', 'giảng dạy',
            'nghiên cứu', 'hội đồng', 'khoa', 'bộ môn', 'chuyên ngành',
            'giang vien', 'phu trach', 'giang day', 'nghien cuu', 'chuyen nganh'
        ]
        
        # ✅ THÊM: Keywords require clarification (câu hỏi mơ hồ)
        self.vague_keywords = [
            'làm sao', 'như thế nào', 'cách nào', 'thủ tục', 'quy trình',
            'thông tin', 'chi tiết', 'hướng dẫn', 'giúp đỡ', 'hỗ trợ',
            'gì', 'nào', 'khi nào', 'ở đâu', 'ai', 'sao', 'có phải',
            'lam sao', 'nhu the nao', 'cach nao', 'thu tuc', 'quy trinh',
            'thong tin', 'chi tiet', 'huong dan', 'giup do', 'ho tro'
        ]
        
        logger.info("✅ LecturerDecisionEngine initialized for LECTURERS with expanded keywords")
    
    def is_education_related(self, query):
        """Enhanced education detection for lecturers with memory context"""
        if not query:
            return False
        
        query_lower = query.lower()
        
        # ✅ CRITICAL: Tìm kiếm bất kỳ từ khóa nào có trong câu hỏi
        found_keywords = []
        for kw in self.education_keywords:
            if kw in query_lower:
                found_keywords.append(kw)
        
        # Count education keywords
        education_count = len(found_keywords)
        lecturer_count = sum(1 for kw in self.lecturer_keywords if kw in query_lower)
        
        # ✅ LOOSENED: Chỉ cần 1 keyword education hoặc lecturer
        is_education = education_count >= 1 or lecturer_count >= 1
        
        # ✅ SPECIAL: Nếu không tìm thấy keyword, kiểm tra các pattern phổ biến
        if not is_education:
            # Kiểm tra các pattern về giáo dục
            education_patterns = [
                r'phí.*(?:học|tốt nghiệp|nhận|cấp)',
                r'(?:học|phí|tiền).*(?:phí|học|cấp|nhận)',
                r'(?:bằng|văn bằng|tốt nghiệp)',
                r'(?:thủ tục|quy trình|cách thức)',
                r'(?:bdu|đại học|trường)',
                r'(?:sinh viên|học sinh)',
                r'(?:giảng viên|thầy|cô|gv)'
            ]
            
            for pattern in education_patterns:
                if re.search(pattern, query_lower):
                    is_education = True
                    found_keywords.append(f"pattern:{pattern}")
                    break
        
        logger.info(f"🎓 Education check: '{query}' -> keywords:{found_keywords} -> {is_education}")
        return is_education
    
    def needs_clarification(self, query, confidence):
        """Check if query needs clarification"""
        if not query:
            return False
            
        query_lower = query.lower()
        
        # Check for vague questions
        vague_count = sum(1 for kw in self.vague_keywords if kw in query_lower)
        word_count = len(query.split())
        
        # Very short + vague OR low confidence
        needs_clarification = (
            (vague_count >= 2 and word_count <= 5) or 
            (confidence < self.confidence_thresholds['low_trust'] and vague_count >= 1)
        )
        
        logger.info(f"❓ Clarification check: vague:{vague_count}, words:{word_count}, conf:{confidence:.3f} -> {needs_clarification}")
        return needs_clarification
    
    def categorize_confidence(self, similarity_score):
        """Categorize confidence level"""
        if similarity_score >= self.confidence_thresholds['high_trust']:
            return 'high_trust'
        elif similarity_score >= self.confidence_thresholds['medium_trust']:
            return 'medium_trust'  
        elif similarity_score >= self.confidence_thresholds['low_trust']:
            return 'low_trust'
        else:
            return 'no_trust'
    
    def make_decision(self, query, retrieval_result, intent_result, session_memory=None):
        """Enhanced decision making for lecturers with memory context"""
        
        # Step 1: Check conversation context first
        context_override = False
        if session_memory and len(session_memory) > 0:
            # Kiểm tra 3 câu hỏi gần nhất có phải về education không
            recent_queries = [item.get('query', '') for item in session_memory[-3:]]
            recent_education_queries = [q for q in recent_queries if self.is_education_related(q)]
            
            # Nếu có ít nhất 1 câu gần đây về education -> cho phép câu hiện tại
            if len(recent_education_queries) >= 1:
                context_override = True
                logger.info(f"🧠 MEMORY OVERRIDE: Recent education context detected - allowing current query")
        
        # Step 2: Check if education-related
        is_education = self.is_education_related(query) or context_override
        
        if not is_education:
            return 'reject_non_education', None, False
        
        # Step 3: Get confidence level
        similarity = retrieval_result.get('confidence', 0)
        confidence_level = self.categorize_confidence(similarity)
        
        # Step 4: Check if needs clarification
        needs_clarification = self.needs_clarification(query, similarity)
        
        logger.info(f"🤖 Decision inputs: education={is_education}, context_override={context_override}, similarity={similarity:.3f}, level={confidence_level}, clarify={needs_clarification}")
        
        # Step 5: Make decision
        if needs_clarification:
            return 'ask_clarification', {
                'query': query,
                'confidence': similarity,
                'instruction': 'clarification_needed',
                'message': 'Question is too vague, need clarification'
            }, True
        
        if confidence_level == 'high_trust':
            decision = 'use_db_direct'
            context = {
                'instruction': 'direct_answer_lecturer',
                'db_answer': retrieval_result.get('response', ''),
                'confidence': similarity,
                'message': 'High confidence - use database answer directly'
            }
            
        elif confidence_level == 'medium_trust':
            decision = 'enhance_db_answer'
            context = {
                'instruction': 'enhance_answer_lecturer',
                'db_answer': retrieval_result.get('response', ''),
                'confidence': similarity,
                'message': 'Medium confidence - enhance database answer'
            }
            
        elif confidence_level == 'low_trust':
            decision = 'ask_clarification'
            context = {
                'instruction': 'clarification_needed',
                'db_answer': retrieval_result.get('response', ''),
                'confidence': similarity,
                'message': 'Low confidence - ask for clarification'
            }
            
        else:  # no_trust
            decision = 'say_dont_know'
            context = {
                'instruction': 'dont_know_lecturer',
                'confidence': similarity,
                'message': 'No relevant information - say dont know'
            }
        
        logger.info(f"🎯 Decision made: {decision}")
        return decision, context, True


class HybridChatbotAI:
    """
    Enhanced Hybrid Chatbot specifically for BDU Lecturers
    """
    
    def __init__(self):
        # Initialize components with lecturer-specific enhancements
        self.sbert_retriever = ChatbotAI()
        self.intent_classifier = PhoBERTIntentClassifier()
        self.response_generator = GeminiResponseGenerator()  # Now uses enhanced version
        self.decision_engine = LecturerDecisionEngine()  # New lecturer-specific engine
        
        # Enhanced conversation memory for lecturers
        self.conversation_memory = {}
        
        logger.info("🚀 HybridChatbotAI initialized specifically for BDU Lecturers")
    
    @property
    def model(self):
        return self.sbert_retriever.model
    
    @property
    def index(self):
        return self.sbert_retriever.index
    
    @property
    def knowledge_data(self):
        return self.sbert_retriever.knowledge_data
    
    def get_system_status(self):
        """Get system status for lecturers"""
        gemini_status = self.response_generator.get_system_status()
        
        return {
            'sbert_model': bool(self.sbert_retriever.model),
            'faiss_index': bool(self.sbert_retriever.index),
            'phobert_available': not self.intent_classifier.fallback_mode,
            'gemini_available': gemini_status.get('gemini_api_available', False),
            'knowledge_entries': len(self.sbert_retriever.knowledge_data),
            'mode': 'lecturer_focused_hybrid_with_clarification',
            'memory_sessions': gemini_status.get('memory_sessions', 0),
            'confidence_thresholds': self.decision_engine.confidence_thresholds,
            'lecturer_features': [
                'lecturer_keyword_detection',
                'clarification_requests', 
                'department_suggestions',
                'formal_addressing',
                'concise_responses',
                'no_fabrication_policy'
            ],
            'gemini_status': gemini_status
        }
    
    def process_query(self, query, session_id=None):
        """
        Main query processing specifically optimized for lecturers
        """
        start_time = time.time()
        
        logger.info(f"👨‍🏫 Processing lecturer query: '{query}' (session: {session_id})")
        
        try:
            # Step 1: Clean and validate input
            query = self._clean_query(query)
            if not query or len(query.strip()) < 2:
                return self._get_empty_query_response_lecturer()
            
            # Step 2: Get intent and entities
            intent_result = self.intent_classifier.classify_intent(query)
            entities = self.intent_classifier.extract_entities(query)
            
            # Step 3: Search knowledge base
            retrieval_result = self.sbert_retriever.generate_response(query)
            
            logger.info(f"🔍 Retrieval result: confidence={retrieval_result.get('confidence', 0):.3f}")
            
            # Step 4: Make lecturer-specific decision WITH MEMORY CONTEXT
            session_memory = self.get_conversation_context(session_id) if session_id else None
            decision_type, gemini_context, should_respond = self.decision_engine.make_decision(
                query, retrieval_result, intent_result, session_memory
            )
            
            # Step 5: Execute decision
            if not should_respond:
                response_text = "Dạ thầy/cô, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ. 🎓 Thầy/cô có câu hỏi nào khác về trường không ạ?"
                method = 'rejected_non_education'
            else:
                response_text = self._execute_lecturer_decision(
                    decision_type, query, gemini_context, intent_result, entities, session_id
                )
                method = decision_type
            
            # Step 6: Update memory WITH MORE DETAILS
            if session_id and should_respond:
                self._update_memory(session_id, query, intent_result, retrieval_result.get('confidence', 0), decision_type, should_respond)
            
            processing_time = time.time() - start_time
            
            return {
                'response': response_text,
                'confidence': retrieval_result.get('confidence', 0),
                'method': method,
                'decision_type': decision_type,
                'intent': intent_result,
                'sources': retrieval_result.get('sources', []),
                'entities': entities,
                'processing_time': processing_time,
                'is_education': gemini_context is not None,
                'lecturer_optimized': True
            }
            
        except Exception as e:
            logger.error(f"❌ Processing error: {str(e)}")
            return {
                'response': "Dạ thầy/cô, em gặp khó khăn kỹ thuật. Thầy/cô có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎓",
                'confidence': 0.0,
                'method': 'error_fallback',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _execute_lecturer_decision(self, decision_type, query, gemini_context, intent_result, entities, session_id):
        """Execute lecturer-specific decisions"""
        
        logger.info(f"🎯 Executing lecturer decision: {decision_type}")
        
        if decision_type == 'use_db_direct':
            # High confidence -> Use database answer directly with lecturer formatting
            response = self.response_generator.generate_response(
                query=query,
                context=gemini_context,
                intent_info=intent_result,
                entities=entities,
                session_id=session_id
            )
            return response.get('response', f"Dạ thầy/cô, {gemini_context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?")
            
        elif decision_type == 'enhance_db_answer':
            # Medium confidence -> Enhance database answer
            response = self.response_generator.generate_response(
                query=query,
                context=gemini_context,
                intent_info=intent_result,
                entities=entities,
                session_id=session_id
            )
            return response.get('response', f"Dạ thầy/cô, {gemini_context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?")
            
        elif decision_type == 'ask_clarification':
            # Need clarification -> Generate clarification request
            response = self.response_generator.generate_response(
                query=query,
                context=gemini_context,
                intent_info=intent_result,
                entities=entities,
                session_id=session_id
            )
            return response.get('response', self._get_default_clarification_request(query))
            
        elif decision_type == 'say_dont_know':
            # No relevant info -> Generate don't know response with department suggestion
            response = self.response_generator.generate_response(
                query=query,
                context=gemini_context,
                intent_info=intent_result,
                entities=entities,
                session_id=session_id
            )
            return response.get('response', self._get_default_dont_know_response(query))
            
        else:
            logger.warning(f"⚠️ Unknown decision type: {decision_type}")
            return "Dạ thầy/cô, em gặp khó khăn trong việc xử lý câu hỏi. Thầy/cô có cần hỗ trợ thêm gì không ạ? 🎓"
    
    def _get_default_clarification_request(self, query):
        """Default clarification request if Gemini fails"""
        # Extract key topic for targeted clarification
        query_words = query.lower().split()
        
        topic_keywords = {
            'ngân hàng đề thi': ['ngân hàng', 'đề thi', 'đề'],
            'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ'],
            'tạp chí': ['tạp chí', 'bài viết'],
            'thi đua khen thưởng': ['thi đua', 'khen thưởng'],
            'báo cáo': ['báo cáo', 'nộp'],
            'lịch giảng dạy': ['lịch', 'giảng dạy']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_words for kw in keywords):
                return f"Dạ thầy/cô, để em hỗ trợ chính xác về {topic}, thầy/cô có thể nói rõ hơn về nội dung cụ thể cần hỗ trợ không ạ? 🎓"
        
        return "Dạ thầy/cô, để em hỗ trợ chính xác nhất, thầy/cô có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
    
    def _get_default_dont_know_response(self, query):
        """Default don't know response with department suggestion"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ngân hàng đề', 'đề thi', 'khảo thí']):
            return "Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ Phòng Đảm bảo chất lượng và Khảo thí qua email ldkham@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        elif any(word in query_lower for word in ['kê khai', 'nhiệm vụ', 'giờ chuẩn']):
            return "Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ Phòng Tổ chức - Cán bộ qua email tcccb@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        elif any(word in query_lower for word in ['tạp chí', 'nghiên cứu', 'khoa học']):
            return "Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ Phòng Nghiên cứu - Hợp tác qua email nghiencuu@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        else:
            return "Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ phòng ban liên quan qua email info@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
    
    def _clean_query(self, query):
        """Clean and prepare query for lecturers"""
        if not query:
            return ""
        
        # Basic cleaning
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[?]{2,}', '?', query)
        query = re.sub(r'[!]{2,}', '!', query)
        
        return query
    
    def _update_memory(self, session_id, query, intent_result, confidence, decision_type=None, was_education=True):
        """Enhanced memory update for lecturers with more context"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        self.conversation_memory[session_id].append({
            'query': query,
            'intent': intent_result.get('intent', 'unknown'),
            'confidence': confidence,
            'timestamp': time.time(),
            'user_type': 'lecturer',  # Track that this is a lecturer session
            'decision_type': decision_type,  # ✅ NEW: Track decision made
            'was_education_related': was_education,  # ✅ NEW: Track if was education
            'is_education_query': self.decision_engine.is_education_related(query)  # ✅ NEW: Direct check
        })
        
        # Keep last 10 interactions for lecturers (more history for work context)
        self.conversation_memory[session_id] = self.conversation_memory[session_id][-10:]
        
        logger.info(f"🧠 Memory updated for session {session_id}: {len(self.conversation_memory[session_id])} total interactions")
    
    def _get_empty_query_response_lecturer(self):
        """Response for empty queries from lecturers"""
        return {
            'response': "Dạ chào thầy/cô! Em có thể hỗ trợ gì cho thầy/cô về công việc tại BDU ạ? 🎓",
            'confidence': 0.9,
            'method': 'empty_query_lecturer',
            'processing_time': 0.01
        }
    
    def get_conversation_context(self, session_id):
        """Get conversation context for a lecturer session"""
        return self.conversation_memory.get(session_id, [])
    
    def get_conversation_memory(self, session_id):
        """Get conversation memory from Gemini service"""
        return self.response_generator.get_conversation_memory(session_id)
    
    def clear_conversation_memory(self, session_id=None):
        """Clear conversation memory"""
        if session_id:
            self.response_generator.clear_conversation_memory(session_id)
            if session_id in self.conversation_memory:
                del self.conversation_memory[session_id]
        else:
            self.response_generator.clear_conversation_memory()
            self.conversation_memory.clear()


# Keep original ChatbotAI for retrieval (unchanged but enhanced for lecturers)
class ChatbotAI:
    def __init__(self):
        self.model = None
        self.index = None
        self.knowledge_data = []
        self.vietnamese_restorer = None  # ✅ THÊM
        self.load_models()
    
    def load_models(self):
        """Load AI models and knowledge base"""
        try:
            self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
            logger.info("✅ Vietnamese SBERT loaded for lecturers")
            
            # ✅ THÊM: Load Vietnamese restorer
            try:
                self.vietnamese_restorer = SimpleVietnameseRestorer(settings.GEMINI_API_KEY)
                logger.info("✅ Vietnamese Restorer loaded for search")
            except Exception as e:
                logger.warning(f"⚠️ Vietnamese Restorer failed to load: {e}")
                self.vietnamese_restorer = None
            
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model = None
    
    def load_knowledge_base(self):
        """Load knowledge base from database and CSV with lecturer focus"""
        try:
            # Load from database
            db_knowledge = list(KnowledgeBase.objects.filter(is_active=True).values(
                'question', 'answer', 'category'
            ))
            
            # Load from CSV file - enhanced for lecturers
            csv_path = os.path.join(settings.BASE_DIR, 'data', 'QA.csv')
            csv_knowledge = []
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding='utf-8')
                if 'question' in df.columns and 'answer' in df.columns:
                    csv_knowledge = df[['question', 'answer']].fillna('').to_dict('records')
                    if 'category' in df.columns:
                        for i, item in enumerate(csv_knowledge):
                            item['category'] = df.iloc[i].get('category', 'Giảng viên')
                    else:
                        for item in csv_knowledge:
                            item['category'] = 'Giảng viên'
            
            # Combine sources with priority for lecturer-specific content
            self.knowledge_data = csv_knowledge + db_knowledge  # CSV first for lecturer priority
            
            # Build FAISS index
            if self.model and self.knowledge_data:
                self.build_faiss_index()
            
            logger.info(f"✅ Loaded {len(self.knowledge_data)} knowledge entries for lecturers")
            
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
            self.knowledge_data = self.get_fallback_knowledge_lecturer()
    
    def get_fallback_knowledge_lecturer(self):
        """Fallback knowledge data specifically for lecturers"""
        return [
            {
                'question': 'ngân hàng đề thi',
                'answer': 'Giảng viên cần báo cáo kết quả xây dựng ngân hàng đề thi kết thúc học phần và lập kế hoạch cho học kỳ tiếp theo. Nộp về Phòng Đảm bảo chất lượng và Khảo thí qua email ldkham@bdu.edu.vn trước hạn quy định.',
                'category': 'Giảng viên'
            },
            {
                'question': 'kê khai nhiệm vụ năm học',
                'answer': 'Giảng viên cơ hữu và thỉnh giảng cần kê khai nhiệm vụ năm học bao gồm giảng dạy, nghiên cứu khoa học và các hoạt động khác. Khoa tổng hợp và báo cáo lên nhà trường.',
                'category': 'Giảng viên'
            },
            {
                'question': 'tạp chí khoa học',
                'answer': 'Tạp chí Khoa học và Công nghệ Trường Đại học Bình Dương nhận bài viết từ giảng viên, nghiên cứu sinh và các nhà khoa học. Gửi bài qua email chỉ định của tòa soạn.',
                'category': 'Giảng viên'
            },
            {
                'question': 'thi đua khen thưởng',
                'answer': 'Nhà trường tổ chức đánh giá thi đua, khen thưởng cá nhân và tập thể xuất sắc trong năm học. Có các danh hiệu như Chiến sĩ thi đua, Lao động tiên tiến...',
                'category': 'Giảng viên'
            }
        ]
    
    def build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        try:
            questions = [item['question'] for item in self.knowledge_data]
            embeddings = self.model.encode(questions)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index built with {len(questions)} entries for lecturers")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.index = None
    
    def semantic_search(self, query, top_k=3):
        """Fast semantic search optimized for lecturer queries"""
        try:
            if not self.model or not self.index:
                return self.keyword_search(query)
            
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_data):
                    result = self.knowledge_data[idx].copy()
                    result['similarity'] = float(score)
                    results.append(result)
            
            return results[0] if results else (None, 0), results
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return self.keyword_search(query)
    
    def keyword_search(self, query):
        """Enhanced keyword fallback search for lecturers"""
        query_words = set(query.lower().split())
        best_match = None
        best_score = 0
        
        for item in self.knowledge_data:
            question_words = set(item['question'].lower().split())
            answer_words = set(item['answer'].lower().split())
            
            # Enhanced matching for lecturer-specific terms
            question_common = query_words & question_words
            answer_common = query_words & answer_words
            
            if question_common or answer_common:
                # Boost score for question matches
                question_score = len(question_common) / len(query_words | question_words) * 2
                answer_score = len(answer_common) / len(query_words | answer_words)
                
                total_score = question_score + answer_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = item
        
        return best_match, best_score
    
    def generate_response(self, query):
        """Generate response optimized for lecturer hybrid system"""
        try:
            if not query.strip():
                return {
                    'response': 'Dạ thầy/cô, vui lòng nhập câu hỏi cụ thể ạ. 🎓',
                    'confidence': 0.1,
                    'method': 'empty_query',
                    'sources': []
                }
            
            # ✅ THÊM: Restore Vietnamese trước khi search
            original_query = query
            if self.vietnamese_restorer and not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Using restored query for search: '{query}' -> '{restored_query}'")
                    query = restored_query  # SỬ DỤNG CÂU ĐÃ RESTORE CHO SEARCH
            
            # Search for match
            if self.model and self.index:
                best_match, all_results = self.semantic_search(query)
            else:
                best_match, confidence = self.keyword_search(query)
                all_results = [best_match] if best_match else []
            
            if best_match:
                similarity = best_match.get('similarity', confidence if 'confidence' in locals() else 0)
                
                return {
                    'response': best_match['answer'],
                    'confidence': similarity,
                    'method': 'retrieval',
                    'sources': self._format_sources(all_results[:2]),
                    'category': best_match.get('category', 'Giảng viên')
                }
            else:
                return {
                    'response': 'Em chưa có thông tin về vấn đề này.',
                    'confidence': 0.1,
                    'method': 'no_match',
                    'sources': []
                }
            
        except Exception as e:
            logger.error(f"Generate response error: {str(e)}")
            return {
                'response': 'Đã có lỗi xảy ra trong hệ thống.',
                'confidence': 0.1,
                'method': 'error',
                'sources': []
            }
    
    def _format_sources(self, results):
        """Format sources for display"""
        sources = []
        for result in results:
            if result and result.get('similarity', 0) > 0.2:
                sources.append({
                    'question': result['question'],
                    'category': result.get('category', 'Giảng viên'),
                    'similarity': result.get('similarity', 0)
                })
        return sources

# Create global instance optimized for lecturers
chatbot_ai = HybridChatbotAI()