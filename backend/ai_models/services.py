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
import random

from .google_drive_service import google_drive_service

logger = logging.getLogger(__name__)

class LecturerDecisionEngine:
    """
    Enhanced Decision Engine specifically for BDU Lecturers
    MODIFIED: Increased generation by lowering direct answer threshold and boosting enhancement
    """
    
    def __init__(self):
        # ✅ MODIFIED: Adjusted confidence thresholds to favor more generation
        self.confidence_thresholds = {
            'high_trust': 0.85,     # ✅ INCREASED from 0.7 to 0.85 - harder to get direct answer
            'medium_trust': 0.45,   # ✅ LOWERED from 0.5 to 0.45 - easier to get enhancement
            'low_trust': 0.25,      # Keep same
            'no_trust': 0.1         # Keep same
        }
        
        # ✅ NEW: Generation boost factors
        self.generation_boost_settings = {
            'enable_boost': True,
            'boost_probability': 0.15,  # 15% chance to force enhancement even with high confidence
            'boost_keywords': [
                # Keywords that benefit from additional context
                'ngân hàng đề thi', 'kê khai nhiệm vụ', 'tạp chí', 'nghiên cứu',
                'thi đua', 'khen thưởng', 'báo cáo', 'lịch giảng dạy',
                'chất lượng', 'đánh giá', 'tiêu chuẩn', 'quy trình',
                'ngan hang de thi', 'ke khai nhiem vu', 'tap chi', 'nghien cuu',
                'thi dua', 'khen thuong', 'bao cao', 'lich giang day',
                'chat luong', 'danh gia', 'tieu chuan', 'quy trinh'
            ]
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
        
        logger.info("✅ Enhanced LecturerDecisionEngine initialized - GENERATION OPTIMIZED")
    
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
    
    def _should_boost_generation(self, query, confidence_level):
        """✅ NEW: Determine if we should boost generation for this query"""
        if not self.generation_boost_settings['enable_boost']:
            return False
        
        query_lower = query.lower()
        
        # Check if query contains boost keywords
        has_boost_keywords = any(keyword in query_lower for keyword in self.generation_boost_settings['boost_keywords'])
        
        # Random boost based on probability
        random_boost = random.random() < self.generation_boost_settings['boost_probability']
        
        # Boost conditions:
        # 1. Query has boost keywords AND confidence is high_trust -> force to medium_trust
        # 2. Random boost probability
        should_boost = (has_boost_keywords and confidence_level == 'high_trust') or random_boost
        
        if should_boost:
            logger.info(f"🚀 GENERATION BOOST ACTIVATED: keywords={has_boost_keywords}, random={random_boost}")
        
        return should_boost
    
    def make_decision(self, query, retrieval_result, intent_result, session_memory=None):
        """Enhanced decision making for lecturers with GENERATION BOOST"""
        
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
        
        # ✅ NEW: Check for generation boost
        should_boost = self._should_boost_generation(query, confidence_level)
        if should_boost and confidence_level == 'high_trust':
            confidence_level = 'medium_trust'  # Force enhancement instead of direct
            logger.info(f"🚀 GENERATION BOOST: Downgraded high_trust to medium_trust for enhancement")
        
        # Step 4: Check if needs clarification
        needs_clarification = self.needs_clarification(query, similarity)
        
        logger.info(f"🤖 Decision inputs: education={is_education}, context_override={context_override}, similarity={similarity:.3f}, level={confidence_level}, clarify={needs_clarification}, boost={should_boost}")
        
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
                'message': 'Medium confidence - enhance database answer',
                'generation_boosted': should_boost  # ✅ NEW: Mark if this was boosted
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
        """Get system status for lecturers with Google Drive info"""
        gemini_status = self.response_generator.get_system_status()
        drive_status = google_drive_service.get_system_status()
        
        # ✅ NEW: Add QA Management status
        qa_management_status = {}
        try:
            from qa_management.models import QAEntry, QASyncLog
            qa_management_status = {
                'available': True,
                'total_entries': QAEntry.objects.count(),
                'active_entries': QAEntry.objects.filter(is_active=True).count(),
                'pending_sync': QAEntry.objects.filter(sync_status='pending').count(),
                'synced_entries': QAEntry.objects.filter(sync_status='synced').count(),
                'error_entries': QAEntry.objects.filter(sync_status='error').count(),
            }
        except Exception as e:
            qa_management_status = {
                'available': False,
                'error': str(e)
            }
        
        return {
            'sbert_model': bool(self.sbert_retriever.model),
            'faiss_index': bool(self.sbert_retriever.index),
            'phobert_available': not self.intent_classifier.fallback_mode,
            'gemini_available': gemini_status.get('gemini_api_available', False),
            'knowledge_entries': len(self.sbert_retriever.knowledge_data),
            'mode': 'lecturer_focused_hybrid_with_qa_management',  # ✅ Updated mode
            'memory_sessions': gemini_status.get('memory_sessions', 0),
            'confidence_thresholds': self.decision_engine.confidence_thresholds,
            'generation_boost_enabled': self.decision_engine.generation_boost_settings['enable_boost'],
            'boost_probability': self.decision_engine.generation_boost_settings['boost_probability'],
            'lecturer_features': [
                'lecturer_keyword_detection',
                'clarification_requests', 
                'department_suggestions',
                'formal_addressing',
                'enhanced_generation_boost',
                'random_generation_enhancement',
                'keyword_based_generation_boost',
                'no_fabrication_policy',
                'qa_management_integration',  # ✅ NEW feature
                'real_time_sync',  # ✅ NEW feature
                'admin_interface'  # ✅ NEW feature
            ],
            'gemini_status': gemini_status,
            'google_drive_status': drive_status,
            'qa_management_status': qa_management_status  # ✅ NEW
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
                'generation_boosted': gemini_context.get('generation_boosted', False) if gemini_context else False,
                'lecturer_optimized': True,
                # ✅ NEW: Add reference links
                'reference_links': retrieval_result.get('reference_links', [])
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
        """✅ ENHANCED: Execute lecturer-specific decisions with generation support"""
        
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
            # ✅ ENHANCED: Medium confidence -> Enhanced generation with boost support
            
            # Check if this should get enhanced generation
            is_boosted = gemini_context.get('generation_boosted', False)
            
            if is_boosted:
                logger.info(f"🚀 GENERATION BOOST: Using enhanced generation for query")
                # Force enhanced generation by modifying context
                enhanced_context = gemini_context.copy()
                enhanced_context['instruction'] = 'enhance_answer_lecturer_boosted'
            else:
                enhanced_context = gemini_context
            
            response = self.response_generator.generate_response(
                query=query,
                context=enhanced_context,
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
    
    # ✅ NEW: Method to force reload after QA Management changes
    def reload_after_qa_update(self):
        """Reload knowledge base after QA Management updates"""
        logger.info("🔄 Reloading knowledge base after QA Management update...")
        
        # Clear caches
        if hasattr(self.sbert_retriever, 'cached_data'):
            self.sbert_retriever.cached_data = None
            self.sbert_retriever.cache_timestamp = 0
        
        # Reload knowledge base
        self.sbert_retriever.load_knowledge_base()
        
        # Rebuild FAISS index
        if self.sbert_retriever.model and self.sbert_retriever.knowledge_data:
            self.sbert_retriever.build_faiss_index()
        
        logger.info("✅ Knowledge base reloaded successfully")


# Keep original ChatbotAI for retrieval (enhanced with links and QA Management integration)
class ChatbotAI:
    def __init__(self):
        self.model = None
        self.index = None
        self.knowledge_data = []
        self.vietnamese_restorer = None
        # ✅ NEW: Link mapping from link.csv
        self.link_mapping = {}
        # ✅ NEW: Cache properties for QA Management integration
        self.cached_data = None
        self.cache_timestamp = 0
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
    
    def load_link_mapping(self):
        """✅ FIXED: Load link mapping with reduced logging"""
        try:
            link_csv_path = os.path.join(settings.BASE_DIR, 'data', 'link.csv')
            logger.info(f"🔗 Loading reference links from: {link_csv_path}")
            
            if os.path.exists(link_csv_path):
                df_links = pd.read_csv(link_csv_path, encoding='utf-8')
                logger.info(f"🔗 CSV loaded successfully. Shape: {df_links.shape}")
                
                # ✅ REDUCED LOGGING: Only show first 3 rows for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"🔗 First 3 rows:\n{df_links.head(3).to_string()}")
                
                # Create mapping from STT to Link
                for index, row in df_links.iterrows():
                    stt = str(row['STT']).strip()
                    link = str(row['Link']).strip()
                    
                    if stt and link and stt != 'nan' and link != 'nan':
                        self.link_mapping[stt] = link
                
                logger.info(f"✅ Total loaded: {len(self.link_mapping)} reference links")
                
                # ✅ REDUCED LOGGING: Only show sample mappings for debugging
                if logger.isEnabledFor(logging.DEBUG) and self.link_mapping:
                    sample_keys = list(self.link_mapping.keys())[:3]
                    sample_mappings = {k: self.link_mapping[k] for k in sample_keys}
                    logger.debug(f"🔗 Sample mappings: {sample_mappings}")
                
            else:
                logger.error(f"❌ link.csv not found at {link_csv_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading link mapping: {str(e)}")
            self.link_mapping = {}
    
    def get_reference_links(self, qa_item):
        """✅ FIXED: Get reference links with reduced logging"""
        reference_links = []
        
        # Get STT from qa_item (if it exists)
        stt_value = qa_item.get('STT', '')
        
        if not stt_value:
            return reference_links
        
        # Handle multiple STT values separated by commas or semicolons
        stt_list = []
        if isinstance(stt_value, str):
            # Split by comma, semicolon, or space
            stt_parts = re.split(r'[,;\s]+', stt_value.strip())
            stt_list = [part.strip() for part in stt_parts if part.strip()]
        else:
            stt_list = [str(stt_value).strip()]
        
        # Get links for each STT
        for stt in stt_list:
            if stt in self.link_mapping:
                link_url = self.link_mapping[stt]
                reference_links.append({
                    'stt': stt,
                    'url': link_url,
                    'title': f"Tài liệu tham khảo {stt}"
                })
                # ✅ REDUCED LOGGING: Only log when found
                logger.debug(f"✅ FOUND reference link: STT '{stt}' -> '{link_url}'")
        
        return reference_links

    def load_knowledge_base(self):
        """✅ ENHANCED: Load knowledge base with QA Management integration"""
        try:
            # ✅ FIRST: Load link mapping
            self.load_link_mapping()
            
            # ✅ NEW: Load from QA Management database FIRST (highest priority)
            db_qa_entries = []
            try:
                from qa_management.models import QAEntry
                qa_entries = QAEntry.objects.filter(is_active=True).order_by('stt')
                
                for entry in qa_entries:
                    db_qa_entries.append({
                        'question': entry.question,
                        'answer': entry.answer,
                        'category': entry.category or 'Giảng viên',
                        'STT': entry.stt,  # ✅ Include STT for reference links
                    })
                logger.info(f"✅ Loaded {len(db_qa_entries)} entries from QA Management database")
            except Exception as e:
                logger.warning(f"⚠️ QA Management not available or no data: {str(e)}")
            
            # Load from legacy database
            db_knowledge = list(KnowledgeBase.objects.filter(is_active=True).values(
                'question', 'answer', 'category'
            ))
            
            # ✅ Load from Google Drive (fallback/backup)
            csv_knowledge = []
            try:
                csv_knowledge = google_drive_service.get_csv_data()
                if csv_knowledge:
                    logger.info(f"✅ Loaded {len(csv_knowledge)} records from Google Drive (backup/fallback)")
                else:
                    logger.warning("⚠️ No data from Google Drive, using empty list")
                    csv_knowledge = []
            except Exception as e:
                logger.error(f"❌ Failed to load from Google Drive: {str(e)}")
                csv_knowledge = []
            
            # ✅ FALLBACK: Nếu Drive thất bại, thử load file local
            if not csv_knowledge and not db_qa_entries:
                logger.info("🔄 Attempting fallback to local CSV")
                csv_path = os.path.join(settings.BASE_DIR, 'data', 'QA.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8')
                        if 'question' in df.columns and 'answer' in df.columns:
                            # ✅ ENHANCED: Include all columns including STT
                            csv_knowledge = df.fillna('').to_dict('records')
                            logger.info(f"✅ Fallback: Loaded {len(csv_knowledge)} records from local CSV")
                    except Exception as e:
                        logger.error(f"❌ Fallback CSV also failed: {str(e)}")
                        csv_knowledge = []
            
            # ✅ PRIORITY: QA Management DB > Drive CSV > Legacy DB
            # This ensures QA Management data takes precedence
            self.knowledge_data = db_qa_entries + csv_knowledge + db_knowledge
            
            # Build FAISS index
            if self.model and self.knowledge_data:
                self.build_faiss_index()
            
            logger.info(f"✅ Total loaded: {len(self.knowledge_data)} knowledge entries for lecturers")
            logger.info(f"   📊 QA Management: {len(db_qa_entries)} entries")
            logger.info(f"   📊 Google Drive: {len(csv_knowledge)} entries") 
            logger.info(f"   📊 Legacy DB: {len(db_knowledge)} entries")
            logger.info(f"✅ Reference links available: {len(self.link_mapping)} links")
            
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
            self.knowledge_data = self.get_fallback_knowledge_lecturer()

    def get_fallback_knowledge_lecturer(self):
        """Fallback knowledge data specifically for lecturers"""
        return [
            {
                'question': 'ngân hàng đề thi',
                'answer': 'Giảng viên cần báo cáo kết quả xây dựng ngân hàng đề thi kết thúc học phần và lập kế hoạch cho học kỳ tiếp theo. Nộp về Phòng Đảm bảo chất lượng và Khảo thí qua email ldkham@bdu.edu.vn trước hạn quy định.',
                'category': 'Giảng viên',
                'STT': '1'
            },
            {
                'question': 'kê khai nhiệm vụ năm học',
                'answer': 'Giảng viên cơ hữu và thỉnh giảng cần kê khai nhiệm vụ năm học bao gồm giảng dạy, nghiên cứu khoa học và các hoạt động khác. Khoa tổng hợp và báo cáo lên nhà trường.',
                'category': 'Giảng viên',
                'STT': '2'
            },
            {
                'question': 'tạp chí khoa học',
                'answer': 'Tạp chí Khoa học và Công nghệ Trường Đại học Bình Dương nhận bài viết từ giảng viên, nghiên cứu sinh và các nhà khoa học. Gửi bài qua email chỉ định của tòa soạn.',
                'category': 'Giảng viên',
                'STT': '3'
            },
            {
                'question': 'thi đua khen thưởng',
                'answer': 'Nhà trường tổ chức đánh giá thi đua, khen thưởng cá nhân và tập thể xuất sắc trong năm học. Có các danh hiệu như Chiến sĩ thi đua, Lao động tiên tiến...',
                'category': 'Giảng viên',
                'STT': '4'
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
                    # ✅ NEW: Add reference links for this result
                    result['reference_links'] = self.get_reference_links(result)
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
                    best_match = item.copy()
                    # ✅ NEW: Add reference links
                    best_match['reference_links'] = self.get_reference_links(best_match)
        
        return best_match, best_score
    
    def generate_response(self, query):
        """Generate response optimized for lecturer hybrid system"""
        try:
            if not query.strip():
                return {
                    'response': 'Dạ thầy/cô, vui lòng nhập câu hỏi cụ thể ạ. 🎓',
                    'confidence': 0.1,
                    'method': 'empty_query',
                    'sources': [],
                    'reference_links': []
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
                
                # ✅ NEW: Collect all reference links from top results
                all_reference_links = []
                for i, result in enumerate(all_results[:3]):  # Top 3 results
                    if result and 'reference_links' in result:
                        all_reference_links.extend(result['reference_links'])
                
                # Remove duplicates based on STT
                unique_links = {}
                for link in all_reference_links:
                    stt = link['stt']
                    if stt not in unique_links:
                        unique_links[stt] = link
                
                final_links = list(unique_links.values())
                
                return {
                    'response': best_match['answer'],
                    'confidence': similarity,
                    'method': 'retrieval',
                    'sources': self._format_sources(all_results[:2]),
                    'category': best_match.get('category', 'Giảng viên'),
                    'reference_links': final_links  # ✅ NEW: Include reference links
                }
            else:
                return {
                    'response': 'Em chưa có thông tin về vấn đề này.',
                    'confidence': 0.1,
                    'method': 'no_match',
                    'sources': [],
                    'reference_links': []
                }
            
        except Exception as e:
            logger.error(f"Generate response error: {str(e)}")
            return {
                'response': 'Đã có lỗi xảy ra trong hệ thống.',
                'confidence': 0.1,
                'method': 'error',
                'sources': [],
                'reference_links': []
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