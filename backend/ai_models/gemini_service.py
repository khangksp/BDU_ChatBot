import logging
import time
import requests
import json
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ✅ SYSTEM PROMPT CỤ THỂ CHO GIẢNG VIÊN
LECTURER_SYSTEM_PROMPT = """Bạn là AI assistant của Đại học Bình Dương (BDU), chuyên hỗ trợ giảng viên.

🎯 QUY TẮC QUAN TRỌNG:
- LUÔN xưng hô: "thầy/cô" (TUYỆT ĐỐI KHÔNG dùng "bạn", "mình", "anh/chị")
- Bắt đầu: "Dạ thầy/cô,"
- Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
- NGẮN GỌN - Chỉ 1-2 câu chính, đi thẳng vào vấn đề
- KHÔNG CHẾ TẠO thông tin không có
- KHÔNG dùng format phức tạp với **1. **2. hay bullets

✅ PHONG CÁCH MẪU:
"Dạ thầy/cô, [thông tin chính ngắn gọn]. [Thêm 1 câu bổ sung nếu cần]. 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"

🚫 TUYỆT ĐỐI TRÁNH:
- Format phức tạp (**1. **2. **3. etc.)
- Bullets (• hoặc *)
- Câu trả lời dài dòng trên 3 câu
- Thông tin không chắc chắn
- Chế tạo số liệu, quy định

📝 KHI KHÔNG HIỂU RÕ:
- Hỏi lại để làm rõ: "Dạ thầy/cô, để em hỗ trợ chính xác, thầy/cô có thể nói rõ hơn về [vấn đề cụ thể] không ạ?"

❌ KHI KHÔNG CÓ THÔNG TIN:
- Nói thẳng: "Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ [bộ phận liên quan] để được hỗ trợ chi tiết ạ."
"""

class ConversationMemory:
    """Quản lý bộ nhớ hội thoại"""
    
    def __init__(self, max_history=10):
        self.conversations = {}  # {session_id: conversation_data}
        self.max_history = max_history
    
    def add_interaction(self, session_id: str, user_query: str, bot_response: str, 
                       intent_info: dict = None, entities: dict = None):
        """Thêm interaction vào memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history': [],
                'context_summary': "",
                'user_interests': set(),
                'conversation_type': 'lecturer'  # ✅ CHANGED: Default to lecturer
            }
        
        # Extract user interests from entities
        if entities:
            if 'major' in entities:
                self.conversations[session_id]['user_interests'].add(entities['major'])
        
        # Add to history
        interaction = {
            'timestamp': time.time(),
            'user_query': user_query,
            'bot_response': bot_response,
            'intent': intent_info.get('intent', 'unknown') if intent_info else 'unknown',
            'entities': entities or {}
        }
        
        self.conversations[session_id]['history'].append(interaction)
        
        # Keep only recent history
        if len(self.conversations[session_id]['history']) > self.max_history:
            self.conversations[session_id]['history'] = self.conversations[session_id]['history'][-self.max_history:]
        
        # Update context summary
        self._update_context_summary(session_id)
    
    def get_conversation_context(self, session_id: str) -> dict:
        """Lấy context của conversation"""
        if session_id not in self.conversations:
            return {'history': [], 'context_summary': '', 'user_interests': []}
        
        conv = self.conversations[session_id]
        return {
            'history': conv['history'][-5:],  # Last 5 interactions
            'context_summary': conv['context_summary'],
            'user_interests': list(conv['user_interests']),
            'conversation_type': conv['conversation_type']
        }
    
    def _update_context_summary(self, session_id: str):
        """Cập nhật tóm tắt context cho giảng viên"""
        conv = self.conversations[session_id]
        recent_queries = [h['user_query'] for h in conv['history'][-3:]]
        
        # ✅ ENHANCED: Context analysis for lecturers
        query_text = ' '.join(recent_queries).lower()
        
        # ✅ LECTURER-SPECIFIC contexts
        if any(word in query_text for word in ['ngân hàng đề', 'đề thi', 'khảo thí']):
            conv['context_summary'] = 'Đang hỏi về ngân hàng đề thi'
        elif any(word in query_text for word in ['kê khai', 'nhiệm vụ', 'giờ chuẩn']):
            conv['context_summary'] = 'Đang hỏi về kê khai nhiệm vụ năm học'
        elif any(word in query_text for word in ['tạp chí', 'nghiên cứu', 'bài viết']):
            conv['context_summary'] = 'Đang hỏi về tạp chí khoa học'
        elif any(word in query_text for word in ['thi đua', 'khen thưởng', 'danh hiệu']):
            conv['context_summary'] = 'Đang hỏi về thi đua khen thưởng'
        elif any(word in query_text for word in ['báo cáo', 'nộp', 'hạn cuối']):
            conv['context_summary'] = 'Đang hỏi về báo cáo và thủ tục'
        elif any(word in query_text for word in ['lịch', 'thời khóa biểu', 'giảng dạy']):
            conv['context_summary'] = 'Đang hỏi về lịch giảng dạy'
        elif any(word in query_text for word in ['học phí', 'tiền', 'chi phí']):
            conv['context_summary'] = 'Đang quan tâm học phí'
        elif any(word in query_text for word in ['tuyển sinh', 'điểm', 'xét tuyển']):
            conv['context_summary'] = 'Đang hỏi về tuyển sinh'
        elif any(word in query_text for word in ['ngành', 'chuyên ngành', 'đào tạo']):
            conv['context_summary'] = 'Đang tìm hiểu về ngành học'
        elif any(word in query_text for word in ['cơ sở', 'phòng', 'trang thiết bị']):
            conv['context_summary'] = 'Đang hỏi về cơ sở vật chất'
        else:
            conv['context_summary'] = 'Hỏi đáp chung về BDU'

class GeminiResponseGenerator:
    """Gemini API Response Generator cho Giảng viên BDU"""
    
    def __init__(self, api_key: str = None):
        from django.conf import settings
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = "gemini-1.5-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        self.memory = ConversationMemory(max_history=10)
        
        # ✅ UPDATED: Role consistency for lecturers
        self.role_consistency_rules = {
            'identity': 'AI assistant của Đại học Bình Dương (BDU) hỗ trợ giảng viên',
            'personality': 'lịch sự, chuyên nghiệp, tôn trọng',
            'knowledge_scope': 'chuyên về thông tin BDU và hỗ trợ giảng viên',
            'addressing': 'luôn xưng hô thầy/cô, không bao giờ dùng bạn/mình',
            'prohibited_roles': [
                'sinh viên', 'học sinh', 'phụ huynh', 'người ngoài trường'
            ]
        }
        
        logger.info("✅ Gemini Response Generator for LECTURERS initialized")
    
    def generate_response(self, query: str, context: Optional[Dict] = None, 
                          intent_info: Optional[Dict] = None, entities: Optional[Dict] = None,
                          session_id: str = None) -> Dict[str, Any]:
        """Tạo phản hồi cho giảng viên với bộ nhớ hội thoại"""
        start_time = time.time()
        
        print(f"\n--- LECTURER REQUEST (Session: {session_id}) ---")
        print(f"🧠 MEMORY DEBUG: Total active sessions = {len(self.memory.conversations)}")

        try:
            # 1. Lấy ngữ cảnh hội thoại
            conversation_context = {}
            if session_id:
                conversation_context = self.memory.get_conversation_context(session_id)
                print(f"🧠 MEMORY DEBUG: History length = {len(conversation_context.get('history', []))}")
                print(f"🧠 MEMORY DEBUG: Context summary = {conversation_context.get('context_summary', 'None')}")

            
            # 2. Xác định chiến lược phản hồi cho giảng viên
            response_strategy = self._determine_lecturer_response_strategy(
                query, context, intent_info, conversation_context
            )
            
            # ✅ ENHANCED: Check for special lecturer instructions
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'direct_answer_lecturer':
                response = self._generate_direct_lecturer_answer(query, context)
            elif instruction == 'enhance_answer_lecturer':
                response = self._generate_enhanced_lecturer_answer(query, context, intent_info, entities, session_id)
            elif instruction == 'clarification_needed':
                response = self._generate_clarification_request(query, context)
            elif instruction == 'dont_know_lecturer':
                response = self._generate_dont_know_response(query, context)
            else:
                # 3. Kiểm tra ngoài phạm vi (cho giảng viên)
                if context and context.get('emergency_education', False):
                    print(f"🚨 GEMINI: Emergency education mode activated")
                    pass 
                elif not self._is_lecturer_education_related(query) and not context.get('force_education_response', False):
                    response = self._get_contextual_out_of_scope_response_lecturer(conversation_context)
                    
                    if session_id:
                        self.memory.add_interaction(session_id, query, response, intent_info, entities)
                    
                    return {
                        'response': response,
                        'method': 'out_of_scope_lecturer',
                        'confidence': 0.9,
                        'generation_time': time.time() - start_time
                    }
                
                # 4. Xây dựng prompt cho giảng viên
                enhanced_prompt = self._build_lecturer_context_aware_prompt(
                    query, context, intent_info, entities, response_strategy, conversation_context
                )
                
                # 5. Gọi Gemini API
                response = self._call_gemini_api_optimized(enhanced_prompt, response_strategy)
                
                # 6. Hậu xử lý để đảm bảo nhất quán cho giảng viên
                if response:
                    response = self._post_process_with_lecturer_consistency(
                        response, query, context, response_strategy, conversation_context
                    )
            
            final_response = response or self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context)
            
            # 7. Lưu vào bộ nhớ
            if session_id:
                print(f"🧠 MEMORY DEBUG: Saving interaction to memory...")
                self.memory.add_interaction(session_id, query, final_response, intent_info, entities)
                print(f"🧠 MEMORY DEBUG: Memory saved. New history length = {len(self.memory.conversations.get(session_id, {}).get('history', []))}")

            return {
                'response': final_response,
                'method': f'lecturer_aware_gemini_{response_strategy}',
                'strategy': response_strategy,
                'conversation_context': conversation_context,
                'generation_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            fallback_response = self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context)
            
            if session_id:
                self.memory.add_interaction(session_id, query, fallback_response, intent_info, entities)
            
            return {
                'response': fallback_response,
                'method': 'lecturer_context_aware_fallback',
                'error': str(e),
                'generation_time': time.time() - start_time
            }

    def _generate_direct_lecturer_answer(self, query, context):
        """Generate direct answer for lecturers with high confidence"""
        
        prompt = f"""
        {LECTURER_SYSTEM_PROMPT}
        
        NHIỆM VỤ: Trả lời TRỰC TIẾP cho giảng viên BDU
        
        CÂU HỎI GIẢNG VIÊN: {query}
        
        THÔNG TIN CHÍNH XÁC TỪ CSDL:
        {context['db_answer']}
        
        YÊU CẦU:
        - Dùng CHÍNH XÁC thông tin từ CSDL
        - Bắt đầu: "Dạ thầy/cô,"
        - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        - NGẮN GỌN, đi thẳng vào vấn đề
        - KHÔNG format phức tạp
        
        Trả lời:
        """
        
        response = self._call_gemini_api_optimized(prompt, 'direct_enhance')
        return response or f"Dạ thầy/cô, {context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
    
    def _generate_enhanced_lecturer_answer(self, query, context, intent_info, entities, session_id):
        """Generate enhanced answer for lecturers"""
        
        prompt = f"""
        {LECTURER_SYSTEM_PROMPT}
        
        NHIỆM VỤ: Trả lời có bổ sung cho giảng viên BDU
        
        CÂU HỎI GIẢNG VIÊN: {query}
        
        THÔNG TIN LIÊN QUAN TỪ CSDL:
        {context['db_answer']}
        
        YÊU CẦU:
        - Sử dụng thông tin CSDL làm gốc
        - Bổ sung ngữ cảnh phù hợp nếu cần
        - Bắt đầu: "Dạ thầy/cô,"
        - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        - NGẮN GỌN, 2-3 câu tối đa
        
        Trả lời:
        """
        
        response = self._call_gemini_api_optimized(prompt, 'balanced')
        return response or f"Dạ thầy/cô, {context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
    
    def _generate_clarification_request(self, query, context):
        """Generate clarification request for lecturers"""
        
        # Extract key topic from query for targeted clarification
        query_words = query.lower().split()
        key_topics = []
        
        topic_keywords = {
            'ngân hàng đề thi': ['ngân hàng', 'đề thi', 'đề'],
            'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ'],
            'tạp chí': ['tạp chí', 'bài viết'],
            'thi đua khen thưởng': ['thi đua', 'khen thưởng'],
            'giờ chuẩn': ['giờ', 'chuẩn'],
            'nghiên cứu': ['nghiên cứu'],
            'báo cáo': ['báo cáo'],
            'lịch giảng dạy': ['lịch', 'giảng dạy', 'thời khóa biểu']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_words for kw in keywords):
                key_topics.append(topic)
        
        if key_topics:
            topic_text = key_topics[0]
            return f"Dạ thầy/cô, để em hỗ trợ chính xác về {topic_text}, thầy/cô có thể nói rõ hơn về nội dung cụ thể cần hỗ trợ không ạ? 🎓"
        else:
            return f"Dạ thầy/cô, để em hỗ trợ chính xác nhất, thầy/cô có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
    
    def _generate_dont_know_response(self, query, context):
        """Generate don't know response for lecturers"""
        
        # Suggest relevant departments based on query content
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ngân hàng đề', 'đề thi', 'khảo thí']):
            dept = "Phòng Đảm bảo chất lượng và Khảo thí"
            contact = "ldkham@bdu.edu.vn"
        elif any(word in query_lower for word in ['kê khai', 'nhiệm vụ', 'giờ chuẩn']):
            dept = "Phòng Tổ chức - Cán bộ"
            contact = "tcccb@bdu.edu.vn"
        elif any(word in query_lower for word in ['tạp chí', 'nghiên cứu', 'khoa học']):
            dept = "Phòng Nghiên cứu - Hợp tác"
            contact = "nghiencuu@bdu.edu.vn"
        elif any(word in query_lower for word in ['khen thưởng', 'thi đua']):
            dept = "Phòng Tổ chức - Cán bộ"
            contact = "tcccb@bdu.edu.vn"
        else:
            dept = "phòng ban liên quan"
            contact = "info@bdu.edu.vn"
        
        return f"Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ {dept} qua email {contact} để được hỗ trợ chi tiết ạ. 🎓"

    def _determine_lecturer_response_strategy(self, query, context, intent_info, conversation_context):
        """Xác định chiến lược phản hồi cho giảng viên"""
        
        has_real_history = bool(conversation_context.get('history') and len(conversation_context['history']) > 0)
        
        print(f"🔍 LECTURER STRATEGY DEBUG: has_real_history = {has_real_history}")
        if not has_real_history:
             print("🔍 LECTURER STRATEGY DEBUG: No history, using standard strategy...")
        else:
            # ✅ ENHANCED: Lecturer-specific follow-up detection
            last_interaction = conversation_context['history'][-1]
            last_query = last_interaction['user_query'].lower()
            current_query = query.lower()
            
            print(f"🔍 LECTURER STRATEGY DEBUG: last_query = '{last_query[:50]}...'")
            print(f"🔍 LECTURER STRATEGY DEBUG: current_query = '{current_query[:50]}...'")
            
            # ✅ LECTURER-SPECIFIC topics
            lecturer_topics = {
                'ngân hàng đề thi': ['ngân hàng', 'đề thi', 'đề', 'khảo thí'],
                'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ', 'giờ chuẩn'],
                'tạp chí khoa học': ['tạp chí', 'bài viết', 'nghiên cứu'],
                'thi đua khen thưởng': ['thi đua', 'khen thưởng', 'danh hiệu'],
                'báo cáo': ['báo cáo', 'nộp', 'hạn cuối'],
                'lịch giảng dạy': ['lịch', 'giảng dạy', 'thời khóa biểu'],
                'cơ sở vật chất': ['cơ sở', 'phòng', 'trang thiết bị'],
                'học phí': ['học phí', 'phí', 'tiền học', 'chi phí'],
                'tuyển sinh': ['tuyển sinh', 'nhập học', 'đăng ký', 'điểm'],
                'ngành học': ['ngành', 'chuyên ngành', 'khoa', 'đào tạo']
            }
            
            last_main_topic = None
            for topic, keywords in lecturer_topics.items():
                if any(kw in last_query for kw in keywords):
                    last_main_topic = topic
                    break
            
            current_main_topic = None
            for topic, keywords in lecturer_topics.items():
                if any(kw in current_query for kw in keywords):
                    current_main_topic = topic
                    break

            print(f"🔍 LECTURER STRATEGY DEBUG: last_main_topic = {last_main_topic}, current_main_topic = {current_main_topic}")

            has_exact_same_topic = last_main_topic is not None and last_main_topic == current_main_topic
            
            strong_continuation_words = ['còn', 'thêm', 'nữa', 'khác', 'và', 'tiếp theo']
            has_strong_continuation = any(word in current_query.split() for word in strong_continuation_words)
            
            strong_clarification_words = ['cụ thể hơn', 'rõ hơn', 'chi tiết hơn', 'giải thích thêm']
            has_strong_clarification = any(phrase in current_query for phrase in strong_clarification_words)
            
            memory_test_words = ['nhớ không', 'hỏi gì', 'nói gì trước', 'vừa nói', 'tổng hợp']
            is_memory_test = any(word in current_query for word in memory_test_words)

            print(f"🔍 LECTURER STRATEGY DEBUG: has_exact_same_topic = {has_exact_same_topic}")
            print(f"🔍 LECTURER STRATEGY DEBUG: has_strong_continuation = {has_strong_continuation}")
            print(f"🔍 LECTURER STRATEGY DEBUG: has_strong_clarification = {has_strong_clarification}")

            # ĐIỀU KIỆN NGHIÊM NGẶT CHO CÁC CHIẾN LƯỢC NGỮ CẢNH
            if has_strong_continuation and has_exact_same_topic:
                print(f"💡 LECTURER STRATEGY SELECTED: → follow_up_continuation")
                return 'follow_up_continuation'
            
            if has_strong_clarification and has_exact_same_topic:
                print(f"💡 LECTURER STRATEGY SELECTED: → follow_up_clarification")
                return 'follow_up_clarification'

            if is_memory_test:
                 print(f"💡 LECTURER STRATEGY SELECTED: → memory_reference")
                 return 'memory_reference'
                 
            if current_main_topic is not None and last_main_topic is not None and current_main_topic != last_main_topic:
                print(f"💡 LECTURER STRATEGY SELECTED: → topic_shift")
                return 'topic_shift'
        
        # MẶC ĐỊNH: Sử dụng logic chiến lược cơ bản cho giảng viên
        print(f"🔍 LECTURER STRATEGY DEBUG: No clear follow-up detected, using standard strategy logic...")
        
        if isinstance(context, dict) and context.get('confidence', 0) > 0.7:
            print(f"💡 LECTURER STRATEGY SELECTED: → direct_enhance")
            return 'direct_enhance'
        
        if intent_info and intent_info.get('intent') in ['greeting', 'general'] and len(query.split()) <= 5:
            print(f"💡 LECTURER STRATEGY SELECTED: → quick_clarify")
            return 'quick_clarify'
        
        if any(word in query.lower() for word in ['khó khăn', 'cần gấp', 'hạn cuối', 'urgent']):
            print(f"💡 LECTURER STRATEGY SELECTED: → supportive_brief")
            return 'supportive_brief'
        
        print(f"💡 LECTURER STRATEGY SELECTED: → balanced (default)")
        return 'balanced'

    def _build_lecturer_context_aware_prompt(self, query, context, intent_info, entities, strategy, conversation_context):
        """Xây dựng prompt cho giảng viên với LECTURER_SYSTEM_PROMPT"""
        
        base_personality = f"""
        {LECTURER_SYSTEM_PROMPT}

        🤖 QUY TẮC VAI TRÒ NGHIÊM NGẶT:
        - LUÔN giữ vai trò: "{self.role_consistency_rules['identity']}"
        - KHÔNG BAO GIỜ xưng hô là: {', '.join(self.role_consistency_rules['prohibited_roles'])}
        - LUÔN nói "em là AI assistant của BDU hỗ trợ giảng viên" nếu được hỏi về vai trò.

        🗣️ PHONG CÁCH CHO GIẢNG VIÊN:
        - LUÔN bắt đầu câu trả lời bằng "Dạ thầy/cô,".
        - LUÔN kết thúc bằng "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        - Dùng emoji phù hợp (🎓, 📚, 📊, 📋).
        - TUYỆT ĐỐI KHÔNG nói dài dòng hay lặp lại.
        - ĐI THẲNG VÀO TRỌNG TÂM.
        """
        
        context_info = str(context.get('response', '')) if isinstance(context, dict) else str(context or '')
        
        memory_context = ""
        conversation_flow = ""
        
        if conversation_context.get('history'):
            flow_items = []
            for h in conversation_context['history'][-3:]:
                flow_items.append(f"Thầy/cô hỏi: '{h['user_query'][:40]}...' -> Em trả lời: '{h['bot_response'][:50]}...'")
            
            conversation_flow = "\n".join(flow_items)
            
            memory_context = f"""
            ---
            📚 NGỮ CẢNH HỘI THOẠI TRƯỚC VỚI GIẢNG VIÊN:
            - Chủ đề chính đang thảo luận: {conversation_context.get('context_summary', 'Chung')}
            - Các lĩnh vực thầy/cô quan tâm: {', '.join(conversation_context.get('user_interests', [])) or 'Chưa rõ'}
            - Dòng chảy hội thoại gần đây:
            {conversation_flow}
            ---
            """
        
        strategy_prompts = {
            'follow_up_continuation': f"""
            {base_personality}
            {memory_context}
            NHIỆM VỤ: Thầy/cô đang hỏi tiếp về CÙNG CHỦ ĐỀ.
            ⚠️ KIỂM TRA: Thầy/cô hỏi "{query}". Đây là câu hỏi tiếp nối về chủ đề "{conversation_context.get('context_summary', 'trước đó')}".
            HÀNH ĐỘNG: Cung cấp thông tin BỔ SUNG, đừng lặp lại ý cũ. Bắt đầu bằng "Dạ thầy/cô, ngoài ra về [chủ đề]..." hoặc một cách tự nhiên. Trả lời ngắn gọn.
            DỮ LIỆU THAM KHẢO (nếu có): {context_info}
            Trả lời:
            """,
            
            'follow_up_clarification': f"""
            {base_personality}
            {memory_context}
            NHIỆM VỤ: Thầy/cô muốn làm RÕ HƠN về CÙNG CHỦ ĐỀ.
            ⚠️ KIỂM TRA: Thầy/cô hỏi "{query}". Đây là yêu cầu làm rõ về chủ đề "{conversation_context.get('context_summary', 'trước đó')}".
            HÀNH ĐỘNG: Giải thích chi tiết, cụ thể hơn. Bắt đầu bằng "Dạ thầy/cô, để làm rõ hơn về [chủ đề]...".
            DỮ LIỆU THAM KHẢO (nếu có): {context_info}
            Trả lời:
            """,
            
            'topic_shift': f"""
            {base_personality}
            {memory_context}
            NHIỆM VỤ: Thầy/cô đã CHUYỂN SANG một chủ đề MỚI.
            ⚠️ KIỂM TRA: Thầy/cô hỏi "{query}". Chủ đề này khác với chủ đề trước đó.
            HÀNH ĐỘNG: Trả lời trực tiếp vào chủ đề mới. TUYỆT ĐỐI KHÔNG dùng các cụm từ như "như đã nói", "ngoài ra". Có thể thừa nhận sự thay đổi một cách nhẹ nhàng nếu muốn.
            DỮ LIỆU THAM KHẢO (nếu có): {context_info}
            Trả lời:
            """,
            
            'memory_reference': f"""
            {base_personality}
            {memory_context}
            NHIỆM VỤ: Thầy/cô đang hỏi về những gì đã nói (kiểm tra trí nhớ).
            ⚠️ KIỂM TRA: Thầy/cô hỏi "{query}".
            HÀNH ĐỘNG: Dựa vào 'Dòng chảy hội thoại gần đây' để tóm tắt ngắn gọn 1-2 ý chính đã trao đổi. Hỏi xem thầy/cô muốn biết thêm gì không.
            Trả lời:
            """,
            
            'balanced': f"""
            {base_personality}
            {memory_context if 'history' in conversation_context else ''}
            NHIỆM VỤ: Trả lời câu hỏi của thầy/cô một cách tự nhiên, cân bằng.
            ⚠️ KIỂM TRA: Thầy/cô hỏi "{query}". Đây có vẻ là một câu hỏi mới hoặc không có liên kết rõ ràng.
            HÀNH ĐỘNG: Trả lời trực tiếp, ngắn gọn, đi thẳng vào vấn đề. KHÔNG tham chiếu đến hội thoại trước trừ khi câu hỏi CỰC KỲ liên quan.
            DỮ LIỆU THAM KHẢO (nếu có): {context_info}
            Trả lời:
            """
        }

        # Dùng 'balanced' làm prompt mặc định cho các strategy khác chưa được định nghĩa riêng
        final_prompt = strategy_prompts.get(strategy, strategy_prompts['balanced'])
        print(f"📝 LECTURER PROMPT DEBUG (Strategy: {strategy}):\n{final_prompt[:400]}...") # In ra một phần prompt để debug
        return final_prompt

    def _post_process_with_lecturer_consistency(self, response, query, context, strategy, conversation_context):
        """Post-process để đảm bảo nhất quán cho giảng viên"""
        if not response:
            return response
        
        # 1. Sửa các vi phạm vai trò cho giảng viên
        prohibited_phrases = [
            'với tư cách là sinh viên', 'tôi là học sinh',
            'bạn', 'mình', 'anh', 'chị', 'em là sinh viên'
        ]
        for phrase in prohibited_phrases:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, 'em là AI assistant của BDU')
        
        # 2. ✅ CRITICAL: Sửa xưng hô không đúng
        response = re.sub(r'\bbạn\b', 'thầy/cô', response, flags=re.IGNORECASE)
        response = re.sub(r'\bmình\b', 'em', response, flags=re.IGNORECASE)
        response = re.sub(r'\btôi\b', 'em', response, flags=re.IGNORECASE)
        
        # 3. ✅ CRITICAL: Đảm bảo bắt đầu bằng "Dạ thầy/cô"
        response_stripped = response.strip()
        if not response_stripped.lower().startswith('dạ thầy/cô'):
            if response_stripped.lower().startswith('dạ'):
                response = 'Dạ thầy/cô, ' + response_stripped[3:].strip()
            else:
                response = 'Dạ thầy/cô, ' + response_stripped
        
        # 4. ✅ CRITICAL: Đảm bảo kết thúc đúng cách
        if not response.strip().endswith('Thầy/cô có cần hỗ trợ thêm gì không ạ?'):
            # Remove existing endings first
            response = re.sub(r'\s*(Thầy/cô có.*?không ạ\?|Cần.*?không\?|Có.*?không\?)?\s*$', '', response.strip())
            response += ' Thầy/cô có cần hỗ trợ thêm gì không ạ?'
        
        # 5. ✅ REMOVE: Loại bỏ format phức tạp
        response = re.sub(r'\*\*\d+\.\s*', '', response)  # Remove **1. **2. etc
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)  # Remove numbered lists
        response = re.sub(r'^\s*[•\-\*]\s*', '', response, flags=re.MULTILINE)  # Remove bullets
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Remove bold formatting
        
        return response.strip()
    
    def _get_contextual_out_of_scope_response_lecturer(self, conversation_context):
        """Out of scope response cho giảng viên"""
        if conversation_context.get('context_summary'):
            return f"Dạ thầy/cô, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 Thầy/cô còn muốn hỏi gì về {conversation_context['context_summary'].lower()} không ạ?"
        
        return "Dạ thầy/cô, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 Thầy/cô có câu hỏi nào khác về trường không ạ?"
    
    def _get_smart_fallback_with_context_lecturer(self, query, intent_info, conversation_context):
        """Smart fallback với conversation context cho giảng viên"""
        intent_name = intent_info.get('intent', 'general') if intent_info else 'general'
        
        if conversation_context.get('context_summary'):
            summary = conversation_context['context_summary']
            context_fallbacks = {
                'Đang hỏi về ngân hàng đề thi': "Dạ thầy/cô, về ngân hàng đề thi, em có thể hỗ trợ thêm! 📋 Thầy/cô có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về kê khai nhiệm vụ năm học': "Dạ thầy/cô, về kê khai nhiệm vụ năm học, em có thể hỗ trợ thêm! 📊 Thầy/cô có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về tạp chí khoa học': "Dạ thầy/cô, về tạp chí khoa học, em có thể hỗ trợ thêm! 📚 Thầy/cô có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về thi đua khen thưởng': "Dạ thầy/cô, về thi đua khen thưởng, em có thể hỗ trợ thêm! 🏆 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
            }
            if summary in context_fallbacks:
                return context_fallbacks[summary]
        
        smart_fallbacks = {
            'greeting': "Dạ chào thầy/cô! 👋 Em có thể hỗ trợ gì cho thầy/cô về BDU ạ?",
            'general': "Dạ thầy/cô, em sẵn sàng hỗ trợ các vấn đề liên quan đến BDU! 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        }
        
        return smart_fallbacks.get(intent_name, smart_fallbacks['general'])
    
    def _is_lecturer_education_related(self, query):
        """Check if education related for lecturers - enhanced keywords"""
        lecturer_education_keywords = [
            # Cơ bản
            'trường', 'học', 'sinh viên', 'tuyển sinh', 'học phí', 'ngành', 
            'đại học', 'bdu', 'gv', 'giảng viên', 'dạy', 'quy định',
            
            # ✅ LECTURER-SPECIFIC
            'hội đồng', 'nghiên cứu', 'công tác', 'báo cáo', 'đánh giá',
            'thi đua', 'thành tích', 'khen thưởng', 'xét', 'xét thi đua',
            'nhiệm vụ', 'chức năng', 'tiêu chuẩn', 'tiêu chí', 'định mức',
            'kiểm tra', 'giám sát', 'quản lý', 'kết quả', 'hiệu quả',
            'phân công', 'giao nhiệm vụ', 'trách nhiệm', 'chuẩn đầu ra',
            'học kỳ', 'năm học', 'kỳ thi', 'bài giảng', 'giáo án',
            'lớp học', 'môn học', 'học phần', 'tín chỉ', 'cố vấn',
            'ngân hàng đề thi', 'file mềm', 'nộp', 'email', 'phòng ban',
            'kê khai', 'giờ chuẩn', 'thỉnh giảng', 'tạp chí', 'bài viết',
            
            # Không dấu
            'truong', 'hoc', 'sinh vien', 'tuyen sinh', 'hoc phi', 'nganh',
            'dai hoc', 'giang vien', 'day', 'quy dinh', 'nghien cuu',
            'thi dua', 'thanh tich', 'khen thuong', 'nhiem vu', 'chuc nang',
            'tieu chuan', 'tieu chi', 'dinh muc', 'kiem tra', 'giam sat',
            'quan ly', 'ket qua', 'hieu qua', 'phan cong', 'giao nhiem vu',
            'hoc ky', 'nam hoc', 'ky thi', 'bai giang', 'giao an',
            'lop hoc', 'mon hoc', 'hoc phan', 'tin chi', 'co van',
            'ngan hang de thi', 'file mem', 'ke khai', 'gio chuan',
            'thinh giang', 'tap chi', 'bai viet'
        ]
        
        if not query:
            return False
        
        query_lower = query.lower()
        return any(kw in query_lower for kw in lecturer_education_keywords)

    # Keep existing methods but ensure they're adapted for lecturers
    def _call_gemini_api_optimized(self, prompt: str, strategy: str) -> Optional[str]:
        """Call Gemini API - same as before"""
        try:
            headers = {'Content-Type': 'application/json'}
            generation_configs = {
                'quick_clarify': {"temperature": 0.3, "maxOutputTokens": 60},
                'direct_enhance': {"temperature": 0.4, "maxOutputTokens": 120},
                'conversational_brief': {"temperature": 0.6, "maxOutputTokens": 90},
                'structured_info': {"temperature": 0.2, "maxOutputTokens": 200},
                'supportive_brief': {"temperature": 0.5, "maxOutputTokens": 150},
                'follow_up_continuation': {"temperature": 0.4, "maxOutputTokens": 120},
                'follow_up_clarification': {"temperature": 0.3, "maxOutputTokens": 180},
                'topic_shift': {"temperature": 0.5, "maxOutputTokens": 120},
                'memory_reference': {"temperature": 0.2, "maxOutputTokens": 100},
                'balanced': {"temperature": 0.5, "maxOutputTokens": 150}
            }
            
            config = generation_configs.get(strategy, generation_configs['balanced'])
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": config,
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        return candidate['content']['parts'][0]['text']
            else:
                logger.error(f"Gemini API Error {response.status_code}: {response.text}")

            return None
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return None
    
    def get_conversation_memory(self, session_id: str):
        return self.memory.get_conversation_context(session_id)
    
    def clear_conversation_memory(self, session_id: str = None):
        if session_id:
            if session_id in self.memory.conversations:
                del self.memory.conversations[session_id]
        else:
            self.memory.conversations.clear()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for lecturers"""
        try:
            test_prompt = "Test ngắn cho giảng viên"
            response = self._call_gemini_api_optimized(test_prompt, 'quick_clarify')
            
            return {
                'gemini_api_available': response is not None,
                'api_key_configured': bool(self.api_key),
                'service_status': 'active' if response else 'error',
                'mode': 'lecturer_focused_with_memory',
                'memory_sessions': len(self.memory.conversations),
                'features': [
                    'lecturer_conversation_memory',
                    'lecturer_role_consistency',
                    'lecturer_context_aware_responses',
                    'lecturer_follow_up_detection',
                    'lecturer_topic_shift_handling',
                    'lecturer_clarification_requests',
                    'lecturer_department_suggestions'
                ]
            }
        except Exception as e:
            return {
                'gemini_api_available': False,
                'service_status': 'error',
                'error': str(e)
            }