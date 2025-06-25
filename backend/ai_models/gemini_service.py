import logging
import time
import requests
import json
import re
from typing import Dict, Any, Optional, List
from unidecode import unidecode
import difflib
import pandas as pd

logger = logging.getLogger(__name__)

def get_fallback_system_prompt(response_style='professional'):
    """Get fallback system prompt based on response style"""
    
    base_prompt = """Bạn là AI assistant của Đại học Bình Dương (BDU), chuyên hỗ trợ giảng viên.

🎯 QUY TẮC QUAN TRỌNG:
- LUÔN xưng hô: "thầy/cô" (TUYỆT ĐỐI KHÔNG dùng "bạn", "mình", "anh/chị")
- Bắt đầu: "Dạ thầy/cô,"
- Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
- KHÔNG CHẾ TẠO thông tin không có
- KHÔNG dùng format phức tạp với **1. **2. hay bullets"""

    style_additions = {
        'professional': """

✅ PHONG CÁCH CHUYÊN NGHIỆP:
- Ngôn từ trang trọng, lịch sự, chuẩn mực
- Sử dụng thuật ngữ chính xác và phù hợp
- Trình bày có hệ thống, logic rõ ràng
- Tôn trọng cấp bậc và quy trình
- NGẮN GỌN - Chỉ 1-2 câu chính, đi thẳng vào vấn đề""",

        'friendly': """

✅ PHONG CÁCH THÂN THIỆN:
- Ngôn từ gần gũi, ấm áp và dễ chịu
- Sử dụng emoji phù hợp để tạo không khí vui vẻ 😊
- Tạo cảm giác thoải mái, gần gũi
- Giọng điệu vui vẻ, nhiệt tình
- Độ dài vừa phải, khoảng 2-3 câu""",

        'technical': """

✅ PHONG CÁCH KỸ THUẬT:
- Sử dụng thuật ngữ chuyên môn chính xác
- Giải thích chi tiết các khía cạnh kỹ thuật  
- Đưa ra ví dụ cụ thể, số liệu thực tế
- Tập trung vào độ chính xác và đầy đủ
- Có thể dài hơn để giải thích kỹ thuật (3-4 câu)""",

        'brief': """

✅ PHONG CÁCH NGẮN GỌN:
- Trả lời súc tích, đi thẳng vào trọng tâm
- Tối đa 1 câu cho mỗi ý chính
- Không giải thích dài dòng hay lòng vòng
- Tập trung vào thông tin cốt lõi nhất
- Loại bỏ các chi tiết không cần thiết""",

        'detailed': """

✅ PHONG CÁCH CHI TIẾT:
- Giải thích đầy đủ, toàn diện từng khía cạnh
- Đưa ra nhiều ví dụ minh họa cụ thể
- Phân tích từ nhiều góc độ khác nhau
- Cung cấp ngữ cảnh và background rộng
- Có thể dài 4-5 câu để giải thích đầy đủ"""
    }
    
    return base_prompt + style_additions.get(response_style, style_additions['professional'])

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

class SimpleVietnameseRestorer:
    """
    Simple Vietnamese accent restorer using Gemini API
    - Based on your original code sample
    - Minimal complexity, maximum effectiveness
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        # Simple cache to avoid repeated API calls
        self.cache = {}
        self.max_cache_size = 500
        
        # Rate limiting - simple approach
        self.last_call_time = 0
        self.min_interval = 4  # 4 seconds between calls (15 calls/min = 4s interval)
        
        logger.info("✅ SimpleVietnameseRestorer initialized")
    
    def has_vietnamese_accents(self, text: str) -> bool:
        """Check if text has Vietnamese accents"""
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        vietnamese_chars += vietnamese_chars.upper()
        return any(char in vietnamese_chars for char in text)
    
    def restore_vietnamese_tone(self, input_text: str) -> str:
        """Restore Vietnamese accents using Gemini API - exactly like your sample"""
        if not input_text or not input_text.strip():
            return input_text
        
        input_text = input_text.strip()
        
        # Check cache first
        cache_key = input_text.lower()
        if cache_key in self.cache:
            logger.debug(f"🎯 Cache hit for: '{input_text}'")
            return self.cache[cache_key]
        
        # If already has accents, return as is
        if self.has_vietnamese_accents(input_text):
            self.cache[cache_key] = input_text
            return input_text
        
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_call_time < self.min_interval:
            logger.warning(f"⚠️ Rate limiting: skipping API call for '{input_text}'")
            self.cache[cache_key] = input_text
            return input_text
        
        # Call Gemini API
        prompt = f'Hãy viết lại câu sau thành tiếng Việt có dấu đầy đủ, không thay đổi ý nghĩa: "{input_text}"'
        
        try:
            self.last_call_time = current_time
            
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 100,
                    "topP": 0.8
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        restored_text = candidate['content']['parts'][0]['text'].strip()
                        
                        # Clean up response
                        restored_text = re.sub(r'^["\'](.*)["\']$', r'\1', restored_text)
                        restored_text = re.sub(r'^(Câu đã có dấu:|Kết quả:|Trả lời:)\s*', '', restored_text, flags=re.IGNORECASE)
                        
                        # Simple validation
                        if self._is_valid_restoration(input_text, restored_text):
                            logger.info(f"✅ Restored: '{input_text}' -> '{restored_text}'")
                            self._cache_result(cache_key, restored_text)
                            return restored_text
                        else:
                            logger.warning(f"⚠️ Invalid restoration: '{restored_text}'")
            elif response.status_code == 429:
                logger.warning(f"⚠️ Rate limit hit, increasing interval")
                self.min_interval = min(10, self.min_interval + 1)
            else:
                logger.error(f"❌ Gemini API Error {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error restoring tone: {e}")
        
        # Fallback: return original
        self._cache_result(cache_key, input_text)
        return input_text
    
    def _is_valid_restoration(self, original: str, restored: str) -> bool:
        """Simple validation"""
        if not restored or len(restored.strip()) == 0:
            return False
        
        # Check length difference
        if abs(len(restored) - len(original)) > len(original) * 0.5:
            return False
        
        # Check similarity without accents
        original_no_accent = unidecode(original).lower()
        restored_no_accent = unidecode(restored).lower()
        
        similarity = difflib.SequenceMatcher(None, original_no_accent, restored_no_accent).ratio()
        return similarity >= 0.8
    
    def _cache_result(self, key: str, result: str):
        """Cache result with size management"""
        self.cache[key] = result
        
        # Simple cache management
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.cache) // 5
            keys_to_remove = list(self.cache.keys())[:items_to_remove]
            for k in keys_to_remove:
                del self.cache[k]

class GeminiResponseGenerator:
    """Gemini API Response Generator cho Giảng viên BDU với Enhanced Personalization"""
    
    def __init__(self, api_key: str = None):
        from django.conf import settings
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = "gemini-2.0-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        self.memory = ConversationMemory(max_history=10)
        self.vietnamese_restorer = SimpleVietnameseRestorer(self.api_key)
        
        # User context cache for personalization
        self._user_context_cache = {}
        
        # ✅ FIXED: Style-specific generation configs
        self.style_generation_configs = {
            'professional': {
                "temperature": 0.3,
                "maxOutputTokens": 120,
                "topP": 0.8
            },
            'friendly': {
                "temperature": 0.6,
                "maxOutputTokens": 150,
                "topP": 0.9
            },
            'technical': {
                "temperature": 0.2,
                "maxOutputTokens": 200,
                "topP": 0.7
            },
            'brief': {
                "temperature": 0.4,
                "maxOutputTokens": 80,
                "topP": 0.8
            },
            'detailed': {
                "temperature": 0.5,
                "maxOutputTokens": 250,
                "topP": 0.8
            }
        }
        
        # Role consistency rules (unchanged)
        self.role_consistency_rules = {
            'identity': 'AI assistant của Đại học Bình Dương (BDU) hỗ trợ giảng viên',
            'personality': 'lịch sự, chuyên nghiệp, tôn trọng',
            'knowledge_scope': 'chuyên về thông tin BDU và hỗ trợ giảng viên',
            'addressing': 'luôn xưng hô thầy/cô, không bao giờ dùng bạn/mình',
            'prohibited_roles': [
                'sinh viên', 'học sinh', 'phụ huynh', 'người ngoài trường'
            ]
        }
        
        logger.info("✅ Gemini Response Generator initialized with Enhanced Style Support")

    # ✅ NEW: Personalization methods
    def set_user_context(self, session_id: str, user_context: dict):
        """Set user context cho session (được gọi từ chat API)"""
        self._user_context_cache[session_id] = user_context
        logger.info(f"✅ Set user context for session {session_id}: {user_context.get('faculty_code', 'Unknown')}")

    def _get_personalized_system_prompt(self, session_id: str = None):
        """Lấy personalized system prompt từ user context"""
        try:
            if not session_id or session_id not in self._user_context_cache:
                return get_fallback_system_prompt()
            
            user_context = self._user_context_cache[session_id]
            if 'personalized_prompt' in user_context:
                logger.info(f"✅ Using personalized prompt for {user_context.get('faculty_code', 'Unknown')}")
                return user_context['personalized_prompt']
            
            return get_fallback_system_prompt()
            
        except Exception as e:
            logger.error(f"Error getting personalized prompt: {e}")
            return get_fallback_system_prompt()
    
    # ✅ NEW: Get response style from user context
    def _get_user_response_style(self, session_id: str = None):
        """Get user's preferred response style"""
        try:
            if session_id and session_id in self._user_context_cache:
                user_context = self._user_context_cache[session_id]
                preferences = user_context.get('preferences', {})
                return preferences.get('response_style', 'professional')
            return 'professional'
        except Exception as e:
            logger.error(f"Error getting response style: {e}")
            return 'professional'    

    # ✅ ENHANCED: Generate response with style-aware processing
    def generate_response(self, query: str, context: Optional[Dict] = None, 
                      intent_info: Optional[Dict] = None, entities: Optional[Dict] = None,
                      session_id: str = None) -> Dict[str, Any]:
        """Generate response with enhanced style-aware generation"""
        start_time = time.time()
        
        print(f"\n--- ENHANCED STYLE-AWARE REQUEST (Session: {session_id}) ---")
        print(f"🧠 MEMORY DEBUG: Total active sessions = {len(self.memory.conversations)}")

        try:
            original_query = query
            if not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Query restored: '{query}' -> '{restored_query}'")
                    query = restored_query

            # ✅ NEW: Get user's response style
            user_response_style = self._get_user_response_style(session_id)
            print(f"🎨 USER STYLE: {user_response_style}")

            # Get conversation context
            conversation_context = {}
            if session_id:
                conversation_context = self.memory.get_conversation_context(session_id)
                print(f"🧠 MEMORY DEBUG: History length = {len(conversation_context.get('history', []))}")

            # Get user context for personalization
            user_context = None
            if session_id and session_id in self._user_context_cache:
                user_context = self._user_context_cache[session_id]
                print(f"👤 USER CONTEXT: {user_context.get('faculty_code', 'Unknown')} - Style: {user_response_style}")

            # Determine response strategy
            response_strategy = self._determine_lecturer_response_strategy(
                query, context, intent_info, conversation_context
            )
            
            # ✅ FIXED: Handle instruction-based responses
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'direct_answer_lecturer':
                response = self._generate_direct_lecturer_answer(query, context, session_id, user_response_style)
            elif instruction in ['enhance_answer_lecturer', 'enhance_answer_lecturer_boosted']:
                response = self._generate_enhanced_lecturer_answer(query, context, intent_info, entities, session_id, user_response_style)
            elif instruction == 'clarification_needed':
                response = self._generate_clarification_request(query, context, session_id, user_response_style)
            elif instruction == 'dont_know_lecturer':
                response = self._generate_dont_know_response(query, context, session_id, user_response_style)
            else:
                # Check out of scope and generate response
                if context and context.get('emergency_education', False):
                    print(f"🚨 GEMINI: Emergency education mode activated")
                    pass 
                elif not self._is_lecturer_education_related(query) and not context.get('force_education_response', False):
                    response = self._get_contextual_out_of_scope_response_lecturer(conversation_context, session_id, user_response_style)
                    
                    if session_id:
                        self.memory.add_interaction(session_id, original_query, response, intent_info, entities)
                    
                    return {
                        'response': response,
                        'method': 'out_of_scope_lecturer',
                        'confidence': 0.9,
                        'generation_time': time.time() - start_time,
                        'original_query': original_query,
                        'restored_query': query,
                        'personalized': session_id in self._user_context_cache,
                        'response_style': user_response_style  # ✅ NEW
                    }
                
                # ✅ FIXED: Use simplified response generation
                response = self._generate_style_aware_response(query, context, session_id, user_response_style)
            
            final_response = response or self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id, user_response_style)
            
            # Save to memory
            if session_id:
                print(f"🧠 MEMORY DEBUG: Saving interaction to memory...")
                self.memory.add_interaction(session_id, original_query, final_response, intent_info, entities)

            return {
                'response': final_response,
                'method': f'lecturer_aware_gemini_{response_strategy}',
                'strategy': response_strategy,
                'conversation_context': conversation_context,
                'generation_time': time.time() - start_time,
                'original_query': original_query,
                'restored_query': query,
                'vietnamese_restoration_used': query != original_query,
                'personalized': bool(user_context),
                'response_style': user_response_style,  # ✅ NEW
                'style_applied': user_response_style,  # ✅ NEW
                'enhanced_generation': response_strategy == 'enhanced_generation'
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            user_response_style = self._get_user_response_style(session_id)
            fallback_response = self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id, user_response_style)
            
            if session_id:
                self.memory.add_interaction(session_id, original_query, fallback_response, intent_info, entities)
            
            return {
                'response': fallback_response,
                'method': 'lecturer_context_aware_fallback',
                'error': str(e),
                'generation_time': time.time() - start_time,
                'original_query': original_query,
                'restored_query': query,
                'personalized': session_id in self._user_context_cache,
                'response_style': user_response_style  # ✅ NEW
            }

    def _generate_direct_lecturer_answer(self, query, context, session_id=None, response_style='professional'):
        """Generate direct answer with style awareness"""
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        
        prompt = f"""
        {system_prompt}
        
        NHIỆM VỤ: Trả lời TRỰC TIẾP cho giảng viên BDU
        
        CÂU HỎI GIẢNG VIÊN: {query}
        
        THÔNG TIN CHÍNH XÁC TỪ CSDL:
        {context['db_answer']}
        
        YÊU CẦU:
        - Dùng CHÍNH XÁC thông tin từ CSDL
        - Bắt đầu: "Dạ thầy/cô,"
        - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        - Áp dụng phong cách: {response_style}
        - KHÔNG format phức tạp
        
        Trả lời:
        """
        
        response = self._call_gemini_api_with_style(prompt, 'direct_enhance', response_style)
        return response or f"Dạ thầy/cô, {context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
    
    def _generate_enhanced_lecturer_answer(self, query, context, intent_info, entities, session_id, response_style='professional'):
        """Generate enhanced answer with style awareness"""
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        
        is_generation_boosted = (
            context.get('generation_boosted', False) or 
            context.get('instruction') == 'enhance_answer_lecturer_boosted'
        )
        
        if is_generation_boosted:
            prompt = f"""
            {system_prompt}
            
            NHIỆM VỤ ĐẶC BIỆT: Trả lời có BỔ SUNG PHONG PHÚ cho giảng viên BDU
            
            CÂU HỎI GIẢNG VIÊN: {query}
            
            THÔNG TIN CƠ BẢN TỪ CSDL:
            {context['db_answer']}
            
            YÊU CẦU TĂNG CƯỜNG:
            - SỬ DỤNG thông tin CSDL làm nền tảng
            - BỔ SUNG thêm ngữ cảnh, lý do, hoặc hướng dẫn chi tiết
            - GIẢI THÍCH tại sao điều này quan trọng cho giảng viên
            - THÊM tips hoặc lưu ý thực tế nếu phù hợp
            - Áp dụng phong cách: {response_style}
            - Bắt đầu: "Dạ thầy/cô,"
            - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
            
            Trả lời:
            """
        else:
            prompt = f"""
            {system_prompt}
            
            NHIỆM VỤ: Trả lời có bổ sung cho giảng viên BDU
            
            CÂU HỎI GIẢNG VIÊN: {query}
            
            THÔNG TIN LIÊN QUAN TỪ CSDL:
            {context['db_answer']}
            
            YÊU CẦU:
            - Sử dụng thông tin CSDL làm gốc
            - Bổ sung ngữ cảnh phù hợp nếu cần
            - Áp dụng phong cách: {response_style}
            - Bắt đầu: "Dạ thầy/cô,"
            - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
            
            Trả lời:
            """
        
        response = self._call_gemini_api_with_style(prompt, 'enhanced_generation' if is_generation_boosted else 'balanced', response_style)
        return response or f"Dạ thầy/cô, {context['db_answer']} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"

    # ✅ NEW: Style-aware API call
    def _call_gemini_api_with_style(self, prompt: str, strategy: str, response_style: str = 'professional') -> Optional[str]:
        """Call Gemini API with style-specific generation config"""
        try:
            headers = {'Content-Type': 'application/json'}
            
            # Get style-specific config
            style_config = self.style_generation_configs.get(response_style, self.style_generation_configs['professional'])
            
            # Merge with strategy-specific adjustments
            generation_configs = {
                'quick_clarify': {"temperature": style_config["temperature"] * 0.8, "maxOutputTokens": min(style_config["maxOutputTokens"], 80)},
                'direct_enhance': {"temperature": style_config["temperature"], "maxOutputTokens": style_config["maxOutputTokens"]},
                'enhanced_generation': {"temperature": style_config["temperature"] * 1.2, "maxOutputTokens": style_config["maxOutputTokens"] + 50},
                'balanced': style_config
            }
            
            config = generation_configs.get(strategy, style_config)
            
            # Ensure temperature is within bounds
            config["temperature"] = max(0.1, min(1.0, config["temperature"]))
            
            print(f"🎨 STYLE CONFIG: {response_style} -> temp={config['temperature']}, tokens={config['maxOutputTokens']}")
            
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
        
    def _generate_clarification_request(self, query, context, session_id=None, response_style='professional'):
        """Generate clarification request with style"""
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        
        if full_name:
            name_suffix = full_name.split()[-1]
            personal_address = f"thầy/cô {name_suffix}"
        else:
            personal_address = "thầy/cô"
        
        # Style-specific clarification
        if response_style == 'friendly':
            return f"Dạ {personal_address}, để em có thể hỗ trợ {personal_address} tốt nhất, {personal_address} có thể chia sẻ thêm chi tiết về vấn đề này được không ạ? 😊 Em rất sẵn lòng giúp đỡ!"
        elif response_style == 'brief':
            return f"Dạ {personal_address}, cần thêm thông tin chi tiết ạ. 🎓"
        elif response_style == 'technical':
            return f"Dạ {personal_address}, để cung cấp hướng dẫn kỹ thuật chính xác, {personal_address} vui lòng cung cấp thêm thông số và yêu cầu cụ thể ạ."
        elif response_style == 'detailed':
            return f"Dạ {personal_address}, để em có thể đưa ra câu trả lời toàn diện và chi tiết nhất, {personal_address} có thể bổ sung thêm về bối cảnh, mục đích sử dụng, và các yêu cầu cụ thể không ạ? Điều này sẽ giúp em hỗ trợ {personal_address} một cách hiệu quả nhất."
        else:  # professional
            return f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
            
    def _generate_dont_know_response(self, query, context, session_id=None, response_style='professional'):
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
        """✅ ENHANCED: Response strategy with generation bias"""
        
        has_real_history = bool(conversation_context.get('history') and len(conversation_context['history']) > 0)
        
        print(f"🔍 LECTURER STRATEGY DEBUG: has_real_history = {has_real_history}")
        
        if has_real_history:
            # Enhanced follow-up detection for lecturers
            last_interaction = conversation_context['history'][-1]
            last_query = last_interaction['user_query'].lower()
            current_query = query.lower()
            
            # Lecturer-specific topics
            lecturer_topics = {
                'ngân hàng đề thi': ['ngân hàng', 'đề thi', 'đề', 'khảo thí'],
                'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ', 'giờ chuẩn'],
                'tạp chí khoa học': ['tạp chí', 'bài viết', 'nghiên cứu'],
                'thi đua khen thưởng': ['thi đua', 'khen thưởng', 'danh hiệu'],
                'báo cáo': ['báo cáo', 'nộp', 'hạn cuối'],
                'lịch giảng dạy': ['lịch', 'giảng dạy', 'thời khóa biểu']
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

            has_exact_same_topic = last_main_topic is not None and last_main_topic == current_main_topic
            
            strong_continuation_words = ['còn', 'thêm', 'nữa', 'khác', 'và', 'tiếp theo']
            has_strong_continuation = any(word in current_query.split() for word in strong_continuation_words)
            
            strong_clarification_words = ['cụ thể hơn', 'rõ hơn', 'chi tiết hơn', 'giải thích thêm']
            has_strong_clarification = any(phrase in current_query for phrase in strong_clarification_words)
            
            memory_test_words = ['nhớ không', 'hỏi gì', 'nói gì trước', 'vừa nói', 'tổng hợp']
            is_memory_test = any(word in current_query for word in memory_test_words)

            # Context-based strategies
            if has_strong_continuation and has_exact_same_topic:
                return 'follow_up_continuation'
            
            if has_strong_clarification and has_exact_same_topic:
                return 'follow_up_clarification'

            if is_memory_test:
                return 'memory_reference'
                
            if current_main_topic is not None and last_main_topic is not None and current_main_topic != last_main_topic:
                return 'topic_shift'
        
        # ✅ MODIFIED: Default strategy logic with generation bias
        if isinstance(context, dict) and context.get('confidence', 0) > 0.85:  # Raised from 0.7
            return 'direct_enhance'
        
        # ✅ NEW: Favor enhancement over brief responses
        if isinstance(context, dict) and context.get('confidence', 0) > 0.45:  # New range for enhancement
            return 'enhanced_generation'  # New strategy for more generation
        
        if intent_info and intent_info.get('intent') in ['greeting', 'general'] and len(query.split()) <= 5:
            return 'quick_clarify'
        
        if any(word in query.lower() for word in ['khó khăn', 'cần gấp', 'hạn cuối', 'urgent']):
            return 'supportive_brief'
        
        return 'balanced'

    def _post_process_with_lecturer_consistency(self, response, query, context, strategy, conversation_context, session_id=None):
        """Post-process để đảm bảo nhất quán cho giảng viên với personalization"""
        if not response:
            return response
        
        # ✅ NEW: Lấy thông tin user để personalize addressing
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        faculty_code = user_context.get('faculty_code', '')
        
        # Tạo xưng hô cá nhân hóa nếu có thông tin
        if full_name:
            name_suffix = full_name.split()[-1]  # Lấy tên
            personal_address = f"thầy/cô {name_suffix}"
        elif faculty_code:
            personal_address = f"thầy/cô {faculty_code}"
        else:
            personal_address = "thầy/cô"
        
        # 1. Sửa các vi phạm vai trò cho giảng viên
        prohibited_phrases = [
            'với tư cách là sinh viên', 'tôi là học sinh',
            'bạn', 'mình', 'anh', 'chị', 'em là sinh viên'
        ]
        for phrase in prohibited_phrases:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, 'em là AI assistant của BDU')
        
        # 2. ✅ CRITICAL: Sửa xưng hô không đúng với personalization
        response = re.sub(r'\bbạn\b', personal_address, response, flags=re.IGNORECASE)
        response = re.sub(r'\bmình\b', 'em', response, flags=re.IGNORECASE)
        response = re.sub(r'\btôi\b', 'em', response, flags=re.IGNORECASE)
        
        # 3. ✅ CRITICAL: Đảm bảo bắt đầu bằng personalized greeting
        response_stripped = response.strip()
        personalized_start = f"Dạ {personal_address},"
        
        if not response_stripped.lower().startswith('dạ thầy/cô') and not response_stripped.lower().startswith(f'dạ {personal_address.lower()}'):
            if response_stripped.lower().startswith('dạ'):
                response = personalized_start + ' ' + response_stripped[3:].strip()
            else:
                response = personalized_start + ' ' + response_stripped
        
        # 4. ✅ CRITICAL: Đảm bảo kết thúc đúng cách với personalization
        personalized_end = f"{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
        if not response.strip().endswith('có cần hỗ trợ thêm gì không ạ?'):
            # Remove existing endings first
            response = re.sub(r'\s*(Thầy/cô có.*?không ạ\?|Cần.*?không\?|Có.*?không\?)?\s*$', '', response.strip())
            response += ' Thầy/cô có cần hỗ trợ thêm gì không ạ?'
            
        # 5. ✅ REMOVE: Loại bỏ format phức tạp
        response = re.sub(r'\*\*\d+\.\s*', '', response)  # Remove **1. **2. etc
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)  # Remove numbered lists
        response = re.sub(r'^\s*[•\-\*]\s*', '', response, flags=re.MULTILINE)  # Remove bullets
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Remove bold formatting
        
        return response.strip()
    
    def _get_contextual_out_of_scope_response_lecturer(self, conversation_context, session_id=None, user_response_style='professional'):
        """Out of scope response cho giảng viên với personalization"""
        
        # ✅ NEW: Personalized out of scope response
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        department_name = user_context.get('department_name', '')
        
        if full_name:
            name_suffix = full_name.split()[-1]
            personal_address = f"thầy/cô {name_suffix}"
        else:
            personal_address = "thầy/cô"
        
        if conversation_context.get('context_summary'):
            if department_name:
                return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} còn muốn hỏi gì về {conversation_context['context_summary'].lower()} cho ngành {department_name} không ạ?"
            else:
                return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} còn muốn hỏi gì về {conversation_context['context_summary'].lower()} không ạ?"
        
        if department_name:
            return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} có câu hỏi nào khác về ngành {department_name} không ạ?"
        else:
            return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} có câu hỏi nào khác về trường không ạ?"
    
    def _get_smart_fallback_with_context_lecturer(self, query, intent_info, conversation_context, session_id=None, user_response_style='professional'):
        """Smart fallback với conversation context cho giảng viên và personalization"""
        
        # ✅ NEW: Personalized fallback
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        department_name = user_context.get('department_name', '')
        
        if full_name:
            name_suffix = full_name.split()[-1]
            personal_address = f"thầy/cô {name_suffix}"
        else:
            personal_address = "thầy/cô"
        
        intent_name = intent_info.get('intent', 'general') if intent_info else 'general'
        
        if conversation_context.get('context_summary'):
            summary = conversation_context['context_summary']
            context_fallbacks = {
                'Đang hỏi về ngân hàng đề thi': f"Dạ {personal_address}, về ngân hàng đề thi, em có thể hỗ trợ thêm! 📋 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về kê khai nhiệm vụ năm học': f"Dạ {personal_address}, về kê khai nhiệm vụ năm học, em có thể hỗ trợ thêm! 📊 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về tạp chí khoa học': f"Dạ {personal_address}, về tạp chí khoa học, em có thể hỗ trợ thêm! 📚 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?",
                'Đang hỏi về thi đua khen thưởng': f"Dạ {personal_address}, về thi đua khen thưởng, em có thể hỗ trợ thêm! 🏆 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
            }
            if summary in context_fallbacks:
                return context_fallbacks[summary]
        
        smart_fallbacks = {
            'greeting': f"Dạ chào {personal_address}! 👋 Em có thể hỗ trợ gì cho {personal_address} về BDU ạ?",
            'general': f"Dạ {personal_address}, em sẵn sàng hỗ trợ các vấn đề liên quan đến BDU! 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        }
        
        if department_name and intent_name == 'general':
            smart_fallbacks['general'] = f"Dạ {personal_address}, em sẵn sàng hỗ trợ các vấn đề liên quan đến BDU và ngành {department_name}! 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
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

    # ✅ ADDED: Missing methods for personalization testing
    def _build_enhanced_prompt(self, query: str, context=None, intent_info=None, entities=None, session_id=None, response_style='professional'):
        """Build enhanced prompt with style awareness"""
        system_prompt = self._get_personalized_system_prompt(session_id)
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        personal_address = f"thầy/cô {full_name.split()[-1]}" if full_name else "thầy/cô"
        
        context_info = str(context.get('response', '')) if isinstance(context, dict) else str(context or '')
        
        prompt = f"""{system_prompt}
        
🎨 PHONG CÁCH: {response_style}
CÂU HỎI: {query}
THÔNG TIN: {context_info}

YÊU CẦU:
- Bắt đầu: "Dạ {personal_address},"
- Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
- Áp dụng phong cách: {response_style}

Trả lời:"""
        return prompt
    
    def _generate_style_aware_response(self, query: str, context=None, session_id=None, response_style='professional'):
        """Generate style-aware response"""
        prompt = self._build_enhanced_prompt(query, context, None, None, session_id, response_style)
        response = self._call_gemini_api_with_style(prompt, 'balanced', response_style)
        if response:
            response = self._post_process_with_lecturer_consistency(response, query, context, 'balanced', {}, session_id)
        return response or self._get_smart_fallback_with_context_lecturer(query, None, {}, session_id, response_style)
    
    def test_response_style(self, test_query: str, response_style: str, session_id=None):
        """Test response style functionality"""
        try:
            test_context = {'response': 'Đây là thông tin test từ database.', 'confidence': 0.8}
            response = self._generate_style_aware_response(test_query, test_context, session_id, response_style)
            
            style_info = {
                'professional': 'Chuyên nghiệp - trang trọng, lịch sự',
                'friendly': 'Thân thiện - gần gũi, vui vẻ với emoji',
                'technical': 'Kỹ thuật - chi tiết, thuật ngữ chuyên môn',
                'brief': 'Ngắn gọn - súc tích, đi thẳng vào vấn đề',
                'detailed': 'Chi tiết - đầy đủ, nhiều ví dụ'
            }
            
            return {
                'success': True,
                'test_query': test_query,
                'response': response,
                'current_style': response_style,
                'current_style_name': style_info.get(response_style, 'Unknown'),
                'style_applied': response_style,
                'config_used': self.style_generation_configs.get(response_style, {}),
                'recommendation': f"Phong cách '{style_info.get(response_style)}' đã được áp dụng thành công."
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'test_query': test_query, 'current_style': response_style}
    
    def validate_user_preferences(self, preferences):
        """Validate user preferences"""
        errors, warnings = [], []
        
        if 'response_style' in preferences:
            style = preferences['response_style']
            if style not in self.style_generation_configs:
                errors.append(f"Invalid response_style: {style}")
        
        if 'user_memory_prompt' in preferences:
            memory = preferences['user_memory_prompt']
            if isinstance(memory, str):
                if len(memory) > 1000:
                    errors.append("user_memory_prompt too long (max 1000 characters)")
                elif len(memory) > 900:
                    warnings.append("user_memory_prompt approaching limit")
            else:
                errors.append("user_memory_prompt must be string")
        
        if 'department_priority' in preferences:
            if not isinstance(preferences['department_priority'], bool):
                errors.append("department_priority must be boolean")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def get_user_context(self, session_id: str):
        """Get user context for session"""
        return self._user_context_cache.get(session_id)
    
    def clear_user_context(self, session_id=None):
        """Clear user context"""
        if session_id:
            if session_id in self._user_context_cache:
                del self._user_context_cache[session_id]
        else:
            self._user_context_cache.clear()

    def get_conversation_memory(self, session_id: str):
        return self.memory.get_conversation_context(session_id)
    
    def clear_conversation_memory(self, session_id: str = None):
        if session_id:
            if session_id in self.memory.conversations:
                del self.memory.conversations[session_id]
        else:
            self.memory.conversations.clear()
    
    # ✅ ENHANCED: System status with style info
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status with style support info"""
        try:
            test_prompt = "Test ngắn cho giảng viên"
            response = self._call_gemini_api_with_style(test_prompt, 'quick_clarify', 'professional')
            
            return {
                'gemini_api_available': response is not None,
                'api_key_configured': bool(self.api_key),
                'service_status': 'active' if response else 'error',
                'mode': 'lecturer_focused_with_style_support',
                'memory_sessions': len(self.memory.conversations),
                'personalization_sessions': len(self._user_context_cache),
                'supported_styles': list(self.style_generation_configs.keys()),  # ✅ NEW
                'style_generation_configs': self.style_generation_configs,  # ✅ NEW
                'features': [
                    'lecturer_conversation_memory',
                    'lecturer_role_consistency',
                    'lecturer_context_aware_responses',
                    'lecturer_follow_up_detection',
                    'lecturer_topic_shift_handling',
                    'lecturer_clarification_requests',
                    'lecturer_department_suggestions',
                    'personalized_system_prompts',
                    'personalized_addressing',
                    'department_specific_responses',
                    'response_style_support',  # ✅ NEW
                    'style_aware_generation',  # ✅ NEW
                    'dynamic_style_configs'  # ✅ NEW
                ]
            }
        except Exception as e:
            return {
                'gemini_api_available': False,
                'service_status': 'error',
                'error': str(e)
            }