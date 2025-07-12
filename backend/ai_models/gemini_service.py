import logging
import time
import requests
import json
import re
from typing import Dict, Any, Optional, List
from unidecode import unidecode
import difflib
import pandas as pd
import os

logger = logging.getLogger(__name__)

class GeminiApiKeyManager:
    """Quản lý và tự động xoay vòng các API key của Gemini để tránh lỗi rate limit."""
    def __init__(self):
        self.keys = []
        self._load_keys_from_env()
        self.current_key_index = 0
        self.key_status = {k: {'is_rate_limited': False, 'limited_until': 0} for k in self.keys}
        
        if not self.keys:
            logger.error("CRITICAL: No Gemini API keys found in .env file (e.g., GEMINI_API_KEY, GEMINI_API_KEY2)!")
        else:
            logger.info(f"✅ GeminiApiKeyManager initialized with {len(self.keys)} keys.")

    def _load_keys_from_env(self):
        """Tự động tải các key từ file .env theo định dạng GEMINI_API_KEY..."""
        main_key = os.getenv('GEMINI_API_KEY')
        if main_key:
            self.keys.append(main_key)
        
        i = 2
        while True:
            extra_key = os.getenv(f'GEMINI_API_KEY{i}')
            if extra_key:
                self.keys.append(extra_key)
                i += 1
            else:
                break
    
    def get_key(self) -> Optional[str]:
        """Lấy một API key hợp lệ để sử dụng (xoay vòng)."""
        if not self.keys:
            return None
        
        start_index = self.current_key_index
        for i in range(len(self.keys)):
            index = (start_index + i) % len(self.keys)
            key = self.keys[index]
            status = self.key_status[key]
            
            if status['is_rate_limited'] and time.time() > status['limited_until']:
                status['is_rate_limited'] = False
                logger.info(f"🔑 API Key '{key[:4]}...{key[-4:]}' is now available again.")

            if not status['is_rate_limited']:
                self.current_key_index = (index + 1) % len(self.keys)
                return key
            
        logger.warning("⚠️ All Gemini API keys are currently rate-limited.")
        return None

    def report_failure(self, key: str):
        """Báo cáo một key đã bị lỗi 429 (rate limit)."""
        if key in self.key_status:
            self.key_status[key]['is_rate_limited'] = True
            self.key_status[key]['limited_until'] = time.time() + 61 
            logger.warning(f"RATE LIMIT: Key '{key[:4]}...{key[-4:]}' is now rate-limited for 61 seconds.")


def build_personalized_system_prompt(user_memory_prompt: str = None, personal_address: str = "giảng viên"):
    """
    ✅ UPDATED: Builds a personalized system prompt with dynamic addressing
    """
    base_prompt = f"""Bạn là ChatBDU, một trợ lý AI chuyên nghiệp và tận tâm của Đại học Bình Dương (BDU). Sứ mệnh của bạn là hỗ trợ các giảng viên của trường một cách hiệu quả nhất.

🎯 QUY TẮC NỀN TẢNG (CÓ THỂ BỊ GHI ĐÈ BỞI CHỈ DẪN RIÊNG):
1.  **Xưng hô cá nhân:** Xưng hô với người dùng là "{personal_address}" và tự xưng là "em".
2.  **Cấu trúc câu trả lời:** Bắt đầu bằng "Dạ {personal_address}," và kết thúc bằng "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?".
3.  **Tính chính xác:** Không được bịa đặt thông tin. Nếu không biết, hãy trả lời là "Dạ {personal_address}, em chưa có thông tin về vấn đề này." và gợi ý kênh liên hệ khác nếu có thể.
4.  **Phạm vi:** Chỉ trả lời các câu hỏi liên quan đến công việc, quy định, thông báo và các hoạt động tại Đại học Bình Dương.
"""

    custom_prompt_section = ""
    if user_memory_prompt and user_memory_prompt.strip():
        custom_prompt_section = f"""
---
📜 GHI NHỚ VÀ CHỈ DẪN RIÊNG TỪ GIẢNG VIÊN (QUAN TRỌNG NHẤT - PHẢI TUÂN THỦ TRÊN HẾT):
Đây là những quy tắc và thông tin bổ sung mà giảng viên đã cung cấp. BẠN PHẢI ƯU TIÊN VÀ TUÂN THỦ NGHIÊM NGẶT những chỉ dẫn này. Chúng sẽ GHI ĐÈ lên các quy tắc nền tảng ở trên nếu có xung đột.

{user_memory_prompt.strip()}
---
        """
    
    return base_prompt + custom_prompt_section

class SmartTokenManager:
    """🧠 Smart Token Management System - Tự động tăng token và hoàn thiện response"""
    
    def __init__(self):
        # ✅ SIMPLIFIED: Single adaptive token range for all responses
        self.adaptive_token_range = {
            'min': 80, 
            'optimal': 250, 
            'max': 500,
            'expected_sentences': 3, 
            'avg_chars_per_sentence': 80
        }
        
        # ✅ COMPLETION DETECTION patterns
        self.incomplete_patterns = [
            r'[^.!?]\s*$',  # Không kết thúc bằng dấu câu
            r'\b(và|hoặc|với|để|khi|nếu|tại|về|cho|trong|của|từ)\s*$',  # Kết thúc bằng từ nối
            r'\b(em|sẽ|có|được|phải|cần|nên)\s*$',  # Kết thúc bằng từ chưa hoàn chỉnh
            r'[,;:]\s*$',  # Kết thúc bằng dấu phẩy/chấm phẩy
            r'\b(Dạ|Ạ|thầy|cô|giảng viên)\s*$',  # Câu chào chưa hoàn chỉnh
        ]
        
        # ✅ SENTENCE ENDING patterns để kiểm tra câu hoàn chỉnh
        self.complete_endings = [
            r'[.!?]\s*$',  # Kết thúc bằng dấu câu
            r'ạ[.!?]\s*$',  # Kết thúc bằng "ạ" + dấu câu
            r'không ạ\?\s*$',  # "có cần hỗ trợ thêm gì không ạ?"
            r'🎓\s*$',  # Emoji kết thúc
            r'@bdu\.edu\.vn\s*$',  # Email ending
        ]
        
        logger.info("✅ SmartTokenManager initialized with adaptive token range")
    
    def calculate_optimal_tokens(self, prompt_length: int, complexity_hint: str = None) -> int:
        """🎯 Tính toán tokens tối ưu dựa trên độ phức tạp"""
        
        # Base tokens from adaptive range
        base_tokens = self.adaptive_token_range['optimal']
        
        # ✅ ADJUSTMENT dựa trên prompt length
        if prompt_length > 500:
            base_tokens += 50  # Prompt dài cần response dài hơn
        elif prompt_length < 200:
            base_tokens -= 30  # Prompt ngắn có thể response ngắn hơn
            
        # ✅ ADJUSTMENT dựa trên complexity hint
        if complexity_hint:
            if complexity_hint in ['enhanced_generation', 'detailed_explanation']:
                base_tokens += 100
            elif complexity_hint in ['quick_clarify', 'simple_answer']:
                base_tokens -= 40
                
        # ✅ BOUNDS checking
        min_tokens = self.adaptive_token_range['min']
        max_tokens = self.adaptive_token_range['max']
        
        return max(min_tokens, min(max_tokens, base_tokens))
    
    def is_response_incomplete(self, response: str) -> Dict[str, Any]:
        """🔍 Kiểm tra response có bị cắt không và mức độ hoàn thiện"""
        
        if not response or not response.strip():
            return {'incomplete': True, 'reason': 'empty_response', 'confidence': 1.0}
        
        response = response.strip()
        
        # ✅ CHECK 1: Pattern matching cho incomplete
        for pattern in self.incomplete_patterns:
            if re.search(pattern, response):
                return {
                    'incomplete': True, 
                    'reason': 'incomplete_pattern',
                    'pattern': pattern,
                    'confidence': 0.8
                }
        
        # ✅ CHECK 2: Expected sentence count
        expected_sentences = self.adaptive_token_range['expected_sentences']
        
        actual_sentences = len(re.findall(r'[.!?]+', response))
        if actual_sentences < expected_sentences * 0.7:  # Ít hơn 70% expected
            return {
                'incomplete': True,
                'reason': 'insufficient_sentences',
                'expected': expected_sentences,
                'actual': actual_sentences,
                'confidence': 0.7
            }
        
        # ✅ CHECK 3: Required ending patterns for Vietnamese lecturer context
        required_ending = r'(có cần hỗ trợ thêm gì không ạ\?|ạ[.!?]|\?)'
        if not re.search(required_ending, response.lower()):
            return {
                'incomplete': True,
                'reason': 'missing_proper_ending',
                'confidence': 0.9
            }
        
        # ✅ CHECK 4: Proper greeting start
        if not re.match(r'dạ\s+(thầy|cô|giảng viên)', response.lower()):
            return {
                'incomplete': True,
                'reason': 'missing_proper_greeting',
                'confidence': 0.6
            }
        
        return {'incomplete': False, 'reason': 'complete', 'confidence': 0.9}
    
    def estimate_completion_tokens(self, incomplete_response: str) -> int:
        """📊 Ước tính tokens cần để hoàn thiện response"""
        
        # Estimate current length in tokens (rough: 1 token ≈ 3-4 chars in Vietnamese)
        current_tokens = len(incomplete_response) // 3
        
        # Target tokens for complete response
        target_tokens = self.adaptive_token_range['optimal']
        
        # Additional tokens needed
        additional_needed = max(20, target_tokens - current_tokens)
        
        return min(additional_needed, 150)  # Cap at 150 additional tokens

class ConversationMemory:
    """🚀 NÂNG CẤP: Quản lý bộ nhớ hội thoại với conversation context summary"""
    
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
                'conversation_type': 'lecturer'
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
        """🚀 NÂNG CẤP: Lấy context của conversation với recent summary"""
        if session_id not in self.conversations:
            return {'history': [], 'context_summary': '', 'user_interests': [], 'recent_conversation_summary': ''}
        
        conv = self.conversations[session_id]
        
        # ✅ NEW: Tạo tóm tắt 2-3 tương tác gần nhất
        recent_summary = self._create_recent_conversation_summary(session_id)
        
        return {
            'history': conv['history'][-5:],  # Last 5 interactions
            'context_summary': conv['context_summary'],
            'user_interests': list(conv['user_interests']),
            'conversation_type': conv['conversation_type'],
            'recent_conversation_summary': recent_summary  # ✅ NEW
        }
    
    def _create_recent_conversation_summary(self, session_id: str) -> str:
        """🚀 NEW: Tạo tóm tắt ngắn gọn 2-3 tương tác gần nhất"""
        if session_id not in self.conversations:
            return ""
        
        history = self.conversations[session_id]['history']
        if len(history) < 2:
            return ""
        
        # Lấy 2-3 tương tác gần nhất
        recent_interactions = history[-3:] if len(history) >= 3 else history[-2:]
        
        summary_parts = []
        for interaction in recent_interactions:
            user_query = interaction['user_query'][:100]  # Tối đa 100 ký tự
            bot_response = interaction['bot_response'][:150]  # Tối đa 150 ký tự
            
            summary_parts.append(f"Hỏi: {user_query}... → Trả lời: {bot_response}...")
        
        return " | ".join(summary_parts)
    
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
    
    def __init__(self, key_manager: GeminiApiKeyManager):
        self.key_manager = key_manager
        self.model_name = "gemini-2.0-flash" # Dùng model flash cho nhanh và rẻ
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.cache = {}
        self.max_cache_size = 500
        logger.info("✅ SimpleVietnameseRestorer initialized with Key Manager.")
    
    def has_vietnamese_accents(self, text: str) -> bool:
        """Check if text has Vietnamese accents"""
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        vietnamese_chars += vietnamese_chars.upper()
        return any(char in vietnamese_chars for char in text)
    
    def restore_vietnamese_tone(self, input_text: str, retry_count=0) -> str:
        """Restore Vietnamese accents using Gemini API with Key Manager."""
        if not input_text or not input_text.strip():
            return input_text
        
        input_text = input_text.strip()
        
        # Check cache first
        cache_key = input_text.lower()
        if cache_key in self.cache:
            logger.debug(f"🎯 Tone-restorer cache hit for: '{input_text}'")
            return self.cache[cache_key]
        
        # If already has accents, return as is
        if self.has_vietnamese_accents(input_text):
            self.cache[cache_key] = input_text
            return input_text

        # Lấy một key hợp lệ từ bộ quản lý
        api_key_to_use = self.key_manager.get_key()
        if not api_key_to_use:
            logger.error("Tone Restorer: All keys are rate-limited. Skipping restoration.")
            self._cache_result(cache_key, input_text)
            return input_text # Trả về text gốc nếu không còn key

        prompt = f'Hãy viết lại câu sau thành tiếng Việt có dấu đầy đủ, không thay đổi ý nghĩa: "{input_text}"'
        
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100, "topP": 0.8},
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            url = f"{self.base_url}?key={api_key_to_use}"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        restored_text = candidate['content']['parts'][0]['text'].strip()
                        restored_text = re.sub(r'^["\'](.*)["\']$', r'\1', restored_text)
                        restored_text = re.sub(r'^(Câu đã có dấu:|Kết quả:|Trả lời:)\s*', '', restored_text, flags=re.IGNORECASE)
                        
                        if self._is_valid_restoration(input_text, restored_text):
                            logger.info(f"✅ Restored: '{input_text}' -> '{restored_text}'")
                            self._cache_result(cache_key, restored_text)
                            return restored_text
                        else:
                            logger.warning(f"⚠️ Invalid restoration: '{restored_text}'")
            
            elif response.status_code == 429:
                # Báo cáo key bị lỗi và thử lại với key khác (chỉ thử lại 1 lần)
                self.key_manager.report_failure(api_key_to_use)
                if retry_count == 0:
                    logger.warning("Tone Restorer: Rate limit hit, retrying with new key...")
                    return self.restore_vietnamese_tone(input_text, retry_count=1)
            
            else:
                logger.error(f"❌ Gemini API Error {response.status_code} for tone restorer")
                
        except Exception as e:
            logger.error(f"❌ Error restoring tone: {e}")
        
        # Fallback: trả về text gốc nếu có lỗi
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
    """🚀 NÂNG CẤP: Advanced Gemini Response Generator với Smart Token Management và Context Summary"""
    
    def __init__(self):
        self.key_manager = GeminiApiKeyManager()
        # Thống nhất phiên bản model ở đây
        self.model_name = "gemini-2.0-flash" 
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.memory = ConversationMemory(max_history=10)
        self.vietnamese_restorer = SimpleVietnameseRestorer(self.key_manager)
        
        # ✅ NEW: Smart Token Manager
        self.token_manager = SmartTokenManager()
        
        # User context cache for personalization
        self._user_context_cache = {}
        
        # ✅ SIMPLIFIED: Single default generation config
        self.default_generation_config = {
            "temperature": 0.4,
            "topP": 0.85
        }
        
        # Role consistency rules (unchanged)
        self.role_consistency_rules = {
            'identity': 'AI assistant của Đại học Bình Dương (BDU) hỗ trợ giảng viên',
            'personality': 'lịch sự, chuyên nghiệp, tôn trọng',
            'knowledge_scope': 'chuyên về thông tin BDU và hỗ trợ giảng viên',
            'addressing': 'luôn xưng hô đúng cách, không bao giờ dùng bạn/mình',
            'prohibited_roles': [
                'sinh viên', 'học sinh', 'phụ huynh', 'người ngoài trường'
            ]
        }
        
        logger.info("✅ Enhanced Gemini Response Generator initialized with Smart Token Management và Consistent Personalization")

    # ✅ NEW: Process external API data
    def _generate_external_api_response(self, query, context, session_id=None):
        """Generate response from external API data"""
        
        api_data = context.get('api_data', {})
        lecturer_info = api_data.get('lecturer_info', {})
        schedule_summary = api_data.get('schedule_summary', {})
        daily_schedule = api_data.get('daily_schedule', {})
        
        # Get personal addressing
        personal_address = self._get_personal_address_from_api_data(lecturer_info, session_id)
        
        # ✅ NEW: Get conversation context summary
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        # Build comprehensive prompt for external API data
        prompt = self._build_external_api_prompt(
            query, api_data, personal_address, recent_summary
        )
        
        # Calculate optimal tokens for external API response
        optimal_tokens = self.token_manager.calculate_optimal_tokens(
            len(prompt), 
            'external_api_processing'
        )
        
        logger.info(f"🌐 Processing external API data with {optimal_tokens} tokens")
        
        # 🚀 CRITICAL FIX: Pass session_id để consistent personalization
        response = self._call_gemini_api_with_smart_tokens(
            prompt, 'external_api_processing', optimal_tokens, session_id
        )
        
        if not response:
            # Fallback if Gemini fails
            return self._get_external_api_fallback_response(api_data, personal_address)
        
        # Post-process for consistency
        response = self._post_process_external_api_response(
            response, lecturer_info, query, session_id
        )
        
        return response
    
    def _build_external_api_prompt(self, query, api_data, personal_address, recent_summary=""):
        """🚀 NÂNG CẤP: Build comprehensive prompt với conversation context"""
        
        lecturer_info = api_data.get('lecturer_info', {})
        schedule_summary = api_data.get('schedule_summary', {})
        daily_schedule = api_data.get('daily_schedule', {})
        query_context = api_data.get('query_context', '')
        
        # Get lecturer details
        ten_giang_vien = lecturer_info.get('ten_giang_vien', personal_address)
        ma_giang_vien = lecturer_info.get('ma_giang_vien', '')
        chuc_danh = lecturer_info.get('chuc_danh', '')
        gmail = lecturer_info.get('gmail', '')
        trinh_do = lecturer_info.get('trinh_do', '')
        
        # Get schedule details
        total_classes = schedule_summary.get('total_classes', 0)
        unique_subjects = schedule_summary.get('unique_subjects', 0)
        total_periods = schedule_summary.get('total_periods', 0)
        
        # Format daily schedule for prompt
        schedule_text = self._format_schedule_for_prompt(daily_schedule)
        
        system_prompt = self._get_personalized_system_prompt_for_external_api(
            lecturer_info
        )
        
        # ✅ NEW: Conversation context section
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Hãy tham khảo ngữ cảnh trên để tạo câu trả lời mạch lạc, tránh lặp lại thông tin đã nói.
"""
        
        prompt = f"""{system_prompt}

🎯 NHIỆM VỤ ĐẶC BIỆT: Xử lý thông tin CÁ NHÂN từ hệ thống của trường

📋 THÔNG TIN GIẢNG VIÊN:
- Mã giảng viên: {ma_giang_vien}
- Họ và tên: {ten_giang_vien}
- Chức danh: {chuc_danh}
- Trình độ: {trinh_do}
- Email: {gmail}

📅 TỔNG QUAN LỊCH GIẢNG DẠY:
- Tổng số buổi học: {total_classes}
- Số môn học: {unique_subjects}
- Tổng số tiết: {total_periods}

📖 CHI TIẾT LỊCH GIẢNG DẠY:
{schedule_text}

{context_section}

❓ CÂU HỎI CỦA GIẢNG VIÊN: {query}
🔍 NGỮ CẢNH TÌM KIẾM: {query_context}

📝 YÊU CẦU TRẢ LỜI:
- Xưng hô: "Dạ {personal_address},"
- Trả lời CHÍNH XÁC dựa trên dữ liệu thực tế từ hệ thống
- Tạo câu trả lời mạch lạc, tránh lặp lại thông tin đã thảo luận
- Định dạng thông tin dễ đọc, rõ ràng
- Bao gồm các chi tiết quan trọng: thời gian, địa điểm, môn học
- Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
- KHÔNG CHẾ TẠO thông tin không có trong dữ liệu

Trả lời:"""

        return prompt
    
    def _format_schedule_for_prompt(self, daily_schedule):
        """Format daily schedule data for Gemini prompt"""
        if not daily_schedule:
            return "Hiện tại không có lịch giảng dạy trong khoảng thời gian này."
        
        formatted_lines = []
        
        # Sort dates
        sorted_dates = sorted(daily_schedule.keys())
        
        for date_str in sorted_dates:
            classes = daily_schedule[date_str]
            
            # Format date
            try:
                from datetime import datetime
                date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
                weekday = weekdays[date_obj.weekday()]
                formatted_date = f"{weekday}, {date_str}"
            except:
                formatted_date = date_str
            
            formatted_lines.append(f"\n📅 {formatted_date}:")
            
            # Sort classes by starting period
            sorted_classes = sorted(classes, key=lambda x: x.get('tiet_bat_dau', 0))
            
            for class_info in sorted_classes:
                ma_mon_hoc = class_info.get('ma_mon_hoc', '')
                ten_mon_hoc = class_info.get('ten_mon_hoc', '')
                ma_lop = class_info.get('ma_lop', '')
                ma_phong = class_info.get('ma_phong', '')
                tiet_bat_dau = class_info.get('tiet_bat_dau', '')
                so_tiet = class_info.get('so_tiet', '')
                so_luong_sv = class_info.get('so_luong_sv', '')
                
                # Format class entry
                class_line = f"   • {ten_mon_hoc} ({ma_mon_hoc})"
                class_line += f" - Lớp {ma_lop}"
                class_line += f" - Phòng {ma_phong}"
                class_line += f" - Tiết {tiet_bat_dau}"
                if so_tiet:
                    class_line += f" ({so_tiet} tiết)"
                if so_luong_sv:
                    class_line += f" - {so_luong_sv} SV"
                
                formatted_lines.append(class_line)
        
        return '\n'.join(formatted_lines) if formatted_lines else "Không có lịch giảng dạy."

    def _get_personalized_system_prompt_for_external_api(self, lecturer_info):
        """Get personalized system prompt for external API processing với gender-based addressing"""
        
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        gender = lecturer_info.get('gender', 'other')  # ✅ NEW: Lấy giới tính từ API
        chuc_danh = lecturer_info.get('chuc_danh', '')
        
        # ✅ NEW: Xác định cách xưng hô dựa trên giới tính
        if gender == 'male':
            salutation = 'thầy'
        elif gender == 'female':
            salutation = 'cô'
        else:
            salutation = 'giảng viên'
        
        name_parts = ten_giang_vien.split() if ten_giang_vien else []
        name_suffix = name_parts[-1] if name_parts else ''
        
        if salutation in ['thầy', 'cô']:
            personal_address = f"{salutation} {name_suffix}" if name_suffix else salutation
        else:
            personal_address = f"{salutation} {ten_giang_vien}" if ten_giang_vien else salutation
        
        base_prompt = f"""Bạn là AI assistant của Đại học Bình Dương (BDU), chuyên hỗ trợ giảng viên.

🎯 THÔNG TIN NGƯỜI DÙNG:
- Bạn đang trả lời cho {chuc_danh} {ten_giang_vien}
- Xưng hô: "{personal_address}" (TUYỆT ĐỐI KHÔNG dùng "bạn", "mình", "anh/chị")
- Đây là thông tin CÁ NHÂN từ hệ thống chính thức của trường

🎯 QUY TẮC QUAN TRỌNG:
- LUÔN bắt đầu: "Dạ {personal_address},"
- Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
- SỬ DỤNG CHÍNH XÁC thông tin từ hệ thống - KHÔNG CHẾ TẠO
- Trình bày thông tin cá nhân một cách tự nhiên, dễ hiểu
- KHÔNG dùng format phức tạp với **1. **2. hay bullets khi không cần thiết"""

        return base_prompt

    def _get_personal_address_from_api_data(self, lecturer_info, session_id):
        """Get personal address from API data or session với gender support"""
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        gender = lecturer_info.get('gender', 'other')
        
        # ✅ NEW: Sử dụng giới tính từ API data
        if gender == 'male':
            salutation = 'thầy'
        elif gender == 'female':
            salutation = 'cô'
        else:
            salutation = 'giảng viên'
        
        if ten_giang_vien:
            if salutation in ['thầy', 'cô']:
                name_suffix = ten_giang_vien.split()[-1]
                return f"{salutation} {name_suffix}"
            else:
                return f"{salutation} {ten_giang_vien}"
        
        # Fallback to session-based addressing
        return self._get_personal_address(session_id)

    def _post_process_external_api_response(self, response, lecturer_info, query, session_id):
        """Post-process external API response for consistency với gender support"""
        if not response:
            return response
        
        # Get personal addressing info với gender
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        gender = lecturer_info.get('gender', 'other')
        
        if gender == 'male':
            salutation = 'thầy'
        elif gender == 'female':
            salutation = 'cô'
        else:
            salutation = 'giảng viên'
        
        if ten_giang_vien:
            if salutation in ['thầy', 'cô']:
                name_suffix = ten_giang_vien.split()[-1]
                personal_address = f"{salutation} {name_suffix}"
            else:
                personal_address = f"{salutation} {ten_giang_vien}"
        else:
            personal_address = salutation
        
        # 1. Fix addressing inconsistencies
        response = re.sub(r'\bbạn\b', personal_address, response, flags=re.IGNORECASE)
        response = re.sub(r'\bmình\b', 'em', response, flags=re.IGNORECASE)
        response = re.sub(r'\btôi\b', 'em', response, flags=re.IGNORECASE)
        
        # 2. Ensure proper greeting
        response_stripped = response.strip()
        personalized_start = f"Dạ {personal_address},"
        
        if not response_stripped.lower().startswith(f'dạ {personal_address.lower()}'):
            if response_stripped.lower().startswith('dạ'):
                response = personalized_start + ' ' + response_stripped[3:].strip()
            else:
                response = personalized_start + ' ' + response_stripped
        
        # 3. Ensure proper ending
        if not response.strip().endswith('có cần hỗ trợ thêm gì không ạ?'):
            response = re.sub(r'\s*(có cần.*?không ạ\?|Cần.*?không\?|Có.*?không\?)?\s*$', '', response.strip())
            response += f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ?'
        
        # 4. Remove excessive formatting
        response = re.sub(r'\*\*\d+\.\s*', '', response)
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^\s*[•\-\*]\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        
        return response.strip()

    def _get_external_api_fallback_response(self, api_data, personal_address):
        """Fallback response when Gemini fails to process external API data"""
        lecturer_info = api_data.get('lecturer_info', {})
        schedule_summary = api_data.get('schedule_summary', {})
        
        ten_giang_vien = lecturer_info.get('ten_giang_vien', personal_address)
        total_classes = schedule_summary.get('total_classes', 0)
        
        return f"""Dạ {personal_address}, em đã tìm thấy thông tin từ hệ thống của trường:

👤 Thông tin của {ten_giang_vien}:
- Mã giảng viên: {lecturer_info.get('ma_giang_vien', 'Không xác định')}
- Chức danh: {lecturer_info.get('chuc_danh', 'Không xác định')}
- Email: {lecturer_info.get('gmail', 'Không có')}

📅 Lịch giảng dạy: {total_classes} buổi học được lên lịch

Để xem chi tiết, {personal_address} có thể truy cập hệ thống quản lý đào tạo của trường ạ. 🎓

{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"""
    
    # ✅ NEW: Personalization methods
    def set_user_context(self, session_id: str, user_context: dict):
        """Set user context cho session (được gọi từ chat API)"""
        
        print("\n" + "="*20 + " DEBUG: set_user_context " + "="*20)
        print(f"🕵️‍♂️ [set_user_context] Đang cài đặt context cho session: {session_id}")
        print(f"🕵️‍♂️ [set_user_context] Dữ liệu context nhận được: {user_context}")
        if 'gender' in user_context:
            print(f"✅ [set_user_context] TÌM THẤY 'gender' trong context: '{user_context['gender']}'")
        else:
            print(f"❌ [set_user_context] KHÔNG TÌM THẤY 'gender' trong context!")
        print("="*60 + "\n")
        
        self._user_context_cache[session_id] = user_context
        logger.info(f"✅ Set user context for session {session_id}: {user_context.get('faculty_code', 'Unknown')}")

    def _get_personalized_system_prompt(self, session_id: str = None):
        """🚀 NÂNG CẤP: Lấy personalized system prompt với conversation context"""
        try:
            # ✅ NEW: Lấy cách xưng hô cá nhân hóa
            personal_address = self._get_personal_address(session_id)
            
            user_context = self._user_context_cache.get(session_id, {})
            user_memory_prompt = user_context.get('preferences', {}).get('user_memory_prompt', '')
            
            # ✅ NEW: Truyền cách xưng hô vào build function
            return build_personalized_system_prompt(user_memory_prompt, personal_address)
        
        except Exception as e:
            logger.error(f"Error getting personalized prompt: {e}")
            return build_personalized_system_prompt()  # Fallback

    # 🚀 ENHANCED: Generate response với Smart Token Management
    def generate_response(self, query: str, context: Optional[Dict] = None, 
                      intent_info: Optional[Dict] = None, entities: Optional[Dict] = None,
                      session_id: str = None) -> Dict[str, Any]:
        """🚀 NÂNG CẤP: Generate response với Smart Token Management & Context Summary"""
        start_time = time.time()
        
        print(f"\n--- 🚀 SMART TOKEN MANAGEMENT REQUEST (Session: {session_id}) ---")
        print(f"🧠 MEMORY DEBUG: Total active sessions = {len(self.memory.conversations)}")

        try:
            original_query = query
            if not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Query restored: '{query}' -> '{restored_query}'")
                    query = restored_query

            # ✅ NEW: Check if query is empty after restoration
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'process_external_api_data':
                # Process external API data
                response = self._generate_external_api_response(query, context, session_id)
                
                token_info = {
                    'smart_tokens_used': True,
                    'method': 'external_api_processing'
                }
                
                # Save to memory
                if session_id:
                    self.memory.add_interaction(session_id, original_query, response, intent_info, entities)

                return {
                    'response': response,
                    'method': 'external_api_processing',
                    'strategy': 'external_api',
                    'generation_time': time.time() - start_time,
                    'original_query': original_query,
                    'restored_query': query,
                    'vietnamese_restoration_used': query != original_query,
                    'personalized': bool(session_id in self._user_context_cache),
                    'external_api_processed': True,
                    'token_info': token_info
                }

            # ✅ ENHANCED: Get conversation context với recent summary
            conversation_context = {}
            if session_id:
                conversation_context = self.memory.get_conversation_context(session_id)
                print(f"🧠 MEMORY DEBUG: History length = {len(conversation_context.get('history', []))}")
                print(f"📝 CONTEXT SUMMARY: {conversation_context.get('recent_conversation_summary', 'None')}")

            # Get user context for personalization
            user_context = None
            if session_id and session_id in self._user_context_cache:
                user_context = self._user_context_cache[session_id]
                print(f"👤 USER CONTEXT: {user_context.get('faculty_code', 'Unknown')}")

            # Determine response strategy
            response_strategy = self._determine_lecturer_response_strategy(
                query, context, intent_info, conversation_context
            )
            
            # ✅ ENHANCED: Handle instruction-based responses với Smart Tokens
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'direct_answer_lecturer':
                response, token_info = self._generate_direct_lecturer_answer_smart(query, context, session_id)
            elif instruction in ['enhance_answer_lecturer', 'enhance_answer_lecturer_boosted']:
                response, token_info = self._generate_enhanced_lecturer_answer_smart(query, context, intent_info, entities, session_id)
            elif instruction == 'clarification_needed':
                response, token_info = self._generate_clarification_request_smart(query, context, session_id)
            elif instruction == 'dont_know_lecturer':
                response, token_info = self._generate_dont_know_response_smart(query, context, session_id)
            else:
                # Check out of scope and generate response
                if context and context.get('emergency_education', False):
                    print(f"🚨 GEMINI: Emergency education mode activated")
                    pass 
                elif not self._is_lecturer_education_related(query) and not context.get('force_education_response', False):
                    response = self._get_contextual_out_of_scope_response_lecturer(conversation_context, session_id)
                    token_info = {'smart_tokens_used': False, 'method': 'predefined_template'}
                    
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
                        'token_info': token_info
                    }
                
                # ✅ ENHANCED: Use Smart Token Generation
                response, token_info = self._generate_smart_response(query, context, session_id, response_strategy)
            
            final_response = response or self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id)
            
            # Save to memory
            if session_id:
                print(f"🧠 MEMORY DEBUG: Saving interaction to memory...")
                self.memory.add_interaction(session_id, original_query, final_response, intent_info, entities)

            return {
                'response': final_response,
                'method': f'smart_lecturer_aware_gemini_{response_strategy}',
                'strategy': response_strategy,
                'conversation_context': conversation_context,
                'generation_time': time.time() - start_time,
                'original_query': original_query,
                'restored_query': query,
                'vietnamese_restoration_used': query != original_query,
                'personalized': bool(user_context),
                'enhanced_generation': response_strategy == 'enhanced_generation',
                'token_info': token_info  # ✅ NEW: Smart token information
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            fallback_response = self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id)
            
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
                'token_info': {'smart_tokens_used': False, 'method': 'fallback'}
            }

    # 🧠 SMART TOKEN GENERATION METHODS

    def _generate_smart_response(self, query: str, context=None, session_id=None, strategy='balanced'):
        """🚀 Generate response with Smart Token Management"""
        
        prompt = self._build_enhanced_prompt(query, context, None, None, session_id)
        
        # ✅ STEP 1: Calculate optimal tokens
        optimal_tokens = self.token_manager.calculate_optimal_tokens(
            len(prompt), 
            complexity_hint=strategy
        )
        
        print(f"🧠 SMART TOKENS: {optimal_tokens} tokens")
        
        # ✅ STEP 2: First attempt with optimal tokens - 🚀 CRITICAL FIX: Pass session_id
        response = self._call_gemini_api_with_smart_tokens(prompt, strategy, optimal_tokens, session_id)
        
        if not response:
            return self._get_smart_fallback_with_context_lecturer(query, None, {}, session_id), {
                'smart_tokens_used': True, 'method': 'fallback_after_api_failure', 'tokens_attempted': optimal_tokens
            }
        
        # ✅ STEP 3: Check if response is complete
        completion_check = self.token_manager.is_response_incomplete(response)
        
        if completion_check['incomplete']:
            print(f"⚠️ INCOMPLETE RESPONSE detected: {completion_check['reason']}")
            
            # ✅ STEP 4: Auto-completion attempt
            completed_response = self._auto_complete_response(response, query, context, session_id, completion_check)
            
            if completed_response and completed_response != response:
                response = completed_response
                completion_check['auto_completed'] = True
                print(f"✅ AUTO-COMPLETION successful")
            else:
                print(f"⚠️ AUTO-COMPLETION failed, using original")
        
        # ✅ STEP 5: Post-process for consistency
        response = self._post_process_with_lecturer_consistency(response, query, context, strategy, {}, session_id)
        
        token_info = {
            'smart_tokens_used': True,
            'method': 'smart_generation',
            'optimal_tokens': optimal_tokens,
            'completion_check': completion_check,
            'strategy': strategy
        }
        
        return response, token_info

    def _auto_complete_response(self, incomplete_response: str, original_query: str, context, session_id: str, completion_info: Dict) -> Optional[str]:
        """🔧 Auto-complete incomplete response"""
        
        if completion_info['confidence'] < 0.6:  # Don't auto-complete if not confident it's incomplete
            return None
        
        # ✅ Calculate completion tokens needed
        completion_tokens = self.token_manager.estimate_completion_tokens(incomplete_response)
        
        # ✅ Build completion prompt
        completion_prompt = self._build_completion_prompt(incomplete_response, original_query, context, session_id, completion_info)
        
        print(f"🔧 AUTO-COMPLETION: Attempting with {completion_tokens} tokens")
        
        # ✅ Call API to complete - 🚀 CRITICAL FIX: Pass session_id
        completion = self._call_gemini_api_with_smart_tokens(completion_prompt, 'completion', completion_tokens, session_id)
        
        if completion:
            # ✅ Merge incomplete + completion
            if completion_info['reason'] == 'missing_proper_ending':
                # Just add proper ending
                personal_address = self._get_personal_address(session_id)
                return incomplete_response.rstrip() + f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ?'
            elif completion_info['reason'] == 'missing_proper_greeting':
                # Add proper greeting
                personal_address = self._get_personal_address(session_id)
                return f"Dạ {personal_address}, " + incomplete_response.lstrip()
            else:
                # Merge content
                merged = self._merge_incomplete_and_completion(incomplete_response, completion)
                return merged
        
        return None

    def _build_completion_prompt(self, incomplete_response: str, original_query: str, context, session_id: str, completion_info: Dict) -> str:
        """🔧 Build prompt to complete incomplete response"""
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        personal_address = self._get_personal_address(session_id)
        
        if completion_info['reason'] == 'incomplete_pattern':
            completion_prompt = f"""
            {system_prompt}
            
            NHIỆM VỤ: HOÀN THIỆN câu trả lời bị cắt
            
            CÂU HỎI GỐC: {original_query}
            
            CÂU TRẢ LỜI BỊ CẮT:
            {incomplete_response}
            
            YÊU CẦU:
            - TIẾP TỤC viết để hoàn thiện câu trả lời
            - Đảm bảo kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
            - CHỈ VIẾT PHẦN TIẾP THEO, không lặp lại phần đã có
            
            Tiếp tục:"""
        else:
            completion_prompt = f"""
            {system_prompt}
            
            NHIỆM VỤ: SỬA LỖI và hoàn thiện câu trả lời
            
            CÂU HỎI GỐC: {original_query}
            
            CÂU TRẢ LỜI CÓ VẤN ĐỀ:
            {incomplete_response}
            
            VẤN ĐỀ PHÁT HIỆN: {completion_info['reason']}
            
            YÊU CẦU:
            - SỬA LỖI và viết lại câu trả lời HOÀN CHỈNH
            - Bắt đầu: "Dạ {personal_address},"
            - Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
            
            Câu trả lời hoàn chỉnh:"""
        
        return completion_prompt

    def _merge_incomplete_and_completion(self, incomplete: str, completion: str) -> str:
        """🔧 Merge incomplete response with completion"""
        
        # Clean completion
        completion = completion.strip()
        
        # Remove redundant greetings from completion
        completion = re.sub(r'^(dạ\s+(thầy|cô|giảng viên),?\s*)', '', completion, flags=re.IGNORECASE)
        
        # If incomplete ends with incomplete word, replace it
        incomplete_words = incomplete.split()
        if incomplete_words:
            last_word = incomplete_words[-1].lower()
            if last_word in ['và', 'với', 'để', 'khi', 'nếu', 'tại', 'về', 'cho', 'trong', 'của', 'từ']:
                # Remove last incomplete word
                incomplete = ' '.join(incomplete_words[:-1])
        
        # Merge
        merged = incomplete.rstrip() + ' ' + completion.lstrip()
        
        return merged

    def _get_personal_address(self, session_id: str) -> str:
        """🚀 NÂNG CẤP: Get personalized address với bảo đảm không fallback về mặc định generic"""
        
        print("\n" + "="*20 + " DEBUG: _get_personal_address " + "="*20)
        print(f"🕵️‍♂️ [_get_personal_address] Đang lấy xưng hô cho session: {session_id}")
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        print(f"🕵️‍♂️ [_get_personal_address] Context đọc từ cache: {user_context}")

        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        gender = user_context.get('gender', 'other')

        print(f"🕵️‍♂️ [_get_personal_address] Giới tính được xác định là: '{gender}'")
        
        if gender == 'male':
            salutation = 'thầy'
        elif gender == 'female':
            salutation = 'cô'
        else:
            # ✅ CRITICAL: Khi không có giới tính, không fallback về "giảng viên", 
            # mà trả về tên đầy đủ nếu có hoặc giữ nguyên để được xử lý riêng
            if full_name:
                print(f"✅ [_get_personal_address] -> Trả về tên đầy đủ: '{full_name}'")
                print("="*60 + "\n")
                return full_name
            else:
                print(f"✅ [_get_personal_address] -> Trả về fallback: 'giảng viên'")
                print("="*60 + "\n")
                return 'giảng viên'

        if full_name:
            name_suffix = full_name.split()[-1]
            address = f"{salutation} {name_suffix}"
            print(f"✅ [_get_personal_address] -> Trả về xưng hô: '{address}'")
            print("="*60 + "\n")
            return address
        
        print(f"✅ [_get_personal_address] -> Trả về xưng hô: '{salutation}'")
        print("="*60 + "\n")
        return salutation

    def _call_gemini_api_with_smart_tokens(self, prompt: str, strategy: str, max_tokens: int, session_id: str = None, retry_count=0) -> Optional[str]:
        """
        🚀 CRITICAL FIX: Call Gemini API với Smart Token Management và Consistent Personalization
        
        Đây là hàm quan trọng nhất - đã được sửa để truyền session_id cho personalization
        """
        
        # Lấy một key hợp lệ từ bộ quản lý
        api_key_to_use = self.key_manager.get_key()
        
        # Nếu tất cả các key đều bị giới hạn, đợi 5 giây rồi thử lại một lần cuối
        if not api_key_to_use:
            if retry_count == 0:
                logger.warning("All keys are limited. Waiting 5 seconds before one last retry...")
                time.sleep(5)
                return self._call_gemini_api_with_smart_tokens(prompt, strategy, max_tokens, session_id, retry_count=1)
            else:
                logger.error("CRITICAL: All Gemini API keys are rate-limited. Aborting call.")
                # 🚀 CRITICAL FIX: Sử dụng session_id thay vì None
                personal_address = self._get_personal_address(session_id)
                return f"Dạ {personal_address}, hiện tại hệ thống đang quá tải, tất cả các kết nối đều đang bận. Vui lòng thử lại sau khoảng 1 phút nữa ạ. 😥"

        try:
            headers = {'Content-Type': 'application/json'}
            
            strategy_temp_adjustments = {
                'quick_clarify': -0.2, 'direct_enhance': 0.0, 'enhanced_generation': +0.2,
                'completion': -0.3, 'balanced': 0.0
            }
            temp_adjustment = strategy_temp_adjustments.get(strategy, 0.0)
            final_temperature = max(0.1, min(1.0, self.default_generation_config["temperature"] + temp_adjustment))
            
            config = {
                "temperature": final_temperature, "maxOutputTokens": max_tokens,
                "topP": self.default_generation_config["topP"]
            }
            
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
            
            url = f"{self.base_url}?key={api_key_to_use}"
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    # Kiểm tra xem có bị block vì lý do an toàn không
                    if 'finishReason' in candidate and candidate['finishReason'] == 'SAFETY':
                        logger.warning("🚨 Gemini response blocked due to SAFETY reasons.")
                        # 🚀 CRITICAL FIX: Sử dụng session_id thay vì None
                        personal_address = self._get_personal_address(session_id)
                        return f"Dạ {personal_address}, em không thể trả lời câu hỏi này vì lý do an toàn và chính sách nội dung."
                    
                    if 'content' in candidate and 'parts' in candidate['content']:
                        return candidate['content']['parts'][0]['text']
            
            elif response.status_code == 429:
                # Lỗi Rate Limit! Báo cáo key bị lỗi và thử lại ngay với key khác
                self.key_manager.report_failure(api_key_to_use)
                if retry_count == 0:
                    logger.warning(f"Rate limit on key. Retrying immediately with a new key...")
                    return self._call_gemini_api_with_smart_tokens(prompt, strategy, max_tokens, session_id, retry_count=1)
                else:
                    logger.error("Rate limit hit on retry attempt as well. Aborting call.")
                    # 🚀 CRITICAL FIX: Sử dụng session_id thay vì None
                    personal_address = self._get_personal_address(session_id)
                    return f"Dạ {personal_address}, hiện tại hệ thống đang quá tải. Vui lòng thử lại sau ít phút ạ."
            
            else:
                logger.error(f"Gemini API Error {response.status_code} with key '{api_key_to_use[:4]}...': {response.text}")
            
            return None
        
        except requests.exceptions.Timeout:
            logger.error("Gemini API call timed out.")
            # 🚀 CRITICAL FIX: Sử dụng session_id thay vì None
            personal_address = self._get_personal_address(session_id)
            return f"Dạ {personal_address}, yêu cầu xử lý mất quá nhiều thời gian và đã bị ngắt. {personal_address.title()} có thể thử lại với câu hỏi ngắn gọn hơn không ạ?"
        except Exception as e:
            logger.error(f"Smart Gemini API call failed: {str(e)}")
            return None

    # 🚀 SMART VERSIONS of generation methods
    def _generate_direct_lecturer_answer_smart(self, query, context, session_id=None):
        """
        🚀 NÂNG CẤP: Use refined prompt với gender-based addressing và conversation context
        """
        
        # ✅ NEW: Lấy cách xưng hô cá nhân hóa
        personal_address = self._get_personal_address(session_id)
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        db_answer = context.get('db_answer', context.get('response', ''))

        # ✅ NEW: Get conversation context
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Tham khảo ngữ cảnh trên để tránh lặp lại thông tin, tạo câu trả lời mạch lạc.
"""

        # ✅ REFINED PROMPT: Đã include personal_address trong system_prompt
        prompt = f"""{system_prompt}

---
BỐI CẢNH VÀ NHIỆM VỤ

1.  **Kiến thức nền (từ CSDL):**
    "{db_answer}"

2.  **Câu hỏi của giảng viên:**
    "{query}"

{context_section}

3.  **YÊU CẦU CUỐI CÙNG (QUAN TRỌNG):**
    Nhiệm vụ chính của bạn bây giờ là **nhập vai một trợ lý AI** với các đặc điểm và quy tắc được giảng viên định nghĩa trong phần "GHI NHỚ RIÊNG".
    Hãy sử dụng "Kiến thức nền" để trả lời "Câu hỏi của giảng viên" trong khi vẫn duy trì đúng vai trò đó.
    Nếu "GHI NHỚ RIÊNG" trống, hãy trả lời một cách chuyên nghiệp, rõ ràng theo quy tắc mặc định.
    Tạo câu trả lời mạch lạc, tự nhiên, tránh lặp lại thông tin đã thảo luận.
---
Trả lời:
"""

        optimal_tokens = self.token_manager.calculate_optimal_tokens(len(prompt), 'direct_enhance')
        # 🚀 CRITICAL FIX: Pass session_id
        response = self._call_gemini_api_with_smart_tokens(prompt, 'direct_enhance', optimal_tokens, session_id)
        
        # ✅ FIXED: Fallback cũng sử dụng personal_address
        fallback = f"Dạ {personal_address}, {db_answer} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True, 
            'method': 'direct_answer_smart_v5_context', 
            'optimal_tokens': optimal_tokens,
            'personal_addressing': personal_address,  # ✅ NEW: Track addressing used
            'context_aware': bool(recent_summary)  # ✅ NEW: Track context usage
        }

        return response or fallback, token_info

    def _generate_enhanced_lecturer_answer_smart(self, query, context, intent_info, entities, session_id):
        """
        🚀 NÂNG CẤP: Use a refined prompt với gender-based addressing và conversation context
        """
        personal_address = self._get_personal_address(session_id)
        system_prompt = self._get_personalized_system_prompt(session_id)
        db_answer = context.get('db_answer', context.get('response', ''))

        # ✅ NEW: Get conversation context
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Tham khảo ngữ cảnh trên để tránh lặp lại thông tin, tạo câu trả lời mạch lạc và tự nhiên.
"""

        # ✅ REFINED PROMPT: Thay đổi cách ra lệnh
        prompt = f"""{system_prompt}

---
BỐI CẢNH VÀ NHIỆM VỤ

1.  **Kiến thức nền (từ CSDL):**
    "{db_answer}"

2.  **Câu hỏi của giảng viên:**
    "{query}"

{context_section}

3.  **YÊU CẦU CUỐI CÙNG (QUAN TRỌNG):**
    Nhiệm vụ chính của bạn bây giờ là **nhập vai một trợ lý AI** với các đặc điểm và quy tắc được giảng viên định nghĩa trong phần "GHI NHỚ RIÊNG".
    Hãy sử dụng "Kiến thức nền" để trả lời "Câu hỏi của giảng viên" trong khi vẫn duy trì đúng vai trò đó.
    Nếu "GHI NHỚ RIÊNG" trống, hãy trả lời một cách chuyên nghiệp, rõ ràng theo quy tắc mặc định.
    Tạo câu trả lời mạch lạc, tự nhiên, tránh lặp lại thông tin đã thảo luận.
---
Trả lời:
"""

        complexity_hint = 'enhanced_generation' if context.get('generation_boosted', False) else 'balanced'
        optimal_tokens = self.token_manager.calculate_optimal_tokens(len(prompt), complexity_hint)
        # 🚀 CRITICAL FIX: Pass session_id
        response = self._call_gemini_api_with_smart_tokens(prompt, complexity_hint, optimal_tokens, session_id)
        
        # ✅ FIXED: Fallback cũng sử dụng personal_address
        fallback = f"Dạ {personal_address}, {db_answer} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True, 
            'method': 'enhanced_answer_smart_v5_context', 
            'optimal_tokens': optimal_tokens, 
            'generation_boosted': context.get('generation_boosted', False),
            'context_aware': bool(recent_summary)  # ✅ NEW
        }

        return response or fallback, token_info

    def _generate_clarification_request_smart(self, query, context, session_id=None):
        """Generate clarification request with Smart Token Management"""
        
        personal_address = self._get_personal_address(session_id)
        
        # ✅ PREDEFINED smart responses
        clarification_templates = {
            'friendly': f"Dạ {personal_address}, để em có thể hỗ trợ {personal_address} tốt nhất, {personal_address} có thể chia sẻ thêm chi tiết về vấn đề này được không ạ? 😊 Em rất sẵn lòng giúp đỡ!",
            'brief': f"Dạ {personal_address}, cần thêm thông tin chi tiết ạ. 🎓",
            'technical': f"Dạ {personal_address}, để cung cấp hướng dẫn kỹ thuật chính xác, {personal_address} vui lòng cung cấp thêm thông số và yêu cầu cụ thể ạ.",
            'detailed': f"Dạ {personal_address}, để em có thể đưa ra câu trả lời toàn diện và chi tiết nhất, {personal_address} có thể bổ sung thêm về bối cảnh, mục đích sử dụng, và các yêu cầu cụ thể không ạ? Điều này sẽ giúp em hỗ trợ {personal_address} một cách hiệu quả nhất.",
            'professional': f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
        }
        
        response = clarification_templates.get('professional', clarification_templates['professional'])
        
        token_info = {
            'smart_tokens_used': False,  # Used predefined template
            'method': 'clarification_template'
        }
        
        return response, token_info

    def _generate_dont_know_response_smart(self, query, context, session_id=None):
        """Generate don't know response with Smart Token Management và gender-based addressing"""
        
        # ✅ NEW: Lấy cách xưng hô cá nhân hóa dựa trên giới tính
        personal_address = self._get_personal_address(session_id)
        
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
        
        # ✅ FIXED: Sử dụng personal_address thay vì hardcode
        response = f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ {dept} qua email {contact} để được hỗ trợ chi tiết ạ. 🎓"
        
        token_info = {
            'smart_tokens_used': False,  # Used predefined template
            'method': 'dont_know_template',
            'suggested_department': dept,
            'personal_addressing': personal_address  # ✅ NEW: Track addressing used
        }
        
        return response, token_info

    # Keep existing methods but update names and add token info where needed...

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
        if isinstance(context, dict) and context.get('confidence', 0) > 0.85:  # Slightly lower due to re-ranking boost
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
        """🚀 FIXED: Post-process với better duplication detection"""
        if not response:
            return response
        
        personal_address = self._get_personal_address(session_id)
        
        # 1. Sửa các vi phạm vai trò cho giảng viên
        prohibited_phrases = [
            'với tư cách là sinh viên', 'tôi là học sinh',
            'bạn', 'mình', 'anh', 'chị', 'em là sinh viên'
        ]
        for phrase in prohibited_phrases:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, 'em là AI assistant của BDU')
        
        # 2. Sửa xưng hô không đúng với personalization
        response = re.sub(r'\bbạn\b', personal_address, response, flags=re.IGNORECASE)
        response = re.sub(r'\bmình\b', 'em', response, flags=re.IGNORECASE)
        response = re.sub(r'\btôi\b', 'em', response, flags=re.IGNORECASE)
        
        # 3. ✅ CRITICAL: Check if already has proper greeting
        response_stripped = response.strip()
        personalized_start = f"Dạ {personal_address},"
        
        if not response_stripped.lower().startswith(f'dạ {personal_address.lower()}'):
            if response_stripped.lower().startswith('dạ'):
                # Already has "Dạ" but wrong name, replace it
                response = personalized_start + ' ' + response_stripped[3:].strip()
            else:
                response = personalized_start + ' ' + response_stripped
        
        # 4. ✅ CRITICAL: Better ending detection and replacement
        # Check if already has a proper ending
        proper_ending_pattern = r'(thầy|cô|giảng viên)\s+[^.!?]*có\s+cần.*?hỗ trợ.*?thêm.*?gì.*?không.*?ạ\?'
        
        if not re.search(proper_ending_pattern, response.lower()):
            # Remove any partial/incorrect endings first
            response = re.sub(r'\s*🎓.*$', '', response.strip())
            response = re.sub(r'\s*(có cần.*?không ạ\?|Cần.*?không\?|Có.*?không\?).*$', '', response.strip())
            
            # Add the correct ending
            if not response.strip().endswith(('.', '!', '?')):
                response += '.'
            response += f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ? 🎓'
        
        # 5. Remove excessive formatting
        response = re.sub(r'\*\*\d+\.\s*', '', response)
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^\s*[•\-\*]\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        
        # 6. ✅ NEW: Final cleanup for duplicates
        # Remove duplicate names in ending
        duplicate_name_pattern = f'({re.escape(personal_address.title())}).*?\\1'
        response = re.sub(duplicate_name_pattern, r'\1', response)
        
        return response.strip()
    
    def _get_contextual_out_of_scope_response_lecturer(self, conversation_context, session_id=None):
        """Out of scope response cho giảng viên với personalization"""
        
        personal_address = self._get_personal_address(session_id)
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        department_name = user_context.get('department_name', '')
        
        if conversation_context.get('context_summary'):
            if department_name:
                return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} còn muốn hỏi gì về {conversation_context['context_summary'].lower()} cho ngành {department_name} không ạ?"
            else:
                return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} còn muốn hỏi gì về {conversation_context['context_summary'].lower()} không ạ?"
        
        if department_name:
            return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} có câu hỏi nào khác về ngành {department_name} không ạ?"
        else:
            return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} có câu hỏi nào khác về trường không ạ?"
    
    def _get_smart_fallback_with_context_lecturer(self, query, intent_info, conversation_context, session_id=None):
        """Smart fallback với conversation context cho giảng viên và personalization"""
        
        personal_address = self._get_personal_address(session_id)
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        department_name = user_context.get('department_name', '')
        
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
            'điểm', 'đạt', 'không đạt', 'học lại', 'nâng điểm', 'cải thiện điểm',
            'điểm trung bình', 'trung bình', 'tính điểm', 'tính',
            'chuyển đổi', 'công nhận', 'khối lượng', 'tối thiểu', 'chương trình', 
            'phần trăm', 'tối đa', 'giới hạn',
            'tốt nghiệp', 'lễ tốt nghiệp', 'tham dự', 'được phép', 'bằng cấp', 
            'văn bằng', 'cử nhân', 'cấp bằng', 'nhận bằng',
            'thường trực', 'kỷ luật', 'hội đồng thi đua', 'danh sách', 'thành phần',
            'theo quy định', 'quy định về', 'thể lệ', 'hướng dẫn', 'thủ tục',
            'điều kiện', 'yêu cầu',
            'như thế nào', 'bao nhiêu', 'là ai', 'ai là', 'làm gì', 'ở đâu', 
            'khi nào', 'có được',
            
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
            'diem', 'dat', 'khong dat', 'hoc lai', 'nang diem', 'cai thien diem',
            'diem trung binh', 'trung binh', 'tb', 'dtb', 'tinh diem', 'tinh',
            'chuyen doi', 'cong nhan', 'khoi luong', 'toi thieu', 'chuong trinh',
            'phan tram', 'toi da', 'gioi han',
            'tot nghiep', 'le tot nghiep', 'tham du', 'duoc phep', 'bang cap',
            'van bang', 'cu nhan', 'cap bang', 'nhan bang',
            'thuong truc', 'ky luat', 'hoi dong thi dua', 'danh sach', 'thanh phan',
            'ai phu trach', 'theo quy dinh', 'quy dinh ve', 'the le', 'huong dan', 'thu tuc',
            'dieu kien', 'yeu cau', 'nhu the nao', 'bao nhieu', 'la ai',
            'ai la', 'lam gi', 'o dau', 'khi nao', 'co duoc'
        ]
        
        if not query:
            return False
        
        query_lower = query.lower()
        return any(kw in query_lower for kw in lecturer_education_keywords)

    # Keep the remaining essential methods...
    def _build_enhanced_prompt(self, query: str, context=None, intent_info=None, entities=None, session_id=None):
        """🚀 NÂNG CẤP: Build enhanced prompt với conversation context"""
        system_prompt = self._get_personalized_system_prompt(session_id)
        personal_address = self._get_personal_address(session_id)
        
        context_info = str(context.get('response', '')) if isinstance(context, dict) else str(context or '')
        
        # ✅ NEW: Get conversation context
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Tham khảo ngữ cảnh trên để tránh lặp lại thông tin, tạo câu trả lời mạch lạc.
"""
        
        prompt = f"""{system_prompt}
        
CÂU HỎI: {query}
THÔNG TIN: {context_info}

{context_section}

YÊU CẦU:
- Bắt đầu: "Dạ {personal_address},"
- Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
- Tạo câu trả lời mạch lạc, tự nhiên, tránh lặp lại thông tin đã thảo luận

Trả lời:"""
        return prompt
    
    def validate_user_preferences(self, preferences):
        """Validate user preferences - simplified for user_memory_prompt only"""
        errors, warnings = [], []
        
        if 'user_memory_prompt' in preferences:
            memory = preferences['user_memory_prompt']
            if isinstance(memory, str):
                if len(memory) > 1500:
                    errors.append("user_memory_prompt too long (max 1500 characters)")
                elif len(memory) > 1400:
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

    def get_system_status(self) -> Dict[str, Any]:
        """🚀 NÂNG CẤP: Get system status với Smart Token Management, External API info và Consistent Personalization"""
        try:
            test_prompt = "Test ngắn cho giảng viên"
            # 🚀 CRITICAL FIX: Pass session_id=None for test (có thể pass test session)
            response = self._call_gemini_api_with_smart_tokens(test_prompt, 'quick_clarify', 80, session_id="test")
            
            return {
                'gemini_api_available': response is not None,
                'api_key_configured': bool(self.key_manager.keys),
                'service_status': 'active' if response else 'error',
                'mode': 'smart_token_lecturer_focused_with_user_memory_and_context_and_consistent_personalization',  # ✅ Updated
                'memory_sessions': len(self.memory.conversations),
                'personalization_sessions': len(self._user_context_cache),
                'adaptive_token_range': self.token_manager.adaptive_token_range,
                'features': [
                    'smart_token_management',
                    'auto_response_completion',
                    'adaptive_token_allocation',
                    'incomplete_response_detection',
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
                    'user_memory_prompt_support',  # ✅ NEW feature
                    'flexible_personalization',    # ✅ NEW feature
                    'external_api_data_processing',
                    'lecturer_schedule_formatting',
                    'personal_information_handling',
                    'gender_based_addressing',  # ✅ NEW feature
                    'conversation_context_summary',  # ✅ NEW feature
                    'mạch_lạc_response_generation',   # ✅ NEW feature
                    'consistent_personalization_in_errors',  # 🚀 NEW feature
                    'session_id_propagation_in_api_calls',  # 🚀 NEW feature
                    'graceful_error_handling_with_personalization'  # 🚀 NEW feature
                ]
            }
        except Exception as e:
            return {
                'gemini_api_available': False,
                'service_status': 'error',
                'error': str(e),
                'consistent_personalization': True,  # 🚀 NEW: Even in error, personalization is supported
                'graceful_degradation': True  # 🚀 NEW: System supports graceful degradation
            }