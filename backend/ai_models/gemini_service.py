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

class SmartTokenManager:
    """🧠 Smart Token Management System - Tự động tăng token và hoàn thiện response"""
    
    def __init__(self):
        # ✅ ADAPTIVE TOKEN RANGES cho từng style
        self.style_token_ranges = {
            'brief': {
                'min': 40, 'optimal': 80, 'max': 120,
                'expected_sentences': 1, 'avg_chars_per_sentence': 60
            },
            'professional': {
                'min': 80, 'optimal': 150, 'max': 220,
                'expected_sentences': 2, 'avg_chars_per_sentence': 80
            },
            'friendly': {
                'min': 100, 'optimal': 180, 'max': 280,
                'expected_sentences': 3, 'avg_chars_per_sentence': 70
            },
            'technical': {
                'min': 150, 'optimal': 250, 'max': 400,
                'expected_sentences': 4, 'avg_chars_per_sentence': 90
            },
            'detailed': {
                'min': 200, 'optimal': 350, 'max': 500,
                'expected_sentences': 5, 'avg_chars_per_sentence': 85
            }
        }
        
        # ✅ COMPLETION DETECTION patterns
        self.incomplete_patterns = [
            r'[^.!?]\s*$',  # Không kết thúc bằng dấu câu
            r'\b(và|hoặc|với|để|khi|nếu|tại|về|cho|trong|của|từ)\s*$',  # Kết thúc bằng từ nối
            r'\b(thầy/cô|em|sẽ|có|được|phải|cần|nên)\s*$',  # Kết thúc bằng từ chưa hoàn chỉnh
            r'[,;:]\s*$',  # Kết thúc bằng dấu phẩy/chấm phẩy
            r'\b(Dạ|Ạ|thầy|cô)\s*$',  # Câu chào chưa hoàn chỉnh
        ]
        
        # ✅ SENTENCE ENDING patterns để kiểm tra câu hoàn chỉnh
        self.complete_endings = [
            r'[.!?]\s*$',  # Kết thúc bằng dấu câu
            r'ạ[.!?]\s*$',  # Kết thúc bằng "ạ" + dấu câu
            r'không ạ\?\s*$',  # "có cần hỗ trợ thêm gì không ạ?"
            r'🎓\s*$',  # Emoji kết thúc
            r'@bdu\.edu\.vn\s*$',  # Email ending
        ]
        
        logger.info("✅ SmartTokenManager initialized with adaptive ranges")
    
    def calculate_optimal_tokens(self, response_style: str, prompt_length: int, complexity_hint: str = None) -> int:
        """🎯 Tính toán tokens tối ưu dựa trên style và độ phức tạp"""
        
        style_config = self.style_token_ranges.get(response_style, self.style_token_ranges['professional'])
        
        # Base tokens từ style
        base_tokens = style_config['optimal']
        
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
        min_tokens = style_config['min']
        max_tokens = style_config['max']
        
        return max(min_tokens, min(max_tokens, base_tokens))
    
    def is_response_incomplete(self, response: str, expected_style: str) -> Dict[str, Any]:
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
        style_config = self.style_token_ranges.get(expected_style, self.style_token_ranges['professional'])
        expected_sentences = style_config['expected_sentences']
        
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
        required_ending = r'(thầy/cô có cần hỗ trợ thêm gì không ạ\?|ạ[.!?]|\?)'
        if not re.search(required_ending, response.lower()):
            return {
                'incomplete': True,
                'reason': 'missing_proper_ending',
                'confidence': 0.9
            }
        
        # ✅ CHECK 4: Proper greeting start
        if not re.match(r'dạ\s+(thầy/cô|cô|thầy)', response.lower()):
            return {
                'incomplete': True,
                'reason': 'missing_proper_greeting',
                'confidence': 0.6
            }
        
        return {'incomplete': False, 'reason': 'complete', 'confidence': 0.9}
    
    def estimate_completion_tokens(self, incomplete_response: str, target_style: str) -> int:
        """📊 Ước tính tokens cần để hoàn thiện response"""
        
        style_config = self.style_token_ranges.get(target_style, self.style_token_ranges['professional'])
        
        # Estimate current length in tokens (rough: 1 token ≈ 3-4 chars in Vietnamese)
        current_tokens = len(incomplete_response) // 3
        
        # Target tokens for complete response
        target_tokens = style_config['optimal']
        
        # Additional tokens needed
        additional_needed = max(20, target_tokens - current_tokens)
        
        return min(additional_needed, 150)  # Cap at 150 additional tokens

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
    """🚀 Advanced Gemini Response Generator với Smart Token Management"""
    
    def __init__(self, api_key: str = None):
        from django.conf import settings
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = "gemini-2.0-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        self.memory = ConversationMemory(max_history=10)
        self.vietnamese_restorer = SimpleVietnameseRestorer(self.api_key)
        
        # ✅ NEW: Smart Token Manager
        self.token_manager = SmartTokenManager()
        
        # User context cache for personalization
        self._user_context_cache = {}
        
        # ✅ ENHANCED: Dynamic style configs với smart token ranges
        self.style_generation_configs = {
            'professional': {
                "temperature": 0.3,
                "topP": 0.8
            },
            'friendly': {
                "temperature": 0.6,
                "topP": 0.9
            },
            'technical': {
                "temperature": 0.2,
                "topP": 0.7
            },
            'brief': {
                "temperature": 0.4,
                "topP": 0.8
            },
            'detailed': {
                "temperature": 0.5,
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
        
        logger.info("✅ Enhanced Gemini Response Generator initialized with Smart Token Management")

    # ✅ NEW: Process external API data
    def _generate_external_api_response(self, query, context, session_id=None, response_style='professional'):
        """Generate response from external API data"""
        
        api_data = context.get('api_data', {})
        lecturer_info = api_data.get('lecturer_info', {})
        schedule_summary = api_data.get('schedule_summary', {})
        daily_schedule = api_data.get('daily_schedule', {})
        
        # Get personal addressing
        personal_address = self._get_personal_address_from_api_data(lecturer_info, session_id)
        
        # Build comprehensive prompt for external API data
        prompt = self._build_external_api_prompt(
            query, api_data, personal_address, response_style
        )
        
        # Calculate optimal tokens for external API response
        optimal_tokens = self.token_manager.calculate_optimal_tokens(
            response_style, 
            len(prompt), 
            'external_api_processing'
        )
        
        logger.info(f"🌐 Processing external API data with {optimal_tokens} tokens")
        
        response = self._call_gemini_api_with_smart_tokens(
            prompt, 'external_api_processing', response_style, optimal_tokens
        )
        
        if not response:
            # Fallback if Gemini fails
            return self._get_external_api_fallback_response(api_data, personal_address)
        
        # Post-process for consistency
        response = self._post_process_external_api_response(
            response, lecturer_info, query, session_id
        )
        
        return response
    
    def _build_external_api_prompt(self, query, api_data, personal_address, response_style):
        """Build comprehensive prompt for external API data processing"""
        
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
            lecturer_info, response_style
        )
        
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

❓ CÂU HỎI CỦA GIẢNG VIÊN: {query}
🔍 NGỮ CẢNH TÌM KIẾM: {query_context}

📝 YÊU CẦU TRẢ LỜI:
- Xưng hô: "Dạ {personal_address},"
- Phong cách: {response_style}
- Trả lời CHÍNH XÁC dựa trên dữ liệu thực tế từ hệ thống
- Định dạng thông tin dễ đọc, rõ ràng
- Bao gồm các chi tiết quan trọng: thời gian, địa điểm, môn học
- Kết thúc: "{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
- KHÔNG CHẾ TẠO thông tin không có trong dữ liệu

🎨 HƯỚNG DẪN THEO PHONG CÁCH:
"""

        # Add style-specific guidance
        if response_style == 'professional':
            prompt += """
- Ngôn từ trang trọng, chính thức
- Trình bày có logic, rõ ràng
- Tập trung vào thông tin cốt lõi"""
        elif response_style == 'friendly':
            prompt += """
- Ngôn từ gần gũi, ấm áp
- Có thể sử dụng emoji phù hợp 😊
- Tạo cảm giác thoải mái"""
        elif response_style == 'detailed':
            prompt += """
- Giải thích chi tiết từng khía cạnh
- Cung cấp ngữ cảnh và thông tin bổ sung
- Phân tích toàn diện"""
        elif response_style == 'brief':
            prompt += """
- Trả lời ngắn gọn, súc tích
- Chỉ thông tin cần thiết nhất
- Tránh dài dòng"""
        elif response_style == 'technical':
            prompt += """
- Sử dụng thuật ngữ chính xác
- Cung cấp thông số kỹ thuật
- Chi tiết về quy trình, hệ thống"""

        prompt += "\n\nTrả lời:"

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

    def _get_personalized_system_prompt_for_external_api(self, lecturer_info, response_style):
        """Get personalized system prompt for external API processing"""
        
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        chuc_danh = lecturer_info.get('chuc_danh', '')
        
        name_parts = ten_giang_vien.split() if ten_giang_vien else []
        name_suffix = name_parts[-1] if name_parts else ''
        
        base_prompt = f"""Bạn là AI assistant của Đại học Bình Dương (BDU), chuyên hỗ trợ giảng viên.

🎯 THÔNG TIN NGƯỜI DÙNG:
- Bạn đang trả lời cho {chuc_danh} {ten_giang_vien}
- Xưng hô: "thầy/cô {name_suffix}" (TUYỆT ĐỐI KHÔNG dùng "bạn", "mình", "anh/chị")
- Đây là thông tin CÁ NHÂN từ hệ thống chính thức của trường

🎯 QUY TẮC QUAN TRỌNG:
- LUÔN bắt đầu: "Dạ thầy/cô {name_suffix},"
- Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
- SỬ DỤNG CHÍNH XÁC thông tin từ hệ thống - KHÔNG CHẾ TẠO
- Trình bày thông tin cá nhân một cách tự nhiên, dễ hiểu
- KHÔNG dùng format phức tạp với **1. **2. hay bullets khi không cần thiết"""

        return base_prompt

    def _get_personal_address_from_api_data(self, lecturer_info, session_id):
        """Get personal address from API data or session"""
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        
        if ten_giang_vien:
            name_suffix = ten_giang_vien.split()[-1]
            return f"thầy/cô {name_suffix}"
        
        # Fallback to session-based addressing
        return self._get_personal_address(session_id)

    def _post_process_external_api_response(self, response, lecturer_info, query, session_id):
        """Post-process external API response for consistency"""
        if not response:
            return response
        
        # Get personal addressing info
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        name_suffix = ten_giang_vien.split()[-1] if ten_giang_vien else ''
        personal_address = f"thầy/cô {name_suffix}" if name_suffix else "thầy/cô"
        
        # 1. Fix addressing inconsistencies
        response = re.sub(r'\bbạn\b', personal_address, response, flags=re.IGNORECASE)
        response = re.sub(r'\bmình\b', 'em', response, flags=re.IGNORECASE)
        response = re.sub(r'\btôi\b', 'em', response, flags=re.IGNORECASE)
        
        # 2. Ensure proper greeting
        response_stripped = response.strip()
        personalized_start = f"Dạ {personal_address},"
        
        if not response_stripped.lower().startswith('dạ thầy/cô') and not response_stripped.lower().startswith(f'dạ {personal_address.lower()}'):
            if response_stripped.lower().startswith('dạ'):
                response = personalized_start + ' ' + response_stripped[3:].strip()
            else:
                response = personalized_start + ' ' + response_stripped
        
        # 3. Ensure proper ending
        if not response.strip().endswith('có cần hỗ trợ thêm gì không ạ?'):
            response = re.sub(r'\s*(Thầy/cô có.*?không ạ\?|Cần.*?không\?|Có.*?không\?)?\s*$', '', response.strip())
            response += ' Thầy/cô có cần hỗ trợ thêm gì không ạ?'
        
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

    # 🚀 ENHANCED: Generate response với Smart Token Management
    def generate_response(self, query: str, context: Optional[Dict] = None, 
                      intent_info: Optional[Dict] = None, entities: Optional[Dict] = None,
                      session_id: str = None) -> Dict[str, Any]:
        """Generate response with Smart Token Management & Auto-completion"""
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
                user_response_style = self._get_user_response_style(session_id)
                response = self._generate_external_api_response(query, context, session_id, user_response_style)
                
                token_info = {
                    'smart_tokens_used': True,
                    'method': 'external_api_processing',
                    'response_style': user_response_style
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
                    'response_style': user_response_style,
                    'external_api_processed': True,
                    'token_info': token_info
                }
            
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
            
            # ✅ ENHANCED: Handle instruction-based responses với Smart Tokens
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'direct_answer_lecturer':
                response, token_info = self._generate_direct_lecturer_answer_smart(query, context, session_id, user_response_style)
            elif instruction in ['enhance_answer_lecturer', 'enhance_answer_lecturer_boosted']:
                response, token_info = self._generate_enhanced_lecturer_answer_smart(query, context, intent_info, entities, session_id, user_response_style)
            elif instruction == 'clarification_needed':
                response, token_info = self._generate_clarification_request_smart(query, context, session_id, user_response_style)
            elif instruction == 'dont_know_lecturer':
                response, token_info = self._generate_dont_know_response_smart(query, context, session_id, user_response_style)
            else:
                # Check out of scope and generate response
                if context and context.get('emergency_education', False):
                    print(f"🚨 GEMINI: Emergency education mode activated")
                    pass 
                elif not self._is_lecturer_education_related(query) and not context.get('force_education_response', False):
                    response = self._get_contextual_out_of_scope_response_lecturer(conversation_context, session_id, user_response_style)
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
                        'response_style': user_response_style,
                        'token_info': token_info
                    }
                
                # ✅ ENHANCED: Use Smart Token Generation
                response, token_info = self._generate_smart_style_aware_response(query, context, session_id, user_response_style, response_strategy)
            
            final_response = response or self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id, user_response_style)
            
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
                'response_style': user_response_style,
                'style_applied': user_response_style,
                'enhanced_generation': response_strategy == 'enhanced_generation',
                'token_info': token_info  # ✅ NEW: Smart token information
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
                'response_style': user_response_style,
                'token_info': {'smart_tokens_used': False, 'method': 'fallback'}
            }

    # 🧠 SMART TOKEN GENERATION METHODS

    def _generate_smart_style_aware_response(self, query: str, context=None, session_id=None, response_style='professional', strategy='balanced'):
        """🚀 Generate response with Smart Token Management"""
        
        prompt = self._build_enhanced_prompt(query, context, None, None, session_id, response_style)
        
        # ✅ STEP 1: Calculate optimal tokens
        optimal_tokens = self.token_manager.calculate_optimal_tokens(
            response_style, 
            len(prompt), 
            complexity_hint=strategy
        )
        
        print(f"🧠 SMART TOKENS: {response_style} -> {optimal_tokens} tokens")
        
        # ✅ STEP 2: First attempt with optimal tokens
        response = self._call_gemini_api_with_smart_tokens(prompt, strategy, response_style, optimal_tokens)
        
        if not response:
            return self._get_smart_fallback_with_context_lecturer(query, None, {}, session_id, response_style), {
                'smart_tokens_used': True, 'method': 'fallback_after_api_failure', 'tokens_attempted': optimal_tokens
            }
        
        # ✅ STEP 3: Check if response is complete
        completion_check = self.token_manager.is_response_incomplete(response, response_style)
        
        if completion_check['incomplete']:
            print(f"⚠️ INCOMPLETE RESPONSE detected: {completion_check['reason']}")
            
            # ✅ STEP 4: Auto-completion attempt
            completed_response = self._auto_complete_response(response, query, context, session_id, response_style, completion_check)
            
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
            'response_style': response_style,
            'strategy': strategy
        }
        
        return response, token_info

    def _auto_complete_response(self, incomplete_response: str, original_query: str, context, session_id: str, response_style: str, completion_info: Dict) -> Optional[str]:
        """🔧 Auto-complete incomplete response"""
        
        if completion_info['confidence'] < 0.6:  # Don't auto-complete if not confident it's incomplete
            return None
        
        # ✅ Calculate completion tokens needed
        completion_tokens = self.token_manager.estimate_completion_tokens(incomplete_response, response_style)
        
        # ✅ Build completion prompt
        completion_prompt = self._build_completion_prompt(incomplete_response, original_query, context, session_id, response_style, completion_info)
        
        print(f"🔧 AUTO-COMPLETION: Attempting with {completion_tokens} tokens")
        
        # ✅ Call API to complete
        completion = self._call_gemini_api_with_smart_tokens(completion_prompt, 'completion', response_style, completion_tokens)
        
        if completion:
            # ✅ Merge incomplete + completion
            if completion_info['reason'] == 'missing_proper_ending':
                # Just add proper ending
                return incomplete_response.rstrip() + ' Thầy/cô có cần hỗ trợ thêm gì không ạ?'
            elif completion_info['reason'] == 'missing_proper_greeting':
                # Add proper greeting
                personal_address = self._get_personal_address(session_id)
                return f"Dạ {personal_address}, " + incomplete_response.lstrip()
            else:
                # Merge content
                merged = self._merge_incomplete_and_completion(incomplete_response, completion)
                return merged
        
        return None

    def _build_completion_prompt(self, incomplete_response: str, original_query: str, context, session_id: str, response_style: str, completion_info: Dict) -> str:
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
            - Phong cách: {response_style}
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
            - Phong cách: {response_style}
            
            Câu trả lời hoàn chỉnh:"""
        
        return completion_prompt

    def _merge_incomplete_and_completion(self, incomplete: str, completion: str) -> str:
        """🔧 Merge incomplete response with completion"""
        
        # Clean completion
        completion = completion.strip()
        
        # Remove redundant greetings from completion
        completion = re.sub(r'^(dạ\s+thầy/cô,?\s*)', '', completion, flags=re.IGNORECASE)
        
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
        """Get personalized address for user"""
        user_context = self._user_context_cache.get(session_id, {}) if session_id else {}
        full_name = user_context.get('full_name', '')
        
        if full_name:
            name_suffix = full_name.split()[-1]
            return f"thầy/cô {name_suffix}"
        else:
            return "thầy/cô"

    # 🚀 ENHANCED API CALL với Smart Tokens
    def _call_gemini_api_with_smart_tokens(self, prompt: str, strategy: str, response_style: str, max_tokens: int) -> Optional[str]:
        """Call Gemini API with Smart Token Management"""
        try:
            headers = {'Content-Type': 'application/json'}
            
            # Get style-specific config (without maxOutputTokens - handled by smart tokens)
            style_config = self.style_generation_configs.get(response_style, self.style_generation_configs['professional'])
            
            # Strategy-specific temperature adjustments
            strategy_temp_adjustments = {
                'quick_clarify': -0.2,
                'direct_enhance': 0.0,
                'enhanced_generation': +0.2,
                'completion': -0.3,  # ✅ NEW: Lower temp for completion
                'balanced': 0.0
            }
            
            temp_adjustment = strategy_temp_adjustments.get(strategy, 0.0)
            final_temperature = max(0.1, min(1.0, style_config["temperature"] + temp_adjustment))
            
            config = {
                "temperature": final_temperature,
                "maxOutputTokens": max_tokens,  # ✅ Use smart calculated tokens
                "topP": style_config["topP"]
            }
            
            print(f"🚀 SMART API CONFIG: {response_style}/{strategy} -> temp={config['temperature']}, tokens={config['maxOutputTokens']}")
            
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
            response = requests.post(url, headers=headers, json=data, timeout=30)  # Increased timeout for larger responses
            
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
            logger.error(f"Smart Gemini API call failed: {str(e)}")
            return None

    # 🚀 SMART VERSIONS of generation methods
    def _generate_direct_lecturer_answer_smart(self, query, context, session_id=None, response_style='professional'):
        """Generate direct answer with Smart Token Management"""
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        
        prompt = f"""
        {system_prompt}
        
        NHIỆM VỤ: Trả lời TRỰC TIẾP cho giảng viên BDU
        
        CÂU HỎI GIẢNG VIÊN: {query}
        
        THÔNG TIN CHÍNH XÁC TỪ CSDL:
        {context.get('db_answer', context.get('response', ''))}
        
        YÊU CẦU:
        - Dùng CHÍNH XÁC thông tin từ CSDL
        - Bắt đầu: "Dạ thầy/cô,"
        - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        - Áp dụng phong cách: {response_style}
        - KHÔNG format phức tạp
        
        Trả lời:
        """
        
        # ✅ Smart token calculation
        optimal_tokens = self.token_manager.calculate_optimal_tokens(response_style, len(prompt), 'direct_enhance')
        
        response = self._call_gemini_api_with_smart_tokens(prompt, 'direct_enhance', response_style, optimal_tokens)
        
        fallback = f"Dạ thầy/cô, {context.get('db_answer', context.get('response', 'thông tin đã cung cấp'))} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True,
            'method': 'direct_answer_smart',
            'optimal_tokens': optimal_tokens,
            'response_style': response_style
        }
        
        return response or fallback, token_info

    def _generate_enhanced_lecturer_answer_smart(self, query, context, intent_info, entities, session_id, response_style='professional'):
        """Generate enhanced answer with Smart Token Management"""
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        
        is_generation_boosted = (
            context.get('generation_boosted', False) or 
            context.get('instruction') == 'enhance_answer_lecturer_boosted'
        )
        
        complexity_hint = 'enhanced_generation' if is_generation_boosted else 'balanced'
        
        if is_generation_boosted:
            prompt = f"""
            {system_prompt}
            
            NHIỆM VỤ ĐẶC BIỆT: Trả lời có BỔ SUNG PHONG PHÚ cho giảng viên BDU
            
            CÂU HỎI GIẢNG VIÊN: {query}
            
            THÔNG TIN CƠ BẢN TỪ CSDL:
            {context.get('db_answer', context.get('response', ''))}
            
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
            {context.get('db_answer', context.get('response', ''))}
            
            YÊU CẦU:
            - Sử dụng thông tin CSDL làm gốc
            - Bổ sung ngữ cảnh phù hợp nếu cần
            - Áp dụng phong cách: {response_style}
            - Bắt đầu: "Dạ thầy/cô,"
            - Kết thúc: "Thầy/cô có cần hỗ trợ thêm gì không ạ?"
            
            Trả lời:
            """
        
        # ✅ Smart token calculation
        optimal_tokens = self.token_manager.calculate_optimal_tokens(response_style, len(prompt), complexity_hint)
        
        response = self._call_gemini_api_with_smart_tokens(prompt, complexity_hint, response_style, optimal_tokens)
        
        fallback = f"Dạ thầy/cô, {context.get('db_answer', context.get('response', 'thông tin đã cung cấp'))} 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True,
            'method': 'enhanced_answer_smart',
            'optimal_tokens': optimal_tokens,
            'response_style': response_style,
            'generation_boosted': is_generation_boosted
        }
        
        return response or fallback, token_info

    def _generate_clarification_request_smart(self, query, context, session_id=None, response_style='professional'):
        """Generate clarification request with Smart Token Management"""
        
        personal_address = self._get_personal_address(session_id)
        
        # ✅ PREDEFINED smart responses based on style
        clarification_templates = {
            'friendly': f"Dạ {personal_address}, để em có thể hỗ trợ {personal_address} tốt nhất, {personal_address} có thể chia sẻ thêm chi tiết về vấn đề này được không ạ? 😊 Em rất sẵn lòng giúp đỡ!",
            'brief': f"Dạ {personal_address}, cần thêm thông tin chi tiết ạ. 🎓",
            'technical': f"Dạ {personal_address}, để cung cấp hướng dẫn kỹ thuật chính xác, {personal_address} vui lòng cung cấp thêm thông số và yêu cầu cụ thể ạ.",
            'detailed': f"Dạ {personal_address}, để em có thể đưa ra câu trả lời toàn diện và chi tiết nhất, {personal_address} có thể bổ sung thêm về bối cảnh, mục đích sử dụng, và các yêu cầu cụ thể không ạ? Điều này sẽ giúp em hỗ trợ {personal_address} một cách hiệu quả nhất.",
            'professional': f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
        }
        
        response = clarification_templates.get(response_style, clarification_templates['professional'])
        
        token_info = {
            'smart_tokens_used': False,  # Used predefined template
            'method': 'clarification_template',
            'response_style': response_style
        }
        
        return response, token_info

    def _generate_dont_know_response_smart(self, query, context, session_id=None, response_style='professional'):
        """Generate don't know response with Smart Token Management"""
        
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
        
        response = f"Dạ thầy/cô, em chưa có thông tin về vấn đề này. Thầy/cô có thể liên hệ {dept} qua email {contact} để được hỗ trợ chi tiết ạ. 🎓"
        
        token_info = {
            'smart_tokens_used': False,  # Used predefined template
            'method': 'dont_know_template',
            'response_style': response_style,
            'suggested_department': dept
        }
        
        return response, token_info

    # Keep existing methods but update names and add token info where needed...
    # [Rest of the methods remain the same but simplified for brevity]

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
        personal_address = self._get_personal_address(session_id)
        
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
    
    def _get_smart_fallback_with_context_lecturer(self, query, intent_info, conversation_context, session_id=None, user_response_style='professional'):
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

    # Keep the remaining essential methods...
    def _build_enhanced_prompt(self, query: str, context=None, intent_info=None, entities=None, session_id=None, response_style='professional'):
        """Build enhanced prompt with style awareness"""
        system_prompt = self._get_personalized_system_prompt(session_id)
        personal_address = self._get_personal_address(session_id)
        
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
    
    def test_response_style(self, test_query: str, response_style: str, session_id=None):
        """Test response style functionality with Smart Token Management"""
        try:
            test_context = {'response': 'Đây là thông tin test từ database.', 'confidence': 0.8}
            response, token_info = self._generate_smart_style_aware_response(test_query, test_context, session_id, response_style, 'balanced')
            
            style_info = {
                'professional': 'Chuyên nghiệp - trang trọng, lịch sự',
                'friendly': 'Thân thiện - gần gũi, vui vẻ với emoji',
                'technical': 'Kỹ thuật - chi tiết, thuật ngữ chuyên môn',
                'brief': 'Ngắn gọn - súc tích, đi thẳng vào vấn đề',
                'detailed': 'Chi tiết - đầy đủ, nhiều ví dụ'
            }
            
            # ✅ Smart token range info
            token_range = self.token_manager.style_token_ranges.get(response_style, {})
            
            return {
                'success': True,
                'test_query': test_query,
                'response': response,
                'current_style': response_style,
                'current_style_name': style_info.get(response_style, 'Unknown'),
                'style_applied': response_style,
                'token_info': token_info,
                'token_range': token_range,
                'recommendation': f"Phong cách '{style_info.get(response_style)}' đã được áp dụng thành công với Smart Token Management."
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
    
    # 🚀 ENHANCED: System status with Smart Token info
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status with Smart Token Management and External API info"""
        try:
            test_prompt = "Test ngắn cho giảng viên"
            response = self._call_gemini_api_with_smart_tokens(test_prompt, 'quick_clarify', 'professional', 80)
            
            return {
                'gemini_api_available': response is not None,
                'api_key_configured': bool(self.api_key),
                'service_status': 'active' if response else 'error',
                'mode': 'smart_token_lecturer_focused_with_external_api',  # ✅ Updated
                'memory_sessions': len(self.memory.conversations),
                'personalization_sessions': len(self._user_context_cache),
                'supported_styles': list(self.style_generation_configs.keys()),
                'smart_token_ranges': self.token_manager.style_token_ranges,
                'features': [
                    'smart_token_management',
                    'auto_response_completion',
                    'adaptive_token_allocation',
                    'style_aware_token_calculation',
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
                    'response_style_support',
                    'style_aware_generation',
                    'dynamic_style_configs',
                    'external_api_data_processing',  # ✅ NEW feature
                    'lecturer_schedule_formatting',  # ✅ NEW feature
                    'personal_information_handling'  # ✅ NEW feature
                ]
            }
        except Exception as e:
            return {
                'gemini_api_available': False,
                'service_status': 'error',
                'error': str(e)
            }