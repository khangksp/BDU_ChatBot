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

class AdvancedConfidenceManager:
    def __init__(self):
        self.MAX_CONFIDENCE = 1.0
        self.confidence_calibration_rules = {
            'high_semantic_match': 0.95,
            'medium_semantic_match': 0.75, 
            'low_semantic_match': 0.45,
            'keyword_match_bonus': 0.1,
            'context_match_bonus': 0.05,
            'document_context_bonus': 0.1
        }
        
        # Thresholds for decision making - adjusted cho Advanced RAG
        self.decision_thresholds = {
            'direct_answer': 0.8,      # Lowered from 0.85 due to re-ranking boost
            'enhanced_answer': 0.45,   # Lowered from 0.5 
            'ask_clarification': 0.25, # Lowered from 0.3
            'dont_know': 0.1
        }
        
        logger.info("✅ AdvancedConfidenceManager initialized with overflow protection")
    
    def normalize_confidence(self, raw_confidence: float, source: str = "unknown") -> float:
        if raw_confidence is None or not isinstance(raw_confidence, (int, float)):
            logger.warning(f"⚠️ Invalid confidence value: {raw_confidence} from {source}")
            return 0.1  # Safe default
        
        # Cap at maximum confidence
        normalized = min(self.MAX_CONFIDENCE, abs(float(raw_confidence)))
        
        if raw_confidence > self.MAX_CONFIDENCE:
            logger.info(f"🛡️ Confidence capped: {raw_confidence:.3f} -> {normalized:.3f} (source: {source})")
        
        return normalized
    
    def calculate_response_confidence(self, semantic_score: float = 0, 
                                   keyword_score: float = 0,
                                   context_bonus: float = 0,
                                   method: str = "hybrid") -> float:
        base_confidence = 0.0
        
        # Base từ semantic score
        if semantic_score >= 0.8:
            base_confidence = self.confidence_calibration_rules['high_semantic_match']
        elif semantic_score >= 0.6:
            base_confidence = self.confidence_calibration_rules['medium_semantic_match']
        else:
            base_confidence = self.confidence_calibration_rules['low_semantic_match']
        
        # Apply bonuses
        if keyword_score > 0.5:
            base_confidence += self.confidence_calibration_rules['keyword_match_bonus']
        
        if context_bonus > 0:
            base_confidence += self.confidence_calibration_rules['context_match_bonus']
        
        # Method-specific adjustments
        method_adjustments = {
            'two_stage_reranking': 0.05,    # Slight boost for advanced method
            'document_context': 0.1,        # Higher confidence for document-based
            'external_api': 0.15,           # Highest confidence for API data
            'hybrid': 0.0,                  # Baseline
            'fallback': -0.2                # Lower confidence for fallback
        }
        
        base_confidence += method_adjustments.get(method, 0.0)
        
        # 🛡️ CRITICAL: Ensure final confidence ≤ 1.0
        final_confidence = self.normalize_confidence(base_confidence, f"response_calculation_{method}")
        
        logger.debug(f"🧮 Confidence calculation: semantic={semantic_score:.3f}, "
                    f"keyword={keyword_score:.3f}, method={method} -> {final_confidence:.3f}")
        
        return final_confidence
    
    def get_response_strategy(self, confidence: float) -> str:
        confidence = self.normalize_confidence(confidence, "strategy_decision")
        
        if confidence >= self.decision_thresholds['direct_answer']:
            return 'direct_answer'
        elif confidence >= self.decision_thresholds['enhanced_answer']:
            return 'enhanced_answer'
        elif confidence >= self.decision_thresholds['ask_clarification']:
            return 'ask_clarification'
        else:
            return 'dont_know'

class SmartTokenManager:    
    def __init__(self):
        self.adaptive_token_range = {
            'min': 80, 
            'optimal': 250, 
            'max': 500,
            'expected_sentences': 3, 
            'avg_chars_per_sentence': 80
        }
        
        self.incomplete_patterns = [
            r'[^.!?]\s*$',  # Không kết thúc bằng dấu câu
            r'\b(và|hoặc|với|để|khi|nếu|tại|về|cho|trong|của|từ)\s*$',  # Kết thúc bằng từ nối
            r'\b(em|sẽ|có|được|phải|cần|nên)\s*$',  # Kết thúc bằng từ chưa hoàn chỉnh
            r'[,;:]\s*$',  # Kết thúc bằng dấu phẩy/chấm phẩy
            r'\b(Dạ|Ạ|thầy|cô|giảng viên)\s*$',  # Câu chào chưa hoàn chỉnh
        ]
        
        self.complete_endings = [
            r'[.!?]\s*$',  # Kết thúc bằng dấu câu
            r'ạ[.!?]\s*$',  # Kết thúc bằng "ạ" + dấu câu
            r'không ạ\?\s*$',  # "có cần hỗ trợ thêm gì không ạ?"
            r'🎓\s*$',  # Emoji kết thúc
            r'@bdu\.edu\.vn\s*$',  # Email ending
        ]
        
        logger.info("✅ SmartTokenManager initialized with adaptive token range")
    
    def calculate_optimal_tokens(self, prompt_length: int, complexity_hint: str = None) -> int:        
        # Base tokens from adaptive range
        base_tokens = self.adaptive_token_range['optimal']
        
        if prompt_length > 500:
            base_tokens += 50  # Prompt dài cần response dài hơn
        elif prompt_length < 200:
            base_tokens -= 30  # Prompt ngắn có thể response ngắn hơn
            
        if complexity_hint:
            if complexity_hint in ['enhanced_generation', 'detailed_explanation', 'document_context', 'two_stage_reranking']:
                base_tokens += 100  # 🚀 NEW: Advanced methods need more tokens
            elif complexity_hint in ['quick_clarify', 'simple_answer']:
                base_tokens -= 40
                
        min_tokens = self.adaptive_token_range['min']
        max_tokens = self.adaptive_token_range['max']
        
        return max(min_tokens, min(max_tokens, base_tokens))
    
    def is_response_incomplete(self, response: str) -> Dict[str, Any]:        
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
        # Estimate current length in tokens (rough: 1 token ≈ 3-4 chars in Vietnamese)
        current_tokens = len(incomplete_response) // 3
        
        # Target tokens for complete response
        target_tokens = self.adaptive_token_range['optimal']
        
        # Additional tokens needed
        additional_needed = max(20, target_tokens - current_tokens)
        
        return min(additional_needed, 150)  # Cap at 150 additional tokens

class ConversationMemory:    
    def __init__(self, max_history=30):
        self.conversations = {}  # {session_id: conversation_data}
        self.max_history = max_history
        try:
            self.entity_extractor = SimpleEntityExtractor()
            logger.info("✅ SimpleEntityExtractor initialized successfully in ConversationMemory")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SimpleEntityExtractor: {str(e)}")
            self.entity_extractor = None
        
    def add_interaction(self, session_id: str, user_query: str, bot_response: str, 
                       intent_info: dict = None, entities: dict = None):
        
        # 🆕 DEBUG LOG
        logger.info(f"🔍 DEBUG add_interaction: session={session_id}")
        logger.info(f"🔍 DEBUG query: '{user_query}'")
        logger.info(f"🔍 DEBUG response preview: '{bot_response[:100]}...'")
        
        if not hasattr(self, 'entity_extractor') or self.entity_extractor is None:
            logger.error("❌ CRITICAL: entity_extractor is None!")
            return  # STOP here if no extractor
        
        # Extract entities from Q&A pair
        qa_text = f"{user_query} {bot_response}"
        extracted_entities = self.entity_extractor.extract_entities(qa_text, user_query)
        
        logger.info(f"🔍 DEBUG extracted entities: {extracted_entities}")
        
        """Thêm interaction vào memory với entity extraction"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'history': [],
                'context_summary': "",
                'user_interests': set(),
                'conversation_type': 'lecturer',
                # ✅ THÊM MỚI: Entity memory và relationships
                'entity_memory': {},  # {entity: {type, related_entities, context, confidence}}
                'entity_relationships': [],  # List các mối quan hệ
                'context_keywords': []  # Keywords để enhance query
            }
        
        # Extract entities from Q&A pair
        qa_text = f"{user_query} {bot_response}"
        extracted_entities = self.entity_extractor.extract_entities(qa_text, user_query)
        
        # Build relationships
        relationships = self.entity_extractor.build_entity_relationships(
            user_query, bot_response, extracted_entities
        )
        
        # Update entity memory
        self._update_entity_memory(session_id, extracted_entities, relationships, user_query, bot_response)
        
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
            'entities': entities or {},
            # ✅ THÊM MỚI: Lưu extracted entities cho interaction này
            'extracted_entities': extracted_entities,
            'entity_relationships': relationships
        }
        
        self.conversations[session_id]['history'].append(interaction)
        
        # Keep only recent history
        if len(self.conversations[session_id]['history']) > self.max_history:
            self.conversations[session_id]['history'] = self.conversations[session_id]['history'][-self.max_history:]
        
        # Update context summary
        self._update_context_summary(session_id)
        
        # ✅ THÊM MỚI: Update context keywords for next queries
        self._update_context_keywords(session_id)
    
    def _update_entity_memory(self, session_id: str, extracted_entities: dict, relationships: list, query: str, response: str):
        """Cập nhật entity memory với thông tin mới"""
        conv = self.conversations[session_id]
        
        # Update entities
        for entity_type, entity_list in extracted_entities.items():
            for entity in entity_list:
                entity_key = entity.lower().strip()
                
                if entity_key not in conv['entity_memory']:
                    conv['entity_memory'][entity_key] = {
                        'original_form': entity,
                        'type': entity_type,
                        'contexts': [],
                        'related_entities': set(),
                        'confidence': 0.5,
                        'first_seen': time.time(),
                        'last_used': time.time()
                    }
                
                # Update context
                context_snippet = f"Q: {query[:100]}... A: {response[:100]}..."
                conv['entity_memory'][entity_key]['contexts'].append({
                    'snippet': context_snippet,
                    'timestamp': time.time(),
                    'query': query,
                    'response_preview': response[:200]
                })
                
                # Keep only recent contexts
                if len(conv['entity_memory'][entity_key]['contexts']) > 3:
                    conv['entity_memory'][entity_key]['contexts'] = conv['entity_memory'][entity_key]['contexts'][-3:]
                
                conv['entity_memory'][entity_key]['last_used'] = time.time()
        
        # Update relationships
        for rel in relationships:
            entity1_key = rel['entity1'].lower().strip()
            entity2_key = rel['entity2'].lower().strip()
            
            # Add bidirectional relationships
            if entity1_key in conv['entity_memory']:
                conv['entity_memory'][entity1_key]['related_entities'].add(entity2_key)
                conv['entity_memory'][entity1_key]['confidence'] = min(0.9, conv['entity_memory'][entity1_key]['confidence'] + 0.1)
            
            if entity2_key in conv['entity_memory']:
                conv['entity_memory'][entity2_key]['related_entities'].add(entity1_key)
                conv['entity_memory'][entity2_key]['confidence'] = min(0.9, conv['entity_memory'][entity2_key]['confidence'] + 0.1)
        
        # Add relationships to list (keep recent)
        conv['entity_relationships'].extend(relationships)
        if len(conv['entity_relationships']) > 20:
            conv['entity_relationships'] = conv['entity_relationships'][-20:]
    
    def _update_context_keywords(self, session_id: str):
        """Cập nhật context keywords từ entity memory"""
        conv = self.conversations[session_id]
        
        # Get high-confidence entities from recent interactions
        recent_entities = []
        current_time = time.time()
        
        for entity_key, entity_data in conv['entity_memory'].items():
            # Entities used in last 5 interactions (roughly)
            time_since_last_use = current_time - entity_data['last_used']
            if time_since_last_use < 300:  # 5 minutes
                if entity_data['confidence'] > 0.6:
                    recent_entities.append({
                        'entity': entity_data['original_form'],
                        'type': entity_data['type'],
                        'confidence': entity_data['confidence'],
                        'recency': time_since_last_use
                    })
        
        # Sort by confidence and recency
        recent_entities.sort(key=lambda x: (x['confidence'], -x['recency']), reverse=True)
        
        # Extract keywords
        context_keywords = []
        for entity_info in recent_entities[:5]:  # Top 5
            entity = entity_info['entity']
            if len(entity.strip()) > 2:
                context_keywords.append(entity)
        
        conv['context_keywords'] = context_keywords
        
        logger.debug(f"📝 Updated context keywords for session {session_id}: {context_keywords}")

    def get_conversation_context(self, session_id: str) -> dict:
        if session_id not in self.conversations:
            return {
                'history': [], 
                'context_summary': '', 
                'user_interests': [], 
                'recent_conversation_summary': '',
                # ✅ THÊM MỚI: Context enhancement data
                'context_keywords': [],
                'entity_memory': {},
                'active_entities': []
            }
        
        conv = self.conversations[session_id]
        # Tạo tóm tắt 2-3 tương tác gần nhất
        recent_summary = self._create_recent_conversation_summary(session_id)
        
        # ✅ THÊM MỚI: Get active entities (high confidence, recently used)
        active_entities = self._get_active_entities(session_id)
        
        return {
            'history': conv['history'][-25:],  # Last 25 interactions
            'context_summary': conv['context_summary'],
            'user_interests': list(conv['user_interests']),
            'conversation_type': conv['conversation_type'],
            'recent_conversation_summary': recent_summary,
            # ✅ THÊM MỚI: Enhanced context data
            'context_keywords': conv.get('context_keywords', []),
            'entity_memory': conv.get('entity_memory', {}),
            'active_entities': active_entities,
            'entity_relationships': conv.get('entity_relationships', [])[-10:]  # Recent 10 relationships
        }
    
    def _get_active_entities(self, session_id: str) -> list:
        """Lấy danh sách entities đang active (confidence cao, dùng gần đây)"""
        if session_id not in self.conversations:
            return []
        
        conv = self.conversations[session_id]
        active_entities = []
        current_time = time.time()
        
        for entity_key, entity_data in conv['entity_memory'].items():
            time_since_last_use = current_time - entity_data['last_used']
            
            # Entity active nếu: confidence > 0.6 VÀ dùng trong 10 phút gần đây
            if entity_data['confidence'] > 0.6 and time_since_last_use < 600:
                active_entities.append({
                    'entity': entity_data['original_form'],
                    'type': entity_data['type'], 
                    'confidence': entity_data['confidence'],
                    'related_entities': list(entity_data['related_entities'])[:3],  # Top 3 related
                    'last_context': entity_data['contexts'][-1]['snippet'] if entity_data['contexts'] else ""
                })
        
        # Sort by confidence
        active_entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return active_entities[:5]
    
    def get_context_for_query(self, session_id: str, current_query: str) -> dict:
        """🔧 IMPROVED: Enhanced context detection with fallback mechanisms"""
        
        logger.info(f"🔍 DEBUG get_context_for_query called: session={session_id}, query='{current_query}'")
        
        if session_id not in self.conversations:
            logger.info(f"🔍 DEBUG: No conversations found for session {session_id}")
            return {
                'context_keywords': [], 
                'related_entities': [], 
                'should_use_context': False,
                'context_strength': 0
            }
        
        conv = self.conversations[session_id]
        logger.info(f"🔍 DEBUG: Found conversation with {len(conv.get('entity_memory', {}))} entities")
        
        # 🔧 IMPROVED: Normalize query for better matching
        current_query_normalized = self._normalize_for_matching(current_query)
        logger.info(f"🔍 DEBUG: Normalized query: '{current_query_normalized}'")
        
        # 🔧 RELAXED: Consider last 3 interactions instead of 2
        recent_interactions = conv['history'][-3:] if len(conv['history']) >= 3 else conv['history']
        
        relevant_entities = []
        extracted_names = []
        
        # 🆕 STEP 1: Check for memory reference patterns (như "còn nhớ...")
        memory_reference_patterns = [
            r'\b(còn|vẫn)\s+(nhớ|biết)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)',
            r'\b(thế|vậy)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+là\s+(ai|gì)',
            r'\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+là\s+(ai|gì)',  # "X là ai"
            r'\bai\s+là\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)',  # "ai là X"
        ]
        
        # Extract names from memory reference patterns  
        for pattern in memory_reference_patterns:
            matches = re.findall(pattern, current_query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for group in match:
                        if group and len(group.strip()) > 2:
                            # Check if it looks like a name (has at least 1 uppercase char)
                            if any(c.isupper() for c in group):
                                extracted_names.append(group.strip())
                else:
                    if len(match.strip()) > 2 and any(c.isupper() for c in match):
                        extracted_names.append(match.strip())
        
        logger.info(f"🔍 DEBUG: Extracted names from patterns: {extracted_names}")
        
        # 🆕 STEP 2: Enhanced entity matching với memory references
        if extracted_names:
            logger.info(f"🔍 Memory/Direct reference detected: {extracted_names}")
            for extracted_name in extracted_names:
                for entity_key, entity_data in conv.get('entity_memory', {}).items():
                    original_form = entity_data.get('original_form', entity_key)
                    
                    # Use flexible name matching for memory references
                    if self._names_match_flexible(extracted_name, original_form):
                        relevant_entities.append({
                            'entity': original_form,
                            'type': entity_data.get('type', 'unknown'),
                            'related': list(entity_data.get('related_entities', set()))[:2],
                            'confidence': 0.8  # Higher confidence for direct references
                        })
                        logger.info(f"🎯 Memory reference matched: {extracted_name} → {original_form}")
        
        # 🆕 STEP 3: Standard entity matching (existing logic)
        for entity_key, entity_data in conv.get('entity_memory', {}).items():
            original_form = entity_data.get('original_form', entity_key)
            
            # Skip if already found via memory reference
            if any(ent['entity'] == original_form for ent in relevant_entities):
                continue
            
            # 🔧 IMPROVED: Use flexible matching
            is_relevant = self._is_entity_relevant_to_query_strict(
                current_query_normalized, 
                entity_key, 
                original_form
            )
            
            if is_relevant:
                relevant_entities.append({
                    'entity': original_form,
                    'type': entity_data.get('type', 'unknown'),
                    'related': list(entity_data.get('related_entities', set()))[:2],
                    'confidence': entity_data.get('confidence', 0.5)
                })
                logger.info(f"🎯 Found relevant entity: {original_form} (key: {entity_key})")
        
        # 🆕 STEP 4: Fallback cho person name queries không tìm thấy context
        if not relevant_entities and extracted_names:
            logger.info(f"🔍 No entity memory found, creating fallback context for: {extracted_names}")
            for name in extracted_names:
                if len(name.split()) >= 2:  # Valid name structure
                    relevant_entities.append({
                        'entity': name,
                        'type': 'person_name', 
                        'related': [],
                        'confidence': 0.6  # Medium confidence for fallback
                    })
                    logger.info(f"🎯 Fallback entity created: {name}")
        
        # 🔧 STEP 5: Decision logic với relaxed thresholds
        context_strength = len(relevant_entities)
        
        # 🔧 MUCH MORE RELAXED: Lower confidence threshold
        should_use_context = (
            len(relevant_entities) > 0 and 
            any(entity['confidence'] > 0.3 for entity in relevant_entities)  # Lowered from 0.5
        )
        
        # 🆕 STEP 6: Force context cho entity query patterns
        entity_query_indicators = [
            'là ai', 'ai là', 'còn nhớ', 'vậy ', 'thế ', 
            'ông ', 'bà ', 'thầy ', 'cô ', 'anh ', 'chị '
        ]
        
        if not should_use_context:
            has_entity_pattern = any(indicator in current_query.lower() for indicator in entity_query_indicators)
            if has_entity_pattern and (relevant_entities or extracted_names):
                should_use_context = True
                logger.info(f"🎯 Force context enabled for entity query pattern")
        
        # 🆕 STEP 7: Generate context keywords
        context_keywords = []
        if should_use_context:
            # Prioritize extracted names first
            for name in extracted_names[:2]:  # Max 2 from direct extraction
                if name not in context_keywords:
                    context_keywords.append(name)
            
            # Then add from relevant entities
            for entity_info in relevant_entities[:3]:  # Max 3 total
                if len(context_keywords) < 3 and entity_info['entity'] not in context_keywords:
                    context_keywords.append(entity_info['entity'])
        
        # 🆕 STEP 8: Enhanced logging
        logger.info(f"🔍 DEBUG: should_use_context={should_use_context}, relevant_entities={len(relevant_entities)}")
        logger.info(f"🔍 DEBUG: context_keywords={context_keywords}")
        logger.info(f"🔍 DEBUG: context_strength={context_strength}")
        
        # Calculate final confidence
        final_confidence = max([e['confidence'] for e in relevant_entities], default=0.0)
        if extracted_names and not relevant_entities:
            final_confidence = 0.6  # Fallback confidence
        
        return {
            'context_keywords': context_keywords,
            'related_entities': relevant_entities,
            'should_use_context': should_use_context,
            'context_strength': context_strength,
            'context_confidence': final_confidence,
            'extracted_names': extracted_names,  # For debugging
            'memory_reference_detected': bool(extracted_names),
            'fallback_used': not relevant_entities and bool(extracted_names)
        }
    
    def _names_match_flexible(self, name1: str, name2: str) -> bool:
        """🆕 NEW: Flexible name matching"""
        if not name1 or not name2:
            return False
        
        # Normalize both names
        norm1 = self._normalize_for_matching(name1.lower())
        norm2 = self._normalize_for_matching(name2.lower())
        
        # Direct match
        if norm1 == norm2:
            return True
        
        # Word-level matching
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # If both have multiple words, check overlap
        if len(words1) >= 2 and len(words2) >= 2:
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            # 🔧 FLEXIBLE: 50% overlap is good enough for names
            return overlap_ratio >= 0.5
        
        # Single word matching
        if len(words1) == 1 and len(words2) == 1:
            # Allow partial matching for single words if they're long enough
            word1, word2 = list(words1)[0], list(words2)[0]
            if len(word1) >= 3 and len(word2) >= 3:
                return word1 in word2 or word2 in word1
        
        return False
    
    def _is_entity_relevant_to_query_strict(self, normalized_query, entity_key, original_form):
        """🔧 IMPROVED: Flexible entity matching để tránh miss các variations"""
        
        # Strategy 1: Direct exact match (case insensitive)
        entity_key_normalized = self._normalize_for_matching(entity_key)
        original_form_normalized = self._normalize_for_matching(original_form)
        
        if entity_key_normalized in normalized_query or original_form_normalized in normalized_query:
            logger.debug(f"🎯 Exact match found for '{original_form}'")
            return True
        
        # Strategy 2: Word-by-word matching for multi-word entities (RELAXED)
        entity_words = set(original_form_normalized.split())
        query_words = set(normalized_query.split())
        
        if len(entity_words) >= 2:
            overlap = len(entity_words.intersection(query_words))
            overlap_ratio = overlap / len(entity_words)
            
            # 🔧 RELAXED: 60% overlap instead of 80%
            if overlap_ratio >= 0.6:  
                logger.debug(f"🎯 Name parts match: {overlap}/{len(entity_words)} = {overlap_ratio:.2f}")
                return True
        
        # Strategy 3: Name parts matching (họ + tên)
        if len(entity_words) >= 2:
            first_name = list(entity_words)[0]  # Họ
            last_name = list(entity_words)[-1]  # Tên
            
            # Check if both first and last name appear
            if first_name in query_words and last_name in query_words:
                logger.debug(f"🎯 First + Last name match: '{first_name}' + '{last_name}'")
                return True
            
            # Check if last name + middle name appear
            if len(entity_words) >= 3:
                middle_name = list(entity_words)[1]
                if last_name in query_words and middle_name in query_words:
                    logger.debug(f"🎯 Middle + Last name match: '{middle_name}' + '{last_name}'")
                    return True
        
        # Strategy 4: Partial name with title matching
        titles = ['gs.ts', 'ts', 'gs', 'thầy', 'cô', 'giáo sư', 'tiến sĩ', 'ông', 'bà']
        query_has_title = any(title in normalized_query for title in titles)
        
        if query_has_title and len(entity_words) >= 2:
            # If query has title, just need last name match
            last_name = list(entity_words)[-1]
            if last_name in query_words and len(last_name) > 2:
                logger.debug(f"🎯 Title + Last name match: '{last_name}'")
                return True
        
        return False
    
    def _normalize_for_matching(self, text):
        """🚀 FIX: Normalize text for better entity matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove Vietnamese particles and common words
        normalized = re.sub(r'\b(dạ|ạ|à|ơi|nhé|vậy|thì|là|ai|gì|như|thế|nào)\b', ' ', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _is_entity_relevant_to_query(self, normalized_query, entity_key, original_form):
        """🚀 FIX: Multiple strategies to check if entity is relevant"""
        
        # Strategy 1: Direct substring match with normalized query
        entity_key_normalized = self._normalize_for_matching(entity_key)
        original_form_normalized = self._normalize_for_matching(original_form)
        
        if entity_key_normalized in normalized_query or entity_key in normalized_query.lower():
            logger.debug(f"📝 Match strategy 1: entity_key '{entity_key_normalized}' in query")
            return True
            
        if original_form_normalized in normalized_query or original_form.lower() in normalized_query.lower():
            logger.debug(f"📝 Match strategy 2: original_form '{original_form_normalized}' in query")
            return True
        
        # Strategy 2: Word-by-word matching (for multi-word entities)
        entity_words = set(original_form_normalized.split())
        query_words = set(normalized_query.split())
        
        # If entity has multiple words, check if most words are in query
        if len(entity_words) > 1:
            overlap = len(entity_words.intersection(query_words))
            overlap_ratio = overlap / len(entity_words)
            
            if overlap_ratio >= 0.7:  # 70% of entity words must be in query
                logger.debug(f"📝 Match strategy 3: word overlap {overlap}/{len(entity_words)} = {overlap_ratio:.2f}")
                return True
        
        # Strategy 3: Partial name matching (for person names)
        if len(entity_words) >= 2:
            # Check if last name (assumed to be last word) is in query
            last_word = list(entity_words)[-1]
            if len(last_word) > 2 and last_word in query_words:
                logger.debug(f"📝 Match strategy 4: last name '{last_word}' found")
                return True
        
        return False
    
    def _create_recent_conversation_summary(self, session_id: str) -> str:
        if session_id not in self.conversations:
            return ""
        
        history = self.conversations[session_id]['history']
        if len(history) < 2:
            return ""
        
        # Lấy tương tác gần nhất
        recent_interactions = history[-20:]
        
        summary_parts = []
        for interaction in recent_interactions:
            user_query = interaction['user_query'][:100]  # Tối đa 100 ký tự
            bot_response = interaction['bot_response'][:150]  # Tối đa 150 ký tự
            
            summary_parts.append(f"Hỏi: {user_query}... → Trả lời: {bot_response}...")
        
        return " | ".join(summary_parts)
    
    def _update_context_summary(self, session_id: str):
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
    def __init__(self, key_manager: GeminiApiKeyManager):
        self.key_manager = key_manager
        self.model_name = "gemini-2.0-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.cache = {}
        self.max_cache_size = 500
        logger.info("✅ SimpleVietnameseRestorer initialized with Key Manager.")
    
    def has_vietnamese_accents(self, text: str) -> bool:
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        vietnamese_chars += vietnamese_chars.upper()
        return any(char in vietnamese_chars for char in text)
    
    def restore_vietnamese_tone(self, input_text: str, retry_count=0) -> str:
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
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100, "topP": 0.8}
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
        self.cache[key] = result
        
        # Simple cache management
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.cache) // 5
            keys_to_remove = list(self.cache.keys())[:items_to_remove]
            for k in keys_to_remove:
                del self.cache[k]

class SimpleEntityExtractor:
    """Trích xuất thực thể đơn giản từ Q&A để build context memory"""
    
    def __init__(self):
        # 🔧 CẢI TIẾN: Patterns chặt chẽ hơn cho các loại entity
        self.entity_patterns = {
            'person_name': [
                # 🔧 IMPROVED: Tên riêng người Việt (2-4 từ, viết hoa đầu từ) - CHẶT CHẼ HỚN
                r'\b([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+)\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+)(?:\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+))?(?:\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+))?\b',
                # Pattern với tiến sĩ, giáo sư
                r'(?:GS\.TS\.|TS\.|GS\.|tiến sĩ|giáo sư)\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+(?:\s+[A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+){1,2})'
            ],
            'position': [
                r'\b(hiệu trưởng|phó hiệu trưởng|trưởng phòng|phó trưởng phòng|trưởng khoa|phó trưởng khoa|giáo sư|phó giáo sư|tiến sĩ|thạc sĩ)\b',
                r'\b(chủ tịch|phó chủ tịch|ủy viên|thành viên|trưởng ban|phó ban)\b'
            ],
            'department': [
                r'(khoa [^.!?]*|phòng [^.!?]*|ban [^.!?]*|bộ môn [^.!?]*)',
                r'(đại học bình dương|bdu|trường đại học)'
            ],
            'numbers': [
                r'(\d+(?:[.,]\d+)*(?:\s*(?:triệu|nghìn|tỷ|đồng|vnđ|usd|phần trăm|%))?)',
                r'(\d+(?:\.\d+)?(?:\s*tín chỉ)?)'
            ],
            'dates': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(ngày \d{1,2}|tháng \d{1,2}|năm \d{4})',
                r'(học kỳ \d+|năm học \d{4}-\d{4})'
            ]
        }
        
        # 🆕 THÊM MỚI: Blacklist để loại bỏ false positives
        self.person_name_blacklist = {
            # Các cụm từ thường bị nhận nhầm là tên người
            'hoc phi chinh', 'quy khanh', 'duc tin', 'duc hanh', 'duc duc', 'nam duc',
            'hoc phi', 'chi phi', 'muc phi', 'le phi', 'phi le', 'thu phi',
            'quy dinh', 'quy che', 'quy trinh', 'quy tac', 'quy luat',
            'duc tinh', 'duc tich', 'duc hanh vi', 'duc han che',
            'nam hoc', 'nam tu', 'nam toi', 'nam sau', 'nam truoc',
            'tin chi', 'tin tin', 'chi tiet', 'chi tieu', 'chi phi',
            'bao cao', 'cao cap', 'cao dang', 'cap hoc', 'cap do',
            'sinh vien', 'giang vien', 'can bo', 'hoc sinh', 'nghien cuu sinh',
            'dai hoc', 'cao hoc', 'tien si', 'thac si', 'cu nhan',
            'mon hoc', 'bai hoc', 'gio hoc', 'lop hoc', 'hoc tap',
            # Thêm các từ khóa BDU thường gặp
            'binh duong', 'bdu', 'truong dai hoc', 'phong ban', 'khoa hoc',
            'nghien cuu', 'dao tao', 'quan ly', 'hanh chinh', 'ky thuat',
            'cong nghe', 'kinh te', 'ngoai ngu', 'su pham', 'y khoa'
        }
        
        # 🆕 THÊM MỚI: Common Vietnamese words that are not names
        self.common_words_blacklist = {
            'co the', 'co ban', 'co so', 'co hoi', 'co quan', 'co mat',
            'la mot', 'la cach', 'la gi', 'la ai', 'la khi', 'la lieu',
            'duoc su', 'duoc cap', 'duoc phep', 'duoc biet', 'duoc tang',
            'hay la', 'hay khong', 'hay nhat', 'hay gi', 'hay co',
            'neu co', 'neu khong', 'neu la', 'neu can', 'neu muon'
        }
        
        logger.info("✅ IMPROVED SimpleEntityExtractor initialized with enhanced patterns and blacklists")

    def extract_entities(self, text, query_context=""):
        """🔧 IMPROVED: Trích xuất entities với filtering tốt hơn"""
        if not text:
            return {}
            
        entities = {}
        text_cleaned = text.strip()
        
        # 🔧 IMPROVED: Clean text - remove common phrases but preserve names
        text_cleaned = re.sub(r'\b(dạ|ạ|thưa|xin chào|chào|em|anh|chị|cảm ơn)\b', ' ', text_cleaned, flags=re.IGNORECASE)
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
        
        # Extract theo từng loại pattern
        for entity_type, patterns in self.entity_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_cleaned, re.IGNORECASE)
                for match in matches:
                    if entity_type == 'person_name':
                        # 🔧 SPECIAL HANDLING: Person names need more careful extraction
                        entity_value = self._extract_person_name_from_match(match)
                    else:
                        entity_value = match.group(1) if match.groups() else match.group(0)
                    
                    entity_value = entity_value.strip()
                    
                    # 🔧 IMPROVED: Strict filtering với blacklist
                    if self._is_valid_entity(entity_value, entity_type):
                        # Normalize entity
                        if entity_type == 'person_name':
                            entity_value = self._normalize_person_name(entity_value)
                        else:
                            entity_value = entity_value.lower()
                            
                        if entity_value not in entities[entity_type]:
                            entities[entity_type].append(entity_value)
        
        # Chỉ giữ lại entities có ít nhất 1 item
        entities = {k: v for k, v in entities.items() if v}
        
        logger.debug(f"🔍 Entity extraction result: {entities}")
        return entities

    def _extract_person_name_from_match(self, match):
        """🆕 THÊM MỚI: Trích xuất tên người từ regex match"""
        if match.groups():
            # Combine all non-empty groups
            name_parts = [group for group in match.groups() if group and group.strip()]
            return ' '.join(name_parts)
        else:
            return match.group(0)

    def _is_valid_entity(self, entity_value, entity_type):
        """🔧 IMPROVED: Validate entity quality với blacklist mở rộng"""
        if not entity_value or len(entity_value.strip()) < 3:
            return False
            
        entity_lower = entity_value.lower().strip()
        
        # 🆕 CHECK BLACKLIST FIRST
        if entity_type == 'person_name':
            # Check person name blacklist
            if entity_lower in self.person_name_blacklist:
                logger.debug(f"🚫 Rejected by person blacklist: '{entity_value}'")
                return False
            
            # Check common words blacklist
            if entity_lower in self.common_words_blacklist:
                logger.debug(f"🚫 Rejected by common words blacklist: '{entity_value}'")
                return False
            
            # Check for parts in blacklist
            entity_words = entity_lower.split()
            for word in entity_words:
                if word in self.person_name_blacklist or word in self.common_words_blacklist:
                    logger.debug(f"🚫 Rejected by word-level blacklist: '{entity_value}' (word: '{word}')")
                    return False
        
        # Remove noise patterns
        noise_patterns = [
            r'\b(có|cần|thể|thêm|gì|không|hỗ|trợ|để|em|là|ai|nói|rõ|hơn|về|vấn|đề|chính|xác|nhất)\b',
            r'^(và|hoặc|với|để|khi|nếu|tại|về|cho|trong|của|từ)',
            r'(ạ|à|ơi|nhé)$'
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, entity_lower):
                logger.debug(f"🚫 Rejected by noise pattern: '{entity_value}' (pattern: {pattern})")
                return False
        
        # Specific validation by type
        if entity_type == 'person_name':
            # 🔧 STRICTER: Must have 2-4 words, each word >= 2 chars
            words = entity_lower.split()
            if len(words) < 2 or len(words) > 4:
                logger.debug(f"🚫 Rejected by word count: '{entity_value}' ({len(words)} words)")
                return False
            
            if any(len(word) < 2 for word in words):
                logger.debug(f"🚫 Rejected by word length: '{entity_value}'")
                return False
            
            # 🆕 NEW: Check for Vietnamese name patterns
            if not self._looks_like_vietnamese_name(entity_value):
                logger.debug(f"🚫 Rejected by Vietnamese name pattern: '{entity_value}'")
                return False
            
            # Must not contain common Vietnamese particles
            if any(word in ['cô', 'thầy', 'anh', 'chị', 'em', 'dạ', 'được', 'phải', 'theo', 'như', 'từ'] for word in words):
                logger.debug(f"🚫 Rejected by particle words: '{entity_value}'")
                return False
                
        elif entity_type == 'position':
            # Must be exactly one of the position words
            valid_positions = ['hiệu trưởng', 'phó hiệu trưởng', 'trưởng phòng', 'phó trưởng phòng', 'trưởng khoa', 'phó trưởng khoa', 'giáo sư', 'phó giáo sư', 'tiến sĩ', 'thạc sĩ', 'chủ tịch', 'phó chủ tịch']
            if entity_lower not in valid_positions:
                logger.debug(f"🚫 Rejected invalid position: '{entity_value}'")
                return False
        
        logger.debug(f"✅ Valid entity: '{entity_value}' (type: {entity_type})")
        return True
    
    def _looks_like_vietnamese_name(self, name):
        """🆕 THÊM MỚI: Kiểm tra xem có giống tên người Việt không"""
        name_lower = name.lower()
        words = name_lower.split()
        
        # Vietnamese surname patterns (họ phổ biến)
        common_surnames = {
            'nguyễn', 'trần', 'lê', 'phạm', 'hoàng', 'huỳnh', 'phan', 'vũ', 'võ', 'đặng', 
            'bùi', 'đỗ', 'hồ', 'ngô', 'dương', 'lý', 'cao', 'đậu', 'lưu', 'tô',
            'nguyen', 'tran', 'le', 'pham', 'hoang', 'huynh', 'phan', 'vu', 'vo', 'dang',
            'bui', 'do', 'ho', 'ngo', 'duong', 'ly', 'cao', 'dau', 'luu', 'to'
        }
        
        # Check if first word (surname) is common Vietnamese surname
        if words[0] in common_surnames:
            return True
        
        # Vietnamese name characteristics
        # - Usually has balanced syllable structure
        # - Contains Vietnamese-specific characters or patterns
        vietnamese_chars = 'ăâêôơưàáạảãằắặẳẵầấậẩẫềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹđ'
        
        # Count Vietnamese-specific characters
        vietnamese_char_count = sum(1 for char in name_lower if char in vietnamese_chars)
        
        # If has Vietnamese chars and reasonable length, likely a Vietnamese name
        if vietnamese_char_count >= 1 and len(words) >= 2:
            return True
        
        # Pattern check: avoid common false positives
        false_positive_patterns = [
            r'phi|phí|fee',  # Avoid fee-related terms
            r'quy|qúy',      # Avoid regulation-related terms
            r'học|hoc',      # Avoid study-related terms
            r'chí|chi',      # Avoid will/credit-related terms
        ]
        
        for pattern in false_positive_patterns:
            if re.search(pattern, name_lower):
                # Double check: if it really contains Vietnamese name elements, still accept
                if vietnamese_char_count >= 2:  # Higher threshold for suspicious cases
                    continue
                else:
                    return False
        
        # Default: accept if passes basic structure checks
        return True
    
    def build_entity_relationships(self, query, answer, entities):
        """Xây dựng mối quan hệ giữa entities dựa vào ngữ cảnh Q&A"""
        relationships = []
        
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Tìm mối quan hệ person - position
        if 'person_name' in entities and 'position' in entities:
            for person in entities['person_name']:
                for position in entities['position']:
                    # Kiểm tra trong query
                    if any(keyword in query_lower for keyword in ['là ai', 'ai là', 'chức vụ']):
                        relationships.append({
                            'type': 'person_position',
                            'entity1': person,
                            'entity2': position,
                            'relation': 'has_position',
                            'confidence': 0.8,
                            'source': 'query_answer_pair'
                        })
        
        # Tìm mối quan hệ person - department
        if 'person_name' in entities and 'department' in entities:
            for person in entities['person_name']:
                for dept in entities['department']:
                    relationships.append({
                        'type': 'person_department', 
                        'entity1': person,
                        'entity2': dept,
                        'relation': 'works_at',
                        'confidence': 0.7,
                        'source': 'context'
                    })
        
        return relationships

    def _normalize_person_name(self, name):
        """🔧 IMPROVED: Normalize person name to consistent format"""
        # Title case each word
        words = name.lower().split()
        normalized_words = []
        for word in words:
            if len(word) > 0:
                normalized_words.append(word[0].upper() + word[1:])
        return ' '.join(normalized_words)

    def get_context_keywords(self, entities, relationships):
        """🚀 FIX: Generate better context keywords"""
        context_keywords = []
        
        # From entities - prioritize person names and positions
        if 'person_name' in entities:
            for entity in entities['person_name'][:2]:  # Max 2 person names
                context_keywords.append(entity)
                
        if 'position' in entities:
            for entity in entities['position'][:1]:  # Max 1 position
                context_keywords.append(entity)
        
        # From relationships
        for rel in relationships[:3]:  # Max 3 relationships
            if rel['confidence'] > 0.7:
                context_keywords.extend([rel['entity1'], rel['entity2']])
        
        # 🚀 FIX: Deduplicate and limit
        context_keywords = list(set(context_keywords))
        context_keywords = [kw for kw in context_keywords if len(kw.strip()) > 2]
        
        return context_keywords[:3]
    
class GeminiResponseGenerator:    
    def __init__(self):
        self.key_manager = GeminiApiKeyManager()
        # Thống nhất phiên bản model ở đây
        self.model_name = "gemini-2.0-flash" 
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.memory = ConversationMemory(max_history=30)
        self.vietnamese_restorer = SimpleVietnameseRestorer(self.key_manager)
        
        self.token_manager = SmartTokenManager()
        self.confidence_manager = AdvancedConfidenceManager()
        self._user_context_cache = {}
        
        self.default_generation_config = {
            "temperature": 0.4,
            "topP": 0.85
        }
        
        self.role_consistency_rules = {
            'identity': 'AI assistant của Đại học Bình Dương (BDU) hỗ trợ giảng viên',
            'personality': 'lịch sự, chuyên nghiệp, tôn trọng',
            'knowledge_scope': 'chuyên về thông tin BDU và hỗ trợ giảng viên',
            'addressing': 'luôn xưng hô đúng cách, không bao giờ dùng bạn/mình',
            'prohibited_roles': [
                'sinh viên', 'học sinh', 'phụ huynh', 'người ngoài trường'
            ]
        }
        
        logger.info("✅ Enhanced Gemini Response Generator initialized with Advanced Confidence Management, Smart Token Management, và Two-Stage Re-ranking Integration")
        # 🆕 DEBUG: Kiểm tra entity extractor được khởi tạo chưa
        if hasattr(self.memory, 'entity_extractor'):
            logger.info("✅ Entity Extractor available in ConversationMemory")
            
            # Test entity extraction
            test_entities = self.memory.entity_extractor.extract_entities(
                "Hiệu trưởng là Cao Việt Hiếu", 
                "hiệu trưởng là ai"
            )
            logger.info(f"🧪 Test entity extraction: {test_entities}")
            
        else:
            logger.error("❌ CRITICAL: Entity Extractor NOT found in ConversationMemory")
            logger.error("❌ This will cause context-aware functionality to fail")
            
        # 🆕 DEBUG: Kiểm tra get_context_for_query method
        if hasattr(self.memory, 'get_context_for_query'):
            logger.info("✅ get_context_for_query method available")
            
            # Test với session giả
            
            test_context = self.memory.get_context_for_query("test", "vậy Cao Việt Hiếu là ai?")
            logger.info(f"🧪 Test get_context_for_query: {test_context}")
            
        else:
            logger.error("❌ CRITICAL: get_context_for_query method NOT found")
            logger.error("❌ This will cause context analysis to always return should_use=False")

    # 🚀 NEW: Document Context Processing Methods
    def _build_document_context_prompt(self, query: str, document_text: str, session_id: str = None) -> str:
        system_prompt = self._get_personalized_system_prompt(session_id)
        personal_address = self._get_personal_address(session_id)
        
        # Get conversation context if available
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        # Build context section
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Hãy tham khảo ngữ cảnh trên để tạo câu trả lời mạch lạc, tránh lặp lại thông tin đã thảo luận.
"""
        
        # Truncate document text if too long (keep within token limits)
        max_doc_length = 3000  # characters
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "\n\n[...tài liệu còn tiếp...]"
        
        # ⭐ NHIỆM VỤ 2: Thêm khối chỉ dẫn đặc biệt cho việc xử lý dữ liệu OCR "bẩn"
        ocr_guidance = """---
⭐ HƯỚNG DẪN XỬ LÝ DỮ LIỆU OCR ĐẶC BIỆT (Rất quan trọng)
Dữ liệu dưới đây được trích xuất tự động từ file PDF/DOCX, do đó có thể chứa các lỗi định dạng, đặc biệt là các bảng (table) bị chuyển thành văn bản thuần túy.
1.  **Xử lý bảng (Table):** Một dòng văn bản có thể chứa nhiều thông tin liên quan (ví dụ: số thứ tự, họ tên, chức vụ, nhiệm vụ). BẠN PHẢI TỰ SUY LUẬN để liên kết các thông tin có vẻ nằm trên cùng một hàng với nhau. Ví dụ: dòng "1 Bà A Chức vụ B Nhiệm vụ C" có nghĩa là Bà A có chức vụ B và nhiệm vụ C.
2.  **Đếm số lượng:** Nếu được hỏi "có mấy điều", "có bao nhiêu thành viên", hãy tìm và đếm số lần xuất hiện của các từ khóa như "Điều 1.", "Điều 2.", hoặc các số thứ tự trong danh sách (1, 2, 3...).
3.  **Tìm kiếm chính xác:** Hãy đọc thật kỹ và tìm kiếm chính xác các từ khóa trong câu hỏi của người dùng trong toàn bộ văn bản, ngay cả khi nó không có cấu trúc.
---"""

        prompt = f"""{system_prompt}

🎯 NHIỆM VỤ ĐẶC BIỆT: Trả lời câu hỏi dựa trên nội dung tài liệu được cung cấp

{ocr_guidance}

📄 NỘI DUNG TÀI LIỆU:
{document_text}

{context_section}

❓ CÂU HỎI CỦA GIẢNG VIÊN: {query}

📝 YÊU CẦU TRẢ LỜI QUAN TRỌNG:
- Xưng hô: "Dạ {personal_address},"
- CHỈ TRẢ LỜI DỰA VÀO nội dung tài liệu được cung cấp ở trên
- KHÔNG SỬ DỤNG kiến thức bên ngoài tài liệu
- Nếu tài liệu không chứa thông tin để trả lời câu hỏi, hãy nói rõ điều đó
- Trích dẫn cụ thể từ tài liệu khi có thể
- Tạo câu trả lời rõ ràng, dễ hiểu và mạch lạc
- Kết thúc: "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?"
- TUYỆT ĐỐI KHÔNG bịa đặt thông tin không có trong tài liệu

Trả lời:"""

        return prompt

    def _generate_external_api_response(self, query, context, session_id=None):        
        api_data = context.get('api_data', {})
        lecturer_info = api_data.get('lecturer_info', {})
        schedule_summary = api_data.get('schedule_summary', {})
        daily_schedule = api_data.get('daily_schedule', {})
        
        personal_address = self._get_personal_address_from_api_data(lecturer_info, session_id)
        
        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        prompt = self._build_external_api_prompt(
            query, api_data, personal_address, recent_summary
        )
        
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
- Kết thúc: "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?"
- KHÔNG CHẾ TẠO thông tin không có trong dữ liệu

Trả lời:"""
        return prompt
    
    def _format_schedule_for_prompt(self, daily_schedule):
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
- Kết thúc: "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?"
- SỬ DỤNG CHÍNH XÁC thông tin từ hệ thống - KHÔNG CHẾ TẠO
- Trình bày thông tin cá nhân một cách tự nhiên, dễ hiểu
- KHÔNG dùng format phức tạp với **1. **2. hay bullets khi không cần thiết"""

        return base_prompt

    def _get_personal_address_from_api_data(self, lecturer_info, session_id):
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
                return f"{salutation} {name_suffix}"
            else:
                return f"{salutation} {ten_giang_vien}"
        
        # Fallback to session-based addressing
        return self._get_personal_address(session_id)

    def _post_process_external_api_response(self, response, lecturer_info, query, session_id):
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

    def set_user_context(self, session_id: str, user_context: dict):
        
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
        try:
            personal_address = self._get_personal_address(session_id)
            
            user_context = self._user_context_cache.get(session_id, {})
            user_memory_prompt = user_context.get('preferences', {}).get('user_memory_prompt', '')
            
            return build_personalized_system_prompt(user_memory_prompt, personal_address)
        
        except Exception as e:
            logger.error(f"Error getting personalized prompt: {e}")
            return build_personalized_system_prompt()  # Fallback

    def generate_response(self, query: str, context: Optional[Dict] = None, 
                      intent_info: Optional[Dict] = None, entities: Optional[Dict] = None,
                      session_id: str = None) -> Dict[str, Any]:
        start_time = time.time()
        
        print(f"\n--- 🚀 ADVANCED RAG GENERATION REQUEST (Session: {session_id}) ---")
        print(f"🧠 MEMORY DEBUG: Total active sessions = {len(self.memory.conversations)}")

        try:
            original_query = query
            if not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Query restored: '{query}' -> '{restored_query}'")
                    query = restored_query

            instruction = context.get('instruction', '') if context else ''
            
            # 🚀 NEW: Handle document context processing
            if instruction == 'answer_from_document':
                logger.info("📄 DOCUMENT CONTEXT: Processing document-based query")
                
                document_text = context.get('document_text', '')
                if not document_text or not document_text.strip():
                    logger.warning("⚠️ Empty document text provided")
                    personal_address = self._get_personal_address(session_id)
                    
                    # 🛡️ CONFIDENCE CAP: Document error response
                    response_confidence = self.confidence_manager.normalize_confidence(0.1, "document_error")
                    
                    return {
                        'response': f"Dạ {personal_address}, em không nhận được nội dung tài liệu để trả lời câu hỏi. {personal_address.title()} có thể gửi lại tài liệu không ạ? 🎓",
                        'method': 'document_context_empty',
                        'strategy': 'document_error',
                        'confidence': response_confidence,  # 🛡️ CAPPED
                        'generation_time': time.time() - start_time,
                        'original_query': original_query,
                        'restored_query': query,
                        'vietnamese_restoration_used': query != original_query,
                        'personalized': bool(session_id in self._user_context_cache),
                        'document_context_processed': True,
                        'token_info': {'smart_tokens_used': False, 'method': 'document_error'}
                    }
                
                # Build document context prompt
                prompt = self._build_document_context_prompt(query, document_text, session_id)
                
                # Calculate optimal tokens for document processing
                optimal_tokens = self.token_manager.calculate_optimal_tokens(
                    len(prompt), 
                    'document_context'
                )
                
                logger.info(f"📄 Processing document context with {optimal_tokens} tokens")
                
                # Call Gemini API for document processing
                response = self._call_gemini_api_with_smart_tokens(
                    prompt, 'document_context', optimal_tokens, session_id
                )
                
                if not response:
                    personal_address = self._get_personal_address(session_id)
                    response = f"Dạ {personal_address}, em gặp khó khăn kỹ thuật khi phân tích tài liệu. {personal_address.title()} có thể thử lại hoặc đặt câu hỏi cụ thể hơn không ạ? 🎓"
                
                # 🛡️ CONFIDENCE CAP: Document processing response
                response_confidence = self.confidence_manager.calculate_response_confidence(
                    semantic_score=0.85,  # High for document-based
                    keyword_score=0.0,
                    context_bonus=0.1,
                    method='document_context'
                )
                
                # Save to memory
                if session_id:
                    self.memory.add_interaction(session_id, original_query, response, intent_info, entities)

                return {
                    'response': response,
                    'method': 'document_context_processing',
                    'strategy': 'document_context',
                    'confidence': response_confidence,  # 🛡️ CAPPED
                    'generation_time': time.time() - start_time,
                    'original_query': original_query,
                    'restored_query': query,
                    'vietnamese_restoration_used': query != original_query,
                    'personalized': bool(session_id in self._user_context_cache),
                    'document_context_processed': True,
                    'token_info': {
                        'smart_tokens_used': True,
                        'method': 'document_context_processing',
                        'optimal_tokens': optimal_tokens
                    }
                }
            
            if instruction == 'process_external_api_data':
                # Process external API data
                response = self._generate_external_api_response(query, context, session_id)
                
                # 🛡️ CONFIDENCE CAP: External API response
                response_confidence = self.confidence_manager.calculate_response_confidence(
                    semantic_score=0.9,   # Very high for API data
                    keyword_score=0.0,
                    context_bonus=0.15,   # High context bonus for personal data
                    method='external_api'
                )
                
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
                    'confidence': response_confidence,  # 🛡️ CAPPED
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
            
            # 🛡️ CONFIDENCE CAP: Extract and normalize confidence from context
            raw_confidence = context.get('confidence', 0.5) if context else 0.5
            normalized_confidence = self.confidence_manager.normalize_confidence(raw_confidence, "input_context")
            
            # Update context with normalized confidence
            if context:
                context['confidence'] = normalized_confidence
            
            instruction = context.get('instruction', '') if context else ''
            
            if instruction == 'direct_answer_lecturer':
                response, token_info = self._generate_direct_lecturer_answer_smart(query, context, session_id)
                # Confidence từ context đã được normalized
                final_confidence = normalized_confidence
            elif instruction in ['enhance_answer_lecturer', 'enhance_answer_lecturer_boosted']:
                response, token_info = self._generate_enhanced_lecturer_answer_smart(query, context, intent_info, entities, session_id)
                # Slight boost for enhanced methods, but still capped
                final_confidence = self.confidence_manager.normalize_confidence(normalized_confidence + 0.05, "enhanced_method")
            elif instruction == 'clarification_needed':
                response, token_info = self._generate_clarification_request_smart(query, context, session_id)
                final_confidence = self.confidence_manager.normalize_confidence(0.3, "clarification")
            elif instruction == 'dont_know_lecturer':
                response, token_info = self._generate_dont_know_response_smart(query, context, session_id)
                final_confidence = self.confidence_manager.normalize_confidence(0.1, "dont_know")
            else:
                # Check out of scope and generate response
                if context and context.get('emergency_education', False):
                    print(f"🚨 GEMINI: Emergency education mode activated")
                    pass 
                elif not self._is_lecturer_education_related(query) and not context.get('force_education_response', False):
                    response = self._get_contextual_out_of_scope_response_lecturer(conversation_context, session_id)
                    token_info = {'smart_tokens_used': False, 'method': 'predefined_template'}
                    final_confidence = self.confidence_manager.normalize_confidence(0.9, "out_of_scope")
                    
                    if session_id:
                        self.memory.add_interaction(session_id, original_query, response, intent_info, entities)
                    
                    return {
                        'response': response,
                        'method': 'out_of_scope_lecturer',
                        'confidence': final_confidence,  # 🛡️ CAPPED
                        'generation_time': time.time() - start_time,
                        'original_query': original_query,
                        'restored_query': query,
                        'personalized': session_id in self._user_context_cache,
                        'token_info': token_info
                    }
                
                response, token_info = self._generate_smart_response(query, context, session_id, response_strategy)
                
                # Calculate confidence based on context and method
                semantic_score = context.get('semantic_score', 0.5) if context else 0.5
                keyword_score = context.get('keyword_score', 0.0) if context else 0.0
                
                final_confidence = self.confidence_manager.calculate_response_confidence(
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    context_bonus=0.05 if conversation_context.get('recent_conversation_summary') else 0.0,
                    method='two_stage_reranking' if context and context.get('two_stage_reranking_used') else 'hybrid'
                )
            
            final_response = response or self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id)
            
            # 🛡️ FINAL CONFIDENCE NORMALIZATION
            if not 'final_confidence' in locals():
                final_confidence = self.confidence_manager.normalize_confidence(normalized_confidence, "final_response")
            
            # Save to memory
            if session_id:
                print(f"🧠 MEMORY DEBUG: Saving interaction to memory...")
                self.memory.add_interaction(session_id, original_query, final_response, intent_info, entities)

            return {
                'response': final_response,
                'method': f'advanced_rag_lecturer_aware_gemini_{response_strategy}',
                'strategy': response_strategy,
                'conversation_context': conversation_context,
                'confidence': final_confidence,  # 🛡️ GUARANTEED ≤ 1.0
                'generation_time': time.time() - start_time,
                'original_query': original_query,
                'restored_query': query,
                'vietnamese_restoration_used': query != original_query,
                'personalized': bool(user_context),
                'enhanced_generation': response_strategy == 'enhanced_generation',
                'token_info': token_info,  # ✅ NEW: Smart token information
                'confidence_management': {  # 🚀 NEW: Advanced confidence info
                    'raw_confidence': raw_confidence,
                    'normalized_confidence': normalized_confidence,
                    'final_confidence': final_confidence,
                    'confidence_capped': final_confidence == 1.0,
                    'confidence_source': 'advanced_calculation'
                }
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            fallback_response = self._get_smart_fallback_with_context_lecturer(query, intent_info, conversation_context, session_id)
            
            # 🛡️ CONFIDENCE CAP: Error fallback
            error_confidence = self.confidence_manager.normalize_confidence(0.1, "error_fallback")
            
            if session_id:
                self.memory.add_interaction(session_id, original_query, fallback_response, intent_info, entities)
            
            return {
                'response': fallback_response,
                'method': 'lecturer_context_aware_fallback',
                'error': str(e),
                'confidence': error_confidence,  # 🛡️ CAPPED
                'generation_time': time.time() - start_time,
                'original_query': original_query,
                'restored_query': query,
                'personalized': session_id in self._user_context_cache,
                'token_info': {'smart_tokens_used': False, 'method': 'fallback'}
            }
            
    def _generate_smart_response(self, query: str, context=None, session_id=None, strategy='balanced'):        
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
        if completion_info['confidence'] < 0.6:  # Don't auto-complete if not confident it's incomplete
            return None
        
        completion_tokens = self.token_manager.estimate_completion_tokens(incomplete_response)
        
        completion_prompt = self._build_completion_prompt(incomplete_response, original_query, context, session_id, completion_info)
        
        print(f"🔧 AUTO-COMPLETION: Attempting with {completion_tokens} tokens")
        
        completion = self._call_gemini_api_with_smart_tokens(completion_prompt, 'completion', completion_tokens, session_id)
        
        if completion:
            if completion_info['reason'] == 'missing_proper_ending':
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
            - Đảm bảo kết thúc: "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?"
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
            - Kết thúc: "{personal_address.title()} có cần em hỗ trợ thêm gì không ạ?"
            
            Câu trả lời hoàn chỉnh:"""
        
        return completion_prompt

    def _merge_incomplete_and_completion(self, incomplete: str, completion: str) -> str:
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
                'completion': -0.3, 'balanced': 0.0, 'document_context': +0.1,  # 🚀 NEW: Document context adjustment
                'two_stage_reranking': +0.05  # 🚀 NEW: Slight adjustment for advanced method
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

    def _generate_direct_lecturer_answer_smart(self, query, context, session_id=None):
        personal_address = self._get_personal_address(session_id)
        
        system_prompt = self._get_personalized_system_prompt(session_id)
        db_answer = context.get('db_answer', context.get('response', ''))

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
        response = self._call_gemini_api_with_smart_tokens(prompt, 'direct_enhance', optimal_tokens, session_id)
        
        fallback = f"Dạ {personal_address}, {db_answer} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True, 
            'method': 'direct_answer_smart_v6_advanced_confidence', 
            'optimal_tokens': optimal_tokens,
            'personal_addressing': personal_address,  # ✅ NEW: Track addressing used
            'context_aware': bool(recent_summary),  # ✅ NEW: Track context usage
            'confidence_managed': True  # 🚀 NEW: Advanced confidence management applied
        }

        return response or fallback, token_info

    def _generate_enhanced_lecturer_answer_smart(self, query, context, intent_info, entities, session_id):
        personal_address = self._get_personal_address(session_id)
        system_prompt = self._get_personalized_system_prompt(session_id)
        db_answer = context.get('db_answer', context.get('response', ''))

        conversation_context = self.memory.get_conversation_context(session_id) if session_id else {}
        recent_summary = conversation_context.get('recent_conversation_summary', '')
        
        context_section = ""
        if recent_summary:
            context_section = f"""
🗣️ NGỮ CẢNH HỘI THOẠI GẦN ĐÂY:
{recent_summary}

💡 LƯU Ý: Tham khảo ngữ cảnh trên để tránh lặp lại thông tin, tạo câu trả lời mạch lạc và tự nhiên.
"""

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
    ĐẶC BIỆT: Tạo câu trả lời chi tiết và toàn diện hơn.
---
Trả lời:
"""

        complexity_hint = 'enhanced_generation' if context.get('generation_boosted', False) else 'two_stage_reranking'
        optimal_tokens = self.token_manager.calculate_optimal_tokens(len(prompt), complexity_hint)
        response = self._call_gemini_api_with_smart_tokens(prompt, complexity_hint, optimal_tokens, session_id)
        
        fallback = f"Dạ {personal_address}, {db_answer} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?"
        
        token_info = {
            'smart_tokens_used': True, 
            'method': 'enhanced_answer_smart_v6_advanced_confidence', 
            'optimal_tokens': optimal_tokens, 
            'generation_boosted': context.get('generation_boosted', False),
            'context_aware': bool(recent_summary),  # ✅ NEW
            'confidence_managed': True,  # 🚀 NEW
            'two_stage_compatible': True  # 🚀 NEW: Compatible with Two-Stage Re-ranking
        }

        return response or fallback, token_info

    def _generate_clarification_request_smart(self, query, context, session_id=None):        
        personal_address = self._get_personal_address(session_id)
        
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
            'method': 'clarification_template_v2',
            'confidence_managed': True,  # 🚀 NEW
            'template_type': 'professional'
        }
        
        return response, token_info

    def _generate_dont_know_response_smart(self, query, context, session_id=None):        
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
        
        response = f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ {dept} qua email {contact} để được hỗ trợ chi tiết ạ. 🎓"
        
        token_info = {
            'smart_tokens_used': False,  # Used predefined template
            'method': 'dont_know_template_v2',
            'suggested_department': dept,
            'personal_addressing': personal_address,  # ✅ NEW: Track addressing used
            'confidence_managed': True  # 🚀 NEW
        }
        
        return response, token_info

    def _determine_lecturer_response_strategy(self, query, context, intent_info, conversation_context):        
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
        
        raw_confidence = context.get('confidence', 0.5) if context else 0.5
        normalized_confidence = self.confidence_manager.normalize_confidence(raw_confidence, "strategy_decision")
        
        if normalized_confidence > 0.75:  # Adjusted threshold for Advanced RAG
            return 'direct_enhance'
        
        if normalized_confidence > 0.4:  # Lower threshold due to advanced re-ranking
            return 'enhanced_generation'  # Enhanced strategy for better generation
        
        if intent_info and intent_info.get('intent') in ['greeting', 'general'] and len(query.split()) <= 5:
            return 'quick_clarify'
        
        if any(word in query.lower() for word in ['khó khăn', 'cần gấp', 'hạn cuối', 'urgent']):
            return 'supportive_brief'
        
        return 'balanced'

    def _post_process_with_lecturer_consistency(self, response, query, context, strategy, conversation_context, session_id=None):
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
            response = re.sub(r'\s*🎓.*', '', response.strip())
            response = re.sub(r'\s*(có cần.*?không ạ\?|Cần.*?không\?|Có.*?không\?).*', '', response.strip())
            
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

    def _build_enhanced_prompt(self, query: str, context=None, intent_info=None, entities=None, session_id=None):
        system_prompt = self._get_personalized_system_prompt(session_id)
        personal_address = self._get_personal_address(session_id)
        
        context_info = str(context.get('response', '')) if isinstance(context, dict) else str(context or '')
        
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
        return self._user_context_cache.get(session_id)    
    def clear_user_context(self, session_id=None):
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
        try:
            test_prompt = "Test ngắn cho giảng viên"
            response = self._call_gemini_api_with_smart_tokens(test_prompt, 'quick_clarify', 80, session_id="test")
            
            return {
                'gemini_api_available': response is not None,
                'api_key_configured': bool(self.key_manager.keys),
                'service_status': 'active' if response else 'error',
                'mode': 'advanced_rag_gemini_with_two_stage_reranking_integration_and_advanced_confidence_management',
                'memory_sessions': len(self.memory.conversations),
                'personalization_sessions': len(self._user_context_cache),
                'adaptive_token_range': self.token_manager.adaptive_token_range,
                'confidence_management': {
                    'max_confidence': self.confidence_manager.MAX_CONFIDENCE,
                    'decision_thresholds': self.confidence_manager.decision_thresholds,
                    'calibration_rules': self.confidence_manager.confidence_calibration_rules,
                    'overflow_protection_enabled': True,
                    'confidence_normalization_active': True
                },
                'features': [
                    'advanced_confidence_management',
                    'confidence_overflow_protection',
                    'confidence_normalization',
                    'two_stage_reranking_integration',
                    'advanced_rag_compatibility',
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
                    'user_memory_prompt_support',
                    'flexible_personalization',
                    'external_api_data_processing',
                    'lecturer_schedule_formatting',
                    'personal_information_handling',
                    'gender_based_addressing',
                    'conversation_context_summary',
                    'mạch_lạc_response_generation',
                    'consistent_personalization_in_errors',
                    'session_id_propagation_in_api_calls',
                    'graceful_error_handling_with_personalization',
                    'document_context_processing',
                    'pdf_docx_text_extraction',
                    'document_based_question_answering',
                    'ocr_integration_support',
                    'fine_tuned_model_compatibility',
                    'cross_encoder_simulation_support',
                    'hybrid_retrieval_enhancement'
                ]
            }
        except Exception as e:
            return {
                'gemini_api_available': False,
                'service_status': 'error',
                'error': str(e),
                'consistent_personalization': True,
                'graceful_degradation': True,
                'document_context_support': True,
                'advanced_confidence_management': True,
                'confidence_overflow_protection': True
            }

gemini_response_generator = GeminiResponseGenerator()