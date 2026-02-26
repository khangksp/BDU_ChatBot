import logging
import re
import time
from .reranker import SemanticReRanker
from .decision_engine import PureSemanticDecisionEngine
from .base_chatbot import ChatbotAI

logger = logging.getLogger(__name__)

class PureSemanticChatbotAI:
    def __init__(self, shared_response_generator):
        from .phobert_service import retriever_service        
        self.sbert_retriever = ChatbotAI(shared_response_generator=shared_response_generator)
        self.retriever_service = retriever_service
        self.response_generator = shared_response_generator
        self.decision_engine = PureSemanticDecisionEngine()        
        self.semantic_reranker = SemanticReRanker(retriever_service=self.retriever_service)        
        self.conversation_memory = {}        
        logger.info("🎯 ENHANCED PureSemanticChatbotAI initialized")
        logger.info("   🛡️ Smart penalty system enabled")
        logger.info("   🧠 Confidence-aware decision making")
        logger.info("   🎯 High-quality answer preservation")
        logger.info("   🔬 Top-5 smart candidate selection")
    
    def _check_direct_entity_query(self, query: str, session_id: str):
        """🔧 IMPROVED: Better detection of entity queries"""
        session_memory = self.get_conversation_context(session_id)
        if not session_memory or len(session_memory) == 0:
            return False, None, None

        query_lower = query.lower().strip()
        
        # 🆕 EXPANDED: More patterns to catch entity questions
        direct_patterns = [
            r'\b(vậy|thế)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+là\s+(ai|gì)\b',  # "vậy X là ai"
            r'\b(vậy|thế)\s+(thầy|cô|ông|bà|anh|chị)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\b',  # "vậy thầy X"
            r'\b(còn|và)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+(thì sao|như thế nào|là ai)\b',  # "còn X thì sao"
            r'\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+là\s+(ai|gì)\b',  # "X là ai"
            r'\b(?:ông|bà|thầy|cô|anh|chị)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s*$'  # "ông X", "bà Y"
        ]
        
        # Traditional direct references
        direct_pronouns = ['ông ấy', 'bà ấy', 'người đó', 'thầy ấy', 'cô ấy', 'anh ấy', 'chị ấy']
        has_direct_pronoun = any(pronoun in query_lower for pronoun in direct_pronouns)
        
        # Check for name patterns
        extracted_name = None
        has_direct_pattern = False
        
        for pattern in direct_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                has_direct_pattern = True
                # Extract name from different groups based on pattern
                groups = match.groups()
                for group in groups:
                    if group and len(group.split()) >= 1 and group[0].isupper():
                        # Handle single name or full name
                        if len(group.split()) == 1:
                            # Single word - check if it could be part of a longer name
                            extracted_name = group
                        else:
                            # Multiple words - likely full name
                            extracted_name = group
                        break
                if extracted_name:
                    break
        
        # ONLY proceed if there's a clear direct reference
        if not (has_direct_pronoun or has_direct_pattern):
            return False, None, None

        # Check last 3 interactions instead of just 1
        recent_interactions = session_memory[-3:] if len(session_memory) >= 3 else session_memory
        
        for interaction in reversed(recent_interactions):
            last_entities_info = interaction.get('semantic_info', {}).get('extracted_entities', {})
            last_person_entities = last_entities_info.get('person_name', [])
            
            if not last_person_entities:
                continue
            
            # If we extracted a specific name, try to match it
            if extracted_name:
                for entity in last_person_entities:
                    if self._names_match_flexible(extracted_name, entity):
                        logger.info(f"🎯 Direct entity match: '{extracted_name}' → '{entity}'")
                        return True, entity, interaction
            
            # For pronouns, use the most recent entity
            if has_direct_pronoun:
                main_entity = last_person_entities[0]
                logger.info(f"🎯 Direct pronoun reference: '{main_entity}'")
                return True, main_entity, interaction
        
        return False, None, None

    def _names_match_flexible(self, name1: str, name2: str) -> bool:
        """🆕 NEW: Flexible name matching"""
        if not name1 or not name2:
            return False
        
        # Normalize both names
        norm1 = name1.lower().strip()
        norm2 = name2.lower().strip()
        
        # Remove Vietnamese particles
        particles = ['dạ', 'ạ', 'ơi', 'nhé', 'vậy', 'thì', 'là', 'của', 'và', 'với']
        for particle in particles:
            norm1 = norm1.replace(particle, ' ')
            norm2 = norm2.replace(particle, ' ')
        
        # Clean up spaces
        norm1 = ' '.join(norm1.split())
        norm2 = ' '.join(norm2.split())
        
        # Direct match
        if norm1 == norm2:
            return True
        
        # Word-level matching
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # If both have multiple words, check overlap
        if len(words1) >= 2 and len(words2) >= 2:
            overlap = len(words1.intersection(words2))
            total_words = min(len(words1), len(words2))  # Use smaller set as denominator
            overlap_ratio = overlap / total_words if total_words > 0 else 0
            
            # 60% overlap is good enough for names
            return overlap_ratio >= 0.6
        
        # Single word vs multi-word (e.g., "cường" vs "lê văn cường")
        if len(words1) == 1 and len(words2) >= 2:
            single_word = list(words1)[0]
            return single_word in words2 and len(single_word) > 2
        elif len(words2) == 1 and len(words1) >= 2:
            single_word = list(words2)[0]
            return single_word in words1 and len(single_word) > 2
        
        # Single word matching with partial
        if len(words1) == 1 and len(words2) == 1:
            word1, word2 = list(words1)[0], list(words2)[0]
            if len(word1) >= 3 and len(word2) >= 3:
                return word1 in word2 or word2 in word1
        
        return False
    
    def _smart_entity_fallback_search(self, query: str, session_id: str = None):
        """🆕 NEW: Smart fallback khi context không match"""
        try:
            query_lower = query.lower()
            
            # Extract potential entity names from query
            potential_names = self._extract_names_from_query(query)
            
            if not potential_names:
                return []
            
            logger.info(f"🔍 Smart entity fallback: searching for {potential_names}")
            
            # Search for each potential name in knowledge base
            best_candidates = []
            
            for name in potential_names:
                # Try different query variations
                search_queries = [
                    f"{name} là ai",
                    f"ai là {name}",
                    f"thông tin {name}",
                    f"chức vụ {name}",
                    name  # Just the name itself
                ]
                
                for search_query in search_queries:
                    candidates = self.sbert_retriever.semantic_search_top_k(
                        search_query, top_k=5
                    )
                    
                    if candidates and candidates[0].get('semantic_score', 0) > 0.5:  # Lowered from 0.6
                        # Add fallback context info
                        candidates[0]['fallback_search_query'] = search_query
                        candidates[0]['extracted_name'] = name
                        candidates[0]['search_method'] = 'smart_entity_fallback'
                        best_candidates.append(candidates[0])
                        logger.info(f"Found entity info via fallback: {name} -> score={candidates[0].get('semantic_score', 0):.3f}")
                        break  # Found good match, no need to try other variations
            
            # Sort by semantic score and return best ones
            best_candidates.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
            return best_candidates[:3]  # Top 3
            
        except Exception as e:
            logger.error(f"Error in smart entity fallback search: {str(e)}")
            return []
    
    def _extract_names_from_query(self, query: str) -> list:
        """🆕 NEW: Extract potential person names from query"""
        potential_names = []
        
        # Pattern 1: "vậy X là ai", "X là ai", "ai là X" 
        patterns = [
            r'(?:vậy|thế)\s+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)*)\s+là\s+ai',
            r'([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)*)\s+là\s+ai',
            r'ai\s+là\s+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)*)',
            r'(?:ông|bà|thầy|cô|anh|chị)\s+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    name = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                else:
                    name = match
                
                if name and len(name.strip()) > 2:
                    clean_name = name.strip()
                    # Validate it looks like a Vietnamese name
                    if self._is_likely_vietnamese_name(clean_name):
                        potential_names.append(clean_name)
        
        # Also extract capitalized words that could be names
        words = query.split()
        current_name = []
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                # Check if it's not a common word
                if word.lower() not in ['Dạ', 'Vậy', 'Thế', 'Còn', 'Và', 'Hay', 'Nhưng']:
                    current_name.append(word)
            else:
                if len(current_name) >= 2:  # At least 2 words for a name
                    full_name = ' '.join(current_name)
                    if self._is_likely_vietnamese_name(full_name):
                        potential_names.append(full_name)
                current_name = []
        
        # Check remaining name at end
        if len(current_name) >= 2:
            full_name = ' '.join(current_name)
            if self._is_likely_vietnamese_name(full_name):
                potential_names.append(full_name)
        
        # Remove duplicates and return
        return list(set(potential_names))
    
    def _is_likely_vietnamese_name(self, name: str) -> bool:
        """Check if text looks like a Vietnamese person name"""
        if not name or len(name.split()) < 1:
            return False
        
        words = name.lower().split()
        
        # Common Vietnamese surnames
        common_surnames = {
            'nguyễn', 'trần', 'lê', 'phạm', 'hoàng', 'huỳnh', 'phan', 'vũ', 'võ', 'đặng', 
            'bùi', 'đỗ', 'hồ', 'ngô', 'dương', 'lý', 'cao', 'đậu', 'lưu', 'tô',
            'nguyen', 'tran', 'le', 'pham', 'hoang', 'huynh', 'phan', 'vu', 'vo', 'dang',
            'bui', 'do', 'ho', 'ngo', 'duong', 'ly', 'cao', 'dau', 'luu', 'to'
        }
        
        # If first word is common surname, likely a name
        if words[0] in common_surnames:
            return True
        
        # Check for Vietnamese name characteristics
        vietnamese_chars = 'ăâêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõôồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹđ'
        vietnamese_char_count = sum(1 for char in name.lower() if char in vietnamese_chars)
        
        # Must have reasonable length and structure
        total_chars = len(''.join(words))
        if len(words) >= 2 and 4 <= total_chars <= 20:
            # If has Vietnamese chars, likely a name
            if vietnamese_char_count >= 1:
                return True
            # Even without accents, if structure looks right, could be name
            elif len(words) <= 4 and all(len(w) >= 2 for w in words):
                return True
        
        return False
    
    def _normalize_for_matching(self, text):
        """🚀 IMPROVED: Better normalization for Vietnamese text"""
        if not text:
            return ""
        
        # Convert to lowercase first
        normalized = text.lower().strip()
        
        # Remove punctuation but keep Vietnamese characters and spaces
        normalized = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', normalized)
        
        # Remove particles but keep meaningful words  
        particles = ['dạ', 'ạ', 'ơi', 'nhé', 'vậy', 'thì', 'là', 'của', 'và', 'với', 'ai', 'gì', 'nào']
        words = normalized.split()
        filtered_words = [w for w in words if w not in particles and len(w) > 1]
        
        normalized = ' '.join(filtered_words)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _create_response_from_memory(self, query: str, entity_name: str, last_interaction: dict, session_id: str):
        """
        Tạo câu trả lời trực tiếp từ bộ nhớ (NÂNG CẤP)
        """
        personal_address = self._get_personal_address(session_id)
        last_bot_response = last_interaction.get('response', '')
        last_qa_info = last_interaction.get('sources', [{}])[0] if last_interaction.get('sources') else {}
        original_question = last_qa_info.get('question', '')
        original_answer = last_qa_info.get('answer', last_bot_response)

        # Cố gắng tìm chức vụ đã được xác định ở lượt trước
        position = "vai trò đã được đề cập" # Fallback
        last_entities = last_interaction.get('semantic_info', {}).get('extracted_entities', {})
        if 'position' in last_entities and last_entities['position']:
            position = last_entities['position'][0]

        # Xây dựng câu trả lời dựa trên thông tin đã có
        response_text = (
            f"Dạ {personal_address}, khi đề cập đến \"{entity_name}\", "
            f"em đang hiểu là {personal_address} hỏi về thông tin từ lượt trao đổi trước. "
            f"Theo đó, {entity_name} giữ chức vụ là {position} ạ. "
            f"{personal_address.title()} có cần em cung cấp thêm chi tiết nào từ thông tin gốc không ạ?"
        )

        final_score = 0.98

        return {
            'response': response_text,
            'confidence': final_score,
            'method': 'conversation_memory_direct_answer_v2',
            'decision_type': 'use_memory_direct',
            'semantic_info': {
                'final_score': final_score,
                'confidence_level': 'very_high',
                'semantic_decision': True,
                'selected_position': 0,
                'original_context': {
                    'question': original_question,
                    'answer': original_answer
                }
            },
            'context_info': {
                'search_method': 'skipped_retrieval',
                'context_used': True,
                'context_keywords': [entity_name],
            },
            'sources': last_interaction.get('sources', []),
            'processing_time': 0.1,
            'context_aware_rag': True,
            'architecture': 'context_aware_semantic_rag',
        }
    
    def _ensure_context_functionality(self):
        """🆕 CRITICAL: Đảm bảo context functionality hoạt động"""
        try:
            # Kiểm tra entity extractor
            if not hasattr(self.response_generator, 'memory'):
                logger.error("❌ CRITICAL: response_generator.memory not found")
                return False
                
            if not hasattr(self.response_generator.memory, 'entity_extractor'):
                logger.error("❌ CRITICAL: entity_extractor not found in memory")
                return False
                
            if not hasattr(self.response_generator.memory, 'get_context_for_query'):
                logger.error("❌ CRITICAL: get_context_for_query method not found")
                return False
                
            # Test basic functionality
            test_entities = self.response_generator.memory.entity_extractor.extract_entities(
                "test text", "test query"
            )
            
            logger.info("✅ Context functionality check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Context functionality check failed: {str(e)}")
            return False
    
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
        gemini_status = self.response_generator.get_system_status()
        
        # Lazy import để tránh circular import và NameError
        try:
            from qa_management.services import drive_service
            drive_status = drive_service.get_system_status()
        except Exception:
            drive_status = {'available': False, 'error': 'drive_service not available'}
        
        try:
            from .external_api_service import external_api_service
            external_api_status = external_api_service.get_system_status()
        except Exception:
            external_api_status = {'available': False, 'error': 'external_api_service not available'}
        
        retriever_status = self.retriever_service.get_system_status()
        
        # 🆕 THÊM: Context-aware features status
        context_features_status = {
            'entity_extractor_available': hasattr(self.response_generator, 'memory') and hasattr(self.response_generator.memory, 'entity_extractor'),
            'dual_search_available': hasattr(self.sbert_retriever, 'dual_semantic_search'),
            'context_memory_available': hasattr(self.response_generator, 'memory'),
            'context_keywords_supported': True,
            'entity_relationships_supported': True
        }
        
        return {
            'sbert_model': bool(self.sbert_retriever.model),
            'faiss_index': bool(self.sbert_retriever.index),
            'retriever_service_available': self.retriever_service.is_available(),
            'fine_tuned_model_available': retriever_status.get('fine_tuned_model_loaded', False),
            'gemini_available': gemini_status.get('gemini_api_available', False),
            'knowledge_entries': len(self.sbert_retriever.knowledge_data),
            'mode': 'context_aware_semantic_rag',  # 🆕 UPDATED
            'architecture': 'context_aware_semantic_rag',
            'semantic_reranking': {
                'enabled': True,
                'smart_penalty_system': True,
                'confidence_preservation': True,
                'adaptive_penalty_rates': True,
                'stage1_candidates': self.semantic_reranker.config['stage1_top_k'],
                'stage2_final': self.semantic_reranker.config['stage2_top_n']
            },
            'decision_engine': {
                'type': 'context_aware_semantic',  # 🆕 UPDATED
                'confidence_thresholds': self.decision_engine.semantic_confidence_thresholds,
                'smart_mismatch_handling': True,
                'high_confidence_preservation': True,
                'adaptive_tolerance': True
            },
            # 🆕 THÊM: Context-aware features
            'context_aware_features': [
                'entity_extraction_from_qa',
                'entity_relationship_building', 
                'context_keyword_generation',
                'dual_semantic_search',
                'context_enhanced_queries',
                'smart_fallback_mechanism',
                'context_quality_analysis',
                'conversation_continuity',
                'multi_turn_understanding',
                'entity_memory_decay'
            ],
            'context_features_status': context_features_status,
            'enhanced_semantic_features': [
                'smart_penalty_system',
                'confidence_preservation', 
                'adaptive_mismatch_tolerance',
                'tiered_decision_logic',
                'targeted_clarification',
                'high_quality_answer_protection',
                'context_aware_retrieval',  # 🆕 NEW
                'entity_based_context',     # 🆕 NEW
                'conversation_memory'       # 🆕 NEW
            ],
            'retriever_service_status': retriever_status,
            'external_api_status': external_api_status,
            'gemini_status': gemini_status
        }

    def _is_social_query(self, query: str) -> bool:
        """Kiểm tra xem câu hỏi có phải là chào hỏi xã giao không"""
        social_patterns = [
            r'^(xin )?chào( bạn| mọi người| ad| admin)?\.?$',
            r'^hi( there| guy)?\.?$',
            r'^hello\.?$',
            r'^alo( alo)?\.?$',
            r'^có ai (ở đây|không).*\??$',
            r'^bạn là ai\??$',
            r'^giới thiệu về bạn\??$',
            r'^test\.?$'
        ]
        query_lower = query.strip().lower()
        return any(re.match(p, query_lower) for p in social_patterns)
    
    def process_query(self, query, session_id=None, jwt_token=None, document_text=None):
        start_time = time.time()
        
        logger.info(f"🎯 IMPROVED Context-Aware Semantic RAG Processing: '{query}' (session: {session_id})")
        
        try:
            # STEP 1: VALIDATE INPUT
            query = self._clean_query(query)
            if not query:
                return self._get_empty_query_response()

            # 🔥 NEW: STEP 1.1 CHECK SOCIAL QUERY (CHIT-CHAT)
            # Nếu là câu xã giao, trả lời ngay, không cần tìm kiếm tài liệu
            if hasattr(self, '_is_social_query') and self._is_social_query(query):
                logger.info(f"👋 Detected social query: '{query}'")
                response_data = self.response_generator.generate_response(
                    query=query,
                    context={'mode': 'chat_only'}, # Chế độ chỉ chat
                    session_id=session_id
                )
                return {
                    'response': response_data['response'],
                    'confidence': 1.0,
                    'method': 'social_chat',
                    'processing_time': time.time() - start_time,
                    'is_education': False
                }
            
            # STEP 2: GET SESSION MEMORY
            session_memory = self.get_conversation_context(session_id) if session_id else []
            logger.info(f"🧠 Session memory: {len(session_memory)} interactions")
            
            # STEP 2.1: CHECK DIRECT ENTITY QUERY
            is_direct_hit, entity_name, last_interaction = self._check_direct_entity_query(query, session_id)
            if is_direct_hit:
                response_data = self._create_response_from_memory(query, entity_name, last_interaction, session_id)
                if session_id:
                     self._update_semantic_memory(
                        session_id, query, response_data['confidence'], response_data['decision_type'], 
                        True, response_data, None
                    )
                return response_data
            
            # STEP 3: DOCUMENT CONTEXT CHECK
            if document_text:
                doc_length = len(document_text.strip())
                logger.info(f"📄 Document context: {doc_length} characters")
            
            # STEP 4: IMPROVED CONTEXT-AWARE RETRIEVAL
            candidates = []
            search_method = 'normal'
            
            context_info = {}
            if session_id and hasattr(self.response_generator, 'memory'):
                context_info = self.response_generator.memory.get_context_for_query(session_id, query)
                logger.info(f"🔍 Context analysis: should_use={context_info.get('should_use_context', False)}, strength={context_info.get('context_strength', 0)}")

            should_try_context = (
                context_info.get('should_use_context', False) and 
                context_info.get('related_entities') and
                context_info.get('context_strength', 0) >= 1.5 
            )
            
            is_entity_query = any(pattern in query.lower() for pattern in [
                'là ai', 'ai là', 'ông ', 'bà ', 'thầy ', 'cô ',
                'vậy ', 'thế ', 'còn ', 'và ', 'gs.ts', 'tiến sĩ'
            ])
            
            if should_try_context:
                logger.info("🔄 Trying DUAL search (context + normal) for comparison")
                context_keywords = context_info.get('context_keywords', [])
                candidates, search_method = self.sbert_retriever.dual_semantic_search(
                    query, context_keywords, top_k=self.semantic_reranker.config['stage1_top_k']
                )
                logger.info(f"🔄 Dual search result: method={search_method}, candidates={len(candidates)}")
                
            elif is_entity_query:
                logger.info("🔍 Entity query detected - trying smart fallback search first")
                fallback_candidates = self._smart_entity_fallback_search(query, session_id)
                
                if fallback_candidates and fallback_candidates[0].get('semantic_score', 0) > 0.6:
                    logger.info("Smart entity fallback found good results, using them")
                    candidates = fallback_candidates
                    search_method = 'smart_entity_fallback'
                else:
                    logger.info("Smart fallback insufficient - using normal search")
                    candidates = self.sbert_retriever.semantic_search_top_k(
                        query, top_k=self.semantic_reranker.config['stage1_top_k']
                    )
                    search_method = 'normal'
            else:
                logger.info("🔍 Using NORMAL search (non-entity query)")
                candidates = self.sbert_retriever.semantic_search_top_k(
                    query, top_k=self.semantic_reranker.config['stage1_top_k']
                )
                search_method = 'normal'
                
            logger.info(f"🔍 Search completed: method={search_method}, candidates={len(candidates)}")
            
            # STEP 5: CONTEXT-AWARE SEMANTIC RE-RANKING
            context_keywords = context_info.get('context_keywords', []) if should_try_context else []
            reranked_candidates = self.semantic_reranker.rerank(candidates, query, context_keywords)
            
            # 🔥 NEW: LOW CONFIDENCE CHECK & FALLBACK (Kiểm tra điểm số thấp)
            top_score = reranked_candidates[0].get('final_score', 0) if reranked_candidates else 0
            
            if not reranked_candidates or top_score < 0.45:
                logger.warning(f"⚠️ Low confidence ({top_score:.3f}) -> Switching to General Knowledge Mode")
                
                # Gọi LLM với chế độ kiến thức chung (không dùng RAG context)
                response_data = self.response_generator.generate_response(
                    query=query,
                    context={'mode': 'general_knowledge'}, # Chế độ kiến thức chung
                    session_id=session_id
                )
                
                return {
                    'response': response_data['response'],
                    'confidence': 0.5, # Tự tin trung bình
                    'method': 'general_knowledge_fallback',
                    'processing_time': time.time() - start_time,
                    'is_education': False,
                    'original_score': top_score
                }

            # STEP 5.1: CONTEXT QUALITY ANALYSIS (Nếu điểm cao thì tiếp tục logic RAG)
            context_quality_score = self._analyze_context_quality(
                query, reranked_candidates[0], context_info, search_method
            )
            
            logger.info(f"📊 Top candidate analysis:")
            best_candidate = reranked_candidates[0]
            final_score = best_candidate.get('final_score', 0)
            logger.info(f"   Score: {final_score:.3f} | Context quality: {context_quality_score:.3f}")
            logger.info(f"   Method: {search_method}")
            
            # STEP 6: ADD CONTEXT INFO TO CANDIDATE
            best_candidate['context_method'] = search_method
            best_candidate['context_quality'] = context_quality_score
            best_candidate['context_keywords_used'] = context_info.get('context_keywords', [])
            
            # STEP 7: DECISION MAKING
            decision_type, context, should_respond = self.decision_engine.make_decision(
                query, reranked_candidates, session_memory, jwt_token, document_text
            )
            
            # STEP 8: EXECUTE DECISION
            if not should_respond:
                response_text = self._get_out_of_scope_response(session_id)
                method = 'rejected_non_education'
            else:
                if context:
                    context['context_method'] = search_method
                    context['context_quality'] = context_quality_score
                    context['context_keywords_used'] = context_info.get('context_keywords', [])
                
                response_text = self._execute_fixed_semantic_decision(
                    decision_type, query, context, session_id
                )
                method = f"{decision_type}_{search_method}"
            
            # STEP 9: UPDATE MEMORY
            if session_id and should_respond:
                self._update_semantic_memory(
                    session_id, query, final_score, decision_type, 
                    should_respond, context, document_text
                )
                
                if hasattr(self.response_generator, 'memory'):
                    self.response_generator.memory.add_interaction(
                        session_id, query, response_text, 
                        intent_info={'intent': decision_type}, 
                        entities={}
                    )
            
            processing_time = time.time() - start_time
            
            return {
                'response': response_text,
                'confidence': final_score,
                'method': method,
                'decision_type': decision_type,
                'semantic_info': {
                    'final_score': final_score,
                    'original_semantic_score': best_candidate.get('semantic_score', final_score),
                    'smart_penalty': best_candidate.get('smart_penalty', 0),
                    'mismatch_issues': best_candidate.get('mismatch_issues', []),
                    'confidence_level': context.get('confidence_level', 'unknown') if context else 'unknown',
                    'confidence_preserved': context.get('confidence_preserved', False) if context else False,
                    'selected_position': context.get('selected_position', 1) if context else 1,
                    'semantic_decision': True
                },
                'context_info': {
                    'search_method': search_method,
                    'context_used': should_try_context,
                    'context_keywords': context_info.get('context_keywords', []),
                    'context_strength': context_info.get('context_strength', 0),
                    'context_quality': context_quality_score,
                    'related_entities': context_info.get('related_entities', []),
                    'context_threshold_met': should_try_context
                },
                'sources': self._format_sources(reranked_candidates[:2]),
                'processing_time': processing_time,
                'is_education': context is not None,
                'context_aware_rag': True,
                'reference_links': best_candidate.get('reference_links', []),
                'external_api_used': decision_type == 'use_external_api',
                'semantic_reranking_used': best_candidate.get('fixed_semantic_reranking', False),
                'session_memory_used': bool(session_memory),
                'document_context_used': bool(document_text),
                'document_context_priority': decision_type == 'use_document_context',
                'architecture': 'improved_context_aware_semantic_rag',
                'enhanced_features': [
                    'smart_penalty', 'confidence_preservation', 'adaptive_tolerance', 
                    'top5_selection', 'smart_candidate_selection',
                    'improved_context_detection', 'dual_search_with_fallback', 'selective_entity_memory',
                    'context_quality_analysis', 'smart_context_threshold', 'chat_fallback'
                ]
            }
            
        except Exception as e:
            logger.error(f"⛔ Context-aware semantic processing error: {str(e)}")
            return {
                'response': self._get_error_response(session_id),
                'confidence': 0.0,
                'method': 'error_fallback',
                'processing_time': time.time() - start_time,
                'error': str(e),
                'context_aware_rag': True,
                'graceful_degradation_used': True
            }

    def _analyze_context_quality(self, query, best_candidate, context_info, search_method):
        """🆕 THÊM MỚI: Phân tích chất lượng context được sử dụng"""
        try:
            if search_method == 'normal' or not context_info.get('should_use_context', False):
                return 0.0
            
            context_keywords = context_info.get('context_keywords', [])
            if not context_keywords:
                return 0.0
            
            # Kiểm tra context keywords có xuất hiện trong candidate không
            question = best_candidate.get('question', '').lower()
            answer = best_candidate.get('answer', '').lower()
            candidate_text = f"{question} {answer}"
            
            keywords_found = 0
            for keyword in context_keywords:
                if keyword.lower() in candidate_text:
                    keywords_found += 1
            
            # Tính quality score
            keyword_ratio = keywords_found / len(context_keywords) if context_keywords else 0
            semantic_score = best_candidate.get('semantic_score', 0)
            context_strength = context_info.get('context_strength', 0)
            
            quality_score = (
                0.4 * keyword_ratio +           # 40% từ keyword match
                0.4 * semantic_score +          # 40% từ semantic score  
                0.2 * min(1.0, context_strength / 3.0)  # 20% từ context strength
            )
            
            logger.debug(f"📊 Context quality: keywords_found={keywords_found}/{len(context_keywords)}, "
                        f"semantic={semantic_score:.3f}, strength={context_strength}, quality={quality_score:.3f}")
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing context quality: {str(e)}")
            return 0.0
    
    def _execute_fixed_semantic_decision(self, decision_type, query, context, session_id):
        logger.info(f"🎯 Executing decision: {decision_type}")

        addr = self._get_personal_address(session_id)

        try:
            # ─── Tài liệu upload ────────────────────────────────────────
            if decision_type == 'use_document_context':
                response = self.response_generator.generate_response(
                    query=query, context=context, session_id=session_id
                )
                text = response.get('response', '') if response else ''
                return text if text.strip() else self._get_document_fallback(session_id)

            # ─── API giảng viên → GỌI QWEN ──────────────────────────────
            elif decision_type == 'use_external_api':
                return self._handle_external_api_decision(query, context, session_id)

            # ─── Cần đăng nhập ───────────────────────────────────────────
            elif decision_type == 'require_authentication':
                return self._handle_authentication_required(session_id)

            # ─── FAQ cao điểm → TRẢ THẲNG không qua LLM ────────────────
            elif decision_type in ('use_db_direct', 'enhance_db_answer'):
                raw = (context or {}).get('db_answer', '').strip() if context else ''
                if raw:
                    return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
                return f"Dạ {addr}, em chưa có thông tin về vấn đề này ạ. 🎓"

            # ─── Cần làm rõ ──────────────────────────────────────────────
            elif decision_type == 'ask_clarification':
                if context and context.get('smart_clarification'):
                    return self.decision_engine._create_smart_clarification_response(
                        query, context.get('mismatch_issues', []), session_id
                    )
                return self._get_clarification_fallback(session_id)

            # ─── Không biết ──────────────────────────────────────────────
            elif decision_type == 'say_dont_know':
                return self._get_dont_know_fallback(session_id)

            else:
                logger.warning(f"⚠️ Unknown decision type: {decision_type}")
                return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)

        except Exception as e:
            logger.error(f"❌ Error executing decision: {e}")
            return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
            logger.error(f"❌ Error executing FIXED semantic decision: {str(e)}")
            return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
        
    def _create_fixed_semantic_fallback_response(self, decision_type, query, context, session_id):
        personal_address = self._get_personal_address(session_id)        
        raw_answer = context.get('db_answer', '') if context else ''
        mismatch_issues = context.get('mismatch_issues', []) if context else []
        confidence_preserved = context.get('confidence_preserved', False) if context else False        
        if mismatch_issues and decision_type in ['use_db_direct', 'enhance_db_answer', 'ask_clarification']:
            logger.info("🤔 FIXED fallback: Using smart clarification due to detected mismatches")
            return self.decision_engine._create_smart_clarification_response(
                query, mismatch_issues, session_id
            )
        
        if decision_type in ['use_db_direct', 'enhance_db_answer']:
            if raw_answer and raw_answer.strip():
                logger.info(f"🔍 DEBUG - Raw database answer: '{raw_answer[:300]}...'")                
                clean_answer = raw_answer.strip()                
                clean_answer = re.sub(r'^(dạ\s+(thầy|cô|giảng viên)[^,]*,?\s*)', '', clean_answer, flags=re.IGNORECASE)
                clean_answer = re.sub(r'^(xin chào|chào)[^.!?]*[.!?]\s*', '', clean_answer, flags=re.IGNORECASE)                
                if clean_answer and not clean_answer[0].isupper():
                    clean_answer = clean_answer[0].upper() + clean_answer[1:]                
                personalized_response = f"Dạ {personal_address}, {clean_answer}"                
                if not personalized_response.strip().endswith(('?', '!', '.')):
                    personalized_response += '.'                
                if confidence_preserved:
                    personalized_response += f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ? 🎯'
                else:
                    personalized_response += f' {personal_address.title()} cần em làm rõ thêm gì không ạ? 🎯'                
                logger.info(f"🛡️ FIXED SEMANTIC FALLBACK: Formatted raw answer for {personal_address}")
                return personalized_response
            else:
                return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯"
        
        return f"Dạ {personal_address}, em sẵn sàng hỗ trợ {personal_address} về các vấn đề liên quan đến BDU. {personal_address.title()} có thể chia sẻ cụ thể hơn về điều cần hỗ trợ không ạ? 🎯"

    def _check_llm_availability(self):
        """Kiểm tra Ollama có đang chạy không."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    # Alias để tương thích với code cũ còn gọi _check_gemini_availability
    def _check_gemini_availability(self):
        return True  # Luôn trả về True, logic mới không kiểm tra Gemini

    def _validate_answer_relevance(self, query, answer):
        try:
            query_lower = query.lower()
            answer_lower = answer.lower()            
            concept_patterns = {
                'báo cáo khối lượng': ['báo cáo', 'khối lượng', 'công việc'],
                'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ'],
                'tốt nghiệp': ['tốt nghiệp', 'graduation'],
                'tạp chí': ['tạp chí', 'journal', 'bài viết'],
                'lịch giảng': ['lịch', 'giảng dạy', 'schedule'],
                'hạn nộp': ['hạn', 'deadline', 'chậm nhất']
            }            
            main_concept = None
            for concept, keywords in concept_patterns.items():
                if any(kw in query_lower for kw in keywords):
                    main_concept = concept
                    break            
            if not main_concept:
                return True
            
            concept_keywords = concept_patterns[main_concept]
            answer_has_concept = any(kw in answer_lower for kw in concept_keywords)            
            relevance_issues = []
            
            if 'báo cáo khối lượng' in query_lower and 'khối lượng học tập' in answer_lower:
                relevance_issues.append("Query về 'báo cáo khối lượng công việc' nhưng answer về 'khối lượng học tập sinh viên'")
            if 'kê khai nhiệm vụ' in query_lower and 'kê khai' not in answer_lower:
                relevance_issues.append("Query về 'kê khai nhiệm vụ' nhưng answer không chứa 'kê khai'")
            if relevance_issues:
                logger.warning(f"🔍 ANSWER RELEVANCE WARNING:")
                for issue in relevance_issues:
                    logger.warning(f"   ⚠️ {issue}")
                return False            
            return answer_has_concept            
        except Exception as e:
            logger.error(f"❌ Error in answer relevance validation: {str(e)}")
            return True

    def _clean_query(self, query):
        if not query:
            return ""
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[?]{2,}', '?', query)
        query = re.sub(r'[!]{2,}', '!', query)
        return query

    def _update_semantic_memory(self, session_id, query, final_score, decision_type, was_education, context, document_text):
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []        
        interaction = {
            'query': query,
            'semantic_info': {
                'final_score': final_score,
                'confidence_level': context.get('confidence_level', 'unknown') if context else 'unknown',
                'confidence_preserved': context.get('confidence_preserved', False) if context else False,
                'smart_penalty': context.get('smart_penalty', 0) if context else 0,
                'mismatch_issues': context.get('mismatch_issues', []) if context else [],
                'semantic_decision': True
            },
            'timestamp': time.time(),
            'user_type': 'lecturer',
            'decision_type': decision_type,
            'was_education_related': was_education,
            'fixed_semantic_processed': True,
            'document_context_used': bool(document_text),
            'document_context_priority': decision_type == 'use_document_context',
            'external_api_used': decision_type == 'use_external_api',
            'query_length': len(query.split()),
            'architecture': 'fixed_semantic_rag'
        }
        
        self.conversation_memory[session_id].append(interaction)        
        self.conversation_memory[session_id] = self.conversation_memory[session_id][-30:]
        
        logger.info(f"🧠 FIXED semantic memory updated for session {session_id}: {len(self.conversation_memory[session_id])} interactions")

    def _get_personal_address(self, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                return self.response_generator._get_personal_address(session_id)
            return ""
        except Exception as e:
            logger.error(f"❌ Error getting personal address: {str(e)}")
            return ""

    def _get_empty_query_response(self):
        return {
            'response': "Dạ, em có thể hỗ trợ gì về công việc tại BDU ạ? 🎯",
            'confidence': 0.9,
            'method': 'empty_query',
            'processing_time': 0.01,
            'fixed_semantic_rag': True
        }

    def _get_no_match_response(self):
        return {
            'response': "Dạ, em chưa có thông tin về vấn đề này. Bạn có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯",
            'confidence': 0.1,
            'method': 'no_match_semantic',
            'decision_type': 'say_dont_know',
            'processing_time': 0.01,
            'fixed_semantic_rag': True
        }

    def _get_out_of_scope_response(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em chỉ hỗ trợ các vấn đề liên quan đến công việc tại BDU thôi ạ! 🎯"
    def _get_error_response(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em gặp khó khăn kỹ thuật. Bạn có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎯"
    def _get_clarification_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} để em hỗ trợ chính xác nhất, bạn có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎯"
    def _get_dont_know_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em chưa có thông tin về vấn đề này. Bạn có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯"
    def _get_document_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em đã xem xét tài liệu nhưng gặp khó khăn trong việc trả lời. Bạn có thể đặt câu hỏi cụ thể hơn không ạ? 🎯"
    def _handle_external_api_decision(self, query, context, session_id):
        personal_address = self._get_personal_address(session_id)
        try:
            jwt_token = context.get('jwt_token')
            api_result = external_api_service.get_lecturer_schedule(jwt_token, query)

            if not api_result.get('success'):
                return self._get_api_error_response(api_result, session_id)

            lecturer_info = api_result.get('lecturer_info', {})

            # ── Câu hỏi về danh tính bản thân ("Tôi là ai", "Tôi tên gì"…) ──
            # Trả thẳng từ lecturer_info, KHÔNG cần gọi LLM với dữ liệu lịch
            identity_patterns = [
                'tôi là ai', 'toi la ai', 'tên tôi là gì', 'tôi tên gì',
                'thông tin của tôi', 'thông tin cá nhân', 'tôi là giảng viên nào',
                'mã giảng viên', 'mã số của tôi', 'số của tôi'
            ]
            query_lower = query.lower()
            if any(p in query_lower for p in identity_patterns):
                name  = lecturer_info.get('ten_giang_vien', '?')
                ma    = lecturer_info.get('ma_giang_vien', '?')
                chuc  = lecturer_info.get('chuc_danh', '')
                vitri = lecturer_info.get('vi_tri_viec_lam', '')
                trinh = lecturer_info.get('trinh_do', '')
                detail = ', '.join(filter(None, [chuc, vitri, trinh]))
                text = (
                    f"Dạ {personal_address}, thông tin của {personal_address} như sau:\n"
                    f"- **Họ và tên:** {name}\n"
                    f"- **Mã giảng viên:** {ma}\n"
                    + (f"- **Chức danh/Vị trí:** {detail}\n" if detail else "")
                    + f"\n{personal_address.title()} cần em hỗ trợ thêm gì không ạ? 🎓"
                )
                return text

            # ── Câu hỏi lịch dạy thật sự ──────────────────────────────────
            # Chỉ lấy daily_schedule (đã được lọc), KHÔNG lấy toàn bộ raw data
            daily_schedule = api_result.get('daily_schedule', {})
            schedule_summary = api_result.get('schedule_summary', {})

            # Giới hạn: tối đa 10 ngày để tránh prompt quá lớn → timeout
            if len(daily_schedule) > 10:
                sorted_dates = sorted(daily_schedule.keys())
                daily_schedule = {k: daily_schedule[k] for k in sorted_dates[:10]}
                logger.info(f"✂️ Schedule trimmed to 10 days (was {len(api_result.get('daily_schedule', {}))}) to prevent Ollama timeout.")

            slim_api_data = {
                'lecturer_info': lecturer_info,
                'schedule_summary': schedule_summary,
                'daily_schedule': daily_schedule,
                'query_context': query,
            }

            enhanced_context = {
                'instruction': 'process_external_api_data',
                'api_data': slim_api_data,
                'original_query': query,
            }

            response = self.response_generator.generate_response(
                query=query, context=enhanced_context, session_id=session_id
            )
            return response.get('response', self._get_api_fallback(api_result, session_id))

        except Exception as e:
            logger.error(f"❌ Error handling external API: {str(e)}")
            return self._get_api_error_fallback(session_id)

        
    def _handle_authentication_required(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, bạn cần đăng nhập vào ứng dụng trước ạ. 🔐"
    def _get_api_fallback(self, api_result, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em đã tìm thấy thông tin lịch giảng dạy nhưng gặp khó khăn trong việc trình bày chi tiết. Bạn có thể truy cập hệ thống quản lý đào tạo để xem thông tin đầy đủ ạ. 🎯"
    def _get_api_error_response(self, api_result, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em gặp khó khăn khi truy xuất thông tin cá nhân. Bạn có thể thử lại sau hoặc liên hệ bộ phận IT để được hỗ trợ ạ. 🎯"
    def _get_api_error_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"{_greeting(personal_address)} em gặp khó khăn kỹ thuật khi truy xuất thông tin cá nhân. Bạn có thể thử lại sau ạ. 🎯"
    def _format_sources(self, results):
        sources = []
        for result in results:
            if result and result.get('final_score', 0) > 0.2:
                sources.append({
                    'question': result['question'],
                    'category': result.get('category', 'Giảng viên'),
                    'final_score': result.get('final_score', 0),
                    'original_semantic_score': result.get('semantic_score', 0),
                    'smart_penalty': result.get('smart_penalty', 0),
                    'stage1_score': result.get('stage1_score', 0),
                    'stage2_score': result.get('stage2_score', 0),
                    'mismatch_issues': result.get('mismatch_issues', []),
                    'fixed_semantic_reranking': result.get('fixed_semantic_reranking', False)
                })
        return sources

    def get_conversation_context(self, session_id):
        return self.conversation_memory.get(session_id, [])
    def get_conversation_memory(self, session_id):
        return self.response_generator.get_conversation_memory(session_id)
    def clear_conversation_memory(self, session_id=None):
        if session_id:
            self.response_generator.clear_conversation_memory(session_id)
            if session_id in self.conversation_memory:
                del self.conversation_memory[session_id]
        else:
            self.response_generator.clear_conversation_memory()
            self.conversation_memory.clear()
    def reload_after_qa_update(self):
        logger.info("🔄 Reloading FIXED semantic knowledge base...")
        
        if hasattr(self.sbert_retriever, 'cached_data'):
            self.sbert_retriever.cached_data = None
            self.sbert_retriever.cache_timestamp = 0
        
        self.sbert_retriever.load_knowledge_base()
        
        if self.sbert_retriever.model and self.sbert_retriever.knowledge_data:
            self.sbert_retriever.build_faiss_index()
        
        logger.info("✅ FIXED semantic knowledge base reloaded successfully")

