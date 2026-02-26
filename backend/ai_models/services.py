import logging
import time
from .llm.core import LocalQwenGenerator, _greeting, _closing
from .query_response_cache import query_response_cache
from .external_api_service import external_api_service
from .semantic_chatbot import PureSemanticChatbotAI

logger = logging.getLogger(__name__)

class BDUChatbotService:
    def __init__(self):
        self.response_generator = LocalQwenGenerator()
        self.query_cache = query_response_cache
        self.semantic_chatbot = PureSemanticChatbotAI(shared_response_generator=self.response_generator)

        # 🤖 LangChain Agent System (architecture mới)
        try:
            from .langchain_agent import BDULangChainAgent
            self.langchain_agent = BDULangChainAgent(
                chatbot_ai=self.semantic_chatbot,
                external_api_service_instance=external_api_service,
            )
            self._agent_available = True
            logger.info("✅ BDULangChainAgent integrated into BDUChatbotService")
        except Exception as e:
            logger.warning(f"⚠️ LangChain Agent init failed, falling back to legacy: {e}")
            self.langchain_agent = None
            self._agent_available = False

        logger.info("🎯 BDUChatbotService initialized (Agent Architecture v2)")


    def _needs_external_api(self, query: str) -> bool:
        """Legacy check — chỉ dùng khi langchain_agent không available."""
        if not query:
            return False
        personal_info_keywords = [
            'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi',
            'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
            'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
            'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
            'hôm nay', 'hom nay', 'ngày mai', 'ngay mai',
            'tuần này', 'tuan nay', 'tuần sau', 'tuan sau', 'tuần tới', 'tuan toi',
            'tháng này', 'thang nay', 'tháng sau', 'thang sau'
        ]
        query_lower = query.lower()
        needs_api = any(keyword in query_lower for keyword in personal_info_keywords)
        logger.debug(f"🌐 Legacy API check: '{query}' -> {needs_api}")
        return needs_api

    def _get_addr(self, session_id: str) -> str:
        """Lấy cách xưng hô cho session hiện tại."""
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                return self.response_generator._get_personal_address(session_id)
        except Exception:
            pass
        return ""


    def process_query(self, query: str, session_id: str = None, jwt_token: str = None, document_text: str = None) -> dict:
        start_time = time.time()
        logger.info(f"🤖 [Agent v2] Processing: '{query}' (session={session_id}, has_token={bool(jwt_token)}, has_doc={bool(document_text)})")

        try:
            # ── Câu rỗng ─────────────────────────────────────────────────────
            if not query or len(query.strip()) < 2:
                addr = self._get_addr(session_id)
                return {
                    'response': f"{_greeting(addr)} em có thể hỗ trợ về công việc tại BDU ạ? 🎓",
                    'confidence': 0.9, 'method': 'empty_query',
                    'processing_time': time.time() - start_time, 'cache_hit': False,
                }

            # ── Bỏ qua cache với câu hỏi cá nhân/lịch ───────────────────────
            is_personal = any(k in query.lower() for k in [
                'lịch', 'dạy', 'tiết', 'tkb', 'hôm nay tôi', 'tuần này tôi',
                'tôi dạy', 'tôi giảng', 'tôi là ai', 'thông tin của tôi'
            ])
            if not is_personal:
                cached = self.query_cache.get(query)
                if cached:
                    cached['processing_time'] = time.time() - start_time
                    cached['cache_hit'] = True
                    logger.info(f"⚡ CACHE HIT: '{query}'")
                    return cached

            logger.info("💨 Proceeding with Agent System")

            # ── 🤖 LangChain Agent System (kiến trúc mới) ────────────────────
            if self._agent_available and self.langchain_agent:
                try:
                    addr = self._get_addr(session_id)
                    # ── Nếu addr rỗng nhưng có JWT → decode để lấy gender/tên ──
                    if not addr and jwt_token:
                        try:
                            lecturer_info = external_api_service.get_lecturer_info_from_token(jwt_token)
                            if lecturer_info:
                                addr = self._build_addr(lecturer_info)
                                logger.info(f"👤 [Agent] addr from JWT = '{addr}'")
                        except Exception as _e:
                            logger.warning(f"⚠️ Cannot build addr from JWT: {_e}")
                    agent_result = self.langchain_agent.run(
                        query=query,
                        session_id=session_id or 'default',
                        jwt_token=jwt_token,
                        document_text=document_text,
                        addr=addr,
                    )

                    # agent.run() trả dict (mới) hoặc str (fallback)
                    if isinstance(agent_result, dict):
                        response_text = agent_result.get('response', '')
                        agent_ref_links = agent_result.get('reference_links', [])
                        agent_qa_source = agent_result.get('qa_source', None)
                    else:
                        response_text = agent_result
                        agent_ref_links = []
                        agent_qa_source = None

                    result = {
                        'response': response_text,
                        'confidence': 0.9,
                        'method': 'langchain_agent_v2',
                        'processing_time': time.time() - start_time,
                        'cache_hit': False,
                        'cache_stored': False,
                        'agent_architecture': 'bdu_langchain_agent_v2',
                        'external_api_used': jwt_token is not None,
                        'reference_links': agent_ref_links,
                        'qa_source': agent_qa_source,
                    }

                    # Cache câu hỏi không cá nhân
                    if not is_personal:
                        stored = self.query_cache.set(query, result)
                        result['cache_stored'] = stored

                    return result

                except Exception as agent_err:
                    logger.error(f"❌ Agent error, falling back: {agent_err}")
                    # Fall through to legacy

            # ── Legacy fallback ───────────────────────────────────────────────
            logger.info("📚 Using Legacy Semantic RAG (fallback)")
            if self._needs_external_api(query):
                if jwt_token and jwt_token.strip():
                    api_result = self._handle_external_api_call(query, session_id, jwt_token)
                    api_result['cache_hit'] = False
                    api_result['cache_skipped'] = 'personal_query_legacy'
                    return api_result
                else:
                    auth_result = self._handle_authentication_required(session_id)
                    auth_result['cache_hit'] = False
                    return auth_result

            result = self.semantic_chatbot.process_query(query, session_id, jwt_token, document_text)
            result['cache_hit'] = False
            if not is_personal:
                stored = self.query_cache.set(query, result)
                result['cache_stored'] = stored
            return result

        except Exception as e:
            logger.error(f"❌ BDU Service Error: {str(e)}")
            addr = self._get_addr(session_id)
            return {
                'response': f"{_greeting(addr)} em gặp khó khăn kỹ thuật. Bạn có thể liên hệ IT qua email it@bdu.edu.vn ạ. 🎓",
                'confidence': 0.0, 'method': 'service_error',
                'processing_time': time.time() - start_time,
                'error': str(e), 'cache_hit': False, 'cache_stored': False,
            }

    # Các pattern câu hỏi danh tính thuần - KHÔNG cần gọi API mạng
    _IDENTITY_PATTERNS = [
        'tôi là ai', 'toi la ai', 'tên tôi là gì', 'tôi tên gì',
        'thông tin của tôi', 'thong tin cua toi', 'tôi là giảng viên nào',
        'mã giảng viên của tôi', 'mã số của tôi', 'số của tôi',
        'tên tôi', 'họ tên tôi', 'họ và tên tôi',
    ]

    def _handle_external_api_call(self, query: str, session_id: str, jwt_token: str) -> dict:
        start = time.time()

        try:
            # ── Bước 1: Decode JWT (nhẹ, không cần mạng) ──────────────
            lecturer_info = external_api_service.get_lecturer_info_from_token(jwt_token)

            # ── Bước 2: Câu hỏi danh tính → trả thẳng, KHÔNG gọi API ──
            query_lower = query.lower()
            is_identity_query = any(p in query_lower for p in self._IDENTITY_PATTERNS)

            if is_identity_query:
                logger.info("🪪 Identity query → returning JWT info directly, skipping network call.")
                if not lecturer_info:
                    return {
                        'response': "Dạ, em không đọc được thông tin từ token xác thực. Bạn vui lòng đăng nhập lại ạ.",
                        'confidence': 0.7, 'method': 'identity_no_token',
                        'processing_time': time.time() - start,
                    }

                name  = lecturer_info.get('ten_giang_vien', '?')
                ma    = lecturer_info.get('ma_giang_vien', '?')
                gender = str(lecturer_info.get('gender', '')).lower()
                salute = 'thầy' if gender in ('male', '0') else ('cô' if gender in ('female', '1') else '')
                last_name = name.strip().split()[-1] if name.strip() else ''
                addr = f"{salute} {last_name}" if last_name else salute

                chuc  = lecturer_info.get('chuc_danh', '')
                vitri = lecturer_info.get('vi_tri_viec_lam', '')
                trinh = lecturer_info.get('trinh_do', '')
                detail = ', '.join(filter(None, [chuc, vitri, trinh]))

                text = (
                    f"{_greeting(addr)} thông tin của bạn trong hệ thống như sau:\n"
                    f"- **Họ và tên:** {name}\n"
                    f"- **Mã giảng viên:** {ma}\n"
                    + (f"- **Chức danh/Vị trí:** {detail}\n" if detail else "")
                    + f"\n{_closing(addr, '🎓')}"
                )
                return {
                    'response': text,
                    'confidence': 0.98,
                    'method': 'identity_from_jwt',
                    'decision_type': 'use_external_api',
                    'processing_time': time.time() - start,
                    'external_api_used': False,
                    'api_priority_activated': True,
                }

            # ── Bước 3: Câu hỏi lịch dạy → cần gọi API ───────────────
            if not lecturer_info:
                logger.warning("⚠️ Cannot decode JWT for schedule query.")
                return {
                    'response': "Dạ, em không đọc được thông tin xác thực. Bạn vui lòng đăng nhập lại ạ.",
                    'confidence': 0.5, 'method': 'schedule_no_token',
                    'processing_time': time.time() - start,
                }

            addr = self._build_addr(lecturer_info)
            logger.info("🌐 Calling WEEKLY schedule API.")
            api_result = external_api_service.get_lecturer_schedule_by_week(jwt_token, query)

            if not api_result.get('success'):
                error_type = api_result.get('error_type', 'unknown')
                return {
                    'response': self._get_api_error_response(error_type, session_id),
                    'confidence': 0.1, 'method': 'external_api_failed',
                    'decision_type': 'api_error',
                    'processing_time': time.time() - start,
                    'external_api_used': True,
                    'api_priority_activated': True,
                    'graceful_degradation_used': True,
                }

            daily_schedule = api_result.get('daily_schedule', {})
            week_start = api_result.get('week_start', '')
            lecturer_name = api_result.get('lecturer_info', {}).get('ten_giang_vien', '')
            last = lecturer_name.strip().split()[-1] if lecturer_name.strip() else ''

            # ── Lịch trống → trả template ngay, không gọi Qwen ──────────
            if not daily_schedule:
                week_desc = f"tuần bắt đầu {week_start}" if week_start else "tuần này"
                return {
                    'response': (
                        f"Dạ {addr}, {week_desc} {addr} không có lịch dạy trong hệ thống.\n"
                        f"{addr.title()} cần em kiểm tra tuần khác không ạ? 🗓️"
                    ),
                    'confidence': 0.92,
                    'method': 'schedule_empty_week',
                    'decision_type': 'use_external_api',
                    'processing_time': time.time() - start,
                    'external_api_used': True,
                    'api_priority_activated': True,
                }

            slim_data = {
                'lecturer_info': {
                    'ten_giang_vien': lecturer_name,
                    'ma_giang_vien': api_result.get('lecturer_info', {}).get('ma_giang_vien', ''),
                },
                'week_start': week_start,
                'daily_schedule': daily_schedule,
                'query_context': query,
            }

            response = self.response_generator.generate_response(
                query=query,
                context={'instruction': 'process_external_api_data', 'api_data': slim_data},
                session_id=session_id,
            )

            return {
                'response': response.get('response', self._get_api_fallback(session_id)),
                'confidence': 0.95,
                'method': 'external_api_success',
                'decision_type': 'use_external_api',
                'processing_time': time.time() - start,
                'external_api_used': True,
                'api_priority_activated': True,
                'fixed_semantic_rag': True,
            }

        except Exception as e:
            logger.error(f"❌ Error in external API call: {str(e)}")
            return {
                'response': f"Dạ, em gặp khó khăn khi truy xuất thông tin. Bạn có thể thử lại sau ạ. 🎯",
                'confidence': 0.1,
                'method': 'external_api_error',
                'processing_time': time.time() - start,
                'error': str(e),
                'api_priority_activated': True,
                'graceful_degradation_used': True,
            }

    @staticmethod
    def _build_addr(lecturer_info: dict) -> str:
        """Xây dựng xưng hô từ thông tin giảng viên."""
        name = lecturer_info.get('ten_giang_vien', '')
        gender = str(lecturer_info.get('gender', '')).lower()
        salute = 'thầy' if gender in ('male', '0') else ('cô' if gender in ('female', '1') else '')
        last_name = name.strip().split()[-1] if name.strip() else ''
        if salute and last_name:
            return f"{salute} {last_name}"
        return salute  # '' nếu không rõ giới tính


    def _handle_authentication_required(self, session_id: str) -> dict:
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = ""
        except:
            personal_address = ""
            
        return {
            'response': f"{_greeting(personal_address)} để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, bạn cần đăng nhập vào ứng dụng trước ạ. 🔐",
            'confidence': 0.9,
            'method': 'authentication_required',
            'decision_type': 'require_authentication',
            'processing_time': 0.01,
            'external_api_used': False,
            'api_priority_activated': True,
            'authentication_required': True
        }

    def _get_api_fallback(self, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = ""
        except:
            personal_address = ""            
        return f"{_greeting(personal_address)} em đã tìm thấy thông tin lịch giảng dạy nhưng gặp khó khăn trong việc trình bày chi tiết. Bạn có thể truy cập hệ thống quản lý đào tạo để xem thông tin đầy đủ ạ. 🎯"

    def _get_api_error_response(self, error_type, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = "giảng viên"
        except:
            personal_address = "giảng viên"
            
        if error_type == 'token_decode_failed':
            return f"Dạ {personal_address}, phiên đăng nhập đã hết hạn. {personal_address.title()} vui lòng đăng nhập lại vào ứng dụng BDU ạ. 🔐"
        elif error_type == 'authentication_failed':
            return f"Dạ {personal_address}, thông tin đăng nhập không hợp lệ. {personal_address.title()} vui lòng đăng nhập lại ạ. 🔐"
        else:
            return f"Dạ {personal_address}, em gặp khó khăn kỹ thuật khi truy xuất thông tin. {personal_address.title()} có thể thử lại sau ạ. 🎯"

    def get_system_status(self):
        semantic_status = self.semantic_chatbot.get_system_status()
        api_status = external_api_service.get_system_status()
        cache_stats = self.query_cache.get_cache_stats()        
        return {
            'service_name': 'BDUChatbotService',
            'architecture': 'context_aware_semantic_rag',  # 🆕 UPDATED
            'chatbot_service': semantic_status,
            'external_api_service': api_status,
            'cache_performance': cache_stats,
            'context_aware_features': [  # 🆕 UPDATED features list
                'entity_extraction_and_memory',
                'context_enhanced_search',
                'dual_search_strategy',
                'smart_context_fallback',
                'conversation_continuity',
                'multi_turn_understanding',
                'entity_relationship_tracking',
                'context_quality_analysis',
                'smart_penalty_system',
                'confidence_preservation', 
                'adaptive_mismatch_tolerance',
                'tiered_decision_logic',
                'targeted_clarification',
                'high_quality_answer_protection',
                'top5_smart_candidate_selection',
                'document_context_processing',
                'external_api_integration',
                'query_response_cache',
                'graceful_degradation'
            ],
            'removed_features': [
                'intent_classification',
                'keyword_matching',
                'ensemble_methods',
                'mega_intent_system',
                'complex_context_analysis',
                'hard_coded_rules',
                'over_aggressive_penalties',
                'single_candidate_limitation',
                'context_lock_in_issues'  # 🆕 REMOVED ISSUE
            ],
            'processing_flow': [  # 🆕 UPDATED flow
                '1. Cache Check',
                '2. Personal Info API Detection', 
                '3. Context Analysis from Conversation Memory',
                '4. Dual Semantic Search (Normal + Context-Enhanced)',
                '5. Smart Search Method Selection',
                '6. Two-Stage Semantic Re-ranking',
                '7. Smart Candidate Selection from Top-5',
                '8. Context Quality Analysis',
                '9. Confidence-Aware Decision Making',
                '10. Response Generation with Smart Fallback',
                '11. Entity Extraction and Memory Update',
                '12. Cache Storage'
            ]
        }

    def test_context_functionality(self, session_id="test_session"):
        """🆕 Test context-aware functionality"""
        logger.info("🧪 Testing context-aware functionality...")
        
        test_results = {
            'entity_extraction': False,
            'context_memory': False, 
            'dual_search': False,
            'context_enhancement': False,
            'conversation_continuity': False
        }        
        try:
            # Test 1: Entity extraction
            if hasattr(self.response_generator, 'memory') and hasattr(self.response_generator.memory, 'entity_extractor'):
                entities = self.response_generator.memory.entity_extractor.extract_entities(
                    "Hiệu trưởng là Cao Việt Hiếu", 
                    "hiệu trưởng là ai"
                )
                test_results['entity_extraction'] = bool(entities)
                logger.info(f"✅ Entity extraction test: {entities}")
            
            # Test 2: Context memory
            if hasattr(self.response_generator, 'memory'):
                self.response_generator.memory.add_interaction(
                    session_id, 
                    "hiệu trưởng là ai?", 
                    "Cao Việt Hiếu", 
                    intent_info={'intent': 'test'}, 
                    entities={}
                )
                
                context_info = self.response_generator.memory.get_context_for_query(
                    session_id, 
                    "vậy Cao Việt Hiếu là ai?"
                )
                test_results['context_memory'] = context_info.get('should_use_context', False)
                logger.info(f"✅ Context memory test: {context_info}")
            
            # Test 3: Dual search
            if hasattr(self.sbert_retriever, 'dual_semantic_search'):
                candidates, method = self.sbert_retriever.dual_semantic_search(
                    "test query", 
                    ["test keyword"], 
                    top_k=5
                )
                test_results['dual_search'] = method in ['normal', 'context', 'fallback']
                logger.info(f"✅ Dual search test: method={method}, candidates={len(candidates)}")
            
            # Test 4: Context enhancement (full pipeline test)
            try:
                result = self.process_query("ai là hiệu trưởng?", session_id=session_id)
                test_results['context_enhancement'] = 'context_info' in result
                logger.info(f"✅ Context enhancement test: {result.get('context_info', {})}")
                
                # Follow-up query để test continuity  
                result2 = self.process_query("vậy người đó làm gì?", session_id=session_id)
                test_results['conversation_continuity'] = result2.get('context_info', {}).get('context_used', False)
                logger.info(f"✅ Conversation continuity test: {result2.get('context_info', {})}")
            except Exception as e:
                logger.error(f"❌ Context enhancement test failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Context functionality test failed: {str(e)}")
        
        # Cleanup test session
        if session_id and hasattr(self.response_generator, 'memory'):
            if session_id in self.response_generator.memory.conversations:
                del self.response_generator.memory.conversations[session_id]
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"🧪 Context functionality test completed: {passed_tests}/{total_tests} tests passed")
        logger.info(f"📊 Test results: {test_results}")
        
        return {
            'test_results': test_results,
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'fully_functional': passed_tests == total_tests
        }
    
    def get_conversation_memory(self, session_id):
        return self.semantic_chatbot.get_conversation_memory(session_id)
    def clear_conversation_memory(self, session_id=None):
        return self.semantic_chatbot.clear_conversation_memory(session_id)
    def reload_after_qa_update(self):
        return self.semantic_chatbot.reload_after_qa_update()
    @property
    def model(self):
        return self.semantic_chatbot.model
    @property
    def index(self):
        return self.semantic_chatbot.index
    @property
    def knowledge_data(self):
        return self.semantic_chatbot.knowledge_data
    def get_cache_stats(self):
        return self.query_cache.get_cache_stats()
    def clear_cache(self):
        return self.query_cache.clear_cache()
    def update_cache_ttl(self, new_ttl: int):
        self.query_cache.update_ttl(new_ttl)
        logger.info(f"🔄 Cache TTL updated to {new_ttl} seconds")

chatbot_ai = BDUChatbotService()