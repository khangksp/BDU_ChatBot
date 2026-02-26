"""
BDU LangChain Agent System
==========================
Kiến trúc Agent mới theo sơ đồ:
  User Query → BDUChatbotService → Decision Engine → Agent System
                                                        ├── Tool Registry
                                                        │     ├── RAG Tool (search_knowledge_base)
                                                        │     └── API Tool (get_lecturer_schedule)
                                                        ├── BDU Student Agent (Qwen2.5 + LangChain)
                                                        └── Enhanced Memory (Buffer + Entity + Summary)

Thay thế hoàn toàn keyword-matching trong needs_external_api().
LLM tự quyết định tool nào cần gọi dựa trên ngữ nghĩa câu hỏi.
"""

import logging
import time
import re
from typing import Dict, Any, Optional, List

from .llm.core import _greeting, _closing

logger = logging.getLogger(__name__)

# ─── System Prompt cho Agent ─────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """Bạn là trợ lý AI chính thức của Trường Đại học Bình Dương (BDU), hỗ trợ GIẢNG VIÊN.

QUY TẮC BẮT BUỘC (TUYỆT ĐỐI KHÔNG VI PHẠM):
1. NGÔN NGỮ: Chỉ dùng TIẾNG VIỆT. Tuyệt đối KHÔNG dùng tiếng Trung, tiếng Anh.
2. TOOLS: Luôn gọi tool phù hợp TRƯỚC khi trả lời. KHÔNG bịa thông tin từ kiến thức bên ngoài.
3. TRUNG THỰC: Nếu tool không có thông tin → nói thẳng "Em chưa có thông tin về vấn đề này."
4. XƯNG HÔ: Xưng "em". Gọi người dùng là "thầy/cô" nếu biết giới tính. KHÔNG thêm tên/chức danh vào cuối câu trả lời.
5. NGẮN GỌN: Trả lời trọng tâm, lịch sự, không lan man.

HƯỚNG DẪN CHỌN TOOL:
- `search_knowledge_base`: Dùng cho câu hỏi về QUY ĐỊNH, QUY TRÌNH, THÔNG TIN CHUNG của BDU
  Ví dụ: "Quy trình kê khai nhiệm vụ?", "Hạn nộp báo cáo?", "BDU có mấy cơ sở?", "Điều kiện xét thi đua?"
- `get_lecturer_schedule`: Dùng KHI VÀ CHỈ KHI giảng viên hỏi về LỊCH DẠY / THỜI KHÓA BIỂU CÁ NHÂN
  Ví dụ: "Tuần này tôi dạy gì?", "Hôm nay tôi có mấy tiết?", "Lịch giảng tuần tới của tôi", "Ngày mai tôi dạy môn gì?"

QUAN TRỌNG: Dựa vào ngữ nghĩa câu hỏi để chọn đúng tool. Đừng chỉ nhìn từ khóa đơn lẻ."""

# ─── Enhanced Session Memory ─────────────────────────────────────────────────

class EnhancedSessionMemory:
    """
    Enhanced Memory kết hợp:
    - Buffer Memory: Lưu N lượt hội thoại gần nhất (raw messages)
    - Entity Memory: Track tên người, môn học được nhắc đến
    - Turn Summary: Tóm tắt context hiện tại cho agent
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        # {session_id: [{"role": "human"|"ai", "content": str, "timestamp": float}]}
        self._buffer: Dict[str, List[dict]] = {}
        # {session_id: {"names": set, "subjects": set, "last_intent": str}}
        self._entities: Dict[str, dict] = {}

    # ── Buffer ───────────────────────────────────────────────────────────────
    def add_turn(self, session_id: str, human_msg: str, ai_msg: str):
        if session_id not in self._buffer:
            self._buffer[session_id] = []
        self._buffer[session_id].append({
            "role": "human", "content": human_msg, "timestamp": time.time()
        })
        self._buffer[session_id].append({
            "role": "ai", "content": ai_msg, "timestamp": time.time()
        })
        # Giữ window_size * 2 messages (mỗi turn = 2 messages)
        max_msgs = self.window_size * 2
        if len(self._buffer[session_id]) > max_msgs:
            self._buffer[session_id] = self._buffer[session_id][-max_msgs:]

        # Update entity tracking
        self._update_entities(session_id, human_msg, ai_msg)

    def get_history_messages(self, session_id: str) -> List[dict]:
        """Trả về list messages để inject vào LangChain prompt."""
        return self._buffer.get(session_id, [])

    def get_recent_turns(self, session_id: str, n: int = 3) -> List[dict]:
        """Lấy n lượt gần nhất dưới dạng pairs."""
        msgs = self._buffer.get(session_id, [])
        pairs = []
        for i in range(0, len(msgs) - 1, 2):
            if i + 1 < len(msgs):
                pairs.append({"human": msgs[i]["content"], "ai": msgs[i+1]["content"]})
        return pairs[-n:] if pairs else []

    # ── Entity Tracking ───────────────────────────────────────────────────────
    def _update_entities(self, session_id: str, human_msg: str, ai_msg: str):
        if session_id not in self._entities:
            self._entities[session_id] = {
                "names": set(),
                "subjects": set(),
                "last_intent": None,
                "last_tool": None,
            }
        entities = self._entities[session_id]

        # Detect intent từ câu hỏi
        q = human_msg.lower()
        if any(k in q for k in ['lịch', 'dạy', 'tiết', 'thời khóa', 'tkb', 'tuần', 'hôm nay', 'ngày mai']):
            entities["last_intent"] = "schedule"
            entities["last_tool"] = "get_lecturer_schedule"
        elif any(k in q for k in ['quy trình', 'quy định', 'hạn', 'deadline', 'nộp', 'kê khai', 'báo cáo']):
            entities["last_intent"] = "knowledge"
            entities["last_tool"] = "search_knowledge_base"

        # Extract Vietnamese names (words có dấu + viết hoa)
        name_pattern = r'\b(?:thầy|cô|giảng viên|ông|bà)\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]*(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]*)*)\b'
        found_names = re.findall(name_pattern, human_msg)
        for name in found_names:
            if name.strip():
                entities["names"].add(name.strip())

    def get_entity_context(self, session_id: str) -> dict:
        return self._entities.get(session_id, {})

    def get_context_summary(self, session_id: str) -> str:
        """Tạo context string để nhúng vào prompt."""
        recent = self.get_recent_turns(session_id, n=5)
        entities = self.get_entity_context(session_id)

        if not recent:
            return ""

        lines = ["[LỊCH SỬ HỘI THOẠI GẦN ĐÂY]"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"Lượt {i}: Người dùng hỏi: {turn['human'][:200]}")
            lines.append(f"         Em trả lời: {turn['ai'][:300]}")

        if entities.get("last_intent"):
            lines.append(f"\n[CONTEXT]: Intent gần nhất = {entities['last_intent']}")
        if entities.get("names"):
            names_str = ", ".join(list(entities["names"])[:5])
            lines.append(f"[CONTEXT]: Tên được đề cập = {names_str}")

        return "\n".join(lines)

    def clear_session(self, session_id: str):
        self._buffer.pop(session_id, None)
        self._entities.pop(session_id, None)

    def clear_all(self):
        self._buffer.clear()
        self._entities.clear()


# ─── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Quản lý các tools và cung cấp cho Agent.
    Mỗi tool wrap logic hiện có của hệ thống.
    """

    def __init__(self, chatbot_ai, external_api_service):
        self._chatbot_ai = chatbot_ai
        self._ext_api = external_api_service
        logger.info("✅ ToolRegistry initialized with 2 tools: search_knowledge_base, get_lecturer_schedule")

    def get_knowledge_base_answer(self, query: str) -> dict:
        """
        Tool: search_knowledge_base
        Gọi PhoBERT + FAISS + SemanticReRanker hiện có.
        Trả về dict gồm:
          - text: string kết quả cho agent sử dụng
          - reference_links: list link tài liệu (nếu có)
          - qa_source: thông tin câu hỏi QA gốc (question, score, stt)
        """
        _no_result = lambda msg: {'text': msg, 'reference_links': [], 'qa_source': None}
        try:
            logger.info(f"🔍 [Tool: search_knowledge_base] Query: '{query}'")
            sbert = self._chatbot_ai.sbert_retriever
            reranker = self._chatbot_ai.semantic_reranker

            # Step 1: FAISS search
            candidates = sbert.semantic_search_top_k(query, top_k=20)
            if not candidates:
                return _no_result("Không tìm thấy thông tin liên quan trong cơ sở tri thức.")

            # Step 2: Re-ranking
            reranked = reranker.rerank(candidates, query=query)
            if not reranked:
                return _no_result("Không tìm thấy thông tin phù hợp.")

            best = reranked[0]
            score = best.get('final_score', 0)
            answer = best.get('answer', '').strip()
            question = best.get('question', '')
            stt = best.get('STT', best.get('stt', ''))

            logger.info(f"🎯 [KB Tool] Best match: score={score:.3f}, question='{question[:60]}...'")

            if score < 0.3 or not answer:
                return _no_result("Không tìm thấy thông tin đủ liên quan trong cơ sở tri thức BDU.")

            # Lấy reference_links từ best match
            reference_links = []
            try:
                if hasattr(sbert, 'get_reference_links'):
                    reference_links = sbert.get_reference_links(best)
            except Exception as _e:
                logger.warning(f"⚠️ [KB Tool] Cannot get reference_links: {_e}")

            # QA source metadata
            qa_source = {
                'question': question,
                'score': round(score, 3),
                'stt': str(stt) if stt else '',
            }

            # Format kết quả text cho agent
            result_text = f"[KẾT QUẢ TỪ CƠ SỞ TRI THỨC BDU]\n"
            result_text += f"Câu hỏi tương tự: {question}\n"
            result_text += f"Thông tin: {answer}"
            if score >= 0.68:  # Ngưỡng CAO: hạ từ 0.75→0.68 để bypass LLM nhiều hơn
                result_text += f"\n[Độ tin cậy: CAO - {score:.0%}]"
            elif score >= 0.50:  # Ngưỡng TRUNG BÌNH: hạ từ 0.55→0.50
                result_text += f"\n[Độ tin cậy: TRUNG BÌNH - {score:.0%}]"
            else:
                result_text += f"\n[Độ tin cậy: THẤP - {score:.0%}, hãy trả lời thận trọng]"


            logger.info(f"🔗 [KB Tool] reference_links={len(reference_links)}, qa_source_q='{question[:50]}'")
            return {
                'text': result_text,
                'reference_links': reference_links,
                'qa_source': qa_source,
            }

        except Exception as e:
            logger.error(f"❌ [Tool: search_knowledge_base] Error: {e}")
            return _no_result(f"Lỗi khi tìm kiếm: {str(e)}")

    def get_schedule_data(self, query: str, jwt_token: str) -> str:
        """
        Tool: get_lecturer_schedule
        Gọi ExternalAPIService.get_lecturer_schedule_by_week() hiện có.
        Trả về dữ liệu lịch đã format để agent dùng.
        """
        if not jwt_token:
            return "[LỖI] Không có JWT token. Giảng viên cần đăng nhập để xem lịch dạy cá nhân."

        try:
            logger.info(f"🗓️ [Tool: get_lecturer_schedule] Query: '{query}'")
            result = self._ext_api.get_lecturer_schedule_by_week(jwt_token, query)

            if not result.get('success'):
                err = result.get('message', 'Không thể kết nối hệ thống lịch')
                return f"[LỖI KHI LẤY LỊCH] {err}. Giảng viên vui lòng thử lại sau."

            daily_schedule = result.get('daily_schedule', {})
            week_start = result.get('week_start', '')
            lecturer_info = result.get('lecturer_info', {})
            lecturer_name = lecturer_info.get('ten_giang_vien', '')

            if not daily_schedule:
                week_desc = f"tuần bắt đầu {week_start}" if week_start else "tuần được yêu cầu"
                return f"[KẾT QUẢ LỊCH DẠY]\nGiảng viên: {lecturer_name}\nTuần: {week_desc}\nKết quả: KHÔNG CÓ LỊCH DẠY trong tuần này."

            # Format lịch thành text rõ ràng
            lines = [f"[KẾT QUẢ LỊCH DẠY]"]
            if lecturer_name:
                lines.append(f"Giảng viên: {lecturer_name}")
            if week_start:
                lines.append(f"Tuần bắt đầu: {week_start}")
            lines.append("")

            for date_str in sorted(daily_schedule.keys()):
                entries = daily_schedule[date_str]
                if not entries:
                    continue
                lines.append(f"📅 {date_str}:")
                for e in entries:
                    mon = e.get('ten_mon_hoc', '?')
                    tiet_bd = e.get('tiet_bat_dau', '?')
                    so_tiet = e.get('so_tiet', 0)
                    tiet_kt = (tiet_bd + so_tiet - 1) if (tiet_bd != '?' and so_tiet) else e.get('tiet_ket_thuc', tiet_bd)
                    phong = e.get('ma_phong') or e.get('phong', '')
                    nhom = e.get('nhom_hoc') or e.get('ma_lop') or e.get('ten_lop', '')
                    line = f"  - {mon} | Tiết {tiet_bd}–{tiet_kt}"
                    if phong:
                        line += f" | Phòng: {phong}"
                    if nhom:
                        line += f" | Nhóm: {nhom}"
                    lines.append(line)

            logger.info(f"✅ [Schedule Tool] Returned schedule with {len(daily_schedule)} days")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"❌ [Tool: get_lecturer_schedule] Error: {e}")
            return f"[LỖI KỸ THUẬT] Không thể lấy lịch dạy: {str(e)}"


# ─── BDU LangChain Agent ──────────────────────────────────────────────────────

class BDULangChainAgent:
    """
    Agent System chính theo kiến trúc sơ đồ:
    - Tự quyết định tool nào cần dùng (không dùng keyword matching)
    - Enhanced Memory per session
    - Fallback sang KB answer nếu LangChain không available
    """

    def __init__(self, chatbot_ai, external_api_service_instance):
        self._tool_registry = ToolRegistry(chatbot_ai, external_api_service_instance)
        self._memory = EnhancedSessionMemory(window_size=10)
        self._llm = None
        self._lc_available = False
        self._chatbot_ai = chatbot_ai  # Fallback

        # Thử init LangChain
        self._try_init_langchain()
        logger.info(f"✅ BDULangChainAgent initialized (langchain_available={self._lc_available})")

    def _try_init_langchain(self):
        """Lazy init LangChain — nếu fail thì dùng fallback."""
        try:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model="qwen2.5:7b",
                temperature=0,
                num_predict=512,
                base_url="http://localhost:11434",
            )
            self._lc_available = True
            logger.info("✅ LangChain + ChatOllama (qwen2.5:7b) initialized")
        except ImportError:
            logger.warning("⚠️ langchain-ollama not installed. Using fallback mode.")
        except Exception as e:
            logger.warning(f"⚠️ LangChain init failed: {e}. Using fallback mode.")

    # ── Decision Switch (theo sơ đồ Decision Engine) ────────────────────────
    def _classify_intent(self, query: str, context_summary: str, jwt_token: Optional[str]) -> str:
        """
        Decision Engine: phân loại intent trước khi gọi agent.
        Trả về: 'schedule' | 'personal_info' | 'knowledge' | 'social'
        """
        q = query.lower().strip()

        # Social / greetings → không cần gọi tool
        social_patterns = [
            'xin chào', 'chào', 'hello', 'hi ', 'alo', 'cảm ơn', 'cảm on',
            'thank', 'tạm biệt', 'bye', 'giỏi không', 'khỏe không', 'bạn là ai'
        ]
        if any(p in q for p in social_patterns) and len(query.split()) <= 8:
            return 'social'

        # Personal identity intent: tôi là ai, thông tin của tôi, mã giảng viên...
        identity_signals = [
            'tôi là ai', 'toi la ai',
            'tên tôi là gì', 'tôi tên gì', 'họ tên tôi',
            'thông tin của tôi', 'thong tin cua toi',
            'thông tin cá nhân', 'mã giảng viên', 'mã số của tôi',
            'tôi là giảng viên nào', 'hồ sơ của tôi',
        ]
        if any(p in q for p in identity_signals):
            return 'personal_info'

        # Schedule intent
        schedule_signals = [
            'lịch dạy', 'lịch giảng', 'thời khóa biểu', 'tkb',
            'hôm nay tôi', 'ngày mai tôi', 'tuần này tôi', 'tuần tới tôi',
            'tôi dạy', 'tôi giảng', 'lịch của tôi', 'lich cua toi',
            'tôi có lớp', 'môn của tôi', 'học kỳ này tôi',
            'tiết hôm nay', 'tiết ngày mai',
        ]
        if any(p in q for p in schedule_signals):
            return 'schedule'

        return 'knowledge'


    # ── Core Agent Execution ─────────────────────────────────────────────────
    def run(
        self,
        query: str,
        session_id: str,
        jwt_token: Optional[str] = None,
        document_text: Optional[str] = None,
        addr: str = ""
    ) -> dict:
        """
        Entry point chính của Agent System.
        Trả về dict: {'response': str, 'reference_links': list, 'qa_source': dict|None, 'intent': str}
        """
        start = time.time()
        logger.info(f"🤖 [BDULangChainAgent] Query='{query}' | session={session_id} | has_jwt={bool(jwt_token)}")

        # ── Document context → xử lý riêng ───────────────────────────────
        if document_text and document_text.strip():
            doc_response = self._handle_document_query(query, document_text, session_id, addr)
            return {
                'response': doc_response,
                'reference_links': [],
                'qa_source': None,
                'intent': 'document',
            }

        # ── Lấy context từ memory ─────────────────────────────────────────
        context_summary = self._memory.get_context_summary(session_id)

        # ── Decision Engine (Switch) ──────────────────────────────────────
        intent = self._classify_intent(query, context_summary, jwt_token)
        logger.info(f"🔀 [Decision Engine] Intent = {intent}")

        response_text = ""
        reference_links = []
        qa_source = None

        if intent == 'social':
            response_text = self._handle_social(query, addr)

        elif intent == 'personal_info':
            # Câu hỏi thông tin cá nhân: tôi là ai, mã giảng viên...
            response_text = self._handle_personal_info(query, jwt_token, session_id, addr)

        elif intent == 'schedule':
            agent_result = self._handle_with_agent(
                query=query,
                session_id=session_id,
                jwt_token=jwt_token,
                context_summary=context_summary,
                addr=addr,
                forced_tool='schedule'
            )
            if isinstance(agent_result, dict):
                response_text = agent_result.get('response', '')
                reference_links = agent_result.get('reference_links', [])
                qa_source = agent_result.get('qa_source', None)
            else:
                response_text = agent_result

        else:  # knowledge
            agent_result = self._handle_with_agent(
                query=query,
                session_id=session_id,
                jwt_token=jwt_token,
                context_summary=context_summary,
                addr=addr,
                forced_tool='knowledge'
            )
            if isinstance(agent_result, dict):
                response_text = agent_result.get('response', '')
                reference_links = agent_result.get('reference_links', [])
                qa_source = agent_result.get('qa_source', None)
            else:
                response_text = agent_result

        # ── Cập nhật memory ───────────────────────────────────────────────
        if response_text:
            self._memory.add_turn(session_id, query, response_text)

        elapsed = time.time() - start
        logger.info(f"✅ [BDULangChainAgent] Done in {elapsed:.2f}s | intent={intent} | links={len(reference_links)}")
        return {
            'response': response_text,
            'reference_links': reference_links,
            'qa_source': qa_source,
            'intent': intent,
        }

    def _handle_personal_info(self, query: str, jwt_token: Optional[str], session_id: str, addr: str) -> str:
        """
        Xử lý câu hỏi thông tin cá nhân: 'Tôi là ai', 'mã giảng viên của tôi'...
        - Có jwt_token → decode JWT và trả thông tin giảng viên
        - Không có jwt_token → yêu cầu đăng nhập
        """
        addr_or_neutral = addr.title() if addr else "thầy/cô"
        if not jwt_token or not jwt_token.strip():
            return (
                f"{_greeting(addr)} để em có thể cung cấp thông tin cá nhân, "
                f"{addr_or_neutral} cần đăng nhập vào ứng dụng trước ạ. 🔐"
            )

        try:
            # Decode JWT để lấy thông tin giảng viên (không cần gọi API mạng)
            lecturer_info = self._tool_registry._ext_api.get_lecturer_info_from_token(jwt_token)
            if not lecturer_info:
                return (
                    f"{_greeting(addr)} em không đọc được thông tin từ token xác thực. "
                    f"{addr_or_neutral} vui lòng đăng nhập lại ạ. 🔐"
                )

            name   = lecturer_info.get('ten_giang_vien', '?')
            ma     = lecturer_info.get('ma_giang_vien', '?')
            chuc   = lecturer_info.get('chuc_danh', '')
            vitri  = lecturer_info.get('vi_tri_viec_lam', '')
            trinh  = lecturer_info.get('trinh_do', '')
            detail = ', '.join(filter(None, [chuc, vitri, trinh]))

            text = (
                f"{_greeting(addr)} thông tin của {addr_or_neutral} trong hệ thống như sau:\n"
                f"- **Họ và tên:** {name}\n"
                f"- **Mã giảng viên:** {ma}\n"
            )
            if detail:
                text += f"- **Chức danh/Vị trí:** {detail}\n"
            text += f"\n{_closing(addr, '🎓')}"

            logger.info(f"✅ [personal_info] Returned identity for: {name}")
            return text

        except Exception as e:
            logger.error(f"❌ [personal_info] Error: {e}")
            return (
                f"{_greeting(addr)} em gặp khó khăn khi đọc thông tin xác thực. "
                f"{addr_or_neutral} thử lại sau hoặc liên hệ IT để được hỗ trợ ạ. 🎓"
            )


    def _handle_social(self, query: str, addr: str) -> str:
        """Câu xã giao → trả thẳng, không gọi tool."""
        q = query.lower()
        addr_or_neutral = addr.title() if addr else "thầy/cô"
        if any(k in q for k in ['xin chào', 'chào', 'hello', 'hi']):
            greeting = _greeting(addr)
            who = f". {addr_or_neutral} cần hỏi về điều gì ạ? 🎓"
            return f"{greeting} Em là trợ lý AI của BDU. Em có thể hỗ trợ về quy định, quy trình và lịch dạy{who}"
        if any(k in q for k in ['cảm ơn', 'thank']):
            return f"Dạ, không có gì ạ! {_closing(addr, '🎓')}"
        if any(k in q for k in ['tạm biệt', 'bye']):
            name_part = f" {addr}" if addr else ""
            return f"Dạ tạm biệt{name_part}! Chúc {addr_or_neutral} một ngày làm việc hiệu quả ạ! 🎓"
        # Default social
        if self._lc_available:
            return self._call_llm_direct(query, addr)
        return f"{_greeting(addr)} em có thể hỗ trợ về các vấn đề liên quan đến BDU ạ! 🎓"

    @staticmethod
    def _should_ask_clarification(query: str) -> bool:
        """
        Kiểm tra câu hỏi có quá mơ hồ/thiếu ngữ cảnh không.
        True → chatbot nên hỏi lại thay vì cố gắng trả lời.
        """
        q = query.strip()
        q_lower = q.lower()
        words = q_lower.split()

        # Câu chỉ có 1 từ ngắn, không rõ người hỏi muốn gì
        if len(words) == 1 and len(q) < 8:
            # Ngoại lệ: từ khóa rõ ràng vẫn trả lời
            clear_one_word = [
                'lịch', 'tkb', 'kpi', 'moodle', 'phí', 'lương',
                'quy', 'cơ sở', 'học phí', 'danh sách',
            ]
            if not any(kw in q_lower for kw in clear_one_word):
                return True

        # Câu bắt đầu bằng từ chỉ thị có ít từ (thiếu chủ ngữ rõ ràng)
        contextual_starters = [
            'vẫy', 'vậy', 'còn', 'thế thì', 'thế?', 'vẫy?',
            'vậy?', 'vẫy thì', 'vậy thôi', 'còn cái', 'đó thì', 'đờ đó',
        ]
        if any(q_lower.startswith(s) for s in contextual_starters) and len(words) <= 4:
            return True

        # Câu chỉ gồm đại từ hoặc từ chỉ thị, không có danh từ thực
        pronoun_only = [
            'vẫy sao', 'sao vậy', 'hả', 'ha',
            'đúng không', 'đúng hông',
        ]
        if q_lower in pronoun_only:
            return True

        return False

    def _ask_clarification(self, addr: str) -> str:
        """Trả về câu hỏi lại lịch sự, đúng giới tính."""
        greeting = _greeting(addr)
        addr_title = addr.title() if addr else ""
        if addr_title:
            return f"{greeting} em chưa rõ câu hỏi ạ. {addr_title} có thể hỏi lại cụ thể hơn được không ạ? 🙏"
        return f"{greeting} em chưa rõ câu hỏi ạ. Có thể hỏi lại cụ thể hơn được không ạ? 🙏"

    def _handle_with_agent(
        self,
        query: str,
        session_id: str,
        jwt_token: Optional[str],
        context_summary: str,
        addr: str,
        forced_tool: str = 'knowledge'
    ) -> dict:
        """
        Gọi Tool trực tiếp dựa trên forced_tool (schedule hoặc knowledge),
        sau đó dùng LLM để format câu trả lời từ kết quả tool.
        Trả về dict: {'response': str, 'reference_links': list, 'qa_source': dict|None}
        """
        # Helper để bao response string thành dict (dùng cho schedule & social paths)
        def _wrap_str(s: str, links=None, source=None):
            return {'response': s, 'reference_links': links or [], 'qa_source': source}

        # ── Bước 0: Kiểm tra câu hỏi mơ hồ → hỏi lại ──────────────────
        if forced_tool == 'knowledge' and self._should_ask_clarification(query):
            logger.info(f"❓ [Agent] Ambiguous query detected, asking clarification: '{query}'")
            return _wrap_str(self._ask_clarification(addr))

        # ── Bước 1: Gọi Tool ─────────────────────────────────────────────
        if forced_tool == 'schedule':
            tool_text = self._tool_registry.get_schedule_data(query, jwt_token)
            reference_links = []
            qa_source = None
        else:
            kb_result = self._tool_registry.get_knowledge_base_answer(query)
            tool_text = kb_result.get('text', '')
            reference_links = kb_result.get('reference_links', [])
            qa_source = kb_result.get('qa_source', None)

        logger.info(f"📦 [Tool Result] {tool_text[:200]}...")

        # ── Bước 2: Kiểm tra lỗi / không có dữ liệu ─────────────────────
        if "[LỖI]" in tool_text or "Không tìm thấy" in tool_text:
            if forced_tool == 'knowledge':
                addr_or_neutral = addr.title() if addr else "thầy/cô"
                return _wrap_str(
                    f"{_greeting(addr)} em chưa có thông tin về vấn đề này trong CSDL BDU. {addr_or_neutral} có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết hơn ạ. 🎓"
                )
            else:
                return _wrap_str(f"{_greeting(addr)} {tool_text}. 🎓")

        # ── Bước 3: Nếu tool schedule → có dữ liệu, format trực tiếp ────
        if forced_tool == 'schedule':
            return _wrap_str(self._format_schedule_response(tool_text, query, addr, jwt_token))

        # ── Bước 4: KB answer → Phân tầng theo confidence ───────────────────
        # CAO (score >= 0.68)    → Bypass LLM, trả thẳng (nhanh nhất)
        # TRUNG BÌNH (0.50-0.68) → Smart check: ngắn/rõ → bypass, phức tạp → LLM
        # THẤP (< 0.50)          → Hỏi lại người dùng để làm rõ câu hỏi
        has_cao   = "[Độ tin cậy: CAO"        in tool_text
        has_trung = "[Độ tin cậy: TRUNG BÌNH" in tool_text
        has_thap  = "[Độ tin cậy: THẤP"       in tool_text

        import re as _re
        # Extract nội dung "Thông tin:" (dùng chung cho cả 3 nhánh)
        m = _re.search(
            r'Thông tin:\s*(.*?)\s*(?=\[Độ tin cậy:|$)',
            tool_text,
            _re.DOTALL,
        )
        answer_line = m.group(1).strip() if m else ""

        if has_cao and answer_line:
            # CAO → bypass LLM, trả thẳng ngay
            logger.info("⚡ [KB] Bypass LLM (confidence=CAO), format direct")
            return _wrap_str(self._format_kb_response(answer_line, addr), reference_links, qa_source)

        if has_trung and answer_line:
            # ── Smart bypass: câu trả lời ngắn & rõ ràng → không cần LLM ──
            _has_complex_list = answer_line.count('\n') > 5 or answer_line.count(';') > 4
            _needs_llm = _has_complex_list and len(answer_line) > 400
            if not _needs_llm:
                logger.info(f"⚡ [KB] Smart bypass (TB, {len(answer_line)}c), format direct")
                return _wrap_str(self._format_kb_response(answer_line, addr), reference_links, qa_source)
            # TRUNG BÌNH phức tạp → LLM (có cache trong _synthesize_with_llm)
            if self._lc_available:
                logger.info("🧠 [KB] Confidence=TB+complex, using LLM")
                return _wrap_str(self._synthesize_with_llm(query, tool_text, context_summary, addr), reference_links, qa_source)
            logger.info("⚡ [KB] Bypass LLM (confidence=TB, LLM unavailable), format direct")
            return _wrap_str(self._format_kb_response(answer_line, addr), reference_links, qa_source)

        if has_thap:
            # THẤP → hỏi lại người dùng để làm rõ câu hỏi
            logger.info("❓ [KB] Confidence=THẤP, asking user for clarification")
            return _wrap_str(self._handle_low_confidence(tool_text, query, addr), reference_links, qa_source)

        # ── Bước 5: Không có KB answer → LLM hoặc format thủ công ───────────
        if self._lc_available:
            logger.info("🔁 [KB] No direct answer extracted, falling back to LLM synthesis")
            return _wrap_str(self._synthesize_with_llm(query, tool_text, context_summary, addr), reference_links, qa_source)

        # ── Fallback: Format thủ công ─────────────────────────────────────
        return _wrap_str(self._format_fallback_response(tool_text, addr), reference_links, qa_source)


    def _format_schedule_response(self, tool_result: str, query: str, addr: str, jwt_token: str) -> str:
        """Format lịch dạy từ tool result với câu từ tự nhiên hơn."""
        from datetime import datetime, timedelta

        addr_or_neutral = addr.title() if addr else "thầy/cô"

        # ── Đọc thông tin tuần từ tool_result (hỗ trợ cả 2 format) ─────────────────
        # Format 1 (có lịch): "Tuần bắt đầu: 23-02-2026"
        # Format 2 (không lịch): "Tuần: tuần bắt đầu 23-02-2026"
        import re as _re
        week_start_raw = ""  # vd: "23-02-2026"
        for line in tool_result.split('\n'):
            stripped = line.strip()
            if stripped.startswith("Tuần bắt đầu:") or stripped.startswith("Tuần:"):
                # Extract ngày theo pattern DD-MM-YYYY
                date_match = _re.search(r'(\d{2}-\d{2}-\d{4})', stripped)
                if date_match:
                    week_start_raw = date_match.group(1)
                break

        # ── Tính ngày kết thúc tuần (Thứ 2 + 6 ngày = Chủ nhật) ──────────
        week_range_str = ""
        if week_start_raw:
            try:
                dt_start = datetime.strptime(week_start_raw, "%d-%m-%Y")
                dt_end = dt_start + timedelta(days=6)
                week_range_str = f"{dt_start.strftime('%d/%m/%Y')} đến {dt_end.strftime('%d/%m/%Y')}"
            except ValueError:
                week_range_str = week_start_raw

        week_label = f"tuần từ {week_range_str}" if week_range_str else "tuần này"

        # ── Trường hợp KHÔNG CÓ LỊCH DẠY ────────────────────────────────
        if "KHÔNG CÓ LỊCH DẠY" in tool_result:
            return (
                f"{_greeting(addr)} em đã kiểm tra lịch dạy của {addr_or_neutral} "
                f"trong {week_label}, {addr_or_neutral} không có lịch dạy trong tuần này ạ.\n"
                f"{_closing(addr, '🗓️')}"
            )

        # ── Trường hợp CÓ LỊCH DẠY ────────────────────────────────────────
        # Lấy các dòng lịch (dòng ngày + dòng tiết)
        schedule_lines = []
        for line in tool_result.split('\n'):
            stripped = line.strip()
            if stripped.startswith('📅') or stripped.startswith('  -') or stripped.startswith('-'):
                schedule_lines.append(stripped)

        if schedule_lines:
            schedule_text = '\n'.join(schedule_lines)
            return (
                f"{_greeting(addr)} em đã kiểm tra lịch dạy của {addr_or_neutral} "
                f"trong {week_label}, {addr_or_neutral} có lịch dạy như sau ạ:\n\n"
                f"{schedule_text}\n\n"
                f"{_closing(addr, '🗓️')}"
            )

        # Fallback: LLM tổng hợp nếu có
        if self._lc_available:
            return self._synthesize_with_llm(query, tool_result, "", addr)

        return (
            f"{_greeting(addr)} em đã kiểm tra lịch dạy của {addr_or_neutral} "
            f"trong {week_label}:\n{tool_result}\n{_closing(addr, '🗓️')}"
        )

    def _format_kb_response(self, answer: str, addr: str) -> str:
        """Format KB answer thành câu trả lời lịch sự."""
        clean = answer.strip()
        # Bỏ lời chào có sẵn trong DB
        import re as _re
        clean = _re.sub(r'^(dạ\s+(thầy|cô|giảng viên)[^,]*,?\s*)', '', clean, flags=_re.IGNORECASE)
        clean = _re.sub(r'^(xin\s+chào|chào)[^.!?\n]*[.!?\n]\s*', '', clean, flags=_re.IGNORECASE)
        if clean and not clean[0].isupper():
            clean = clean[0].upper() + clean[1:]
        result = f"{_greeting(addr)} {clean}"
        if not result.rstrip().endswith(('.', '!', '?')):
            result += '.'
        result += f' {_closing(addr, "🎓")}'
        return result

    def _format_fallback_response(self, tool_result: str, addr: str) -> str:
        """Fallback format khi không có LLM."""
        answer = ""
        for line in tool_result.split('\n'):
            if line.startswith("Thông tin:"):
                answer = line.replace("Thông tin:", "").strip()
                break
        if answer:
            return self._format_kb_response(answer, addr)
        addr_or_neutral = addr.title() if addr else "thầy/cô"
        return f"{_greeting(addr)} em đã tra cứu nhưng chưa có thông tin đủ rõ. {addr_or_neutral} có thể hỏi cụ thể hơn không ạ? 🎓"

    def _handle_low_confidence(self, tool_result: str, query: str, addr: str) -> str:
        """
        Xử lý khi confidence THẤP (score < 0.55):
        Trả về nội dung gần nhất tìm được + hỏi lại để làm rõ câu hỏi.
        """
        addr_or_neutral = addr.title() if addr else "thầy/cô"
        # Lấy thông tin gần nhất từ KB để gợi ý
        similar_q = ""
        answer_snippet = ""
        for line in tool_result.split('\n'):
            if line.startswith("Câu hỏi tương tự:") and not similar_q:
                similar_q = line.replace("Câu hỏi tương tự:", "").strip()
            if line.startswith("Thông tin:") and not answer_snippet:
                answer_snippet = line.replace("Thông tin:", "").strip()[:150]

        if answer_snippet:
            logger.info(f"❓ [KB] Low confidence, showing nearest result + asking clarification")
            return (
                f"{_greeting(addr)} em tìm được thông tin gần nhất, nhưng không chắc đây là điều "
                f"{addr_or_neutral} muốn hỏi:\n\n"
                f"📌 *{answer_snippet}{'...' if len(answer_snippet) >= 150 else ''}*\n\n"
                f"{addr_or_neutral} có thể hỏi cụ thể hơn không ạ? Em sẽ tìm chính xác hơn. 🙏"
            )

        # Không có gì gần → hỏi lại ngay
        return (
            f"{_greeting(addr)} em chưa tìm được thông tin phù hợp với câu hỏi này. "
            f"{addr_or_neutral} có thể mô tả cụ thể hơn hoặc thử hỏi cách khác không ạ? 🙏"
        )

    # ── In-memory LLM Output Cache ─────────────────────────────────────────
    # Cache theo hash của KB answer → lần 2 cùng KB entry: 0ms
    _llm_output_cache: dict = {}
    _LLM_CACHE_MAX = 500      # tối đa 500 entries
    _LLM_CACHE_TTL = 3600     # 1 giờ

    def _synthesize_with_llm(
        self, query: str, tool_result: str, context_summary: str, addr: str
    ) -> str:
        """Dùng Qwen2.5 để tổng hợp câu trả lời từ kết quả tool (có in-memory cache)."""
        import hashlib as _hashlib, time as _time

        # ── Cache lookup ──────────────────────────────────────────────────
        # Key theo tool_result (KB answer) không phụ thuộc query cụ thể
        cache_key = _hashlib.md5(tool_result.encode("utf-8")).hexdigest()
        now = _time.time()
        cached = self.__class__._llm_output_cache.get(cache_key)
        if cached and (now - cached["ts"]) < self._LLM_CACHE_TTL:
            logger.info(f"⚡ [LLM Cache HIT] key={cache_key[:8]} age={int(now-cached['ts'])}s")
            # Cập nhật addr greeting nếu khác
            text = cached["text"]
            return text

        try:
            begin = f'Bắt đầu bằng "Dạ {addr},".' if addr else 'Bắt đầu bằng "Dạ,".'
            user_prompt = f"""Bạn đang hỗ trợ người dùng BDU{' - ' + addr if addr else ''} tại BDU.

{context_summary}

Câu hỏi: {query}

Thông tin từ hệ thống:
{tool_result}

Hãy trả lời câu hỏi dựa HOÀN TOÀN vào thông tin trên. {begin}
TUYỆT ĐỐI KHÔNG thêm tên, chức danh, hay footer vào cuối câu trả lời."""

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            from langchain_core.messages import HumanMessage, SystemMessage
            lc_msgs = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = self._llm.invoke(lc_msgs)
            text = response.content.strip()

            # Hậu xử lý: Remove Chinese characters nếu có
            if re.search(r'[\u4e00-\u9fff]', text):
                logger.warning("⚠️ Chinese detected in LLM response, using fallback format")
                return self._format_fallback_response(tool_result, addr)

            # Remove LLM signature
            text = re.sub(r'\n[^\n]*[Tt]r\u1ee3\s*l\u00fd\s*BDU\s*$', '', text, flags=re.IGNORECASE).rstrip()
            text = re.sub(r'\n\s*\n([A-ZÀÁÂÃÈÉÊÌ][a-zàáâãèéê]+(?:\s+[A-ZÀÁÂÃÈÉÊÌ][a-zàáâãèéê]+){1,3})\s*$', '', text).rstrip()

            # ── Cache store ───────────────────────────────────────────────
            if len(self.__class__._llm_output_cache) >= self._LLM_CACHE_MAX:
                # Evict oldest entry (FIFO đơn giản)
                oldest = min(self.__class__._llm_output_cache, key=lambda k: self.__class__._llm_output_cache[k]["ts"])
                del self.__class__._llm_output_cache[oldest]
            self.__class__._llm_output_cache[cache_key] = {"text": text, "ts": now}
            logger.info(f"💾 [LLM Cache STORE] key={cache_key[:8]}, cache_size={len(self.__class__._llm_output_cache)}")

            return text

        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            return self._format_fallback_response(tool_result, addr)

    def _call_llm_direct(self, query: str, addr: str) -> str:
        """Gọi LLM trực tiếp cho câu social mà không cần tool."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            begin = f"Bắt đầu bằng 'Dạ {addr},'" if addr else "Bắt đầu bằng 'Dạ,'"
            resp = self._llm.invoke([
                SystemMessage(content="Em là trợ lý AI BDU. Trả lời thân thiện, ngắn gọn bằng TIẾNG VIỆT. Xưng 'em', gọi người dùng là 'thầy/cô'."),
                HumanMessage(content=f"Người dùng hỏi: {query}\n{begin}"),
            ])
            text = resp.content.strip()
            if re.search(r'[\u4e00-\u9fff]', text):
                return f"{_greeting(addr)} em có thể hỗ trợ về các vấn đề tại BDU ạ! 🎓"
            return text
        except Exception as e:
            logger.error(f"❌ LLM direct call failed: {e}")
            return f"{_greeting(addr)} em có thể hỗ trợ về các vấn đề tại BDU ạ! 🎓"

    def _handle_document_query(self, query: str, doc_text: str, session_id: str, addr: str) -> str:
        """Xử lý câu hỏi về tài liệu được upload."""
        doc_excerpt = doc_text[:4000]
        if self._lc_available:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                begin = f'Bắt đầu bằng "Dạ {addr},".' if addr else 'Bắt đầu bằng "Dạ,".'
                prompt = f"""Tài liệu được gửi:

[TÀI LIỆU]
{doc_excerpt}
[KẾT THÚC TÀI LIỆU]

Câu hỏi: {query}
Hãy trả lời dựa trên tài liệu. {begin}"""

                resp = self._llm.invoke([
                    SystemMessage(content=AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                return resp.content.strip()
            except Exception as e:
                logger.error(f"❌ Document query LLM error: {e}")

        addr_or_neutral = addr.title() if addr else "thầy/cô"
        return f"{_greeting(addr)} em đã xem tài liệu nhưng gặp khó khăn kỹ thuật. {addr_or_neutral} thử lại sau ạ. 🎓"

    # ── Streaming Methods ────────────────────────────────────────────────────

    def _synthesize_with_llm_stream(
        self, query: str, tool_result: str, context_summary: str, addr: str
    ):
        """
        Phiên bản streaming của _synthesize_with_llm.
        Dùng LangChain .stream() để yield từng text chunk.
        """
        try:
            begin = f'Bắt đầu bằng "Dạ {addr},".' if addr else 'Bắt đầu bằng "Dạ,".'
            user_prompt = f"""Bạn đang hỗ trợ người dùng BDU{' - ' + addr if addr else ''} tại BDU.

{context_summary}

Câu hỏi: {query}

Thông tin từ hệ thống:
{tool_result}

Hãy trả lời câu hỏi dựa HOÀN TOÀN vào thông tin trên. {begin}
TUYỆT ĐỐI KHÔNG thêm tên, chức danh, hay footer vào cuối câu trả lời."""

            from langchain_core.messages import HumanMessage, SystemMessage
            lc_msgs = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            buffer = []
            for chunk in self._llm.stream(lc_msgs):
                delta = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if delta:
                    buffer.append(delta)
                    yield delta

            # Kiểm tra Chinese sau khi stream xong
            full_text = "".join(buffer)
            if re.search(r'[\u4e00-\u9fff]', full_text):
                logger.warning("⚠️ Chinese detected in LLM stream, yielding fallback")
                fallback = self._format_fallback_response(tool_result, addr)
                yield "\n[RETRY]" + fallback

        except Exception as e:
            logger.error(f"❌ LLM stream synthesis failed: {e}")
            yield self._format_fallback_response(tool_result, addr)

    def run_stream(
        self,
        query: str,
        session_id: str,
        jwt_token: Optional[str] = None,
        document_text: Optional[str] = None,
        addr: str = ""
    ):
        """
        Phiên bản streaming của run().
        Generator yield: các str chunk hoặc dict metadata (chunk cuối).
        Format dict cuối: {"done": True, "reference_links": [...], "qa_source": {...}, "intent": str}
        """
        start = time.time()
        logger.info(f"🌊 [BDULangChainAgent.run_stream] Query='{query}' | session={session_id}")

        reference_links = []
        qa_source = None
        intent = "knowledge"

        try:
            # ── Document context ─────────────────────────────────────────────
            if document_text and document_text.strip():
                intent = "document"
                if self._lc_available:
                    try:
                        from langchain_core.messages import HumanMessage, SystemMessage
                        doc_excerpt = document_text[:4000]
                        begin = f'Bắt đầu bằng "Dạ {addr},".' if addr else 'Bắt đầu bằng "Dạ,".'
                        prompt = f"""Tài liệu được gửi:\n\n[TÀI LIỆU]\n{doc_excerpt}\n[KẾT THÚC TÀI LIỆU]\n\nCâu hỏi: {query}\nHãy trả lời dựa trên tài liệu. {begin}"""
                        for chunk in self._llm.stream([
                            SystemMessage(content=AGENT_SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ]):
                            delta = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            if delta:
                                yield delta
                    except Exception as e:
                        logger.error(f"❌ Document stream error: {e}")
                        addr_or_neutral = addr.title() if addr else "thầy/cô"
                        yield f"{_greeting(addr)} em gặp khó khăn kỹ thuật. {addr_or_neutral} thử lại sau ạ. 🎓"
                else:
                    addr_or_neutral = addr.title() if addr else "thầy/cô"
                    yield f"{_greeting(addr)} em đã xem tài liệu nhưng LLM chưa sẵn sàng. {addr_or_neutral} thử lại sau ạ. 🎓"
                yield {"done": True, "reference_links": [], "qa_source": None, "intent": intent}
                return

            # ── Context & Intent ──────────────────────────────────────────────
            context_summary = self._memory.get_context_summary(session_id)
            intent = self._classify_intent(query, context_summary, jwt_token)
            logger.info(f"🌊 [Stream Intent] = {intent}")

            response_text_parts = []

            if intent == 'social':
                text = self._handle_social(query, addr)
                yield text
                response_text_parts.append(text)

            elif intent == 'personal_info':
                text = self._handle_personal_info(query, jwt_token, session_id, addr)
                yield text
                response_text_parts.append(text)

            elif intent == 'schedule':
                # Schedule: lấy dữ liệu từ tool rồi format (không cần LLM thường)
                tool_text = self._tool_registry.get_schedule_data(query, jwt_token)
                if "[LỖI]" in tool_text or "Không tìm thấy" in tool_text:
                    text = f"{_greeting(addr)} {tool_text}. 🎓"
                    yield text
                    response_text_parts.append(text)
                else:
                    text = self._format_schedule_response(tool_text, query, addr, jwt_token)
                    yield text
                    response_text_parts.append(text)

            else:  # knowledge
                # Kiểm tra câu mơ hồ
                if self._should_ask_clarification(query):
                    text = self._ask_clarification(addr)
                    yield text
                    response_text_parts.append(text)
                else:
                    kb_result = self._tool_registry.get_knowledge_base_answer(query)
                    tool_text = kb_result.get('text', '')
                    reference_links = kb_result.get('reference_links', [])
                    qa_source = kb_result.get('qa_source', None)

                    if "[LỖI]" in tool_text or "Không tìm thấy" in tool_text:
                        addr_or_neutral = addr.title() if addr else "thầy/cô"
                        text = f"{_greeting(addr)} em chưa có thông tin về vấn đề này trong CSDL BDU. {addr_or_neutral} có thể liên hệ phòng ban liên quan ạ. 🎓"
                        yield text
                        response_text_parts.append(text)
                    else:
                        has_cao   = "[Độ tin cậy: CAO"        in tool_text
                        has_trung = "[Độ tin cậy: TRUNG BÌNH" in tool_text
                        has_thap  = "[Độ tin cậy: THẤP"       in tool_text

                        import re as _re
                        m = _re.search(r'Thông tin:\s*(.*?)\s*(?=\[Độ tin cậy:|$)', tool_text, _re.DOTALL)
                        answer_line = m.group(1).strip() if m else ""

                        if has_cao and answer_line:
                            # CAO → bypass LLM, yield ngay
                            text = self._format_kb_response(answer_line, addr)
                            yield text
                            response_text_parts.append(text)
                        elif has_trung and answer_line and self._lc_available:
                            # TRUNG BÌNH → stream LLM
                            for chunk in self._synthesize_with_llm_stream(query, tool_text, context_summary, addr):
                                yield chunk
                                if isinstance(chunk, str):
                                    response_text_parts.append(chunk)
                        elif has_thap:
                            text = self._handle_low_confidence(tool_text, query, addr)
                            yield text
                            response_text_parts.append(text)
                        elif self._lc_available:
                            for chunk in self._synthesize_with_llm_stream(query, tool_text, context_summary, addr):
                                yield chunk
                                if isinstance(chunk, str):
                                    response_text_parts.append(chunk)
                        else:
                            text = self._format_fallback_response(tool_text, addr)
                            yield text
                            response_text_parts.append(text)

            # ── Cập nhật memory ──────────────────────────────────────────────
            full_response = "".join(response_text_parts)
            if full_response:
                self._memory.add_turn(session_id, query, full_response)

        except Exception as e:
            logger.error(f"❌ [run_stream] Error: {e}")
            text = f"{_greeting(addr)} em gặp sự cố kỹ thuật. Vui lòng thử lại ạ. 🎓"
            yield text

        elapsed = time.time() - start
        logger.info(f"✅ [run_stream] Done in {elapsed:.2f}s | intent={intent}")
        yield {"done": True, "reference_links": reference_links, "qa_source": qa_source, "intent": intent}

    # ── Memory Public API ────────────────────────────────────────────────────
    def get_memory(self) -> EnhancedSessionMemory:
        return self._memory

    def clear_session(self, session_id: str):
        self._memory.clear_session(session_id)

    def clear_all_sessions(self):
        self._memory.clear_all()

    def get_system_status(self) -> dict:
        return {
            "langchain_available": self._lc_available,
            "llm_model": "qwen2.5:7b" if self._lc_available else "unavailable",
            "memory_sessions": len(self._memory._buffer),
            "tools": ["search_knowledge_base", "get_lecturer_schedule"],
            "architecture": "bdu_langchain_agent_v2",
        }
