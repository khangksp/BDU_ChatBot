import logging
import json
import re
import time
import requests

logger = logging.getLogger(__name__)

# ============================================================
# SYSTEM PROMPT BẮT BUỘC - TIẾNG VIỆT TUYỆT ĐỐI
# ============================================================
SYSTEM_PROMPT_VI = """Bạn là trợ lý AI chính thức của Trường Đại học Bình Dương (BDU).

QUY TẮC BẮT BUỘC (TUYỆT ĐỐI KHÔNG VI PHẠM):
1. NGÔN NGỮ: Chỉ dùng TIẾNG VIỆT. Tuyệt đối KHÔNG dùng tiếng Trung, tiếng Anh hay bất kỳ ngôn ngữ nào khác.
2. DỮ LIỆU: Chỉ trả lời dựa trên thông tin trong phần [DỮ LIỆU], không được bịa hoặc suy diễn.
3. TRUNG THỰC: Nếu [DỮ LIỆU] không có thông tin liên quan → trả lời: "Dạ, em chưa có thông tin về vấn đề này."
4. NGẮN GỌN: Trả lời vào trọng tâm, lịch sự, không lan man.
5. XƯNG HÔ: Xưng "em". Gọi người dùng là "thầy/cô" nếu biết giới tính, không thêm chức danh vào cuối câu trả lời.
6. CHỮ KÝ: TUYỆT ĐỐI KHÔNG thêm tên người, chức danh, chữ ký hay footer bất kỳ vào cuối câu trả lời. Kết thúc ngay sau nội dung câu trả lời.
"""

SYSTEM_PROMPT_SOCIAL = """Bạn là trợ lý AI BDU. Hãy trả lời thân thiện, ngắn gọn bằng TIẾNG VIỆT."""

# ============================================================
# HELPER FUNCTIONS - XỬ LÝ XƯNG HÔ
# ============================================================
def _get_address(user_context: dict) -> str:
    """Lấy cách xưng hô phù hợp từ thông tin giảng viên.
    Trả về '' nếu không rõ giới tính (không xưng 'giảng viên').
    """
    if not user_context:
        return ""
    full_name = user_context.get("full_name", "")
    gender = str(user_context.get("gender", "other")).lower()
    salutation = ""
    if gender in ["male", "nam", "0"]:
        salutation = "thầy"
    elif gender in ["female", "nữ", "1"]:
        salutation = "cô"
    # Nếu gender='other' hoặc không rõ → salutation = "" (không xưng "giảng viên")
    if full_name and salutation:
        last = full_name.strip().split()[-1]
        return f"{salutation} {last}"
    if salutation:
        return salutation
    # Không rõ giới tính: trả về rỗng
    return ""


def _greeting(addr: str) -> str:
    """Tạo câu mở đầu: 'Dạ {addr},' hoặc 'Dạ,' nếu addr rỗng."""
    return f"Dạ {addr}," if addr else "Dạ,"


def _closing(addr: str, emoji: str = "🎓") -> str:
    """Tạo câu kết thúc hỏi thêm, không xưng hô nếu addr rỗng."""
    if addr:
        return f"{addr.title()} cần em hỗ trợ thêm gì không ạ? {emoji}"
    return f"Cần em hỗ trợ thêm gì không ạ? {emoji}"


# ============================================================
# CÁC TEMPLATE CỐ ĐỊNH (không cần gọi LLM)
# ============================================================
TEMPLATES = {
    "dont_know": lambda addr: f"{_greeting(addr)} em chưa có thông tin về vấn đề này. {'Thầy/cô' if not addr else addr.title()} có thể liên hệ phòng ban liên quan để được hỗ trợ ạ. 🎓",
    "clarification": lambda addr: f"{_greeting(addr)} để em hỗ trợ chính xác nhất, {'thầy/cô' if not addr else addr.title()} có thể mô tả rõ hơn về vấn đề không ạ? 🎓",
    "out_of_scope": lambda addr: f"{_greeting(addr)} em chỉ hỗ trợ các vấn đề liên quan đến công việc và quy định tại BDU thôi ạ. 🎓",
    "auth_required": lambda addr: f"{_greeting(addr)} thông tin này yêu cầu đăng nhập ạ. {'Thầy/cô' if not addr else addr.title()} vui lòng đăng nhập vào ứng dụng giảng viên để em có thể hỗ trợ. 🎓",
    "error": lambda addr: f"{_greeting(addr)} em gặp sự cố kỹ thuật, {'thầy/cô' if not addr else addr.title()} thử lại sau ít phút nhé ạ. 🎓",
}

def _fill_template(template_key: str, addr: str) -> str:
    tpl_fn = TEMPLATES.get(template_key, TEMPLATES["error"])
    return tpl_fn(addr)


# ============================================================
# LỚP CHÍNH
# ============================================================
class LocalQwenGenerator:
    """
    Lớp sinh câu trả lời cục bộ với Ollama qwen2.5:7b.
    
    Chiến lược phân nhánh:
    - FAQ / QA cao điểm → Trả thẳng câu trả lời từ CSDL, KHÔNG gọi LLM.
    - API giảng viên   → Gọi Qwen2.5 với dữ liệu API được nhúng vào prompt, ràng buộc chặt.
    - Document context → Gọi Qwen2.5 với nội dung tài liệu.
    - Các trường hợp còn lại → Template cứng (clarification, dont_know, social, v.v.).
    """

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "qwen2.5:7b"
        self.timeout = 60  # giây
        self._user_context_cache: dict = {}  # cache thông tin giảng viên
        logger.info("✅ LocalQwenGenerator (Ollama qwen2.5:7b) initialized.")

    # ----------------------------------------------------------
    # Public: set_user_context / get_user_context / clear
    # ----------------------------------------------------------
    def set_user_context(self, session_id: str, user_context: dict):
        self._user_context_cache[session_id] = user_context
        logger.info(f"👤 User context set for session {session_id}: {user_context.get('full_name', '?')}")

    def get_user_context(self, session_id: str) -> dict:
        return self._user_context_cache.get(session_id, {})

    def clear_user_context(self, session_id: str = None):
        if session_id:
            self._user_context_cache.pop(session_id, None)
        else:
            self._user_context_cache.clear()

    def get_system_status(self) -> dict:
        """Kiểm tra Ollama có đang chạy không."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            available = resp.status_code == 200
        except Exception:
            available = False
        return {
            "gemini_api_available": False,
            "llm_backend": "ollama_local",
            "model": self.model,
            "ollama_available": available,
        }

    # ----------------------------------------------------------
    # Public: generate_response (entry point từ services.py)
    # ----------------------------------------------------------
    def generate_response(self, query: str, context: dict = None, 
                          intent_info: dict = None, entities: dict = None,
                          session_id: str = None) -> dict:
        """
        Hàm chính, tương thích với interface cũ của GeminiResponseGenerator.
        Trả về dict chứa key 'response'.
        """
        start = time.time()
        addr = _get_address(self._user_context_cache.get(session_id, {}))
        ctx = context or {}
        instruction = ctx.get("instruction", "")
        mode = ctx.get("mode", "")

        try:
            # ── Chế độ chat xã giao ──────────────────────────────────
            if mode == "chat_only":
                text = self._call_ollama(
                    system=SYSTEM_PROMPT_SOCIAL,
                    user=query,
                )
                greeting = _greeting(addr)
                return self._wrap(text or f"{greeting} em có thể giúp gì không ạ?", "social_chat", start)

            # ── Chế độ kiến thức chung (low confidence fallback) ────
            if mode == "general_knowledge":
                # Không bịa thông tin BDU, chỉ trả lời nếu câu hỏi rõ và không liên quan DB
                text = _fill_template("dont_know", addr)
                return self._wrap(text, "general_knowledge_template", start)

            # ── FAQ: câu trả lời trực tiếp từ CSDL → KHÔNG gọi LLM ──
            if instruction in ("direct_answer_lecturer", "enhance_answer_lecturer"):
                db_answer = ctx.get("db_answer", "").strip()
                if db_answer:
                    text = self._format_qa_answer(db_answer, addr)
                    return self._wrap(text, "direct_qa_no_llm", start)
                # fallback nếu db_answer trống
                return self._wrap(_fill_template("dont_know", addr), "direct_qa_empty", start)

            # ── Tài liệu được chia sẻ (OCR / upload) ────────────────
            if instruction in ("answer_from_document", "process_document_context"):
                doc_text = ctx.get("document_text", "")
                if not doc_text.strip():
                    return self._wrap(_fill_template("dont_know", addr), "document_empty", start)
                prompt = self._build_document_prompt(query, doc_text, addr)
                text = self._call_ollama(system=SYSTEM_PROMPT_VI, user=prompt)
                return self._wrap(text or _fill_template("dont_know", addr), "document_qwen", start)

            # ── API giảng viên (lịch dạy, thông tin cá nhân) ────────
            if instruction in ("external_api_lecturer", "process_external_api_data"):
                api_data = ctx.get("api_data") or ctx.get("schedule_data") or ctx
                prompt = self._build_api_prompt(query, api_data, addr)
                text = self._call_ollama(system=SYSTEM_PROMPT_VI, user=prompt)
                return self._wrap(text or _fill_template("error", addr), "api_qwen", start)

            # ── Làm rõ câu hỏi ───────────────────────────────────────
            if instruction in ("clarification_needed", "smart_clarification_needed",
                                "authentication_required"):
                template_key = "auth_required" if instruction == "authentication_required" else "clarification"
                return self._wrap(_fill_template(template_key, addr), "template_clarification", start)

            # ── Không biết ───────────────────────────────────────────
            if instruction == "dont_know_lecturer":
                return self._wrap(_fill_template("dont_know", addr), "template_dont_know", start)

            # ── Fallback mặc định ────────────────────────────────────
            db_answer = ctx.get("db_answer", "").strip()
            if db_answer:
                return self._wrap(self._format_qa_answer(db_answer, addr), "fallback_qa", start)
            return self._wrap(_fill_template("dont_know", addr), "fallback_template", start)

        except Exception as e:
            logger.error(f"❌ LocalQwenGenerator.generate_response error: {e}")
            return self._wrap(_fill_template("error", addr), "exception_fallback", start)

    # ----------------------------------------------------------
    # Ollama HTTP call
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Streaming: generate_response_stream + _call_ollama_stream
    # ----------------------------------------------------------
    def generate_response_stream(self, query: str, context: dict = None,
                                  intent_info: dict = None, entities: dict = None,
                                  session_id: str = None):
        """
        Phiên bản streaming của generate_response.
        Yield từng text chunk (str). Chunk cuối cùng là dict metadata.
        Frontend đọc từng chunk và hiển thị ngay lập tức.
        """
        start = time.time()
        addr = _get_address(self._user_context_cache.get(session_id, {}))
        ctx = context or {}
        instruction = ctx.get("instruction", "")
        mode = ctx.get("mode", "")

        try:
            # ── Chat xã giao ──
            if mode == "chat_only":
                yield from self._call_ollama_stream(SYSTEM_PROMPT_SOCIAL, query)
                yield {"done": True, "method": "social_chat_stream", "generation_time": time.time() - start}
                return

            # ── General knowledge → template ngay, không cần stream ──
            if mode == "general_knowledge":
                text = _fill_template("dont_know", addr)
                yield text
                yield {"done": True, "method": "general_knowledge_template", "generation_time": time.time() - start}
                return

            # ── FAQ: trả thẳng, không cần stream (đã nhanh) ──
            if instruction in ("direct_answer_lecturer", "enhance_answer_lecturer"):
                db_answer = ctx.get("db_answer", "").strip()
                if db_answer:
                    text = self._format_qa_answer(db_answer, addr)
                else:
                    text = _fill_template("dont_know", addr)
                yield text
                yield {"done": True, "method": "direct_qa_stream", "generation_time": time.time() - start}
                return

            # ── Tài liệu ──
            if instruction in ("answer_from_document", "process_document_context"):
                doc_text = ctx.get("document_text", "")
                if not doc_text.strip():
                    yield _fill_template("dont_know", addr)
                    yield {"done": True, "method": "document_empty", "generation_time": time.time() - start}
                    return
                prompt = self._build_document_prompt(query, doc_text, addr)
                yield from self._call_ollama_stream(SYSTEM_PROMPT_VI, prompt)
                yield {"done": True, "method": "document_qwen_stream", "generation_time": time.time() - start}
                return

            # ── API giảng viên ──
            if instruction in ("external_api_lecturer", "process_external_api_data"):
                api_data = ctx.get("api_data") or ctx.get("schedule_data") or ctx
                prompt = self._build_api_prompt(query, api_data, addr)
                yield from self._call_ollama_stream(SYSTEM_PROMPT_VI, prompt)
                yield {"done": True, "method": "api_qwen_stream", "generation_time": time.time() - start}
                return

            # ── Templates ──
            if instruction in ("clarification_needed", "smart_clarification_needed",
                               "authentication_required"):
                template_key = "auth_required" if instruction == "authentication_required" else "clarification"
                yield _fill_template(template_key, addr)
                yield {"done": True, "method": "template_stream", "generation_time": time.time() - start}
                return

            if instruction == "dont_know_lecturer":
                yield _fill_template("dont_know", addr)
                yield {"done": True, "method": "template_dont_know_stream", "generation_time": time.time() - start}
                return

            # ── Fallback ──
            db_answer = ctx.get("db_answer", "").strip()
            if db_answer:
                yield self._format_qa_answer(db_answer, addr)
            else:
                yield _fill_template("dont_know", addr)
            yield {"done": True, "method": "fallback_stream", "generation_time": time.time() - start}

        except Exception as e:
            logger.error(f"❌ generate_response_stream error: {e}")
            yield _fill_template("error", addr)
            yield {"done": True, "method": "exception_stream", "generation_time": time.time() - start}

    def _call_ollama_stream(self, system: str, user: str):
        """
        Generator: gọi Ollama với stream=True, yield từng text delta.
        Tự động lọc tiếng Trung và strip signature ở chunk cuối.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "options": {
                    "temperature": 0,
                    "top_p": 0.9,
                    "num_predict": 512,
                },
            }
            buffer = []  # thu thập toàn bộ để hậu xử lý cuối
            with requests.post(
                self.ollama_url, json=payload,
                timeout=self.timeout, stream=True
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"Ollama stream error {resp.status_code}")
                    return
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        buffer.append(delta)
                        yield delta
                    if data.get("done"):
                        break
            # Hậu xử lý: nếu toàn bộ text có tiếng Trung → retry không-stream
            full_text = "".join(buffer)
            if self._contains_chinese(full_text):
                logger.warning("⚠️ Chinese detected in stream, retrying...")
                retry_text = self._retry_Vietnamese_only(system, user)
                if retry_text:
                    retry_text = self._strip_llm_signature(retry_text)
                    yield "\n[RETRY]" + retry_text  # frontend nên replace toàn bộ
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama stream request timed out.")
        except Exception as e:
            logger.error(f"❌ Ollama stream failed: {e}")

    def _call_ollama(self, system: str, user: str) -> str | None:
        """Gọi Ollama REST API, trả về nội dung text hoặc None nếu lỗi."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": 0,     # Hoàn toàn deterministic - cùng câu hỏi → cùng câu trả lời
                    "top_p": 0.9,
                    "num_predict": 512,   # Giới hạn độ dài để phản hồi nhanh
                },
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("message", {}).get("content", "").strip()
                # Phát hiện và loại bỏ tiếng Trung nếu lọt qua (thêm phòng ngừa)
                if self._contains_chinese(text):
                    logger.warning("⚠️ Chinese detected in response, retrying with stricter prompt...")
                    text = self._retry_Vietnamese_only(system, user)
                # Post-process: loại bỏ chữ ký/footer hallucinate
                text = self._strip_llm_signature(text)
                return text
            else:
                logger.error(f"Ollama API error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timed out.")
            return None
        except Exception as e:
            logger.error(f"❌ Ollama call failed: {e}")
            return None

    def _retry_Vietnamese_only(self, system: str, user: str) -> str | None:
        """Retry với system prompt cứng hơn khi phát hiện tiếng Trung."""
        strict_system = system + "\n\n⚠️ CẢNH BÁO: Phản hồi trước bị phát hiện có tiếng Trung. BẮT BUỘC chỉ dùng tiếng Việt."
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": strict_system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            }
            resp = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Kiểm tra xem text có chứa ký tự tiếng Trung không."""
        return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))

    @staticmethod
    def _strip_llm_signature(text: str) -> str:
        """
        Loại bỏ chữ ký / footer mà LLM tự thêm vào cuối câu trả lời.
        Ví dụ LLM hay thêm: 'Trần Thị Thu Trang Trợ lý BDU',
        'Trợ lý BDU', tên người + chức danh, ...
        """
        if not text:
            return text
        # Pattern 1: dòng cuối chứa "Trợ lý BDU" (có hoặc không có tên trước)
        text = re.sub(r'\n[^\n]*[Tt]r\u1ee3\s*l\u00fd\s*BDU\s*$', '', text, flags=re.IGNORECASE).rstrip()
        # Pattern 2: "Trợ lý BDU" xuất hiện bất kỳ đâu ở cuối (inline)
        text = re.sub(r'\s+[^\n]*[Tt]r\u1ee3\s*l\u00fd\s*BDU\s*$', '', text, flags=re.IGNORECASE).rstrip()
        # Pattern 3: Dòng trống + dòng cuối chỉ có tên người (2-4 từ, viết hoa đầu)
        text = re.sub(r'\n\s*\n([A-Z\u00C0-\u1EF9][a-z\u00C0-\u1EF9]+(?:\s+[A-Z\u00C0-\u1EF9][a-z\u00C0-\u1EF9]+){1,3})\s*$', '', text).rstrip()
        return text

    # ----------------------------------------------------------
    # Prompt builders
    # ----------------------------------------------------------
    def _build_api_prompt(self, query: str, api_data, addr: str) -> str:
        """
        Xây dựng prompt cho câu hỏi lịch dạy.
        Format schedule thành text rõ ràng để Qwen không hallucinate.
        """
        # Lấy thông tin cơ bản
        lecturer_name = ''
        week_start = ''
        daily_schedule = {}

        if isinstance(api_data, dict):
            info = api_data.get('lecturer_info', {})
            lecturer_name = info.get('ten_giang_vien', '') if isinstance(info, dict) else ''
            week_start = api_data.get('week_start', '')
            daily_schedule = api_data.get('daily_schedule', {})

        # Format lịch dạy thành văn bản dễ đọc
        if daily_schedule:
            lines = []
            for date_str in sorted(daily_schedule.keys()):
                entries = daily_schedule[date_str]
                if not entries:
                    continue
                lines.append(f"\n📅 {date_str}:")
                for e in entries:
                    mon = e.get('ten_mon_hoc', '?')
                    tiet_bd = e.get('tiet_bat_dau', '?')
                    # tiet_ket_thuc không có trong API → tính từ tiet_bat_dau + so_tiet - 1
                    so_tiet = e.get('so_tiet', 0)
                    if tiet_bd != '?' and so_tiet:
                        tiet_kt = tiet_bd + so_tiet - 1
                    else:
                        tiet_kt = e.get('tiet_ket_thuc', tiet_bd)  # fallback nếu API sau có field này
                    # phong → ma_phong, ten_lop → nhom_hoc (theo field names API thực tế)
                    phong = e.get('ma_phong') or e.get('phong', '')
                    nhom = e.get('nhom_hoc') or e.get('ma_lop') or e.get('ten_lop', '')
                    line = f"  - {mon} | Tiết {tiet_bd}–{tiet_kt}"
                    if phong:
                        line += f" | Phòng: {phong}"
                    if nhom:
                        line += f" | Nhóm: {nhom}"
                    lines.append(line)
            schedule_text = '\n'.join(lines) if lines else '(không có tiết nào)'
        else:
            schedule_text = '(không có lịch dạy)'

        week_label = f"Tuần bắt đầu {week_start}" if week_start else "Tuần hiện tại"
        # Mở đầu prompt: không dùng "giảng viên" cứng nhắc
        name_display = lecturer_name or addr or "người dùng"
        addr_display = addr if addr else ""
        begin_rule = f'Bắt đầu bằng "Dạ {addr_display},".' if addr_display else 'Bắt đầu bằng "Dạ,".'

        return f"""Bạn đang hỗ trợ người dùng BDU tên {name_display}.

DƯỚI ĐÂY LÀ TẤT CẢ LỊCH DẠY CÓ TRONG HỆ THỐNG ({week_label}):
{schedule_text}

CÂU HỎI: {query}

QUY TẮC BẮT BUỘC KHI TRẢ LỜI:
1. CHỈ đề cập đến các ngày và tiết học CÓ TRONG DANH SÁCH TRÊN.
2. TUYỆT ĐỐI KHÔNG đề cập, nhận xét, hay nói gì về các ngày KHÔNG CÓ trong danh sách.
3. Nếu danh sách trống → nói "tuần này không có lịch dạy".
4. Trình bày theo từng ngày, ngắn gọn, rõ ràng.
5. {begin_rule} Chỉ dùng tiếng Việt.
6. TUYỆT ĐỐI KHÔNG thêm tên người, chức danh, chữ ký hay bất kỳ footer nào vào cuối câu trả lời.

Trả lời:"""


    def _build_document_prompt(self, query: str, doc_text: str, addr: str) -> str:
        """Prompt cho câu hỏi dựa trên tài liệu được upload."""
        doc_excerpt = doc_text[:4000] if len(doc_text) > 4000 else doc_text
        begin = f"Bắt đầu bằng \"Dạ {addr},\"." if addr else "Bắt đầu bằng \"Dạ,\"."
        return f"""Dưới đây là nội dung tài liệu được gửi:

[TÀI LIỆU]
{doc_excerpt}
[KẾT THÚC TÀI LIỆU]

Câu hỏi: {query}

Hãy trả lời dựa trên nội dung tài liệu trên. Nếu tài liệu không đề cập đến thông tin được hỏi, hãy nói rõ. {begin}

Trả lời:"""

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _format_qa_answer(db_answer: str, addr: str) -> str:
        """Định dạng câu trả lời từ CSDL thành văn phong lịch sự."""
        clean = db_answer.strip()
        # Xoá lời chào có sẵn trong câu trả lời DB để tránh lặp
        clean = re.sub(r'^(dạ\s+(thầy|cô|giảng viên)[^,]*,?\s*)', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'^(xin\s+chào|chào)[^.!?\n]*[.!?\n]\s*', '', clean, flags=re.IGNORECASE)
        if clean and not clean[0].isupper():
            clean = clean[0].upper() + clean[1:]
        # Mở đầu
        greeting = _greeting(addr)
        result = f"{greeting} {clean}"
        if not result.rstrip().endswith(('.', '!', '?')):
            result += '.'
        # Kết thúc
        result += f' {_closing(addr, "🎓")}'
        return result

    @staticmethod
    def _wrap(text: str, method: str, start: float) -> dict:
        """Đóng gói kết quả trả về với cấu trúc tương thích interface cũ."""
        return {
            "response": text,
            "method": method,
            "generation_time": time.time() - start,
            "model": "qwen2.5:7b",
            "llm_used": method not in ("direct_qa_no_llm", "template_clarification",
                                        "template_dont_know", "template_auth",
                                        "fallback_template", "general_knowledge_template"),
        }


# ─── Alias để tương thích ngược với code cũ ─────────────────
GeminiResponseGenerator = LocalQwenGenerator
