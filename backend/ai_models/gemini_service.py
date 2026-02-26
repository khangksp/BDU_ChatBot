import logging
from .llm.core import LocalQwenGenerator, GeminiResponseGenerator

logger = logging.getLogger(__name__)

# Instance global dùng chung toàn bộ ứng dụng
gemini_response_generator = LocalQwenGenerator()

logger.info("🚀 LLM Service loaded: LocalQwenGenerator (Ollama qwen2.5:7b)")