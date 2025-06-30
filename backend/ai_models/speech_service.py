# ai_models/speech_service.py

import os
import tempfile
import torch
import logging
import time
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path

# Try to import faster_whisper, fallback if not available
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: faster_whisper not installed. Speech-to-text will use fallback mode.")

# ✅ THÊM CÁC IMPORT MỚI CHO TTS
import uuid
import base64
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not installed. Text-to-speech will be unavailable.")

logger = logging.getLogger(__name__)

class SpeechToTextService:
    """
    Vietnamese Speech-to-Text Service using Faster Whisper
    Optimized for Django backend integration
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.compute_type = None
        
        # Model settings
        self.model_size = "large-v3"  # Best quality for Vietnamese
        self.language = "vi"
        
        # Audio settings
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
        self.max_file_size_mb = 25  # 25MB limit
        
        # Performance settings
        self.beam_size = 5
        self.temperature = 0.0
        
        # Initialize model if available
        if WHISPER_AVAILABLE:
            try:
                self._setup_device()
                self._load_model()
                logger.info("✅ Speech-to-Text service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Speech-to-Text: {e}")
                self.model = None
        else:
            logger.warning("Speech-to-Text service running in fallback mode")
    
    def _setup_device(self):
        """Setup optimal device and compute type"""
        if torch.cuda.is_available():
            try:
                torch.cuda.get_device_name(0)
                self.device = "cuda"
                self.compute_type = "float16"
                logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info("⚡ Using GPU acceleration (float16)")
            except Exception as e:
                logger.warning(f"GPU setup failed: {e}")
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            logger.info("🔄 Using CPU optimization (int8)")
    
    def _load_model(self):
        """Load Whisper model with optimal settings"""
        if not WHISPER_AVAILABLE:
            raise ImportError("faster_whisper not available")
        
        logger.info(f"🚀 Loading Whisper model '{self.model_size}' on {self.device}...")
        
        # Calculate optimal worker settings
        cpu_count = os.cpu_count() or 4
        num_workers = max(1, cpu_count // 2)
        cpu_threads = max(1, cpu_count // 2)
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=num_workers,
            cpu_threads=cpu_threads,
            download_root=None  # Use default cache
        )
        
        logger.info("✅ Whisper model loaded successfully")
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return WHISPER_AVAILABLE and self.model is not None
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Validate audio file before processing"""
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return {"valid": False, "error": "File not found"}
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                return {
                    "valid": False, 
                    "error": f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
                }
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return {
                    "valid": False,
                    "error": f"File too large ({file_size_mb:.1f}MB). Max: {self.max_file_size_mb}MB"
                }
            
            return {"valid": True, "size_mb": file_size_mb}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def transcribe_audio(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional options (language, beam_size, etc.)
        
        Returns:
            Dict with transcription results
        """
        start_time = time.time()
        
        # Check service availability
        if not self.is_available():
            return {
                "success": False,
                "error": "Speech-to-Text service not available",
                "text": "",
                "processing_time": 0,
                "method": "unavailable"
            }
        
        # Validate file
        validation = self.validate_audio_file(file_path)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["error"],
                "text": "",
                "processing_time": time.time() - start_time,
                "method": "validation_failed"
            }
        
        try:
            # Extract options
            language = kwargs.get('language', self.language)
            beam_size = kwargs.get('beam_size', self.beam_size)
            temperature = kwargs.get('temperature', self.temperature)
            
            logger.info(f"🎤 Transcribing audio file: {file_path}")
            logger.info(f"📊 File size: {validation['size_mb']:.1f}MB")
            
            # Transcribe with VAD filter for better results
            segments, info = self.model.transcribe(
                str(file_path),
                language=language,
                beam_size=beam_size,
                temperature=temperature,
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    max_speech_duration_s=30
                ),
                word_timestamps=False  # Disable for faster processing
            )
            
            # Extract text from segments
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text.strip())
            
            final_text = " ".join(text_segments).strip()
            
            # Validate result
            if not final_text or len(final_text) < 2:
                return {
                    "success": False,
                    "error": "No valid speech detected in audio",
                    "text": "",
                    "processing_time": time.time() - start_time,
                    "method": "no_speech_detected"
                }
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Transcription completed in {processing_time:.2f}s")
            logger.info(f"📝 Text: {final_text[:100]}...")
            
            return {
                "success": True,
                "text": final_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "processing_time": processing_time,
                "method": "faster_whisper",
                "model": self.model_size,
                "device": self.device
            }
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "processing_time": time.time() - start_time,
                "method": "transcription_error"
            }
    
    def transcribe_audio_data(self, audio_data: bytes, format: str = "wav") -> Dict[str, Any]:
        """
        Transcribe audio data directly from memory
        
        Args:
            audio_data: Raw audio bytes
            format: Audio format (wav, mp3, etc.)
        
        Returns:
            Dict with transcription results
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.flush()
            
            try:
                # Transcribe temporary file
                result = self.transcribe_audio(tmp_file.name)
                return result
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get service status information"""
        return {
            "available": self.is_available(),
            "whisper_installed": WHISPER_AVAILABLE,
            "model_loaded": self.model is not None,
            "model_size": self.model_size if self.model else None,
            "device": self.device,
            "compute_type": self.compute_type,
            "supported_formats": self.supported_formats,
            "max_file_size_mb": self.max_file_size_mb
        }
    
    def __del__(self):
        """Cleanup GPU memory if using CUDA"""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass


# ✅ THÊM CLASS MỚI CHO TTS
class TextToSpeechService:
    """
    Service để chuyển văn bản thành giọng nói (TTS) sử dụng gTTS.
    Tối ưu cho việc trả về dữ liệu audio qua API.
    """
    
    def __init__(self):
        self.is_available = GTTS_AVAILABLE
        self.default_language = "vi"
        self.default_speed = False  # False = normal speed, True = slow
        
        if self.is_available:
            logger.info("✅ Text-to-Speech service (gTTS) initialized successfully")
        else:
            logger.warning("⚠️ gTTS library not installed. TTS service is unavailable.")
    
    def text_to_audio_base64(self, text: str, language: str = None, slow: bool = None) -> Optional[str]:
        """
        Chuyển văn bản thành audio MP3 và trả về dưới dạng chuỗi base64.
        
        Args:
            text: Văn bản cần chuyển đổi
            language: Mã ngôn ngữ (mặc định: 'vi')
            slow: Tốc độ nói (mặc định: False - tốc độ bình thường)
            
        Returns:
            Một chuỗi base64 của file audio MP3, hoặc None nếu có lỗi.
        """
        if not self.is_available:
            logger.error("❌ gTTS service not available")
            return None
            
        if not text or not text.strip():
            logger.error("❌ Text is empty or None")
            return None
        
        # Use default values if not provided
        language = language or self.default_language
        # slow = slow if slow is not None else self.default_speed
        slow = False
        
        output_path = ""
        
        try:
            # Clean and prepare text
            clean_text = self._clean_text_for_tts(text)
            
            if not clean_text:
                logger.error("❌ No valid text after cleaning")
                return None
            
            logger.info(f"🔊 Generating TTS for text: '{clean_text[:50]}...' (language: {language}, slow: {slow})")
            
            # Create gTTS object
            tts_object = gTTS(text=clean_text, lang=language, slow=slow)
            
            # Sử dụng tempfile để quản lý file tạm một cách an toàn
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                output_path = tmp_file.name
            
            # Save audio to temporary file
            tts_object.save(output_path)
            
            # Verify file was created and has content
            if not os.path.exists(output_path):
                logger.error("❌ TTS audio file was not created")
                return None
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                logger.error("❌ TTS audio file is empty")
                return None
            
            # Đọc nội dung file audio vừa tạo dưới dạng bytes
            with open(output_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Mã hóa sang base64 để gửi qua JSON
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            logger.info(f"✅ Generated TTS audio ({len(audio_bytes)} bytes), encoded to base64 ({len(audio_base64)} chars)")
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"❌ Error generating TTS audio: {e}")
            return None
        
        finally:
            # Luôn đảm bảo file tạm được xóa
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                    logger.debug(f"🗑️ Cleaned up temporary TTS file: {output_path}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not delete temp TTS file {output_path}: {cleanup_error}")
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        Làm sạch văn bản để phù hợp với TTS.
        
        Args:
            text: Văn bản gốc
            
        Returns:
            Văn bản đã được làm sạch
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        clean_text = text.strip()
        
        # Replace multiple spaces with single space
        import re
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove markdown formatting that might interfere with TTS
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Remove bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Remove italic
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # Remove code
        
        # Remove URLs (they don't sound good when spoken)
        clean_text = re.sub(r'https?://[^\s<>]+', '', clean_text)
        
        # Remove email addresses (they don't sound good when spoken)
        clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', clean_text)
        
        # Clean up any resulting multiple spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Limit length to prevent excessively long audio
        max_length = 1000  # Adjust as needed
        if len(clean_text) > max_length:
            # Try to cut at a sentence boundary
            if '. ' in clean_text[:max_length]:
                cut_point = clean_text[:max_length].rfind('. ') + 1
                clean_text = clean_text[:cut_point]
            else:
                clean_text = clean_text[:max_length] + "..."
        
        return clean_text
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get TTS service status information"""
        return {
            "available": self.is_available,
            "gtts_installed": GTTS_AVAILABLE,
            "default_language": self.default_language,
            "default_speed": self.default_speed,
            "supported_languages": self._get_supported_languages() if self.is_available else []
        }
    
    def _get_supported_languages(self) -> list:
        """Get list of supported languages for TTS"""
        # Common languages supported by gTTS
        return [
            'vi',  # Vietnamese
            'en',  # English
            'zh',  # Chinese
            'ja',  # Japanese
            'ko',  # Korean
            'th',  # Thai
            'id',  # Indonesian
            'ms',  # Malay
        ]

# Global service instances
speech_service = SpeechToTextService()
# ✅ TẠO INSTANCE MỚI CHO TTS SERVICE
tts_service = TextToSpeechService()