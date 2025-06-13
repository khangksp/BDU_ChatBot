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

# Global service instance
speech_service = SpeechToTextService()