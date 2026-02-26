import jwt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from knowledge.models import ChatHistory, UserFeedback
import traceback

# ✅ SAFE IMPORTS - with fallbacks
try:
    from ai_models.services import chatbot_ai
    if chatbot_ai is None:
        print("⚠️ WARNING: ai_models.services imported but chatbot_ai is None")
except Exception as e:
    chatbot_ai = None
    print("\n" + "!"*50)
    print("❌ CRITICAL ERROR IMPORTING CHATBOT_AI:")
    traceback.print_exc() # 👈 Dòng này sẽ in chi tiết lỗi ra màn hình console
    print("!"*50 + "\n")

try:
    # Import Class thay vì instance
    from ai_models.speech_service import SpeechToTextService, TextToSpeechService
    speech_service = SpeechToTextService() # Khởi tạo tại đây
    tts_service = TextToSpeechService()
except ImportError:
    speech_service = None
    tts_service = None

try:
    from ai_models.ocr_service import ocr_service
except ImportError:
    ocr_service = None

# 🚀 NEW: Import training module with fallback
try:
    from ai_models.train_retriever import run_training, check_gpu_availability
    TRAINING_AVAILABLE = True
except ImportError:
    run_training = None
    check_gpu_availability = None
    TRAINING_AVAILABLE = False

import uuid
import time
import logging
import json
import tempfile
import os
import base64
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from django.db import models
import threading
from datetime import datetime

from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.permissions import IsAdminUser
from django.http import StreamingHttpResponse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import queue

logger = logging.getLogger(__name__)

# ─── Thread Pool cho xử lý chatbot song song ───────────────────────────────
# Dùng stdlib ThreadPoolExecutor - không cần cài thêm thư viện
# I/O-bound workload (chờ Ollama HTTP) → GIL được giải phóng → thực sự song song
_CHATBOT_POOL_SIZE = int(os.environ.get('CHATBOT_THREAD_POOL_SIZE', 8))
_CHATBOT_THREAD_POOL = ThreadPoolExecutor(
    max_workers=_CHATBOT_POOL_SIZE,
    thread_name_prefix="chatbot_worker",
)
logger.info(f"🚀 ChatBot ThreadPool initialized: {_CHATBOT_POOL_SIZE} workers")

# 🚀 NEW: Global variable to track training status
TRAINING_STATUS = {
    'is_running': False,
    'started_at': None,
    'completed_at': None,
    'success': None,
    'error': None,
    'progress': 0,
    'output_dir': None,
    'training_examples': 0,
    'evaluation_results': None
}

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def extract_jwt_token(request):
    """
    Extract JWT token from request headers or data
    Returns: token string or None
    """
    try:
        # Method 1: Check Authorization header
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            logger.info(f"🔑 JWT token found in Authorization header")
            return token
        
        # Method 2: Check request data (for mobile apps)
        if hasattr(request, 'data') and 'token' in request.data:
            token = request.data.get('token', '').strip()
            if token:
                logger.info(f"🔑 JWT token found in request data")
                return token
        
        # Method 3: Check JSON body for token field
        if hasattr(request, 'body') and request.body:
            try:
                body_data = json.loads(request.body)
                if 'token' in body_data:
                    token = body_data.get('token', '').strip()
                    if token:
                        logger.info(f"🔑 JWT token found in JSON body")
                        return token
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Method 4: Check query parameters (less secure, mainly for testing)
        token = request.GET.get('token', '').strip()
        if token:
            logger.info(f"🔑 JWT token found in query parameters")
            return token
        
        logger.info("🔑 No JWT token found in request")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error extracting JWT token: {str(e)}")
        return None

def validate_jwt_token_format(token):
    """
    Basic validation of JWT token format
    Returns: (is_valid, error_message)
    """
    if not token:
        return False, "Token is empty"
    
    if not isinstance(token, str):
        return False, "Token must be a string"
    
    # Remove Bearer prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    # JWT should have 3 parts separated by dots
    parts = token.split('.')
    if len(parts) != 3:
        return False, f"Invalid JWT format - expected 3 parts, got {len(parts)}"
    
    # Each part should be base64 encoded (basic check)
    try:
        import base64
        for i, part in enumerate(parts[:2]):  # Don't check signature part
            # Add padding if needed
            padded = part + '=' * (4 - len(part) % 4)
            base64.b64decode(padded)
    except Exception as e:
        return False, f"Invalid base64 encoding in JWT: {str(e)}"
    
    return True, "Valid JWT format"

def auto_setup_user_context_from_jwt(session_id: str, jwt_token: str):
    """
    🚀 AUTO-SETUP user context từ JWT token để đảm bảo personal addressing
    """
    try:
        # Import external_api_service
        from ai_models.external_api_service import external_api_service
        
        # Decode JWT để lấy thông tin user
        lecturer_info = external_api_service.get_lecturer_info_from_token(jwt_token)
        
        if lecturer_info:
            # Setup user context với thông tin từ JWT
            user_context = {
                'full_name': lecturer_info.get('ten_giang_vien', ''),
                'gender': lecturer_info.get('gender', 'other'),  # Đã được convert trong external_api_service
                'ma_giang_vien': lecturer_info.get('ma_giang_vien', ''),
                'chuc_danh': lecturer_info.get('chuc_danh', ''),
                'gmail': lecturer_info.get('gmail', ''),
                'faculty_code': 'JWT_AUTO',
                'department_name': lecturer_info.get('vi_tri_viec_lam', ''),
                'preferences': {
                    'user_memory_prompt': '',  # Có thể load từ database user preferences
                    'department_priority': True
                },
                'jwt_source': True,  # Flag để biết context này từ JWT
                'auto_detected': True
            }
            
            # Set context vào response generator
            if chatbot_ai and hasattr(chatbot_ai, 'response_generator'):
                chatbot_ai.response_generator.set_user_context(session_id, user_context)
                logger.info(f"✅ JWT Auto-setup: {lecturer_info['ten_giang_vien']} ({lecturer_info['gender']}) -> session {session_id}")
                return True
            else:
                logger.warning("⚠️ Chatbot AI service not available for context setup")
                return False
            
    except Exception as e:
        logger.error(f"❌ Error auto-setup user context from JWT: {str(e)}")
        return False

# 🚀 NEW: Training functions
def run_training_background(csv_path=None, output_dir='./fine_tuned_phobert'):
    """Run training in background thread"""
    global TRAINING_STATUS
    
    try:
        TRAINING_STATUS.update({
            'is_running': True,
            'started_at': datetime.now().isoformat(),
            'progress': 0,
            'error': None,
            'success': None
        })
        
        logger.info("🚀 Starting PhoBERT fine-tuning in background...")
        
        if not TRAINING_AVAILABLE or not run_training:
            raise Exception("Training module not available")
        
        # Run the actual training
        TRAINING_STATUS['progress'] = 25
        result = run_training(csv_path, output_dir)
        TRAINING_STATUS['progress'] = 90
        
        if result and result.get('success'):
            TRAINING_STATUS.update({
                'success': True,
                'completed_at': datetime.now().isoformat(),
                'progress': 100,
                'output_dir': result.get('output_dir'),
                'training_examples': result.get('training_examples', 0),
                'evaluation_results': result.get('evaluation_results')
            })
            
            # 🚀 CRITICAL: Reload the chatbot with new model
            if chatbot_ai and hasattr(chatbot_ai, 'reload_after_qa_update'):
                logger.info("🔄 Reloading chatbot with fine-tuned model...")
                chatbot_ai.reload_after_qa_update()
                logger.info("✅ Chatbot reloaded with fine-tuned model")
            
            logger.info("✅ Background training completed successfully!")
            
        else:
            error_msg = result.get('error', 'Unknown training error') if result else 'Training function returned None'
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Background training failed: {str(e)}")
        TRAINING_STATUS.update({
            'success': False,
            'error': str(e),
            'completed_at': datetime.now().isoformat(),
            'progress': 0,
            'is_running': False
        })
    finally:
        TRAINING_STATUS['is_running'] = False

# ✅ SAFE SYSTEM STATUS FUNCTIONS
def get_safe_system_status():
    """Get system status with safe fallbacks"""
    try:
        if chatbot_ai:
            return chatbot_ai.get_system_status()
    except Exception as e:
        logger.error(f"Error getting chatbot_ai status: {e}")
    
    return {
        'ai_service': {
            'available': False,
            'error': 'AI service not available'
        },
        'status': 'limited'
    }

def get_safe_speech_status():
    """Get speech status with safe fallbacks"""
    try:
        if speech_service:
            return speech_service.get_system_status()
    except Exception as e:
        logger.error(f"Error getting speech status: {e}")
    
    return {
        'available': False,
        'error': 'Speech service not available'
    }

def get_safe_tts_status():
    """Get TTS status with safe fallbacks"""
    try:
        if tts_service:
            return tts_service.get_system_status()
    except Exception as e:
        logger.error(f"Error getting TTS status: {e}")
    
    return {
        'available': False,
        'error': 'TTS service not available'
    }
    
class HotReloadView(APIView):
    """
    🔥 API Hot Reload: Cập nhật dữ liệu từ Drive và Nạp lại RAM ngay lập tức
    Yêu cầu quyền Admin/Staff để tránh người lạ nghịch.
    """
    permission_classes = [IsAuthenticated, IsAdminUser] # Chỉ cho phép Admin

    def post(self, request):
        start_time = time.time()
        logger.info("🔥 Hot Reload triggered by Admin...")

        try:
            # BƯỚC 1: Import dữ liệu từ Google Drive (Sync Disk)
            # Import dynamic để tránh lỗi vòng lặp nếu chưa khởi tạo
            from qa_management.services import drive_service
            
            import_result = drive_service.import_from_drive()
            
            if not import_result.get('success'):
                return Response({
                    'success': False,
                    'step': 'import_drive',
                    'error': import_result.get('error'),
                    'message': '❌ Lỗi khi tải dữ liệu từ Google Drive'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            drive_stats = f"{import_result.get('imported')} mới, {import_result.get('updated')} cập nhật"
            logger.info(f"✅ Drive Sync OK: {drive_stats}")

            # BƯỚC 2: Nạp lại kiến thức vào RAM (Reload Memory)
            ai_stats = {}
            if chatbot_ai:
                # Kiểm tra xem chatbot có hàm reload không (đã thêm ở bước trước)
                if hasattr(chatbot_ai, 'reload_knowledge'):
                    ai_stats = chatbot_ai.reload_knowledge()
                    logger.info("✅ Chatbot RAM Reload OK")
                elif hasattr(chatbot_ai, 'reload_after_qa_update'):
                    # Hỗ trợ tên hàm cũ nếu bạn dùng tên này
                    chatbot_ai.reload_after_qa_update()
                    logger.info("✅ Chatbot RAM Reload OK (Legacy method)")
            
            total_time = time.time() - start_time

            return Response({
                'success': True,
                'message': '🔥 Hệ thống đã cập nhật dữ liệu nóng thành công!',
                'details': {
                    'drive_sync': drive_stats,
                    'ai_knowledge': ai_stats,
                    'duration': f"{total_time:.2f}s"
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"❌ Hot Reload Failed: {str(e)}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class APIRootView(APIView):
    """API Root - Hiển thị danh sách endpoints"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        test_memory = request.GET.get('test_memory')
        if test_memory and chatbot_ai:
            try:
                memory = chatbot_ai.get_conversation_memory(test_memory)
                return Response({
                    'memory_test': True,
                    'session_id': test_memory,
                    'memory': memory,
                    'total_sessions': len(chatbot_ai.response_generator.memory.conversations) if hasattr(chatbot_ai, 'response_generator') else 0
                })
            except Exception as e:
                return Response({
                    'memory_test': True,
                    'error': str(e)
                })
        
        system_status = get_safe_system_status()
        speech_status = get_safe_speech_status()
        tts_status = get_safe_tts_status()
        
        # ✅ SAFE: External API status check
        external_api_status = {'external_api_service': {'available': False, 'error': 'Service not available'}}
        try:
            from ai_models.external_api_service import external_api_service
            external_api_status = external_api_service.get_system_status()
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error getting external API status: {e}")

        # ✅ SAFE: Personalization status with fallbacks
        personalization_status = {
            'enabled': True,
            'active_personalized_sessions': 0,
            'user_memory_prompt_support': True,
            'flexible_personalization': True,
            'external_api_integration': external_api_status.get('external_api_service', {}).get('available', False),
            'jwt_token_support': True,
            'lecturer_schedule_access': True
        }
        
        try:
            if chatbot_ai and hasattr(chatbot_ai, 'response_generator'):
                personalization_status['active_personalized_sessions'] = len(chatbot_ai.response_generator._user_context_cache)
        except Exception as e:
            logger.error(f"Error getting personalization stats: {e}")
        
        # 🚀 NEW: Training status
        training_status = {
            'training_available': TRAINING_AVAILABLE,
            'training_endpoint': '/api/train-retriever/',
            'current_training_status': dict(TRAINING_STATUS) if TRAINING_AVAILABLE else None
        }
        
        return Response({
            'message': 'Enhanced Chatbot API với Text-to-Speech và Fine-tuned Model Training - Đại học Bình Dương',
            'version': '6.2.0',  # 🚀 Updated version
            'status': 'active',
            'system_status': system_status,
            'speech_status': speech_status,
            'tts_status': tts_status,
            'personalization_status': personalization_status,
            'external_api_status': external_api_status,
            'training_status': training_status,  # 🚀 NEW
            'endpoints': {
                'chat': '/api/chat/',
                'health': '/api/health/',
                'history': '/api/history/',
                'feedback': '/api/feedback/',
                'speech_to_text': '/api/speech-to-text/',
                'speech_status': '/api/speech-status/',
                'personalized_context': '/api/personalized-context/',
                'personalized_status': '/api/personalized-status/',
                'train_retriever': '/api/train-retriever/',
                'hot_reload': '/api/hot-reload/',
            },
            'features': [
                'Natural Language Generation',
                'Intent Classification',
                'Conversation Memory',
                'Emotional Context',
                'UTF-8 Safe Encoding',
                'Speech-to-Text (Whisper)',
                'Text-to-Speech (gTTS)',
                'Voice Conversation Mode',
                'Enhanced Personalization',
                'User Memory Prompt Support',
                'Flexible Personalization',
                'Dynamic System Prompts',
                'Custom User Instructions',
                'User Memory Integration',
                'Department-Specific Responses',
                'JWT Token Authentication',
                'External API Integration',
                'Lecturer Schedule Access',
                'Personal Information Queries',
                'Fine-tuned Model Training',  # 🚀 NEW
                'Two-Stage Re-ranking',  # 🚀 NEW
                'Advanced RAG Architecture',  # 🚀 NEW
            ]
        })

class HealthCheckView(APIView):
    """
    ✅ SAFE Health check endpoint - Always works regardless of AI services
    """
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            # Basic health check that always works
            health_data = {
                'status': 'healthy',
                'message': 'BDU ChatBot API is running! 🚀',
                'timestamp': timezone.now().isoformat(),
                'database': 'connected',
                'encoding': 'utf-8',
                'version': '6.2.0'  # 🚀 Updated version
            }
            
            # ✅ SAFE: Add service status only if available
            try:
                health_data['system_status'] = get_safe_system_status()
                health_data['speech_status'] = get_safe_speech_status()
                health_data['tts_status'] = get_safe_tts_status()
                
                # Voice interaction capability
                speech_available = health_data['speech_status'].get('available', False)
                tts_available = health_data['tts_status'].get('available', False)
                
                health_data['voice_interaction'] = {
                    'stt_available': speech_available,
                    'tts_available': tts_available,
                    'full_voice_chat': speech_available and tts_available
                }
                
                health_data['personalization'] = 'enabled'
                
                # 🚀 NEW: Training capability
                health_data['training_capability'] = {
                    'available': TRAINING_AVAILABLE,
                    'current_status': TRAINING_STATUS['is_running'],
                    'fine_tuned_model_support': True
                }
                
            except Exception as e:
                logger.error(f"Error getting service status in health check: {e}")
                health_data['services_status'] = 'limited'
                health_data['services_error'] = str(e)
            
            return Response(health_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return Response({
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Health check encountered an error',
                'timestamp': timezone.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatView(APIView):
    """Enhanced Chat API with Natural Responses and TTS"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """GET method - API information with personalization and TTS"""
        try:
            system_status = get_safe_system_status()
            speech_status = get_safe_speech_status()
            tts_status = get_safe_tts_status()
            
            # ✅ SAFE: External API status
            external_api_status = {'external_api_service': {'available': False, 'error': 'Service not available'}}
            try:
                from ai_models.external_api_service import external_api_service
                external_api_status = external_api_service.get_system_status()
            except:
                pass
            
            # ✅ SAFE: User personalization
            user_personalization = None
            if request.user.is_authenticated:
                try:
                    user_personalization = {
                        'faculty_code': getattr(request.user, 'faculty_code', 'N/A'),
                        'full_name': getattr(request.user, 'full_name', 'N/A'),
                        'department': getattr(request.user, 'department', 'N/A'),
                        'position': getattr(request.user, 'position', 'N/A'),
                        'has_user_memory_prompt': bool(getattr(request.user, 'chatbot_preferences', {}).get('user_memory_prompt', '').strip()),
                        'memory_length': len(getattr(request.user, 'chatbot_preferences', {}).get('user_memory_prompt', '')),
                        'department_priority': getattr(request.user, 'chatbot_preferences', {}).get('department_priority', True),
                        'personalized_prompt_available': True
                    }
                except Exception as e:
                    logger.error(f"Error getting user personalization: {e}")
            
            return Response({
                'message': 'Enhanced Personalized Chat API với Text-to-Speech và Fine-tuned Training - Open Access',
                'authentication': 'Optional - Works with or without token',
                'jwt_token_support': 'Send JWT token for personal schedule/info access',
                'system_status': system_status,
                'speech_status': speech_status,
                'tts_status': tts_status,
                'external_api_status': external_api_status,
                'user_personalization': user_personalization,
                'method': 'POST để gửi tin nhắn với personalization, JWT token và TTS',
                'jwt_token_usage': {
                    'header': 'Authorization: Bearer <token>',
                    'body_field': 'token',
                    'query_param': 'token (for testing only)',
                    'purpose': 'Access personal schedule and lecturer information'
                },
                'tts_usage': {
                    'mode_field': 'mode',
                    'voice_mode': 'voice - Tạo audio từ response text',
                    'text_mode': 'text - Chỉ trả về text (default)',
                    'audio_format': 'MP3 encoded as base64 string',
                    'supported_languages': tts_status.get('supported_languages', ['vi', 'en'])
                },
                'features': [
                    'PhoBERT Intent Classification',
                    'SBERT + FAISS Retrieval',
                    'Fine-tuned Model Support',  # 🚀 NEW
                    'Two-Stage Re-ranking',  # 🚀 NEW
                    'Advanced RAG Architecture',  # 🚀 NEW
                    'Conversation Memory',
                    'UTF-8 Safe Processing',
                    'Speech-to-Text Integration',
                    'Text-to-Speech Integration',
                    'Voice Conversation Mode',
                    'User Memory Prompt Support (with authentication)',
                    'Dynamic Personalized System Prompts (with authentication)',
                    'Flexible User Instructions (with authentication)',
                    'User Memory Integration (with authentication)',
                    'Anonymous Chat Support',
                    'JWT Token Authentication',
                    'External API Integration',
                    'Personal Schedule Access',
                    'Lecturer Information Queries',
                    'Model Training Capability'  # 🚀 NEW
                ]
            })
        except Exception as e:
            logger.error(f"Error in ChatView GET: {e}")
            return Response({
                'error': 'Chat API information not available',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        """POST method - Process chat with enhanced personalization support and TTS"""
        start_time = time.time()
        
        try:
            # Get and validate input
            user_message = request.data.get('message', '').strip()
            session_id = request.data.get('session_id', str(uuid.uuid4()))
            request_mode = request.data.get('mode', 'text').lower()
            
            logger.info(f"🎯 Request mode: {request_mode}")
            
            # ✅ NEW: Xử lý file tài liệu đính kèm (OCR)
            document_text = None
            document_file = request.FILES.get('document') # Frontend sẽ gửi file với key là 'document'
            if document_file and ocr_service:
                logger.info(f"📄 Document file received: {document_file.name}")
                # Lưu file tạm thời để xử lý
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(document_file.name)[1]) as tmp_file:
                    for chunk in document_file.chunks():
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Gọi OCR service để đọc file
                    pages_data = ocr_service.read_document(tmp_file_path)
                    if pages_data:
                        # Ghép nối text từ tất cả các trang
                        document_text = "\n\n".join([page['text'] for page in pages_data if page['text'].strip()])
                        logger.info(f"✅ OCR extracted {len(document_text)} characters.")
                    else:
                        logger.error("❌ OCR failed to extract text from document.")
                finally:
                    # Luôn xóa file tạm sau khi xử lý xong
                    os.unlink(tmp_file_path)
            elif document_file and not ocr_service:
                logger.error("❌ Document received, but OCR service is not available.")
            
            # Extract JWT token
            jwt_token = extract_jwt_token(request)
            
            if jwt_token:
                is_valid_format, format_message = validate_jwt_token_format(jwt_token)
                logger.info(f"🔑 JWT Token received: format_valid={is_valid_format}, message='{format_message}'")
                
                # 🚀 CRITICAL FIX: Auto-setup user context từ JWT
                if is_valid_format:
                    logger.info("🔧 Setting up user context from JWT token...")
                    setup_success = auto_setup_user_context_from_jwt(session_id, jwt_token)
                    if setup_success:
                        logger.info("✅ JWT auto-setup completed successfully")
                    else:
                        logger.warning("⚠️ JWT auto-setup failed")
                else:
                    logger.warning(f"⚠️ Invalid JWT format: {format_message}")
            else:
                logger.info("🔑 No JWT token provided - using standard QA mode")
            
            # Get user context (existing logic - keep as backup)
            user_id = request.user.id if request.user.is_authenticated else None
            user_context = None
            personalization_info = {}
            
            if user_id and request.user.is_authenticated:
                try:
                    if hasattr(request.user, 'get_chatbot_context'):
                        user_context = request.user.get_chatbot_context()
                    
                    # Extract personalization info
                    personalization_info = {
                        'department_priority': getattr(request.user, 'chatbot_preferences', {}).get('department_priority', True),
                        'department': getattr(request.user, 'department', 'Unknown'),
                        'position': getattr(request.user, 'position', 'Unknown'),
                        'has_user_memory_prompt': bool(getattr(request.user, 'chatbot_preferences', {}).get('user_memory_prompt', '').strip()),
                        'memory_length': len(getattr(request.user, 'chatbot_preferences', {}).get('user_memory_prompt', '')),
                        'personalized_prompt_available': True
                    }
                    
                    logger.info(f"👤 USER CONTEXT: {user_context.get('role_description', 'Unknown') if user_context else 'None'}")
                    
                    # 🚀 ADDITIONAL: Set authenticated user context as backup
                    if user_context and chatbot_ai and hasattr(chatbot_ai, 'response_generator'):
                        chatbot_ai.response_generator.set_user_context(session_id, user_context)
                        logger.info("✅ Authenticated user context also set as backup")
                    
                except Exception as e:
                    logger.warning(f"Could not get enhanced user context: {e}")
                    personalization_info['error'] = str(e)
            
            # ✅ CRITICAL: Nếu có file upload nhưng không có tin nhắn, tạo tin nhắn mặc định
            if document_text and not user_message:
                user_message = f"Dựa vào nội dung tài liệu này ({document_file.name}), hãy tóm tắt ý chính."
                logger.info(f"📝 No user message, generated default query: '{user_message}'")
            
            if not user_message:
                return Response(
                    {'error': 'Tin nhắn không được để trống'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if len(user_message) > 1000:
                return Response(
                    {'error': 'Tin nhắn quá dài (tối đa 1000 ký tự)'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # ENSURE UTF-8 encoding
            try:
                user_message = user_message.encode('utf-8').decode('utf-8')
            except UnicodeError:
                user_message = user_message.encode('utf-8', errors='ignore').decode('utf-8')
            
            logger.info(f"💬 Processing message: {user_message[:50]}... (User: {user_context.get('faculty_code') if user_context else 'Anonymous'}, JWT: {bool(jwt_token)}, Mode: {request_mode})")

            # ✅ SAFE: Process with AI service if available
            ai_response = None
            if chatbot_ai:
                try:
                    # 🔀 PARALLEL: Chạy trong thread pool để không block Django worker
                    future = _CHATBOT_THREAD_POOL.submit(
                        chatbot_ai.process_query,
                        user_message, session_id, jwt_token, document_text
                    )
                    ai_response = future.result(timeout=90)  # 90s timeout tránh treo
                except FutureTimeoutError:
                    logger.error("❌ ChatBot process_query timed out after 90s")
                    ai_response = None
                except Exception as e:
                    logger.error(f"Error processing with AI service: {e}")
            
            # Fallback response if AI service fails
            if not ai_response:
                ai_response = {
                    'response': self._get_fallback_response(user_message, user_context, personalization_info, jwt_token, request_mode),
                    'confidence': 0.3,
                    'method': 'fallback',
                    'sources': [],
                    'reference_links': [],
                    'user_memory_prompt_used': False,
                    'external_api_used': False,
                    'decision_type': 'fallback'
                }
            
            # ENSURE UTF-8 safe response
            response_text = ai_response['response']
            try:
                response_text = response_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean response text
            response_text = self._clean_response_text(response_text)
            
            # ✅ SAFE: TTS processing if requested and available
            audio_content_base64 = None
            tts_processing_time = 0
            tts_error = None
            
            if request_mode == 'voice' and response_text and tts_service:
                logger.info("🔊 Voice mode detected. Generating TTS response...")
                tts_start_time = time.time()
                
                try:
                    if hasattr(tts_service, 'text_to_audio_base64'):
                        audio_content_base64 = tts_service.text_to_audio_base64(response_text)
                        tts_processing_time = time.time() - tts_start_time
                        
                        if audio_content_base64:
                            logger.info(f"✅ TTS audio generated successfully in {tts_processing_time:.2f}s")
                        else:
                            logger.warning("⚠️ TTS audio generation failed - no audio returned")
                            tts_error = "TTS service returned no audio"
                    else:
                        tts_error = "TTS service method not available"
                        
                except Exception as e:
                    tts_processing_time = time.time() - tts_start_time
                    tts_error = str(e)
                    logger.error(f"❌ TTS audio generation failed: {e}")
            elif request_mode == 'voice' and not tts_service:
                logger.warning("⚠️ Voice mode requested but TTS service not available")
                tts_error = "TTS service not available"
            elif request_mode == 'voice':
                logger.warning("⚠️ Voice mode requested but no response text available")
                tts_error = "No response text available for TTS"
            else:
                logger.info(f"📝 Text mode - no TTS processing (mode: {request_mode})")
            
            processing_time = time.time() - start_time
            
            # ✅ SAFE: Save chat history
            try:
                enhanced_entities = {
                    'user_context': user_context,
                    'personalization_info': personalization_info,
                    'personalized': bool(user_context),
                    'user_memory_prompt_applied': ai_response.get('user_memory_prompt_used', False),
                    'department_priority_used': user_context.get('department_priority_enabled') if user_context else False,
                    'jwt_token_provided': bool(jwt_token),
                    'jwt_auto_setup_used': jwt_token and is_valid_format,  # 🚀 NEW FLAG
                    'external_api_used': ai_response.get('external_api_used', False),
                    'external_api_method': ai_response.get('method', '') if ai_response.get('external_api_used') else None,
                    'decision_type': ai_response.get('decision_type', ''),
                    'token_preview': f"{jwt_token[:10]}...{jwt_token[-10:]}" if jwt_token and len(jwt_token) > 20 else None,
                    'request_mode': request_mode,
                    'tts_generated': bool(audio_content_base64),
                    'tts_processing_time': tts_processing_time,
                    'tts_error': tts_error,
                    'reference_links': ai_response.get('reference_links', []),  # ✅ PRESERVED
                    'two_stage_reranking_used': ai_response.get('two_stage_reranking_used', False),  # 🚀 NEW
                    'fine_tuned_model_used': ai_response.get('fine_tuned_model_used', False),  # 🚀 NEW
                    'confidence_capped': ai_response.get('confidence_capped', False)  # 🚀 NEW
                }
                
                chat_record = ChatHistory.objects.create(
                    session_id=session_id,
                    user_message=user_message,
                    bot_response=response_text,
                    confidence_score=ai_response.get('confidence', 0.7),
                    response_time=processing_time,
                    user_ip=get_client_ip(request),
                    user=request.user if request.user.is_authenticated else None,
                    entities=json.dumps(enhanced_entities) if enhanced_entities else None
                )
                logger.info(f"✅ Enhanced chat saved: {chat_record.id}")
            except Exception as e:
                logger.error(f"Error saving chat history: {str(e)}")
            
            return Response({
                'session_id': session_id,
                'response': response_text,
                'confidence': ai_response['confidence'],
                'method': ai_response.get('method', 'hybrid'),
                'intent': ai_response.get('intent', {}).get('intent', 'general'),
                'sources': ai_response.get('sources', []),
                'response_time': processing_time,
                'status': 'success',
                'encoding': 'utf-8',
                'reference_links': ai_response.get('reference_links', []),  # ✅ PRESERVED
                'qa_source': ai_response.get('qa_source', None),  # 🆕 QA source info (KB only)
                
                # TTS fields
                'audio_content': audio_content_base64,
                'mode': request_mode,
                'tts_info': {
                    'enabled': request_mode == 'voice',
                    'processing_time': tts_processing_time,
                    'success': bool(audio_content_base64),
                    'error': tts_error,
                    'audio_format': 'mp3_base64' if audio_content_base64 else None
                },
                
                # 🚀 ENHANCED: Personalization info với JWT auto-setup
                'personalization': {
                    'enabled': bool(user_context) or bool(jwt_token and is_valid_format),
                    'jwt_auto_setup': jwt_token and is_valid_format,  # 🚀 NEW
                    'user_info': {
                        'department': personalization_info.get('department'),
                        'position': personalization_info.get('position'),
                        'faculty_code': user_context.get('faculty_code') if user_context else None
                    } if user_context else None,
                    'user_memory_info': {
                        'has_user_memory_prompt': personalization_info.get('has_user_memory_prompt', False),
                        'memory_length': personalization_info.get('memory_length', 0),
                        'memory_applied': ai_response.get('user_memory_prompt_used', False)
                    } if user_context else None,
                    'department_priority_used': personalization_info.get('department_priority', False)
                },
                
                # External API information
                'external_api': {
                    'jwt_token_provided': bool(jwt_token),
                    'jwt_format_valid': is_valid_format if jwt_token else None,  # 🚀 NEW
                    'external_api_used': ai_response.get('external_api_used', False),
                    'decision_type': ai_response.get('decision_type', ''),
                    'method_used': ai_response.get('method', ''),
                    'personal_info_accessed': ai_response.get('external_api_used', False),
                    'token_valid_format': validate_jwt_token_format(jwt_token)[0] if jwt_token else None
                },
                
                # 🚀 NEW: Advanced RAG information
                'advanced_rag': {
                    'two_stage_reranking_used': ai_response.get('two_stage_reranking_used', False),
                    'fine_tuned_model_used': ai_response.get('fine_tuned_model_used', False),
                    'confidence_capped': ai_response.get('confidence_capped', False),
                    'reranking_stats': ai_response.get('reranking_stats', {}),
                    'enhanced_processing': True
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            
            fallback_response = self._get_fallback_response(
                locals().get('user_message', ''),
                locals().get('user_context'),
                locals().get('personalization_info', {}),
                locals().get('jwt_token'),
                locals().get('request_mode', 'text')
            )
            
            return Response({
                'session_id': locals().get('session_id', str(uuid.uuid4())),
                'response': fallback_response,
                'confidence': 0.3,
                'method': 'fallback',
                'response_time': time.time() - start_time,
                'status': 'fallback',
                'audio_content': None,
                'mode': locals().get('request_mode', 'text'),
                'tts_info': {
                    'enabled': False,
                    'processing_time': 0,
                    'success': False,
                    'error': 'Fallback mode - TTS disabled',
                    'audio_format': None
                },
                'personalization': {
                    'enabled': bool(locals().get('user_context')),
                    'jwt_auto_setup': False,  # 🚀 NEW
                    'fallback_used': True,
                    'error': str(e)
                },
                'external_api': {
                    'jwt_token_provided': bool(locals().get('jwt_token')),
                    'jwt_format_valid': None,  # 🚀 NEW
                    'external_api_used': False,
                    'fallback_used': True,
                    'error': str(e)
                },
                'advanced_rag': {  # 🚀 NEW
                    'two_stage_reranking_used': False,
                    'fine_tuned_model_used': False,
                    'confidence_capped': False,
                    'fallback_used': True
                }
            }, status=status.HTTP_200_OK)
    
    def _get_fallback_response(self, user_message='', user_context=None, personalization_info={}, jwt_token=None, request_mode='text'):
        """Enhanced fallback response"""
        if user_context:
            full_name = user_context.get('full_name', '')
            faculty_code = user_context.get('faculty_code', '')
            name_suffix = full_name.split()[-1] if full_name else faculty_code
            personal_address = f"thầy/cô {name_suffix}"
            department_name = user_context.get('department_name', 'BDU')
            
            if jwt_token:
                base_message = f"""Dạ xin lỗi {personal_address}, hệ thống đang được nâng cấp để phục vụ {personal_address} tốt hơn.

Mặc dù em đã nhận được thông tin đăng nhập của {personal_address}, nhưng hiện tại có một số khó khăn kỹ thuật. 

{personal_address} có thể:
• Thử lại sau vài phút ⏰
• Truy cập trực tiếp hệ thống quản lý đào tạo của trường 🌐
• Liên hệ khoa {department_name} để được hỗ trợ trực tiếp 📞
• Gọi bộ phận IT: 0274.xxx.xxxx 📧

Em sẽ cố gắng khắc phục để phục vụ {personal_address} tốt hơn! 🎓✨"""
            else:
                has_user_memory = personalization_info.get('has_user_memory_prompt', False)
                
                if has_user_memory:
                    base_message = f"""Dạ xin lỗi {personal_address}, hệ thống đang được cải thiện để phục vụ {personal_address} tốt hơn theo những yêu cầu riêng mà {personal_address} đã thiết lập! 🧠

Để truy cập thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng BDU trước ạ. 🔐

Trong thời gian này, {personal_address} có thể:
• Liên hệ trực tiếp khoa {department_name} 📞
• Gọi tổng đài: 0274.xxx.xxxx  
• Email: info@bdu.edu.vn 📧
• Website: www.bdu.edu.vn 🌐

Em sẽ cố gắng hỗ trợ {personal_address} tốt hơn theo những ghi nhớ mà {personal_address} đã cung cấp! 🎓✨"""
                else:
                    base_message = f"""Dạ xin lỗi {personal_address}, hệ thống đang được cải thiện để phục vụ {personal_address} tốt hơn.

Để truy cập thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng BDU trước ạ. 🔐

Trong thời gian này, {personal_address} có thể:
• Liên hệ trực tiếp khoa {department_name}
• Gọi tổng đài: 0274.xxx.xxxx  
• Email: info@bdu.edu.vn
• Website: www.bdu.edu.vn

Cảm ơn {personal_address} đã kiên nhẫn! 🎓"""
            
            if request_mode == 'voice':
                base_message += f"\n\n🔊 Lưu ý: Chức năng chuyển văn bản thành giọng nói tạm thời không khả dụng. {personal_address} vẫn có thể đọc phản hồi này."
            
            return base_message
        
        # Fallback for non-authenticated users
        base_message = """Xin chào! Tôi đã nhận được thông tin đăng nhập, nhưng hiện tại gặp khó khăn kỹ thuật.

Bạn có thể thử lại sau hoặc liên hệ:
• Hotline: 0274.xxx.xxxx
• Email: info@bdu.edu.vn
• Website: www.bdu.edu.vn

Cảm ơn bạn đã kiên nhẫn! 🎓"""
        
        if request_mode == 'voice':
            base_message += "\n\n🔊 Lưu ý: Chức năng chuyển văn bản thành giọng nói tạm thời không khả dụng."
        
        return base_message if jwt_token else self._get_safe_fallback_response(user_message)
    
    def _clean_response_text(self, text):
        """Clean and ensure safe UTF-8 text"""
        import re
        
        # Remove control characters and invalid UTF-8
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Fix common encoding issues
        encoding_fixes = {
            'â€™': "'",
            'â€œ': '"', 
            'â€': '"',
            'â€"': '-',
            'â€¦': '...',
            'Ã¡': 'á',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã£': 'ã',
            'Ã¨': 'è',
            'Ã©': 'é',
            'Ãª': 'ê',
            'Ã¬': 'ì',
            'Ã­': 'í',
            'Ã²': 'ò',
            'Ã³': 'ó',
            'Ã´': 'ô',
            'Ã¹': 'ù',
            'Ãº': 'ú',
            'Ã½': 'ý',
            'Ä': 'đ',
            'Ä': 'Đ'
        }
        
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        # Clean up spaces and newlines only
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _get_safe_fallback_response(self, user_message=''):
        """Safe fallback response with proper UTF-8"""
        return f"""Xin chào! Tôi đã nhận được câu hỏi của bạn. 

Hiện tại hệ thống đang được cải thiện để phục vụ bạn tốt hơn. Trong thời gian này, bạn có thể:

• Liên hệ trực tiếp: 0274.xxx.xxxx
• Email: info@bdu.edu.vn  
• Website: www.bdu.edu.vn

Cảm ơn bạn đã kiên nhẫn! 😊"""

# ✅ KEEP ALL EXISTING VIEWS UNCHANGED TO PRESERVE MIC/SPEECH FUNCTIONALITY

class SpeechToTextView(APIView):
    """Speech-to-Text API endpoint"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """GET method - Service information"""
        speech_status = get_safe_speech_status()
        return Response({
            'message': 'Speech-to-Text API',
            'method': 'POST để upload audio file',
            'speech_service': speech_status,
            'supported_formats': ['wav', 'mp3', 'webm', 'm4a'],
            'max_file_size_mb': 10,
            'usage': {
                'method': 'POST',
                'content_type': 'multipart/form-data',
                'fields': {
                    'audio': 'Audio file (required)',
                    'language': 'Language code (optional, default: vi)',
                    'beam_size': 'Beam size for better accuracy (optional, default: 5)'
                }
            }
        })
    
    def post(self, request):
        """POST method - Process audio file"""
        start_time = time.time()
        
        try:
            # Check if service is available
            if not speech_service:
                return Response({
                    'success': False,
                    'error': 'Speech-to-Text service not available. Service not loaded.',
                    'text': ''
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            if not hasattr(speech_service, 'is_available') or not speech_service.is_available():
                return Response({
                    'success': False,
                    'error': 'Speech-to-Text service not available. Please install faster-whisper.',
                    'text': '',
                    'status': get_safe_speech_status()
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Check if file is in request
            if 'audio' not in request.FILES:
                return Response({
                    'success': False,
                    'error': 'No audio file provided. Please upload an audio file.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            audio_file = request.FILES['audio']
            
            # Validate file size (10MB max)
            max_size = 10 * 1024 * 1024
            if audio_file.size > max_size:
                return Response({
                    'success': False,
                    'error': f'File too large. Maximum size: 10MB'
                }, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
            # Check minimum file size
            if audio_file.size < 1024:  # Less than 1KB
                return Response({
                    'success': False,
                    'error': 'Audio file too small. Please record longer audio.',
                    'text': ''
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get optional parameters
            language = request.data.get('language', 'vi')
            beam_size = int(request.data.get('beam_size', 5))
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(audio_file.name)[1] or '.webm'
            ) as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                tmp_file.flush()
                
                try:
                    # Process with speech service
                    if hasattr(speech_service, 'transcribe_audio'):
                        result = speech_service.transcribe_audio(
                            tmp_file.name,
                            language=language,
                            beam_size=beam_size
                        )
                    else:
                        result = {
                            'success': False,
                            'error': 'Speech service method not available',
                            'text': ''
                        }
                    
                    if result.get('success'):
                        transcribed_text = result.get('text', '').strip()
                        if not transcribed_text:
                            return Response({
                                'success': False,
                                'error': 'No speech detected in audio. Please speak louder or check microphone.',
                                'text': ''
                            }, status=status.HTTP_200_OK)
                    
                    # Add additional metadata
                    result['file_name'] = audio_file.name
                    result['file_size_mb'] = round(audio_file.size / (1024 * 1024), 2)
                    result['total_processing_time'] = time.time() - start_time
                    
                    return Response(result, status=status.HTTP_200_OK)
                    
                finally:
                    # Clean up temporary file
                    try:
                        if os.path.exists(tmp_file.name):
                            os.unlink(tmp_file.name)
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Speech-to-text error: {str(e)}")
            
            return Response({
                'success': False,
                'error': f'Server error: {str(e)}',
                'text': '',
                'processing_time': time.time() - start_time
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SpeechStatusView(APIView):
    """Get Speech service status"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            speech_status = get_safe_speech_status()
            tts_status = get_safe_tts_status()
            
            return Response({
                'status': 'ok',
                'message': 'Speech Services Status (STT + TTS)',
                'speech_service': speech_status,
                'tts_service': tts_status,
                'endpoints': {
                    'speech_to_text': '/api/speech-to-text/',
                    'speech_status': '/api/speech-status/'
                },
                'capabilities': {
                    'stt_languages': ['vi', 'en'],
                    'tts_languages': tts_status.get('supported_languages', ['vi', 'en']),
                    'supported_formats': ['wav', 'mp3', 'webm', 'm4a'],
                    'max_file_size_mb': 10,
                    'features': [
                        'Voice Activity Detection',
                        'Noise Suppression', 
                        'Automatic Language Detection',
                        'GPU Acceleration (if available)',
                        'Text-to-Speech (gTTS)',
                        'Voice Conversation Mode',
                        'Multi-language TTS Support'
                    ]
                },
                'voice_interaction': {
                    'full_duplex_available': speech_status.get('available', False) and tts_status.get('available', False),
                    'stt_available': speech_status.get('available', False),
                    'tts_available': tts_status.get('available', False),
                    'recommended_workflow': [
                        '1. User speaks (STT)',
                        '2. AI processes text',
                        '3. AI responds with text + audio (TTS)',
                        '4. User hears response'
                    ]
                }
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error getting speech status: {str(e)}")
            return Response({
                'status': 'error',
                'error': str(e),
                'speech_service': {
                    'available': False,
                    'error': 'Service status check failed'
                },
                'tts_service': {
                    'available': False,
                    'error': 'Service status check failed'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TextToSpeechTestView(APIView):
    """TTS test endpoint"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        tts_status = get_safe_tts_status()
        return Response({
            'message': 'Text-to-Speech Test API',
            'method': 'POST để test TTS conversion',
            'tts_service': tts_status,
            'usage': {
                'method': 'POST',
                'content_type': 'application/json',
                'fields': {
                    'text': 'Text to convert to speech (required)',
                    'language': 'Language code (optional, default: vi)',
                    'slow': 'Slow speech (optional, default: false)'
                }
            }
        })
    
    def post(self, request):
        try:
            if not tts_service:
                return Response({
                    'success': False,
                    'error': 'TTS service not available.',
                    'audio_content': None
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            text_to_convert = request.data.get('text', '').strip()
            if not text_to_convert:
                return Response({
                    'success': False,
                    'error': 'Text field is required.',
                    'audio_content': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate TTS audio
            if hasattr(tts_service, 'text_to_audio_base64'):
                audio_base64 = tts_service.text_to_audio_base64(text_to_convert)
                
                if audio_base64:
                    return Response({
                        'success': True,
                        'audio_content': audio_base64,
                        'text_processed': text_to_convert,
                        'audio_format': 'mp3_base64'
                    })
                else:
                    return Response({
                        'success': False,
                        'error': 'Failed to generate TTS audio.',
                        'audio_content': None
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                return Response({
                    'success': False,
                    'error': 'TTS service method not available.',
                    'audio_content': None
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
        except Exception as e:
            return Response({
                'success': False,
                'error': f'TTS test failed: {str(e)}',
                'audio_content': None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ KEEP ALL OTHER EXISTING VIEWS UNCHANGED

class ChatHistoryView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request, session_id=None):
        try:
            if session_id:
                history = ChatHistory.objects.filter(session_id=session_id).order_by('timestamp')
            else:
                history = ChatHistory.objects.all().order_by('-timestamp')[:50]
            
            data = [{
                'id': chat.id,
                'session_id': chat.session_id,
                'user_message': chat.user_message,
                'bot_response': chat.bot_response,
                'timestamp': chat.timestamp.isoformat(),
                'confidence': chat.confidence_score,
                'response_time': chat.response_time
            } for chat in history]
            
            return Response({
                'count': len(data),
                'results': data
            })
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return Response(
                {'error': 'Không thể lấy lịch sử chat'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FeedbackView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            chat_id = request.data.get('chat_id')
            feedback_type = request.data.get('feedback_type')
            comment = request.data.get('comment', '')
            
            if not chat_id or not feedback_type:
                return Response(
                    {'error': 'chat_id và feedback_type là bắt buộc'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                chat_history = ChatHistory.objects.get(id=chat_id)
            except ChatHistory.DoesNotExist:
                return Response(
                    {'error': 'Không tìm thấy cuộc trò chuyện'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            feedback = UserFeedback.objects.create(
                chat_history=chat_history,
                feedback_type=feedback_type,
                comment=comment
            )
            
            return Response({
                'message': 'Cảm ơn phản hồi của bạn!',
                'feedback_id': feedback.id
            })
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return Response(
                {'error': 'Không thể lưu phản hồi'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PersonalizedChatContextView(APIView):
    """Lấy context cá nhân hóa cho chat"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            if not request.user.is_authenticated:
                return Response({
                    'personalization_enabled': False,
                    'message': 'User not authenticated'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user = request.user
            
            # Safe user context extraction
            user_context = {}
            try:
                if hasattr(user, 'get_chatbot_context'):
                    user_context = user.get_chatbot_context()
            except Exception as e:
                logger.error(f"Error getting user context: {e}")
            
            tts_status = get_safe_tts_status()
            
            context_info = {
                'personalization_enabled': True,
                'user_context': user_context,
                'personalized_greeting': f"Chào {getattr(user, 'position_name', 'giảng viên')} {getattr(user, 'full_name', 'N/A')}!",
                'department_focus': getattr(user, 'department', 'BDU'),
                'tts_capabilities': {
                    'available': tts_status.get('available', False),
                    'supported_languages': tts_status.get('supported_languages', []),
                    'default_language': tts_status.get('default_language', 'vi'),
                    'voice_mode_enabled': tts_status.get('available', False)
                },
                'current_settings': {
                    'department_priority': getattr(user, 'chatbot_preferences', {}).get('department_priority', True),
                    'has_user_memory_prompt': bool(getattr(user, 'chatbot_preferences', {}).get('user_memory_prompt', '').strip()),
                    'external_api_ready': True,
                    'tts_enabled': tts_status.get('available', False)
                }
            }
            
            return Response(context_info, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Personalized context error: {str(e)}")
            return Response({
                'personalization_enabled': False,
                'error': 'Could not load personalized context',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PersonalizedSystemStatusView(APIView):
    """System status với thông tin personalization"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            status_data = get_safe_system_status()
            speech_status = get_safe_speech_status()
            tts_status = get_safe_tts_status()
            
            personalization_status = {
                'personalization_enabled': True,
                'version': '6.2.0',  # 🚀 Updated version
                'features': {
                    'user_memory_prompt_support': True,
                    'flexible_personalization': True,
                    'dynamic_system_prompts': True,
                    'custom_user_instructions': True,
                    'department_priority': True,
                    'personalized_addressing': True,
                    'jwt_token_authentication': True,
                    'external_api_integration': True,
                    'personal_schedule_access': True,
                    'lecturer_info_queries': True,
                    'text_to_speech_support': True,
                    'voice_conversation_mode': True,
                    'speech_to_text_support': True,
                    'full_voice_interaction': True,
                    'fine_tuned_model_training': TRAINING_AVAILABLE,  # 🚀 NEW
                    'two_stage_reranking': True,  # 🚀 NEW
                    'advanced_rag_architecture': True  # 🚀 NEW
                },
                'statistics': {
                    'total_faculty': 0,
                    'active_personalized_sessions': 0,
                    'departments_available': 0,
                    'positions_available': 0
                }
            }
            
            # Add current user info if authenticated
            if request.user.is_authenticated:
                personalization_status['current_user'] = {
                    'faculty_code': getattr(request.user, 'faculty_code', 'N/A'),
                    'department': getattr(request.user, 'department', 'N/A'),
                    'position': getattr(request.user, 'position', 'N/A'),
                    'has_user_memory_prompt': bool(getattr(request.user, 'chatbot_preferences', {}).get('user_memory_prompt', '').strip()),
                    'department_priority': getattr(request.user, 'chatbot_preferences', {}).get('department_priority', True),
                    'preferences_configured': bool(getattr(request.user, 'chatbot_preferences', {})),
                    'external_api_ready': True,
                    'tts_ready': tts_status.get('available', False)
                }
            
            # Merge with system status
            status_data.update({
                'personalization': personalization_status,
                'speech_status': speech_status,
                'tts_status': tts_status,
                'training_status': {  # 🚀 NEW
                    'training_available': TRAINING_AVAILABLE,
                    'current_training': dict(TRAINING_STATUS)
                }
            })
            
            return Response(status_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            return Response({
                'error': 'Could not retrieve system status',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatSessionsView(APIView):
    """Quản lý chat sessions của user"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Lấy danh sách sessions của user"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            sessions = ChatHistory.objects.filter(user=request.user) \
                .values('session_id', 'session_title') \
                .annotate(
                    last_message_time=models.Max('timestamp'),
                    message_count=models.Count('id')
                ) \
                .order_by('-last_message_time')[:20]
            
            sessions_list = []
            for session in sessions:
                last_chat = ChatHistory.objects.filter(
                    user=request.user,
                    session_id=session['session_id']
                ).order_by('-timestamp').first()
                
                sessions_list.append({
                    'session_id': session['session_id'],
                    'title': session['session_title'] or f"Chat {session['session_id'][-8:]}",
                    'last_message_time': session['last_message_time'],
                    'message_count': session['message_count'],
                    'preview': last_chat.user_message[:50] + '...' if last_chat and len(last_chat.user_message) > 50 else last_chat.user_message if last_chat else '',
                    'active': False
                })
            
            return Response({
                'success': True,
                'sessions': sessions_list,
                'total_sessions': len(sessions_list)
            })
            
        except Exception as e:
            logger.error(f"Error loading chat sessions: {str(e)}")
            return Response({
                'success': False,
                'error': 'Could not load chat sessions'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request):
        """Tạo session mới"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            session_title = request.data.get('title', '')
            new_session_id = f"session_{getattr(request.user, 'faculty_code', 'user')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            return Response({
                'success': True,
                'session_id': new_session_id,
                'title': session_title or f"Chat mới - {timezone.now().strftime('%H:%M')}"
            })
            
        except Exception as e:
            logger.error(f"Error creating new session: {str(e)}")
            return Response({
                'success': False,
                'error': 'Could not create new session'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatSessionDetailView(APIView):
    """Chi tiết một chat session"""
    permission_classes = [AllowAny]
    
    def get(self, request, session_id):
        """Lấy toàn bộ chat history của session"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            chat_history = ChatHistory.objects.filter(
                user=request.user,
                session_id=session_id
            ).order_by('timestamp')
            
            messages = []
            for chat in chat_history:
                # User message
                messages.append({
                    'type': 'user',
                    'content': chat.user_message,
                    'timestamp': chat.timestamp.isoformat()
                })
                
                bot_entities = {}
                if chat.entities:
                    try:
                        bot_entities = json.loads(chat.entities)
                    except json.JSONDecodeError:
                        bot_entities = {}
                        
                # Bot message
                messages.append({
                    'type': 'bot',
                    'content': chat.bot_response,
                    'timestamp': chat.timestamp.isoformat(),
                    'confidence': chat.confidence_score,
                    'response_time': chat.response_time,
                    'sources': bot_entities.get('sources', []),
                    'reference_links': bot_entities.get('reference_links', []),  # ✅ PRESERVED
                    'chat_id': chat.id,
                    'two_stage_reranking_used': bot_entities.get('two_stage_reranking_used', False),  # 🚀 NEW
                    'fine_tuned_model_used': bot_entities.get('fine_tuned_model_used', False)  # 🚀 NEW
                })
            
            return Response({
                'success': True,
                'session_id': session_id,
                'messages': messages,
                'total_messages': len(messages)
            })
            
        except Exception as e:
            logger.error(f"Error loading session detail: {str(e)}")
            return Response({
                'success': False,
                'error': 'Could not load session messages'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def patch(self, request, session_id):
        """Cập nhật thông tin session (rename)"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            new_title = request.data.get('title', '').strip()
            
            if not new_title:
                return Response({
                    'success': False,
                    'error': 'Title không được để trống'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Cập nhật session_title
            updated_count = ChatHistory.objects.filter(
                user=request.user,
                session_id=session_id
            ).update(session_title=new_title)
            
            return Response({
                'success': True,
                'session_id': session_id,
                'new_title': new_title,
                'updated_records': updated_count,
                'message': 'Đã đổi tên đoạn chat thành công'
            })
            
        except Exception as e:
            logger.error(f"Error updating session title: {str(e)}")
            return Response({
                'success': False,
                'error': 'Không thể cập nhật tên session'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request, session_id):
        """Xóa session"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            deleted_count = ChatHistory.objects.filter(
                user=request.user,
                session_id=session_id
            ).delete()[0]
            
            return Response({
                'success': True,
                'deleted_messages': deleted_count
            })
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return Response({
                'success': False,
                'error': 'Could not delete session'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ FUNCTION-BASED VIEWS FOR COMPATIBILITY (Google Drive endpoints)
def health_check(request):
    """Health check function-based view"""
    return JsonResponse({
        'status': 'healthy',
        'message': 'BDU ChatBot API is running!',
        'version': '6.2.0',
        'advanced_rag': True,
        'training_available': TRAINING_AVAILABLE
    })

def speech_status(request):
    """Speech status function-based view"""
    try:
        speech_status = get_safe_speech_status()
        tts_status = get_safe_tts_status()
        
        return JsonResponse({
            'speech_service': speech_status,
            'tts_service': tts_status,
            'voice_interaction_available': speech_status.get('available', False) and tts_status.get('available', False)
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'speech_service': {'available': False},
            'tts_service': {'available': False}
        })

def speech_to_text(request):
    """Speech to text function-based view"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    # Redirect to class-based view
    view = SpeechToTextView()
    return view.post(request)

def force_refresh_drive_data(request):
    """Force refresh Google Drive data"""
    try:
        from qa_management.services import drive_service
        result = drive_service.force_refresh()
        
        # Reload chatbot after data refresh
        if result.get('success') and chatbot_ai:
            chatbot_ai.reload_after_qa_update()
        
        return JsonResponse(result)
    except ImportError:
        return JsonResponse({
            'success': False,
            'error': 'QA Management service not available'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

def google_drive_status(request):
    """Get Google Drive sync status"""
    try:
        from qa_management.services import drive_service
        status = drive_service.get_system_status()
        return JsonResponse(status)
    except ImportError:
        return JsonResponse({
            'available': False,
            'error': 'QA Management service not available'
        })
    except Exception as e:
        return JsonResponse({
            'available': False,
            'error': str(e)
        })


# ─── Streaming Chat View (SSE) ────────────────────────────────────────────────

class ChatStreamView(APIView):
    """
    🌊 Streaming Chat API - Server-Sent Events (SSE)
    POST /api/chat/stream/

    Response format (chunked):
      data: {"delta": "...", "done": false}\n\n
      data: {"delta": "", "done": true, "reference_links": [...], "qa_source": {...}, "intent": "..."}\n\n

    Frontend đọc EventSource hoặc fetch với ReadableStream để hiển thị text ngay khi nhận.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        user_message = request.data.get('message', '').strip()
        session_id = request.data.get('session_id', str(uuid.uuid4()))

        if not user_message:
            return Response({'error': 'Tin nhắn không được để trống'}, status=status.HTTP_400_BAD_REQUEST)
        if len(user_message) > 1000:
            return Response({'error': 'Tin nhắn quá dài (tối đa 1000 ký tự)'}, status=status.HTTP_400_BAD_REQUEST)

        # Chuẩn hóa encoding
        try:
            user_message = user_message.encode('utf-8').decode('utf-8')
        except UnicodeError:
            user_message = user_message.encode('utf-8', errors='ignore').decode('utf-8')

        # Extract JWT token
        jwt_token = extract_jwt_token(request)
        if jwt_token:
            is_valid_format, _ = validate_jwt_token_format(jwt_token)
            if is_valid_format:
                auto_setup_user_context_from_jwt(session_id, jwt_token)

        # Document (OCR)
        document_text = None
        document_file = request.FILES.get('document')
        if document_file and ocr_service:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(document_file.name)[1]) as tmp_file:
                for chunk in document_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            try:
                pages_data = ocr_service.read_document(tmp_file_path)
                if pages_data:
                    document_text = "\n\n".join([p['text'] for p in pages_data if p['text'].strip()])
            finally:
                os.unlink(tmp_file_path)

        if document_text and not user_message:
            user_message = f"Dựa vào nội dung tài liệu này ({document_file.name}), hãy tóm tắt ý chính."

        # Lấy addr để greeting đúng
        addr = ""
        try:
            if chatbot_ai and hasattr(chatbot_ai, '_get_addr'):
                addr = chatbot_ai._get_addr(session_id)
            if not addr and jwt_token:
                from ai_models.external_api_service import external_api_service as _ext
                info = _ext.get_lecturer_info_from_token(jwt_token)
                if info and chatbot_ai:
                    addr = chatbot_ai._build_addr(info)
        except Exception:
            pass

        # Kiểm tra chatbot available
        if not chatbot_ai or not getattr(chatbot_ai, '_agent_available', False) or not chatbot_ai.langchain_agent:
            def _fallback_gen():
                msg = '{"delta": "Dạ, hệ thống đang khởi động. Vui lòng thử lại sau ạ. 🎓", "done": false}'
                yield f"data: {msg}\n\n"
                yield 'data: {"delta": "", "done": true, "reference_links": [], "qa_source": null, "intent": "error"}\n\n'
            resp = StreamingHttpResponse(_fallback_gen(), content_type='text/event-stream')
            resp['Cache-Control'] = 'no-cache'
            resp['X-Accel-Buffering'] = 'no'
            return resp

        logger.info(f"🌊 [ChatStreamView] '{user_message[:60]}' | session={session_id}")

        def _stream_generator():
            """Generator chạy trong HTTP response context."""
            # Dùng queue để đưa kết quả từ thread pool về main thread
            result_queue = queue.Queue()

            def _worker():
                try:
                    gen = chatbot_ai.langchain_agent.run_stream(
                        query=user_message,
                        session_id=session_id,
                        jwt_token=jwt_token,
                        document_text=document_text,
                        addr=addr,
                    )
                    for item in gen:
                        result_queue.put(item)
                except Exception as exc:
                    result_queue.put({'_error': str(exc)})
                finally:
                    result_queue.put(None)  # sentinel

            future = _CHATBOT_THREAD_POOL.submit(_worker)

            while True:
                try:
                    item = result_queue.get(timeout=95)  # > Ollama timeout
                except queue.Empty:
                    # Timeout toàn cục
                    err_json = json.dumps({"delta": " [Timeout]", "done": True,
                                           "reference_links": [], "qa_source": None, "intent": "error"},
                                          ensure_ascii=False)
                    yield f"data: {err_json}\n\n"
                    break

                if item is None:  # sentinel
                    break

                if isinstance(item, dict):
                    if '_error' in item:
                        err_json = json.dumps({"delta": "Dạ, em gặp sự cố kỹ thuật. Vui lòng thử lại ạ. 🎓",
                                               "done": True, "reference_links": [], "qa_source": None,
                                               "intent": "error"}, ensure_ascii=False)
                        yield f"data: {err_json}\n\n"
                        break
                    # Chunk cuối: done metadata
                    done_chunk = {
                        "delta": "",
                        "done": True,
                        "reference_links": item.get("reference_links", []),
                        "qa_source": item.get("qa_source"),
                        "intent": item.get("intent", ""),
                    }
                    yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
                    break
                else:
                    # str chunk - text token
                    text_chunk = {"delta": item, "done": False}
                    yield f"data: {json.dumps(text_chunk, ensure_ascii=False)}\n\n"

        streaming_response = StreamingHttpResponse(
            _stream_generator(),
            content_type='text/event-stream; charset=utf-8',
        )
        streaming_response['Cache-Control'] = 'no-cache'
        streaming_response['X-Accel-Buffering'] = 'no'  # Tắt nginx buffering
        streaming_response['Access-Control-Allow-Origin'] = '*'
        return streaming_response
