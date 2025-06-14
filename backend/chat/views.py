from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from knowledge.models import ChatHistory, UserFeedback
from ai_models.services import chatbot_ai
from ai_models.speech_service import speech_service  # ← THÊM IMPORT
import uuid
import time
import logging
import json
import tempfile
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

class APIRootView(APIView):
    """API Root - Hiển thị danh sách endpoints"""
    def get(self, request):
        
        test_memory = request.GET.get('test_memory')
        if test_memory:
            try:
                memory = chatbot_ai.get_conversation_memory(test_memory)
                return Response({
                    'memory_test': True,
                    'session_id': test_memory,
                    'memory': memory,
                    'total_sessions': len(chatbot_ai.response_generator.memory.conversations)
                })
            except Exception as e:
                return Response({
                    'memory_test': True,
                    'error': str(e)
                })
        
        system_status = chatbot_ai.get_system_status()
        
        # ✅ THÊM: Speech service status
        speech_status = speech_service.get_system_status()
        
        return Response({
            'message': 'Chatbot API - Đại học Bình Dương',
            'version': '3.1.0',  # ← Tăng version
            'status': 'active',
            'system_status': system_status,
            'speech_status': speech_status,  # ← THÊM
            'endpoints': {
                'chat': '/api/chat/',
                'health': '/api/health/',
                'history': '/api/history/',
                'feedback': '/api/feedback/',
                'speech_to_text': '/api/speech-to-text/',  # ← THÊM
                'speech_status': '/api/speech-status/',    # ← THÊM
            },
            'features': [
                'Natural Language Generation',
                'Intent Classification',
                'Conversation Memory',
                'Emotional Context',
                'UTF-8 Safe Encoding',
                'Speech-to-Text (Whisper)',  # ← THÊM
            ]
        })

class ChatView(APIView):
    """Enhanced Chat API with Natural Responses"""
    
    def get(self, request):
        """GET method - API information"""
        system_status = chatbot_ai.get_system_status()
        speech_status = speech_service.get_system_status()  # ← THÊM
        
        return Response({
            'message': 'Natural Language Chat API',
            'system_status': system_status,
            'speech_status': speech_status,  # ← THÊM
            'method': 'POST để gửi tin nhắn',
            'features': [
                'PhoBERT Intent Classification',
                'SBERT + FAISS Retrieval',
                'Conversation Memory',
                'UTF-8 Safe Processing',
                'Speech-to-Text Integration'  # ← THÊM
            ]
        })
    
    def post(self, request):
        """POST method - Process chat with personalization support"""
        start_time = time.time()
        
        try:
            # Get and validate input
            user_message = request.data.get('message', '').strip()
            session_id = request.data.get('session_id', str(uuid.uuid4()))
            
            # ✅ THÊM: Lấy user_id để personalization
            user_id = request.user.id if request.user.is_authenticated else None
            
            print(f"🔍 CHAT DEBUG: user_id = {user_id}, session_id = {session_id}")
            print(f"🔍 CHAT DEBUG: User message = {user_message}")
            
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
            
            # ✅ THÊM: Lấy user context nếu có
            user_context = None
            if user_id and request.user.is_authenticated:
                try:
                    user_context = request.user.get_chatbot_context()
                    print(f"👤 USER CONTEXT: {user_context.get('role_description', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"Could not get user context: {e}")
            
            logger.info(f"💬 Processing: {user_message[:50]}... (User: {user_context.get('faculty_code') if user_context else 'Anonymous'})")
            
            # ✅ THÊM: Process với user context
            if user_context:
                # Sử dụng personalized processing
                ai_response = self._process_with_personalization(user_message, session_id, user_context)
            else:
                # Sử dụng processing thông thường
                ai_response = chatbot_ai.process_query(user_message, session_id)
            
            print(f"🔍 CHAT DEBUG: AI response method = {ai_response.get('method', 'unknown')}")
            
            # ENSURE UTF-8 safe response
            response_text = ai_response['response']
            try:
                response_text = response_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean response text
            response_text = self._clean_response_text(response_text)
            
            processing_time = time.time() - start_time
            
            # Save chat history với user context
            try:
                chat_record = ChatHistory.objects.create(
                    session_id=session_id,
                    user_message=user_message,
                    bot_response=response_text,
                    confidence_score=ai_response.get('confidence', 0.7),
                    response_time=processing_time,
                    user_ip=get_client_ip(request),
                    # ✅ THÊM: Lưu user context vào entities
                    entities=json.dumps({
                        'user_context': user_context,
                        'personalized': bool(user_context)
                    }) if user_context else None
                )
                logger.info(f"✅ Chat saved: {chat_record.id}")
            except Exception as e:
                logger.error(f"Error saving chat: {str(e)}")
            
            # Return enhanced response
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
                # ✅ THÊM: Personalization info
                'personalized': bool(user_context),
                'user_context': {
                    'department': user_context.get('department_name') if user_context else None,
                    'position': user_context.get('position_name') if user_context else None,
                    'faculty_code': user_context.get('faculty_code') if user_context else None
                } if user_context else None
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            
            # Safe fallback response với personalization
            fallback_response = self._get_safe_fallback_response_personalized(
                locals().get('user_message', ''),
                locals().get('user_context')
            )
            
            return Response({
                'session_id': locals().get('session_id', str(uuid.uuid4())),
                'response': fallback_response,
                'confidence': 0.3,
                'method': 'safe_fallback',
                'response_time': time.time() - start_time,
                'status': 'fallback',
                'personalized': bool(locals().get('user_context'))
            })
    
    def _get_safe_fallback_response_personalized(self, user_message='', user_context=None):
        """Safe fallback response với personalization"""
        if user_context:
            full_name = user_context.get('full_name', '')
            faculty_code = user_context.get('faculty_code', '')
            name_suffix = full_name.split()[-1] if full_name else faculty_code
            personal_address = f"thầy/cô {name_suffix}"
            department_name = user_context.get('department_name', 'BDU')
            
            return f"""Dạ xin lỗi {personal_address}, hệ thống đang được cải thiện để phục vụ {personal_address} tốt hơn.

Trong thời gian này, {personal_address} có thể:
• Liên hệ trực tiếp khoa {department_name}
• Gọi tổng đài: 0274.xxx.xxxx  
• Email: info@bdu.edu.vn
• Website: www.bdu.edu.vn

Cảm ơn {personal_address} đã kiên nhẫn! 😊"""
        
        return self._get_safe_fallback_response(user_message)
    
    # ✅ THÊM: Method mới để xử lý personalization
    def _process_with_personalization(self, message, session_id, user_context):
        """Process message với personalization"""
        try:
            # Sử dụng gemini service với personalization
            from ai_models.gemini_service import GeminiResponseGenerator
            
            # Tạo enhanced context
            enhanced_context = {
                'user_context': user_context,
                'force_education_response': True,
                'personalized': True
            }
            
            # Gọi generate_response_personalized nếu có
            gemini_generator = GeminiResponseGenerator()
            base_response = chatbot_ai.process_query(message, session_id)

            # Sau đó enhance với personalization
            if hasattr(gemini_generator, 'enhance_with_personalization'):
                return gemini_generator.enhance_with_personalization(base_response, user_context)
            else:
                return base_response
                
        except Exception as e:
            logger.error(f"Personalized processing error: {e}")
            # Fallback to regular processing
            return chatbot_ai.process_query(message, session_id)
    
    def _clean_response_text(self, text):
        """Clean and ensure safe UTF-8 text"""
        import re
        
        # Remove control characters and invalid UTF-8
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '-')
        
        # # Remove any garbled Vietnamese characters patterns
        # text = re.sub(r'[ẤẬẦẨẪĂẮẶẰẲẴÂẤẬẦẨẪÉẾỆỀỂỄÊẾỆỀỂỄÍỊÌỈĨÓỘÒỎÕÔỐỘỒỔỖƠỚỢỜỞỠÚỤÙỦŨƯỨỰỪỬỮÝỴỲỶỸĐ]+(?=[^aăâeêiouôơưy\s])', '', text)
        
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

class PersonalizedChatContextView(APIView):
    """Lấy context cá nhân hóa cho chat"""
    
    def get(self, request):
        """GET method - Lấy personalized context"""
        try:
            if not request.user.is_authenticated:
                return Response({
                    'personalization_enabled': False,
                    'message': 'User not authenticated'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user = request.user
            user_context = user.get_chatbot_context()
            
            # Thêm thông tin hữu ích cho frontend
            context_info = {
                'personalization_enabled': True,
                'user_context': user_context,
                'personalized_greeting': f"Chào {user_context.get('position_name', 'giảng viên')} {user.full_name}!",
                'department_focus': user_context.get('department_name', 'BDU'),
                'suggested_topics': _get_suggested_topics_for_department(user.department),
                'quick_actions': _get_quick_actions_for_position(user.position),
                'chatbot_tips': [
                    f"Hỏi về thông tin chuyên ngành {user_context.get('department_name')}",
                    f"Tìm hiểu quy định dành cho {user_context.get('position_name')}",
                    "Hỏi về cơ sở vật chất và thiết bị",
                    "Tư vấn về nghiên cứu và hợp tác"
                ]
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
    
    def get(self, request):
        """GET method - System status với personalization"""
        try:
            # Lấy system status cơ bản
            status_data = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()
            
            # Thêm thông tin personalization
            personalization_status = {
                'personalization_enabled': True,
                'total_faculty': 0,
                'departments_available': [],
                'positions_available': []
            }
            
            if request.user.is_authenticated:
                user_context = request.user.get_chatbot_context()
                personalization_status.update({
                    'user_department': user_context.get('department_name', 'Unknown'),
                    'user_position': user_context.get('position_name', 'Unknown'),
                    'has_preferences': bool(request.user.chatbot_preferences),
                    'personalized_prompts_available': True
                })
            
            # Lấy thống kê từ database
            try:
                from authentication.models import Faculty
                personalization_status['total_faculty'] = Faculty.objects.count()
                personalization_status['departments_available'] = [
                    choice[1] for choice in Faculty.DEPARTMENT_CHOICES
                ]
                personalization_status['positions_available'] = [
                    choice[1] for choice in Faculty.POSITION_CHOICES
                ]
            except:
                pass
            
            # Merge với system status
            status_data.update({
                'personalization': personalization_status,
                'speech_status': speech_status
            })
            
            return Response(status_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            return Response({
                'error': 'Could not retrieve system status',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ✅ THÊM: Helper functions (copy từ authentication/views.py)
def _get_suggested_topics_for_department(department):
    """Lấy các chủ đề gợi ý theo ngành"""
    topics_map = {
        'cntt': ['Chương trình đào tạo CNTT', 'Phòng lab tin học', 'Thiết bị máy tính', 'Hợp tác doanh nghiệp IT'],
        'duoc': ['Chương trình đào tạo Dược', 'Phòng thí nghiệm Dược', 'Thiết bị phân tích', 'Thực tập bệnh viện'],
        'dien_tu': ['Chương trình Điện tử', 'Lab vi xử lý', 'Thiết bị đo lường', 'Dự án IoT'],
        'co_khi': ['Chương trình Cơ khí', 'Phòng CAD/CAM', 'Máy gia công CNC', 'Thực tập nhà máy'],
        'y_khoa': ['Chương trình Y khoa', 'Phòng giải phẫu', 'Thực hành lâm sàng', 'Bệnh viện liên kết'],
        'kinh_te': ['Chương trình Kinh tế', 'Phần mềm phân tích', 'Thực tập ngân hàng', 'Nghiên cứu thị trường'],
        'luat': ['Chương trình Luật', 'Phiên tòa giả định', 'Thực tập tòa án', 'Văn phòng luật sư']
    }
    return topics_map.get(department, ['Thông tin chung về trường', 'Quy định đào tạo', 'Cơ sở vật chất'])

def _get_quick_actions_for_position(position):
    """Lấy các quick actions theo chức vụ"""
    actions_map = {
        'giang_vien': ['Xem lịch giảng dạy', 'Quản lý điểm sinh viên', 'Tài liệu giảng dạy', 'Nghiên cứu khoa học'],
        'truong_khoa': ['Quản lý khoa', 'Kế hoạch đào tạo', 'Báo cáo hoạt động', 'Nhân sự khoa'],
        'truong_bo_mon': ['Quản lý bộ môn', 'Phân công giảng dạy', 'Tài liệu chuyên ngành', 'Hoạt động chuyên môn'],
        'tro_giang': ['Hỗ trợ giảng dạy', 'Chuẩn bị bài giảng', 'Chấm bài tập', 'Tương tác sinh viên']
    }
    return actions_map.get(position, ['Thông tin chung', 'Hỗ trợ kỹ thuật', 'Liên hệ phòng ban'])


# ✅ Speech-to-Text Views
class SpeechToTextView(APIView):
    """
    API endpoint for Speech-to-Text conversion
    Accepts audio file upload and returns transcribed text
    """
    
    def get(self, request):
        """GET method - Service information"""
        speech_status = speech_service.get_system_status()
        return Response({
            'message': 'Speech-to-Text API',
            'method': 'POST để upload audio file',
            'speech_service': speech_status,
            'supported_formats': speech_service.supported_formats,
            'max_file_size_mb': speech_service.max_file_size_mb,
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
        """POST method - Process audio file với enhanced debugging"""
        start_time = time.time()
        
        try:
            # Check if service is available
            if not speech_service.is_available():
                logger.error("🚨 Speech service not available")
                return Response({
                    'success': False,
                    'error': 'Speech-to-Text service not available. Please install faster-whisper.',
                    'text': '',
                    'status': speech_service.get_system_status()
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Check if file is in request
            if 'audio' not in request.FILES:
                logger.error("🚨 No audio file in request")
                return Response({
                    'success': False,
                    'error': 'No audio file provided. Please upload an audio file.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            audio_file = request.FILES['audio']
            
            # ✅ ENHANCED DEBUG LOGGING
            logger.info(f"🎤 Received audio file: {audio_file.name}")
            logger.info(f"🎤 File size: {audio_file.size} bytes ({audio_file.size / 1024 / 1024:.2f} MB)")
            logger.info(f"🎤 Content type: {audio_file.content_type}")
            
            # Validate file size
            if audio_file.size > speech_service.max_file_size_mb * 1024 * 1024:
                logger.error(f"🚨 File too large: {audio_file.size} bytes")
                return Response({
                    'success': False,
                    'error': f'File too large. Maximum size: {speech_service.max_file_size_mb}MB'
                }, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            
            # ✅ CHECK: Minimum file size
            if audio_file.size < 1024:  # Less than 1KB
                logger.error(f"🚨 File too small: {audio_file.size} bytes")
                return Response({
                    'success': False,
                    'error': 'Audio file too small. Please record longer audio.',
                    'text': ''
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get optional parameters
            language = request.data.get('language', 'vi')
            beam_size = int(request.data.get('beam_size', 5))
            
            logger.info(f"🎤 Processing with language={language}, beam_size={beam_size}")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(audio_file.name)[1] or '.webm'
            ) as tmp_file:
                # Write file data
                bytes_written = 0
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                    bytes_written += len(chunk)
                tmp_file.flush()
                
                logger.info(f"🎤 Saved temp file: {tmp_file.name} ({bytes_written} bytes)")
                
                try:
                    # Process with speech service
                    logger.info("🔄 Starting transcription...")
                    result = speech_service.transcribe_audio(
                        tmp_file.name,
                        language=language,
                        beam_size=beam_size
                    )
                    
                    # ✅ ENHANCED RESULT LOGGING
                    logger.info(f"🔍 Transcription result: {result}")
                    
                    if result.get('success'):
                        transcribed_text = result.get('text', '').strip()
                        logger.info(f"✅ Transcribed text: '{transcribed_text}' (length: {len(transcribed_text)})")
                        
                        if not transcribed_text:
                            logger.warning("⚠️ Empty transcription result")
                            return Response({
                                'success': False,
                                'error': 'No speech detected in audio. Please speak louder or check microphone.',
                                'text': '',
                                'debug_info': result
                            }, status=status.HTTP_200_OK)
                    else:
                        logger.error(f"❌ Transcription failed: {result.get('error')}")
                    
                    # Add additional metadata
                    result['file_name'] = audio_file.name
                    result['file_size_mb'] = round(audio_file.size / (1024 * 1024), 2)
                    result['total_processing_time'] = time.time() - start_time
                    
                    return Response(result, status=status.HTTP_200_OK)
                    
                finally:
                    # ✅ FIX: Clean up temporary file with better error handling
                    try:
                        if os.path.exists(tmp_file.name):
                            # Brief delay for Windows file system
                            import threading
                            def delayed_cleanup():
                                import time as time_module  # ✅ FIX: Use different name
                                time_module.sleep(0.1)
                                try:
                                    os.unlink(tmp_file.name)
                                    logger.info(f"🗑️ Cleaned up temp file: {tmp_file.name}")
                                except:
                                    pass
                            
                            # Run cleanup in background thread
                            cleanup_thread = threading.Thread(target=delayed_cleanup)
                            cleanup_thread.daemon = True
                            cleanup_thread.start()
                            
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Failed to cleanup temp file: {cleanup_error}")
                        # Not a critical error, continue
        
        except Exception as e:
            logger.error(f"💥 Speech-to-text error: {str(e)}")
            import traceback
            logger.error(f"💥 Full traceback: {traceback.format_exc()}")
            
            return Response({
                'success': False,
                'error': f'Server error: {str(e)}',
                'text': '',
                'processing_time': time.time() - start_time
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SpeechStatusView(APIView):
    """
    Get Speech-to-Text service status and capabilities
    """
    
    def get(self, request):
        """GET method - Service status"""
        try:
            speech_status = speech_service.get_system_status()
            
            return Response({
                'status': 'ok',
                'message': 'Speech-to-Text Service Status',
                'speech_service': speech_status,
                'endpoints': {
                    'speech_to_text': '/api/speech-to-text/',
                    'speech_status': '/api/speech-status/'
                },
                'capabilities': {
                    'languages': ['vi', 'en'],  # Vietnamese and English
                    'supported_formats': speech_service.supported_formats,
                    'max_file_size_mb': speech_service.max_file_size_mb,
                    'features': [
                        'Voice Activity Detection',
                        'Noise Suppression', 
                        'Automatic Language Detection',
                        'GPU Acceleration (if available)'
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
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ EXISTING VIEWS - Unchanged
class ChatHistoryView(APIView):
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

class HealthCheckView(APIView):
    def get(self, request):
        try:
            system_status = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()  # ← THÊM
            
            return Response({
                'status': 'healthy',
                'message': 'Natural Language Chatbot with Speech-to-Text is running! 🚀',
                'database': 'connected',
                'encoding': 'utf-8',
                'system_status': system_status,
                'speech_status': speech_status,  # ← THÊM
                'version': '3.1.0'  # ← Tăng version
            })
        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)