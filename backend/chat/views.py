import jwt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from knowledge.models import ChatHistory, UserFeedback
from ai_models.services import chatbot_ai
from ai_models.speech_service import speech_service, tts_service  # ✅ THÊM TTS SERVICE
import uuid
import time
import logging
import json
import tempfile
import os
import base64  # ✅ THÊM IMPORT BASE64
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from django.db import models

from rest_framework.permissions import AllowAny

logger = logging.getLogger(__name__)

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# ✅ NEW: Helper function to extract JWT token
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

# ✅ NEW: Helper function to validate JWT token format
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

class APIRootView(APIView):
    """API Root - Hiển thị danh sách endpoints"""
    permission_classes = [AllowAny]
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
        speech_status = speech_service.get_system_status()
        
        # ✅ THÊM TTS STATUS
        tts_status = tts_service.get_system_status()
        
        try:
            from ai_models.external_api_service import external_api_service
            external_api_status = external_api_service.get_system_status()
        except ImportError:
            external_api_status = {'external_api_service': {'available': False, 'error': 'Service not imported'}}

        # ✅ ENHANCED: Add personalization status with external API
        personalization_status = {
            'enabled': True,
            'active_personalized_sessions': len(chatbot_ai.response_generator._user_context_cache),
            'user_memory_prompt_support': True,  # ✅ NEW: Updated feature
            'flexible_personalization': True,    # ✅ NEW: Updated feature
            'external_api_integration': external_api_status.get('external_api_service', {}).get('available', False),
            'jwt_token_support': True,
            'lecturer_schedule_access': True
        }
        
        return Response({
            'message': 'Enhanced Chatbot API với Text-to-Speech - Đại học Bình Dương',  # ✅ UPDATED
            'version': '6.1.0',  # ✅ Version bump for TTS feature
            'status': 'active',
            'system_status': system_status,
            'speech_status': speech_status,
            'tts_status': tts_status,  # ✅ THÊM TTS STATUS
            'personalization_status': personalization_status,
            'external_api_status': external_api_status,
            'endpoints': {
                'chat': '/api/chat/',
                'health': '/api/health/',
                'history': '/api/history/',
                'feedback': '/api/feedback/',
                'speech_to_text': '/api/speech-to-text/',
                'speech_status': '/api/speech-status/',
                'personalized_context': '/api/personalized-context/',
                'personalized_status': '/api/personalized-status/',
            },
            'features': [
                'Natural Language Generation',
                'Intent Classification',
                'Conversation Memory',
                'Emotional Context',
                'UTF-8 Safe Encoding',
                'Speech-to-Text (Whisper)',
                'Text-to-Speech (gTTS)',           # ✅ NEW feature
                'Voice Conversation Mode',         # ✅ NEW feature
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
            ]
        })

class ChatView(APIView):
    """Enhanced Chat API with Natural Responses and TTS"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """GET method - API information with personalization and TTS"""
        system_status = chatbot_ai.get_system_status()
        speech_status = speech_service.get_system_status()
        tts_status = tts_service.get_system_status()  # ✅ THÊM TTS STATUS
        
        # ✅ NEW: Get external API status
        try:
            from ai_models.external_api_service import external_api_service
            external_api_status = external_api_service.get_system_status()
        except ImportError:
            external_api_status = {'external_api_service': {'available': False, 'error': 'Service not available'}}
        
        # ✅ UPDATED: Handle unauthenticated users
        user_personalization = None
        if request.user.is_authenticated:
            user_personalization = {
                'faculty_code': request.user.faculty_code,
                'full_name': request.user.full_name,
                'department': request.user.get_department_display(),
                'position': request.user.get_position_display(),
                'has_user_memory_prompt': bool(request.user.chatbot_preferences.get('user_memory_prompt', '').strip()),  # ✅ UPDATED
                'memory_length': len(request.user.chatbot_preferences.get('user_memory_prompt', '')),  # ✅ UPDATED
                'department_priority': request.user.chatbot_preferences.get('department_priority', True),
                'personalized_prompt_available': True
            }
        
        return Response({
            'message': 'Enhanced Personalized Chat API với Text-to-Speech - Open Access',  # ✅ UPDATED
            'authentication': 'Optional - Works with or without token',
            'jwt_token_support': 'Send JWT token for personal schedule/info access',
            'system_status': system_status,
            'speech_status': speech_status,
            'tts_status': tts_status,  # ✅ THÊM TTS STATUS
            'external_api_status': external_api_status,
            'user_personalization': user_personalization,
            'method': 'POST để gửi tin nhắn với personalization, JWT token và TTS',
            'jwt_token_usage': {
                'header': 'Authorization: Bearer <token>',
                'body_field': 'token',
                'query_param': 'token (for testing only)',
                'purpose': 'Access personal schedule and lecturer information'
            },
            'tts_usage': {  # ✅ THÊM HƯỚNG DẪN TTS
                'mode_field': 'mode',
                'voice_mode': 'voice - Tạo audio từ response text',
                'text_mode': 'text - Chỉ trả về text (default)',
                'audio_format': 'MP3 encoded as base64 string',
                'supported_languages': tts_status.get('supported_languages', ['vi', 'en'])
            },
            'features': [
                'PhoBERT Intent Classification',
                'SBERT + FAISS Retrieval',
                'Conversation Memory',
                'UTF-8 Safe Processing',
                'Speech-to-Text Integration',
                'Text-to-Speech Integration (NEW)',        # ✅ NEW
                'Voice Conversation Mode (NEW)',           # ✅ NEW
                'User Memory Prompt Support (with authentication)',
                'Dynamic Personalized System Prompts (with authentication)',
                'Flexible User Instructions (with authentication)',
                'User Memory Integration (with authentication)',
                'Anonymous Chat Support',
                'JWT Token Authentication',
                'External API Integration',
                'Personal Schedule Access',
                'Lecturer Information Queries'
            ]
        })

    def post(self, request):
        """POST method - Process chat with enhanced personalization support and TTS"""
        start_time = time.time()
        
        try:
            # Get and validate input
            user_message = request.data.get('message', '').strip()
            session_id = request.data.get('session_id', str(uuid.uuid4()))
            
            # ✅ BƯỚC 1: ĐỌC "MODE" TỪ REQUEST
            request_mode = request.data.get('mode', 'text').lower()  # Mặc định là 'text'
            logger.info(f"🎯 Request mode: {request_mode}")
            
            # ✅ NEW: Extract JWT token from request
            jwt_token = extract_jwt_token(request)
            
            # ✅ NEW: Log token information for debugging
            if jwt_token:
                is_valid_format, format_message = validate_jwt_token_format(jwt_token)
                logger.info(f"🔑 JWT Token received: format_valid={is_valid_format}, message='{format_message}'")
                
                # Log first and last few characters for debugging (without exposing full token)
                if len(jwt_token) > 20:
                    token_preview = f"{jwt_token[:10]}...{jwt_token[-10:]}"
                else:
                    token_preview = "SHORT_TOKEN"
                logger.info(f"🔑 Token preview: {token_preview}")
                
                if not is_valid_format:
                    logger.warning(f"⚠️ Invalid JWT token format: {format_message}")
                    # Don't fail the request, just log the warning
            else:
                logger.info("🔑 No JWT token provided - using standard QA mode")
            
            # ✅ DEBUG: Log all token extraction sources
            token_from_request = request.data.get('token', 'No token in data')
            token_from_auth = request.META.get('HTTP_AUTHORIZATION', 'No auth header')
            print(f"🔑 TOKEN DEBUG: data='{token_from_request[:20] if isinstance(token_from_request, str) else token_from_request}...', auth='{token_from_auth[:30] if isinstance(token_from_auth, str) else token_from_auth}...'")

            # ✅ ENHANCED: Get user_id and personalization info
            user_id = request.user.id if request.user.is_authenticated else None
            
            print(f"🔍 ENHANCED CHAT DEBUG: user_id = {user_id}, session_id = {session_id}")
            print(f"🔍 ENHANCED CHAT DEBUG: User message = {user_message}")
            print(f"🔊 TTS MODE DEBUG: mode = {request_mode}")
            
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
            
            # ✅ ENHANCED: Get comprehensive user context với user memory prompt
            user_context = None
            personalization_info = {}
            
            if user_id and request.user.is_authenticated:
                try:
                    user_context = request.user.get_chatbot_context()
                    
                    # ✅ UPDATED: Extract personalization info for user memory prompt
                    personalization_info = {
                        'department_priority': user_context.get('department_priority_enabled', True),
                        'department': user_context.get('department_name', 'Unknown'),
                        'position': user_context.get('position_name', 'Unknown'),
                        'has_user_memory_prompt': bool(request.user.chatbot_preferences.get('user_memory_prompt', '').strip()),  # ✅ UPDATED
                        'memory_length': len(request.user.chatbot_preferences.get('user_memory_prompt', '')),  # ✅ UPDATED
                        'personalized_prompt_available': True
                    }
                    
                    print(f"👤 ENHANCED USER CONTEXT: {user_context.get('role_description', 'Unknown')}")
                    print(f"🧠 USER MEMORY PROMPT: length={personalization_info['memory_length']}, has_custom={personalization_info['has_user_memory_prompt']}")
                    
                except Exception as e:
                    logger.warning(f"Could not get enhanced user context: {e}")
                    personalization_info['error'] = str(e)
            
            logger.info(f"💬 Processing with enhanced personalization + JWT + TTS: {user_message[:50]}... (User: {user_context.get('faculty_code') if user_context else 'Anonymous'}, JWT: {bool(jwt_token)}, Mode: {request_mode})")

            # ✅ ENHANCED: Process với comprehensive user context
            if user_context:
                # Set enhanced user context vào gemini service
                enhanced_context = {
                    'personalized_prompt': request.user.get_personalized_system_prompt(),
                    'faculty_code': user_context.get('faculty_code'),
                    'full_name': user_context.get('full_name'),
                    'department': user_context.get('department'),
                    'department_name': user_context.get('department_name'),
                    'position_name': user_context.get('position_name'),
                    'preferences': user_context.get('preferences'),
                    # ✅ UPDATED: User memory prompt specific context
                    'user_memory_prompt': request.user.chatbot_preferences.get('user_memory_prompt', ''),  # ✅ UPDATED
                    'department_priority_enabled': personalization_info.get('department_priority', True)
                }
                
                chatbot_ai.response_generator.set_user_context(session_id, enhanced_context)
                
                # ✅ NEW: Process with personalization AND JWT token
                ai_response = chatbot_ai.process_query(user_message, session_id, jwt_token)
            else:
                # ✅ NEW: Process without personalization but WITH JWT token for external API
                ai_response = chatbot_ai.process_query(user_message, session_id, jwt_token)
            
            print(f"🔍 ENHANCED CHAT DEBUG: AI response method = {ai_response.get('method', 'unknown')}")
            print(f"🧠 ENHANCED CHAT DEBUG: User memory prompt used = {ai_response.get('user_memory_prompt_used', False)}")  # ✅ UPDATED
            print(f"🌐 ENHANCED CHAT DEBUG: External API used = {ai_response.get('external_api_used', False)}")
            
            # ENSURE UTF-8 safe response
            response_text = ai_response['response']
            try:
                response_text = response_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean response text
            response_text = self._clean_response_text(response_text)
            
            # ✅ BƯỚC 2: KIỂM TRA MODE VÀ TẠO AUDIO
            audio_content_base64 = None
            tts_processing_time = 0
            tts_error = None
            
            if request_mode == 'voice' and response_text:
                logger.info("🔊 Voice mode detected. Generating TTS response...")
                tts_start_time = time.time()
                
                try:
                    audio_content_base64 = tts_service.text_to_audio_base64(response_text)
                    tts_processing_time = time.time() - tts_start_time
                    
                    if audio_content_base64:
                        logger.info(f"✅ TTS audio generated successfully in {tts_processing_time:.2f}s")
                    else:
                        logger.warning("⚠️ TTS audio generation failed - no audio returned")
                        tts_error = "TTS service returned no audio"
                        
                except Exception as e:
                    tts_processing_time = time.time() - tts_start_time
                    tts_error = str(e)
                    logger.error(f"❌ TTS audio generation failed: {e}")
            elif request_mode == 'voice':
                logger.warning("⚠️ Voice mode requested but no response text available")
                tts_error = "No response text available for TTS"
            else:
                logger.info(f"📝 Text mode - no TTS processing (mode: {request_mode})")
            
            processing_time = time.time() - start_time
            
            # ✅ ENHANCED: Save chat history với JWT, external API info và TTS info
            try:
                # Enhanced entities with personalization + external API info + TTS info
                enhanced_entities = {
                    'user_context': user_context,
                    'personalization_info': personalization_info,
                    'personalized': bool(user_context),
                    'user_memory_prompt_applied': ai_response.get('user_memory_prompt_used', False),  # ✅ UPDATED
                    'department_priority_used': user_context.get('department_priority_enabled') if user_context else False,
                    # ✅ NEW: External API related fields
                    'jwt_token_provided': bool(jwt_token),
                    'external_api_used': ai_response.get('external_api_used', False),
                    'external_api_method': ai_response.get('method', '') if ai_response.get('external_api_used') else None,
                    'decision_type': ai_response.get('decision_type', ''),
                    'token_preview': f"{jwt_token[:10]}...{jwt_token[-10:]}" if jwt_token and len(jwt_token) > 20 else None,
                    # ✅ NEW: TTS related fields
                    'request_mode': request_mode,
                    'tts_generated': bool(audio_content_base64),
                    'tts_processing_time': tts_processing_time,
                    'tts_error': tts_error,
                    'reference_links': ai_response.get('reference_links', [])
                }
                
                chat_record = ChatHistory.objects.create(
                    session_id=session_id,
                    user_message=user_message,
                    bot_response=response_text,
                    confidence_score=ai_response.get('confidence', 0.7),
                    response_time=processing_time,
                    user_ip=get_client_ip(request),
                    user=request.user if request.user.is_authenticated else None,
                    entities=json.dumps(enhanced_entities) if enhanced_entities else None  # ✅ Enhanced entities
                )
                logger.info(f"✅ Enhanced chat with external API and TTS saved: {chat_record.id}")
            except Exception as e:
                logger.error(f"Error saving enhanced chat: {str(e)}")
            
            # ✅ BƯỚC 3: CẬP NHẬT JSON RESPONSE
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
                'reference_links': ai_response.get('reference_links', []),
                
                # ✅ CÁC TRƯỜNG MỚI CHO TTS
                'audio_content': audio_content_base64,  # Sẽ là null nếu mode='text'
                'mode': request_mode,
                'tts_info': {
                    'enabled': request_mode == 'voice',
                    'processing_time': tts_processing_time,
                    'success': bool(audio_content_base64),
                    'error': tts_error,
                    'audio_format': 'mp3_base64' if audio_content_base64 else None
                },
                # -----------------------------
                
                # ✅ ENHANCED: Detailed personalization response
                'personalization': {
                    'enabled': bool(user_context),
                    'user_info': {
                        'department': personalization_info.get('department'),
                        'position': personalization_info.get('position'),
                        'faculty_code': user_context.get('faculty_code') if user_context else None
                    } if user_context else None,
                    'user_memory_info': {  # ✅ UPDATED: User memory prompt info
                        'has_user_memory_prompt': personalization_info.get('has_user_memory_prompt', False),
                        'memory_length': personalization_info.get('memory_length', 0),
                        'memory_applied': ai_response.get('user_memory_prompt_used', False)
                    } if user_context else None,
                    'department_priority_used': personalization_info.get('department_priority', False)
                },
                
                # ✅ NEW: External API information
                'external_api': {
                    'jwt_token_provided': bool(jwt_token),
                    'external_api_used': ai_response.get('external_api_used', False),
                    'decision_type': ai_response.get('decision_type', ''),
                    'method_used': ai_response.get('method', ''),
                    'personal_info_accessed': ai_response.get('external_api_used', False),
                    'token_valid_format': validate_jwt_token_format(jwt_token)[0] if jwt_token else None
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"❌ Enhanced chat error with external API and TTS: {str(e)}")
            
            # Enhanced fallback response với personalization + external API + TTS
            fallback_response = self._get_enhanced_fallback_response_with_external_api_and_tts(
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
                'method': 'enhanced_fallback_with_external_api_and_tts',
                'response_time': time.time() - start_time,
                'status': 'fallback',
                'audio_content': None,  # ✅ Không tạo TTS cho fallback
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
                    'fallback_used': True,
                    'error': str(e)
                },
                'external_api': {
                    'jwt_token_provided': bool(locals().get('jwt_token')),
                    'external_api_used': False,
                    'fallback_used': True,
                    'error': str(e)
                }
            }) 
    
    def _get_enhanced_fallback_response_with_external_api_and_tts(self, user_message='', user_context=None, personalization_info={}, jwt_token=None, request_mode='text'):
        """Enhanced fallback response với comprehensive personalization + external API + TTS"""
        if user_context:
            full_name = user_context.get('full_name', '')
            faculty_code = user_context.get('faculty_code', '')
            name_suffix = full_name.split()[-1] if full_name else faculty_code
            personal_address = f"thầy/cô {name_suffix}"
            department_name = user_context.get('department_name', 'BDU')
            
            # ✅ NEW: Different messages based on JWT token availability and TTS mode
            if jwt_token:
                # Have JWT token but still failed
                base_message = f"""Dạ xin lỗi {personal_address}, hệ thống đang được nâng cấp để phục vụ {personal_address} tốt hơn.

Mặc dù em đã nhận được thông tin đăng nhập của {personal_address}, nhưng hiện tại có một số khó khăn kỹ thuật. 

{personal_address} có thể:
• Thử lại sau vài phút ⏰
• Truy cập trực tiếp hệ thống quản lý đào tạo của trường 🌐
• Liên hệ khoa {department_name} để được hỗ trợ trực tiếp 📞
• Gọi bộ phận IT: 0274.xxx.xxxx 📧

Em sẽ cố gắng khắc phục để phục vụ {personal_address} tốt hơn! 🎓✨"""
            else:
                # No JWT token - use user memory prompt considerations
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
            
            # ✅ Add TTS-specific note for voice mode
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
        """Clean and ensure safe UTF-8 text (unchanged from original)"""
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
        """Safe fallback response with proper UTF-8 (unchanged from original)"""
        return f"""Xin chào! Tôi đã nhận được câu hỏi của bạn. 

Hiện tại hệ thống đang được cải thiện để phục vụ bạn tốt hơn. Trong thời gian này, bạn có thể:

• Liên hệ trực tiếp: 0274.xxx.xxxx
• Email: info@bdu.edu.vn  
• Website: www.bdu.edu.vn

Cảm ơn bạn đã kiên nhẫn! 😊"""

class PersonalizedChatContextView(APIView):
    """Lấy context cá nhân hóa cho chat"""
    
    def get(self, request):
        """GET method - Enhanced personalized context with TTS"""
        try:
            if not request.user.is_authenticated:
                return Response({
                    'personalization_enabled': False,
                    'message': 'User not authenticated'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user = request.user
            user_context = user.get_chatbot_context()
            
            # ✅ UPDATED: Enhanced context info với user memory prompt
            user_memory_prompt = user.chatbot_preferences.get('user_memory_prompt', '').strip()
            
            # ✅ THÊM TTS STATUS
            tts_status = tts_service.get_system_status()
            
            context_info = {
                'personalization_enabled': True,
                'user_context': user_context,
                'personalized_greeting': f"Chào {user_context.get('position_name', 'giảng viên')} {user.full_name}!",
                'department_focus': user_context.get('department_name', 'BDU'),
                
                # ✅ UPDATED: User memory prompt information
                'user_memory_info': {
                    'has_user_memory_prompt': bool(user_memory_prompt),
                    'memory_length': len(user_memory_prompt),
                    'memory_preview': user_memory_prompt[:150] + '...' if len(user_memory_prompt) > 150 else user_memory_prompt,
                    'using_default_prompt': not bool(user_memory_prompt),
                    'memory_effectiveness': 'high' if len(user_memory_prompt) > 100 else 'medium' if len(user_memory_prompt) > 50 else 'low'
                },
                
                # ✅ THÊM TTS CAPABILITIES
                'tts_capabilities': {
                    'available': tts_status.get('available', False),
                    'supported_languages': tts_status.get('supported_languages', []),
                    'default_language': tts_status.get('default_language', 'vi'),
                    'voice_mode_enabled': tts_status.get('available', False)
                },
                
                # ✅ ENHANCED: Better suggested topics
                'suggested_topics': self._get_enhanced_suggested_topics_for_department(user.department),
                'quick_actions': self._get_quick_actions_for_position(user.position),
                
                # ✅ UPDATED: Personalization tips WITH USER MEMORY PROMPT AND TTS
                'personalization_tips': [
                    f"Sử dụng 'Ghi nhớ và chỉ dẫn' để ChatBDU hiểu và phục vụ bạn tốt hơn",
                    f"Viết những quy tắc, sở thích riêng vào ô 'User Memory Prompt'", 
                    f"Hỏi về thông tin chuyên ngành {user_context.get('department_name')}",
                    "Bật/tắt ưu tiên chuyên ngành theo nhu cầu",
                    "Đăng nhập ứng dụng để truy cập lịch giảng dạy cá nhân",
                    "Hỏi về 'lịch của tôi' để xem thời khóa biểu riêng",
                    "Sử dụng chế độ giọng nói để trò chuyện tự nhiên hơn 🎤🔊"  # ✅ NEW TTS tip
                ],
                
                # ✅ UPDATED: External API capabilities
                'external_api_features': {
                    'personal_schedule_access': True,
                    'lecturer_info_access': True,
                    'jwt_token_required': True,
                    'example_queries': [
                        "Lịch giảng dạy của tôi hôm nay",
                        "Thông tin cá nhân của tôi", 
                        "Tôi dạy môn gì tuần này?",
                        "Lịch làm việc ngày mai"
                    ]
                },
                
                # ✅ UPDATED: Settings summary
                'current_settings': {
                    'department_priority': user.chatbot_preferences.get('department_priority', True),
                    'has_user_memory_prompt': bool(user_memory_prompt),
                    'memory_prompt_length': len(user_memory_prompt),
                    'total_preferences': len(user.chatbot_preferences),
                    'external_api_ready': True,
                    'tts_enabled': tts_status.get('available', False),  # ✅ NEW
                    'voice_conversation_ready': tts_status.get('available', False),  # ✅ NEW
                    'personalization_strength': 'high' if bool(user_memory_prompt) else 'medium'  # ✅ NEW
                }
            }
            
            return Response(context_info, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Enhanced personalized context error: {str(e)}")
            return Response({
                'personalization_enabled': False,
                'error': 'Could not load enhanced personalized context',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _get_enhanced_suggested_topics_for_department(self, department):
        """Enhanced suggested topics based on department"""
        base_topics = {
            'cntt': ['Chương trình đào tạo CNTT', 'Phòng lab tin học', 'Thiết bị máy tính', 'Hợp tác doanh nghiệp IT'],
            'duoc': ['Chương trình đào tạo Dược', 'Phòng thí nghiệm Dược', 'Thiết bị phân tích', 'Thực tập bệnh viện'],
            'dien_tu': ['Chương trình Điện tử', 'Lab vi xử lý', 'Thiết bị đo lường', 'Dự án IoT'],
            'co_khi': ['Chương trình Cơ khí', 'Phòng CAD/CAM', 'Máy gia công CNC', 'Thực tập nhà máy'],
            'y_khoa': ['Chương trình Y khoa', 'Phòng giải phẫu', 'Thực hành lâm sàng', 'Bệnh viện liên kết'],
            'kinh_te': ['Chương trình Kinh tế', 'Phần mềm phân tích', 'Thực tập ngân hàng', 'Nghiên cứu thị trường'],
            'luat': ['Chương trình Luật', 'Phiên tòa giả định', 'Thực tập tòa án', 'Văn phòng luật sư']
        }
        
        topics = base_topics.get(department, ['Thông tin chung về trường', 'Quy định đào tạo', 'Cơ sở vật chất'])
        
        # ✅ NEW: Add personal schedule topics
        topics.extend([
            'Lịch giảng dạy của tôi',
            'Thông tin cá nhân của tôi',
            'Môn học tôi phụ trách'
        ])
        
        return topics

    def _get_quick_actions_for_position(self, position):
        """Quick actions based on position"""
        base_actions = {
            'giang_vien': ['Xem lịch giảng dạy', 'Quản lý điểm sinh viên', 'Tài liệu giảng dạy', 'Nghiên cứu khoa học'],
            'truong_khoa': ['Quản lý khoa', 'Kế hoạch đào tạo', 'Báo cáo hoạt động', 'Nhân sự khoa'],
            'truong_bo_mon': ['Quản lý bộ môn', 'Phân công giảng dạy', 'Tài liệu chuyên ngành', 'Hoạt động chuyên môn'],
            'tro_giang': ['Hỗ trợ giảng dạy', 'Chuẩn bị bài giảng', 'Chấm bài tập', 'Tương tác sinh viên']
        }
        
        actions = base_actions.get(position, ['Thông tin chung', 'Hỗ trợ kỹ thuật', 'Liên hệ phòng ban'])
        
        # ✅ NEW: Add personal actions for all positions
        actions.extend([
            'Xem lịch cá nhân của tôi',
            'Thông tin tài khoản của tôi',
            'Lịch làm việc hôm nay'
        ])
        
        return actions

class PersonalizedSystemStatusView(APIView):
    """System status với thông tin personalization và TTS"""
    
    def get(self, request):
        """GET method - Enhanced system status với personalization + external API + TTS"""
        try:
            # Base system status
            status_data = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()
            tts_status = tts_service.get_system_status()  # ✅ THÊM TTS STATUS
            
            # ✅ NEW: Get external API status
            try:
                from ai_models.external_api_service import external_api_service
                external_api_status = external_api_service.get_system_status()
            except ImportError:
                external_api_status = {'external_api_service': {'available': False, 'error': 'Service not available'}}
            
            # ✅ UPDATED: Comprehensive personalization status với user memory prompt và TTS
            personalization_status = {
                'personalization_enabled': True,
                'version': '6.1.0',  # ✅ Version bump for TTS feature
                'features': {
                    'user_memory_prompt_support': True,     # ✅ NEW
                    'flexible_personalization': True,      # ✅ NEW
                    'dynamic_system_prompts': True,         # ✅ NEW
                    'custom_user_instructions': True,       # ✅ NEW
                    'department_priority': True,
                    'personalized_addressing': True,
                    'jwt_token_authentication': True,
                    'external_api_integration': True,
                    'personal_schedule_access': True,
                    'lecturer_info_queries': True,
                    'text_to_speech_support': True,         # ✅ NEW TTS
                    'voice_conversation_mode': True,        # ✅ NEW TTS
                    'speech_to_text_support': True,
                    'full_voice_interaction': True          # ✅ NEW - STT + TTS combined
                },
                'statistics': {
                    'total_faculty': 0,
                    'active_personalized_sessions': len(chatbot_ai.response_generator._user_context_cache),
                    'departments_available': 0,  # Will be updated below
                    'positions_available': 0     # Will be updated below
                },
                'external_api_integration': external_api_status,
                'tts_integration': tts_status  # ✅ THÊM TTS INTEGRATION
            }
            
            # Add current user info if authenticated
            if request.user.is_authenticated:
                user_memory_prompt = request.user.chatbot_preferences.get('user_memory_prompt', '').strip()
                personalization_status['current_user'] = {
                    'faculty_code': request.user.faculty_code,
                    'department': request.user.get_department_display(),
                    'position': request.user.get_position_display(),
                    'has_user_memory_prompt': bool(user_memory_prompt),  # ✅ UPDATED
                    'memory_prompt_length': len(user_memory_prompt),     # ✅ UPDATED
                    'department_priority': request.user.chatbot_preferences.get('department_priority', True),
                    'preferences_configured': bool(request.user.chatbot_preferences),
                    'external_api_ready': True,
                    'tts_ready': tts_status.get('available', False),  # ✅ NEW TTS
                    'voice_interaction_ready': tts_status.get('available', False) and speech_status.get('available', False),  # ✅ NEW
                    'personalization_strength': 'high' if bool(user_memory_prompt) else 'medium'  # ✅ NEW
                }
            
            # Get statistics from database
            try:
                from authentication.models import Faculty
                personalization_status['statistics']['total_faculty'] = Faculty.objects.count()
                personalization_status['statistics']['active_faculty'] = Faculty.objects.filter(is_active_faculty=True).count()
                personalization_status['statistics']['with_personalization'] = Faculty.objects.exclude(chatbot_preferences={}).count()
                
                # ✅ UPDATED: User memory prompt statistics
                faculty_with_memory = Faculty.objects.filter(
                    chatbot_preferences__user_memory_prompt__isnull=False
                ).exclude(
                    chatbot_preferences__user_memory_prompt__exact=''
                ).count()
                
                personalization_status['statistics']['with_user_memory_prompt'] = faculty_with_memory
                personalization_status['statistics']['memory_prompt_adoption_rate'] = (
                    faculty_with_memory / max(1, personalization_status['statistics']['total_faculty']) * 100
                )
                
            except Exception as e:
                personalization_status['statistics']['database_error'] = str(e)
            
            # Merge with system status
            status_data.update({
                'personalization': personalization_status,
                'speech_status': speech_status,
                'tts_status': tts_status,  # ✅ THÊM TTS STATUS
                'external_api_status': external_api_status
            })
            
            return Response(status_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Enhanced system status error: {str(e)}")
            return Response({
                'error': 'Could not retrieve enhanced system status',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ Speech-to-Text Views (unchanged)
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
        """GET method - Service status including TTS"""
        try:
            speech_status = speech_service.get_system_status()
            tts_status = tts_service.get_system_status()  # ✅ THÊM TTS STATUS
            
            return Response({
                'status': 'ok',
                'message': 'Speech Services Status (STT + TTS)',  # ✅ UPDATED
                'speech_service': speech_status,
                'tts_service': tts_status,  # ✅ THÊM TTS SERVICE STATUS
                'endpoints': {
                    'speech_to_text': '/api/speech-to-text/',
                    'speech_status': '/api/speech-status/'
                },
                'capabilities': {
                    'stt_languages': ['vi', 'en'],  # Vietnamese and English
                    'tts_languages': tts_status.get('supported_languages', ['vi', 'en']),  # ✅ NEW
                    'supported_formats': speech_service.supported_formats,
                    'max_file_size_mb': speech_service.max_file_size_mb,
                    'features': [
                        'Voice Activity Detection',
                        'Noise Suppression', 
                        'Automatic Language Detection',
                        'GPU Acceleration (if available)',
                        'Text-to-Speech (gTTS)',           # ✅ NEW
                        'Voice Conversation Mode',         # ✅ NEW
                        'Multi-language TTS Support'       # ✅ NEW
                    ]
                },
                'voice_interaction': {  # ✅ NEW SECTION
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

class ChatSessionsView(APIView):
    """Quản lý chat sessions của user"""
    
    def get(self, request):
        """Lấy danh sách sessions của user"""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        try:
            # Lấy các sessions duy nhất của user
            sessions = ChatHistory.objects.filter(user=request.user) \
                .values('session_id', 'session_title') \
                .annotate(
                    last_message_time=models.Max('timestamp'),
                    message_count=models.Count('id')
                ) \
                .order_by('-last_message_time')[:20]  # Giới hạn 20 sessions gần nhất
            
            sessions_list = []
            for session in sessions:
                # Lấy tin nhắn cuối cùng để làm preview
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
                    'active': False  # Frontend sẽ set active
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
            new_session_id = f"session_{request.user.faculty_code}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
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
                    'reference_links': bot_entities.get('reference_links', []),
                    'chat_id': chat.id
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
        """✅ THÊM METHOD MỚI: Cập nhật thông tin session (rename)"""
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
            
            if len(new_title) > 200:
                return Response({
                    'success': False,
                    'error': 'Title quá dài (tối đa 200 ký tự)'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Tìm một chat record bất kỳ của session này để cập nhật session_title
            chat_record = ChatHistory.objects.filter(
                user=request.user,
                session_id=session_id
            ).first()
            
            if not chat_record:
                return Response({
                    'success': False,
                    'error': 'Session không tồn tại hoặc không thuộc về bạn'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Cập nhật session_title cho tất cả chat records của session này
            updated_count = ChatHistory.objects.filter(
                user=request.user,
                session_id=session_id
            ).update(session_title=new_title)
            
            logger.info(f"Updated session title for {updated_count} records: {session_id} -> '{new_title}'")
            
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

class TextToSpeechTestView(APIView):
    """
    ✅ NEW: Endpoint test riêng cho TTS service
    """
    permission_classes = [AllowAny]
    
    def get(self, request):
        """GET method - TTS service information"""
        tts_status = tts_service.get_system_status()
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
                },
                'response': {
                    'success': 'Boolean indicating success',
                    'audio_content': 'Base64 encoded MP3 audio (if successful)',
                    'text_processed': 'The text that was processed',
                    'processing_time': 'Time taken to generate audio'
                }
            },
            'examples': {
                'vietnamese': {
                    'text': 'Xin chào, tôi là trợ lý AI của Đại học Bình Dương',
                    'language': 'vi'
                },
                'english': {
                    'text': 'Hello, I am the AI assistant of Binh Duong University',
                    'language': 'en'
                }
            }
        })
    
    def post(self, request):
        """POST method - Test TTS conversion"""
        start_time = time.time()
        
        try:
            # Check if TTS service is available
            if not tts_service.is_available:
                return Response({
                    'success': False,
                    'error': 'TTS service not available. Please install gTTS.',
                    'audio_content': None,
                    'tts_status': tts_service.get_system_status()
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Get and validate input
            text_to_convert = request.data.get('text', '').strip()
            language = request.data.get('language', 'vi')
            slow = request.data.get('slow', False)
            
            if not text_to_convert:
                return Response({
                    'success': False,
                    'error': 'Text field is required and cannot be empty.',
                    'audio_content': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(text_to_convert) > 1000:
                return Response({
                    'success': False,
                    'error': 'Text too long. Maximum 1000 characters.',
                    'audio_content': None
                }, status=status.HTTP_400_BAD_REQUEST)
            
            logger.info(f"🔊 TTS Test: Converting text to speech: '{text_to_convert[:50]}...' (lang: {language}, slow: {slow})")
            
            # Generate TTS audio
            audio_base64 = tts_service.text_to_audio_base64(
                text=text_to_convert,
                language=language,
                slow=slow
            )
            
            processing_time = time.time() - start_time
            
            if audio_base64:
                logger.info(f"✅ TTS Test: Successfully generated audio in {processing_time:.2f}s")
                return Response({
                    'success': True,
                    'audio_content': audio_base64,
                    'text_processed': text_to_convert,
                    'language': language,
                    'slow': slow,
                    'processing_time': processing_time,
                    'audio_format': 'mp3_base64',
                    'audio_size_chars': len(audio_base64),
                    'message': 'TTS conversion successful! Use the audio_content in your frontend.'
                })
            else:
                logger.error(f"❌ TTS Test: Failed to generate audio")
                return Response({
                    'success': False,
                    'error': 'Failed to generate TTS audio. Check server logs for details.',
                    'audio_content': None,
                    'text_processed': text_to_convert,
                    'processing_time': processing_time
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"💥 TTS Test error: {str(e)}")
            
            return Response({
                'success': False,
                'error': f'TTS test failed: {str(e)}',
                'audio_content': None,
                'processing_time': processing_time
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HealthCheckView(APIView):
    def get(self, request):
        try:
            system_status = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()
            tts_status = tts_service.get_system_status()  # ✅ THÊM TTS STATUS
            
            return Response({
                'status': 'healthy',
                'message': 'Enhanced Personalized Chatbot với Text-to-Speech is running! 🚀🔊',  # ✅ UPDATED
                'database': 'connected',
                'encoding': 'utf-8',
                'system_status': system_status,
                'speech_status': speech_status,
                'tts_status': tts_status,  # ✅ THÊM TTS STATUS
                'personalization': 'enabled',
                'voice_interaction': {  # ✅ NEW
                    'stt_available': speech_status.get('available', False),
                    'tts_available': tts_status.get('available', False),
                    'full_voice_chat': speech_status.get('available', False) and tts_status.get('available', False)
                },
                'version': '6.1.0'  # ✅ Updated version for TTS
            })
        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)