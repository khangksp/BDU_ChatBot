import jwt
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
        
        try:
            from ai_models.external_api_service import external_api_service
            external_api_status = external_api_service.get_system_status()
        except ImportError:
            external_api_status = {'external_api_service': {'available': False, 'error': 'Service not imported'}}

        # ✅ ENHANCED: Add personalization status with external API
        personalization_status = {
            'enabled': True,
            'active_personalized_sessions': len(chatbot_ai.response_generator._user_context_cache),
            'supported_response_styles': list(chatbot_ai.response_generator.style_generation_configs.keys()),
            'style_aware_generation': True,
            'external_api_integration': external_api_status.get('external_api_service', {}).get('available', False),  # ✅ NEW
            'jwt_token_support': True,  # ✅ NEW
            'lecturer_schedule_access': True  # ✅ NEW
        }
        
        return Response({
            'message': 'Enhanced Chatbot API - Đại học Bình Dương với External API Integration',  # ✅ UPDATED
            'version': '5.0.0',  # ✅ Version bump for external API
            'status': 'active',
            'system_status': system_status,
            'speech_status': speech_status,
            'personalization_status': personalization_status,
            'external_api_status': external_api_status,  # ✅ NEW
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
                'Enhanced Personalization',
                'Response Style Adaptation',
                'User Memory Integration',
                'Department-Specific Responses',
                'JWT Token Authentication',  # ✅ NEW
                'External API Integration',  # ✅ NEW
                'Lecturer Schedule Access',  # ✅ NEW
                'Personal Information Queries',  # ✅ NEW
            ]
        })

class ChatView(APIView):
    """Enhanced Chat API with Natural Responses"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """GET method - API information with personalization"""
        system_status = chatbot_ai.get_system_status()
        speech_status = speech_service.get_system_status()
        
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
                'current_style': request.user.chatbot_preferences.get('response_style', 'professional'),
                'department_priority': request.user.chatbot_preferences.get('department_priority', True),
                'has_custom_memory': bool(request.user.chatbot_preferences.get('user_memory_prompt', '').strip())
            }
        
        return Response({
            'message': 'Enhanced Personalized Chat API with External API - Open Access',  # ✅ UPDATED
            'authentication': 'Optional - Works with or without token',
            'jwt_token_support': 'Send JWT token for personal schedule/info access',  # ✅ NEW
            'system_status': system_status,
            'speech_status': speech_status,
            'external_api_status': external_api_status,  # ✅ NEW
            'user_personalization': user_personalization,
            'method': 'POST để gửi tin nhắn với personalization và JWT token',  # ✅ UPDATED
            'jwt_token_usage': {  # ✅ NEW
                'header': 'Authorization: Bearer <token>',
                'body_field': 'token',
                'query_param': 'token (for testing only)',
                'purpose': 'Access personal schedule and lecturer information'
            },
            'features': [
                'PhoBERT Intent Classification',
                'SBERT + FAISS Retrieval',
                'Conversation Memory',
                'UTF-8 Safe Processing',
                'Speech-to-Text Integration',
                'Response Style Adaptation (with authentication)',
                'Personalized System Prompts (with authentication)',
                'User Memory Integration (with authentication)',
                'Anonymous Chat Support',
                'JWT Token Authentication',  # ✅ NEW
                'External API Integration',  # ✅ NEW
                'Personal Schedule Access',  # ✅ NEW
                'Lecturer Information Queries'  # ✅ NEW
            ]
        })

    def post(self, request):
        """POST method - Process chat with enhanced personalization support"""
        start_time = time.time()
        
        try:
            # Get and validate input
            user_message = request.data.get('message', '').strip()
            session_id = request.data.get('session_id', str(uuid.uuid4()))
            
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
            
            # ✅ ENHANCED: Get comprehensive user context với detailed preferences
            user_context = None
            personalization_info = {}
            
            if user_id and request.user.is_authenticated:
                try:
                    user_context = request.user.get_chatbot_context()
                    
                    # ✅ NEW: Extract detailed personalization info
                    personalization_info = {
                        'response_style': user_context.get('current_response_style', 'professional'),
                        'department_priority': user_context.get('department_priority_enabled', True),
                        'department': user_context.get('department_name', 'Unknown'),
                        'position': user_context.get('position_name', 'Unknown'),
                        'has_custom_memory': bool(request.user.chatbot_preferences.get('user_memory_prompt', '').strip()),
                        'memory_length': len(request.user.chatbot_preferences.get('user_memory_prompt', '')),
                        'personalized_prompt_available': True
                    }
                    
                    print(f"👤 ENHANCED USER CONTEXT: {user_context.get('role_description', 'Unknown')}")
                    print(f"🎨 PERSONALIZATION INFO: Style={personalization_info['response_style']}, Dept_Priority={personalization_info['department_priority']}")
                    
                except Exception as e:
                    logger.warning(f"Could not get enhanced user context: {e}")
                    personalization_info['error'] = str(e)
            
            logger.info(f"💬 Processing with enhanced personalization + JWT: {user_message[:50]}... (User: {user_context.get('faculty_code') if user_context else 'Anonymous'}, JWT: {bool(jwt_token)})")

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
                    # ✅ NEW: Style and memory specific context
                    'response_style': personalization_info.get('response_style', 'professional'),
                    'user_memory_prompt': request.user.chatbot_preferences.get('user_memory_prompt', ''),
                    'department_priority_enabled': personalization_info.get('department_priority', True)
                }
                
                chatbot_ai.response_generator.set_user_context(session_id, enhanced_context)
                
                # ✅ NEW: Process with personalization AND JWT token
                ai_response = chatbot_ai.process_query(user_message, session_id, jwt_token)
            else:
                # ✅ NEW: Process without personalization but WITH JWT token for external API
                ai_response = chatbot_ai.process_query(user_message, session_id, jwt_token)
            
            print(f"🔍 ENHANCED CHAT DEBUG: AI response method = {ai_response.get('method', 'unknown')}")
            print(f"🎨 ENHANCED CHAT DEBUG: Applied style = {ai_response.get('response_style', 'none')}")
            print(f"🌐 ENHANCED CHAT DEBUG: External API used = {ai_response.get('external_api_used', False)}")
            
            # ENSURE UTF-8 safe response
            response_text = ai_response['response']
            try:
                response_text = response_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean response text
            response_text = self._clean_response_text(response_text)
            
            processing_time = time.time() - start_time
            
            # ✅ ENHANCED: Save chat history với JWT and external API info
            try:
                # Enhanced entities with personalization + external API info
                enhanced_entities = {
                    'user_context': user_context,
                    'personalization_info': personalization_info,
                    'personalized': bool(user_context),
                    'response_style_applied': ai_response.get('response_style', 'none'),
                    'style_applied': ai_response.get('style_applied', 'none'),
                    'department_priority_used': user_context.get('department_priority_enabled') if user_context else False,
                    # ✅ NEW: External API related fields
                    'jwt_token_provided': bool(jwt_token),
                    'external_api_used': ai_response.get('external_api_used', False),
                    'external_api_method': ai_response.get('method', '') if ai_response.get('external_api_used') else None,
                    'decision_type': ai_response.get('decision_type', ''),
                    'token_preview': f"{jwt_token[:10]}...{jwt_token[-10:]}" if jwt_token and len(jwt_token) > 20 else None
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
                logger.info(f"✅ Enhanced chat with external API saved: {chat_record.id}")
            except Exception as e:
                logger.error(f"Error saving enhanced chat: {str(e)}")
            
            # ✅ ENHANCED: Return comprehensive response with external API details
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
                
                # ✅ ENHANCED: Detailed personalization response
                'personalization': {
                    'enabled': bool(user_context),
                    'user_info': {
                        'department': personalization_info.get('department'),
                        'position': personalization_info.get('position'),
                        'faculty_code': user_context.get('faculty_code') if user_context else None
                    } if user_context else None,
                    'style_info': {
                        'requested_style': personalization_info.get('response_style', 'professional'),
                        'applied_style': ai_response.get('response_style', 'none'),
                        'style_applied_successfully': ai_response.get('response_style') == personalization_info.get('response_style')
                    } if user_context else None,
                    'memory_info': {
                        'has_custom_memory': personalization_info.get('has_custom_memory', False),
                        'memory_length': personalization_info.get('memory_length', 0)
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
            logger.error(f"❌ Enhanced chat error with external API: {str(e)}")
            
            # Enhanced fallback response với personalization + external API
            fallback_response = self._get_enhanced_fallback_response_with_external_api(
                locals().get('user_message', ''),
                locals().get('user_context'),
                locals().get('personalization_info', {}),
                locals().get('jwt_token')
            )
            
            return Response({
                'session_id': locals().get('session_id', str(uuid.uuid4())),
                'response': fallback_response,
                'confidence': 0.3,
                'method': 'enhanced_fallback_with_external_api',
                'response_time': time.time() - start_time,
                'status': 'fallback',
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
    
    def _get_enhanced_fallback_response_with_external_api(self, user_message='', user_context=None, personalization_info={}, jwt_token=None):
        """Enhanced fallback response với comprehensive personalization + external API"""
        if user_context:
            full_name = user_context.get('full_name', '')
            faculty_code = user_context.get('faculty_code', '')
            name_suffix = full_name.split()[-1] if full_name else faculty_code
            personal_address = f"thầy/cô {name_suffix}"
            department_name = user_context.get('department_name', 'BDU')
            
            # ✅ NEW: Different messages based on JWT token availability
            if jwt_token:
                # Have JWT token but still failed
                return f"""Dạ xin lỗi {personal_address}, hệ thống đang được nâng cấp để phục vụ {personal_address} tốt hơn.

Mặc dù em đã nhận được thông tin đăng nhập của {personal_address}, nhưng hiện tại có một số khó khăn kỹ thuật. 

{personal_address} có thể:
• Thử lại sau vài phút ⏰
• Truy cập trực tiếp hệ thống quản lý đào tạo của trường 🌐
• Liên hệ khoa {department_name} để được hỗ trợ trực tiếp 📞
• Gọi bộ phận IT: 0274.xxx.xxxx 📧

Em sẽ cố gắng khắc phục để phục vụ {personal_address} tốt hơn! 🎓✨"""
            else:
                # No JWT token
                response_style = personalization_info.get('response_style', 'professional')
                
                if response_style == 'friendly':
                    return f"""Dạ xin lỗi {personal_address}, hệ thống đang được cải thiện để phục vụ {personal_address} tốt hơn nhé! 😊

Để truy cập thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng BDU trước ạ. 🔐

Trong thời gian này, {personal_address} có thể:
• Liên hệ trực tiếp khoa {department_name} 📞
• Gọi tổng đài: 0274.xxx.xxxx  
• Email: info@bdu.edu.vn 📧
• Website: www.bdu.edu.vn 🌐

Em sẽ cố gắng hỗ trợ {personal_address} tốt hơn! 🎓✨"""

                elif response_style == 'brief':
                    return f"""Dạ {personal_address}, hệ thống đang cải thiện. 

Để xem thông tin cá nhân, vui lòng đăng nhập ứng dụng BDU. 🔐
Liên hệ: khoa {department_name} hoặc 0274.xxx.xxxx

Cảm ơn {personal_address}! 🎓"""

                else:  # professional (default)
                    return f"""Dạ xin lỗi {personal_address}, hệ thống đang được cải thiện để phục vụ {personal_address} tốt hơn.

Để truy cập thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng BDU. 🔐

Trong thời gian này, {personal_address} có thể:
• Liên hệ trực tiếp khoa {department_name}
• Gọi tổng đài: 0274.xxx.xxxx  
• Email: info@bdu.edu.vn
• Website: www.bdu.edu.vn

Cảm ơn {personal_address} đã kiên nhẫn! 🎓"""
        
        # Fallback for non-authenticated users
        if jwt_token:
            return """Xin chào! Tôi đã nhận được thông tin đăng nhập, nhưng hiện tại gặp khó khăn kỹ thuật.

Bạn có thể thử lại sau hoặc liên hệ:
• Hotline: 0274.xxx.xxxx
• Email: info@bdu.edu.vn
• Website: www.bdu.edu.vn

Cảm ơn bạn đã kiên nhẫn! 🎓"""
        else:
            return self._get_safe_fallback_response(user_message)
    
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

# ✅ NEW: JWT Token Test View (for development/testing)
class JWTTokenTestView(APIView):
    """Test endpoint for JWT token validation"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        """Test JWT token extraction and validation"""
        try:
            # Extract token
            jwt_token = extract_jwt_token(request)
            
            if not jwt_token:
                return Response({
                    'success': False,
                    'message': 'No JWT token found in request',
                    'token_sources_checked': [
                        'Authorization header (Bearer token)',
                        'Request data (token field)',
                        'JSON body (token field)', 
                        'Query parameters (token field)'
                    ]
                })
            
            # Validate format
            is_valid_format, format_message = validate_jwt_token_format(jwt_token)
            
            # Try to decode (without verification for testing)
            decoded_payload = None
            decode_error = None
            try:
                decoded_payload = jwt.decode(jwt_token, options={"verify_signature": False})
            except Exception as e:
                decode_error = str(e)
            
            # Test external API service
            external_api_test = None
            try:
                from ai_models.external_api_service import external_api_service
                lecturer_info = external_api_service.get_lecturer_info_from_token(jwt_token)
                external_api_test = {
                    'service_available': True,
                    'lecturer_info_extracted': bool(lecturer_info),
                    'lecturer_info': lecturer_info
                }
            except Exception as e:
                external_api_test = {
                    'service_available': False,
                    'error': str(e)
                }
            
            return Response({
                'success': True,
                'token_found': True,
                'token_preview': f"{jwt_token[:15]}...{jwt_token[-15:]}" if len(jwt_token) > 30 else jwt_token,
                'token_length': len(jwt_token),
                'format_validation': {
                    'is_valid': is_valid_format,
                    'message': format_message
                },
                'decode_test': {
                    'success': decoded_payload is not None,
                    'payload_preview': {
                        'sub': decoded_payload.get('sub') if decoded_payload else None,
                        'vien_chuc_ma': decoded_payload.get('vien_chuc', {}).get('ma_vien_chuc') if decoded_payload else None,
                        'vien_chuc_ten': decoded_payload.get('vien_chuc', {}).get('ho_va_ten') if decoded_payload else None
                    } if decoded_payload else None,
                    'error': decode_error
                },
                'external_api_test': external_api_test,
                'message': 'JWT token test completed successfully'
            })
            
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e),
                'message': 'JWT token test failed'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PersonalizedChatContextView(APIView):
    """Lấy context cá nhân hóa cho chat"""
    
    def get(self, request):
        """GET method - Enhanced personalized context"""
        try:
            if not request.user.is_authenticated:
                return Response({
                    'personalization_enabled': False,
                    'message': 'User not authenticated'
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user = request.user
            user_context = user.get_chatbot_context()
            
            # ✅ ENHANCED: More comprehensive context info
            current_style = user.chatbot_preferences.get('response_style', 'professional')
            user_memory = user.chatbot_preferences.get('user_memory_prompt', '').strip()
            
            context_info = {
                'personalization_enabled': True,
                'user_context': user_context,
                'personalized_greeting': f"Chào {user_context.get('position_name', 'giảng viên')} {user.full_name}!",
                'department_focus': user_context.get('department_name', 'BDU'),
                
                # ✅ NEW: Enhanced style information
                'style_info': {
                    'current_style': current_style,
                    'style_name': dict(user.RESPONSE_STYLE_CHOICES).get(current_style),
                    'style_description': _get_style_description_for_context(current_style),
                    'available_styles': [
                        {
                            'code': choice[0],
                            'name': choice[1],
                            'description': _get_style_description_for_context(choice[0])
                        }
                        for choice in user.RESPONSE_STYLE_CHOICES
                    ]
                },
                
                # ✅ NEW: Memory information
                'memory_info': {
                    'has_custom_memory': bool(user_memory),
                    'memory_length': len(user_memory),
                    'memory_preview': user_memory[:100] + '...' if len(user_memory) > 100 else user_memory,
                    'using_default_memory': not bool(user_memory)
                },
                
                # ✅ ENHANCED: Better suggested topics
                'suggested_topics': _get_enhanced_suggested_topics_for_department(user.department, current_style),
                'quick_actions': _get_style_aware_quick_actions_for_position(user.position, current_style),
                
                # ✅ NEW: Personalization tips WITH EXTERNAL API
                'personalization_tips': [
                    f"Sử dụng phong cách '{dict(user.RESPONSE_STYLE_CHOICES).get(current_style)}' cho câu trả lời phù hợp",
                    f"Hỏi về thông tin chuyên ngành {user_context.get('department_name')}",
                    f"Tùy chỉnh memory prompt để ChatBDU hiểu bạn hơn",
                    "Bật/tắt ưu tiên chuyên ngành theo nhu cầu",
                    "Đăng nhập ứng dụng để truy cập lịch giảng dạy cá nhân",  # ✅ NEW
                    "Hỏi về 'lịch của tôi' để xem thời khóa biểu riêng"  # ✅ NEW
                ],
                
                # ✅ NEW: External API capabilities
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
                
                # ✅ NEW: Settings summary
                'current_settings': {
                    'response_style': current_style,
                    'department_priority': user.chatbot_preferences.get('department_priority', True),
                    'has_custom_memory': bool(user_memory),
                    'total_preferences': len(user.chatbot_preferences),
                    'external_api_ready': True  # ✅ NEW
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
class PersonalizedSystemStatusView(APIView):
    """System status với thông tin personalization"""
    
    def get(self, request):
        """GET method - Enhanced system status với personalization + external API"""
        try:
            # Base system status
            status_data = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()
            
            # ✅ NEW: Get external API status
            try:
                from ai_models.external_api_service import external_api_service
                external_api_status = external_api_service.get_system_status()
            except ImportError:
                external_api_status = {'external_api_service': {'available': False, 'error': 'Service not available'}}
            
            # ✅ ENHANCED: Comprehensive personalization status với external API
            personalization_status = {
                'personalization_enabled': True,
                'version': '5.0.0',  # ✅ Version bump for external API
                'features': {
                    'response_style_support': True,
                    'user_memory_prompts': True,
                    'department_priority': True,
                    'style_aware_generation': True,
                    'personalized_addressing': True,
                    'dynamic_style_configs': True,
                    'jwt_token_authentication': True,  # ✅ NEW
                    'external_api_integration': True,  # ✅ NEW
                    'personal_schedule_access': True,  # ✅ NEW
                    'lecturer_info_queries': True  # ✅ NEW
                },
                'statistics': {
                    'total_faculty': 0,
                    'active_personalized_sessions': len(chatbot_ai.response_generator._user_context_cache),
                    'supported_styles': len(chatbot_ai.response_generator.style_generation_configs),
                    'departments_available': len(Faculty.DEPARTMENT_CHOICES) if 'Faculty' in globals() else 0,
                    'positions_available': len(Faculty.POSITION_CHOICES) if 'Faculty' in globals() else 0
                },
                'external_api_integration': external_api_status  # ✅ NEW
            }
            
            # Add current user info if authenticated
            if request.user.is_authenticated:
                personalization_status['current_user'] = {
                    'faculty_code': request.user.faculty_code,
                    'department': request.user.get_department_display(),
                    'position': request.user.get_position_display(),
                    'current_style': request.user.chatbot_preferences.get('response_style', 'professional'),
                    'has_custom_memory': bool(request.user.chatbot_preferences.get('user_memory_prompt', '').strip()),
                    'department_priority': request.user.chatbot_preferences.get('department_priority', True),
                    'preferences_configured': bool(request.user.chatbot_preferences),
                    'external_api_ready': True  # ✅ NEW - ready to use external API
                }
            
            # Get statistics from database
            try:
                from authentication.models import Faculty
                personalization_status['statistics']['total_faculty'] = Faculty.objects.count()
                personalization_status['statistics']['active_faculty'] = Faculty.objects.filter(is_active_faculty=True).count()
                personalization_status['statistics']['with_personalization'] = Faculty.objects.exclude(chatbot_preferences={}).count()
                
                # Style distribution
                style_distribution = {}
                for faculty in Faculty.objects.exclude(chatbot_preferences={}):
                    style = faculty.chatbot_preferences.get('response_style', 'professional')
                    style_distribution[style] = style_distribution.get(style, 0) + 1
                personalization_status['statistics']['style_distribution'] = style_distribution
                
            except Exception as e:
                personalization_status['statistics']['database_error'] = str(e)
            
            # Merge with system status
            status_data.update({
                'personalization': personalization_status,
                'speech_status': speech_status,
                'external_api_status': external_api_status  # ✅ NEW
            })
            
            return Response(status_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Enhanced system status error: {str(e)}")
            return Response({
                'error': 'Could not retrieve enhanced system status',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ✅ THÊM: Helper functions (copy từ authentication/views.py)
# def _get_suggested_topics_for_department(department):
#     """Lấy các chủ đề gợi ý theo ngành"""
#     topics_map = {
#         'cntt': ['Chương trình đào tạo CNTT', 'Phòng lab tin học', 'Thiết bị máy tính', 'Hợp tác doanh nghiệp IT'],
#         'duoc': ['Chương trình đào tạo Dược', 'Phòng thí nghiệm Dược', 'Thiết bị phân tích', 'Thực tập bệnh viện'],
#         'dien_tu': ['Chương trình Điện tử', 'Lab vi xử lý', 'Thiết bị đo lường', 'Dự án IoT'],
#         'co_khi': ['Chương trình Cơ khí', 'Phòng CAD/CAM', 'Máy gia công CNC', 'Thực tập nhà máy'],
#         'y_khoa': ['Chương trình Y khoa', 'Phòng giải phẫu', 'Thực hành lâm sàng', 'Bệnh viện liên kết'],
#         'kinh_te': ['Chương trình Kinh tế', 'Phần mềm phân tích', 'Thực tập ngân hàng', 'Nghiên cứu thị trường'],
#         'luat': ['Chương trình Luật', 'Phiên tòa giả định', 'Thực tập tòa án', 'Văn phòng luật sư']
#     }
#     return topics_map.get(department, ['Thông tin chung về trường', 'Quy định đào tạo', 'Cơ sở vật chất'])

# def _get_quick_actions_for_position(position):
#     """Lấy các quick actions theo chức vụ"""
#     actions_map = {
#         'giang_vien': ['Xem lịch giảng dạy', 'Quản lý điểm sinh viên', 'Tài liệu giảng dạy', 'Nghiên cứu khoa học'],
#         'truong_khoa': ['Quản lý khoa', 'Kế hoạch đào tạo', 'Báo cáo hoạt động', 'Nhân sự khoa'],
#         'truong_bo_mon': ['Quản lý bộ môn', 'Phân công giảng dạy', 'Tài liệu chuyên ngành', 'Hoạt động chuyên môn'],
#         'tro_giang': ['Hỗ trợ giảng dạy', 'Chuẩn bị bài giảng', 'Chấm bài tập', 'Tương tác sinh viên']
#     }
#     return actions_map.get(position, ['Thông tin chung', 'Hỗ trợ kỹ thuật', 'Liên hệ phòng ban'])

# ✅ HELPER FUNCTIONS

def _get_style_description_for_context(style_code):
    """Get style description for context API"""
    descriptions = {
        'professional': 'Trang trọng, lịch sự, chuẩn mực - phù hợp cho công việc chính thức',
        'friendly': 'Gần gũi, ấm áp, vui vẻ - tạo không khí thoải mái',
        'technical': 'Chi tiết, chuyên môn, kỹ thuật - phù hợp cho giải thích phức tạp',
        'brief': 'Ngắn gọn, súc tích, đi thẳng vào vấn đề - tiết kiệm thời gian',
        'detailed': 'Đầy đủ, toàn diện, nhiều ví dụ - hiểu sâu vấn đề'
    }
    return descriptions.get(style_code, 'Mô tả không có sẵn')

def _get_enhanced_suggested_topics_for_department(department, response_style):
    """Enhanced suggested topics based on department and style"""
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
    
    # Add style-specific suffix
    if response_style == 'technical':
        topics = [f"{topic} (chi tiết kỹ thuật)" for topic in topics]
    elif response_style == 'brief':
        topics = [f"{topic} (tóm tắt)" for topic in topics]
    
    return topics

def _get_style_aware_quick_actions_for_position(position, response_style):
    """Style-aware quick actions based on position"""
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
    
    # Add style context
    if response_style == 'detailed':
        actions.append('Hướng dẫn chi tiết các quy trình')
    elif response_style == 'friendly':
        actions.append('Chat thân thiện về công việc')
    
    return actions

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

# 1. THÊM VÀO chat/views.py - 2 views mới

import time
import uuid
from django.utils import timezone
from django.db import models
import json

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
                # Bot message
                messages.append({
                    'type': 'bot',
                    'content': chat.bot_response,
                    'timestamp': chat.timestamp.isoformat(),
                    'confidence': chat.confidence_score,
                    'response_time': chat.response_time,
                    'sources': [],
                    'reference_links': [],
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

class HealthCheckView(APIView):
    def get(self, request):
        try:
            system_status = chatbot_ai.get_system_status()
            speech_status = speech_service.get_system_status()
            
            return Response({
                'status': 'healthy',
                'message': 'Enhanced Personalized Chatbot is running! 🚀',
                'database': 'connected',
                'encoding': 'utf-8',
                'system_status': system_status,
                'speech_status': speech_status,
                'personalization': 'enabled',  # ✅ NEW
                'version': '4.0.0'  # ✅ Updated version
            })
        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)