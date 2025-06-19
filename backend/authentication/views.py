from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.hashers import check_password
from django.utils import timezone
from django.conf import settings
from rest_framework import status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from datetime import timedelta
import logging

from .models import Faculty, PasswordResetToken, LoginAttempt
from .serializers import (
    LoginSerializer, FacultyProfileSerializer, 
    PasswordResetRequestSerializer, PasswordResetConfirmSerializer,
    ChangePasswordSerializer
)

logger = logging.getLogger(__name__)


def get_client_ip(request):
    """Lấy IP client"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_login_attempt(faculty_code, request, success, failure_reason=None):
    """Log lại các lần đăng nhập"""
    try:
        LoginAttempt.objects.create(
            faculty_code=faculty_code,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            success=success,
            failure_reason=failure_reason or ''
        )
    except Exception as e:
        logger.error(f"Error logging login attempt: {e}")


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def login_view(request):
    """
    API đăng nhập cho giảng viên với auto-load vai trò
    """
    serializer = LoginSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'success': False,
            'message': 'Dữ liệu không hợp lệ',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    faculty_code = serializer.validated_data['faculty_code']
    password = serializer.validated_data['password']
    remember_me = serializer.validated_data.get('remember_me', False)
    
    try:
        # Kiểm tra tài khoản có tồn tại không
        try:
            faculty = Faculty.objects.get(faculty_code=faculty_code)
        except Faculty.DoesNotExist:
            log_login_attempt(faculty_code, request, False, "Faculty not found")
            return Response({
                'success': False,
                'message': 'Mã giảng viên không tồn tại'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # Kiểm tra tài khoản có bị khóa không
        if not faculty.is_active or not faculty.is_active_faculty:
            log_login_attempt(faculty_code, request, False, "Account inactive")
            return Response({
                'success': False,
                'message': 'Tài khoản đã bị khóa hoặc không còn hoạt động'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # Xác thực password
        if not check_password(password, faculty.password):
            log_login_attempt(faculty_code, request, False, "Wrong password")
            return Response({
                'success': False,
                'message': 'Mật khẩu không chính xác'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # ✅ NEW: Auto-setup chatbot preferences khi đăng nhập
        preferences_setup_info = setup_chatbot_preferences_on_login(faculty)
        
        # Đăng nhập thành công
        login(request, faculty)
        
        # Tạo hoặc lấy token
        token, created = Token.objects.get_or_create(user=faculty)
        
        # Cập nhật thông tin đăng nhập
        faculty.last_login = timezone.now()
        faculty.last_login_ip = get_client_ip(request)
        faculty.save()
        
        # Set session timeout
        if remember_me:
            request.session.set_expiry(settings.SESSION_COOKIE_AGE)  # 2 weeks
        else:
            request.session.set_expiry(86400)  # 1 day
        
        # Log thành công
        log_login_attempt(faculty_code, request, True)
        
        # Serialize user data
        user_data = FacultyProfileSerializer(faculty).data
        
        # ✅ NEW: Thêm thông tin về chatbot setup
        return Response({
            'success': True,
            'message': 'Đăng nhập thành công',
            'data': {
                'user': user_data,
                'token': token.key,
                'session_id': request.session.session_key,
                'chatbot_setup': preferences_setup_info  # ✅ NEW: Info về việc setup
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Login error for {faculty_code}: {e}")
        log_login_attempt(faculty_code, request, False, "System error")
        return Response({
            'success': False,
            'message': 'Lỗi hệ thống. Vui lòng thử lại sau.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ✅ NEW: Helper function để setup chatbot preferences
def setup_chatbot_preferences_on_login(faculty):
    """
    Auto-setup chatbot preferences khi đăng nhập
    Returns info về những gì đã được setup
    """
    setup_info = {
        'was_setup': False,
        'is_first_time': False,
        'role_loaded': faculty.get_role_description(),
        'department': faculty.get_department_display(),
        'preferences_count': 0
    }
    
    try:
        # Kiểm tra xem đã có preferences chưa
        if not faculty.chatbot_preferences:
            # Lần đầu tiên đăng nhập - setup từ đầu
            faculty.chatbot_preferences = faculty.get_default_chatbot_preferences()
            faculty.save(update_fields=['chatbot_preferences'])
            
            setup_info.update({
                'was_setup': True,
                'is_first_time': True,
                'message': f'Đã tự động thiết lập vai trò {faculty.get_role_description()}',
                'preferences_count': len(faculty.chatbot_preferences)
            })
            
            logger.info(f"✅ First-time chatbot setup for {faculty.faculty_code}: {faculty.get_role_description()}")
            
        else:
            # Đã có preferences - kiểm tra xem có cần update không
            current_prefs = faculty.chatbot_preferences
            needs_update = False
            
            # Kiểm tra user_memory_prompt có empty không
            if not current_prefs.get('user_memory_prompt', '').strip():
                current_prefs['user_memory_prompt'] = faculty.get_default_memory_prompt()
                needs_update = True
            
            # Kiểm tra response_style
            if 'response_style' not in current_prefs:
                current_prefs['response_style'] = 'professional'
                needs_update = True
            
            # Kiểm tra department_priority
            if 'department_priority' not in current_prefs:
                current_prefs['department_priority'] = True
                needs_update = True
            
            if needs_update:
                current_prefs['last_login_update'] = timezone.now().isoformat()
                faculty.save(update_fields=['chatbot_preferences'])
                setup_info.update({
                    'was_setup': True,
                    'message': f'Đã cập nhật cài đặt cho vai trò {faculty.get_role_description()}',
                    'preferences_count': len(current_prefs)
                })
                logger.info(f"✅ Updated chatbot preferences for {faculty.faculty_code}")
            else:
                setup_info.update({
                    'message': f'Vai trò {faculty.get_role_description()} đã được thiết lập trước đó',
                    'preferences_count': len(current_prefs)
                })
        
        return setup_info
        
    except Exception as e:
        logger.error(f"Error setting up chatbot preferences for {faculty.faculty_code}: {e}")
        return {
            'was_setup': False,
            'error': str(e),
            'message': 'Có lỗi khi thiết lập chatbot preferences'
        }


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def logout_view(request):
    """
    API đăng xuất
    """
    try:
        # Xóa token
        try:
            token = Token.objects.get(user=request.user)
            token.delete()
        except Token.DoesNotExist:
            pass
        
        # Đăng xuất session
        logout(request)
        
        return Response({
            'success': True,
            'message': 'Đăng xuất thành công'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi đăng xuất'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def profile_view(request):
    """
    API lấy thông tin profile
    """
    try:
        serializer = FacultyProfileSerializer(request.user)
        return Response({
            'success': True,
            'data': serializer.data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy thông tin profile'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def password_reset_request(request):
    """
    API yêu cầu reset password
    """
    serializer = PasswordResetRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'success': False,
            'message': 'Dữ liệu không hợp lệ',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    faculty_code = serializer.validated_data['faculty_code']
    email = serializer.validated_data['email']
    
    try:
        # Kiểm tra tài khoản
        try:
            faculty = Faculty.objects.get(faculty_code=faculty_code, email=email)
        except Faculty.DoesNotExist:
            # Không tiết lộ thông tin tài khoản có tồn tại hay không
            return Response({
                'success': True,
                'message': 'Nếu thông tin chính xác, email reset password sẽ được gửi trong vài phút'
            }, status=status.HTTP_200_OK)
        
        # Tạo token reset
        expires_at = timezone.now() + timedelta(hours=1)  # Token hết hạn sau 1 giờ
        reset_token = PasswordResetToken.objects.create(
            faculty=faculty,
            expires_at=expires_at
        )
        
        # TODO: Gửi email với token (implement sau)
        # send_password_reset_email(faculty, reset_token.token)
        
        logger.info(f"Password reset requested for {faculty_code}")
        
        return Response({
            'success': True,
            'message': 'Email reset password đã được gửi',
            'debug_info': {
                'token': str(reset_token.token),  # Chỉ để debug, xóa khi production
                'expires_at': reset_token.expires_at.isoformat()
            } if settings.DEBUG else None
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi hệ thống. Vui lòng thử lại sau.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def password_reset_confirm(request):
    """
    API xác nhận reset password
    """
    serializer = PasswordResetConfirmSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'success': False,
            'message': 'Dữ liệu không hợp lệ',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    token = serializer.validated_data['token']
    new_password = serializer.validated_data['new_password']
    
    try:
        # Kiểm tra token
        try:
            reset_token = PasswordResetToken.objects.get(token=token)
        except PasswordResetToken.DoesNotExist:
            return Response({
                'success': False,
                'message': 'Token không hợp lệ'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not reset_token.is_valid():
            return Response({
                'success': False,
                'message': 'Token đã hết hạn hoặc đã được sử dụng'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Reset password
        faculty = reset_token.faculty
        faculty.set_password(new_password)
        faculty.save()
        
        # Đánh dấu token đã sử dụng
        reset_token.mark_as_used()
        
        logger.info(f"Password reset completed for {faculty.faculty_code}")
        
        return Response({
            'success': True,
            'message': 'Mật khẩu đã được thay đổi thành công'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Password reset confirm error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi hệ thống. Vui lòng thử lại sau.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def change_password(request):
    """
    API đổi mật khẩu khi đã đăng nhập
    """
    serializer = ChangePasswordSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            'success': False,
            'message': 'Dữ liệu không hợp lệ',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    current_password = serializer.validated_data['current_password']
    new_password = serializer.validated_data['new_password']
    
    try:
        # Kiểm tra mật khẩu hiện tại
        if not check_password(current_password, request.user.password):
            return Response({
                'success': False,
                'message': 'Mật khẩu hiện tại không chính xác'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Đổi mật khẩu
        request.user.set_password(new_password)
        request.user.save()
        
        logger.info(f"Password changed for {request.user.faculty_code}")
        
        return Response({
            'success': True,
            'message': 'Mật khẩu đã được thay đổi thành công'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Change password error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi hệ thống. Vui lòng thử lại sau.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def auth_status(request):
    """
    API kiểm tra trạng thái authentication
    """
    if request.user.is_authenticated:
        user_data = FacultyProfileSerializer(request.user).data
        return Response({
            'authenticated': True,
            'user': user_data
        })
    else:
        return Response({
            'authenticated': False,
            'user': None
        })
        
# ===============================
# 🎯 PERSONALIZATION ENDPOINTS - UPDATED
# ===============================

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def chatbot_preferences(request):
    """API lấy chatbot preferences của Faculty - UPDATED"""
    try:
        faculty = request.user
        
        # ✅ NEW: Ensure preferences exist
        if not faculty.chatbot_preferences:
            faculty.chatbot_preferences = faculty.get_default_chatbot_preferences()
            faculty.save(update_fields=['chatbot_preferences'])
        
        preferences = faculty.chatbot_preferences
        
        return Response({
            'success': True,
            'data': {
                'preferences': preferences,
                'user_context': faculty.get_chatbot_context(),
                'department_info': {
                    'code': faculty.department,
                    'name': faculty.get_department_display(),
                    'position': faculty.get_position_display()
                },
                'system_prompt': faculty.get_personalized_system_prompt()  # ✅ NEW: Include system prompt
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Get chatbot preferences error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy cấu hình chatbot'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def update_chatbot_preferences(request):
    """API cập nhật chatbot preferences - UPDATED để hỗ trợ structure mới"""
    try:
        faculty = request.user
        new_preferences = request.data.get('preferences', {})
        
        # ✅ NEW: Validate new structure
        # user_memory_prompt validation
        user_memory_prompt = new_preferences.get('user_memory_prompt', '').strip()
        if len(user_memory_prompt) > 1000:
            return Response({
                'success': False,
                'message': 'Memory prompt không được vượt quá 1000 ký tự'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # response_style validation
        valid_response_styles = ['professional', 'friendly', 'technical', 'brief', 'detailed']
        response_style = new_preferences.get('response_style', 'professional')
        if response_style not in valid_response_styles:
            return Response({
                'success': False,
                'message': 'Response style không hợp lệ',
                'valid_options': valid_response_styles
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # department_priority validation (boolean)
        department_priority = new_preferences.get('department_priority', True)
        if not isinstance(department_priority, bool):
            return Response({
                'success': False,
                'message': 'Department priority phải là true hoặc false'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # ✅ NEW: Cập nhật preferences với structure mới
        updated_preferences = {
            'user_memory_prompt': user_memory_prompt,
            'response_style': response_style,
            'department_priority': department_priority,
            'last_updated': timezone.now().isoformat(),
            'updated_by': 'user'  # Flag để biết user tự update
        }
        
        faculty.update_chatbot_preferences(updated_preferences)
        
        logger.info(f"Updated chatbot preferences for {faculty.faculty_code}: memory_prompt={len(user_memory_prompt)} chars, style={response_style}, dept_priority={department_priority}")
        
        return Response({
            'success': True,
            'message': 'Cài đặt chatbot đã được cập nhật thành công',
            'data': {
                'preferences': faculty.chatbot_preferences,
                'user_context': faculty.get_chatbot_context(),
                'system_prompt': faculty.get_personalized_system_prompt(),  # ✅ NEW: Return updated prompt
                'changes': {
                    'memory_prompt_length': len(user_memory_prompt),
                    'response_style': response_style,
                    'department_priority': department_priority
                }
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Update chatbot preferences error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi cập nhật cài đặt chatbot'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def personalized_system_prompt(request):
    """API lấy system prompt cá nhân hóa - UPDATED"""
    try:
        faculty = request.user
        
        # ✅ NEW: Ensure preferences exist
        if not faculty.chatbot_preferences:
            faculty.chatbot_preferences = faculty.get_default_chatbot_preferences()
            faculty.save(update_fields=['chatbot_preferences'])
        
        return Response({
            'success': True,
            'data': {
                'system_prompt': faculty.get_personalized_system_prompt(),
                'user_context': faculty.get_chatbot_context(),
                'preferences': faculty.chatbot_preferences,
                'department_info': {
                    'code': faculty.department,
                    'name': faculty.get_department_display(),
                    'position': faculty.get_position_display(),
                    'specialization': faculty.specialization,
                    'department_priority_enabled': faculty.chatbot_preferences.get('department_priority', True)
                }
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Get personalized prompt error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy system prompt'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ✅ NEW: API để test department priority
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def test_department_priority(request):
    """API để test xem department priority có hoạt động không"""
    try:
        faculty = request.user
        test_query = request.data.get('test_query', '')
        
        if not test_query:
            return Response({
                'success': False,
                'message': 'Cần có test_query để kiểm tra'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Tạo 2 system prompts: có và không có department priority
        
        # Test với department_priority = True
        faculty.chatbot_preferences['department_priority'] = True
        prompt_with_dept = faculty.get_personalized_system_prompt()
        
        # Test với department_priority = False  
        faculty.chatbot_preferences['department_priority'] = False
        prompt_without_dept = faculty.get_personalized_system_prompt()
        
        # Restore original setting
        original_dept_priority = request.data.get('original_dept_priority', True)
        faculty.chatbot_preferences['department_priority'] = original_dept_priority
        faculty.save(update_fields=['chatbot_preferences'])
        
        return Response({
            'success': True,
            'data': {
                'test_query': test_query,
                'department': faculty.get_department_display(),
                'prompts': {
                    'with_department_priority': prompt_with_dept,
                    'without_department_priority': prompt_without_dept
                },
                'differences': {
                    'has_department_knowledge': 'CHUYÊN MÔN NGÀNH' in prompt_with_dept,
                    'length_difference': len(prompt_with_dept) - len(prompt_without_dept)
                },
                'recommendation': 'Bật department priority để được hỗ trợ chuyên sâu về ngành' if faculty.department != 'general' else 'Department priority không cần thiết cho ngành chung'
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Test department priority error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi test department priority'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ✅ NEW: API để reset về default role
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated]) 
def reset_to_auto_role(request):
    """API để reset về vai trò tự động theo ngành"""
    try:
        faculty = request.user
        old_preferences = faculty.chatbot_preferences.copy() if faculty.chatbot_preferences else {}
        
        # Reset về default
        new_preferences = faculty.reset_to_auto_role()
        
        logger.info(f"Reset chatbot preferences to auto role for {faculty.faculty_code}: {faculty.get_role_description()}")
        
        return Response({
            'success': True,
            'message': f'Đã reset về vai trò tự động: {faculty.get_role_description()}',
            'data': {
                'old_preferences': old_preferences,
                'new_preferences': new_preferences,
                'role_description': faculty.get_role_description(),
                'department': faculty.get_department_display(),
                'system_prompt': faculty.get_personalized_system_prompt()
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Reset to auto role error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi reset về vai trò tự động'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===============================
# 🛠️ HELPER FUNCTIONS - SIMPLIFIED
# ===============================

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_department_suggestions(request):
    """API lấy gợi ý theo ngành - SIMPLIFIED"""
    try:
        faculty = request.user
        
        # Simplified suggestions based on department
        department_info = {
            'code': faculty.department,
            'name': faculty.get_department_display(),
            'has_specific_knowledge': faculty.department != 'general'
        }
        
        position_info = {
            'code': faculty.position,
            'name': faculty.get_position_display()
        }
        
        return Response({
            'success': True,
            'data': {
                'department': department_info,
                'position': position_info,
                'personalized_greeting': f"Xin chào {faculty.get_position_display()} {faculty.full_name}!",
                'role_description': faculty.get_role_description(),
                'auto_setup_available': True,
                'department_priority_recommended': faculty.department != 'general'
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Get department suggestions error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy gợi ý'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)