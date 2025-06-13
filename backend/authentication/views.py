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
    API đăng nhập cho giảng viên
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
        
        return Response({
            'success': True,
            'message': 'Đăng nhập thành công',
            'data': {
                'user': user_data,
                'token': token.key,
                'session_id': request.session.session_key
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Login error for {faculty_code}: {e}")
        log_login_attempt(faculty_code, request, False, "System error")
        return Response({
            'success': False,
            'message': 'Lỗi hệ thống. Vui lòng thử lại sau.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
# 🎯 PERSONALIZATION ENDPOINTS
# ===============================

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def chatbot_preferences(request):
    """API lấy chatbot preferences của Faculty"""
    try:
        faculty = request.user
        preferences = faculty.chatbot_preferences or {}
        
        return Response({
            'success': True,
            'data': {
                'preferences': preferences,
                'user_context': faculty.get_chatbot_context(),
                'department_info': {
                    'code': faculty.department,
                    'name': faculty.get_department_display(),
                    'position': faculty.get_position_display()
                }
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
    """API cập nhật chatbot preferences"""
    try:
        faculty = request.user
        new_preferences = request.data.get('preferences', {})
        
        # Validate preferences
        valid_response_styles = ['professional', 'friendly', 'technical', 'brief', 'detailed']
        if 'response_style' in new_preferences:
            if new_preferences['response_style'] not in valid_response_styles:
                return Response({
                    'success': False,
                    'message': 'Response style không hợp lệ',
                    'valid_options': valid_response_styles
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate focus_areas theo department
        if 'focus_areas' in new_preferences:
            valid_focus_areas = _get_valid_focus_areas_for_department(faculty.department)
            invalid_areas = [area for area in new_preferences['focus_areas'] if area not in valid_focus_areas]
            if invalid_areas:
                return Response({
                    'success': False,
                    'message': f'Focus areas không hợp lệ: {invalid_areas}',
                    'valid_options': valid_focus_areas
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Cập nhật preferences
        faculty.update_chatbot_preferences(new_preferences)
        
        logger.info(f"Updated chatbot preferences for {faculty.faculty_code}")
        
        return Response({
            'success': True,
            'message': 'Cấu hình chatbot đã được cập nhật thành công',
            'data': {
                'preferences': faculty.chatbot_preferences,
                'user_context': faculty.get_chatbot_context()
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Update chatbot preferences error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi cập nhật cấu hình chatbot'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def personalized_system_prompt(request):
    """API lấy system prompt cá nhân hóa"""
    try:
        faculty = request.user
        
        return Response({
            'success': True,
            'data': {
                'system_prompt': faculty.get_personalized_system_prompt(),
                'user_context': faculty.get_chatbot_context(),
                'department_info': {
                    'code': faculty.department,
                    'name': faculty.get_department_display(),
                    'position': faculty.get_position_display(),
                    'specialization': faculty.specialization
                }
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Get personalized prompt error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy system prompt'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def update_department_focus(request):
    """API cập nhật focus areas theo ngành"""
    try:
        faculty = request.user
        focus_areas = request.data.get('focus_areas', [])
        
        # Validate focus areas theo department
        valid_focus_areas = _get_valid_focus_areas_for_department(faculty.department)
        filtered_focus_areas = [area for area in focus_areas if area in valid_focus_areas]
        
        # Cập nhật preferences
        faculty.update_chatbot_preferences({
            'focus_areas': filtered_focus_areas,
            'department_priority': True
        })
        
        logger.info(f"Updated focus areas for {faculty.faculty_code}: {filtered_focus_areas}")
        
        return Response({
            'success': True,
            'message': 'Focus areas đã được cập nhật thành công',
            'data': {
                'focus_areas': filtered_focus_areas,
                'valid_options': valid_focus_areas,
                'department': faculty.get_department_display()
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Update focus areas error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi cập nhật focus areas'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_department_suggestions(request):
    """API lấy gợi ý theo ngành"""
    try:
        faculty = request.user
        
        # Lấy suggested topics và quick actions
        suggested_topics = _get_suggested_topics_for_department(faculty.department)
        quick_actions = _get_quick_actions_for_position(faculty.position)
        valid_focus_areas = _get_valid_focus_areas_for_department(faculty.department)
        
        return Response({
            'success': True,
            'data': {
                'department': {
                    'code': faculty.department,
                    'name': faculty.get_department_display()
                },
                'position': {
                    'code': faculty.position,
                    'name': faculty.get_position_display()
                },
                'suggested_topics': suggested_topics,
                'quick_actions': quick_actions,
                'valid_focus_areas': valid_focus_areas,
                'personalized_greeting': f"Xin chào {faculty.get_position_display()} {faculty.full_name}!"
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Get department suggestions error: {e}")
        return Response({
            'success': False,
            'message': 'Lỗi khi lấy gợi ý'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ===============================
# 🛠️ HELPER FUNCTIONS
# ===============================

def _get_valid_focus_areas_for_department(department):
    """Lấy các focus areas hợp lệ cho từng ngành"""
    focus_areas_map = {
        'cntt': [
            'Lập trình Web', 'Mobile App', 'AI/Machine Learning', 
            'Database Management', 'Network Security', 'Cloud Computing',
            'Data Science', 'IoT Development', 'Software Engineering'
        ],
        'duoc': [
            'Dược lý học', 'Hóa dược', 'Dược động học', 'Vi sinh dược',
            'Phân tích dược', 'Công nghệ dược', 'Dược lâm sàng',
            'Quản lý dược', 'Dược thảo'
        ],
        'dien_tu': [
            'Mạch điện tử', 'Vi xử lý', 'IoT', 'Robotics', 'Automation',
            'Truyền thông', 'Điều khiển tự động', 'Embedded Systems',
            'Signal Processing', 'Power Electronics'
        ],
        'co_khi': [
            'Thiết kế máy', 'CAD/CAM', 'Gia công CNC', 'Nhiệt động lực',
            'Cơ học chất lưu', 'Vật liệu', 'Tự động hóa sản xuất',
            'Bảo trì thiết bị', 'Quản lý sản xuất'
        ],
        'y_khoa': [
            'Nội khoa', 'Ngoại khoa', 'Sản phụ khoa', 'Nhi khoa',
            'Mắt', 'Tai mũi họng', 'Da liễu', 'Tâm thần',
            'Chẩn đoán hình ảnh', 'Xét nghiệm y học'
        ],
        'kinh_te': [
            'Tài chình doanh nghiệp', 'Ngân hàng', 'Chứng khoán',
            'Bảo hiểm', 'Kinh tế vĩ mô', 'Kinh tế vi mô',
            'Kinh tế lượng', 'Thương mại quốc tế', 'Marketing'
        ],
        'luat': [
            'Luật dân sự', 'Luật hình sự', 'Luật kinh tế',
            'Luật lao động', 'Luật hành chính', 'Luật quốc tế',
            'Luật môi trường', 'Luật đất đai', 'Luật sở hữu trí tuệ'
        ]
    }
    
    return focus_areas_map.get(department, ['Tổng quát'])


def _get_suggested_topics_for_department(department):
    """Lấy các chủ đề gợi ý theo ngành"""
    topics_map = {
        'cntt': [
            'Chương trình đào tạo CNTT',
            'Phòng lab tin học',
            'Thiết bị máy tính và server',
            'Hợp tác doanh nghiệp IT',
            'Nghiên cứu AI/ML',
            'Đào tạo lập trình'
        ],
        'duoc': [
            'Chương trình đào tạo Dược',
            'Phòng thí nghiệm Dược',
            'Thiết bị phân tích dược',
            'Thực tập bệnh viện',
            'Chứng chỉ hành nghề Dược sĩ',
            'Nghiên cứu dược liệu'
        ],
        'dien_tu': [
            'Chương trình Điện tử viễn thông',
            'Lab vi xử lý và IoT',
            'Thiết bị đo lường điện tử',
            'Dự án IoT và Robotics',
            'Thực tập doanh nghiệp',
            'Nghiên cứu embedded systems'
        ],
        'co_khi': [
            'Chương trình Cơ khí',
            'Phòng CAD/CAM',
            'Máy gia công CNC',
            'Thực tập nhà máy',
            'Thiết kế sản phẩm',
            'Nghiên cứu automation'
        ],
        'y_khoa': [
            'Chương trình Y khoa',
            'Phòng giải phẫu',
            'Thực hành lâm sàng',
            'Bệnh viện liên kết',
            'Chứng chỉ hành nghề',
            'Nghiên cứu y sinh'
        ],
        'kinh_te': [
            'Chương trình Kinh tế',
            'Phần mềm phân tích tài chính',
            'Thực tập ngân hàng',
            'Nghiên cứu thị trường',
            'Chứng chỉ CFA/FRM',
            'Tư vấn tài chính'
        ],
        'luat': [
            'Chương trình Luật',
            'Phiên tòa giả định',
            'Thực tập tòa án',
            'Văn phòng luật sư',
            'Chứng chỉ luật sư',
            'Tư vấn pháp lý'
        ]
    }
    
    return topics_map.get(department, [
        'Thông tin chung về trường',
        'Quy định đào tạo',
        'Cơ sở vật chất',
        'Hoạt động nghiên cứu'
    ])


def _get_quick_actions_for_position(position):
    """Lấy các quick actions theo chức vụ"""
    actions_map = {
        'giang_vien': [
            'Xem lịch giảng dạy',
            'Quản lý điểm sinh viên',
            'Tài liệu giảng dạy',
            'Nghiên cứu khoa học',
            'Đề cương môn học'
        ],
        'truong_khoa': [
            'Quản lý khoa',
            'Kế hoạch đào tạo',
            'Báo cáo hoạt động',
            'Nhân sự khoa',
            'Ngân sách khoa'
        ],
        'pho_truong_khoa': [
            'Hỗ trợ quản lý khoa',
            'Giám sát đào tạo',
            'Phối hợp hoạt động',
            'Báo cáo tình hình'
        ],
        'truong_bo_mon': [
            'Quản lý bộ môn',
            'Phân công giảng dạy',
            'Tài liệu chuyên ngành',
            'Hoạt động chuyên môn',
            'Kế hoạch bộ môn'
        ],
        'tro_giang': [
            'Hỗ trợ giảng dạy',
            'Chuẩn bị bài giảng',
            'Chấm bài tập',
            'Tương tác sinh viên',
            'Học tập nâng cao'
        ],
        'can_bo': [
            'Công tác hành chính',
            'Xử lý thủ tục',
            'Hỗ trợ giảng viên',
            'Quản lý tài liệu'
        ]
    }
    
    return actions_map.get(position, [
        'Thông tin chung',
        'Hỗ trợ kỹ thuật',
        'Liên hệ phòng ban'
    ])