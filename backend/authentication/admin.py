from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import Faculty, PasswordResetToken, LoginAttempt


@admin.register(Faculty)
class FacultyAdmin(UserAdmin):
    """
    Admin interface cho Faculty model với personalization
    """
    # ✅ NÂNG CẤP: Thêm các trường personalization
    list_display = [
        'faculty_code', 'full_name', 'email', 'department_display', 'position_display',
        'is_active_faculty', 'last_login', 'login_count', 'has_chatbot_preferences'
    ]
    
    # ✅ NÂNG CẤP: Thêm filter theo department và position
    list_filter = [
        'is_active', 'is_active_faculty', 'department', 'position',
        'date_joined', 'last_login'
    ]
    
    search_fields = ['faculty_code', 'full_name', 'email', 'department', 'specialization']
    ordering = ['-date_joined']
    
    # ✅ NÂNG CẤP: Custom fieldsets với personalization
    fieldsets = (
        ('Thông tin đăng nhập', {
            'fields': ('faculty_code', 'password')
        }),
        ('Thông tin cá nhân', {
            'fields': ('full_name', 'email', 'phone', 'office_room')
        }),
        ('Thông tin vai trò & chuyên môn', {
            'fields': ('department', 'position', 'specialization'),
            'classes': ('wide',)
        }),
        ('Tùy chọn Chatbot', {
            'fields': ('chatbot_preferences',),
            'classes': ('collapse',),
            'description': 'Cấu hình cá nhân hóa chatbot cho giảng viên'
        }),
        ('Trạng thái', {
            'fields': ('is_active', 'is_active_faculty', 'is_staff', 'is_superuser')
        }),
        ('Metadata', {
            'fields': ('last_login', 'last_login_ip', 'date_joined'),
            'classes': ('collapse',)
        }),
    )
    
    # ✅ NÂNG CẤP: Add fieldsets với personalization
    add_fieldsets = (
        ('Tạo tài khoản mới', {
            'classes': ('wide',),
            'fields': ('faculty_code', 'full_name', 'email', 'password1', 'password2')
        }),
        ('Thông tin vai trò', {
            'classes': ('wide',),
            'fields': ('department', 'position', 'specialization', 'office_room')
        }),
    )
    
    readonly_fields = ['last_login', 'last_login_ip', 'date_joined']
    
    # ✅ THÊM: Custom display methods
    def department_display(self, obj):
        """Hiển thị department với màu sắc"""
        colors = {
            'cntt': '#007bff',
            'duoc': '#28a745', 
            'dien_tu': '#ffc107',
            'co_khi': '#dc3545',
            'y_khoa': '#e83e8c',
            'kinh_te': '#6f42c1',
            'luat': '#fd7e14',
            'general': '#6c757d'
        }
        color = colors.get(obj.department, '#6c757d')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_department_display()
        )
    department_display.short_description = 'Khoa/Ngành'
    
    def position_display(self, obj):
        """Hiển thị position với icon"""
        icons = {
            'truong_khoa': '👨‍💼',
            'pho_truong_khoa': '👩‍💼',
            'truong_bo_mon': '🎯',
            'giang_vien': '👨‍🏫',
            'tro_giang': '👩‍🎓',
            'can_bo': '👤',
            'admin': '🔧'
        }
        icon = icons.get(obj.position, '👤')
        return format_html(
            '{} {}',
            icon,
            obj.get_position_display()
        )
    position_display.short_description = 'Chức vụ'
    
    def has_chatbot_preferences(self, obj):
        """Hiển thị trạng thái cấu hình chatbot"""
        if obj.chatbot_preferences:
            return format_html(
                '<span style="color: green;">✓ Đã cấu hình</span>'
            )
        return format_html(
            '<span style="color: orange;">○ Chưa cấu hình</span>'
        )
    has_chatbot_preferences.short_description = 'Chatbot Setup'
    
    def login_count(self, obj):
        """Hiển thị số lần đăng nhập thành công"""
        count = LoginAttempt.objects.filter(
            faculty_code=obj.faculty_code, 
            success=True
        ).count()
        return format_html(
            '<span style="color: green; font-weight: bold;">{}</span>',
            count
        )
    login_count.short_description = 'Lần đăng nhập'
    
    def save_model(self, request, obj, form, change):
        """Override save để đảm bảo username = faculty_code"""
        obj.username = obj.faculty_code
        
        # ✅ THÊM: Khởi tạo chatbot preferences mặc định
        if not obj.chatbot_preferences:
            obj.chatbot_preferences = {
                'response_style': 'professional',
                'department_priority': True,
                'focus_areas': [],
                'notification_preferences': {
                    'email_updates': True,
                    'system_notifications': True
                }
            }
        
        super().save_model(request, obj, form, change)
    
    # ✅ THÊM: Actions
    actions = ['setup_default_chatbot_preferences', 'reset_chatbot_preferences']
    
    def setup_default_chatbot_preferences(self, request, queryset):
        """Thiết lập cấu hình chatbot mặc định"""
        count = 0
        for faculty in queryset:
            if not faculty.chatbot_preferences:
                faculty.update_chatbot_preferences({
                    'response_style': 'professional',
                    'department_priority': True,
                    'focus_areas': [],
                    'notification_preferences': {
                        'email_updates': True,
                        'system_notifications': True
                    }
                })
                count += 1
        
        self.message_user(request, f'Đã thiết lập cấu hình chatbot cho {count} giảng viên.')
    setup_default_chatbot_preferences.short_description = 'Thiết lập cấu hình chatbot mặc định'
    
    def reset_chatbot_preferences(self, request, queryset):
        """Reset cấu hình chatbot"""
        count = queryset.update(chatbot_preferences={})
        self.message_user(request, f'Đã reset cấu hình chatbot cho {count} giảng viên.')
    reset_chatbot_preferences.short_description = 'Reset cấu hình chatbot'


# ✅ GIỮ NGUYÊN: Các admin khác không thay đổi
@admin.register(PasswordResetToken)
class PasswordResetTokenAdmin(admin.ModelAdmin):
    """
    Admin interface cho PasswordResetToken
    """
    list_display = [
        'faculty', 'token_short', 'created_at', 'expires_at', 
        'is_used', 'is_expired'
    ]
    list_filter = ['created_at', 'expires_at', 'used_at']
    search_fields = ['faculty__faculty_code', 'faculty__full_name']
    readonly_fields = ['token', 'created_at']
    ordering = ['-created_at']
    
    def token_short(self, obj):
        """Hiển thị token ngắn gọn"""
        return f"{str(obj.token)[:8]}..."
    token_short.short_description = 'Token'
    
    def is_used(self, obj):
        """Hiển thị trạng thái đã sử dụng"""
        if obj.used_at:
            return format_html(
                '<span style="color: red;">✓ Đã dùng</span>'
            )
        return format_html(
            '<span style="color: green;">○ Chưa dùng</span>'
        )
    is_used.short_description = 'Trạng thái'
    
    def is_expired(self, obj):
        """Hiển thị trạng thái hết hạn"""
        if not obj.is_valid():
            return format_html(
                '<span style="color: red;">✗ Hết hạn</span>'
            )
        return format_html(
            '<span style="color: green;">✓ Còn hạn</span>'
        )
    is_expired.short_description = 'Hạn sử dụng'


@admin.register(LoginAttempt)
class LoginAttemptAdmin(admin.ModelAdmin):
    """
    Admin interface cho LoginAttempt
    """
    list_display = [
        'faculty_code', 'success_status', 'ip_address', 
        'attempt_time', 'failure_reason'
    ]
    list_filter = [
        'success', 'attempt_time', 'failure_reason'
    ]
    search_fields = ['faculty_code', 'ip_address']
    readonly_fields = ['faculty_code', 'ip_address', 'user_agent', 'attempt_time']
    ordering = ['-attempt_time']
    
    def success_status(self, obj):
        """Hiển thị trạng thái đăng nhập"""
        if obj.success:
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Thành công</span>'
            )
        return format_html(
            '<span style="color: red; font-weight: bold;">✗ Thất bại</span>'
        )
    success_status.short_description = 'Kết quả'