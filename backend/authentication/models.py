from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
import uuid

class Faculty(AbstractUser):
    """
    Custom User model cho giảng viên
    Extends AbstractUser để dễ customize sau này
    """
    
    # ✅ THÊM: Choices cho department và position
    DEPARTMENT_CHOICES = [
        ('cntt', 'Công nghệ thông tin'),
        ('duoc', 'Dược'),
        ('dien_tu', 'Điện tử viễn thông'),
        ('co_khi', 'Cơ khí'),
        ('kinh_te', 'Kinh tế'),
        ('luat', 'Luật'),
        ('y_khoa', 'Y khoa'),
        ('ngoai_ngu', 'Ngoại ngữ'),
        ('xay_dung', 'Xây dựng'),
        ('quan_tri', 'Quản trị kinh doanh'),
        ('ke_toan', 'Kế toán'),
        ('marketing', 'Marketing'),
        ('tai_chinh', 'Tài chính ngân hàng'),
        ('general', 'Chung (không chuyên ngành)')
    ]
    
    POSITION_CHOICES = [
        ('giang_vien', 'Giảng viên'),
        ('tro_giang', 'Trợ giảng'),
        ('truong_khoa', 'Trưởng khoa'),
        ('pho_truong_khoa', 'Phó trưởng khoa'),
        ('truong_bo_mon', 'Trưởng bộ môn'),
        ('can_bo', 'Cán bộ'),
        ('admin', 'Quản trị viên')
    ]
    
    # Thông tin cơ bản (giữ nguyên)
    faculty_code = models.CharField(
        max_length=20, 
        unique=True, 
        help_text="Mã giảng viên (VD: GV001, BDU2024001)"
    )
    full_name = models.CharField(max_length=100, help_text="Họ và tên đầy đủ")
    
    # ✅ NÂNG CẤP: Thay đổi department thành có choices
    department = models.CharField(
        max_length=20, 
        choices=DEPARTMENT_CHOICES, 
        default='general',
        help_text="Khoa/Ngành chuyên môn"
    )
    
    phone = models.CharField(max_length=15, blank=True)
    
    # ✅ THÊM: Thông tin vai trò và personalization
    position = models.CharField(
        max_length=20,
        choices=POSITION_CHOICES,
        default='giang_vien',
        verbose_name="Chức vụ"
    )
    
    # ✅ THÊM: Thông tin chuyên môn
    specialization = models.TextField(
        blank=True, 
        verbose_name="Chuyên môn/Lĩnh vực nghiên cứu"
    )
    office_room = models.CharField(
        max_length=20, 
        blank=True, 
        verbose_name="Phòng làm việc"
    )
    
    # ✅ THÊM: Tùy chọn chatbot cá nhân hóa
    chatbot_preferences = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Tùy chọn chatbot",
        help_text="Lưu các tùy chọn cá nhân hóa chatbot"
    )
    
    # Trạng thái tài khoản (giữ nguyên)
    is_active_faculty = models.BooleanField(default=True, help_text="Có đang làm việc không")
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    
    # Metadata (giữ nguyên)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Override username field (giữ nguyên)
    USERNAME_FIELD = 'faculty_code'
    REQUIRED_FIELDS = ['email', 'full_name']
    
    class Meta:
        db_table = 'faculty'
        verbose_name = 'Giảng viên'
        verbose_name_plural = 'Danh sách giảng viên'
        
    def __str__(self):
        return f"{self.faculty_code} - {self.full_name} ({self.get_department_display()})"
    
    def save(self, *args, **kwargs):
        # Auto set username = faculty_code (giữ nguyên)
        self.username = self.faculty_code
        super().save(*args, **kwargs)
    
    # ✅ THÊM: Các method hỗ trợ personalization
    def get_role_description(self):
        """Lấy mô tả vai trò đầy đủ"""
        dept_name = self.get_department_display()
        pos_name = self.get_position_display()
        return f"{pos_name} {dept_name}"
    
    def get_chatbot_context(self):
        """Lấy context cho chatbot dựa trên vai trò"""
        return {
            'user_id': self.id,
            'faculty_code': self.faculty_code,
            'full_name': self.full_name,
            'department': self.department,
            'department_name': self.get_department_display(),
            'position': self.position,
            'position_name': self.get_position_display(),
            'role_description': self.get_role_description(),
            'specialization': self.specialization,
            'office_room': self.office_room,
            'preferences': self.chatbot_preferences,
            'is_lecturer': self.position in ['giang_vien', 'tro_giang', 'truong_khoa', 'pho_truong_khoa', 'truong_bo_mon']
        }
    
    def update_chatbot_preferences(self, preferences_data):
        """Cập nhật tùy chọn chatbot"""
        if not self.chatbot_preferences:
            self.chatbot_preferences = {}
        
        self.chatbot_preferences.update(preferences_data)
        self.save(update_fields=['chatbot_preferences'])
    
    def get_personalized_system_prompt(self):
        """Tạo system prompt cá nhân hóa dựa trên vai trò"""
        base_prompt = f"""Bạn là AI assistant chuyên nghiệp của Đại học Bình Dương (BDU), được thiết kế ĐẶC BIỆT để hỗ trợ {self.get_role_description()}.

🎯 THÔNG TIN NGƯỜI DÙNG:
- Mã GV: {self.faculty_code}
- Họ tên: {self.full_name}
- Vai trò: {self.get_role_description()}
- Khoa/Ngành: {self.get_department_display()}
- Chuyên môn: {self.specialization or 'Chung'}
- Phòng làm việc: {self.office_room or 'Không xác định'}

🤖 QUY TẮC VAI TRÒ:
- LUÔN xưng hô: "thầy/cô {self.full_name.split()[-1] if self.full_name else self.faculty_code}"
- Ưu tiên thông tin liên quan đến ngành {self.get_department_display()}
- Tập trung vào nhu cầu của {self.get_position_display()}
- Cung cấp thông tin chuyên sâu về {self.get_department_display()}

🎓 CHUYÊN MÔN THEO NGÀNH:"""
        
        # Thêm kiến thức chuyên ngành
        department_knowledge = self._get_department_specific_knowledge()
        if department_knowledge:
            base_prompt += f"\n{department_knowledge}"
        
        base_prompt += f"""

✅ QUY TẮC TRẢ LỜI:
1. Luôn bắt đầu với "Dạ thầy/cô {self.full_name.split()[-1] if self.full_name else self.faculty_code},"
2. Ưu tiên thông tin phục vụ công việc {self.get_position_display()}
3. Tập trung vào ngành {self.get_department_display()}
4. Sử dụng thuật ngữ chuyên ngành phù hợp
5. Kết thúc: "Thầy/cô cần hỗ trợ thêm gì về {self.get_department_display()} không ạ?"

🚫 TRÁNH:
- Thông tin không liên quan đến {self.get_department_display()}
- Tư vấn ngoài chuyên môn
- Quyết định thay cho lãnh đạo trường"""
        
        return base_prompt
    
    def _get_department_specific_knowledge(self):
        """Lấy kiến thức chuyên ngành"""
        knowledge_map = {
            'cntt': """
- Ngành CNTT: Lập trình, Cơ sở dữ liệu, Mạng máy tính, AI/ML
- Phòng lab: Lab tin học, Lab mạng, Lab phần mềm
- Thiết bị: Máy tính, Server, Thiết bị mạng
- Nghiên cứu: AI, IoT, Big Data, Cyber Security
- Hợp tác doanh nghiệp: FPT, Viettel, VNPT""",
            
            'duoc': """
- Ngành Dược: Dược lý, Hóa dược, Dược động học
- Phòng lab: Lab hóa phân tích, Lab vi sinh, Lab dược lý
- Thiết bị: Máy quang phổ, Máy sắc ký, Kính hiển vi
- Thực hành: Bệnh viện, Nhà thuốc, Công ty dược
- Chứng chỉ: Chứng chỉ hành nghề Dược sĩ""",
            
            'dien_tu': """
- Ngành Điện tử: Mạch điện tử, Vi xử lý, Truyền thông
- Phòng lab: Lab điện tử, Lab vi xử lý, Lab truyền thông
- Thiết bị: Oscilloscope, Function generator, Multimeter
- Ứng dụng: IoT, Embedded system, Robotics
- Ngành liên quan: Tự động hóa, Điều khiển""",
            
            'co_khi': """
- Ngành Cơ khí: Thiết kế máy, Gia công, Nhiệt động lực
- Phòng lab: Lab CAD/CAM, Lab gia công, Lab đo lường
- Thiết bị: Máy tiện, Máy phay, Máy đo CMM
- Phần mềm: AutoCAD, SolidWorks, Mastercam
- Thực tập: Nhà máy, Xí nghiệp cơ khí""",
            
            'y_khoa': """
- Ngành Y khoa: Giải phẫu, Sinh lý, Bệnh lý, Lâm sàng
- Phòng lab: Lab giải phẫu, Lab sinh lý, Lab vi sinh
- Thực hành: Bệnh viện, Trung tâm y tế
- Chứng chỉ: Bằng Bác sĩ, Chứng chỉ hành nghề
- Chuyên khoa: Nội, Ngoại, Sản, Nhi, Mắt, Răng Hàm Mặt""",
            
            'kinh_te': """
- Ngành Kinh tế: Vi mô, Vĩ mô, Kinh tế lượng, Tài chính
- Phần mềm: Excel, SPSS, Stata, EViews
- Thực tập: Ngân hàng, Công ty tài chính, Doanh nghiệp
- Chứng chỉ: CFA, FRM, Kế toán trưởng
- Nghiên cứu: Thị trường tài chính, Chính sách kinh tế""",
            
            'luat': """
- Ngành Luật: Luật dân sự, Luật hình sự, Luật kinh tế
- Thực hành: Tòa án, Văn phòng luật sư, Công ty
- Chứng chỉ: Chứng chỉ hành nghề Luật sư
- Kỹ năng: Biện luận, Soạn thảo hợp đồng, Tư vấn pháp lý
- Moot court: Phiên tòa giả định"""
        }
        
        return knowledge_map.get(self.department, "")


# ✅ GIỮ NGUYÊN: Các model khác không thay đổi
class PasswordResetToken(models.Model):
    """
    Token để reset password
    """
    faculty = models.ForeignKey(Faculty, on_delete=models.CASCADE)
    token = models.UUIDField(default=uuid.uuid4, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    used_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField()
    
    class Meta:
        db_table = 'password_reset_tokens'
        
    def is_valid(self):
        """Kiểm tra token có còn hợp lệ không"""
        return (
            self.used_at is None and 
            timezone.now() < self.expires_at
        )
    
    def mark_as_used(self):
        """Đánh dấu token đã được sử dụng"""
        self.used_at = timezone.now()
        self.save()


class LoginAttempt(models.Model):
    """
    Theo dõi các lần đăng nhập để bảo mật
    """
    faculty_code = models.CharField(max_length=20)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    success = models.BooleanField()
    attempt_time = models.DateTimeField(auto_now_add=True)
    failure_reason = models.CharField(max_length=100, blank=True)
    
    class Meta:
        db_table = 'login_attempts'
        indexes = [
            models.Index(fields=['faculty_code', 'attempt_time']),
            models.Index(fields=['ip_address', 'attempt_time']),
        ]