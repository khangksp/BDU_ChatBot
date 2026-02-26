from django.contrib.auth.models import AbstractUser
from django.contrib.auth.base_user import BaseUserManager
from django.db import models
from django.utils import timezone
import uuid

class FacultyManager(BaseUserManager):
    def create_user(self, faculty_code, email, password=None, **extra_fields):
        if not faculty_code:
            raise ValueError('Phải có mã giảng viên (faculty_code)')
        email = self.normalize_email(email)
        # Vì kế thừa AbstractUser nên vẫn cần field username, ta gán nó bằng faculty_code
        user = self.model(faculty_code=faculty_code, email=email, username=faculty_code, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, faculty_code, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('is_active_faculty', True) # Set luôn cái này cho superuser

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(faculty_code, email, password, **extra_fields)

class Faculty(AbstractUser):
    """
    Custom User model cho giảng viên với Enhanced Personalization
    """
    
    # CHOICES
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
    
    # ✅ NEW: Response style choices
    RESPONSE_STYLE_CHOICES = [
        ('professional', 'Chuyên nghiệp'),
        ('friendly', 'Thân thiện'),
        ('technical', 'Kỹ thuật'),
        ('brief', 'Ngắn gọn'),
        ('detailed', 'Chi tiết')
    ]
    
    # Basic fields
    faculty_code = models.CharField(
        max_length=20, 
        unique=True, 
        help_text="Mã giảng viên (VD: GV001, BDU2024001)"
    )
    full_name = models.CharField(max_length=100, help_text="Họ và tên đầy đủ")
    
    # ✅ NEW: Thêm trường giới tính
    GENDER_CHOICES = [
        ('male', 'Nam'),
        ('female', 'Nữ'), 
        ('other', 'Khác'),
    ]

    gender = models.CharField(
        max_length=10,
        choices=GENDER_CHOICES,
        default='other',
        blank=True,
        verbose_name="Giới tính",
        help_text="Giới tính để xác định cách xưng hô (thầy/cô)"
    )
    
    department = models.CharField(
        max_length=20, 
        choices=DEPARTMENT_CHOICES, 
        default='general',
        help_text="Khoa/Ngành chuyên môn"
    )
    
    phone = models.CharField(max_length=15, blank=True)
    
    position = models.CharField(
        max_length=20,
        choices=POSITION_CHOICES,
        default='giang_vien',
        verbose_name="Chức vụ"
    )
    
    specialization = models.TextField(
        blank=True, 
        verbose_name="Chuyên môn/Lĩnh vực nghiên cứu"
    )
    office_room = models.CharField(
        max_length=20, 
        blank=True, 
        verbose_name="Phòng làm việc"
    )
    
    # ✅ ENHANCED: Chatbot preferences với better structure
    chatbot_preferences = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Tùy chọn chatbot",
        help_text="Lưu các tùy chọn cá nhân hóa chatbot"
    )
    
    # Status fields
    is_active_faculty = models.BooleanField(default=True, help_text="Có đang làm việc không")
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'faculty_code'
    REQUIRED_FIELDS = ['email', 'full_name']
    
    objects = FacultyManager()
    
    class Meta:
        db_table = 'faculty'
        verbose_name = 'Giảng viên'
        verbose_name_plural = 'Danh sách giảng viên'
        
    def __str__(self):
        return f"{self.faculty_code} - {self.full_name} ({self.get_department_display()})"
    
    def save(self, *args, **kwargs):
        # Auto set username = faculty_code
        self.username = self.faculty_code
        
        # ✅ ENHANCED: Auto-setup chatbot preferences với better defaults
        if not self.chatbot_preferences:
            self.chatbot_preferences = self.get_default_chatbot_preferences()
        
        super().save(*args, **kwargs)
    
    # ✅ ENHANCED: Better default preferences
    def get_default_chatbot_preferences(self):
        """Tạo default chatbot preferences theo ngành với response_style"""
        return {
            'user_memory_prompt': self.get_default_memory_prompt(),
            'response_style': 'professional',  # Default style
            'department_priority': True,
            'auto_role_loaded': True,
            'role_setup_date': timezone.now().isoformat()
        }
    
    def get_default_memory_prompt(self):
        """Tạo default memory prompt theo ngành và chức vụ"""
        base_info = f"Tôi là {self.get_position_display()} {self.get_department_display()}"
        
        department_prompts = {
            'cntt': f"{base_info}. Tôi quan tâm đến lập trình, AI/ML, cơ sở dữ liệu, và công nghệ web. Tôi thích câu trả lời có ví dụ code và giải pháp thực tế.",
            'duoc': f"{base_info}. Tôi chuyên về dược lý, hóa dược, và quản lý dược. Tôi quan tâm đến an toàn thuốc, tương tác thuốc, và quy định ngành dược.",
            'dien_tu': f"{base_info}. Tôi làm việc với mạch điện tử, vi xử lý, IoT và tự động hóa. Tôi thích thông tin về thiết bị, datasheet và ứng dụng thực tế.",
            'co_khi': f"{base_info}. Tôi chuyên về thiết kế máy, CAD/CAM, gia công và sản xuất. Tôi quan tâm đến công nghệ sản xuất và quản lý chất lượng.",
            'y_khoa': f"{base_info}. Tôi làm việc trong lĩnh vực lâm sàng, chẩn đoán và điều trị. Tôi cần thông tin y khoa chính xác và cập nhật.",
            'kinh_te': f"{base_info}. Tôi quan tâm đến tài chính, đầu tư, phân tích kinh tế và chính sách. Tôi thích dữ liệu và phân tích số liệu.",
            'luat': f"{base_info}. Tôi chuyên về pháp lý, hợp đồng và tư vấn luật. Tôi cần thông tin chính xác về quy định và thủ tục pháp lý.",
            'ngoai_ngu': f"{base_info}. Tôi giảng dạy ngoại ngữ và quan tâm đến phương pháp giảng dạy, văn hóa và giao tiếp.",
            'xay_dung': f"{base_info}. Tôi chuyên về kết cấu, vật liệu xây dựng và quản lý dự án. Tôi quan tâm đến tiêu chuẩn kỹ thuật và quy chuẩn.",
            'quan_tri': f"{base_info}. Tôi quan tâm đến quản lý doanh nghiệp, chiến lược và phát triển tổ chức.",
            'ke_toan': f"{base_info}. Tôi chuyên về kế toán, kiểm toán và báo cáo tài chính. Tôi cần thông tin về chuẩn mực và quy định kế toán.",
            'marketing': f"{base_info}. Tôi quan tâm đến marketing digital, thương hiệu và hành vi người tiêu dùng.",
            'tai_chinh': f"{base_info}. Tôi chuyên về tài chính doanh nghiệp, ngân hàng và đầu tư. Tôi quan tâm đến phân tích tài chính và quản lý rủi ro."
        }
        
        return department_prompts.get(self.department, f"{base_info}. Tôi quan tâm đến thông tin chung về giáo dục đại học.")
    
    # ✅ NEW: Style-specific prompt templates
    def get_style_specific_instructions(self, response_style):
        """Get style-specific instructions for system prompt"""
        style_instructions = {
            'professional': """
✅ PHONG CÁCH CHUYÊN NGHIỆP:
- Ngôn từ trang trọng, lịch sự, chuẩn mực
- Sử dụng thuật ngữ chính xác và phù hợp
- Trình bày có hệ thống, logic rõ ràng
- Tôn trọng cấp bậc và quy trình
- Giọng điệu nghiêm túc nhưng thân thiện""",
            
            'friendly': """
✅ PHONG CÁCH THÂN THIỆN:
- Ngôn từ gần gũi, ấm áp và dễ chịu
- Sử dụng emoji phù hợp để tạo không khí vui vẻ 😊
- Tạo cảm giác thoải mái, gần gũi
- Giọng điệu vui vẻ, nhiệt tình
- Thể hiện sự quan tâm và sẵn sàng giúp đỡ""",
            
            'technical': """
✅ PHONG CÁCH KỸ THUẬT:
- Sử dụng thuật ngữ chuyên môn chính xác
- Giải thích chi tiết các khía cạnh kỹ thuật  
- Đưa ra ví dụ cụ thể, số liệu thực tế
- Tập trung vào độ chính xác và đầy đủ
- Phân tích sâu các vấn đề phức tạp""",
            
            'brief': """
✅ PHONG CÁCH NGẮN GỌN:
- Trả lời súc tích, đi thẳng vào trọng tâm
- Tối đa 1-2 câu cho mỗi ý chính
- Không giải thích dài dòng hay lòng vòng
- Tập trung vào thông tin cốt lõi nhất
- Loại bỏ các chi tiết không cần thiết""",
            
            'detailed': """
✅ PHONG CÁCH CHI TIẾT:
- Giải thích đầy đủ, toàn diện từng khía cạnh
- Đưa ra nhiều ví dụ minh họa cụ thể
- Phân tích từ nhiều góc độ khác nhau
- Cung cấp ngữ cảnh và background rộng
- Bao gồm các thông tin liên quan và tham khảo"""
        }
        
        return style_instructions.get(response_style, style_instructions['professional'])
    
    # ✅ ENHANCED: Personalized system prompt với response_style
    def get_personalized_system_prompt(self):
        """Tạo system prompt cá nhân hóa dựa trên vai trò và preferences"""
        
        # Lấy preferences
        user_memory = self.chatbot_preferences.get('user_memory_prompt', '').strip()
        if not user_memory:
            user_memory = self.get_default_memory_prompt()
            
        response_style = self.chatbot_preferences.get('response_style', 'professional')
        department_priority = self.chatbot_preferences.get('department_priority', True)
        
        # ✅ FIXED: Personal addressing - sử dụng self.get_personal_address()
        personal_address = self.get_personal_address()
        
        base_prompt = f"""Bạn là AI assistant chuyên nghiệp của Đại học Bình Dương (BDU).

🎯 THÔNG TIN NGƯỜI DÙNG:
- Mã GV: {self.faculty_code}
- Họ tên: {self.full_name}
- Vai trò: {self.get_role_description()}

🧠 THÔNG TIN CÁ NHÂN:
{user_memory}

🤖 QUY TẮC GIAO TIẾP:
- LUÔN xưng hô: "{personal_address}" (chỉ khi biết giới tính, ngược lại bỏ qua)
- Bắt đầu: "Dạ {personal_address}," nếu biết, hoặc "Dạ," nếu không rõ
- Kết thúc: "Cần em hỗ trợ thêm gì không ạ?" (KHÔNG kèm chức danh nếu không rõ)
- KHÔNG CHẾ TẠO thông tin không có

{self.get_style_specific_instructions(response_style)}"""

        # Add department knowledge if enabled
        if department_priority and self.department != 'general':
            department_knowledge = self._get_department_specific_knowledge()
            if department_knowledge:
                base_prompt += f"""

🎓 CHUYÊN MÔN NGÀNH {self.get_department_display().upper()}:
{department_knowledge}

📚 ƯU TIÊN CHUYÊN NGÀNH:
- Tập trung vào thông tin liên quan đến ngành {self.get_department_display()}
- Sử dụng thuật ngữ chuyên ngành phù hợp
- Ưu tiên giải pháp thực tế trong ngành"""
        else:
            base_prompt += """

🔄 CHẾ ĐỘ TỔNG QUÁT:
- Trả lời thông tin chung về BDU
- Không tập trung vào chuyên ngành cụ thể"""

        return base_prompt
    
    def _get_department_specific_knowledge(self):
        """Lấy kiến thức chuyên ngành (existing method - no changes)"""
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
            # ... other departments remain the same
        }
        
        return knowledge_map.get(self.department, "")
    
    # ✅ ENHANCED: Better validation cho preferences
    def update_chatbot_preferences(self, preferences_data):
        """Cập nhật tùy chọn chatbot với validation"""
        if not self.chatbot_preferences:
            self.chatbot_preferences = {}
        
        # ✅ NEW: Validate response_style
        if 'response_style' in preferences_data:
            valid_styles = [choice[0] for choice in self.RESPONSE_STYLE_CHOICES]
            if preferences_data['response_style'] not in valid_styles:
                raise ValueError(f"Invalid response_style. Must be one of: {valid_styles}")
        
        # ✅ NEW: Validate user_memory_prompt
        if 'user_memory_prompt' in preferences_data:
            memory_prompt = preferences_data['user_memory_prompt']
            if len(memory_prompt) > 1000:
                raise ValueError("user_memory_prompt cannot exceed 1000 characters")
        
        # ✅ NEW: Validate department_priority
        if 'department_priority' in preferences_data:
            if not isinstance(preferences_data['department_priority'], bool):
                raise ValueError("department_priority must be a boolean")
        
        self.chatbot_preferences.update(preferences_data)
        self.chatbot_preferences['last_updated'] = timezone.now().isoformat()
        self.save(update_fields=['chatbot_preferences'])
    
    # Helper methods (unchanged)
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
            'gender': self.gender,  # ✅ NEW: Thêm giới tính
            'department': self.department,
            'department_name': self.get_department_display(),
            'position': self.position,
            'position_name': self.get_position_display(),
            'role_description': self.get_role_description(),
            'specialization': self.specialization,
            'office_room': self.office_room,
            'preferences': self.chatbot_preferences,
            'is_lecturer': self.position in ['giang_vien', 'tro_giang', 'truong_khoa', 'pho_truong_khoa', 'truong_bo_mon'],
            'department_priority_enabled': self.chatbot_preferences.get('department_priority', True),
            'current_response_style': self.chatbot_preferences.get('response_style', 'professional')
        }
    
    def reset_to_auto_role(self):
        """Reset về vai trò tự động theo ngành"""
        self.chatbot_preferences = self.get_default_chatbot_preferences()
        self.save(update_fields=['chatbot_preferences'])
        return self.chatbot_preferences

    def get_salutation(self):
        """Xác định cách xưng hô dựa trên giới tính. Trả '' nếu không rõ."""
        if self.gender == 'male':
            return 'thầy'
        elif self.gender == 'female':
            return 'cô'
        else:
            return ''  # Không rõ giới tính → không xưng hô "giảng viên"

    def get_personal_address(self):
        """Lấy cách xưng hô kèm tên. Trả '' nếu không rõ giới tính."""
        salutation = self.get_salutation()
        if self.full_name:
            name_suffix = self.full_name.split()[-1]
            # Nếu là thầy/cô thì đi kèm tên
            if salutation in ['thầy', 'cô']:
                return f"{salutation} {name_suffix}"
        # salutation rỗng (không rõ giới tính) → trả '' để chatbot bắt đầu bằng "Dạ,"
        if salutation:
            return salutation
        return ''
    
# Existing models remain unchanged
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