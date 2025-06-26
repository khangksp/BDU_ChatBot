import os
import sys 
from pathlib import Path
from dotenv import load_dotenv
from .settings import *

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv()

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-your-secret-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() in ['true', '1', 'yes']

# =============================================================================
# 🌐 CẤU HÌNH MẠNG - QUAN TRỌNG KHI DEPLOY
# =============================================================================

# 🔥 KHI DEPLOY: Thêm IP server thật vào đây
ALLOWED_HOSTS = [
    '*',
    'localhost',           # Cho development trên máy local
    '127.0.0.1',          # IP local
    '0.0.0.0',            # Cho phép tất cả IP (chỉ dùng khi test)
    # '192.168.1.100',    # 🔥 DEPLOY: Bỏ # và thay bằng IP server thật
    # 'your-domain.com',  # 🔥 DEPLOY: Nếu có tên miền thì bỏ # và sửa
    '*.ngrok.io',  # Allow all ngrok subdomains
    '*.ngrok-free.app',
]

# =============================================================================
# ĐỊNH NGHĨA ỨNG DỤNG
# =============================================================================

DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

THIRD_PARTY_APPS = [
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
]

LOCAL_APPS = [
    'authentication',
    'knowledge',
    'chat',
    'ai_models',
    'qa_management',
]

QA_MANAGEMENT = {
    'AUTO_SYNC_ON_SAVE': False,  # Automatically sync to Drive when saving in admin
    'BACKUP_BEFORE_SYNC': True,  # Create backup before major sync operations
    'MAX_ENTRIES_PER_PAGE': 50,  # Pagination in admin
    'SYNC_BATCH_SIZE': 100,  # Number of entries to process in one batch
    'ADMIN_PERMISSIONS': {
        'SUPERUSER_ONLY': False,  # If True, only superusers can access QA management
        'STAFF_REQUIRED': True,  # Staff permission required
        'CUSTOM_PERMISSIONS': []  # Custom permissions if needed
    },
    'UI_SETTINGS': {
        'SHOW_PREVIEW_ROWS': 5,  # Number of rows to show in CSV preview
        'MAX_UPLOAD_SIZE_MB': 10,  # Maximum CSV upload size
        'AUTO_REFRESH_INTERVAL': 30,  # Auto-refresh interval for sync status (seconds)
    }
}

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# Custom User Model
AUTH_USER_MODEL = 'authentication.Faculty'

# =============================================================================
# CẤU HÌNH MIDDLEWARE
# =============================================================================

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'backend.middleware.CSRFExemptMiddleware',
]

ROOT_URLCONF = 'backend.urls'

# =============================================================================
# CẤU HÌNH TEMPLATES
# =============================================================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# =============================================================================
# CẤU HÌNH DATABASE
# =============================================================================

# Mặc định: SQLite cho development (dễ setup)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'timeout': 20,
        }
    }
}

# =============================================================================
# KIỂM TRA MẬT KHẨU
# =============================================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# =============================================================================
# CHỈNH THỜI GIAN VÀ NGÔN NGỮ
# =============================================================================

LANGUAGE_CODE = 'vi-vn'
TIME_ZONE = 'Asia/Ho_Chi_Minh'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# =============================================================================
# FILE STATIC VÀ MEDIA
# =============================================================================

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# =============================================================================
# CẤU HÌNH SESSION
# =============================================================================

SESSION_COOKIE_AGE = int(os.getenv('SESSION_COOKIE_AGE', 1209600))  # 2 tuần
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_COOKIE_SECURE = not DEBUG  # True khi production
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_SAVE_EVERY_REQUEST = False

# =============================================================================
# CẤU HÌNH REST FRAMEWORK
# =============================================================================

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# =============================================================================
# 🌐 CẤU HÌNH CORS - CHO PHÉP FRONTEND KẾT NỐI
# =============================================================================

# 🔥 KHI DEPLOY: Thêm IP frontend thật vào đây
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Development - React dev server
    "http://127.0.0.1:3000",     # Development - Local  
    "http://localhost:8080",      # Port khác
    "http://127.0.0.1:8080",     # Port khác
    # "http://192.168.1.100:3000",  # 🔥 DEPLOY: Bỏ # và thay IP thật
    # "http://192.168.1.100:80",    # 🔥 DEPLOY: Nếu frontend chạy port 80
    # "https://your-domain.com",    # 🔥 DEPLOY: Nếu có HTTPS domain
    "https://3558-113-161-163-160.ngrok-free.app",
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Chỉ cho phép tất cả khi DEBUG=True

CORS_ALLOWED_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# =============================================================================
# 🔒 CẤU HÌNH BẢO MẬT
# =============================================================================

if not DEBUG:
    # Cài đặt bảo mật cho production
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_HSTS_SECONDS = 31536000  # 1 năm
    SECURE_REDIRECT_EXEMPT = []
    
    SECURE_SSL_REDIRECT = False # Chỉ bật khi có HTTPS
    
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    X_FRAME_OPTIONS = 'DENY'

# 🔥 KHI DEPLOY: Thêm IP/domain frontend thật vào đây
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",      # Development
    "http://127.0.0.1:3000",     # Development
    # "http://192.168.1.100:3000",  # 🔥 DEPLOY: Bỏ # và thay IP thật
    # "https://your-domain.com",    # 🔥 DEPLOY: Nếu có HTTPS domain
    "https://*.ngrok.io",
    "https://*.ngrok-free.app",
    "https://3558-113-161-163-160.ngrok-free.app",
]

# =============================================================================
# CẤU HÌNH LOGGING (SỬA LỖI CHO WINDOWS)
# ============================================================================= 

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'console_safe': {
            # Formatter an toàn không có emoji cho Windows console
            'format': '[{levelname}] {asctime} - {name} - {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG' if DEBUG else 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'console_safe',  # Dùng formatter an toàn
            'stream': sys.stdout,
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'authentication': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'chat': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'ai_models': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        
        'qa_management': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'qa_management.models': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'qa_management.services': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'qa_management.admin': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Đảm bảo các thư mục cần thiết tồn tại
os.makedirs(BASE_DIR / 'static', exist_ok=True)
os.makedirs(BASE_DIR / 'media', exist_ok=True)

# =============================================================================
# 🤖 CẤU HÌNH AI MODELS
# =============================================================================

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Cấu hình Speech-to-text
SPEECH_RECOGNITION_ENABLED = os.getenv('SPEECH_RECOGNITION_ENABLED', 'True').lower() in ['true', '1', 'yes']
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')

# Cấu hình Chat
MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', 50))
CHAT_RESPONSE_TIMEOUT = int(os.getenv('CHAT_RESPONSE_TIMEOUT', 30))

# =============================================================================
# 🎯 CẤU HÌNH PERSONALIZATION CHO FACULTY
# =============================================================================

# Personalization settings
CHATBOT_PERSONALIZATION = {
    'ENABLE_DEPARTMENT_BOOST': True,
    'ENABLE_ROLE_BASED_PROMPTS': True,
    'DEFAULT_RESPONSE_STYLE': 'professional',
    'MAX_FOCUS_AREAS': 5,
    'DEPARTMENT_CONFIDENCE_BOOST': 1.2,
    'FACULTY_SESSION_TIMEOUT': 3600,  # 1 hour
}

APPEND_SLASH = False

# Department specific configurations
DEPARTMENT_CONFIGS = {
    'cntt': {
        'keywords': ['lập trình', 'phần mềm', 'database', 'AI', 'machine learning', 'mạng', 'website'],
        'boost_factor': 1.3,
        'specializations': ['Web Development', 'Mobile App', 'AI/ML', 'Database', 'Network Security'],
        'emoji': '💻'
    },
    'duoc': {
        'keywords': ['thuốc', 'dược phẩm', 'hóa dược', 'vi sinh', 'phân tích', 'dược lý'],
        'boost_factor': 1.2,
        'specializations': ['Dược lý', 'Hóa dược', 'Công nghệ dược', 'Dược lâm sàng'],
        'emoji': '💊'
    },
    'dien_tu': {
        'keywords': ['mạch điện', 'vi xử lý', 'IoT', 'embedded', 'robot', 'sensor'],
        'boost_factor': 1.2,
        'specializations': ['IoT', 'Robotics', 'Automation', 'Signal Processing'],
        'emoji': '🔌'
    },
    'co_khi': {
        'keywords': ['máy móc', 'thiết kế', 'CAD', 'gia công', 'sản xuất', 'chế tạo'],
        'boost_factor': 1.2,
        'specializations': ['Thiết kế máy', 'CAD/CAM', 'Automation', 'Manufacturing'],
        'emoji': '⚙️'
    },
    'y_khoa': {
        'keywords': ['y tế', 'bệnh', 'điều trị', 'chẩn đoán', 'bệnh viện', 'bác sĩ'],
        'boost_factor': 1.2,
        'specializations': ['Nội khoa', 'Ngoại khoa', 'Sản phụ khoa', 'Nhi khoa'],
        'emoji': '🏥'
    },
    'kinh_te': {
        'keywords': ['tài chính', 'ngân hàng', 'đầu tư', 'kinh doanh', 'thị trường', 'kế toán'],
        'boost_factor': 1.2,
        'specializations': ['Tài chính doanh nghiệp', 'Ngân hàng', 'Chứng khoán', 'Marketing'],
        'emoji': '💰'
    },
    'luat': {
        'keywords': ['luật', 'pháp lý', 'hợp đồng', 'quy định', 'tòa án', 'luật sư'],
        'boost_factor': 1.2,
        'specializations': ['Luật dân sự', 'Luật hình sự', 'Luật kinh tế', 'Luật lao động'],
        'emoji': '⚖️'
    }
}

# Faculty position configurations
POSITION_CONFIGS = {
    'truong_khoa': {
        'priority_level': 'high',
        'access_level': 'management',
        'response_style': 'formal_detailed'
    },
    'pho_truong_khoa': {
        'priority_level': 'high',
        'access_level': 'management',
        'response_style': 'formal_detailed'
    },
    'truong_bo_mon': {
        'priority_level': 'medium',
        'access_level': 'department',
        'response_style': 'professional'
    },
    'giang_vien': {
        'priority_level': 'standard',
        'access_level': 'faculty',
        'response_style': 'professional'
    },
    'tro_giang': {
        'priority_level': 'standard',
        'access_level': 'faculty',
        'response_style': 'supportive'
    }
}

# Response style templates
RESPONSE_STYLES = {
    'professional': {
        'tone': 'formal_friendly',
        'detail_level': 'moderate',
        'technical_terms': True
    },
    'formal_detailed': {
        'tone': 'very_formal',
        'detail_level': 'comprehensive',
        'technical_terms': True
    },
    'supportive': {
        'tone': 'encouraging',
        'detail_level': 'detailed_with_examples',
        'technical_terms': False
    },
    'technical': {
        'tone': 'precise',
        'detail_level': 'technical_focused',
        'technical_terms': True
    },
    'brief': {
        'tone': 'direct',
        'detail_level': 'concise',
        'technical_terms': False
    }
}

# ✅ CẬP NHẬT: Logging configuration cho personalization
LOGGING['loggers'].update({
    'authentication.models': {
        'handlers': ['console'],
        'level': 'INFO',
        'propagate': True,
    },
    'authentication.views': {
        'handlers': ['console'],
        'level': 'INFO',
        'propagate': True,
    },
    'ai_models.gemini_service': {
        'handlers': ['console'],
        'level': 'INFO',
        'propagate': True,
    },
    'django.security': {
        'handlers': ['console'],
        'level': 'DEBUG',
        'propagate': True,
    },
})

# Faculty preferences validation
VALID_RESPONSE_STYLES = ['professional', 'friendly', 'technical', 'brief', 'detailed']
VALID_NOTIFICATION_TYPES = ['email_updates', 'system_notifications', 'department_news']
MAX_FOCUS_AREAS_PER_FACULTY = 5

# Personalization cache settings
PERSONALIZATION_CACHE = {
    'FACULTY_CONTEXT_TIMEOUT': 1800,  # 30 minutes
    'DEPARTMENT_KEYWORDS_TIMEOUT': 3600,  # 1 hour
    'SYSTEM_PROMPT_TIMEOUT': 1800,  # 30 minutes
}

# =============================================================================
# 🔗 CẤU HÌNH GOOGLE DRIVE 
# =============================================================================

# Google Drive settings
GOOGLE_DRIVE = {
    'ENABLED': True,
    'FOLDER_ID': '1589N-eP0KW3SLZwQtibxXnVoMrubPCqM',  # ID từ link Drive
    'CSV_FILENAME': 'QA.csv',
    'SERVICE_ACCOUNT_FILE': BASE_DIR / 'thinking-armor-463404-n1-627b306232a8.json',
    'CACHE_TIMEOUT': 60,  # 1 phút
    'SCOPES': ['https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/drive']
}

GOOGLE_DRIVE.update({
    'WRITE_ENABLED': True,  # Enable write operations
    'BACKUP_ENABLED': True,  # Enable automatic backups
    'BACKUP_RETENTION_DAYS': 30,  # Keep backups for 30 days
    'CONFLICT_RESOLUTION': 'database_wins',  # Options: 'database_wins', 'drive_wins', 'ask_user'
    'BATCH_UPLOAD_SIZE': 1000,  # Number of entries per batch upload
})

# Data source priority
KNOWLEDGE_BASE_SOURCES = {
    'PRIMARY': 'google_drive',    # Ưu tiên Google Drive
    'FALLBACK': 'local_csv',      # Fallback về local CSV
    'SECONDARY': 'database'       # Database là nguồn phụ
}

# =============================================================================
# 🔗 CẤU HÌNH QA MANAGEMENT
# =============================================================================

# ✅ NEW: Task scheduling (for future cron jobs)
QA_SYNC_SCHEDULE = {
    'AUTO_IMPORT_ENABLED': False,  # Enable automatic import from Drive
    'AUTO_IMPORT_INTERVAL': 3600,  # Import every hour (seconds)
    'AUTO_EXPORT_ENABLED': False,  # Enable automatic export to Drive
    'AUTO_EXPORT_INTERVAL': 1800,  # Export every 30 minutes (seconds)
    'INDEX_REBUILD_ENABLED': True,  # Enable automatic FAISS index rebuild
    'INDEX_REBUILD_AFTER_SYNC': True,  # Rebuild index after successful sync
}

# ✅ ENHANCED: Admin interface customization
ADMIN_INTERFACE = {
    'QA_MANAGEMENT': {
        'SHOW_DASHBOARD_STATS': True,
        'ENABLE_BULK_ACTIONS': True,
        'SHOW_SYNC_STATUS': True,
        'AUTO_SAVE_DRAFTS': False,  # For future implementation
        'ENABLE_PREVIEW_MODE': True,
    }
}

# ✅ NEW: Data validation settings
QA_DATA_VALIDATION = {
    'STT_PATTERN': r'^[A-Za-z0-9_-]+$',  # STT format validation
    'MIN_QUESTION_LENGTH': 5,  # Minimum question length
    'MIN_ANSWER_LENGTH': 10,  # Minimum answer length
    'MAX_QUESTION_LENGTH': 500,  # Maximum question length
    'MAX_ANSWER_LENGTH': 2000,  # Maximum answer length
    'FORBIDDEN_WORDS': [],  # Words that should not appear in Q&A
    'REQUIRED_KEYWORDS': [],  # Keywords that should appear (optional)
}

# ✅ ENHANCED: Security settings for QA Management
if not DEBUG:
    # Production security for QA Management
    QA_MANAGEMENT.update({
        'REQUIRE_2FA': False,  # Require 2FA for QA management (future)
        'IP_WHITELIST': [],  # IP whitelist for QA admin access
        'SESSION_TIMEOUT': 3600,  # Session timeout for QA admin (seconds)
        'AUDIT_LOG_ENABLED': True,  # Enable audit logging
    })

# ✅ NEW: Integration settings
CHATBOT_INTEGRATION = {
    'AUTO_REBUILD_INDEX': True,  # Automatically rebuild FAISS index after QA changes
    'CACHE_INVALIDATION': True,  # Invalidate chatbot cache after QA changes
    'NOTIFICATION_ENABLED': False,  # Send notifications on QA updates (future)
    'WEBHOOK_URLS': [],  # Webhook URLs to call after QA updates (future)
}

# ✅ ENHANCED: File handling
FILE_UPLOAD_SETTINGS = {
    'QA_UPLOAD_PATH': 'qa_uploads/',
    'ALLOWED_EXTENSIONS': ['.csv', '.xlsx'],
    'MAX_FILE_SIZE': 10 * 1024 * 1024,  # 10MB
    'SCAN_FOR_VIRUSES': False,  # Enable virus scanning (future)
    'AUTO_CLEANUP_DAYS': 7,  # Auto-delete uploaded files after 7 days
}

# =============================================================================
# 🌐 EXTERNAL API INTEGRATION - MINIMAL CONFIG
# =============================================================================

# External API Settings - Chỉ cần những cái cơ bản nhất
SCHOOL_API_BASE_URL = 'https://cds.bdu.edu.vn'
JWT_SECRET_KEY = None  # None = test mode, không cần verify signature
JWT_ALGORITHM = 'HS256'

# External API Configuration
EXTERNAL_API_SETTINGS = {
    'ENABLE_EXTERNAL_API': True,
    'CACHE_DURATION_SECONDS': 300,  # 5 phút
    'REQUEST_TIMEOUT_SECONDS': 30,
    'LECTURER_SCHEDULE_ENDPOINT': '/app_cbgv/odp/vien_chuc/thoi_khoa_bieu',
    'LOW_CONFIDENCE_THRESHOLD': 0.3,
}

# Feature Flags
FEATURE_FLAGS = {
    'EXTERNAL_API_ENABLED': True,
    'JWT_AUTHENTICATION_ENABLED': True,
    'PERSONAL_SCHEDULE_ACCESS_ENABLED': True,
}

# Tạo thư mục logs nếu chưa có
os.makedirs(BASE_DIR / 'logs', exist_ok=True)

# Debug info
if DEBUG:
    print("🚀 External API Integration: ✅ ENABLED")
    print(f"📡 School API URL: {SCHOOL_API_BASE_URL}")
    print(f"🔑 JWT Test Mode: ✅ ENABLED (No signature verification)")