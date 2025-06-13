import os
import sys 
from pathlib import Path
from dotenv import load_dotenv

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
    'localhost',           # Cho development trên máy local
    '127.0.0.1',          # IP local
    '0.0.0.0',            # Cho phép tất cả IP (chỉ dùng khi test)
    # '192.168.1.100',    # 🔥 DEPLOY: Bỏ # và thay bằng IP server thật
    # 'your-domain.com',  # 🔥 DEPLOY: Nếu có tên miền thì bỏ # và sửa
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
]

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
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    X_FRAME_OPTIONS = 'DENY'

# 🔥 KHI DEPLOY: Thêm IP/domain frontend thật vào đây
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",      # Development
    "http://127.0.0.1:3000",     # Development
    # "http://192.168.1.100:3000",  # 🔥 DEPLOY: Bỏ # và thay IP thật
    # "https://your-domain.com",    # 🔥 DEPLOY: Nếu có HTTPS domain
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