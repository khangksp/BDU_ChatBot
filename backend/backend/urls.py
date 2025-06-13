<<<<<<< HEAD
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse

def home_view(request):
    """Trang chủ API với personalization info"""
    return JsonResponse({
        'message': 'Chào mừng đến với Chatbot cá nhân hóa của Đại học Bình Dương!',
        'version': '2.0.0',  # ✅ NÂNG CẤP VERSION
        'features': [
            'Natural Language Processing',
            'Speech-to-Text Integration', 
            'Faculty Personalization',  # ✅ THÊM
            'Department-specific Responses',  # ✅ THÊM
            'Role-based Chatbot Prompts'  # ✅ THÊM
        ],
        'endpoints': {
            'admin': '/admin/',
            'health': '/api/health/',
            'chat': '/api/chat/',
            'knowledge': '/api/knowledge/',
            'authentication': '/api/auth/',
            'personalized_chat': '/api/personalized-context/',  # ✅ THÊM
            'api_docs': '/api/',
        },
        'personalization': {
            'enabled': True,
            'supported_departments': [
                'Công nghệ thông tin', 'Dược', 'Điện tử viễn thông',
                'Cơ khí', 'Y khoa', 'Kinh tế', 'Luật'
            ],
            'supported_positions': [
                'Giảng viên', 'Trợ giảng', 'Trưởng khoa', 
                'Phó trưởng khoa', 'Trưởng bộ môn', 'Cán bộ'
            ]
        },
        'status': 'running'
    })

urlpatterns = [
    path('', home_view, name='home'),
    path('admin/', admin.site.urls),
    path('api/', include('chat.urls')),
    path('api/knowledge/', include('knowledge.urls')),
    path('api/auth/', include('authentication.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
=======
"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
>>>>>>> e8b177af1a7d44e5e53eef8ef515df70c4164c31
