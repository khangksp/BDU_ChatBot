from django.urls import path
from . import views

app_name = 'ai_models'

urlpatterns = [
    # Public endpoints (no authentication required)
    path('health/', views.health_check, name='health_check'),
    path('speech-status/', views.speech_status, name='speech_status'),
    
    # Protected endpoints (authentication required)
    path('speech-to-text/', views.speech_to_text, name='speech_to_text'),
]