from django.urls import path
from . import views

urlpatterns = [
    path('', views.APIRootView.as_view(), name='api-root'),  # ← API root
    path('chat/', views.ChatView.as_view(), name='chat'),
    path('history/', views.ChatHistoryView.as_view(), name='chat-history'),
    path('history/<str:session_id>/', views.ChatHistoryView.as_view(), name='chat-history-session'),
    path('feedback/', views.FeedbackView.as_view(), name='feedback'),
    path('health/', views.HealthCheckView.as_view(), name='health-check'),
    
    # Speech-to-Text
    path('speech-to-text/', views.SpeechToTextView.as_view(), name='speech-to-text'),
    path('speech-status/', views.SpeechStatusView.as_view(), name='speech-status'),
    
    # Personalized chat endpoints
    path('personalized-context/', views.PersonalizedChatContextView.as_view(), name='personalized-chat-context'),
    path('system-status-personalized/', views.PersonalizedSystemStatusView.as_view(), name='system-status-personalized'),
]