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
    
    # ✅ NEW: Text-to-Speech test endpoint
    path('text-to-speech-test/', views.TextToSpeechTestView.as_view(), name='text-to-speech-test'),
    
    # Personalized chat endpoints
    path('personalized-context/', views.PersonalizedChatContextView.as_view(), name='personalized-chat-context'),
    path('system-status-personalized/', views.PersonalizedSystemStatusView.as_view(), name='system-status-personalized'),
    
    # Chat sessions management
    path('chat-sessions/', views.ChatSessionsView.as_view(), name='chat-sessions'),
    path('chat-sessions/<str:session_id>/', views.ChatSessionDetailView.as_view(), name='chat-session-detail'),
]