from django.urls import path
from . import views

app_name = 'authentication'

urlpatterns = [
    # Authentication APIs
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('status/', views.auth_status, name='auth_status'),
    
    # Password management
    path('password/reset/request/', views.password_reset_request, name='password_reset_request'),
    path('password/reset/confirm/', views.password_reset_confirm, name='password_reset_confirm'),
    path('password/change/', views.change_password, name='change_password'),
    
    # Personalization endpoints
    path('chatbot/preferences/', views.chatbot_preferences, name='chatbot_preferences'),
    path('chatbot/preferences/update/', views.update_chatbot_preferences, name='update_chatbot_preferences'),
    path('chatbot/system-prompt/', views.personalized_system_prompt, name='personalized_system_prompt'),
    path('chatbot/focus-areas/', views.update_department_focus, name='update_department_focus'),
    path('chatbot/suggestions/', views.get_department_suggestions, name='get_department_suggestions'),
]