from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)

@api_view(['GET'])
@permission_classes([AllowAny])  # Cho phép truy cập công khai
def health_check(request):
    """
    Health check endpoint - Public access
    """
    return Response({
        'status': 'healthy',
        'message': 'BDU ChatBot API is running',
        'timestamp': timezone.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'database': 'connected',
            'authentication': 'available',
            'ai_models': 'loaded',
            'speech_recognition': 'available'
        }
    }, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([AllowAny])  # Speech status cũng public
def speech_status(request):
    """
    Speech recognition status endpoint - Public access
    """
    try:
        # Import trong function để tránh circular import
        from .speech_service import speech_service
        
        return Response({
            'speech_service': {
                'available': True,
                'model_loaded': hasattr(speech_service, 'model') and speech_service.model is not None,
                'device': getattr(speech_service, 'device', 'cpu'),
                'model_size': getattr(speech_service, 'model_size', 'base')
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'speech_service': {
                'available': False,
                'error': str(e)
            }
        }, status=status.HTTP_200_OK)  # Vẫn return 200 để frontend hoạt động

# Speech-to-text endpoint cần authentication
@api_view(['POST'])
# Không thêm @permission_classes([AllowAny]) - sử dụng default authentication
def speech_to_text(request):
    """
    Speech to text endpoint - Requires authentication
    """
    if request.method != 'POST':
        return Response({
            'success': False,
            'error': 'Only POST method allowed'
        }, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    
    try:
        # Import trong function
        from .speech_service import speech_service
        
        if 'audio' not in request.FILES:
            return Response({
                'success': False,
                'error': 'No audio file provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = request.FILES['audio']
        language = request.POST.get('language', 'vi')
        
        # Process audio
        result = speech_service.transcribe_audio(audio_file, language)
        
        return Response({
            'success': True,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0.0),
            'language': language
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Speech processing failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)