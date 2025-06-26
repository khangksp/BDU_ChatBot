import jwt
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
from django.conf import settings

logger = logging.getLogger(__name__)

class ExternalAPIService:
    """
    Service for handling external API calls to school systems
    Handles JWT token decoding and API communication
    """
    
    def __init__(self):
        # API endpoints - có thể config trong settings
        self.base_url = getattr(settings, 'SCHOOL_API_BASE_URL', 'https://cds.bdu.edu.vn')
        self.schedule_endpoint = f"{self.base_url}/app_cbgv/odp/vien_chuc/thoi_khoa_bieu"
        
        # JWT settings
        self.jwt_secret = getattr(settings, 'JWT_SECRET_KEY', None)
        self.jwt_algorithm = getattr(settings, 'JWT_ALGORITHM', 'HS256')
        
        # Cache to avoid repeated API calls within short timeframe
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("✅ ExternalAPIService initialized")
    
    def decode_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode JWT token to extract lecturer information
        Returns payload with ma_vien_chuc, ho_va_ten, etc.
        """
        try:
            if token.startswith('Bearer '):
                token = token[7:]  # Remove 'Bearer ' prefix
            
            # Nếu không có secret key, thử decode without verification (for testing)
            if not self.jwt_secret:
                logger.warning("⚠️ JWT_SECRET_KEY not configured, decoding without verification")
                decoded = jwt.decode(token, options={"verify_signature": False})
            else:
                decoded = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            logger.info(f"✅ JWT decoded successfully for user: {decoded.get('vien_chuc', {}).get('ho_va_ten', 'Unknown')}")
            return decoded
            
        except jwt.ExpiredSignatureError:
            logger.error("❌ JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"❌ Invalid JWT token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Error decoding JWT: {str(e)}")
            return None
    
    def get_lecturer_info_from_token(self, token: str) -> Optional[Dict[str, str]]:
        """
        Extract lecturer basic info from JWT token
        Returns: {ma_giang_vien, ten_giang_vien, gmail, chuc_danh}
        """
        payload = self.decode_jwt_token(token)
        if not payload:
            return None
        
        try:
            vien_chuc = payload.get('vien_chuc', {})
            
            lecturer_info = {
                'ma_giang_vien': vien_chuc.get('ma_vien_chuc', ''),
                'ten_giang_vien': vien_chuc.get('ho_va_ten', ''),
                'gmail': vien_chuc.get('gmail', ''),
                'chuc_danh': vien_chuc.get('chuc_danh', ''),
                'vi_tri_viec_lam': vien_chuc.get('vi_tri_viec_lam', ''),
                'trinh_do': vien_chuc.get('trinh_do', ''),
                'ma_don_vi': vien_chuc.get('ma_don_vi', ''),
                'so_dien_thoai': vien_chuc.get('so_dien_thoai', '')
            }
            
            logger.info(f"📋 Extracted lecturer info: {lecturer_info['ma_giang_vien']} - {lecturer_info['ten_giang_vien']}")
            return lecturer_info
            
        except Exception as e:
            logger.error(f"❌ Error extracting lecturer info: {str(e)}")
            return None
    
    def get_lecturer_schedule(self, token: str, query_context: str = '') -> Dict[str, Any]:
        """
        Get lecturer schedule from external API
        Returns formatted data for Gemini processing
        """
        try:
            # 1. Get lecturer info from token
            lecturer_info = self.get_lecturer_info_from_token(token)
            if not lecturer_info:
                return {
                    'success': False,
                    'error': 'Không thể xác thực thông tin giảng viên từ token',
                    'error_type': 'token_decode_failed'
                }
            
            ma_giang_vien = lecturer_info['ma_giang_vien']
            ten_giang_vien = lecturer_info['ten_giang_vien']
            
            # 2. Check cache first
            cache_key = f"schedule_{ma_giang_vien}"
            if cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if datetime.now() - cache_data['timestamp'] < timedelta(seconds=self.cache_duration):
                    logger.info(f"🎯 Using cached schedule for {ma_giang_vien}")
                    cache_data['data']['from_cache'] = True
                    return cache_data['data']
            
            # 3. Call external API
            logger.info(f"🌐 Calling schedule API for {ma_giang_vien} - {ten_giang_vien}")
            
            headers = {
                'Authorization': f'Bearer {token.replace("Bearer ", "")}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                self.schedule_endpoint,
                headers=headers,
                timeout=30
            )
            
            logger.info(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                api_data = response.json()
                schedule_data = api_data.get('data', [])
                
                logger.info(f"📅 Retrieved {len(schedule_data)} schedule entries")
                
                # 4. Process and format data
                formatted_data = self._process_schedule_data(
                    schedule_data, 
                    lecturer_info, 
                    query_context
                )
                
                # 5. Cache the result
                self.cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': formatted_data
                }
                
                return formatted_data
                
            elif response.status_code == 401:
                logger.error("❌ API Authentication failed - token expired or invalid")
                return {
                    'success': False,
                    'error': 'Token đã hết hạn hoặc không hợp lệ. Vui lòng đăng nhập lại.',
                    'error_type': 'authentication_failed'
                }
            else:
                logger.error(f"❌ API call failed with status: {response.status_code}")
                logger.error(f"❌ Response: {response.text}")
                return {
                    'success': False,
                    'error': 'Không thể kết nối đến hệ thống thời khóa biểu của trường',
                    'error_type': 'api_call_failed',
                    'status_code': response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error calling schedule API: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi kết nối mạng. Vui lòng thử lại sau.',
                'error_type': 'network_error'
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error in get_lecturer_schedule: {str(e)}")
            return {
                'success': False,
                'error': 'Đã xảy ra lỗi không mong muốn khi lấy thông tin lịch giảng dạy',
                'error_type': 'unexpected_error'
            }
    
    def _process_schedule_data(self, schedule_data: List[Dict], lecturer_info: Dict[str, str], query_context: str) -> Dict[str, Any]:
        """
        Process raw schedule data from API into formatted structure for Gemini
        """
        try:
            ma_giang_vien = lecturer_info['ma_giang_vien']
            ten_giang_vien = lecturer_info['ten_giang_vien']
            
            # Filter schedule entries for this lecturer
            lecturer_schedule = [
                entry for entry in schedule_data 
                if entry.get('ma_giang_vien') == ma_giang_vien
            ]
            
            logger.info(f"📊 Found {len(lecturer_schedule)} schedule entries for {ma_giang_vien}")
            
            # Group by date and sort
            daily_schedule = {}
            for entry in lecturer_schedule:
                date_str = entry.get('ngay_hoc', '')
                if date_str:
                    if date_str not in daily_schedule:
                        daily_schedule[date_str] = []
                    daily_schedule[date_str].append(entry)
            
            # Sort dates and entries within each date
            sorted_schedule = {}
            for date_str in sorted(daily_schedule.keys()):
                # Sort by tiet_bat_dau (starting period)
                sorted_entries = sorted(
                    daily_schedule[date_str], 
                    key=lambda x: x.get('tiet_bat_dau', 0)
                )
                sorted_schedule[date_str] = sorted_entries
            
            # Analyze query context for specific time filtering
            filtered_schedule = self._filter_schedule_by_query(sorted_schedule, query_context)
            
            # Format for Gemini
            formatted_result = {
                'success': True,
                'lecturer_info': lecturer_info,
                'schedule_summary': {
                    'total_classes': len(lecturer_schedule),
                    'date_range': {
                        'start': min(daily_schedule.keys()) if daily_schedule else None,
                        'end': max(daily_schedule.keys()) if daily_schedule else None
                    },
                    'unique_subjects': len(set([entry.get('ma_mon_hoc', '') for entry in lecturer_schedule])),
                    'total_periods': sum([entry.get('so_tiet', 0) for entry in lecturer_schedule])
                },
                'daily_schedule': filtered_schedule,
                'query_context': query_context,
                'processed_at': datetime.now().isoformat(),
                'from_cache': False
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"❌ Error processing schedule data: {str(e)}")
            return {
                'success': False,
                'error': 'Lỗi xử lý dữ liệu thời khóa biểu',
                'error_type': 'data_processing_error'
            }
    
    def _filter_schedule_by_query(self, schedule: Dict, query_context: str) -> Dict:
        """
        Filter schedule based on query context (today, tomorrow, this week, etc.)
        """
        if not query_context:
            return schedule
        
        query_lower = query_context.lower()
        today = datetime.now()
        
        # Define date filters based on common queries
        if any(keyword in query_lower for keyword in ['hôm nay', 'hom nay', 'today']):
            today_str = today.strftime('%d-%m-%Y')
            return {k: v for k, v in schedule.items() if k == today_str}
            
        elif any(keyword in query_lower for keyword in ['ngày mai', 'ngay mai', 'tomorrow']):
            tomorrow = today + timedelta(days=1)
            tomorrow_str = tomorrow.strftime('%d-%m-%Y')
            return {k: v for k, v in schedule.items() if k == tomorrow_str}
            
        elif any(keyword in query_lower for keyword in ['tuần này', 'tuan nay', 'this week']):
            # Get dates for current week (Monday to Sunday)
            start_of_week = today - timedelta(days=today.weekday())
            week_dates = []
            for i in range(7):
                date = start_of_week + timedelta(days=i)
                week_dates.append(date.strftime('%d-%m-%Y'))
            return {k: v for k, v in schedule.items() if k in week_dates}
            
        elif any(keyword in query_lower for keyword in ['tuần tới', 'tuan toi', 'next week']):
            # Get dates for next week
            start_of_next_week = today + timedelta(days=(7 - today.weekday()))
            week_dates = []
            for i in range(7):
                date = start_of_next_week + timedelta(days=i)
                week_dates.append(date.strftime('%d-%m-%Y'))
            return {k: v for k, v in schedule.items() if k in week_dates}
            
        # Return full schedule if no specific time filter
        return schedule
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for external API service"""
        return {
            'external_api_service': {
                'available': True,
                'base_url': self.base_url,
                'endpoints': {
                    'schedule': self.schedule_endpoint
                },
                'jwt_configured': bool(self.jwt_secret),
                'cache_entries': len(self.cache),
                'cache_duration_seconds': self.cache_duration,
                'features': [
                    'jwt_token_decoding',
                    'lecturer_schedule_retrieval',
                    'query_context_filtering',
                    'response_caching',
                    'error_handling'
                ]
            }
        }

# Singleton instance
external_api_service = ExternalAPIService()