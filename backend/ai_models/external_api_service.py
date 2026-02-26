import jwt
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import re # ✅ NÂNG CẤP: Thêm thư viện re để tìm ngày/tháng
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
        self.schedule_week_endpoint = f"{self.base_url}/app_cbgv/odp/vien_chuc/thoi_khoa_bieu_tuan"
        
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
            
            # ✅ FIXED: Chuyển đổi giới tính đúng — None/unknown → 'other'
            gioi_tinh = vien_chuc.get('gioi_tinh')
            if gioi_tinh == 0:
                gender_str = 'male'
            elif gioi_tinh == 1:
                gender_str = 'female'
            else:
                gender_str = 'other'  # None hoặc giá trị lạ → không rõ giới tính
            
            lecturer_info = {
                'ma_giang_vien': vien_chuc.get('ma_vien_chuc', ''),
                'ten_giang_vien': vien_chuc.get('ho_va_ten', ''),
                'gender': gender_str,
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
    
    def get_lecturer_schedule_by_week(self, token: str, query_context: str = '') -> Dict[str, Any]:
        """
        Lấy lịch giảng dạy theo tuần qua API mới.
        Truyền ngày thư 2 (Monday) đầu tuần định dạng dd-mm-yyyy.
        API chỉ trả về dữ liệu của tuần đó, không có tình trạng 210 records.
        """
        try:
            lecturer_info = self.get_lecturer_info_from_token(token)
            if not lecturer_info:
                return {'success': False, 'error': 'Không xác thực được token', 'error_type': 'token_decode_failed'}

            # Xác định ngày thư 2 cần lấy dựa vào câu hỏi
            monday = self._get_target_monday(query_context)
            ngay_thu_2 = monday.strftime('%d-%m-%Y')

            logger.info(f"📅 Weekly schedule API: ngay_thu_2={ngay_thu_2} (query='{query_context}')")

            # Kiểm tra cache
            cache_key = f"week_{lecturer_info['ma_giang_vien']}_{ngay_thu_2}"
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if datetime.now() - cached['timestamp'] < timedelta(seconds=self.cache_duration):
                    logger.info(f"🎯 Using cached weekly schedule for {ngay_thu_2}")
                    return cached['data']

            headers = {
                'Authorization': f'Bearer {token.replace("Bearer ", "")}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            }
            resp = requests.get(
                self.schedule_week_endpoint,
                headers=headers,
                params={'ngay_thu_2': ngay_thu_2},
                timeout=15,
            )

            logger.info(f"📡 Weekly API Response: {resp.status_code}")

            if resp.status_code == 200:
                raw = resp.json()
                schedule_data = raw.get('data', [])
                logger.info(f"📅 Weekly schedule: {len(schedule_data)} entries for week {ngay_thu_2}")

                # Log field names từ entry đầu tiên để biết tên field thật
                if schedule_data:
                    logger.info(f"🔑 Weekly API field names: {list(schedule_data[0].keys())}")
                    logger.info(f"🔍 First entry sample: {schedule_data[0]}")

                # Group theo ngày, giữ TOÀN BỘ fields (không filter trước khi biết tên field thật)
                grouped: Dict[str, list] = {}
                for entry in schedule_data:
                    # Thử cả 2 tên field phổ biến cho ngày học
                    day = entry.get('ngay_hoc', '') or entry.get('ngayHoc', '') or entry.get('date', '')
                    if day:
                        grouped.setdefault(day, []).append(entry)

                result = {
                    'success': True,
                    'lecturer_info': lecturer_info,
                    'week_start': ngay_thu_2,
                    'daily_schedule': grouped,
                    'total_entries': len(schedule_data),
                    'query_context': query_context,
                }
                self.cache[cache_key] = {'timestamp': datetime.now(), 'data': result}
                return result

            elif resp.status_code == 401:
                return {'success': False, 'error': 'Token hết hạn hoặc không hợp lệ.', 'error_type': 'authentication_failed'}
            else:
                logger.error(f"❌ Weekly API error {resp.status_code}: {resp.text[:200]}")
                return {'success': False, 'error': 'Lỗi API từ server trường.', 'error_type': 'api_call_failed'}

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Weekly API network error: {e}")
            return {'success': False, 'error': 'Lỗi kết nối mạng.', 'error_type': 'network_error'}
        except Exception as e:
            logger.error(f"❌ Weekly API unexpected error: {e}")
            return {'success': False, 'error': str(e), 'error_type': 'unexpected_error'}

    def _get_target_monday(self, query_context: str) -> datetime:
        """
        Xác định ngày thư 2 (Monday) cần lấy dựa vào câu hỏi.
        Mặc định: tuần hiện tại.
        """
        today = datetime.now()
        # Ngày thư 2 của tuần hiện tại
        current_monday = today - timedelta(days=today.weekday())  # weekday(): 0=Mon

        if not query_context:
            return current_monday

        q = query_context.lower()

        # Ngày cụ thể dd/mm
        date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})', q)
        if date_match:
            d, m = int(date_match.group(1)), int(date_match.group(2))
            try:
                specific = datetime(today.year, m, d)
                # Lấy thư 2 của tuần chứa ngày đó
                return specific - timedelta(days=specific.weekday())
            except ValueError:
                pass

        # Hôm nay
        if any(k in q for k in ['hôm nay', 'hom nay', 'today', 'ngày hôm nay']):
            return today - timedelta(days=today.weekday())

        # Ngày mai
        if any(k in q for k in ['ngày mai', 'ngay mai', 'tomorrow']):
            tomorrow = today + timedelta(days=1)
            return tomorrow - timedelta(days=tomorrow.weekday())

        # Tuần trước
        if any(k in q for k in ['tuần trước', 'tuan truoc', 'last week']):
            return current_monday - timedelta(weeks=1)

        # Tuần sau / tuần tới
        if any(k in q for k in ['tuần sau nua', 'tuan sau nua', '2 tuần', '2 tuan']):
            return current_monday + timedelta(weeks=2)
        if any(k in q for k in ['tuần sau', 'tuan sau', 'tuần tới', 'tuan toi', 'next week']):
            return current_monday + timedelta(weeks=1)

        # Thứ trong tuần (sẽ lấy tuần chứa thứ đó gần nhất)
        weekday_map = {
            'thứ 2': 0, 'thu 2': 0, 'thứ hai': 0,
            'thứ 3': 1, 'thu 3': 1, 'thứ ba': 1,
            'thứ 4': 2, 'thu 4': 2, 'thứ tư': 2,
            'thứ 5': 3, 'thu 5': 3, 'thứ năm': 3,
            'thứ 6': 4, 'thu 6': 4, 'thứ sáu': 4,
            'thứ 7': 5, 'thu 7': 5, 'thứ bảy': 5,
            'chủ nhật': 6, 'chu nhat': 6,
        }
        for name, wd in weekday_map.items():
            if name in q:
                target = current_monday + timedelta(days=wd)
                if target < today:
                    target += timedelta(weeks=1)
                return target - timedelta(days=target.weekday())

        # Mặc định: tuần hiện tại
        return current_monday

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
            api_data = None # Khởi tạo biến api_data
            
            # ✅ SỬA LỖI 1: Logic cache
            # Thay đổi cache key để cache dữ liệu thô, chưa lọc
            cache_key = f"schedule_raw_{ma_giang_vien}" 
            if cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if datetime.now() - cache_data['timestamp'] < timedelta(seconds=self.cache_duration):
                    logger.info(f"🎯 Using RAW cached schedule for {ma_giang_vien}")
                    api_data = cache_data['data'] # Lấy dữ liệu thô từ cache
            
            # Nếu không có cache hoặc cache hết hạn, gọi API
            if api_data is None:
                logger.info(f"🌐 Calling schedule API for {ma_giang_vien}")
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
                    # Lưu dữ liệu THÔ vào cache
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': api_data
                    }
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

            # Sau khi có dữ liệu thô (từ cache hoặc API), tiến hành xử lý
            schedule_data = api_data.get('data', [])
            logger.info(f"📅 Retrieved {len(schedule_data)} schedule entries to process")
            
            # Việc xử lý và lọc sẽ diễn ra ở đây, đảm bảo mỗi lần hỏi đều được lọc lại
            formatted_data = self._process_schedule_data(
                schedule_data, 
                lecturer_info, 
                query_context
            )
            
            return formatted_data
            
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
            
            # Filter schedule entries for this lecturer
            lecturer_schedule = [
                entry for entry in schedule_data 
                if entry.get('ma_giang_vien') == ma_giang_vien
            ]
            
            logger.info(f"📊 Found {len(lecturer_schedule)} schedule entries for {ma_giang_vien}")
            
            # Group by date and sort
            daily_schedule_raw = {}
            for entry in lecturer_schedule:
                date_str = entry.get('ngay_hoc', '')
                if date_str:
                    if date_str not in daily_schedule_raw:
                        daily_schedule_raw[date_str] = []
                    daily_schedule_raw[date_str].append(entry)
            
            # Sort dates and entries within each date
            sorted_schedule_raw = {}
            for date_str in sorted(daily_schedule_raw.keys()):
                # Sort by tiet_bat_dau (starting period)
                sorted_entries = sorted(
                    daily_schedule_raw[date_str], 
                    key=lambda x: x.get('tiet_bat_dau', 0)
                )
                sorted_schedule_raw[date_str] = sorted_entries
            
            # Analyze query context for specific time filtering
            filtered_schedule = self._filter_schedule_by_query(sorted_schedule_raw, query_context)
            
            # Format for Gemini
            formatted_result = {
                'success': True,
                'lecturer_info': lecturer_info,
                'schedule_summary': {
                    'total_classes': len(lecturer_schedule),
                    'date_range': {
                        'start': min(daily_schedule_raw.keys()) if daily_schedule_raw else None,
                        'end': max(daily_schedule_raw.keys()) if daily_schedule_raw else None
                    },
                    'unique_subjects': len(set([entry.get('ma_mon_hoc', '') for entry in lecturer_schedule])),
                    'total_periods': sum([entry.get('so_tiet', 0) for entry in lecturer_schedule])
                },
                'daily_schedule': filtered_schedule,
                'query_context': query_context,
                'processed_at': datetime.now().isoformat(),
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
        🚀 ENHANCED FIX: Ultra-reliable schedule filtering với deterministic logic
        """
        if not query_context:
            return schedule
        
        query_lower = query_context.lower()
        today = datetime.now()
        
        logger.info(f"🔍 ENHANCED TIME FILTER: query='{query_context}', today={today.strftime('%d-%m-%Y')}")
        
        # ✅ CRITICAL FIX: Xử lý theo thứ tự ưu tiên STRICT
        
        # PRIORITY 1: Specific complex patterns (highest priority)
        complex_patterns = {
            'tuần sau nữa': lambda: self._get_week_dates(today, weeks_ahead=2),
            'tuan sau nua': lambda: self._get_week_dates(today, weeks_ahead=2),
            '2 tuần tới': lambda: self._get_week_dates(today, weeks_ahead=2),
            '2 tuan toi': lambda: self._get_week_dates(today, weeks_ahead=2),
            'cuối tuần này': lambda: self._get_weekend_dates(today, current_week=True),
            'cuoi tuan nay': lambda: self._get_weekend_dates(today, current_week=True),
            'đầu tuần sau': lambda: self._get_early_week_dates(today, weeks_ahead=1),
            'dau tuan sau': lambda: self._get_early_week_dates(today, weeks_ahead=1),
        }
        
        for pattern, date_func in complex_patterns.items():
            if pattern in query_lower:
                dates = date_func()
                logger.info(f"✅ COMPLEX PATTERN MATCH: '{pattern}' -> {dates}")
                return {k: v for k, v in schedule.items() if k in dates}
        
        # PRIORITY 2: Number + time unit patterns
        number_pattern = re.search(r'(\d+)\s*tuần\s*(tới|sau|tiếp theo)', query_lower)
        if number_pattern:
            num_weeks = int(number_pattern.group(1))
            dates = self._get_week_dates(today, weeks_ahead=num_weeks)
            logger.info(f"✅ NUMBER PATTERN: {num_weeks} tuần -> {dates}")
            return {k: v for k, v in schedule.items() if k in dates}
        
        # PRIORITY 3: Basic time keywords (medium priority)
        basic_time_patterns = {
            'tuần này': lambda: self._get_week_dates(today, weeks_ahead=0),
            'tuan nay': lambda: self._get_week_dates(today, weeks_ahead=0),
            'this week': lambda: self._get_week_dates(today, weeks_ahead=0),
            'tuần tới': lambda: self._get_week_dates(today, weeks_ahead=1),
            'tuan toi': lambda: self._get_week_dates(today, weeks_ahead=1),
            'tuần sau': lambda: self._get_week_dates(today, weeks_ahead=1),
            'tuan sau': lambda: self._get_week_dates(today, weeks_ahead=1),
            'next week': lambda: self._get_week_dates(today, weeks_ahead=1),
            'hôm nay': lambda: [today.strftime('%d-%m-%Y')],
            'hom nay': lambda: [today.strftime('%d-%m-%Y')],
            'today': lambda: [today.strftime('%d-%m-%Y')],
            'ngày mai': lambda: [(today + timedelta(days=1)).strftime('%d-%m-%Y')],
            'ngay mai': lambda: [(today + timedelta(days=1)).strftime('%d-%m-%Y')],
            'tomorrow': lambda: [(today + timedelta(days=1)).strftime('%d-%m-%Y')],
        }
        
        for pattern, date_func in basic_time_patterns.items():
            if pattern in query_lower:
                dates = date_func()
                logger.info(f"✅ BASIC TIME PATTERN: '{pattern}' -> {dates}")
                return {k: v for k, v in schedule.items() if k in dates}
        
        # PRIORITY 4: Weekday patterns
        weekday_patterns = {
            'thứ 2': 0, 'thu 2': 0, 'thứ hai': 0, 'thu hai': 0,
            'thứ 3': 1, 'thu 3': 1, 'thứ ba': 1, 'thu ba': 1,
            'thứ 4': 2, 'thu 4': 2, 'thứ tư': 2, 'thu tu': 2,
            'thứ 5': 3, 'thu 5': 3, 'thứ năm': 3, 'thu nam': 3,
            'thứ 6': 4, 'thu 6': 4, 'thứ sáu': 4, 'thu sau': 4,
            'thứ 7': 5, 'thu 7': 5, 'thứ bảy': 5, 'thu bay': 5,
            'chủ nhật': 6, 'chu nhat': 6, 'sunday': 6
        }
        
        for weekday_name, target_weekday in weekday_patterns.items():
            # Sử dụng regex để đảm bảo khớp từ hoàn chỉnh (tránh "thứ" trong "thứ bậc")
            pattern = r'\b' + re.escape(weekday_name) + r'\b'
            
            if re.search(pattern, query_lower):
                # KIỂM TRA NGỮ CẢNH: Chỉ kích hoạt khi có ý định xem lịch rõ ràng
                
                # Ngữ cảnh 1: Có từ bổ nghĩa thời gian (vd: "tuần sau", "tới")
                has_time_modifier = any(mod in query_lower for mod in ['tuần này', 'tuần sau', 'tuần tới', 'tới', 'nay', 'sau'])
                
                # Ngữ cảnh 2: Có từ khóa về lịch trình (vd: "xem lịch", "tkb")
                has_schedule_keyword = any(kw in query_lower for kw in ['lịch', 'tkb', 'thời khóa biểu', 'xem lịch', 'có lịch'])

                # Ngữ cảnh 3 (QUAN TRỌNG): Loại trừ các trường hợp là số thứ tự
                is_ordinal_context = any(kw in query_lower for kw in ['lần', 'vi phạm', 'hạng', 'điều'])

                # Chỉ khi có ngữ cảnh (1 hoặc 2) VÀ không phải là ngữ cảnh số thứ tự (3) thì mới xử lý
                if (has_time_modifier or has_schedule_keyword) and not is_ordinal_context:
                    target_date = self._get_next_weekday(today, target_weekday)
                    target_date_str = target_date.strftime('%d-%m-%Y')
                    logger.info(f"✅ CONTEXTUAL WEEKDAY PATTERN: '{weekday_name}' with context -> {target_date_str}")
                    return {k: v for k, v in schedule.items() if k == target_date_str}
                else:
                    # Ghi log lý do bỏ qua để dễ gỡ lỗi sau này
                    logger.info(f"ℹ️ SKIPPING WEEKDAY PATTERN for '{weekday_name}': "
                                f"has_time_modifier={has_time_modifier}, "
                                f"has_schedule_keyword={has_schedule_keyword}, "
                                f"is_ordinal_context={is_ordinal_context}")
        
        # PRIORITY 5: Specific date patterns (lowest priority)
        # Chỉ xử lý khi không có từ khóa thời gian chung trong câu ngắn
        word_count = len(query_context.split())
        has_general_keywords = any(kw in query_lower for kw in [
            'tuần', 'tuan', 'week', 'hôm nay', 'hom nay', 'ngày mai', 'ngay mai'
        ])
        
        date_pattern = re.search(r'(\d{1,2})[/-](\d{1,2})', query_lower)
        if date_pattern and not (word_count <= 10 and has_general_keywords):
            day = int(date_pattern.group(1))
            month = int(date_pattern.group(2))
            year = today.year
            try:
                specific_date_str = datetime(year, month, day).strftime('%d-%m-%Y')
                logger.info(f"✅ SPECIFIC DATE: {day}/{month} -> {specific_date_str}")
                return {k: v for k, v in schedule.items() if k == specific_date_str}
            except ValueError:
                logger.warning(f"⚠️ Invalid date: {day}/{month}")
        
        # Nếu không có pattern nào khớp và câu hỏi có liên quan đến lịch/TKB
        # → Trả về tuần hiện tại theo mặc định (KHÔNG trả về toàn bộ)
        schedule_keywords = [
            'lịch', 'thai khoa bieu', 'thời khóa biểu', 'tkb',
            'giảng dạy', 'dạy', 'giảng',
        ]
        if any(kw in query_lower for kw in schedule_keywords):
            default_dates = self._get_week_dates(today, weeks_ahead=0)
            logger.info(f"ℹ️ DEFAULT TO CURRENT WEEK (no time pattern): {default_dates[0]} – {default_dates[-1]}")
            result = {k: v for k, v in schedule.items() if k in default_dates}
            if result:
                return result
            # Nếu tuần này không có lịch, trả về tuần sau
            next_week_dates = self._get_week_dates(today, weeks_ahead=1)
            logger.info(f"ℹ️ Current week empty, falling back to next week: {next_week_dates[0]}")
            return {k: v for k, v in schedule.items() if k in next_week_dates}

        logger.info(f"✅ NO PATTERN MATCHED -> returning full schedule")
        return schedule

    def _get_week_dates(self, today: datetime, weeks_ahead: int) -> List[str]:
        """Helper: Get dates for a specific week"""
        if weeks_ahead == 0:
            # Current week: Monday to Sunday
            start_of_week = today - timedelta(days=today.weekday())
        else:
            # Future weeks: start from Monday of target week
            start_of_week = today + timedelta(days=(7 * weeks_ahead - today.weekday()))
        
        return [(start_of_week + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(7)]

    def _get_weekend_dates(self, today: datetime, current_week: bool = True) -> List[str]:
        """Helper: Get weekend dates (Saturday + Sunday)"""
        if current_week:
            days_until_saturday = (5 - today.weekday()) % 7
            saturday = today + timedelta(days=days_until_saturday)
        else:
            saturday = today + timedelta(days=(12 - today.weekday()))
        
        sunday = saturday + timedelta(days=1)
        return [saturday.strftime('%d-%m-%Y'), sunday.strftime('%d-%m-%Y')]

    def _get_early_week_dates(self, today: datetime, weeks_ahead: int) -> List[str]:
        """Helper: Get early week dates (Mon, Tue, Wed)"""
        start_of_target_week = today + timedelta(days=(7 * weeks_ahead - today.weekday()))
        return [(start_of_target_week + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(3)]

    def _get_next_weekday(self, today: datetime, target_weekday: int) -> datetime:
        """Helper: Get next occurrence of a specific weekday"""
        days_ahead = target_weekday - today.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
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
                    'enhanced_query_context_filtering',  # ✅ UPDATED
                    'response_caching',
                    'error_handling',
                    'complex_time_pattern_processing',   # ✅ NEW
                    'priority_based_time_filtering'      # ✅ NEW
                ]
            }
        }

# Singleton instance
external_api_service = ExternalAPIService()