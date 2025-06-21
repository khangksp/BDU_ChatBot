"""
Google Drive Integration Service - BDU Chatbot
Cache thành CSV thay vì PKL để dễ debug
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from django.conf import settings

logger = logging.getLogger(__name__)

class GoogleDriveService:
    def __init__(self):
        self.service = None
        
        # ✅ ĐỌC CONFIG TỪ SETTINGS
        drive_config = getattr(settings, 'GOOGLE_DRIVE', {})
        self.folder_id = drive_config.get('FOLDER_ID', '1TCozZMg3kFaRXtyRWsrBqwr9T8DbFnq3')
        self.csv_filename = drive_config.get('CSV_FILENAME', 'QA.csv')
        self.cache_timeout = drive_config.get('CACHE_TIMEOUT', 300)
        
        # ✅ CACHE THÀNH CSV THAY VÌ PKL
        self.cache_file = Path(settings.BASE_DIR) / 'data' / 'drive_cache.csv'
        self.cache_meta_file = Path(settings.BASE_DIR) / 'data' / 'drive_cache_meta.json'
        self.local_fallback_path = Path(settings.BASE_DIR) / 'data' / 'QA.csv'
        
        # Memory cache
        self.cached_data = None
        self.cache_timestamp = 0
        
        # Ensure data directory exists
        self.cache_file.parent.mkdir(exist_ok=True)
        
        logger.info(f"🚀 GoogleDriveService initialized - Folder: {self.folder_id}, Cache: CSV format")
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Drive API"""
        try:
            service_account_file = Path(settings.BASE_DIR) / 'thinking-armor-463404-n1-627b306232a8.json'
            
            if not service_account_file.exists():
                logger.error(f"❌ Service account file not found: {service_account_file}")
                return False
            
            scopes = ['https://www.googleapis.com/auth/drive.readonly']
            credentials = Credentials.from_service_account_file(str(service_account_file), scopes=scopes)
            self.service = build('drive', 'v3', credentials=credentials)
            
            logger.info("✅ Google Drive authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Google Drive authentication failed: {str(e)}")
            return False

    def _find_csv_file(self):
        """Find CSV file in the specified folder"""
        try:
            query = f"name='{self.csv_filename}' and parents in '{self.folder_id}' and trashed=false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, modifiedTime, size)"
            ).execute()
            
            files = results.get('files', [])
            
            if files:
                file_info = files[0]
                logger.info(f"📁 Found file: {file_info['name']} (ID: {file_info['id']})")
                return file_info
            else:
                logger.warning(f"⚠️ File {self.csv_filename} not found in folder {self.folder_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error finding CSV file: {str(e)}")
            return None

    def _download_csv_content(self, file_id):
        """Download CSV content from Google Drive"""
        try:
            # Get file content
            file_content = self.service.files().get_media(fileId=file_id).execute()
            csv_content = file_content.decode('utf-8')
            
            logger.info(f"📥 Downloaded CSV content ({len(csv_content)} chars)")
            return csv_content
            
        except Exception as e:
            logger.error(f"❌ Error downloading CSV: {str(e)}")
            return None

    def _parse_csv_content(self, csv_content):
        """Parse CSV content to structured data"""
        try:
            from io import StringIO
            
            # Parse CSV
            df = pd.read_csv(StringIO(csv_content))
            
            # Validate required columns
            if 'question' not in df.columns or 'answer' not in df.columns:
                logger.error("❌ CSV missing required columns: 'question', 'answer'")
                return []
            
            # Clean and process data
            df = df.fillna('')
            df['question'] = df['question'].astype(str).str.strip()
            df['answer'] = df['answer'].astype(str).str.strip()
            
            # Add category if not exists
            if 'category' not in df.columns:
                df['category'] = 'Giảng viên'
            
            # Convert to list of dicts
            records = df.to_dict('records')
            
            logger.info(f"✅ Parsed {len(records)} records from CSV")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error parsing CSV: {str(e)}")
            return []

    def _save_cache_csv(self, data):
        """✅ Save cache as CSV file instead of PKL"""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Save as CSV
            df.to_csv(self.cache_file, index=False, encoding='utf-8')
            
            # Save metadata separately
            metadata = {
                'timestamp': time.time(),
                'data_count': len(data),
                'created_at': datetime.now().isoformat(),
                'source': 'google_drive',
                'folder_id': self.folder_id,
                'csv_filename': self.csv_filename
            }
            
            with open(self.cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Cache saved as CSV: {self.cache_file} ({len(data)} records)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving CSV cache: {str(e)}")
            return False

    def _load_cache_csv(self):
        """✅ Load cache from CSV file"""
        try:
            # Check if cache files exist
            if not self.cache_file.exists() or not self.cache_meta_file.exists():
                logger.info("📂 No CSV cache files found")
                return False
            
            # Load metadata
            with open(self.cache_meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            cache_timestamp = metadata.get('timestamp', 0)
            cache_age = time.time() - cache_timestamp
            
            # Check if cache is still valid
            if cache_age > self.cache_timeout:
                logger.info(f"⏰ CSV cache expired ({cache_age:.1f}s > {self.cache_timeout}s)")
                return False
            
            # Load CSV data
            df = pd.read_csv(self.cache_file, encoding='utf-8')
            data = df.to_dict('records')
            
            # Update memory cache
            self.cached_data = data
            self.cache_timestamp = cache_timestamp
            
            logger.info(f"📂 Loaded {len(data)} records from CSV cache (age: {cache_age:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV cache: {str(e)}")
            return False

    def _is_cache_valid(self):
        """Check if memory cache is still valid"""
        if not self.cached_data:
            return False
        
        cache_age = time.time() - self.cache_timestamp
        return cache_age <= self.cache_timeout

    def _download_and_parse(self):
        """Download and parse CSV from Google Drive"""
        try:
            file_info = self._find_csv_file()
            if not file_info:
                return []
            
            csv_content = self._download_csv_content(file_info['id'])
            if not csv_content:
                return []
            
            parsed_data = self._parse_csv_content(csv_content)
            return parsed_data
            
        except Exception as e:
            logger.error(f"❌ Error downloading and parsing: {str(e)}")
            return []

    def get_csv_data(self, force_refresh=False):
        """
        Get CSV data with intelligent caching
        Returns list of dicts with question, answer, category
        """
        try:
            # 1. Check memory cache first (fastest)
            if not force_refresh and self._is_cache_valid():
                logger.info(f"⚡ Using memory cache ({len(self.cached_data)} records)")
                return self.cached_data
            
            # 2. Try to load from CSV cache file
            if not force_refresh and self._load_cache_csv():
                return self.cached_data
            
            # 3. Fetch from Google Drive
            logger.info("🔄 Fetching fresh data from Google Drive...")
            parsed_data = self._download_and_parse()
            
            if parsed_data:
                # Update memory cache
                self.cached_data = parsed_data
                self.cache_timestamp = time.time()
                
                # ✅ Save to CSV cache instead of PKL
                self._save_cache_csv(parsed_data)
                
                logger.info(f"✅ Successfully loaded {len(parsed_data)} records from Drive")
                return parsed_data
            else:
                logger.warning("⚠️ No data from Drive, trying fallback...")
                return self._load_fallback_csv()
                
        except Exception as e:
            logger.error(f"❌ Error in get_csv_data: {str(e)}")
            return self._load_fallback_csv()

    def _load_fallback_csv(self):
        """Load fallback CSV from local file"""
        try:
            if self.local_fallback_path.exists():
                df = pd.read_csv(self.local_fallback_path, encoding='utf-8')
                
                if 'question' in df.columns and 'answer' in df.columns:
                    df = df.fillna('')
                    if 'category' not in df.columns:
                        df['category'] = 'Giảng viên'
                    
                    data = df.to_dict('records')
                    logger.info(f"🔄 Loaded {len(data)} records from fallback CSV")
                    return data
                    
            logger.warning("⚠️ No fallback data available")
            return []
            
        except Exception as e:
            logger.error(f"❌ Error loading fallback CSV: {str(e)}")
            return []

    def force_refresh(self):
        """Force refresh data from Google Drive"""
        logger.info("🔄 Force refresh requested")
        return self.get_csv_data(force_refresh=True)

    def get_system_status(self):
        """Get system status for debugging"""
        cache_age = time.time() - self.cache_timestamp if self.cache_timestamp else 0
        
        # Check CSV cache file status
        csv_cache_exists = self.cache_file.exists()
        csv_cache_size = self.cache_file.stat().st_size if csv_cache_exists else 0
        
        meta_cache_exists = self.cache_meta_file.exists()
        
        return {
            'drive_authenticated': bool(self.service),
            'folder_id': self.folder_id,
            'csv_filename': self.csv_filename,
            'cache_timeout': self.cache_timeout,
            'cache_valid': self._is_cache_valid(),
            'cache_age_seconds': cache_age,
            'last_data_count': len(self.cached_data) if self.cached_data else 0,
            'cache_format': 'CSV',  # ✅ MỚI: Hiển thị format cache
            'csv_cache_file': str(self.cache_file),
            'csv_cache_exists': csv_cache_exists,
            'csv_cache_size_bytes': csv_cache_size,
            'meta_cache_exists': meta_cache_exists,
            'fallback_available': self.local_fallback_path.exists()
        }

    def clear_cache(self):
        """Clear all cache (memory + files)"""
        try:
            # Clear memory
            self.cached_data = None
            self.cache_timestamp = 0
            
            # Remove cache files
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"🗑️ Removed CSV cache file: {self.cache_file}")
            
            if self.cache_meta_file.exists():
                self.cache_meta_file.unlink()
                logger.info(f"🗑️ Removed meta cache file: {self.cache_meta_file}")
            
            logger.info("✅ All cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")
            return False

# ✅ Global instance
google_drive_service = GoogleDriveService()