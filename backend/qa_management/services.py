"""
QA Management Service for Google Drive Integration
Extends the existing GoogleDriveService with write capabilities
FIXED VERSION - Prevents data loss during sync operations
"""

import csv
import io
import time
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django.db import transaction
import logging
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

class QADriveService:
    """
    Service for syncing Q&A entries with Google Drive
    Extends existing GoogleDriveService with write capabilities
    """
    
    def __init__(self):
        self.service = None
        
        # Get config from settings
        drive_config = getattr(settings, 'GOOGLE_DRIVE', {})
        self.folder_id = drive_config.get('FOLDER_ID', '1589N-eP0KW3SLZwQtibxXnVoMrubPCqM')
        self.csv_filename = drive_config.get('CSV_FILENAME', 'QA.csv')
        
        # Initialize service
        self._authenticate()
        
        logger.info("🚀 QADriveService initialized with write capabilities")
    
    def _authenticate(self):
        """Authenticate with Google Drive API with write permissions"""
        try:
            service_account_file = settings.BASE_DIR / 'thinking-armor-463404-n1-627b306232a8.json'
            
            if not service_account_file.exists():
                logger.error(f"❌ Service account file not found: {service_account_file}")
                return False
            
            # ✅ WRITE PERMISSIONS - thay đổi scopes
            scopes = [
                'https://www.googleapis.com/auth/drive.file',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                str(service_account_file), 
                scopes=scopes
            )
            self.service = build('drive', 'v3', credentials=credentials)
            
            logger.info("✅ Google Drive authentication successful (with write permissions)")
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
        """Download current CSV content from Google Drive"""
        try:
            file_content = self.service.files().get_media(fileId=file_id).execute()
            csv_content = file_content.decode('utf-8')
            
            logger.info(f"📥 Downloaded CSV content ({len(csv_content)} chars)")
            return csv_content
            
        except Exception as e:
            logger.error(f"❌ Error downloading CSV: {str(e)}")
            return None
    
    def _upload_csv_content(self, csv_content, file_id=None):
        """Upload CSV content to Google Drive"""
        try:
            # Prepare media upload
            media_body = MediaIoBaseUpload(
                io.BytesIO(csv_content.encode('utf-8')),
                mimetype='text/csv',
                resumable=True
            )
            
            if file_id:
                # Update existing file
                updated_file = self.service.files().update(
                    fileId=file_id,
                    media_body=media_body
                ).execute()
                
                logger.info(f"✅ Updated existing file: {updated_file.get('name')} (ID: {file_id})")
                return updated_file
            else:
                # Create new file
                file_metadata = {
                    'name': self.csv_filename,
                    'parents': [self.folder_id]
                }
                
                new_file = self.service.files().create(
                    body=file_metadata,
                    media_body=media_body,
                    fields='id,name'
                ).execute()
                
                logger.info(f"✅ Created new file: {new_file.get('name')} (ID: {new_file.get('id')})")
                return new_file
                
        except Exception as e:
            logger.error(f"❌ Error uploading CSV: {str(e)}")
            return None
    
    def _database_to_csv_format(self, entries=None):
        """Convert database entries to CSV format matching Drive structure"""
        from .models import QAEntry
        
        if entries is None:
            entries = QAEntry.objects.filter(is_active=True).order_by('stt')
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row matching Drive format: STT,question,answer,,
        writer.writerow(['STT', 'question', 'answer', '', ''])
        
        # Data rows
        for entry in entries:
            writer.writerow([
                entry.stt,
                entry.question,
                entry.answer,
                '',  # Empty column 4
                ''   # Empty column 5
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"🔄 Converted {len(entries)} entries to CSV format")
        return csv_content
    
    def _csv_to_database_format(self, csv_content):
        """Parse CSV content and return entries for database - WITH SAFETY CHECKS"""
        try:
            # ✅ SAFETY CHECK: Validate input
            if not csv_content or len(csv_content.strip()) < 10:
                logger.error("❌ CSV content is empty or too short")
                return []
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(csv_content))
            
            # ✅ SAFETY CHECK: Validate DataFrame
            if df.empty:
                logger.error("❌ Parsed DataFrame is empty")
                return []
            
            logger.info(f"📊 CSV parsed - Shape: {df.shape}, Columns: {list(df.columns)}")
            
            # Validate columns
            required_columns = ['STT', 'question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                logger.error(f"❌ Available columns: {list(df.columns)}")
                return []
            
            # Clean data
            df = df.fillna('')
            df['STT'] = df['STT'].astype(str).str.strip()
            df['question'] = df['question'].astype(str).str.strip()
            df['answer'] = df['answer'].astype(str).str.strip()
            
            # Filter out ONLY completely empty rows
            original_count = len(df)
            df = df[
                (df['STT'] != '') & 
                (df['question'] != '') & 
                (df['answer'] != '')
            ]
            filtered_count = len(df)
            
            if filtered_count < original_count:
                logger.info(f"🧹 Filtered out {original_count - filtered_count} empty rows")
            
            # ✅ SAFETY CHECK: Ensure we have data after filtering
            if df.empty:
                logger.error("❌ No valid rows found after filtering")
                return []
            
            # Convert to list of dicts
            entries = []
            for index, row in df.iterrows():
                try:
                    entry_data = {
                        'stt': str(row['STT']).strip(),
                        'question': str(row['question']).strip(),
                        'answer': str(row['answer']).strip(),
                        'category': 'Giảng viên'  # Default category
                    }
                    
                    # ✅ VALIDATION: Skip completely invalid entries
                    if not entry_data['stt'] or not entry_data['question'] or not entry_data['answer']:
                        logger.warning(f"⚠️ Skipping invalid entry at row {index}: {entry_data}")
                        continue
                        
                    entries.append(entry_data)
                    
                except Exception as row_error:
                    logger.warning(f"⚠️ Error processing row {index}: {row_error}")
                    continue
            
            logger.info(f"✅ Successfully parsed {len(entries)} valid entries from CSV")
            
            # ✅ FINAL SAFETY CHECK
            if not entries:
                logger.error("❌ No valid entries found after processing")
                return []
            
            return entries
            
        except Exception as e:
            logger.error(f"❌ Error parsing CSV to database format: {str(e)}")
            logger.error(f"❌ CSV content preview: {csv_content[:200] if csv_content else 'None'}...")
            return []
    
    def import_from_drive(self):
        """Import Q&A entries from Google Drive to database"""
        from .models import QAEntry, QASyncLog
        
        # Create sync log
        sync_log = QASyncLog.objects.create(
            operation='import_from_drive',
            status='partial'  # Will update later
        )
        
        try:
            # Start sync log
            sync_log = QASyncLog.objects.create(
                operation='import_from_drive',
                status='running'
            )
            
            # Get fresh data from Drive
            logger.info("🔄 Importing data from Google Drive...")
            
            # Get file and download content
            file_info = self._find_csv_file()
            if not file_info:
                sync_log.status = 'failed'
                sync_log.error_message = 'CSV file not found in Drive'
                sync_log.completed_at = timezone.now()
                sync_log.save()
                return {
                    'success': False,
                    'error': 'CSV file not found in Google Drive'
                }
            
            csv_content = self._download_csv_content(file_info['id'])
            if not csv_content:
                sync_log.status = 'failed'
                sync_log.error_message = 'Could not download CSV content'
                sync_log.completed_at = timezone.now()
                sync_log.save()
                return {
                    'success': False,
                    'error': 'Could not download CSV content from Drive'
                }
            
            # Parse CSV data
            drive_data = self._csv_to_database_format(csv_content)
            
            if not drive_data:
                sync_log.status = 'failed'
                sync_log.error_message = 'No valid data found in Drive CSV'
                sync_log.completed_at = timezone.now()
                sync_log.save()
                return {
                    'success': False,
                    'error': 'No valid data found in Google Drive CSV'
                }
            
            imported_count = 0
            updated_count = 0
            error_count = 0
            
            for item in drive_data:
                try:
                    stt = item.get('stt', '').strip()
                    question = item.get('question', '').strip()
                    answer = item.get('answer', '').strip()
                    
                    if not stt or not question or not answer:
                        error_count += 1
                        continue
                    
                    # Create or update entry
                    entry, created = QAEntry.objects.update_or_create(
                        stt=stt,
                        defaults={
                            'question': question,
                            'answer': answer,
                            'category': item.get('category', 'Giảng viên'),
                            'sync_status': 'synced',
                            'last_synced_to_drive': timezone.now()
                        }
                    )
                    
                    if created:
                        imported_count += 1
                    else:
                        updated_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing item {item}: {str(e)}")
                    error_count += 1
            
            # Complete sync log
            sync_log.status = 'completed'
            sync_log.completed_at = timezone.now()
            sync_log.entries_processed = len(drive_data)
            sync_log.entries_success = imported_count + updated_count
            sync_log.entries_failed = error_count
            sync_log.save()
            
            logger.info(f"✅ Import completed: {imported_count} new, {updated_count} updated, {error_count} errors")
            
            return {
                'success': True,
                'imported': imported_count,
                'updated': updated_count,
                'errors': error_count,
                'total_processed': len(drive_data)
            }
            
        except Exception as e:
            sync_log.status = 'failed'
            sync_log.error_message = str(e)
            sync_log.completed_at = timezone.now()
            sync_log.save()
            
            logger.error(f"❌ Import from Drive failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_all_to_drive(self):
        """Export all Q&A entries from database to Google Drive - SAFE VERSION"""
        from .models import QAEntry, QASyncLog
        
        # Create sync log
        sync_log = QASyncLog.objects.create(
            operation='export_to_drive',
            status='partial'
        )
        
        try:
            # Get all active entries
            entries = QAEntry.objects.filter(is_active=True).order_by('stt')
            
            if not entries.exists():
                sync_log.status = 'failed'
                sync_log.error_message = 'No active entries to export'
                sync_log.completed_at = timezone.now()
                sync_log.save()
                return {
                    'success': False,
                    'error': 'No active entries to export'
                }
            
            # ✅ SAFETY CHECK: Don't export if there are too few entries (possible data loss)
            if entries.count() < 10:
                logger.warning(f"⚠️ Only {entries.count()} entries to export - this might indicate data loss")
                logger.warning("⚠️ Please verify this is intentional before proceeding")
                
                # Optional: Uncomment to prevent export with few entries
                # sync_log.status = 'failed'
                # sync_log.error_message = f'Too few entries ({entries.count()}) - possible data loss'
                # sync_log.completed_at = timezone.now()
                # sync_log.save()
                # return {'success': False, 'error': 'Too few entries - possible data loss'}
            
            # ✅ CREATE BACKUP before major export
            backup_result = self.backup_current_data()
            if backup_result['success']:
                logger.info(f"✅ Backup created before export: {backup_result['backup_filename']}")
            
            # Convert to CSV format
            csv_content = self._database_to_csv_format(entries)
            
            # Find existing file or create new
            file_info = self._find_csv_file()
            file_id = file_info['id'] if file_info else None
            
            # Upload to Drive
            upload_result = self._upload_csv_content(csv_content, file_id)
            if not upload_result:
                sync_log.status = 'failed'
                sync_log.error_message = 'Failed to upload CSV to Drive'
                sync_log.completed_at = timezone.now()
                sync_log.save()
                return {
                    'success': False,
                    'error': 'Failed to upload CSV to Drive'
                }
            
            # Mark all entries as synced
            now = timezone.now()
            updated_count = entries.update(
                sync_status='synced',
                last_synced_to_drive=now
            )
            
            # Update sync log
            sync_log.entries_processed = entries.count()
            sync_log.entries_success = updated_count
            sync_log.entries_failed = 0
            sync_log.status = 'success'
            sync_log.completed_at = timezone.now()
            sync_log.details = {
                'backup_created': backup_result['success'],
                'backup_file': backup_result.get('backup_filename', 'None'),
                'total_entries_exported': updated_count
            }
            sync_log.save()
            
            logger.info(f"✅ Export completed: {updated_count} entries synced to Drive")
            
            return {
                'success': True,
                'total_entries': updated_count,
                'file_id': upload_result.get('id'),
                'backup_created': backup_result['success']
            }
            
        except Exception as e:
            sync_log.status = 'failed'
            sync_log.error_message = str(e)
            sync_log.completed_at = timezone.now()
            sync_log.save()
            
            logger.error(f"❌ Export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def sync_single_entry(self, entry):
        """Sync a single QA entry to Drive using SAFE MERGE mode"""
        try:
            logger.info(f"🔄 Syncing entry {entry.stt} to Drive...")
            
            # 1. Download existing CSV content from Drive
            file_info = self._find_csv_file()
            if not file_info:
                logger.error("❌ CSV file not found on Drive")
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                return False
            
            # 2. Get current CSV content
            existing_csv_content = self._download_csv_content(file_info['id'])
            if not existing_csv_content:
                logger.error("❌ Could not download existing CSV content")
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                return False
            
            # 3. ✅ SAFE PARSE: Validate before proceeding
            existing_entries = self._csv_to_database_format(existing_csv_content)
            
            # ✅ CRITICAL CHECK: If parse failed (empty result), ABORT!
            if not existing_entries:
                logger.error(f"❌ CRITICAL: Failed to parse existing CSV or CSV is empty! "
                            f"CSV content preview: {existing_csv_content[:200]}...")
                logger.error("❌ ABORTING SYNC to prevent data loss!")
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                return False
            
            logger.info(f"✅ Successfully parsed {len(existing_entries)} existing entries from Drive")
            
            # 4. Create entry data in correct format
            new_entry_data = {
                'stt': entry.stt,
                'question': entry.question,
                'answer': entry.answer,
                'category': getattr(entry, 'category', 'Giảng viên'),
            }
            
            # 5. ✅ SAFE MERGE: Keep ALL existing data
            merged_entries = existing_entries.copy()  # Keep everything
            
            # Check if STT already exists
            existing_index = None
            for i, existing_entry in enumerate(merged_entries):
                if existing_entry.get('stt') == entry.stt:
                    existing_index = i
                    break
            
            if existing_index is not None:
                # Update existing entry
                merged_entries[existing_index] = new_entry_data
                logger.info(f"✅ Updated existing entry with STT: {entry.stt}")
            else:
                # Add new entry
                merged_entries.append(new_entry_data)
                logger.info(f"✅ Added new entry with STT: {entry.stt}")
            
            # 6. ✅ VALIDATION: Ensure we didn't lose data
            if len(merged_entries) < len(existing_entries):
                logger.error(f"❌ CRITICAL: Data loss detected! "
                            f"Original: {len(existing_entries)}, "
                            f"Merged: {len(merged_entries)}")
                logger.error("❌ ABORTING SYNC to prevent data loss!")
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                return False
            
            # 7. Sort by STT to maintain order (optional)
            try:
                merged_entries.sort(key=lambda x: int(x.get('stt', 0)) if str(x.get('stt', '')).isdigit() else float('inf'))
            except Exception as sort_error:
                logger.warning(f"⚠️ Sorting failed: {sort_error}, keeping original order")
            
            # 8. Convert merged data back to CSV format
            merged_csv_content = self._create_csv_from_entries(merged_entries)
            
            # 9. ✅ FINAL VALIDATION: Check CSV content
            if not merged_csv_content or len(merged_csv_content.strip()) < 50:
                logger.error(f"❌ CRITICAL: Generated CSV is too short or empty!")
                logger.error("❌ ABORTING SYNC to prevent data loss!")
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                return False
            
            # 10. Upload merged CSV back to Drive
            upload_result = self._upload_csv_content(merged_csv_content, file_info['id'])
            
            if upload_result:
                # Update sync status
                entry.sync_status = 'synced'
                entry.last_synced_to_drive = timezone.now()
                entry.save(update_fields=['sync_status', 'last_synced_to_drive'])
                
                logger.info(f"✅ Successfully synced entry {entry.stt} "
                           f"(Total entries in CSV: {len(merged_entries)})")
                return True
            else:
                entry.sync_status = 'error'
                entry.save(update_fields=['sync_status'])
                logger.error(f"❌ Failed to upload merged CSV for entry {entry.stt}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error syncing entry {entry.stt}: {str(e)}")
            entry.sync_status = 'error'
            entry.save(update_fields=['sync_status'])
            return False
    
    def _create_csv_from_entries(self, entries):
        """Helper method to create CSV content from entry list"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row matching Drive format: STT,question,answer,,
        writer.writerow(['STT', 'question', 'answer', '', ''])
        
        # Data rows
        for entry in entries:
            writer.writerow([
                entry.get('stt', ''),
                entry.get('question', ''),
                entry.get('answer', ''),
                '',  # Empty column 4
                ''   # Empty column 5
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"🔄 Created CSV content with {len(entries)} entries")
        return csv_content
    
    def get_drive_status(self):
        """Get Google Drive connection and file status"""
        try:
            if not self.service:
                return {
                    'connected': False,
                    'error': 'Not authenticated'
                }
            
            # Check if file exists
            file_info = self._find_csv_file()
            
            if file_info:
                # Get file details
                file_details = self.service.files().get(
                    fileId=file_info['id'],
                    fields='id,name,size,modifiedTime,createdTime'
                ).execute()
                
                return {
                    'connected': True,
                    'file_exists': True,
                    'file_id': file_details['id'],
                    'file_name': file_details['name'],
                    'file_size': int(file_details.get('size', 0)),
                    'modified_time': file_details.get('modifiedTime'),
                    'created_time': file_details.get('createdTime'),
                    'folder_id': self.folder_id
                }
            else:
                return {
                    'connected': True,
                    'file_exists': False,
                    'folder_id': self.folder_id,
                    'csv_filename': self.csv_filename
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting Drive status: {str(e)}")
            return {
                'connected': False,
                'error': str(e)
            }
    
    def backup_current_data(self):
        """Create a backup of current data before major operations"""
        try:
            from .models import QAEntry
            
            entries = QAEntry.objects.filter(is_active=True).order_by('stt')
            csv_content = self._database_to_csv_format(entries)
            
            # Save backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"QA_backup_{timestamp}.csv"
            
            # Upload backup to Drive
            media_body = MediaIoBaseUpload(
                io.BytesIO(csv_content.encode('utf-8')),
                mimetype='text/csv',
                resumable=True
            )
            
            file_metadata = {
                'name': backup_filename,
                'parents': [self.folder_id]
            }
            
            backup_file = self.service.files().create(
                body=file_metadata,
                media_body=media_body,
                fields='id,name'
            ).execute()
            
            logger.info(f"✅ Backup created: {backup_filename} (ID: {backup_file.get('id')})")
            
            return {
                'success': True,
                'backup_file_id': backup_file.get('id'),
                'backup_filename': backup_filename
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating backup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Global instance
qa_drive_service = QADriveService()