"""
Django signals for QA Management
Handles automatic chatbot reload when Q&A data changes
"""

from django.db.models.signals import post_save, post_delete, pre_delete
from django.dispatch import receiver
from django.utils import timezone
from django.conf import settings
import logging
import threading

logger = logging.getLogger(__name__)

@receiver(post_save, sender='qa_management.QAEntry')
def qa_entry_saved(sender, instance, created, **kwargs):
    """Handle QA Entry save - trigger chatbot reload"""
    try:
        # Import here to avoid circular imports
        from ai_models.services import chatbot_ai
        
        action = "created" if created else "updated"
        logger.info(f"🔄 QA Entry {action}: {instance.stt}")
        
        # Mark as pending sync (only for updates, not new entries)
        if not created and not getattr(instance, '_syncing', False):
            instance.sync_status = 'pending'
            # Use update to avoid triggering signal again
            type(instance).objects.filter(pk=instance.pk).update(sync_status='pending')
        
        # Get auto-reload setting
        chatbot_integration = getattr(settings, 'CHATBOT_INTEGRATION', {})
        auto_rebuild = chatbot_integration.get('AUTO_REBUILD_INDEX', True)
        
        if auto_rebuild:
            # Trigger chatbot reload in background thread to avoid blocking
            def reload_chatbot():
                try:
                    chatbot_ai.sbert_retriever.load_knowledge_base()
                    logger.info("✅ Chatbot knowledge base reloaded")
                except Exception as e:
                    logger.error(f"❌ Failed to reload chatbot: {str(e)}")
            
            # Run in background thread
            reload_thread = threading.Thread(target=reload_chatbot)
            reload_thread.daemon = True
            reload_thread.start()
            
    except Exception as e:
        logger.error(f"❌ QA Entry save signal error: {str(e)}")

@receiver(post_delete, sender='qa_management.QAEntry')
def qa_entry_deleted(sender, instance, **kwargs):
    """Handle QA Entry deletion - trigger chatbot reload"""
    try:
        from ai_models.services import chatbot_ai
        
        logger.info(f"🗑️ QA Entry deleted: {instance.stt}")
        
        # Get auto-reload setting
        chatbot_integration = getattr(settings, 'CHATBOT_INTEGRATION', {})
        auto_rebuild = chatbot_integration.get('AUTO_REBUILD_INDEX', True)
        
        if auto_rebuild:
            # Trigger chatbot reload in background
            def reload_chatbot():
                try:
                    chatbot_ai.sbert_retriever.load_knowledge_base()
                    logger.info("✅ Chatbot knowledge base reloaded after deletion")
                except Exception as e:
                    logger.error(f"❌ Failed to reload chatbot after deletion: {str(e)}")
            
            reload_thread = threading.Thread(target=reload_chatbot)
            reload_thread.daemon = True
            reload_thread.start()
        
    except Exception as e:
        logger.error(f"❌ QA Entry delete signal error: {str(e)}")

# Bulk operations signals
@receiver(pre_delete, sender='qa_management.QAEntry')
def qa_entry_pre_delete(sender, instance, **kwargs):
    """Handle before deletion - log for audit"""
    logger.info(f"📝 Preparing to delete QA Entry: {instance.stt} - '{instance.question[:50]}...'")

# Auto-sync to Google Drive (optional)
@receiver(post_save, sender='qa_management.QAEntry')
def auto_sync_to_drive(sender, instance, created, **kwargs):
    """Auto-sync to Drive if enabled in settings - FIXED VERSION"""
    try:
        # Check if auto-sync is enabled
        qa_settings = getattr(settings, 'QA_MANAGEMENT', {})
        if not qa_settings.get('AUTO_SYNC_ON_SAVE', False):
            return
        
        # Don't auto-sync if already syncing or during bulk operations
        if getattr(instance, '_syncing', False) or getattr(instance, '_bulk_operation', False):
            return
            
        # Don't sync immediately after creation (wait for user to finish editing)
        if created:
            logger.info(f"⏳ New QA Entry created: {instance.stt} - auto-sync will happen on next edit")
            return
        
        # ✅ FIX: Don't auto-sync during debug tests
        if instance.stt.startswith('DEBUG_TEST_') or instance.stt.startswith('QUICK_TEST_'):
            logger.info(f"🔍 Skipping auto-sync for debug entry: {instance.stt}")
            return
        
        # Import here to avoid circular imports
        from .services import drive_service
        
        logger.info(f"🔄 Auto-sync triggered for: {instance.stt}")
        
        # ✅ FIX: Use sync_single_entry instead of export_all_to_drive
        def sync_to_drive():
            try:
                # Mark to prevent recursive sync
                instance._syncing = True
                
                # ✅ IMPORTANT: Use single entry sync to avoid overwriting all data
                result = drive_service.sync_single_entry(instance)
                
                if result:
                    logger.info(f"✅ Auto-sync successful for entry: {instance.stt}")
                else:
                    logger.warning(f"⚠️ Auto-sync failed for entry: {instance.stt}")
                    
            except Exception as e:
                logger.error(f"❌ Auto-sync error for {instance.stt}: {str(e)}")
            finally:
                # Remove syncing flag
                if hasattr(instance, '_syncing'):
                    delattr(instance, '_syncing')
        
        # Run sync in background thread
        sync_thread = threading.Thread(target=sync_to_drive)
        sync_thread.daemon = True
        sync_thread.start()
        
    except Exception as e:
        logger.error(f"❌ Auto-sync signal error: {str(e)}")

# Cache invalidation signal
@receiver([post_save, post_delete], sender='qa_management.QAEntry')
def invalidate_chatbot_cache(sender, **kwargs):
    """Invalidate chatbot cache when Q&A data changes"""
    try:
        # Get cache invalidation setting
        chatbot_integration = getattr(settings, 'CHATBOT_INTEGRATION', {})
        cache_invalidation = chatbot_integration.get('CACHE_INVALIDATION', True)
        
        if cache_invalidation:
            from ai_models.services import chatbot_ai
            
            # Clear any cached data
            if hasattr(chatbot_ai.sbert_retriever, 'cached_data'):
                chatbot_ai.sbert_retriever.cached_data = None
                chatbot_ai.sbert_retriever.cache_timestamp = 0
            
            # Clear Google Drive cache
            # FIX: Update import path or remove if not available
            try:
                from ai_models.services import drive_service
                drive_service.clear_cache()
            except ImportError:
                logger.warning("Google Drive cache service not available for cache invalidation")
            
            logger.info("🗑️ Chatbot cache invalidated")
            
    except Exception as e:
        logger.error(f"❌ Cache invalidation error: {str(e)}")

# Notification signal (for future use)
@receiver(post_save, sender='qa_management.QAEntry')
def send_qa_update_notification(sender, instance, created, **kwargs):
    """Send notifications when Q&A data changes (future feature)"""
    try:
        # Check if notifications are enabled
        chatbot_integration = getattr(settings, 'CHATBOT_INTEGRATION', {})
        notifications_enabled = chatbot_integration.get('NOTIFICATION_ENABLED', False)
        
        if not notifications_enabled:
            return
        
        # Future implementation:
        # - Send email notifications
        # - Call webhook URLs
        # - Send Slack/Discord notifications
        # - Update dashboard
        
        action = "created" if created else "updated"
        logger.info(f"📢 QA Entry {action}: {instance.stt} - notifications would be sent here")
        
    except Exception as e:
        logger.error(f"❌ Notification error: {str(e)}")

# Audit logging signal
@receiver([post_save, post_delete], sender='qa_management.QAEntry')
def audit_qa_changes(sender, instance, **kwargs):
    """Log Q&A changes for audit purposes"""
    try:
        # Check if audit logging is enabled
        qa_settings = getattr(settings, 'QA_MANAGEMENT', {})
        audit_enabled = qa_settings.get('AUDIT_LOG_ENABLED', True)
        
        if audit_enabled:
            # In a full implementation, you would save to an audit log table
            # For now, just log to the standard logger
            
            if 'created' in kwargs:
                action = "CREATED" if kwargs['created'] else "UPDATED"
            else:
                action = "DELETED"
            
            logger.info(f"📋 AUDIT: {action} QA Entry {instance.stt} - '{instance.question[:30]}...'")
            
    except Exception as e:
        logger.error(f"❌ Audit logging error: {str(e)}")