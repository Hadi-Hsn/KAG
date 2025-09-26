import httpx
import asyncio
import logging
from typing import Optional
from app.config.settings import PUSHOVER_USER, PUSHOVER_TOKEN

# Set up logging
logger = logging.getLogger(__name__)

class NotificationService:
    """
    Service for sending notifications via Pushover when users interact with the system.
    This provides monitoring capabilities for service usage.
    """
    
    @staticmethod
    async def send_pushover_notification(message: str, title: str = "KAG Service Usage") -> bool:
        """Send notification via Pushover when someone uses the service"""
        if not PUSHOVER_USER or not PUSHOVER_TOKEN:
            logger.info("Pushover credentials not configured, skipping notification")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.pushover.net/1/messages.json",
                    data={
                        "token": PUSHOVER_TOKEN,
                        "user": PUSHOVER_USER,
                        "message": message,
                        "title": title,
                        "priority": 0  # Normal priority
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info("Pushover notification sent successfully")
                    return True
                else:
                    logger.warning(f"Pushover notification failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Pushover notification: {str(e)}")
            return False
    
    @staticmethod
    async def notify_file_upload(files_count: int, total_size_mb: float, user_ip: str = "Unknown") -> None:
        """Send notification when files are uploaded"""
        message = f"📄 File Upload Activity\n"
        message += f"Files: {files_count}\n"
        message += f"Total Size: {total_size_mb:.2f} MB\n"
        message += f"IP: {user_ip}"
        
        await NotificationService.send_pushover_notification(
            message, "KAG - File Upload"
        )
    
    @staticmethod
    async def notify_query_request(query: str, user_ip: str = "Unknown", results_count: int = 0) -> None:
        """Send notification when a query is made"""
        # Truncate query if too long
        display_query = query[:100] + "..." if len(query) > 100 else query
        
        message = f"🔍 Query Activity\n"
        message += f"Query: {display_query}\n"
        message += f"Results: {results_count}\n"
        message += f"IP: {user_ip}"
        
        await NotificationService.send_pushover_notification(
            message, "KAG - Query Request"
        )

# Global instance for easy access
notification_service = NotificationService()
