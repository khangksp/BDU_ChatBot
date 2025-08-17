# debug_speech_service.py
import os
import torch
import logging
from pathlib import Path
import sys

# Thêm thư mục gốc của dự án vào sys.path để có thể import speech_service
# Điều này giả định bạn đặt file debug_speech_service.py trong thư mục `backend`
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Cấu hình logging để hiển thị thông tin chi tiết trên console
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Hàm chuyển đổi bytes sang GB cho dễ đọc
def bytes_to_gb(b):
    return round(b / (1024**3), 2)

def check_gpu_status(stage=""):
    """Kiểm tra và in ra trạng thái bộ nhớ GPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem, total_mem_info = torch.cuda.mem_get_info(0)
        
        logging.info(f"--- Trạng thái GPU {stage} ---")
        logging.info(f"Tên thiết bị: {device_name}")
        logging.info(f"Tổng VRAM: {bytes_to_gb(total_mem)} GB")
        logging.info(f"VRAM còn trống: {bytes_to_gb(free_mem)} GB")
        logging.info(f"VRAM đã sử dụng: {bytes_to_gb(total_mem_info - free_mem)} GB")
        logging.info("----------------------------------")
    else:
        logging.warning("--- Không tìm thấy GPU tương thích CUDA. ---")


if __name__ == "__main__":
    logging.info(" BẮT ĐẦU KỊCH BẢN GỠ LỖI CHO SPEECHTOTEXTSERVICE ")
    logging.info("=" * 50)

    try:
        # Import service SAU KHI đã cấu hình sys.path
        from ai_models.speech_service import SpeechToTextService, WHISPER_AVAILABLE

        if not WHISPER_AVAILABLE:
            logging.error("Lỗi nghiêm trọng: Thư viện faster_whisper chưa được cài đặt.")
            sys.exit(1)

        # 1. Kiểm tra trạng thái GPU ban đầu
        check_gpu_status("TRƯỚC KHI TẢI MODEL")

        # 2. Cố gắng khởi tạo service
        logging.info(">>> Bắt đầu khởi tạo SpeechToTextService (đây là bước có thể gây lỗi)...")
        service = None # Khởi tạo là None
        
        try:
            service = SpeechToTextService()
            
            if service and service.is_available():
                logging.info("✅✅✅ THÀNH CÔNG: SpeechToTextService đã được khởi tạo thành công!")
                logging.info(f"Thông tin service: {service.get_system_status()}")
            else:
                logging.error("❌❌❌ THẤT BẠI: Service đã khởi tạo nhưng không khả dụng (model is None).")
                logging.error(f"Thông tin service: {service.get_system_status() if service else 'Service object is None'}")

        except Exception as e:
            logging.error("❌❌❌ THẤT BẠI: Đã xảy ra lỗi nghiêm trọng trong quá trình khởi tạo service.")
            # logging.exception(e) sẽ in ra đầy đủ traceback của lỗi
            logging.exception(e)
            
            # 3. Kiểm tra lại trạng thái GPU sau khi lỗi
            check_gpu_status("SAU KHI XẢY RA LỖI")

    except ImportError as ie:
        logging.error(f"Lỗi Import: Không thể tìm thấy module. Vui lòng đảm bảo file này nằm trong thư mục 'backend'. Lỗi: {ie}")
    except Exception as general_error:
        logging.error(f"Lỗi không xác định: {general_error}")
    
    logging.info("=" * 50)
    logging.info(" KẾT THÚC KỊCH BẢN GỠ LỖI ")