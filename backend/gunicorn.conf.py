"""
Gunicorn Configuration - BDU ChatBot GV
Tối ưu hóa cho CPU multi-core (không có NVIDIA GPU)
"""
import multiprocessing

# -------------------------------------------------------
# Binding
# -------------------------------------------------------
bind = "0.0.0.0:3019"

# -------------------------------------------------------
# Workers
# ⚠️ Mỗi worker load lại toàn bộ AI model (PhoBERT, SBERT)
# → Dùng ít workers để tránh cạn RAM
# 2 workers × 2 threads = xử lý được 4 concurrent requests
# Khi lên EC2 có nhiều RAM hơn, có thể tăng lên 3-4
# -------------------------------------------------------
workers = 2

# Dùng gthread để mỗi worker có thể xử lý nhiều request
# song song qua threads (tốt hơn sync khi có I/O chờ LLM)
worker_class = "gthread"
threads = 2

# -------------------------------------------------------
# Timeout
# 120s vì LLM (Ollama) đôi khi cần thời gian dài hơn
# -------------------------------------------------------
timeout = 120
graceful_timeout = 30
keepalive = 5

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
accesslog = "-"       # stdout
errorlog = "-"        # stderr
loglevel = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# -------------------------------------------------------
# Performance
# -------------------------------------------------------
preload_app = True    # Nạp app 1 lần, fork ra workers → tiết kiệm RAM
max_requests = 500    # Restart worker sau 500 requests (chống memory leak)
max_requests_jitter = 50
