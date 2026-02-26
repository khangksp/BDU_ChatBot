"""
Script debug: kiểm tra CPU/GPU usage và đo hiệu năng chatbot
"""
import requests
import json
import time
import threading
import subprocess

SESSION = "gpu_debug_session_001"
BASE_URL = "http://localhost:8000/api/chat/"

# ──────────────────────────────── Monitor GPU ───────────────────────────────

gpu_samples = []
stop_monitoring = threading.Event()

def monitor_gpu():
    while not stop_monitoring.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            line = res.stdout.strip()
            if line:
                parts = [x.strip() for x in line.split(",")]
                gpu_samples.append({
                    "t": time.time(),
                    "gpu_util": int(parts[0]),
                    "mem_util": int(parts[1]),
                    "mem_used_mib": int(parts[2]),
                    "mem_free_mib": int(parts[3]),
                })
        except Exception as e:
            pass
        time.sleep(0.5)


def print_gpu_report(label, t_start):
    if not gpu_samples:
        print("  [GPU] Không có sample nào (nvidia-smi không hoạt động?)")
        return
    relevant = [s for s in gpu_samples if s["t"] >= t_start]
    if not relevant:
        relevant = gpu_samples
    max_gpu = max(s["gpu_util"] for s in relevant)
    max_mem = max(s["mem_used_mib"] for s in relevant)
    avg_gpu = sum(s["gpu_util"] for s in relevant) / len(relevant)
    print(f"  [GPU] Peak GPU util  : {max_gpu}%")
    print(f"  [GPU] Avg  GPU util  : {avg_gpu:.1f}%")
    print(f"  [GPU] Peak VRAM used : {max_mem} MiB / 6141 MiB")
    if max_gpu > 5:
        print("  [GPU] ✅ GPU đang được SỬ DỤNG bởi Ollama")
    else:
        print("  [GPU] ⚠️  GPU KHÔNG được dùng (Ollama chạy CPU-only)")
    gpu_samples.clear()


# ──────────────────────────────── Test helper ────────────────────────────────

def test_request(label, message, session=None):
    sid = session or SESSION
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Query: {message}")
    t_start = time.time()
    try:
        r = requests.post(
            BASE_URL,
            json={"message": message, "session_id": sid},
            timeout=120
        )
        elapsed = time.time() - t_start
        data = r.json()
        resp_text = data.get("response", "")
        method = data.get("method", "?")
        confidence = data.get("confidence", "?")
        intent = data.get("intent", "?")
        print(f"  HTTP: {r.status_code}  |  Time: {elapsed:.2f}s")
        print(f"  method={method}  confidence={confidence}  intent={intent}")
        print(f"  Response (80c): {resp_text[:80]}")
        print_gpu_report(label, t_start)
        return elapsed
    except requests.Timeout:
        elapsed = time.time() - t_start
        print(f"  [TIMEOUT] after {elapsed:.1f}s")
        print_gpu_report(label, t_start)
        return elapsed
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  [ERROR] {e}")
        return elapsed


# ──────────────────────────────── Main ────────────────────────────────────

def main():
    # Start GPU monitor thread
    mon = threading.Thread(target=monitor_gpu, daemon=True)
    mon.start()
    time.sleep(0.5)  # let a few samples accumulate

    print("="*60)
    print("CHATBOT PERFORMANCE & GPU DIAGNOSTIC REPORT")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Baseline GPU (idle) ──────────────────────────────────────
    print("\n[Baseline GPU - idle before any request]")
    time.sleep(2)
    if gpu_samples:
        idle_util = gpu_samples[-1]["gpu_util"]
        idle_mem  = gpu_samples[-1]["mem_used_mib"]
        print(f"  GPU util: {idle_util}%  |  VRAM used: {idle_mem} MiB")
    gpu_samples.clear()

    # ── Test 1: KB CAO (bypass LLM) ─────────────────────────────
    t1 = test_request(
        "KB CAO - Bypass LLM (rất nhanh)",
        "Chức năng chấm công hoạt động như thế nào"
    )

    # ── Test 2: KB TRUNG BÌNH (LLM needed) ──────────────────────
    t2 = test_request(
        "KB TRUNG BINH - LLM invoked (chậm nhất)",
        "Hiệu trưởng BDU tên gì"
    )

    # ── Test 3: Schedule (API call) ──────────────────────────────
    t3 = test_request(
        "Schedule - External API (nhanh)",
        "Thời khóa biểu tuần sau"
    )

    # ── Test 4: Social (template, instant) ───────────────────────
    t4 = test_request(
        "Social - Template (instant)",
        "Xin chào"
    )

    # ── Concurrent test ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST: CONCURRENT - 3 users cùng lúc (KB CAO)")
    gpu_samples.clear()
    results = []
    lock = threading.Lock()

    def concurrent_req(i):
        t = time.time()
        try:
            r = requests.post(
                BASE_URL,
                json={"message": "BDU có bao nhiêu cơ sở", "session_id": f"concurrent_{i}"},
                timeout=60
            )
            elapsed = time.time() - t
            with lock:
                results.append((i, r.status_code, round(elapsed, 2)))
        except Exception as e:
            with lock:
                results.append((i, 0, round(time.time() - t, 2)))

    t_conc_start = time.time()
    threads = [threading.Thread(target=concurrent_req, args=(i,)) for i in range(3)]
    for th in threads: th.start()
    for th in threads: th.join()
    t_conc_total = time.time() - t_conc_start

    for r in sorted(results):
        print(f"  User {r[0]}: HTTP {r[1]}, {r[2]}s")
    print(f"  Total wall time   : {t_conc_total:.2f}s")
    print(f"  Sequential would  : ~{round(t_conc_total/3, 1) * 3:.1f}s (giả sử ~{round(t_conc_total/3,1)}s/req)")
    if t_conc_total < min(r[2] for r in results) * 2.5:
        print("  ✅ Parallel: users xử lý CÙNG LÚC")
    else:
        print("  ⚠️  Sequential: users bị XẾP HÀNG chờ")
    print_gpu_report("concurrent", t_conc_start)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  KB CAO  (bypass LLM) : {t1:.2f}s   ← nhanh vì không gọi LLM")
    print(f"  KB TB   (dùng LLM)   : {t2:.2f}s   ← chậm vì Ollama generate")
    print(f"  Schedule (API)       : {t3:.2f}s")
    print(f"  Social  (template)   : {t4:.2f}s")
    print()
    print("PHÂN TÍCH BOTTLENECK:")
    if t2 > 10:
        print(f"  ⚠️  LLM response ({t2:.0f}s) là bottleneck chính")
        print("  Nguyên nhân có thể:")
        print("    1. Ollama đang chạy CPU-only (nếu GPU util thấp ở trên)")
        print("    2. Model quá lớn so với VRAM 6GB (vram overflow → CPU offload)")
        print("    3. num_predict=512 tokens * tốc độ token/s của model")
        print()
        print("  GỢI Ý TỐI ƯU:")
        print("    - Chạy: ollama run qwen2.5:7b (kiểm tra xem có dùng GPU không)")
        print("    - Hoặc dùng model nhỏ hơn: ollama pull qwen2.5:3b")
        print("    - Hoặc giảm num_predict từ 512 → 256 cho câu trả lời ngắn")

    stop_monitoring.set()
    mon.join(timeout=2)
    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
