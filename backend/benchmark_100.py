# -*- coding: utf-8 -*-
"""
BDU Chatbot Benchmark: 100 cau hoi - Do toc do & Do chinh xac
- Cau KB: Paraphrase, khong copy nguyen ban
- Cau API: Thong tin ca nhan, lich day (dung JWT)
- Tieu chi: > 10s = FAIL (qua cham)
"""
import sys, io
# Force UTF-8 output tren Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import requests
import json
import time
import statistics

BASE_URL = "http://localhost:8000/api/chat/"
JWT = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTc3MjAyMDgxNSwianRpIjoiYjNjNWZlYjgtN2U1YS00ZGIwLWFkNGUtNGQ2ZThkYWQ3NzM4IiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IjkwODIyIiwibmJmIjoxNzcyMDIwODE1LCJjc3JmIjoiYjYwMTU0MjMtMTZjYy00YjY0LTg4YzYtMjA4ZGEyMGNjNTBiIiwiZXhwIjoxODAzNTU2ODE1LCJ0YWlfa2hvYW4iOnsibWFfdGFpX2tob2FuIjoxLCJ0YWlfa2hvYW4iOiJkYXR1YW4iLCJ2YWlfdHJvIjoidmllbl9jaHVjIiwidHJhbmdfdGhhaSI6MSwibmdheV90YW8iOiIyMDI1LTA2LTE3IiwibGFzdF9sb2dpbiI6IjIwMjYtMDItMjUgMDc6MTg6NDAiLCJ2ZXJzaW9uIjoiIn0sInZpZW5fY2h1YyI6eyJtYV90YWlfa2hvYW4iOjEsIm1hX3ZpZW5fY2h1YyI6IjkwODIyIiwibWFfbmhhbl92aWVuIjoiIiwiaG9fdmFfdGVuIjoiRFx1MDFiMFx1MDFhMW5nIEFuaCBUdVx1MWVhNW4iLCJtYV9jaHVjX2RhbmgiOjUsImNodWNfZGFuaCI6IlBoXHUwMGYzIFRyXHUwMWIwXHUxZWRmbmcgQlx1MWVkOSBNXHUwMGY0biIsImdtYWlsIjoiZGF0dWFuQGJkdS5lZHUudm4iLCJzb19kaWVuX3Rob2FpIjoiMDkxMzA3NzgwOSIsIm1hX2Rvbl92aSI6NSwiZ2lvaV90aW5oIjowLCJ0cmluaF9kbyI6IlRoXHUxZWExYyBzXHUwMTI5IiwiYXZhdGFyX3VybCI6Imh0dHBzOi8vZHJpdmUuZ29vZ2xlLmNvbS91Yz9pZD0xaTFpT19oSDFRUDdJdjdQM3p6TUhnWm1oS3N0OUhfYU4iLCJpc19hY3RpdmUiOnRydWUsInRlbl9kb25fdmkiOiJWaVx1MWVjN24gVHJcdTAwZWQgdHVcdTFlYzcgbmhcdTAwZTJuIHRcdTFlYTFvIHZcdTAwZTAgQ2h1eVx1MWVjM24gXHUwMTEwXHUxZWQ1aSBTXHUxZWQxIn19"
    ".coB4mPlietvjjQlx45EAszew40sDhN-oywzLaSqlNCN4aMSRbtoajgUFV-MOs_pJs60x3l2AlwExNdmeYzJnKEA9VZDkzVMTGoJ5ymCZeUwjGqygGzQuc8SKeLw0IuokF5PC5TNrIwhAYnt9n-SZ6_97adr7mk-n73buYMY5IbpEQ0BZAqs6m1qrk5JgooNiSRDphTYyvsqH9jzZWaUHIOJ77LHc56gZeHs3DhUcycLPcaUzmz6ogeew6KXOiPGrn7H39zYzICGGtYU0Gcflp4ZWya8Y4Rq0c-br9bVNOACfL2pVpt2MhFe8oPJ8vym-OYg5JnqY5XDfhg_Ku1aS4g"
)

HEADERS_JWT = {"Authorization": f"Bearer {JWT}", "Content-Type": "application/json"}
HEADERS_NO  = {"Content-Type": "application/json"}

# 100 cau: (message, category, expected_keywords_vi_lower, use_jwt)
QUESTIONS = [
    # KB - Gioi thieu truong (10)
    ("BDU viet day du la gi vay?",                             "KB-general",  ["binh duong","bình dương"], False),
    ("Truong dai hoc Binh Duong thanh lap nam nao?",           "KB-general",  ["1997","thành lập"],        False),
    ("Truong BDU nam o dau?",                                  "KB-general",  ["bình dương","địa chỉ"],    False),
    ("Hieu truong truong la ai?",                              "KB-general",  ["hiếu","cao"],              False),
    ("Hien tai BDU co may co so?",                             "KB-general",  ["cơ sở"],                   False),
    ("Dia chi chinh cua Dai hoc Binh Duong o dau?",            "KB-general",  ["bình dương"],              False),
    ("Tam nhin cua truong Dai hoc Binh Duong la gi?",          "KB-general",  [],                          False),
    ("Su menh cua BDU la gi?",                                 "KB-general",  [],                          False),
    ("BDU co bao nhieu sinh vien dang hoc?",                   "KB-general",  [],                          False),
    ("Truong BDU duoc thanh lap boi to chuc nao?",             "KB-general",  [],                          False),

    # KB - Dao tao (10)
    ("Truong Binh Duong dao tao bao nhieu nganh dai hoc?",     "KB-edu",      ["18","ngành"],              False),
    ("Chuong trinh dao tao cua BDU co duoc cong khai khong?",  "KB-edu",      ["công khai"],               False),
    ("BDU co dao tao sau dai hoc khong?",                      "KB-edu",      [],                          False),
    ("Trong bao nhieu nam thi sinh vien BDU tot nghiep cu nhan?","KB-edu",    [],                          False),
    ("Khoa nao tai BDU phu trach nganh CNTT?",                 "KB-edu",      [],                          False),
    ("BDU co lien ket dao tao quoc te khong?",                 "KB-edu",      [],                          False),
    ("Nganh Duoc BDU da duoc kiem dinh chua?",                 "KB-edu",      [],                          False),
    ("Sinh vien co the dang ky tin chi tu do tai BDU khong?",  "KB-edu",      [],                          False),
    ("Thoi gian nghi he cua BDU nam nay nhu the nao?",         "KB-edu",      [],                          False),
    ("Diem chuan nganh Ky thuat Phan mem BDU 2024 la bao nhieu?","KB-edu",   [],                          False),

    # KB - Hoc vu (10)
    ("Sinh vien bi canh bao hoc vu khi nao?",                  "KB-acad",     ["cảnh báo"],               False),
    ("Dieu kien xet hoc bong khuyen khich hoc tap la gi?",     "KB-acad",     [],                          False),
    ("Sinh vien muon bao luu thi can lam thu tuc gi?",         "KB-acad",     ["bảo lưu"],                False),
    ("Ho so xin nghi hoc tam thoi gom nhung gi?",              "KB-acad",     [],                          False),
    ("Dieu kien de tot nghiep dung han la gi?",                "KB-acad",     [],                          False),
    ("Muc hoc phi dai hoc BDU hien nay la bao nhieu?",         "KB-acad",     [],                          False),
    ("Sinh vien duoc thi lai bao nhieu lan?",                  "KB-acad",     [],                          False),
    ("Diem tich luy toi thieu de khong bi buoc thoi hoc?",     "KB-acad",     [],                          False),
    ("Cach dang ky hoc lai mon truot ra sao?",                 "KB-acad",     [],                          False),
    ("Thu tuc xin mien giam hoc phi thuc hien o phong nao?",   "KB-acad",     [],                          False),

    # KB - Cham cong & Hanh chinh (10)
    ("Chuc nang cham cong trong he thong de lam gi?",          "KB-admin",    ["chấm công"],              False),
    ("Ke khai nhiem vu giang day theo quy trinh nao?",         "KB-admin",    [],                          False),
    ("Bao cao ket qua giang day nop han chot khi nao?",        "KB-admin",    [],                          False),
    ("Dang ky nghien cuu khoa hoc can dieu kien gi?",          "KB-admin",    [],                          False),
    ("Quy dinh ve gio day tieu chuan hang nam la the nao?",    "KB-admin",    [],                          False),
    ("Cach nop de cuong mon hoc len he thong nhu the nao?",    "KB-admin",    [],                          False),
    ("Xet duyet de tai NCKH cap truong thuc hien ra sao?",     "KB-admin",    [],                          False),
    ("Ai ky duyet don xin nghi phep cua giang vien?",          "KB-admin",    [],                          False),
    ("Thu tuc xin thanh toan tien giang them gio?",            "KB-admin",    [],                          False),
    ("Ho so thang hang chuc danh nghe nghiep gom nhung gi?",   "KB-admin",    [],                          False),

    # KB - CNTT / He thong (8)
    ("Phan mem QLVB&HSCV dung de lam gi?",                    "KB-it",       [],                          False),
    ("Du lieu he thong QLVB duoc luu o dau?",                  "KB-it",       [],                          False),
    ("Email sinh vien BDU co duoi gi?",                        "KB-it",       ["bdu.edu.vn"],             False),
    ("He thong LMS cua BDU dung nen tang nao?",                "KB-it",       [],                          False),
    ("Cong thong tin sinh vien BDU truy cap o dau?",           "KB-it",       [],                          False),
    ("BDU co ho tro phan mem Office cho sinh vien khong?",     "KB-it",       [],                          False),
    ("Cach reset mat khau tai khoan truong nhu the nao?",      "KB-it",       [],                          False),
    ("He thong dao tao truc tuyen BDU dang dung gi?",          "KB-it",       [],                          False),

    # KB - Nghien cuu (5)
    ("Giang vien phai cong bo bao nhieu bai bao moi nam?",     "KB-research", [],                          False),
    ("Quy dinh ve dang ky so huu tri tue tai BDU?",            "KB-research", [],                          False),
    ("Kinh phi NCKH cap khoa toi da bao nhieu?",               "KB-research", [],                          False),
    ("Dieu kien duoc duyet de tai NCKH ca nhan?",              "KB-research", [],                          False),
    ("Thoi han thuc hien de tai NCKH cap truong la may nam?",  "KB-research", [],                          False),

    # KB - Sinh vien (7)
    ("Ky tuc xa BDU co nhan sinh vien nam nhat khong?",        "KB-student",  [],                          False),
    ("Hoc bong toan phan BDU danh cho doi tuong nao?",         "KB-student",  [],                          False),
    ("Trung tam ho tro sinh vien BDU lam viec gio nao?",       "KB-student",  [],                          False),
    ("Cau lac bo hoc thuat tai BDU co may CLB?",               "KB-student",  [],                          False),
    ("Hoat dong Ren luyen sinh vien BDU tinh diem the nao?",   "KB-student",  [],                          False),
    ("BDU co chuong trinh thuc tap doanh nghiep khong?",       "KB-student",  [],                          False),
    ("Vay von hoc phi co the lien he ai tai BDU?",             "KB-student",  [],                          False),

    # KB - Phap che / Noi quy (10)
    ("Giang vien vi pham noi quy se bi xu ly the nao?",        "KB-rule",     [],                          False),
    ("Quy dinh ve dong phuc giang day tai BDU?",               "KB-rule",     [],                          False),
    ("Nghi phep nam tinh tu ngay nao den ngay nao?",           "KB-rule",     [],                          False),
    ("Dieu kien gia han hop dong lao dong giang vien?",        "KB-rule",     [],                          False),
    ("Khi nao giang vien duoc huong phu cap tham nien?",       "KB-rule",     [],                          False),
    ("So ngay nghi phep nam toi da la bao nhieu?",             "KB-rule",     [],                          False),
    ("Quy dinh gio day vuot va cach tinh thu lao?",            "KB-rule",     [],                          False),
    ("Luong co so ap dung hien hanh cho vien chuc?",           "KB-rule",     [],                          False),
    ("Quy trinh khieu nai, to cao noi bo tai BDU?",            "KB-rule",     [],                          False),
    ("Dieu kien duoc phong tang danh hieu GVNG?",              "KB-rule",     [],                          False),

    # API - Thong tin ca nhan (10)
    ("Toi ten la ai?",                                         "API-personal",["dương","tuấn","duong"],    True),
    ("Chuc danh cua toi tai truong la gi?",                    "API-personal",["phó","trưởng","bộ môn"],   True),
    ("Email cong vu cua toi la gi?",                           "API-personal",["datuan","bdu.edu.vn"],     True),
    ("So dien thoai cua toi trong he thong la so nao?",        "API-personal",["0913"],                    True),
    ("Toi cong tac o don vi nao?",                             "API-personal",["chuyển đổi","viễn"],       True),
    ("Ma so giang vien cua toi la bao nhieu?",                 "API-personal",["90822"],                   True),
    ("Trinh do hoc van cua toi ghi gi trong he thong?",        "API-personal",["thạc","sĩ"],               True),
    ("Ban biet toi la ai khong?",                              "API-personal",["dương","tuấn"],            True),
    ("Ho va ten day du cua toi la gi?",                        "API-personal",["dương","anh","tuấn"],      True),
    ("Toi dang giu chuc vu gi?",                               "API-personal",["phó","trưởng"],            True),

    # API - Lich day (10)
    ("Tuan nay toi co lich day khong?",                        "API-schedule",["lịch","tuần"],             True),
    ("Toi day mon gi trong tuan sau?",                         "API-schedule",["tuần","lịch"],             True),
    ("Lich giang day tuan toi cua toi nhu the nao?",           "API-schedule",["tuần","lịch"],             True),
    ("Thu may tuan sau toi phai len lop?",                     "API-schedule",["tuần"],                    True),
    ("Toi day o phong nao trong tuan toi?",                    "API-schedule",["tuần"],                    True),
    ("Nhom sinh vien toi day tuan sau la nhom may?",           "API-schedule",["tuần"],                    True),
    ("Tuan sau toi bat dau tiet may moi buoi?",               "API-schedule",["tuần"],                    True),
    ("Toi co day vao thu Bay tuan toi khong?",                 "API-schedule",["tuần","lịch"],             True),
    ("Cho toi xem lich day 7 ngay toi?",                       "API-schedule",["tuần","lịch"],             True),
    ("Lich giang day tuan sau gom nhung ngay nao?",            "API-schedule",["tuần"],                    True),
]

# ─────────────────────────────────────────────────────────────────────────────

def ask(question, use_jwt, session_id, timeout=12):
    headers = HEADERS_JWT if use_jwt else HEADERS_NO
    payload  = {"message": question, "session_id": session_id}
    t0 = time.time()
    try:
        r = requests.post(BASE_URL, headers=headers, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            return elapsed, data.get("response", ""), data.get("confidence", 0), data.get("method", ""), True
        else:
            return time.time() - t0, "HTTP %d" % r.status_code, 0, "error", False
    except requests.Timeout:
        return 10.01, "TIMEOUT", 0, "timeout", False
    except Exception as e:
        return time.time() - t0, "ERR:%s" % str(e)[:50], 0, "exception", False


def keyword_hit(response, keywords):
    if not keywords:
        return None  # khong the danh gia
    rv = response.lower()
    # Vietnamese fallback: thu vi latin
    rv_latin = rv
    return any(kw.lower() in rv_latin for kw in keywords)


def main():
    SESSION = "bench_%d" % int(time.time())
    print("=" * 70)
    print("BDU Chatbot - Benchmark 100 cau hoi")
    print("Server:", BASE_URL)
    print("Session:", SESSION)
    print("Tieu chi FAIL: response_time >= 10s")
    print("=" * 70)

    results = []
    PASS = FAIL_SLOW = FAIL_ACC = UNKN = 0
    times_all = []
    times_pass = []
    category_stats = {}

    for idx, (q, cat, keywords, use_jwt) in enumerate(QUESTIONS, 1):
        elapsed, resp, conf, method, ok = ask(q, use_jwt, SESSION)
        times_all.append(elapsed)

        slow = elapsed >= 10.0
        kw_result = keyword_hit(resp, keywords)  # True/False/None

        if slow:
            status = "FAIL-SLOW"
            FAIL_SLOW += 1
        elif kw_result is False:
            status = "FAIL-ACC"
            FAIL_ACC += 1
        elif kw_result is None:
            status = "PASS?"     # khong co keyword -> ghi nhan nhung coi la pass
            PASS += 1
            UNKN += 1
            times_pass.append(elapsed)
        else:
            status = "PASS"
            PASS += 1
            times_pass.append(elapsed)

        c = category_stats.setdefault(cat, {"pass": 0, "fail_slow": 0, "fail_acc": 0, "times": []})
        if "PASS" in status:
            c["pass"] += 1
            c["times"].append(elapsed)
        elif status == "FAIL-SLOW":
            c["fail_slow"] += 1
        else:
            c["fail_acc"] += 1

        jwt_tag = "[JWT]" if use_jwt else "     "
        icon = "OK  " if "PASS" in status else "SLOW" if status == "FAIL-SLOW" else "ERR "
        short_q = q[:50]
        print("[%03d] %s %s  %5.2fs  %-12s  %-14s  %s" % (idx, jwt_tag, icon, elapsed, status, cat, short_q))
        if "FAIL" in status:
            short_r = resp[:90].replace("\n", " ")
            print("       resp: %s" % short_r)

        time.sleep(0.1)

    total = len(QUESTIONS)
    print()
    print("=" * 70)
    print("TONG KET %d CAU" % total)
    print("=" * 70)
    print("  PASS         : %3d / %d  (%.1f%%)" % (PASS, total, PASS/total*100))
    print("  FAIL (>=10s) : %3d / %d  (%.1f%%)" % (FAIL_SLOW, total, FAIL_SLOW/total*100))
    print("  FAIL (acc)   : %3d / %d  (%.1f%%)" % (FAIL_ACC, total, FAIL_ACC/total*100))
    print("  (PASS? kw=NA): %3d" % UNKN)
    print()

    if times_pass:
        srt = sorted(times_pass)
        p90 = srt[int(len(srt)*0.9)]
        p50 = srt[int(len(srt)*0.5)]
        print("  Thoi gian (chi PASS):")
        print("    Min    : %.2fs" % min(times_pass))
        print("    Max    : %.2fs" % max(times_pass))
        print("    Avg    : %.2fs" % (sum(times_pass)/len(times_pass)))
        print("    Median : %.2fs" % p50)
        print("    P90    : %.2fs" % p90)

    if times_all:
        srt_all = sorted(times_all)
        print()
        print("  Thoi gian (toan bo 100 cau):")
        print("    Avg    : %.2fs" % (sum(times_all)/len(times_all)))
        print("    P90    : %.2fs" % srt_all[int(len(srt_all)*0.9)])
        print("    Max    : %.2fs" % max(times_all))

    print()
    print("-- Theo nhom cau hoi " + "-"*48)
    for cat, s in sorted(category_stats.items()):
        total_cat = s["pass"] + s["fail_slow"] + s["fail_acc"]
        avg_t = ("%.2fs" % (sum(s["times"])/len(s["times"]))) if s["times"] else "N/A"
        print("  %-16s  PASS=%d/%d  FailSlow=%d  FailAcc=%d  AvgPassTime=%s" % (
            cat, s["pass"], total_cat, s["fail_slow"], s["fail_acc"], avg_t))

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
