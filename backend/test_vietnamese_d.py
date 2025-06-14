# test_vietnamese_d.py
# Chạy script này để test ngay

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_clean_function():
    """Test hàm clean để xem có mất chữ Đ không"""
    
    # Simulate the BROKEN version
    def broken_clean(text):
        import re
        # Đây là regex GÂY LỖI trong code của bạn
        text = re.sub(r'[ẤẬẦẨẪĂẮẶẰẲẴÂẤẬẦẨẪÉẾỆỀỂỄÊẾỆỀỂỄÍỊÌỈĨÓỘÒỎÕÔỐỘỒỔỖƠỚỢỜỞỠÚỤÙỦŨƯỨỰỪỬỮÝỴỲỶỸĐ]+(?=[^aăâeêiouôơưy\s])', '', text)
        return text
    
    # FIXED version
    def fixed_clean(text):
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        # Chỉ fix encoding issues, KHÔNG xóa Vietnamese chars
        encoding_fixes = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-', 'Ä': 'Đ'
        }
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        return text
    
    test_cases = [
        "Đại học Bình Dương",
        "Đảm bảo chất lượng",
        "Điện thoại di động", 
        "Đồng bằng sông Cửu Long"
    ]
    
    print("🧪 TESTING VIETNAMESE 'Đ' CHARACTER")
    print("=" * 50)
    
    for test in test_cases:
        broken_result = broken_clean(test)
        fixed_result = fixed_clean(test)
        
        print(f"\nOriginal:     {test}")
        print(f"Broken clean: {broken_result} {'❌' if 'Đ' not in broken_result and 'Đ' in test else '✅'}")
        print(f"Fixed clean:  {fixed_result} {'✅' if 'Đ' in fixed_result or 'Đ' not in test else '❌'}")

if __name__ == "__main__":
    test_clean_function()
    
    print("\n" + "="*50)
    print("🔧 TO FIX THE ISSUE:")
    print("1. Open chat/views.py")
    print("2. Find _clean_response_text() function") 
    print("3. REMOVE the regex line that deletes Vietnamese chars")
    print("4. Restart Django server")
    print("5. Test with: 'Đại học Bình Dương'")