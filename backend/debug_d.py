#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script để tìm lỗi encoding với chữ "đ"
"""

import os
import sys
import django
import json
import logging

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from ai_models.services import HybridChatbotAI

def test_encoding_pipeline():
    """Test từng bước encoding trong pipeline"""
    
    # Test data với chữ "đ"
    test_queries = [
        "KT. HIỆU TRƯỞNG PHÓ HIỆU TRƯỞNG THƯỜNG TRỰC - ĐỖ DOÃN TRANG",
        "Thông báo được ký bởi Hiệu trưởng",
        "Đào tạo và phát triển",
        "Địa chỉ trường đại học",
        "Đăng ký học phần"
    ]
    
    chatbot = HybridChatbotAI()
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING: {query}")
        print(f"{'='*80}")
        
        # Step 1: Check input encoding
        print(f"📥 INPUT ENCODING:")
        print(f"   Original: {repr(query)}")
        print(f"   UTF-8 bytes: {query.encode('utf-8')}")
        print(f"   Length: {len(query)} chars")
        
        # Step 2: Process with chatbot
        result = chatbot.process_query(query, session_id="debug_session")
        response = result.get('response', '')
        
        # Step 3: Check output encoding
        print(f"\n📤 OUTPUT ENCODING:")
        print(f"   Response: {repr(response)}")
        print(f"   UTF-8 bytes: {response.encode('utf-8') if response else b''}")
        print(f"   Length: {len(response)} chars")
        
        # Step 4: Check for missing "đ"
        input_d_count = query.count('đ') + query.count('Đ')
        output_d_count = response.count('đ') + response.count('Đ')
        
        print(f"\n🔍 CHARACTER ANALYSIS:")
        print(f"   Input 'đ/Đ' count: {input_d_count}")
        print(f"   Output 'đ/Đ' count: {output_d_count}")
        
        if input_d_count != output_d_count:
            print(f"   ❌ MISSING CHARACTERS: {input_d_count - output_d_count}")
            
            # Find specific missing positions
            for i, char in enumerate(query):
                if char in ['đ', 'Đ']:
                    if i < len(response) and response[i] != char:
                        print(f"   🔍 Position {i}: '{char}' → '{response[i] if i < len(response) else 'EOF'}'")
        else:
            print(f"   ✅ All 'đ/Đ' characters preserved")
        
        # Step 5: Test JSON serialization
        print(f"\n🔄 JSON SERIALIZATION TEST:")
        try:
            json_str = json.dumps({'response': response}, ensure_ascii=False)
            json_parsed = json.loads(json_str)
            json_response = json_parsed['response']
            
            print(f"   JSON string: {repr(json_str[:100])}...")
            print(f"   JSON parsed: {repr(json_response)}")
            
            json_d_count = json_response.count('đ') + json_response.count('Đ')
            print(f"   JSON 'đ/Đ' count: {json_d_count}")
            
            if json_d_count != output_d_count:
                print(f"   ❌ JSON SERIALIZATION ISSUE!")
            else:
                print(f"   ✅ JSON serialization OK")
                
        except Exception as e:
            print(f"   ❌ JSON Error: {e}")

def test_gemini_api_directly():
    """Test Gemini API trực tiếp"""
    print(f"\n{'='*80}")
    print(f"🤖 TESTING GEMINI API DIRECTLY")
    print(f"{'='*80}")
    
    from ai_models.gemini_service import GeminiResponseGenerator
    
    gemini = GeminiResponseGenerator()
    
    test_prompt = """
    Bạn là AI assistant của Đại học Bình Dương (BDU), chuyên hỗ trợ giảng viên.
    
    Câu hỏi: Ai ký thông báo này?
    
    Thông tin: Thông báo được ký bởi KT. HIỆU TRƯỞNG PHÓ HIỆU TRƯỞNG THƯỜNG TRỰC - ĐỖ DOÃN TRANG
    
    Trả lời ngắn gọn với format: "Dạ thầy/cô, [thông tin]. Thầy/cô có cần hỗ trợ thêm gì không ạ?"
    """
    
    print(f"📤 SENDING TO GEMINI:")
    print(f"   Prompt contains 'đ/Đ': {test_prompt.count('đ') + test_prompt.count('Đ')} chars")
    
    try:
        response = gemini._call_gemini_api_optimized(test_prompt, 'balanced')
        
        print(f"\n📥 GEMINI RESPONSE:")
        print(f"   Response: {repr(response)}")
        print(f"   Contains 'đ/Đ': {response.count('đ') + response.count('Đ') if response else 0} chars")
        
        if response:
            # Check specific characters
            if 'ĐỖ DOÃN TRANG' in response:
                print(f"   ✅ Full name preserved")
            elif 'Ỗ DOÃN TRANG' in response:
                print(f"   ❌ Missing initial 'Đ'")
            elif 'DOÃN TRANG' in response:
                print(f"   ❌ Missing 'ĐỖ'")
            else:
                print(f"   ❌ Name heavily corrupted")
                
    except Exception as e:
        print(f"   ❌ Gemini API Error: {e}")

def test_database_encoding():
    """Test database encoding"""
    print(f"\n{'='*80}")
    print(f"🗄️  TESTING DATABASE ENCODING")
    print(f"{'='*80}")
    
    from knowledge.models import KnowledgeBase
    
    # Test data in database
    test_entries = KnowledgeBase.objects.filter(
        question__icontains='đ'
    )[:5]
    
    print(f"📊 Found {test_entries.count()} entries with 'đ'")
    
    for entry in test_entries:
        print(f"\n🔍 Database Entry ID {entry.id}:")
        print(f"   Question: {repr(entry.question)}")
        print(f"   Answer: {repr(entry.answer)}")
        print(f"   'đ/Đ' in question: {entry.question.count('đ') + entry.question.count('Đ')}")
        print(f"   'đ/Đ' in answer: {entry.answer.count('đ') + entry.answer.count('Đ')}")

def test_django_response():
    """Test Django HTTP response encoding"""
    print(f"\n{'='*80}")
    print(f"🌐 TESTING DJANGO HTTP RESPONSE")
    print(f"{'='*80}")
    
    from django.http import JsonResponse
    import json
    
    test_data = {
        'response': 'Dạ thầy/cô, Thông báo được ký bởi KT. HIỆU TRƯỞNG PHÓ HIỆU TRƯỞNG THƯỜNG TRỰC - ĐỖ DOÃN TRANG. 🎓 Thầy/cô có cần hỗ trợ thêm gì không ạ?'
    }
    
    print(f"📤 Original data:")
    print(f"   {repr(test_data['response'])}")
    print(f"   'đ/Đ' count: {test_data['response'].count('đ') + test_data['response'].count('Đ')}")
    
    # Test JsonResponse
    response = JsonResponse(test_data, json_dumps_params={'ensure_ascii': False})
    content = response.content.decode('utf-8')
    
    print(f"\n📥 JsonResponse content:")
    print(f"   {repr(content)}")
    
    # Parse back
    parsed = json.loads(content)
    parsed_response = parsed['response']
    
    print(f"\n🔄 Parsed back:")
    print(f"   {repr(parsed_response)}")
    print(f"   'đ/Đ' count: {parsed_response.count('đ') + parsed_response.count('Đ')}")

if __name__ == "__main__":
    print("🚀 Starting Encoding Debug Tests...")
    
    # Test 1: Full pipeline
    test_encoding_pipeline()
    
    # Test 2: Gemini API directly
    test_gemini_api_directly()
    
    # Test 3: Database
    test_database_encoding()
    
    # Test 4: Django response
    test_django_response()
    
    print(f"\n{'='*80}")
    print(f"✅ Debug tests completed!")
    print(f"{'='*80}")