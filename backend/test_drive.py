#!/usr/bin/env python
"""
Test Google Drive Integration
Chạy: python test_drive.py
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
sys.path.append(str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from ai_models.google_drive_service import google_drive_service
from ai_models.services import chatbot_ai

def test_google_drive_connection():
    """Test kết nối Google Drive"""
    print("🔍 Testing Google Drive connection...")
    
    # Test authentication
    if google_drive_service.service:
        print("✅ Google Drive authenticated successfully")
    else:
        print("❌ Google Drive authentication failed")
        return False
    
    # Test folder access
    try:
        file_info = google_drive_service._find_csv_file()
        if file_info:
            print(f"✅ Found CSV file: {file_info['name']} (ID: {file_info['id']})")
        else:
            print("⚠️ CSV file not found in Drive folder")
            print("📋 Make sure you:")
            print("   1. Shared folder with: bdu-chatbot-service@thinking-armor-463404-n1.iam.gserviceaccount.com")
            print("   2. Uploaded QA.csv to the folder")
            return False
    except Exception as e:
        print(f"❌ Error accessing folder: {str(e)}")
        return False
    
    return True

def test_csv_data_loading():
    """Test load dữ liệu CSV"""
    print("\n🔍 Testing CSV data loading...")
    
    try:
        # Test load data
        data = google_drive_service.get_csv_data()
        
        if data:
            print(f"✅ Loaded {len(data)} records from Drive")
            
            # Hiển thị vài record đầu
            for i, record in enumerate(data[:3]):
                print(f"   Record {i+1}: {record.get('question', '')[:50]}...")
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error loading CSV data: {str(e)}")
        return False
    
    return True

def test_system_status():
    """Test trạng thái hệ thống"""
    print("\n🔍 Testing system status...")
    
    try:
        # Google Drive status
        drive_status = google_drive_service.get_system_status()
        print("📊 Google Drive Status:")
        for key, value in drive_status.items():
            print(f"   {key}: {value}")
        
        # Chatbot system status
        system_status = chatbot_ai.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Knowledge entries: {system_status.get('knowledge_entries', 0)}")
        print(f"   Mode: {system_status.get('mode', 'unknown')}")
        print(f"   Google Drive integration: {'google_drive_integration' in system_status.get('lecturer_features', [])}")
        
    except Exception as e:
        print(f"❌ Error getting system status: {str(e)}")
        return False
    
    return True

def test_chatbot_response():
    """Test chatbot response với dữ liệu từ Drive"""
    print("\n🔍 Testing chatbot response...")
    
    try:
        # Test query
        test_query = "ngân hàng đề thi"
        result = chatbot_ai.process_query(test_query, session_id="test_drive")
        
        print(f"Query: '{test_query}'")
        print(f"Response: {result.get('response', '')[:100]}...")
        print(f"Confidence: {result.get('confidence', 0):.3f}")
        print(f"Method: {result.get('method', 'unknown')}")
        
        if result.get('confidence', 0) > 0.5:
            print("✅ Chatbot working with Drive data")
        else:
            print("⚠️ Low confidence - may need more data or better questions")
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {str(e)}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Testing Google Drive Integration for BDU Chatbot")
    print("=" * 60)
    
    tests = [
        ("Google Drive Connection", test_google_drive_connection),
        ("CSV Data Loading", test_csv_data_loading),
        ("System Status", test_system_status),
        ("Chatbot Response", test_chatbot_response)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Google Drive integration is working!")
        print("\n📋 Next steps:")
        print("   1. Start Django server: python manage.py runserver")
        print("   2. Test chatbot with Drive data")
        print("   3. Update CSV on Drive to see auto-refresh (5 minutes)")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Verify service account email is shared on Drive folder")
        print("   2. Check QA.csv file is uploaded and named correctly")
        print("   3. Ensure internet connection for Drive API")

if __name__ == "__main__":
    main()