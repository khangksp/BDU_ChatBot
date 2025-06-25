#!/usr/bin/env python3
"""
Test script for Enhanced Personalization Features
Tests response_style and user_memory_prompt functionality

Usage:
    python test_personalization.py

Requirements:
    - Django project setup with authentication app
    - Faculty model with personalization
    - API endpoints configured
"""

import os
import sys
import django
import json
import requests
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.contrib.auth import get_user_model
from authentication.models import Faculty
from ai_models.gemini_service import GeminiResponseGenerator
from ai_models.services import chatbot_ai

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

class PersonalizationTester:
    """Comprehensive tester for personalization features"""
    
    def __init__(self):
        self.test_results = {}
        self.test_faculty = None
        self.api_token = None
        
    def run_all_tests(self):
        """Run comprehensive personalization tests"""
        print("🧪 PERSONALIZATION TESTING SUITE")
        print("=" * 50)
        
        try:
            # 1. Setup test environment
            self.setup_test_data()
            
            # 2. Test model-level functionality
            self.test_faculty_model_personalization()
            
            # 3. Test response styles
            self.test_response_styles()
            
            # 4. Test user memory prompt
            self.test_user_memory_prompt()
            
            # 5. Test API endpoints
            self.test_api_endpoints()
            
            # 6. Test chat integration
            self.test_chat_personalization()
            
            # 7. Generate test report
            self.generate_test_report()
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_test_data(self):
        """Setup test faculty data"""
        print("\n1️⃣ Setting up test data...")
        
        # Create or get test faculty
        self.test_faculty, created = Faculty.objects.get_or_create(
            faculty_code='TEST_PERSON_001',
            defaults={
                'full_name': 'Nguyễn Văn Test',
                'email': 'test.personalization@bdu.edu.vn',
                'department': 'cntt',
                'position': 'giang_vien',
                'specialization': 'AI và Machine Learning',
                'office_room': 'A.101'
            }
        )
        
        if created:
            self.test_faculty.set_password('test123456')
            self.test_faculty.save()
            print(f"✅ Created test faculty: {self.test_faculty.faculty_code}")
        else:
            print(f"✅ Using existing test faculty: {self.test_faculty.faculty_code}")
        
        # Ensure clean preferences
        self.test_faculty.chatbot_preferences = self.test_faculty.get_default_chatbot_preferences()
        self.test_faculty.save()
        
        self.test_results['setup'] = {'status': 'success', 'faculty_code': self.test_faculty.faculty_code}
    
    def test_faculty_model_personalization(self):
        """Test Faculty model personalization methods"""
        print("\n2️⃣ Testing Faculty model personalization...")
        
        tests = {}
        
        # Test 1: Default preferences
        try:
            default_prefs = self.test_faculty.get_default_chatbot_preferences()
            assert 'response_style' in default_prefs
            assert 'user_memory_prompt' in default_prefs
            assert 'department_priority' in default_prefs
            tests['default_preferences'] = '✅ PASS'
        except Exception as e:
            tests['default_preferences'] = f'❌ FAIL: {e}'
        
        # Test 2: Style-specific instructions
        try:
            for style_code, _ in Faculty.RESPONSE_STYLE_CHOICES:
                instructions = self.test_faculty.get_style_specific_instructions(style_code)
                assert len(instructions) > 50  # Should have substantial content
                assert 'PHONG CÁCH' in instructions.upper()
            tests['style_instructions'] = '✅ PASS'
        except Exception as e:
            tests['style_instructions'] = f'❌ FAIL: {e}'
        
        # Test 3: Personalized system prompt
        try:
            system_prompt = self.test_faculty.get_personalized_system_prompt()
            assert self.test_faculty.faculty_code in system_prompt
            assert self.test_faculty.full_name in system_prompt
            assert 'PHONG CÁCH' in system_prompt
            tests['personalized_prompt'] = '✅ PASS'
        except Exception as e:
            tests['personalized_prompt'] = f'❌ FAIL: {e}'
        
        # Test 4: Preferences validation
        try:
            # Valid update
            self.test_faculty.update_chatbot_preferences({
                'response_style': 'friendly',
                'department_priority': False
            })
            
            # Invalid style should raise error
            try:
                self.test_faculty.update_chatbot_preferences({
                    'response_style': 'invalid_style'
                })
                tests['preferences_validation'] = '❌ FAIL: Should have raised error for invalid style'
            except ValueError:
                tests['preferences_validation'] = '✅ PASS'
                
        except Exception as e:
            tests['preferences_validation'] = f'❌ FAIL: {e}'
        
        self.test_results['model_tests'] = tests
        
        for test_name, result in tests.items():
            print(f"   {test_name}: {result}")
    
    def test_response_styles(self):
        """Test all response styles"""
        print("\n3️⃣ Testing response styles...")
        
        style_tests = {}
        test_query = "Ngân hàng đề thi là gì?"
        
        for style_code, style_name in Faculty.RESPONSE_STYLE_CHOICES:
            try:
                # Update faculty style
                self.test_faculty.update_chatbot_preferences({
                    'response_style': style_code
                })
                
                # Generate system prompt with this style
                system_prompt = self.test_faculty.get_personalized_system_prompt()
                
                # Verify style is applied
                style_instructions = self.test_faculty.get_style_specific_instructions(style_code)
                
                # Test with Gemini service
                gemini = GeminiResponseGenerator()
                gemini.set_user_context('test_session', {
                    'personalized_prompt': system_prompt,
                    'faculty_code': self.test_faculty.faculty_code,
                    'preferences': self.test_faculty.chatbot_preferences
                })
                
                # Verify style is detected correctly
                detected_style = gemini._get_user_response_style('test_session')
                
                if detected_style == style_code:
                    style_tests[f'{style_code}_{style_name}'] = '✅ PASS'
                else:
                    style_tests[f'{style_code}_{style_name}'] = f'❌ FAIL: Expected {style_code}, got {detected_style}'
                    
            except Exception as e:
                style_tests[f'{style_code}_{style_name}'] = f'❌ FAIL: {e}'
        
        self.test_results['style_tests'] = style_tests
        
        for test_name, result in style_tests.items():
            print(f"   {test_name}: {result}")
    
    def test_user_memory_prompt(self):
        """Test user memory prompt functionality"""
        print("\n4️⃣ Testing user memory prompt...")
        
        memory_tests = {}
        
        # Test 1: Custom memory prompt
        try:
            custom_memory = "Tôi là giảng viên CNTT, thích lập trình Python và AI. Tôi muốn câu trả lời có ví dụ code."
            
            self.test_faculty.update_chatbot_preferences({
                'user_memory_prompt': custom_memory
            })
            
            system_prompt = self.test_faculty.get_personalized_system_prompt()
            
            if custom_memory in system_prompt:
                memory_tests['custom_memory'] = '✅ PASS'
            else:
                memory_tests['custom_memory'] = '❌ FAIL: Custom memory not found in prompt'
                
        except Exception as e:
            memory_tests['custom_memory'] = f'❌ FAIL: {e}'
        
        # Test 2: Memory prompt length validation
        try:
            # Test max length (should fail)
            long_memory = "x" * 1001  # Exceeds 1000 char limit
            
            try:
                self.test_faculty.update_chatbot_preferences({
                    'user_memory_prompt': long_memory
                })
                memory_tests['length_validation'] = '❌ FAIL: Should have rejected long prompt'
            except ValueError:
                memory_tests['length_validation'] = '✅ PASS'
                
        except Exception as e:
            memory_tests['length_validation'] = f'❌ FAIL: {e}'
        
        # Test 3: Empty memory prompt fallback
        try:
            self.test_faculty.update_chatbot_preferences({
                'user_memory_prompt': ''
            })
            
            system_prompt = self.test_faculty.get_personalized_system_prompt()
            default_memory = self.test_faculty.get_default_memory_prompt()
            
            if default_memory in system_prompt:
                memory_tests['empty_fallback'] = '✅ PASS'
            else:
                memory_tests['empty_fallback'] = '❌ FAIL: Default memory not used for empty prompt'
                
        except Exception as e:
            memory_tests['empty_fallback'] = f'❌ FAIL: {e}'
        
        self.test_results['memory_tests'] = memory_tests
        
        for test_name, result in memory_tests.items():
            print(f"   {test_name}: {result}")
    
    def test_api_endpoints(self):
        """Test API endpoints (if server is running)"""
        print("\n5️⃣ Testing API endpoints...")
        
        api_tests = {}
        
        try:
            # Try to login and get token
            login_response = requests.post(f"{API_BASE}/auth/login/", {
                'faculty_code': self.test_faculty.faculty_code,
                'password': 'test123456'
            }, timeout=5)
            
            if login_response.status_code == 200:
                self.api_token = login_response.json()['data']['token']
                headers = {'Authorization': f'Token {self.api_token}'}
                
                # Test 1: Get preferences
                prefs_response = requests.get(f"{API_BASE}/auth/chatbot/preferences/", headers=headers, timeout=5)
                if prefs_response.status_code == 200:
                    api_tests['get_preferences'] = '✅ PASS'
                else:
                    api_tests['get_preferences'] = f'❌ FAIL: Status {prefs_response.status_code}'
                
                # Test 2: Update preferences
                update_response = requests.post(f"{API_BASE}/auth/chatbot/preferences/update/", {
                    'preferences': {
                        'response_style': 'technical',
                        'user_memory_prompt': 'Test memory prompt for API',
                        'department_priority': True
                    }
                }, headers=headers, timeout=5)
                
                if update_response.status_code == 200:
                    api_tests['update_preferences'] = '✅ PASS'
                else:
                    api_tests['update_preferences'] = f'❌ FAIL: Status {update_response.status_code}'
                
                # Test 3: Get system prompt
                prompt_response = requests.get(f"{API_BASE}/auth/chatbot/system-prompt/", headers=headers, timeout=5)
                if prompt_response.status_code == 200:
                    api_tests['get_system_prompt'] = '✅ PASS'
                else:
                    api_tests['get_system_prompt'] = f'❌ FAIL: Status {prompt_response.status_code}'
                    
            else:
                api_tests['login'] = f'❌ FAIL: Login failed with status {login_response.status_code}'
                
        except requests.ConnectionError:
            api_tests['connection'] = '⚠️ SKIP: Server not running'
        except Exception as e:
            api_tests['error'] = f'❌ FAIL: {e}'
        
        self.test_results['api_tests'] = api_tests
        
        for test_name, result in api_tests.items():
            print(f"   {test_name}: {result}")
    
    def test_chat_personalization(self):
        """Test chat integration with personalization"""
        print("\n6️⃣ Testing chat personalization integration...")
        
        chat_tests = {}
        
        try:
            # Test with different styles
            test_queries = [
                "Ngân hàng đề thi có quy trình như thế nào?",
                "Kê khai nhiệm vụ năm học bao gồm những gì?",
                "Tạp chí khoa học nhận bài từ đâu?"
            ]
            
            for style_code, style_name in Faculty.RESPONSE_STYLE_CHOICES:
                try:
                    # Update style
                    self.test_faculty.update_chatbot_preferences({
                        'response_style': style_code,
                        'user_memory_prompt': f'Test với phong cách {style_name}'
                    })
                    
                    # Test with chatbot_ai
                    user_context = self.test_faculty.get_chatbot_context()
                    system_prompt = self.test_faculty.get_personalized_system_prompt()
                    
                    # Simulate chat processing
                    session_id = f'test_session_{style_code}'
                    
                    # Set user context in gemini service
                    chatbot_ai.response_generator.set_user_context(session_id, {
                        'personalized_prompt': system_prompt,
                        'faculty_code': user_context.get('faculty_code'),
                        'full_name': user_context.get('full_name'),
                        'preferences': user_context.get('preferences')
                    })
                    
                    # Test query processing
                    response = chatbot_ai.process_query(test_queries[0], session_id)
                    
                    if response.get('response') and response.get('response_style') == style_code:
                        chat_tests[f'chat_{style_code}'] = '✅ PASS'
                    else:
                        chat_tests[f'chat_{style_code}'] = f'❌ FAIL: Style not properly applied'
                        
                except Exception as e:
                    chat_tests[f'chat_{style_code}'] = f'❌ FAIL: {e}'
        
        except Exception as e:
            chat_tests['integration_error'] = f'❌ FAIL: {e}'
        
        self.test_results['chat_tests'] = chat_tests
        
        for test_name, result in chat_tests.items():
            print(f"   {test_name}: {result}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 COMPREHENSIVE TEST REPORT")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if '✅ PASS' in result:
                        passed_tests += 1
                    elif '❌ FAIL' in result:
                        failed_tests += 1
                    elif '⚠️ SKIP' in result:
                        skipped_tests += 1
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⚠️ Skipped: {skipped_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Detailed report
        print(f"\n📋 DETAILED RESULTS:")
        for category, tests in self.test_results.items():
            print(f"\n{category.upper()}:")
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    print(f"   {test_name}: {result}")
            else:
                print(f"   {tests}")
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'success_rate': passed_tests/total_tests*100 if total_tests > 0 else 0
            },
            'detailed_results': self.test_results
        }
        
        with open('personalization_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test report saved to: personalization_test_report.json")
        
        # Clean up
        self.cleanup()
        
    def cleanup(self):
        """Clean up test data"""
        try:
            if self.test_faculty:
                # Reset to default preferences
                self.test_faculty.chatbot_preferences = self.test_faculty.get_default_chatbot_preferences()
                self.test_faculty.save()
                print(f"\n🧹 Cleaned up test faculty: {self.test_faculty.faculty_code}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


def run_quick_test():
    """Quick test for immediate feedback"""
    print("🚀 QUICK PERSONALIZATION TEST")
    print("=" * 30)
    
    try:
        # Test Faculty model
        faculty = Faculty.objects.create(
            faculty_code='QUICK_TEST',
            full_name='Quick Test User',
            email='quick@test.com',
            department='cntt',
            position='giang_vien'
        )
        
        # Test 1: Default preferences
        default_prefs = faculty.get_default_chatbot_preferences()
        print(f"✅ Default preferences: {default_prefs.get('response_style')}")
        
        # Test 2: Style instructions
        instructions = faculty.get_style_specific_instructions('friendly')
        print(f"✅ Style instructions length: {len(instructions)} characters")
        
        # Test 3: System prompt
        system_prompt = faculty.get_personalized_system_prompt()
        print(f"✅ System prompt includes faculty code: {faculty.faculty_code in system_prompt}")
        
        # Test 4: Update preferences
        faculty.update_chatbot_preferences({
            'response_style': 'technical',
            'user_memory_prompt': 'I am a technical person',
            'department_priority': True
        })
        print(f"✅ Preferences updated: {faculty.chatbot_preferences.get('response_style')}")
        
        # Cleanup
        faculty.delete()
        print("✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_test()
    else:
        tester = PersonalizationTester()
        tester.run_all_tests()