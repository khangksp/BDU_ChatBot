#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive ChatBot Testing Script for BDU Lecturer Assistant
Mục đích: Test tất cả câu hỏi và tìm ra những keyword/pattern còn thiếu
"""

import os
import sys
import django
import pandas as pd
import json
import time
from datetime import datetime
from collections import defaultdict, Counter
import re

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')  # Fixed: backend.settings
django.setup()

# Import your chatbot services
from ai_models.services import HybridChatbotAI  # Fixed: ai_models instead of ai_services

class ChatBotTester:
    """Comprehensive Testing Suite for ChatBot"""
    
    def __init__(self):
        self.chatbot = HybridChatbotAI()
        self.test_results = []
        self.failed_queries = []
        self.keyword_analysis = defaultdict(list)
        self.session_id = f"test_session_{int(time.time())}"
        
        # ✅ CATEGORIES để phân loại câu hỏi
        self.categories = {
            'tuition_fees': ['học phí', 'lệ phí', 'phí', 'tiền', 'chi phí', 'thanh toán'],
            'admission': ['tuyển sinh', 'nhập học', 'đăng ký', 'xét tuyển', 'điều kiện'],
            'graduation': ['tốt nghiệp', 'bằng', 'văn bằng', 'nhận bằng', 'tốt nghiệp'],
            'lecturer_tasks': ['ngân hàng đề thi', 'kê khai', 'nhiệm vụ', 'báo cáo', 'giảng viên'],
            'academic_journal': ['tạp chí', 'bài viết', 'nghiên cứu', 'khoa học'],
            'competition_awards': ['thi đua', 'khen thưởng', 'danh hiệu', 'bằng khen'],
            'schedule': ['lịch', 'thời khóa biểu', 'giảng dạy', 'học'],
            'departments': ['phòng', 'khoa', 'bộ môn', 'đơn vị'],
            'facilities': ['cơ sở', 'phòng học', 'thư viện', 'lab'],
            'programs': ['ngành', 'chuyên ngành', 'chương trình', 'đào tạo'],
            'general': ['thông tin', 'hỗ trợ', 'giúp', 'hướng dẫn']
        }
        
        # ✅ EXPECTED PATTERNS cho reject/accept
        self.rejection_patterns = [
            "em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên",
            "em chỉ hỗ trợ",
            "không thể hỗ trợ",
            "ngoài phạm vi"
        ]
        
        self.clarification_patterns = [
            "để em hỗ trợ chính xác",
            "thầy/cô có thể nói rõ hơn",
            "cần làm rõ",
            "thông tin cụ thể"
        ]
        
        print("🚀 ChatBot Tester initialized!")
    
    def load_test_data(self, csv_path=None):
        """Load test data from CSV or create sample data"""
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8')
            test_queries = df['question'].fillna('').tolist()
            
            # Handle expected answers
            if 'answer' in df.columns:
                expected_answers = df['answer'].fillna('').tolist()
            else:
                expected_answers = ['']*len(test_queries)
            
            # Handle categories - fixed logic
            if 'category' in df.columns:
                categories = df['category'].fillna('general').tolist()
            else:
                categories = ['general']*len(test_queries)
                
            print(f"📊 Loaded from CSV: {len(test_queries)} queries")
        else:
            # ✅ SAMPLE TEST DATA nếu không có CSV
            test_queries, expected_answers, categories = self.generate_sample_test_data()
            print(f"📊 Using sample data: {len(test_queries)} queries")
        
        return test_queries, expected_answers, categories
    
    def generate_sample_test_data(self):
        """Generate comprehensive sample test data"""
        test_data = [
            # ✅ EDUCATION RELATED - should be accepted
            ("học phí của trường là bao nhiêu?", "", "tuition_fees"),
            ("lệ phí nhận bằng tốt nghiệp là bao nhiêu?", "", "graduation"),
            ("vậy còn lệ phí thuật lê phục vả thì sao?", "", "tuition_fees"),
            ("phí chuyển khoản là bao nhiêu?", "", "tuition_fees"),
            ("cách thức nộp học phí như thế nào?", "", "tuition_fees"),
            ("tuyển sinh năm 2024 có gì mới?", "", "admission"),
            ("điều kiện tốt nghiệp như thế nào?", "", "graduation"),
            ("thủ tục nhận bằng cần gì?", "", "graduation"),
            ("ngân hàng đề thi nộp khi nào?", "", "lecturer_tasks"),
            ("kê khai nhiệm vụ năm học ở đâu?", "", "lecturer_tasks"),
            ("tạp chí khoa học nhận bài không?", "", "academic_journal"),
            ("thi đua khen thưởng năm nay thế nào?", "", "competition_awards"),
            ("lịch giảng dạy cập nhật ở đâu?", "", "schedule"),
            ("phòng đảm bảo chất lượng ở đâu?", "", "departments"),
            ("ngành công nghệ thông tin học những gì?", "", "programs"),
            
            # ✅ VAGUE QUESTIONS - should ask for clarification
            ("thông tin gì?", "", "general"),
            ("hướng dẫn làm sao?", "", "general"),
            ("cách nào để?", "", "general"),
            ("thủ tục như thế nào?", "", "general"),
            
            # ✅ NON-EDUCATION - should be rejected (test carefully)
            ("thời tiết hôm nay thế nào?", "", "non_education"),
            ("món ăn ngon ở đâu?", "", "non_education"),
            ("cách nấu phở như thế nào?", "", "non_education"),
            
            # ✅ EDGE CASES
            ("", "", "empty"),
            ("bdu", "", "general"),
            ("đại học bình dương", "", "general"),
            ("giảng viên", "", "general"),
            ("thầy cô", "", "general"),
            
            # ✅ FOLLOW-UP QUESTIONS (test memory)
            ("còn về phí khác thì sao?", "", "tuition_fees"),
            ("vậy thêm thông tin gì nữa?", "", "general"),
            ("chi tiết hơn về vấn đề này?", "", "general"),
            
            # ✅ MIXED LANGUAGE
            ("học phí tuition fee là bao nhiêu?", "", "tuition_fees"),
            ("admission requirements điều kiện gì?", "", "admission"),
            
            # ✅ TYPOS & INFORMAL
            ("hoc phi la bao nhieu?", "", "tuition_fees"),
            ("tôt nghiep can gi?", "", "graduation"),
            ("hê thống có hỗ trợ không?", "", "general"),
        ]
        
        queries, answers, cats = zip(*test_data)
        return list(queries), list(answers), list(cats)
    
    def categorize_query(self, query):
        """Automatically categorize query based on keywords"""
        query_lower = query.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'unknown'
    
    def analyze_response(self, query, response_data):
        """Analyze chatbot response and categorize result"""
        response_text = response_data.get('response', '')
        confidence = response_data.get('confidence', 0)
        method = response_data.get('method', '')
        decision_type = response_data.get('decision_type', '')
        
        # ✅ CLASSIFY RESPONSE TYPE
        if any(pattern in response_text.lower() for pattern in self.rejection_patterns):
            response_type = 'rejected'
        elif any(pattern in response_text.lower() for pattern in self.clarification_patterns):
            response_type = 'clarification'
        elif 'em chưa có thông tin' in response_text.lower():
            response_type = 'no_info'
        elif 'dạ thầy/cô' in response_text.lower():
            response_type = 'answered'
        else:
            response_type = 'unknown'
        
        return {
            'response_type': response_type,
            'confidence': confidence,
            'method': method,
            'decision_type': decision_type,
            'response_length': len(response_text),
            'has_emoji': '🎓' in response_text or '📚' in response_text,
            'proper_addressing': 'thầy/cô' in response_text.lower()
        }
    
    def test_single_query(self, query, expected_category=None):
        """Test a single query and return detailed results"""
        print(f"\n🔍 Testing: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        start_time = time.time()
        
        try:
            # Skip empty queries
            if not query or not query.strip():
                result = {
                    'query': query,
                    'test_result': 'SKIPPED - Empty query',
                    'timestamp': datetime.now().isoformat()
                }
                print(f"   ⏭️  SKIPPED - Empty query")
                self.test_results.append(result)
                return result
            
            # Get chatbot response
            response_data = self.chatbot.process_query(query, session_id=self.session_id)
            
            # Analyze response
            analysis = self.analyze_response(query, response_data)
            
            # Categorize query
            detected_category = self.categorize_query(query)
            
            # Check if query should be education-related
            is_education_query = self.chatbot.decision_engine.is_education_related(query)
            
            result = {
                'query': query,
                'expected_category': expected_category or detected_category,
                'detected_category': detected_category,
                'is_education_query': is_education_query,
                'response': response_data.get('response', ''),
                'confidence': response_data.get('confidence', 0),
                'method': response_data.get('method', ''),
                'decision_type': response_data.get('decision_type', ''),
                'processing_time': time.time() - start_time,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # ✅ CLASSIFY SUCCESS/FAILURE
            if analysis['response_type'] == 'rejected' and is_education_query:
                result['test_result'] = 'FAILED - Education query was rejected'
                self.failed_queries.append(result)
            elif analysis['response_type'] == 'answered' and not is_education_query:
                result['test_result'] = 'WARNING - Non-education query was answered'
            elif analysis['response_type'] == 'unknown':
                result['test_result'] = 'FAILED - Unknown response type'
                self.failed_queries.append(result)
            else:
                result['test_result'] = 'PASSED'
            
            print(f"   📊 Result: {result['test_result']}")
            print(f"   📈 Confidence: {result['confidence']:.3f}")
            print(f"   🤖 Response Type: {analysis['response_type']}")
            print(f"   ⚡ Time: {result['processing_time']:.3f}s")
            
        except Exception as e:
            result = {
                'query': query,
                'test_result': f'ERROR - {str(e)}',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            print(f"   ❌ ERROR: {str(e)}")
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_test(self, csv_path=None, output_dir='test_results'):
        """Run comprehensive test suite"""
        print("🚀 Starting Comprehensive ChatBot Test...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_queries, expected_answers, categories = self.load_test_data(csv_path)
        
        print(f"📝 Loaded {len(test_queries)} test queries")
        
        # Run tests
        for i, (query, expected, category) in enumerate(zip(test_queries, expected_answers, categories)):
            print(f"\n[{i+1}/{len(test_queries)}]", end="")
            self.test_single_query(query, category)
            
            # Small delay to avoid overwhelming
            time.sleep(0.1)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(output_dir)
        
        print(f"\n✅ Test completed! Results saved to {output_dir}/")
    
    def generate_comprehensive_report(self, output_dir):
        """Generate comprehensive test report with analysis"""
        
        # ✅ 1. SUMMARY STATISTICS
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get('test_result', '').startswith('PASSED')])
        failed_tests = len([r for r in self.test_results if r.get('test_result', '').startswith('FAILED')])
        error_tests = len([r for r in self.test_results if r.get('test_result', '').startswith('ERROR')])
        warning_tests = len([r for r in self.test_results if r.get('test_result', '').startswith('WARNING')])
        
        # ✅ 2. RESPONSE TYPE ANALYSIS
        response_types = Counter([r.get('analysis', {}).get('response_type', 'unknown') for r in self.test_results])
        
        # ✅ 3. CONFIDENCE ANALYSIS
        confidences = [r.get('confidence', 0) for r in self.test_results if 'confidence' in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # ✅ 4. FAILED QUERIES ANALYSIS
        failed_education_queries = []
        rejected_education_queries = []
        
        for result in self.test_results:
            if result.get('is_education_query', False):
                analysis = result.get('analysis', {})
                if analysis.get('response_type') == 'rejected':
                    rejected_education_queries.append(result)
                elif result.get('test_result', '').startswith('FAILED'):
                    failed_education_queries.append(result)
        
        # ✅ 5. KEYWORD GAP ANALYSIS
        missing_keywords = self.analyze_missing_keywords()
        
        # ✅ 6. GENERATE REPORTS
        
        # Summary Report
        summary_report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'warnings': warning_tests,
                'pass_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'response_analysis': {
                'response_types': dict(response_types),
                'average_confidence': f"{avg_confidence:.3f}",
                'total_rejected_education_queries': len(rejected_education_queries),
                'total_failed_education_queries': len(failed_education_queries)
            },
            'keyword_gaps': missing_keywords,
            'recommendations': self.generate_recommendations()
        }
        
        # Save summary
        with open(f"{output_dir}/test_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        # Save detailed results
        df_results = pd.DataFrame(self.test_results)
        df_results.to_csv(f"{output_dir}/detailed_results.csv", encoding='utf-8', index=False)
        
        # Save failed queries for analysis
        if failed_education_queries or rejected_education_queries:
            failed_df = pd.DataFrame(rejected_education_queries + failed_education_queries)
            failed_df.to_csv(f"{output_dir}/failed_education_queries.csv", encoding='utf-8', index=False)
        
        # ✅ 7. PRINT SUMMARY TO CONSOLE
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({(passed_tests/total_tests*100):.1f}%)")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Warnings: {warning_tests}")
        print(f"🔥 Errors: {error_tests}")
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        
        print(f"\n🤖 Response Types:")
        for resp_type, count in response_types.most_common():
            print(f"   {resp_type}: {count}")
        
        print(f"\n❌ Education Queries Rejected: {len(rejected_education_queries)}")
        if rejected_education_queries:
            print("   Top rejected education queries:")
            for result in rejected_education_queries[:5]:
                print(f"   - '{result['query']}'")
        
        if missing_keywords:
            print(f"\n🔍 Missing Keywords Found: {len(missing_keywords)}")
            for keyword in missing_keywords[:10]:
                print(f"   - '{keyword}'")
        
        print(f"\n📁 Full reports saved to: {output_dir}/")
        print("="*80)
    
    def analyze_missing_keywords(self):
        """Analyze queries to find missing keywords"""
        missing_keywords = []
        
        # Analyze rejected education queries
        for result in self.test_results:
            if (result.get('is_education_query', False) and 
                result.get('analysis', {}).get('response_type') == 'rejected'):
                
                query = result['query'].lower()
                
                # Extract potential keywords
                words = re.findall(r'\b\w+\b', query)
                
                # Filter out common words
                common_words = {'là', 'của', 'có', 'thể', 'được', 'như', 'thế', 'nào', 'gì', 'sao', 'và', 'với', 'trong', 'về', 'để', 'khi', 'đã', 'sẽ', 'các', 'một', 'những'}
                
                potential_keywords = [w for w in words if len(w) > 2 and w not in common_words]
                
                # Check if any of these words are in current keywords
                current_keywords = self.chatbot.decision_engine.education_keywords
                for word in potential_keywords:
                    if word not in current_keywords and word not in missing_keywords:
                        missing_keywords.append(word)
        
        return missing_keywords[:20]  # Top 20 missing keywords
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check pass rate
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get('test_result', '').startswith('PASSED')])
        pass_rate = (passed_tests/total_tests) if total_tests > 0 else 0
        
        if pass_rate < 0.8:
            recommendations.append("🔴 Pass rate is below 80%. Consider reviewing education keyword detection logic.")
        
        # Check rejected education queries
        rejected_education = len([r for r in self.test_results 
                                if r.get('is_education_query', False) and 
                                   r.get('analysis', {}).get('response_type') == 'rejected'])
        
        if rejected_education > 0:
            recommendations.append(f"🔴 {rejected_education} education queries were rejected. Review education_keywords list.")
        
        # Check confidence levels
        low_confidence_queries = len([r for r in self.test_results if r.get('confidence', 0) < 0.3])
        
        if low_confidence_queries > total_tests * 0.3:
            recommendations.append("🟡 High number of low-confidence responses. Consider improving knowledge base or similarity thresholds.")
        
        # Check processing time
        slow_queries = len([r for r in self.test_results if r.get('processing_time', 0) > 2.0])
        
        if slow_queries > 0:
            recommendations.append(f"🟡 {slow_queries} queries took longer than 2 seconds. Consider performance optimization.")
        
        if not recommendations:
            recommendations.append("✅ All metrics look good! System is performing well.")
        
        return recommendations
    
    def test_memory_functionality(self):
        """Test conversation memory functionality"""
        print("\n🧠 Testing Memory Functionality...")
        
        memory_test_session = f"memory_test_{int(time.time())}"
        
        # Test sequence
        queries = [
            "học phí của trường là bao nhiêu?",
            "vậy còn lệ phí khác thì sao?",
            "chi tiết hơn về phí này?",
            "cảm ơn thông tin"
        ]
        
        for i, query in enumerate(queries):
            print(f"\n[Memory Test {i+1}] {query}")
            result = self.chatbot.process_query(query, session_id=memory_test_session)
            
            print(f"Response: {result['response'][:100]}...")
            print(f"Decision: {result.get('decision_type', 'unknown')}")
            
            # Check memory
            memory = self.chatbot.get_conversation_context(memory_test_session)
            print(f"Memory entries: {len(memory)}")
    
    def run_specific_category_test(self, category):
        """Test specific category of queries"""
        print(f"\n🎯 Testing {category} category...")
        
        if category in self.categories:
            keywords = self.categories[category]
            
            # Generate test queries for this category
            test_queries = [
                f"{keyword} là gì?",
                f"thông tin về {keyword}",
                f"hướng dẫn {keyword}",
                f"cách thức {keyword} như thế nào?"
            ]
            
            for query in test_queries:
                self.test_single_query(query, category)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChatBot Testing Script')
    parser.add_argument('--csv', help='Path to CSV file with test queries')
    parser.add_argument('--output', default='test_results', help='Output directory for results')
    parser.add_argument('--category', help='Test specific category only')
    parser.add_argument('--memory', action='store_true', help='Test memory functionality')
    
    args = parser.parse_args()
    
    tester = ChatBotTester()
    
    if args.memory:
        tester.test_memory_functionality()
    elif args.category:
        tester.run_specific_category_test(args.category)
    else:
        tester.run_comprehensive_test(csv_path=args.csv, output_dir=args.output)


if __name__ == "__main__":
    main()