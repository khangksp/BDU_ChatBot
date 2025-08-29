#!/usr/bin/env python3
"""
test_context.py - Comprehensive Context Memory Testing
Test interaction-based memory persistence and entity extraction
"""

import os
import sys
import django
import time
from datetime import datetime

# Setup Django environment
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from ai_models.services import chatbot_ai

class ContextMemoryTester:
    def __init__(self):
        self.session_id = f"test_context_{int(time.time())}"
        self.test_results = {}
        print(f"🧪 Context Memory Tester initialized")
        print(f"📋 Session ID: {self.session_id}")
        print(f"🕐 Test started at: {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)

    def print_separator(self, title):
        print(f"\n{'='*20} {title} {'='*20}")

    def analyze_response(self, query, response_data, expected_features=None):
        """Analyze response for context usage and memory features"""
        result = {
            "query": query,
            "response": response_data.get('response', 'NO RESPONSE'),
            "confidence": response_data.get('confidence', 0),
            "context_used": response_data.get('context_info', {}).get('context_used', False),
            "method": response_data.get('method', 'unknown'),
        }

        # Check expected features
        if expected_features:
            result["expected_check"] = {}
            for feature, expected in expected_features.items():
                actual = self._extract_feature_value(response_data, feature)
                result["expected_check"][feature] = {
                    "expected": expected,
                    "actual": actual,
                    "pass": (actual == expected),
                }

        # In chi tiết trong quá trình chạy (debug log)
        print(f"❓ Query: {query}")
        print(f"✅ Response: {result['response']}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"🔧 Method: {result['method']}")
        print(f"📊 Context used: {result['context_used']}")
        print("-" * 60)

        return result

    def _extract_feature_value(self, response_data, feature):
        """Extract specific feature value from response"""
        if feature == 'context_used':
            return response_data.get('context_info', {}).get('context_used', False)
        elif feature == 'has_context_keywords':
            return bool(response_data.get('context_info', {}).get('context_keywords', []))
        elif feature == 'confidence_above_05':
            return response_data.get('confidence', 0) > 0.5
        elif feature == 'method_contains_context':
            return 'context' in response_data.get('method', '').lower()
        return None

    def test_basic_memory(self):
        """Test 1: Basic entity establishment and immediate recall"""
        self.print_separator("TEST 1: BASIC MEMORY")
        results = []

        resp1 = chatbot_ai.process_query("thầy Tuấn là ai", session_id=self.session_id)
        results.append(self.analyze_response("thầy Tuấn là ai", resp1))

        time.sleep(1)

        resp2 = chatbot_ai.process_query("vậy thầy Tuấn làm gì", session_id=self.session_id)
        results.append(self.analyze_response("vậy thầy Tuấn làm gì", resp2, {
            'context_used': True,
            'has_context_keywords': True
        }))
        return results

    def test_memory_persistence(self):
        """Test 2: Memory persistence over multiple interactions"""
        self.print_separator("TEST 2: MEMORY PERSISTENCE")
        queries = [
            "hiệu trưởng là ai",
            "phó hiệu trưởng là ai",
            "giám đốc trung tâm CDS là ai",
            "học phí bao nhiêu",
            "thủ tục tốt nghiệp",
            "vậy hiệu trưởng tên gì",
            "còn phó hiệu trưởng thì sao",
            "thầy Tuấn ở đâu"
        ]

        results = []
        for i, query in enumerate(queries, 1):
            resp = chatbot_ai.process_query(query, session_id=self.session_id)
            expected_context = i > 5
            results.append(self.analyze_response(query, resp, {
                'context_used': expected_context
            } if expected_context else None))
            time.sleep(0.5)

        # Tóm tắt: số lần context_used thành công
        context_success = sum(1 for r in results if r['context_used'])
        return {"steps": results, "context_success": context_success, "total": len(queries)}

    def test_entity_extraction(self):
        """Test 3: Entity extraction accuracy"""
        self.print_separator("TEST 3: ENTITY EXTRACTION")
        summary = {}
        try:
            memory_data = chatbot_ai.response_generator.memory.get_conversation_context(self.session_id)
            summary = {
                "history_length": len(memory_data.get('history', [])),
                "entity_memory": len(memory_data.get('entity_memory', {})),
                "active_entities": len(memory_data.get('active_entities', [])),
                "context_keywords": memory_data.get('context_keywords', [])
            }
            print(f"📊 Current Memory State: {summary}")
        except Exception as e:
            summary["error"] = str(e)
        return summary

    def test_context_functionality(self):
        """Test 4: Context functionality methods"""
        self.print_separator("TEST 4: CONTEXT FUNCTIONALITY")
        results = {}
        try:
            if hasattr(chatbot_ai.response_generator, 'memory') and hasattr(chatbot_ai.response_generator.memory, 'entity_extractor'):
                entities = chatbot_ai.response_generator.memory.entity_extractor.extract_entities(
                    "GS.TS. Cao Việt Hiếu là Hiệu trưởng trường", 
                    "hiệu trưởng là ai"
                )
                results["entity_extractor"] = entities
                print(f"✅ Entity extractor test: {entities}")

            if hasattr(chatbot_ai.response_generator.memory, 'get_context_for_query'):
                context_info = chatbot_ai.response_generator.memory.get_context_for_query(
                    self.session_id, "vậy Cao Việt Hiếu là ai?"
                )
                results["context_query"] = context_info
                print(f"✅ Context query analysis: {context_info}")

            if hasattr(chatbot_ai.response_generator.memory, 'interaction_counter'):
                counter = chatbot_ai.response_generator.memory.interaction_counter.get(self.session_id, 0)
                results["interaction_counter"] = counter
                print(f"✅ Interaction counter: {counter}")
        except Exception as e:
            results["error"] = str(e)
        return results

    def test_long_context_sequence(self):
        """Test 5: Long context sequence (15+ interactions)"""
        self.print_separator("TEST 5: LONG CONTEXT SEQUENCE")
        long_sequence = [
            # Dãy câu hỏi mới của bạn
            "5 ông thầy",
            "thầy Tuấn là ai",
            "giám đốc trung tâm CDS là ai",
            "hiệu trưởng là ai",
            "cao việt hiếu là ai",
            "phó hiệu trưởng là ai",
            "giám đốc CDS",
            "ai là giám đốc trung tâm CDS",
            "Lê Văn Cường là ai",
            "Đỗ Đoan Trang là ai",
            # Bắt đầu các câu hỏi kiểm tra ghi nhớ
            "vậy Lê Văn Cường là ai",      # Nhớ từ câu "Lê Văn Cường là ai"
            "Cao Việt Hiếu là ai?",         # Nhớ từ câu "cao việt hiếu là ai"
            "học phí",                      # Chuyển chủ đề
            "học phí chính quy",           # Câu hỏi liên quan chủ đề mới
            "còn thầy Tuấn thì sao",       # << Quay lại kiểm tra ghi nhớ cũ
            "bà Trang làm gì ở trường"    # << Quay lại kiểm tra ghi nhớ cũ
        ]
        context_success = 0
        # Tăng số lượng bài test recall để đánh giá chính xác hơn
        recall_queries = [
            "vậy lê văn cường là ai",
            "cao việt hiếu là ai?",
            "còn thầy tuấn thì sao",
            "bà trang làm gì ở trường"
        ]
        total_recall_tests = len(recall_queries)
        results = []

        for i, query in enumerate(long_sequence, 1):
            resp = chatbot_ai.process_query(query, session_id=self.session_id)
            r = self.analyze_response(query, resp)
            results.append(r)
            # Kiểm tra chính xác hơn
            if query.lower().strip() in recall_queries and r['context_used']:
                context_success += 1
            time.sleep(0.3)

        success_rate = (context_success / total_recall_tests) * 100 if total_recall_tests > 0 else 0
        return {"steps": results, "success_rate": success_rate, "recalls": context_success, "total_recall": total_recall_tests}

    def test_memory_limits(self):
        """Test 6: Memory limits and cleanup"""
        self.print_separator("TEST 6: MEMORY LIMITS")
        summary = {}
        try:
            memory_data = chatbot_ai.response_generator.memory.get_conversation_context(self.session_id)
            summary = {
                "history_length": len(memory_data.get('history', [])),
                "entity_count": len(memory_data.get('entity_memory', {})),
                "max_history": getattr(chatbot_ai.response_generator.memory, "max_history", "unknown"),
                "interaction_count": getattr(chatbot_ai.response_generator.memory, "interaction_counter", {}).get(self.session_id, 0)
            }
            print(f"📊 Memory Statistics: {summary}")
        except Exception as e:
            summary["error"] = str(e)
        return summary

    def generate_summary_report(self):
        """Generate comprehensive test summary"""
        self.print_separator("SUMMARY REPORT")
        print("📊 Test Results Summary:")

        for test_name, results in self.test_results.items():
            print(f"\n--- {test_name.upper()} ---")
            if isinstance(results, dict) and "steps" in results:
                print(f"Steps: {len(results['steps'])}")
                if "context_success" in results:
                    print(f"Context success: {results['context_success']} / {results['total']}")
                if "success_rate" in results:
                    print(f"Recall success: {results['recalls']} / {results['total_recall']} ({results['success_rate']:.1f}%)")
            else:
                print(results)

        print("\n🎯 Tổng hợp hoàn tất. Chi tiết ở log phía trên.")

    def run_all_tests(self):
        """Run complete test suite"""
        print(f"🚀 Starting comprehensive context memory test...")
        try:
            self.test_results['basic_memory'] = self.test_basic_memory()
            self.test_results['memory_persistence'] = self.test_memory_persistence()
            self.test_results['entity_extraction'] = self.test_entity_extraction()
            self.test_results['context_functionality'] = self.test_context_functionality()
            self.test_results['long_sequence'] = self.test_long_context_sequence()
            self.test_results['memory_limits'] = self.test_memory_limits()
            self.generate_summary_report()
        finally:
            try:
                chatbot_ai.clear_conversation_memory(self.session_id)
                print(f"🧹 Test session cleaned up")
            except:
                pass

if __name__ == "__main__":
    print("🧪 BDU Chatbot Context Memory Test")
    print("="*50)
    tester = ContextMemoryTester()
    tester.run_all_tests()
