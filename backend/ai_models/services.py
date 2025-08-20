import numpy as np
import faiss
import time
import os
import re
from django.conf import settings
from knowledge.models import KnowledgeBase
import logging
from .gemini_service import GeminiResponseGenerator, SimpleVietnameseRestorer
import pandas as pd

from .external_api_service import external_api_service
from qa_management.services import drive_service
from .query_response_cache import query_response_cache

logger = logging.getLogger(__name__)

class SemanticReRanker:
    def __init__(self, retriever_service):
        # Pure semantic configuration with FIXED smart penalties
        self.retriever_service = retriever_service
        
        # FIXED re-ranking configuration with TOP-5 enhancement
        self.config = {
            'stage1_top_k': 20,      # Get more candidates for re-ranking
            'stage2_top_n': 8,       # 🔬 ENHANCED: 3 → 5 to catch more correct answers
            'cross_encoder_enabled': True,
            'semantic_weight': 0.6,   # Weight for original semantic score
            'cross_encoder_weight': 0.4,  # Weight for cross-encoder score
            'min_score_threshold': 0.1,    # Minimum score to consider
            'smart_penalty_enabled': True,  # FIXED: Smart penalty system
            'confidence_preservation': True,  # FIXED: Preserve high confidence
            'adaptive_penalty_rates': {    # FIXED: Adaptive penalty rates
                'very_high': 0.05,  # Very light penalty for very high confidence
                'high': 0.1,        # Light penalty for high confidence
                'medium': 0.15,     # Moderate penalty for medium confidence
                'low': 0.25         # Heavy penalty for low confidence
            }
        }
        
        logger.info("🎯 ENHANCED SemanticReRanker initialized with smart penalty + top-5 selection")
        logger.info(f"   📊 Stage 1: Top-{self.config['stage1_top_k']} semantic retrieval")
        logger.info(f"   🔄 Stage 2: Top-{self.config['stage2_top_n']} cross-encoder re-ranking")
        logger.info(f"   🧠 Smart penalty: {self.config['smart_penalty_enabled']}")
        logger.info(f"   🛡️ Confidence preservation: {self.config['confidence_preservation']}")
        logger.info(f"   🔬 Top-5 candidate selection enabled")

    def calculate_semantic_boost(self, candidate, query):
        """
        🎯 PURE SEMANTIC: Calculate semantic-only boost factors
        """
        boost = 0.0
        
        # Document length normalization (shorter answers might be more precise)
        answer_length = len(candidate.get('answer', ''))
        if 100 <= answer_length <= 500:  # Optimal length range
            boost += 0.05
        elif answer_length > 1000:  # Penalty for very long answers
            boost -= 0.05
        
        # Question-answer semantic coherence
        question = candidate.get('question', '')
        answer = candidate.get('answer', '')
        
        # Simple lexical overlap as semantic coherence proxy
        query_words = set(query.lower().split())
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Semantic coherence between query and question
        question_overlap = len(query_words.intersection(question_words)) / max(len(query_words), 1)
        if question_overlap > 0.3:
            boost += 0.1
        
        # Ensure boost doesn't exceed reasonable limits
        return min(0.2, boost)

    def _detect_mismatch_severity(self, candidate, query):
        """
        🔍 FIXED: Detect mismatch severity (not just binary)
        
        Returns:
            dict: {
                'concept_severity': float,  # 0.0-1.0
                'topic_severity': float,    # 0.0-1.0
                'context_severity': float,  # 0.0-1.0
                'issues': list
            }
        """
        query_lower = query.lower()
        question_lower = candidate.get('question', '').lower()
        answer_lower = candidate.get('answer', '').lower()
        
        mismatch_analysis = {
            'concept_severity': 0.0,
            'topic_severity': 0.0, 
            'context_severity': 0.0,
            'issues': []
        }
        
        # CONCEPT MISMATCH ANALYSIS (more nuanced)
        concept_conflicts = [
            # HIGH SEVERITY conflicts
            {
                'query_concepts': ['báo cáo khối lượng công việc', 'báo cáo nhiệm vụ giảng viên'],
                'wrong_concepts': ['khối lượng học tập sinh viên', 'tín chỉ sinh viên'],
                'severity': 0.8,  # High severity
                'description': 'Work reporting vs Student credit hours'
            },
            {
                'query_concepts': ['tài khoản đóng học phí', 'số tài khoản ngân hàng'],
                'wrong_concepts': ['tài khoản đăng nhập', 'tài khoản khảo sát'],
                'severity': 0.7,  # High severity
                'description': 'Bank account vs Login account'
            },
            # MEDIUM SEVERITY conflicts
            {
                'query_concepts': ['kê khai nhiệm vụ giảng viên'],
                'wrong_concepts': ['đăng ký môn học sinh viên'],
                'severity': 0.5,  # Medium severity
                'description': 'Faculty duty vs Student registration'
            },
            {
                'query_concepts': ['lịch giảng dạy giảng viên'],
                'wrong_concepts': ['lịch học sinh viên'],
                'severity': 0.4,  # Medium-low severity
                'description': 'Teaching vs Learning schedule'
            }
        ]
        
        for conflict in concept_conflicts:
            query_has = any(concept in query_lower for concept in conflict['query_concepts'])
            answer_has_wrong = any(concept in answer_lower for concept in conflict['wrong_concepts'])
            
            if query_has and answer_has_wrong:
                mismatch_analysis['concept_severity'] = max(
                    mismatch_analysis['concept_severity'], 
                    conflict['severity']
                )
                mismatch_analysis['issues'].append(f"Concept: {conflict['description']}")
        
        # TOPIC MISMATCH ANALYSIS
        topic_irrelevance = [
            {
                'query_topics': ['học phí', 'lệ phí'],
                'irrelevant_topics': ['cuộc thi', 'moswc', 'viettel', 'robot'],
                'severity': 0.9,  # Very high - completely different domain
                'description': 'Education fees vs Competition'
            },
            {
                'query_topics': ['báo cáo'],
                'irrelevant_topics': ['sinh viên tham gia cuộc thi'],
                'severity': 0.6,  # Medium-high
                'description': 'Reporting vs Student activities'
            },
            {
                'query_topics': ['tài khoản ngân hàng'],
                'irrelevant_topics': ['khảo sát đánh giá'],
                'severity': 0.7,  # High
                'description': 'Banking vs Survey system'
            }
        ]
        
        for irrelevance in topic_irrelevance:
            query_has_topic = any(topic in query_lower for topic in irrelevance['query_topics'])
            answer_has_irrelevant = any(topic in answer_lower for topic in irrelevance['irrelevant_topics'])
            
            if query_has_topic and answer_has_irrelevant:
                mismatch_analysis['topic_severity'] = max(
                    mismatch_analysis['topic_severity'],
                    irrelevance['severity']
                )
                mismatch_analysis['issues'].append(f"Topic: {irrelevance['description']}")
        
        # CONTEXT MISMATCH ANALYSIS (lighter penalties)
        context_checks = [
            {
                'query_pattern': ['giảng viên', 'cán bộ'],
                'answer_wrong': ['sinh viên chỉ', 'dành riêng sinh viên'],
                'severity': 0.3,  # Light penalty for context mismatch
                'description': 'Faculty vs Student role'
            }
        ]
        
        for check in context_checks:
            query_has_pattern = any(pattern in query_lower for pattern in check['query_pattern'])
            answer_has_wrong = any(wrong in answer_lower for wrong in check['answer_wrong'])
            
            if query_has_pattern and answer_has_wrong:
                mismatch_analysis['context_severity'] = max(
                    mismatch_analysis['context_severity'],
                    check['severity']
                )
                mismatch_analysis['issues'].append(f"Context: {check['description']}")
        
        return mismatch_analysis

    def _calculate_smart_penalty(self, candidate, query, base_semantic_score):
        if not self.config['smart_penalty_enabled']:
            return 0.0, []
            
        # Get mismatch analysis
        mismatch_analysis = self._detect_mismatch_severity(candidate, query)
        
        if not mismatch_analysis['issues']:
            return 0.0, []  # No mismatch detected
        
        # FIXED: Determine confidence tier based on base semantic score
        if base_semantic_score >= 0.8:
            confidence_tier = 'very_high'
        elif base_semantic_score >= 0.65:
            confidence_tier = 'high'
        elif base_semantic_score >= 0.45:
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'
        
        # FIXED: Get adaptive penalty rate based on confidence
        max_penalty_rate = self.config['adaptive_penalty_rates'][confidence_tier]
        
        # Calculate penalty based on severity
        concept_penalty = mismatch_analysis['concept_severity'] * max_penalty_rate * 0.6  # 60% weight
        topic_penalty = mismatch_analysis['topic_severity'] * max_penalty_rate * 0.3     # 30% weight  
        context_penalty = mismatch_analysis['context_severity'] * max_penalty_rate * 0.1 # 10% weight
        
        total_penalty = concept_penalty + topic_penalty + context_penalty
        
        # FIXED: Additional protection for very high confidence answers
        if confidence_tier == 'very_high' and self.config['confidence_preservation']:
            total_penalty = min(total_penalty, 0.08)  # Cap at 8% penalty for very high confidence
            logger.debug(f"🛡️ Confidence preservation applied: penalty capped at {total_penalty:.3f}")
        elif confidence_tier == 'high' and self.config['confidence_preservation']:
            total_penalty = min(total_penalty, 0.12)  # Cap at 12% penalty for high confidence
            logger.debug(f"🛡️ Confidence preservation applied: penalty capped at {total_penalty:.3f}")
        
        if total_penalty > 0:
            logger.debug(f"🔍 Smart penalty calculated:")
            logger.debug(f"   📊 Base score: {base_semantic_score:.3f}")
            logger.debug(f"   🎯 Confidence tier: {confidence_tier}")
            logger.debug(f"   📉 Concept penalty: {concept_penalty:.3f}")
            logger.debug(f"   📉 Topic penalty: {topic_penalty:.3f}")
            logger.debug(f"   📉 Context penalty: {context_penalty:.3f}")
            logger.debug(f"   📉 Total penalty: {total_penalty:.3f}")
            
        return total_penalty, mismatch_analysis['issues']

    def stage1_semantic_scoring(self, candidates, query):
        if not candidates:
            return []
        
        enhanced_candidates = []
        
        for candidate in candidates:
            if not candidate:
                continue
            
            # Get original semantic score from retrieval
            semantic_score = candidate.get('similarity', candidate.get('semantic_score', 0.0))
            
            # Calculate pure semantic boost
            semantic_boost = self.calculate_semantic_boost(candidate, query)
            
            # FIXED: Calculate smart penalty based on confidence
            concept_penalty, mismatch_issues = self._calculate_smart_penalty(candidate, query, semantic_score)
            
            # Final stage 1 score: semantic + boost - smart_penalty
            stage1_score = semantic_score + semantic_boost - concept_penalty
            stage1_score = max(0.0, min(1.0, stage1_score))  # Clamp to [0,1]
            
            # Create enhanced candidate
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'semantic_score': semantic_score,
                'semantic_boost': semantic_boost,
                'smart_penalty': concept_penalty,
                'mismatch_issues': mismatch_issues,
                'stage1_score': stage1_score,
                'ranking_method': 'stage1_fixed_smart_penalty'
            })
            
            enhanced_candidates.append(enhanced_candidate)
            
            logger.debug(f"🎯 FIXED Stage 1: semantic={semantic_score:.3f}, boost={semantic_boost:.3f}, penalty={concept_penalty:.3f}, final={stage1_score:.3f}")
        
        # Sort by stage1_score in descending order
        enhanced_candidates.sort(key=lambda x: x['stage1_score'], reverse=True)
        
        # Return top-k candidates for stage 2
        stage1_candidates = enhanced_candidates[:self.config['stage1_top_k']]
        
        logger.info(f"🎯 FIXED Stage 1: {len(stage1_candidates)} candidates selected for cross-encoder re-ranking")
        
        return stage1_candidates

    def stage2_cross_encoder_simulation(self, candidates, query):
        if not candidates:
            logger.info("🔄 Stage 2 skipped: No candidates available")
            return []
        
        logger.info(f"🔄 Stage 2: Cross-encoder re-ranking {len(candidates)} candidates")
        
        try:
            # Enhanced cross-encoder simulation with semantic focus
            cross_encoder_scores = self._simulate_cross_encoder_semantic(query, candidates)
            
            # Combine Stage 1 and Stage 2 scores
            final_candidates = []
            for i, candidate in enumerate(candidates):
                stage1_score = candidate.get('stage1_score', 0.0)
                stage2_score = cross_encoder_scores[i] if i < len(cross_encoder_scores) else 0.0
                
                # Weighted combination
                final_score = (
                    self.config['semantic_weight'] * stage1_score + 
                    self.config['cross_encoder_weight'] * stage2_score
                )
                
                # Ensure final_score doesn't exceed 1.0
                final_score = min(1.0, final_score)
                
                final_candidate = candidate.copy()
                final_candidate.update({
                    'stage2_score': stage2_score,
                    'final_score': final_score,
                    'ranking_method': 'stage2_fixed_smart_semantic',
                    'fixed_semantic_reranking': True
                })
                
                final_candidates.append(final_candidate)
                
                logger.debug(f"🔄 Stage 2: s1={stage1_score:.3f}, s2={stage2_score:.3f}, final={final_score:.3f}")
            
            # Sort by final_score and return top-n
            final_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"✅ Stage 2 Complete: Top-{self.config['stage2_top_n']} candidates selected")
            
            return final_candidates[:self.config['stage2_top_n']]
            
        except Exception as e:
            logger.error(f"❌ Stage 2 cross-encoder failed: {str(e)}, falling back to Stage 1 results")
            return candidates[:self.config['stage2_top_n']]

    def _simulate_cross_encoder_semantic(self, query, candidates):
        scores = []
        query_words = set(query.lower().split())

        for candidate in candidates:
            question = candidate.get('question', '').lower()
            answer = candidate.get('answer', '').lower()

            # Factor 1: Query-Question semantic overlap (increased weight)
            question_words = set(question.split())
            question_overlap = len(query_words.intersection(question_words)) / max(len(query_words), 1)

            # Factor 2: Answer completeness and relevance
            answer_words = set(answer.split())
            answer_coverage = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
            
            # Factor 3: Question-Answer semantic coherence
            qa_shared_words = len(question_words.intersection(answer_words))
            qa_coherence = qa_shared_words / max(len(question_words.union(answer_words)), 1)

            # Factor 4: Answer length optimization (not too short, not too long)
            answer_length = len(answer)
            if 100 <= answer_length <= 800:
                length_score = 1.0
            elif answer_length < 100:
                length_score = answer_length / 100.0
            else:
                length_score = max(0.5, 1000.0 / answer_length)

            # Combine factors with semantic focus
            cross_encoder_score = (
                0.4 * question_overlap +      # Primary: query-question match
                0.3 * answer_coverage +       # Secondary: answer relevance  
                0.2 * qa_coherence +          # Coherence between Q&A
                0.1 * length_score            # Length optimization
            )

            cross_encoder_score = min(1.0, cross_encoder_score)
            scores.append(cross_encoder_score)

        return scores

    def rerank(self, candidates, query=""):
        """
        🎯 FIXED MAIN METHOD: Two-stage re-ranking with smart penalty system
        """
        if not candidates:
            return []
        
        logger.info(f"🎯 Starting FIXED semantic two-stage re-ranking for {len(candidates)} candidates")
        
        # STAGE 1: FIXED semantic scoring with smart penalties
        stage1_candidates = self.stage1_semantic_scoring(candidates, query)
        
        if not stage1_candidates:
            logger.warning("⚠️ No candidates after FIXED Stage 1")
            return []
        
        # STAGE 2: Cross-encoder re-ranking
        final_candidates = self.stage2_cross_encoder_simulation(stage1_candidates, query)
        
        logger.info(f"✅ FIXED semantic two-stage re-ranking complete: {len(final_candidates)} final candidates")
        
        return final_candidates


class PureSemanticDecisionEngine:
    def __init__(self):
        # 🎯 FIXED: Balanced semantic confidence thresholds
        self.semantic_confidence_thresholds = {
            'very_high': 0.8,    # Keep original - for truly excellent matches
            'high': 0.65,        # Slightly raised - for good matches  
            'medium': 0.45,      # Slightly raised - for decent matches
            'low': 0.25,         # Kept original - for poor matches
            'very_low': 0.1      # Kept original - for very poor matches
        }
        
        # 🧠 FIXED: Smart decision factors
        self.decision_factors = {
            'preserve_high_confidence': True,     # Don't over-penalize good answers
            'mismatch_tolerance': {               # Tolerance levels by confidence
                'very_high': 0.8,  # High tolerance for high confidence
                'high': 0.6,       # Medium tolerance for good confidence
                'medium': 0.4,     # Low tolerance for medium confidence
                'low': 0.2         # Very low tolerance for poor confidence
            },
            'smart_clarification_threshold': 0.3,  # When to use smart vs generic clarification
            # 🚀 NEW: Generative response thresholds
            'generative_enabled': True,           # Enable generative responses
            'generative_confidence_threshold': 0.4,  # Below this + non-education = generative
            'generative_education_override': False   # Don't use generative for education queries
        }
        
        # External API detection keywords (kept minimal)
        self.personal_info_keywords = [
            'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
            'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
            'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
            'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi',
            'hôm nay', 'hom nay', 'ngày mai', 'ngay mai', 'tuần này', 'tuan nay'
        ]
        
        # Basic education keywords for scope checking
        self.education_keywords = [
            'học', 'trường', 'sinh viên', 'giảng viên', 'dạy', 'bdu', 'đại học',
            'ngân hàng đề thi', 'báo cáo', 'kê khai', 'tạp chí', 'nghiên cứu'
        ]
        
        # 🚀 NEW: General knowledge indicators
        self.general_knowledge_patterns = [
            # Science & Technology
            'khoa học', 'kỹ thuật', 'công nghệ', 'toán học', 'vật lý', 'hóa học',
            'sinh học', 'lập trình', 'máy tính', 'internet', 'ai', 'robot',
            
            # General knowledge
            'lịch sử', 'địa lý', 'văn học', 'thời tiết', 'ẩm thực', 'du lịch',
            'thể thao', 'âm nhạc', 'điện ảnh', 'sức khỏe', 'y tế',
            
            # Daily life
            'nấu ăn', 'công thức', 'cách làm', 'hướng dẫn', 'mẹo', 'kinh nghiệm',
            'tư vấn', 'gợi ý', 'giải thích', 'định nghĩa', 'ý nghĩa'
        ]
        
        logger.info("✅ ENHANCED PureSemanticDecisionEngine initialized with Generative Support")
        logger.info("   🎯 FIXED decision making với smart confidence preservation")
        logger.info("   🛡️ High confidence answer protection")
        logger.info("   🧠 Adaptive mismatch tolerance")
        logger.info("   🚀 NEW: Generative response for general knowledge")

    def categorize_semantic_confidence(self, final_score):
        """
        🎯 PURE SEMANTIC: Categorize confidence based only on semantic score
        """
        if final_score >= self.semantic_confidence_thresholds['very_high']:
            return 'very_high'
        elif final_score >= self.semantic_confidence_thresholds['high']:
            return 'high'
        elif final_score >= self.semantic_confidence_thresholds['medium']:
            return 'medium'
        elif final_score >= self.semantic_confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def is_education_related(self, query):
        """
        🎯 SIMPLIFIED: Basic education scope check (UNCHANGED)
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Simple keyword-based education detection
        education_found = any(kw in query_lower for kw in self.education_keywords)
        
        # Basic pattern matching for education context
        if not education_found:
            education_patterns = [
                r'(?:bdu|đại học|trường)',
                r'(?:giảng viên|thầy|cô)',
                r'(?:sinh viên|học sinh)',
                r'(?:báo cáo|kê khai)',
                r'(?:đề thi|tạp chí)'
            ]
            
            for pattern in education_patterns:
                if re.search(pattern, query_lower):
                    education_found = True
                    break
        
        logger.debug(f"🎓 Education check: '{query}' -> {education_found}")
        return education_found

    def needs_external_api(self, query, final_score=0.0):
        """
        🎯 UNCHANGED: External API detection
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Simple personal keyword detection
        needs_api = any(keyword in query_lower for keyword in self.personal_info_keywords)
        
        logger.debug(f"🌐 API check: '{query}' -> {needs_api}")
        return needs_api

    def _assess_mismatch_impact(self, best_candidate, original_score):
        """
        🧠 FIXED: Assess if mismatch issues should affect decision (UNCHANGED)
        """
        if not best_candidate:
            return False, []
        
        mismatch_issues = best_candidate.get('mismatch_issues', [])
        smart_penalty = best_candidate.get('smart_penalty', 0.0)
        
        if not mismatch_issues:
            return False, []  # No mismatch issues
        
        # FIXED: Determine confidence tier based on ORIGINAL score (before penalty)
        confidence_tier = self.categorize_semantic_confidence(original_score)
        
        # FIXED: Get mismatch tolerance for this confidence level
        tolerance = self.decision_factors['mismatch_tolerance'].get(confidence_tier, 0.5)
        
        # FIXED: Calculate mismatch severity score
        severity_score = smart_penalty / 0.3  # Normalize to 0-1 scale (max penalty is ~0.3)
        
        # FIXED: Decision logic
        should_impact_decision = severity_score > tolerance
        
        logger.debug(f"🧠 Mismatch impact assessment:")
        logger.debug(f"   📊 Original score: {original_score:.3f}")
        logger.debug(f"   🎯 Confidence tier: {confidence_tier}")
        logger.debug(f"   📉 Smart penalty: {smart_penalty:.3f}")
        logger.debug(f"   🔍 Severity score: {severity_score:.3f}")
        logger.debug(f"   🛡️ Tolerance: {tolerance:.3f}")
        logger.debug(f"   💡 Should impact decision: {should_impact_decision}")
        
        return should_impact_decision, mismatch_issues

    def _create_smart_clarification_response(self, query, mismatch_issues, session_id):
        """
        🤔 FIXED: Create smart clarification based on detected mismatches (UNCHANGED)
        """
        try:
            personal_address = "giảng viên"  # Default fallback
        except:
            personal_address = "giảng viên"
        
        # Analyze mismatch to provide specific clarification
        if any('Work reporting vs Student' in issue for issue in mismatch_issues):
            return f"""Dạ {personal_address}, em thấy câu hỏi về "báo cáo khối lượng công việc" của giảng viên, nhưng thông tin em tìm được lại về khối lượng học tập của sinh viên.

{personal_address.title()} có thể làm rõ hơn:
- {personal_address.title()} cần thông tin về báo cáo khối lượng giờ giảng của giảng viên?
- Hay về thời gian nộp báo cáo nhiệm vụ giảng dạy?
- Hoặc về quy trình báo cáo công tác của khoa/bộ môn?

Em sẽ tìm thông tin chính xác hơn khi {personal_address} làm rõ! 🎯"""

        elif any('Bank account vs Login' in issue for issue in mismatch_issues):
            return f"""Dạ {personal_address}, em hiểu {personal_address} hỏi về "số tài khoản để đóng học phí", nhưng thông tin em tìm được lại về tài khoản đăng nhập hệ thống.

{personal_address.title()} có thể xác nhận:
- {personal_address.title()} cần số tài khoản ngân hàng để chuyển tiền học phí?
- Hay cần thông tin về cách đóng học phí online?
- Hoặc về thủ tục thanh toán học phí tại trường?

Em sẽ tìm đúng thông tin {personal_address} cần! 💳"""

        elif any('Education fees vs Competition' in issue for issue in mismatch_issues):
            return f"""Dạ {personal_address}, em tìm thấy thông tin nhưng có vẻ không đúng chủ đề {personal_address} quan tâm (thông tin về cuộc thi thay vì học phí).

{personal_address.title()} có thể nói rõ hơn về:
- Loại học phí cụ thể {personal_address} cần biết?
- Phòng ban hoặc thủ tục liên quan?
- Đối tượng áp dụng?

Em sẽ tìm thông tin chính xác hơn! 🎓"""
        
        else:
            # Generic clarification
            return f"""Dạ {personal_address}, để em có thể hỗ trợ chính xác nhất, {personal_address} có thể làm rõ hơn về vấn đề cần hỗ trợ không ạ?

Em sẽ tìm thông tin phù hợp nhất cho {personal_address}! 🎯"""

    # 🚀 NEW: Generative response support methods
    def _is_general_knowledge_query(self, query):
        """
        🚀 NEW: Check if query is general knowledge that could be answered generatively
        """
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Check for general knowledge patterns
        is_general = any(pattern in query_lower for pattern in self.general_knowledge_patterns)
        
        # Check for question patterns
        question_patterns = [
            r'\b(gì|ai|nào|sao|thế nào|tại sao|vì sao|làm thế nào)\b',
            r'\b(có phải|có đúng|có nên|có thể)\b',
            r'\b(cách|phương pháp|bí quyết|mẹo)\b',
            r'\?$'  # Ends with question mark
        ]
        
        has_question_pattern = any(re.search(pattern, query_lower) for pattern in question_patterns)
        
        logger.debug(f"🧠 General knowledge check: '{query}' -> patterns={is_general}, questions={has_question_pattern}")
        
        return is_general or has_question_pattern

    def _create_generative_context(self, query, session_id):
        """
        🚀 NEW: Create context for generative response
        """
        return {
            'instruction': 'generative_answer',
            'query': query,
            'confidence': 0.3,  # Medium-low confidence for generative
            'message': 'Using generative AI for general knowledge',
            'semantic_decision': True,
            'generative_mode': True,
            'session_id': session_id
        }

    def make_decision(self, query, candidates_list, session_memory=None, jwt_token=None, document_text=None):
        """
        🎯 ENHANCED: Decision making with generative support for non-education queries
        """
        # 🎯 DOCUMENT CONTEXT PRIORITY (unchanged)
        if document_text and document_text.strip():
            logger.info("📄 DOCUMENT CONTEXT PRIORITY: Document text provided")
            return 'use_document_context', {
                'instruction': 'answer_from_document',
                'query': query,
                'document_text': document_text,
                'confidence': 0.95,
                'message': 'Answering based on document content',
                'semantic_decision': True
            }, True
        
        # 🎯 BASIC EDUCATION SCOPE CHECK
        is_education = self.is_education_related(query)
        
        # 🎯 EXTERNAL API CHECK (unchanged)
        if self.needs_external_api(query, 0.0):  # Use dummy score for API check
            if jwt_token and jwt_token.strip():
                return 'use_external_api', {
                    'instruction': 'external_api_lecturer',
                    'query': query,
                    'jwt_token': jwt_token,
                    'fallback_qa_answer': candidates_list[0].get('answer', '') if candidates_list else '',
                    'confidence': candidates_list[0].get('final_score', 0) if candidates_list else 0,
                    'message': 'Using external API for personal information',
                    'semantic_decision': True
                }, True
            else:
                return 'require_authentication', {
                    'instruction': 'authentication_required',
                    'query': query,
                    'confidence': candidates_list[0].get('final_score', 0) if candidates_list else 0,
                    'message': 'Personal information requires authentication',
                    'semantic_decision': True
                }, True
        
        # 🚀 NEW: NON-EDUCATION + FIRST MESSAGE LOGIC
        if not is_education and session_memory and len(session_memory) == 0:  # First message
            # Check if it's a general knowledge query that could be answered generatively
            if (self.decision_factors['generative_enabled'] and 
                self._is_general_knowledge_query(query)):
                
                logger.info("🚀 GENERATIVE MODE: Non-education general knowledge query")
                return 'use_generative', self._create_generative_context(query, None), True
            else:
                # Original behavior: reject non-education
                logger.info("📚 SCOPE: Rejecting non-education query")
                return 'reject_non_education', None, False
        
        # 📬 ENHANCED: Smart candidate selection from top 5 (unchanged logic)
        if not candidates_list:
            logger.warning("⚠️ No candidates provided for decision making")
            
            # 🚀 NEW: If no candidates and it's general knowledge, try generative
            if (self.decision_factors['generative_enabled'] and 
                not is_education and 
                self._is_general_knowledge_query(query)):
                
                logger.info("🚀 GENERATIVE FALLBACK: No candidates + general knowledge")
                return 'use_generative', self._create_generative_context(query, None), True
            
            return 'say_dont_know', {
                'instruction': 'dont_know_lecturer',
                'confidence': 0.0,
                'message': 'No candidates available',
                'semantic_decision': True
            }, True
        
        best_candidate = None
        best_suitability = -1
        selection_info = []
        
        if len(candidates_list) > 1:
            # 📬 SMART SELECTION: Pick best from top 5 based on suitability
            logger.info(f"📬 SMART SELECTION: Analyzing {len(candidates_list)} candidates")
            
            for i, candidate in enumerate(candidates_list[:5]):
                score = candidate.get('final_score', 0)
                mismatch_count = len(candidate.get('mismatch_issues', []))
                semantic_score = candidate.get('semantic_score', 0)
                
                # Suitability = semantic_score - mismatch_penalty + position_bonus
                position_bonus = (5 - i) * 0.01  # Small bonus for higher positions
                suitability = semantic_score - (mismatch_count * 0.1) + position_bonus
                
                selection_info.append({
                    'position': i + 1,
                    'score': score,
                    'semantic_score': semantic_score,
                    'mismatch_count': mismatch_count,
                    'suitability': suitability
                })
                
                if suitability > best_suitability:
                    best_suitability = suitability
                    best_candidate = candidate
                    
                logger.debug(f"📬 Candidate #{i+1}: score={score:.3f}, semantic={semantic_score:.3f}, mismatches={mismatch_count}, suitability={suitability:.3f}")
            
            if best_candidate:
                original_pos = None
                for info in selection_info:
                    if info['suitability'] == best_suitability:
                        original_pos = info['position']
                        break
                logger.info(f"📬 SMART SELECTION: Chose candidate #{original_pos} (suitability: {best_suitability:.3f})")
        else:
            best_candidate = candidates_list[0]
            logger.info("📬 SINGLE CANDIDATE: Using the only available candidate")
        
        # Get final score from selected candidate
        final_score = best_candidate.get('final_score', 0.0)
        original_score = best_candidate.get('semantic_score', final_score)
        
        # 🧠 FIXED: Smart mismatch assessment (unchanged)
        should_impact, mismatch_issues = self._assess_mismatch_impact(best_candidate, original_score)
        
        # 🎯 FIXED: Determine confidence level and decision
        confidence_level = self.categorize_semantic_confidence(final_score)
        
        logger.info(f"🎯 ENHANCED Semantic Decision Analysis:")
        logger.info(f"   📊 Selected candidate position: {original_pos if 'original_pos' in locals() else 1}")
        logger.info(f"   📊 Original semantic score: {original_score:.3f}")
        logger.info(f"   📊 Final score: {final_score:.3f}")
        logger.info(f"   🎯 Confidence level: {confidence_level}")
        logger.info(f"   🧠 Mismatch should impact: {should_impact}")
        logger.info(f"   🔍 Mismatch issues: {len(mismatch_issues)}")
        
        # 🚀 NEW: Generative fallback for low confidence + non-education
        if (confidence_level in ['very_low', 'low'] and 
            final_score < self.decision_factors['generative_confidence_threshold'] and
            not is_education and 
            self.decision_factors['generative_enabled'] and
            self._is_general_knowledge_query(query)):
            
            logger.info("🚀 GENERATIVE FALLBACK: Low confidence + non-education + general knowledge")
            return 'use_generative', self._create_generative_context(query, None), True
        
        # 🛡️ FIXED DECISION LOGIC: Preserve high confidence + smart mismatch handling (ALL UNCHANGED)
        
        if confidence_level == 'very_high':
            # VERY HIGH CONFIDENCE: Almost always answer, even with light mismatch
            decision = 'use_db_direct'
            context = {
                'instruction': 'direct_answer_lecturer',
                'db_answer': best_candidate.get('answer', ''),
                'confidence': final_score,
                'message': f'Very high confidence - direct answer (preserved)',
                'semantic_decision': True,
                'confidence_level': confidence_level,
                'mismatch_issues': mismatch_issues,
                'confidence_preserved': True,
                'selected_position': original_pos if 'original_pos' in locals() else 1
            }
            logger.info(f"✅ ENHANCED Decision: {decision} (very high confidence preserved)")
            
        elif confidence_level == 'high':
            # HIGH CONFIDENCE: Answer unless serious mismatch
            if should_impact and mismatch_issues:
                decision = 'ask_clarification'
                context = {
                    'instruction': 'smart_clarification_needed',
                    'db_answer': best_candidate.get('answer', ''),
                    'confidence': final_score,
                    'message': f'High confidence but serious mismatch → smart clarification',
                    'semantic_decision': True,
                    'confidence_level': confidence_level,
                    'mismatch_issues': mismatch_issues,
                    'smart_clarification': True,
                    'selected_position': original_pos if 'original_pos' in locals() else 1
                }
                logger.info(f"🤔 ENHANCED Decision: {decision} (high confidence + serious mismatch)")
            else:
                decision = 'use_db_direct'
                context = {
                    'instruction': 'direct_answer_lecturer',
                    'db_answer': best_candidate.get('answer', ''),
                    'confidence': final_score,
                    'message': f'High confidence - direct answer',
                    'semantic_decision': True,
                    'confidence_level': confidence_level,
                    'mismatch_issues': mismatch_issues,
                    'selected_position': original_pos if 'original_pos' in locals() else 1
                }
                logger.info(f"✅ ENHANCED Decision: {decision} (high confidence)")
                
        elif confidence_level == 'medium':
            # MEDIUM CONFIDENCE: Smart handling based on mismatch
            if should_impact and mismatch_issues:
                decision = 'ask_clarification'
                context = {
                    'instruction': 'smart_clarification_needed',
                    'db_answer': best_candidate.get('answer', ''),
                    'confidence': final_score,
                    'message': f'Medium confidence + mismatch → smart clarification',
                    'semantic_decision': True,
                    'confidence_level': confidence_level,
                    'mismatch_issues': mismatch_issues,
                    'smart_clarification': True,
                    'selected_position': original_pos if 'original_pos' in locals() else 1
                }
                logger.info(f"🤔 ENHANCED Decision: {decision} (medium confidence + mismatch)")
            else:
                decision = 'enhance_db_answer'
                context = {
                    'instruction': 'enhance_answer_lecturer',
                    'db_answer': best_candidate.get('answer', ''),
                    'confidence': final_score,
                    'message': 'Medium confidence - enhanced answer',
                    'semantic_decision': True,
                    'confidence_level': confidence_level,
                    'selected_position': original_pos if 'original_pos' in locals() else 1
                }
                logger.info(f"✅ ENHANCED Decision: {decision} (medium confidence)")
                
        elif confidence_level == 'low':
            # LOW CONFIDENCE: Ask clarification (smart if mismatch)
            smart_clarification = bool(mismatch_issues)
            decision = 'ask_clarification'
            context = {
                'instruction': 'smart_clarification_needed' if smart_clarification else 'clarification_needed',
                'db_answer': best_candidate.get('answer', ''),
                'confidence': final_score,
                'message': f'Low confidence - need clarification',
                'semantic_decision': True,
                'confidence_level': confidence_level,
                'mismatch_issues': mismatch_issues,
                'smart_clarification': smart_clarification,
                'selected_position': original_pos if 'original_pos' in locals() else 1
            }
            logger.info(f"🤔 ENHANCED Decision: {decision} (low confidence)")
            
        else:  # very_low
            # VERY LOW CONFIDENCE: Don't know
            decision = 'say_dont_know'
            context = {
                'instruction': 'dont_know_lecturer',
                'confidence': final_score,
                'message': f'Very low confidence - no relevant information',
                'semantic_decision': True,
                'confidence_level': confidence_level,
                'mismatch_issues': mismatch_issues,
                'selected_position': original_pos if 'original_pos' in locals() else 1
            }
            logger.info(f"❌ ENHANCED Decision: {decision} (very low confidence)")
        
        return decision, context, True


class PureSemanticChatbotAI:
    def __init__(self, shared_response_generator):
        # Import the simplified retriever service
        from .phobert_service import retriever_service
        
        # Initialize components with FIXED semantic approach
        self.sbert_retriever = ChatbotAI(shared_response_generator=shared_response_generator)
        self.retriever_service = retriever_service
        self.response_generator = shared_response_generator
        self.decision_engine = PureSemanticDecisionEngine()
        
        # 🎯 FIXED: Use SemanticReRanker with smart penalty system
        self.semantic_reranker = SemanticReRanker(retriever_service=self.retriever_service)
        
        self.conversation_memory = {}
        
        logger.info("🎯 ENHANCED PureSemanticChatbotAI initialized")
        logger.info("   🛡️ Smart penalty system enabled")
        logger.info("   🧠 Confidence-aware decision making")
        logger.info("   🎯 High-quality answer preservation")
        logger.info("   🔬 Top-5 smart candidate selection")
        logger.info("   🚀 NEW: Generative support for general knowledge")
    
    @property
    def model(self):
        return self.sbert_retriever.model
    
    @property
    def index(self):
        return self.sbert_retriever.index
    
    @property
    def knowledge_data(self):
        return self.sbert_retriever.knowledge_data
    
    def get_system_status(self):
        gemini_status = self.response_generator.get_system_status()
        drive_status = drive_service.get_system_status()
        external_api_status = external_api_service.get_system_status()
        retriever_status = self.retriever_service.get_system_status()
        
        return {
            'sbert_model': bool(self.sbert_retriever.model),
            'faiss_index': bool(self.sbert_retriever.index),
            'retriever_service_available': self.retriever_service.is_available(),
            'fine_tuned_model_available': retriever_status.get('fine_tuned_model_loaded', False),
            'gemini_available': gemini_status.get('gemini_api_available', False),
            'knowledge_entries': len(self.sbert_retriever.knowledge_data),
            'mode': 'enhanced_semantic_rag_with_generative',  # 🚀 UPDATED
            'architecture': 'enhanced_semantic_rag_with_generative',
            'semantic_reranking': {
                'enabled': True,
                'smart_penalty_system': True,
                'confidence_preservation': True,
                'adaptive_penalty_rates': True,
                'stage1_candidates': self.semantic_reranker.config['stage1_top_k'],
                'stage2_final': self.semantic_reranker.config['stage2_top_n']
            },
            'decision_engine': {
                'type': 'enhanced_semantic_with_generative',  # 🚀 UPDATED
                'confidence_thresholds': self.decision_engine.semantic_confidence_thresholds,
                'smart_mismatch_handling': True,
                'high_confidence_preservation': True,
                'adaptive_tolerance': True,
                'generative_support': self.decision_engine.decision_factors['generative_enabled']  # 🚀 NEW
            },
            'enhanced_semantic_features': [
                'smart_penalty_system',
                'confidence_preservation', 
                'adaptive_mismatch_tolerance',
                'tiered_decision_logic',
                'targeted_clarification',
                'high_quality_answer_protection',
                # 🚀 NEW: Generative features
                'generative_general_knowledge_support',
                'general_knowledge_detection',
                'context_aware_generative_responses',
                'adaptive_generative_fallback'
            ],
            'retriever_service_status': retriever_status,
            'external_api_status': external_api_status,
            'gemini_status': gemini_status
        }

    def process_query(self, query, session_id=None, jwt_token=None, document_text=None):
        start_time = time.time()
        
        logger.info(f"🎯 ENHANCED Semantic RAG Processing: '{query}' (session: {session_id}, has_token: {bool(jwt_token)}, has_document: {bool(document_text)})")
        
        try:
            # STEP 1: VALIDATE INPUT
            query = self._clean_query(query)
            if not query:
                return self._get_empty_query_response()
            
            # STEP 2: GET SESSION MEMORY
            session_memory = self.get_conversation_context(session_id) if session_id else []
            logger.info(f"🧠 Session memory: {len(session_memory)} interactions")
            
            # STEP 3: DOCUMENT CONTEXT LOG
            if document_text:
                doc_length = len(document_text.strip())
                logger.info(f"📄 Document context: {doc_length} characters")
            
            # STEP 4: PURE SEMANTIC RETRIEVAL
            candidates = self.sbert_retriever.semantic_search_top_k(query, top_k=self.semantic_reranker.config['stage1_top_k'])
            
            if not candidates:
                logger.warning("⚠️ No candidates found from semantic search")
                return self._get_no_match_response()
            
            # STEP 5: FIXED SEMANTIC RE-RANKING
            reranked_candidates = self.semantic_reranker.rerank(candidates, query)
            
            if not reranked_candidates:
                logger.warning("⚠️ No candidates after FIXED semantic re-ranking")
                return self._get_no_match_response()
            
            # 🔍 DEBUG: Log top 3 candidates for quality analysis
            logger.info(f"🔍 DEBUG - Top 3 candidates quality check:")
            for i, candidate in enumerate(reranked_candidates[:3]):
                question_preview = candidate.get('question', '')[:80]
                answer_preview = candidate.get('answer', '')[:100]
                final_score = candidate.get('final_score', 0)
                smart_penalty = candidate.get('smart_penalty', 0)
                mismatch_issues = candidate.get('mismatch_issues', [])
                logger.info(f"   #{i+1}: score={final_score:.3f}, penalty={smart_penalty:.3f}, issues={len(mismatch_issues)} | Q: '{question_preview}...' | A: '{answer_preview}...'")
            
            # STEP 6: GET BEST CANDIDATE WITH SEMANTIC SCORE
            best_candidate = reranked_candidates[0]
            final_score = best_candidate.get('final_score', 0.0)
            final_score = min(1.0, final_score)  # Ensure score cap
            
            # 🔍 DEBUG: Log FIXED candidate details
            original_semantic = best_candidate.get('semantic_score', 0)
            smart_penalty = best_candidate.get('smart_penalty', 0)
            mismatch_issues = best_candidate.get('mismatch_issues', [])
            
            logger.info(f"🎯 ENHANCED Best candidate analysis:")
            logger.info(f"   📊 Original semantic: {original_semantic:.3f}")
            logger.info(f"   📉 Smart penalty: {smart_penalty:.3f}")
            logger.info(f"   📊 Final score: {final_score:.3f}")
            logger.info(f"   🔍 Mismatch issues: {len(mismatch_issues)}")
            for issue in mismatch_issues:
                logger.info(f"     ⚠️ {issue}")
            
            # STEP 7: ENHANCED SEMANTIC DECISION MAKING with top 5 candidates + generative support
            decision_type, context, should_respond = self.decision_engine.make_decision(
                query, reranked_candidates, session_memory, jwt_token, document_text
            )
            
            # STEP 8: EXECUTE DECISION
            if not should_respond:
                response_text = self._get_out_of_scope_response(session_id)
                method = 'rejected_non_education'
            else:
                response_text = self._execute_enhanced_semantic_decision(
                    decision_type, query, context, session_id
                )
                method = decision_type
            
            # STEP 9: UPDATE MEMORY
            if session_id and should_respond:
                self._update_enhanced_semantic_memory(
                    session_id, query, final_score, decision_type, 
                    should_respond, context, document_text
                )
            
            processing_time = time.time() - start_time
            
            return {
                'response': response_text,
                'confidence': final_score,
                'method': method,
                'decision_type': decision_type,
                'semantic_info': {
                    'final_score': final_score,
                    'original_semantic_score': original_semantic,
                    'smart_penalty': smart_penalty,
                    'mismatch_issues': mismatch_issues,
                    'confidence_level': context.get('confidence_level', 'unknown') if context else 'unknown',
                    'confidence_preserved': context.get('confidence_preserved', False) if context else False,
                    'selected_position': context.get('selected_position', 1) if context else 1,
                    'semantic_decision': True,
                    'generative_mode': context.get('generative_mode', False) if context else False  # 🚀 NEW
                },
                'sources': self._format_sources(reranked_candidates[:2]),
                'processing_time': processing_time,
                'is_education': context is not None,
                'enhanced_semantic_rag': True,
                'generative_support': True,  # 🚀 NEW
                'generative_response_used': decision_type == 'use_generative',  # 🚀 NEW
                'reference_links': best_candidate.get('reference_links', []),
                'external_api_used': decision_type == 'use_external_api',
                'semantic_reranking_used': best_candidate.get('fixed_semantic_reranking', False),
                'session_memory_used': bool(session_memory),
                'document_context_used': bool(document_text),
                'document_context_priority': decision_type == 'use_document_context',
                'architecture': 'enhanced_semantic_rag_with_generative',  # 🚀 UPDATED
                'enhanced_features': ['smart_penalty', 'confidence_preservation', 'adaptive_tolerance', 'top5_selection', 'smart_candidate_selection', 'generative_support'],  # 🚀 UPDATED
                'reranking_stats': {
                    'original_semantic_score': original_semantic,
                    'semantic_boost': best_candidate.get('semantic_boost', 0),
                    'smart_penalty': smart_penalty,
                    'stage1_score': best_candidate.get('stage1_score', 0),
                    'stage2_score': best_candidate.get('stage2_score', 0),
                    'final_score': final_score,
                    'selected_position': context.get('selected_position', 1) if context else 1,
                    'top5_enhanced': True,
                    'generative_capable': True  # 🚀 NEW
                }
            }
            
        except Exception as e:
            logger.error(f"❌ ENHANCED semantic processing error: {str(e)}")
            return {
                'response': self._get_error_response(session_id),
                'confidence': 0.0,
                'method': 'error_fallback',
                'processing_time': time.time() - start_time,
                'error': str(e),
                'enhanced_semantic_rag': True,
                'generative_support': True,  # 🚀 NEW
                'graceful_degradation_used': True
            }

    def _execute_enhanced_semantic_decision(self, decision_type, query, context, session_id):
        """
        🎯 ENHANCED: Execute semantic decision with generative support
        """
        logger.info(f"🎯 Executing ENHANCED semantic decision: {decision_type}")
        
        # 🛡️ CHECK GEMINI AVAILABILITY FIRST
        gemini_available = self._check_gemini_availability()
        
        if not gemini_available:
            logger.warning("⚠️ Gemini API not available - using ENHANCED graceful degradation")
            return self._create_enhanced_semantic_fallback_response(decision_type, query, context, session_id)
        
        try:
            if decision_type == 'use_document_context':
                response = self.response_generator.generate_response(
                    query=query, context=context, intent_info=None, entities={}, session_id=session_id
                )
                response_text = response.get('response', '') if response else ''
                
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning("⚠️ Empty/invalid response from Gemini - using fallback")
                    return self._get_document_fallback(session_id)
                
                return response_text
            
            elif decision_type == 'use_external_api':
                return self._handle_external_api_decision(query, context, session_id)
            
            elif decision_type == 'require_authentication':
                return self._handle_authentication_required(session_id)
            
            # 🚀 NEW: Handle generative response
            elif decision_type == 'use_generative':
                logger.info("🚀 GENERATIVE EXECUTION: Processing general knowledge query")
                
                response = self.response_generator.generate_response(
                    query=query, context=context, intent_info=None, entities={}, session_id=session_id
                )
                response_text = response.get('response', '') if response else ''
                
                # 🛡️ CRITICAL: Validate generative response and fallback if needed
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning("⚠️ Empty/invalid generative response from Gemini - using enhanced fallback")
                    return self._create_generative_fallback_response(query, context, session_id)
                
                # 🚀 ENHANCED: Additional validation for generative responses
                if self._is_generative_response_appropriate(response_text, query, session_id):
                    return response_text
                else:
                    logger.warning("⚠️ Generative response failed appropriateness check - using fallback")
                    return self._create_generative_fallback_response(query, context, session_id)
            
            elif decision_type in ['use_db_direct', 'enhance_db_answer']:
                response = self.response_generator.generate_response(
                    query=query, context=context, intent_info=None, entities={}, session_id=session_id
                )
                response_text = response.get('response', '') if response else ''
                
                # 🛡️ CRITICAL: Validate response and fallback if needed
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning("⚠️ Empty/invalid response from Gemini - using ENHANCED semantic fallback")
                    return self._create_enhanced_semantic_fallback_response(decision_type, query, context, session_id)
                
                return response_text
            
            elif decision_type == 'ask_clarification':
                # 🤔 ENHANCED: Check if smart clarification is needed
                if context and context.get('smart_clarification', False):
                    logger.info("🤔 Creating ENHANCED smart clarification response")
                    mismatch_issues = context.get('mismatch_issues', [])
                    return self.decision_engine._create_smart_clarification_response(
                        query, mismatch_issues, session_id
                    )
                else:
                    # Standard clarification via Gemini
                    response = self.response_generator.generate_response(
                        query=query, context=context, intent_info=None, entities={}, session_id=session_id
                    )
                    response_text = response.get('response', '') if response else ''
                    
                    if not response_text or len(response_text.strip()) < 10:
                        return self._get_clarification_fallback(session_id)
                    
                    return response_text
            
            elif decision_type == 'say_dont_know':
                response = self.response_generator.generate_response(
                    query=query, context=context, intent_info=None, entities={}, session_id=session_id
                )
                response_text = response.get('response', '') if response else ''
                
                if not response_text or len(response_text.strip()) < 10:
                    return self._get_dont_know_fallback(session_id)
                
                return response_text
            
            else:
                logger.warning(f"⚠️ Unknown decision type: {decision_type}")
                return self._create_enhanced_semantic_fallback_response(decision_type, query, context, session_id)
                
        except Exception as e:
            logger.error(f"❌ Error executing ENHANCED semantic decision: {str(e)}")
            return self._create_enhanced_semantic_fallback_response(decision_type, query, context, session_id)

    # 🚀 NEW: Generative response validation and fallback methods

    def _is_generative_response_appropriate(self, response_text, query, session_id):
        """
        🚀 NEW: Check if generative response is appropriate and well-formed
        """
        if not response_text or len(response_text.strip()) < 20:
            return False
        
        personal_address = self._get_personal_address(session_id)
        
        # Check proper greeting
        if not response_text.lower().startswith(f'dạ {personal_address.lower()}'):
            logger.debug("🚀 Generative validation: Missing proper greeting")
            return False
        
        # Check for BDU context mention
        bdu_indicators = [
            'bdu', 'đại học bình dương', 'không phải chuyên môn', 
            'ngoài phạm vi', 'kiến thức tổng quát', 'hỗ trợ.*bdu'
        ]
        
        has_bdu_context = any(indicator in response_text.lower() for indicator in bdu_indicators)
        
        if not has_bdu_context:
            logger.debug("🚀 Generative validation: Missing BDU context")
            return False
        
        # Check response isn't just an apology/disclaimer
        negative_patterns = [
            r'không thể trả lời', r'không biết', r'xin lỗi.*không',
            r'không có thông tin', r'không thể hỗ trợ'
        ]
        
        is_just_negative = any(re.search(pattern, response_text.lower()) for pattern in negative_patterns)
        
        if is_just_negative and len(response_text) < 200:  # Short negative response
            logger.debug("🚀 Generative validation: Too negative/short")
            return False
        
        return True

    def _create_generative_fallback_response(self, query, context, session_id):
        """
        🚀 NEW: Create fallback response for failed generative attempts
        """
        personal_address = self._get_personal_address(session_id)
        
        # Analyze query type for better fallback
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['là gì', 'định nghĩa', 'giải thích']):
            # Definition/explanation query
            return f"""Dạ {personal_address}, em hiểu {personal_address} muốn tìm hiểu về vấn đề này. Tuy nhiên, đây không phải chuyên môn về BDU của em.

Em khuyến khích {personal_address} tham khảo các nguồn tài liệu chuyên môn hoặc hỏi các chuyên gia trong lĩnh vực đó để có thông tin chính xác nhất.

Để được hỗ trợ tốt nhất, {personal_address} có thể hỏi em về các vấn đề liên quan đến công việc và hoạt động tại Đại học Bình Dương ạ! 🎓"""

        elif any(word in query_lower for word in ['cách', 'làm thế nào', 'hướng dẫn']):
            # How-to query
            return f"""Dạ {personal_address}, em nhận thấy {personal_address} cần hướng dẫn về vấn đề này. Tuy nhiên, đây không phải lĩnh vực chuyên môn BDU của em.

Em gợi ý {personal_address} tìm hiểu từ các nguồn hướng dẫn chuyên môn hoặc tham khảo ý kiến của các chuyên gia có kinh nghiệm trong lĩnh vực đó.

Để được hỗ trợ tốt nhất, {personal_address} có thể hỏi em về các quy trình, thủ tục và hoạt động tại Đại học Bình Dương ạ! 🎓"""

        elif '?' in query or any(word in query_lower for word in ['ai', 'gì', 'nào', 'sao']):
            # General question
            return f"""Dạ {personal_address}, em hiểu {personal_address} quan tâm đến câu hỏi này. Tuy nhiên, đây không phải phạm vi chuyên môn về BDU của em.

Em khuyến khích {personal_address} tìm hiểu từ các nguồn thông tin uy tín hoặc chuyên gia trong lĩnh vực liên quan để có câu trả lời chính xác nhất.

Để được hỗ trợ tốt nhất, {personal_address} có thể hỏi em về các vấn đề liên quan đến công việc giảng viên và hoạt động tại Đại học Bình Dương ạ! 🎓"""

        else:
            # Generic fallback
            return f"""Dạ {personal_address}, em nhận thấy câu hỏi này nằm ngoài phạm vi chuyên môn BDU của em.

Để được hỗ trợ tốt nhất, {personal_address} có thể hỏi em về các vấn đề liên quan đến công việc và hoạt động tại Đại học Bình Dương ạ! 🎓"""

    def _create_enhanced_semantic_fallback_response(self, decision_type, query, context, session_id):
        """
        🛡️ ENHANCED: Create semantic fallback response with generative support
        """
        personal_address = self._get_personal_address(session_id)
        
        # Get information from context
        raw_answer = context.get('db_answer', '') if context else ''
        mismatch_issues = context.get('mismatch_issues', []) if context else []
        confidence_preserved = context.get('confidence_preserved', False) if context else False
        generative_mode = context.get('generative_mode', False) if context else False
        
        # 🚀 NEW: Handle generative fallback
        if decision_type == 'use_generative' or generative_mode:
            return self._create_generative_fallback_response(query, context, session_id)
        
        # 🤔 ENHANCED: If there are mismatch issues, provide smart clarification
        if mismatch_issues and decision_type in ['use_db_direct', 'enhance_db_answer', 'ask_clarification']:
            logger.info("🤔 ENHANCED fallback: Using smart clarification due to detected mismatches")
            return self.decision_engine._create_smart_clarification_response(
                query, mismatch_issues, session_id
            )
        
        if decision_type in ['use_db_direct', 'enhance_db_answer']:
            if raw_answer and raw_answer.strip():
                # 🔍 DEBUG: Log raw answer for analysis
                logger.info(f"🔍 DEBUG - Raw database answer: '{raw_answer[:300]}...'")
                
                # Format raw database answer with minimal enhancement
                clean_answer = raw_answer.strip()
                
                # Remove any existing personalized parts to avoid duplication
                clean_answer = re.sub(r'^(dạ\s+(thầy|cô|giảng viên)[^,]*,?\s*)', '', clean_answer, flags=re.IGNORECASE)
                clean_answer = re.sub(r'^(xin chào|chào)[^.!?]*[.!?]\s*', '', clean_answer, flags=re.IGNORECASE)
                
                # Ensure it starts with capital letter
                if clean_answer and not clean_answer[0].isupper():
                    clean_answer = clean_answer[0].upper() + clean_answer[1:]
                
                # Add personalized greeting
                personalized_response = f"Dạ {personal_address}, {clean_answer}"
                
                # Ensure proper ending
                if not personalized_response.strip().endswith(('?', '!', '.')):
                    personalized_response += '.'
                
                # Add ENHANCED closing
                if confidence_preserved:
                    personalized_response += f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ? 🎯'
                else:
                    personalized_response += f' {personal_address.title()} cần em làm rõ thêm gì không ạ? 🎯'
                
                logger.info(f"🛡️ ENHANCED SEMANTIC FALLBACK: Formatted raw answer for {personal_address}")
                return personalized_response
            else:
                return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯"
        
        # Other fallback cases...
        return f"Dạ {personal_address}, em sẵn sàng hỗ trợ {personal_address} về các vấn đề liên quan đến BDU. {personal_address.title()} có thể chia sẻ cụ thể hơn về điều cần hỗ trợ không ạ? 🎯"

    def _check_gemini_availability(self):
        """🛡️ Check if Gemini API is available"""
        try:
            if not self.response_generator:
                return False
            
            if not hasattr(self.response_generator, 'key_manager') or not self.response_generator.key_manager.keys:
                return False
            
            test_key = self.response_generator.key_manager.get_key()
            if not test_key:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking Gemini availability: {str(e)}")
            return False

    def _clean_query(self, query):
        """Clean and prepare query"""
        if not query:
            return ""
        
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[?]{2,}', '?', query)
        query = re.sub(r'[!]{2,}', '!', query)
        
        return query

    def _update_enhanced_semantic_memory(self, session_id, query, final_score, decision_type, was_education, context, document_text):
        """
        🧠 ENHANCED: Update conversation memory with semantic + generative info
        """
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        interaction = {
            'query': query,
            'semantic_info': {
                'final_score': final_score,
                'confidence_level': context.get('confidence_level', 'unknown') if context else 'unknown',
                'confidence_preserved': context.get('confidence_preserved', False) if context else False,
                'smart_penalty': context.get('smart_penalty', 0) if context else 0,
                'mismatch_issues': context.get('mismatch_issues', []) if context else [],
                'semantic_decision': True,
                'generative_mode': context.get('generative_mode', False) if context else False  # 🚀 NEW
            },
            'timestamp': time.time(),
            'user_type': 'lecturer',
            'decision_type': decision_type,
            'was_education_related': was_education,
            'enhanced_semantic_processed': True,  # 🚀 UPDATED
            'document_context_used': bool(document_text),
            'document_context_priority': decision_type == 'use_document_context',
            'external_api_used': decision_type == 'use_external_api',
            'generative_response_used': decision_type == 'use_generative',  # 🚀 NEW
            'query_length': len(query.split()),
            'architecture': 'enhanced_semantic_rag_with_generative'  # 🚀 UPDATED
        }
        
        self.conversation_memory[session_id].append(interaction)
        
        # Keep only recent history
        self.conversation_memory[session_id] = self.conversation_memory[session_id][-15:]
        
        logger.info(f"🧠 ENHANCED semantic memory updated for session {session_id}: {len(self.conversation_memory[session_id])} interactions")

    # Helper methods with ENHANCED personal addressing
    def _get_personal_address(self, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                return self.response_generator._get_personal_address(session_id)
            return "giảng viên"
        except Exception as e:
            logger.error(f"❌ Error getting personal address: {str(e)}")
            return "giảng viên"

    def _get_empty_query_response(self):
        return {
            'response': "Dạ chào giảng viên! Em có thể hỗ trợ gì cho giảng viên về công việc tại BDU ạ? 🎯",
            'confidence': 0.9,
            'method': 'empty_query',
            'processing_time': 0.01,
            'enhanced_semantic_rag': True,
            'generative_support': True  # 🚀 NEW
        }

    def _get_no_match_response(self):
        return {
            'response': "Dạ giảng viên, em chưa có thông tin về vấn đề này. Giảng viên có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯",
            'confidence': 0.1,
            'method': 'no_match_semantic',
            'decision_type': 'say_dont_know',
            'processing_time': 0.01,
            'enhanced_semantic_rag': True,
            'generative_support': True  # 🚀 NEW
        }

    def _get_out_of_scope_response(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎯"

    def _get_error_response(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em gặp khó khăn kỹ thuật. {personal_address.title()} có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎯"

    def _get_clarification_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎯"

    def _get_dont_know_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯"

    def _get_document_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em đã xem xét tài liệu nhưng gặp khó khăn trong việc trả lời. {personal_address.title()} có thể đặt câu hỏi cụ thể hơn không ạ? 🎯"

    def _handle_external_api_decision(self, query, context, session_id):
        """Handle external API call"""
        try:
            jwt_token = context.get('jwt_token')
            api_result = external_api_service.get_lecturer_schedule(jwt_token, query)
            
            if api_result.get('success'):
                enhanced_context = {
                    'instruction': 'process_external_api_data',
                    'api_data': api_result,
                    'original_query': query,
                    'fallback_qa_answer': context.get('fallback_qa_answer', ''),
                    'confidence': context.get('confidence', 0)
                }
                
                response = self.response_generator.generate_response(
                    query=query, context=enhanced_context, intent_info=None, entities={}, session_id=session_id
                )
                
                return response.get('response', self._get_api_fallback(api_result, session_id))
            else:
                return self._get_api_error_response(api_result, session_id)
                
        except Exception as e:
            logger.error(f"❌ Error handling external API: {str(e)}")
            return self._get_api_error_fallback(session_id)

    def _handle_authentication_required(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng trước ạ. 🔐"

    def _get_api_fallback(self, api_result, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em đã tìm thấy thông tin lịch giảng dạy nhưng gặp khó khăn trong việc trình bày chi tiết. {personal_address.title()} có thể truy cập hệ thống quản lý đào tạo để xem thông tin đầy đủ ạ. 🎯"

    def _get_api_error_response(self, api_result, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em gặp khó khăn khi truy xuất thông tin cá nhân. {personal_address.title()} có thể thử lại sau hoặc liên hệ bộ phận IT để được hỗ trợ ạ. 🎯"

    def _get_api_error_fallback(self, session_id):
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em gặp khó khăn kỹ thuật khi truy xuất thông tin cá nhân. {personal_address.title()} có thể thử lại sau ạ. 🎯"

    def _format_sources(self, results):
        """Format sources for display with ENHANCED semantic scores"""
        sources = []
        for result in results:
            if result and result.get('final_score', 0) > 0.2:
                sources.append({
                    'question': result['question'],
                    'category': result.get('category', 'Giảng viên'),
                    'final_score': result.get('final_score', 0),
                    'original_semantic_score': result.get('semantic_score', 0),
                    'smart_penalty': result.get('smart_penalty', 0),
                    'stage1_score': result.get('stage1_score', 0),
                    'stage2_score': result.get('stage2_score', 0),
                    'mismatch_issues': result.get('mismatch_issues', []),
                    'enhanced_semantic_reranking': result.get('fixed_semantic_reranking', False),  # 🚀 UPDATED
                    'generative_capable': True  # 🚀 NEW
                })
        return sources

    # Delegate methods
    def get_conversation_context(self, session_id):
        return self.conversation_memory.get(session_id, [])

    def get_conversation_memory(self, session_id):
        return self.response_generator.get_conversation_memory(session_id)

    def clear_conversation_memory(self, session_id=None):
        if session_id:
            self.response_generator.clear_conversation_memory(session_id)
            if session_id in self.conversation_memory:
                del self.conversation_memory[session_id]
        else:
            self.response_generator.clear_conversation_memory()
            self.conversation_memory.clear()

    def reload_after_qa_update(self):
        logger.info("🔄 Reloading ENHANCED semantic knowledge base...")
        
        if hasattr(self.sbert_retriever, 'cached_data'):
            self.sbert_retriever.cached_data = None
            self.sbert_retriever.cache_timestamp = 0
        
        self.sbert_retriever.load_knowledge_base()
        
        if self.sbert_retriever.model and self.sbert_retriever.knowledge_data:
            self.sbert_retriever.build_faiss_index()
        
        logger.info("✅ ENHANCED semantic knowledge base reloaded successfully")


class ChatbotAI:
    def __init__(self, shared_response_generator):
        self.model = None
        self.index = None
        self.knowledge_data = []
        self.vietnamese_restorer = shared_response_generator.vietnamese_restorer
        self.link_mapping = {}
        self.cached_data = None
        self.cache_timestamp = 0
        
        self.load_models()

    def load_models(self):
        """Load AI models and knowledge base"""
        try:
            from sentence_transformers import SentenceTransformer
            # Load fine-tuned model if available
            fine_tuned_path = os.path.join(settings.BASE_DIR, 'fine_tuned_phobert')
            if os.path.exists(fine_tuned_path):
                self.model = SentenceTransformer(fine_tuned_path)
                logger.info("✅ Fine-tuned SBERT loaded from: fine_tuned_phobert")
            else:
                # Fallback to base model
                self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
                logger.info("✅ Base Vietnamese SBERT loaded")
            
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model = None

    def load_link_mapping(self):
        """Load link mapping"""
        try:
            link_csv_path = os.path.join(settings.BASE_DIR, 'data', 'link.csv')
            if os.path.exists(link_csv_path):
                df_links = pd.read_csv(link_csv_path, encoding='utf-8')
                for index, row in df_links.iterrows():
                    stt = str(row['STT']).strip()
                    link = str(row['Link']).strip()
                    if stt and link and stt != 'nan' and link != 'nan':
                        self.link_mapping[stt] = link
                logger.info(f"✅ Loaded {len(self.link_mapping)} reference links")
        except Exception as e:
            logger.error(f"❌ Error loading link mapping: {str(e)}")
            self.link_mapping = {}

    def get_reference_links(self, qa_item):
        """Get reference links"""
        reference_links = []
        stt_value = qa_item.get('STT', '')
        
        if not stt_value:
            return reference_links
        
        stt_list = []
        if isinstance(stt_value, str):
            stt_parts = re.split(r'[,;\s]+', stt_value.strip())
            stt_list = [part.strip() for part in stt_parts if part.strip()]
        else:
            stt_list = [str(stt_value).strip()]
        
        for stt in stt_list:
            if stt in self.link_mapping:
                link_url = self.link_mapping[stt]
                reference_links.append({
                    'stt': stt,
                    'url': link_url,
                    'title': f"Tài liệu tham khảo {stt}"
                })
        
        return reference_links
    
    def load_knowledge_base(self):
        """Load knowledge base with auto-keywords"""
        try:
            self.load_link_mapping()
            
            # Load from QA Management database
            db_qa_entries = []
            try:
                from qa_management.models import QAEntry
                qa_entries = QAEntry.objects.filter(is_active=True).order_by('stt')
                
                for entry in qa_entries:
                    db_qa_entries.append({
                        'question': entry.question,
                        'answer': entry.answer,
                        'category': entry.category or 'Giảng viên',
                        'STT': entry.stt
                    })
                logger.info(f"✅ Loaded {len(db_qa_entries)} entries from QA Management database")
            except Exception as e:
                logger.warning(f"⚠️ QA Management not available: {str(e)}")
            
            # Load from CSV files
            csv_knowledge = []
            try:
                drive_data = drive_service.get_csv_data()
                if drive_data:
                    csv_knowledge = drive_data
                    logger.info(f"✅ Loaded {len(csv_knowledge)} records from Google Drive")
            except Exception as e:
                logger.error(f"❌ Failed to load from Google Drive: {str(e)}")
            
            # Fallback to local CSV
            if not csv_knowledge and not db_qa_entries:
                csv_path = os.path.join(settings.BASE_DIR, 'data', 'QA.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8')
                        for index, row in df.iterrows():
                            if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
                                continue
                            csv_knowledge.append({
                                'question': str(row['question']),
                                'answer': str(row['answer']),
                                'category': str(row.get('category', 'Chung')),
                                'STT': str(row.get('STT', ''))
                            })
                        logger.info(f"✅ Fallback: Loaded {len(csv_knowledge)} records from local CSV")
                    except Exception as e:
                        logger.error(f"❌ Fallback CSV also failed: {str(e)}")
            
            # Load from legacy database
            db_knowledge = list(KnowledgeBase.objects.filter(is_active=True).values(
                'question', 'answer', 'category'
            ))
            
            # Combine all sources
            self.knowledge_data = db_qa_entries + csv_knowledge + db_knowledge
            
            # Build FAISS index
            if self.model and self.knowledge_data:
                self.build_faiss_index()
            
            logger.info(f"✅ ENHANCED semantic knowledge base loaded: {len(self.knowledge_data)} entries")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            self.knowledge_data = []

    def build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        try:
            questions = [item['question'] for item in self.knowledge_data]
            embeddings = self.model.encode(questions)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index built with {len(questions)} entries")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.index = None

    def semantic_search_top_k(self, query, top_k=20):
        """Semantic search with fine-tuned model"""
        try:
            if not self.model or not self.index:
                logger.warning("⚠️ Model or index not available")
                return []
            
            # Restore Vietnamese if needed
            if self.vietnamese_restorer and not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Using restored query: '{query}' -> '{restored_query}'")
                    query = restored_query
            
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.knowledge_data)))
            
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_data) and score > 0.1:
                    candidate = self.knowledge_data[idx].copy()
                    candidate['semantic_score'] = float(score)
                    candidate['similarity'] = float(score)
                    candidate['reference_links'] = self.get_reference_links(candidate)
                    candidates.append(candidate)
            
            logger.info(f"🔍 Semantic search found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []


class BDUChatbotService:
    def __init__(self):
        # Create shared response_generator
        self.response_generator = GeminiResponseGenerator()
        
        # Initialize conversation memory
        self.query_cache = query_response_cache
        
        # 🎯 ENHANCED: Use PureSemanticChatbotAI with GENERATIVE logic
        self.semantic_chatbot = PureSemanticChatbotAI(shared_response_generator=self.response_generator)
        
        # Keywords phát hiện thông tin cá nhân
        self.personal_info_keywords = [
            # Từ khóa cốt lõi
            'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi',
            'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
            'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
            'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
            
            # Từ khóa thời gian
            'hôm nay', 'hom nay', 'ngày mai', 'ngay mai', 
            'tuần này', 'tuan nay', 'tuần sau', 'tuan sau', 'tuần tới', 'tuan toi',
            'tháng này', 'thang nay', 'tháng sau', 'thang sau'
        ]
        
        logger.info("🎯 ENHANCED BDUChatbotService initialized with Generative General Knowledge Support")
        logger.info("   🚀 NEW: Generative responses for non-education queries")
        logger.info("   🧠 ENHANCED: Context-aware general knowledge handling")
        logger.info("   🛡️ PRESERVED: All existing education-focused functionality")

    def _needs_external_api(self, query: str) -> bool:
        """
        🎯 SIMPLIFIED: Basic personal info detection for API calls
        """
        if not query:
            return False
        
        query_lower = query.lower()
        needs_api = any(keyword in query_lower for keyword in self.personal_info_keywords)
        
        logger.debug(f"🌐 API check: '{query}' -> {needs_api}")
        return needs_api

    def process_query(self, query: str, session_id: str = None, jwt_token: str = None, document_text: str = None) -> dict:
        start_time = time.time()
        
        logger.info(f"🎯 ENHANCED BDU Service Processing: '{query}' (session: {session_id}, has_token: {bool(jwt_token)}, has_document: {bool(document_text)})")
        
        try:
            if not query or len(query.strip()) < 2:
                # Get personal address for empty query response
                try:
                    if hasattr(self.response_generator, '_get_personal_address') and session_id:
                        personal_address = self.response_generator._get_personal_address(session_id)
                        response_text = f"Dạ chào {personal_address}! Em có thể hỗ trợ gì cho {personal_address} về công việc tại BDU ạ? 🎯"
                    else:
                        response_text = "Dạ chào giảng viên! Em có thể hỗ trợ gì cho giảng viên về công việc tại BDU ạ? 🎯"
                except:
                    response_text = "Dạ chào giảng viên! Em có thể hỗ trợ gì cho giảng viên về công việc tại BDU ạ? 🎯"
                    
                return {
                    'response': response_text,
                    'confidence': 0.9,
                    'method': 'empty_query',
                    'processing_time': time.time() - start_time,
                    'enhanced_semantic_rag': True,
                    'generative_support': True,  # 🚀 NEW
                    'cache_hit': False
                }
            
            # 🎯 CACHE CHECK
            cached_response = self.query_cache.get(query)
            if cached_response:
                cached_response['processing_time'] = time.time() - start_time
                # 🚀 NEW: Add generative support flag to cached responses
                cached_response['generative_support'] = True
                logger.info(f"⚡ CACHE HIT: Response served in {cached_response['processing_time']:.3f}s")
                return cached_response
            
            logger.info("💨 CACHE MISS: Proceeding with ENHANCED semantic processing")
            
            # 🎯 SIMPLIFIED API PRIORITY CHECK
            if self._needs_external_api(query):
                logger.info("🚨 API PRIORITY: Personal info query detected")
                
                if jwt_token and jwt_token.strip():
                    # Has token -> Call external API
                    api_result = self._handle_external_api_call(query, session_id, jwt_token)
                    api_result['cache_hit'] = False
                    api_result['cache_skipped'] = 'personal_query'
                    api_result['generative_support'] = True  # 🚀 NEW
                    return api_result
                else:
                    # No token -> Require authentication
                    auth_result = self._handle_authentication_required(session_id)
                    auth_result['cache_hit'] = False
                    auth_result['cache_skipped'] = 'authentication_required'
                    auth_result['generative_support'] = True  # 🚀 NEW
                    return auth_result
            
            # 🚀 ENHANCED: Generative-capable semantic RAG processing
            logger.info("📚 Using ENHANCED Semantic RAG System with Generative General Knowledge Support")
            result = self.semantic_chatbot.process_query(query, session_id, jwt_token, document_text)
            
            # 🚀 NEW: Add generative support flags
            result['api_priority_activated'] = False
            result['fallback_to_enhanced_semantic'] = True
            result['cache_hit'] = False
            result['generative_support'] = True  # 🚀 NEW
            result['generative_capability'] = True  # 🚀 NEW
            
            # 🚀 NEW: Track if generative was actually used
            if result.get('decision_type') == 'use_generative':
                result['generative_response_used'] = True
                result['generative_mode_activated'] = True
                logger.info("🚀 GENERATIVE MODE: Successfully used generative response for non-education query")
            else:
                result['generative_response_used'] = False
                result['generative_mode_activated'] = False
            
            # 🎯 CACHE STORE (with generative info)
            cache_stored = self.query_cache.set(query, result)
            result['cache_stored'] = cache_stored
            
            if cache_stored:
                logger.info(f"💾 Response cached for future requests (generative support included)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ ENHANCED BDU Service Error: {str(e)}")
            
            # Get personal address for error response
            try:
                if hasattr(self.response_generator, '_get_personal_address'):
                    personal_address = self.response_generator._get_personal_address(session_id)
                else:
                    personal_address = "giảng viên"
            except:
                personal_address = "giảng viên"
                
            return {
                'response': f"Dạ {personal_address}, em gặp khó khăn kỹ thuật. {personal_address.title()} có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎯",
                'confidence': 0.0,
                'method': 'service_error',
                'processing_time': time.time() - start_time,
                'error': str(e),
                'enhanced_semantic_rag': True,
                'generative_support': True,  # 🚀 NEW
                'graceful_degradation_used': True,
                'cache_hit': False,
                'cache_stored': False
            }

    def _handle_external_api_call(self, query: str, session_id: str, jwt_token: str) -> dict:
        """Handle external API call for personal info with generative support"""
        try:
            logger.info("🌐 Calling external API for personal information")
            
            api_result = external_api_service.get_lecturer_schedule(jwt_token, query)
            
            if api_result.get('success'):
                enhanced_context = {
                    'instruction': 'process_external_api_data',
                    'api_data': api_result,
                    'original_query': query,
                    'confidence': 0.95
                }
                
                response = self.response_generator.generate_response(
                    query=query,
                    context=enhanced_context,
                    intent_info=None,
                    entities={},
                    session_id=session_id
                )
                
                return {
                    'response': response.get('response', self._get_api_fallback(session_id)),
                    'confidence': 0.95,
                    'method': 'external_api_success',
                    'decision_type': 'use_external_api',
                    'processing_time': 0.5,
                    'external_api_used': True,
                    'api_priority_activated': True,
                    'enhanced_semantic_rag': True,
                    'generative_support': True  # 🚀 NEW
                }
            else:
                error_type = api_result.get('error_type', 'unknown')
                return {
                    'response': self._get_api_error_response(error_type, session_id),
                    'confidence': 0.1,
                    'method': 'external_api_failed',
                    'decision_type': 'api_error',
                    'processing_time': 0.3,
                    'external_api_used': True,
                    'api_error': api_result.get('error', ''),
                    'api_priority_activated': True,
                    'graceful_degradation_used': True,
                    'generative_support': True  # 🚀 NEW
                }
                
        except Exception as e:
            logger.error(f"❌ Error in external API call: {str(e)}")
            
            # Get personal address for error response
            try:
                if hasattr(self.response_generator, '_get_personal_address'):
                    personal_address = self.response_generator._get_personal_address(session_id)
                else:
                    personal_address = "giảng viên"
            except:
                personal_address = "giảng viên"
                
            return {
                'response': f"Dạ {personal_address}, em gặp khó khăn khi truy xuất thông tin cá nhân. {personal_address.title()} có thể thử lại sau ạ. 🎯",
                'confidence': 0.1,
                'method': 'external_api_error',
                'processing_time': 0.2,
                'error': str(e),
                'api_priority_activated': True,
                'graceful_degradation_used': True,
                'generative_support': True  # 🚀 NEW
            }

    def _handle_authentication_required(self, session_id: str) -> dict:
        """Handle authentication required with generative support"""
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = "giảng viên"
        except:
            personal_address = "giảng viên"
            
        return {
            'response': f"Dạ {personal_address}, để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng trước ạ. 🔐",
            'confidence': 0.9,
            'method': 'authentication_required',
            'decision_type': 'require_authentication',
            'processing_time': 0.01,
            'external_api_used': False,
            'api_priority_activated': True,
            'authentication_required': True,
            'generative_support': True  # 🚀 NEW
        }

    def _get_api_fallback(self, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = "giảng viên"
        except:
            personal_address = "giảng viên"
            
        return f"Dạ {personal_address}, em đã tìm thấy thông tin lịch giảng dạy nhưng gặp khó khăn trong việc trình bày chi tiết. {personal_address.title()} có thể truy cập hệ thống quản lý đào tạo để xem thông tin đầy đủ ạ. 🎯"

    def _get_api_error_response(self, error_type, session_id):
        try:
            if hasattr(self.response_generator, '_get_personal_address'):
                personal_address = self.response_generator._get_personal_address(session_id)
            else:
                personal_address = "giảng viên"
        except:
            personal_address = "giảng viên"
            
        if error_type == 'token_decode_failed':
            return f"Dạ {personal_address}, phiên đăng nhập đã hết hạn. {personal_address.title()} vui lòng đăng nhập lại vào ứng dụng BDU ạ. 🔐"
        elif error_type == 'authentication_failed':
            return f"Dạ {personal_address}, thông tin đăng nhập không hợp lệ. {personal_address.title()} vui lòng đăng nhập lại ạ. 🔐"
        else:
            return f"Dạ {personal_address}, em gặp khó khăn kỹ thuật khi truy xuất thông tin. {personal_address.title()} có thể thử lại sau ạ. 🎯"

    def get_system_status(self):
        """
        🚀 ENHANCED: Get system status with full generative support information
        """
        semantic_status = self.semantic_chatbot.get_system_status()
        api_status = external_api_service.get_system_status()
        cache_stats = self.query_cache.get_cache_stats()
        
        return {
            'service_name': 'BDUChatbotService',
            'architecture': 'enhanced_semantic_rag_with_generative_knowledge',  # 🚀 UPDATED
            'version': '2.0_generative',  # 🚀 NEW
            'chatbot_service': semantic_status,
            'external_api_service': api_status,
            'cache_performance': cache_stats,
            'enhanced_semantic_features': [
                'smart_penalty_system',
                'confidence_preservation', 
                'adaptive_mismatch_tolerance',
                'tiered_decision_logic',
                'targeted_clarification',
                'high_quality_answer_protection',
                'top5_smart_candidate_selection',
                'relevance_analysis_debugging',
                'suitability_based_selection',
                'document_context_processing',
                'external_api_integration',
                'conversation_memory',
                'query_response_cache',
                'graceful_degradation',
                # 🚀 NEW: Generative features
                'generative_general_knowledge_support',
                'generative_response_validation',
                'non_education_query_handling',
                'general_knowledge_detection',
                'context_aware_generative_responses',
                'adaptive_generative_fallback',
                'generative_appropriateness_check',
                'seamless_education_generative_transition',
                'confidence_based_generative_activation'
            ],
            'removed_features': [
                'intent_classification',
                'keyword_matching',
                'ensemble_methods',
                'mega_intent_system',
                'complex_context_analysis',
                'hard_coded_rules',
                'over_aggressive_penalties',
                'single_candidate_limitation',
                'strict_education_only_limitation',  # 🚀 Key removal
                'non_education_query_rejection'      # 🚀 Key removal
            ],
            'processing_flow': [
                '1. Cache Check',
                '2. Personal Info API Detection',
                '3. General Knowledge vs Education Classification',  # 🚀 ENHANCED
                '4. ENHANCED Semantic Retrieval (Fine-tuned Model)',
                '5. Smart Two-Stage Semantic Re-ranking (Top-5)',
                '6. Smart Candidate Selection from Top-5',
                '7. Confidence-Aware Decision Making with Generative Support',  # 🚀 ENHANCED
                '8. Education Answer OR Generative General Knowledge Response',  # 🚀 NEW
                '9. Response Validation and Smart Fallback',  # 🚀 ENHANCED
                '10. Cache Storage'
            ],
            'generative_capabilities': {  # 🚀 COMPREHENSIVE section
                'general_knowledge_support': True,
                'context_aware_responses': True,
                'personalized_addressing': True,
                'bdu_context_preservation': True,
                'response_validation': True,
                'adaptive_fallback': True,
                'confidence_threshold': 0.4,
                'education_override': False,
                'seamless_integration': True,
                'quality_assurance': True,
                'conversation_continuity': True,
                'fallback_graceful_degradation': True
            },
            'query_handling_modes': {  # 🚀 NEW section
                'education_bdu_queries': 'Enhanced Semantic RAG with confidence management',
                'personal_info_queries': 'External API with authentication',
                'general_knowledge_queries': 'Generative AI with BDU context preservation',
                'mixed_session_queries': 'Adaptive mode switching based on query type',
                'document_based_queries': 'Document context processing with OCR support'
            },
            'confidence_management': {  # 🚀 ENHANCED section
                'high_confidence_education': 'Direct answer from knowledge base',
                'medium_confidence_education': 'Enhanced answer with additional context',
                'low_confidence_education': 'Smart clarification request',
                'very_low_confidence_education': 'Don\'t know with department suggestions',
                'low_confidence_non_education': 'Generative general knowledge response',
                'inappropriate_queries': 'Polite redirection to BDU topics'
            }
        }

    # 🚀 NEW: Generative support utility methods
    def has_generative_support(self):
        """
        🚀 NEW: Check if generative support is available and properly configured
        """
        try:
            # Check if decision engine has generative enabled
            decision_engine = getattr(self.semantic_chatbot, 'decision_engine', None)
            if not decision_engine:
                return False
            
            generative_enabled = decision_engine.decision_factors.get('generative_enabled', False)
            
            # Check if response generator has the generative methods
            response_generator = getattr(self, 'response_generator', None)
            if not response_generator:
                return False
            
            has_generative_method = hasattr(response_generator, '_generate_general_knowledge_response_smart')
            
            # Check if Gemini API is available
            gemini_available = self.semantic_chatbot._check_gemini_availability()
            
            return generative_enabled and has_generative_method and gemini_available
            
        except Exception as e:
            logger.error(f"❌ Error checking generative support: {str(e)}")
            return False

    def toggle_generative_support(self, enabled: bool):
        """
        🚀 NEW: Enable or disable generative support at runtime
        """
        try:
            decision_engine = getattr(self.semantic_chatbot, 'decision_engine', None)
            if decision_engine:
                decision_engine.decision_factors['generative_enabled'] = enabled
                logger.info(f"🚀 Generative support {'enabled' if enabled else 'disabled'}")
                return True
            else:
                logger.error("❌ Decision engine not found")
                return False
        except Exception as e:
            logger.error(f"❌ Error toggling generative support: {str(e)}")
            return False

    def get_generative_stats(self):
        """
        🚀 NEW: Get statistics about generative usage
        """
        try:
            total_sessions = len(self.semantic_chatbot.conversation_memory)
            generative_sessions = 0
            generative_queries = 0
            
            for session_id, interactions in self.semantic_chatbot.conversation_memory.items():
                session_has_generative = False
                for interaction in interactions:
                    if interaction.get('generative_response_used', False):
                        generative_queries += 1
                        if not session_has_generative:
                            generative_sessions += 1
                            session_has_generative = True
            
            return {
                'total_sessions': total_sessions,
                'generative_sessions': generative_sessions,
                'generative_queries': generative_queries,
                'generative_session_percentage': (generative_sessions / max(total_sessions, 1)) * 100,
                'support_available': self.has_generative_support(),
                'feature_enabled': getattr(self.semantic_chatbot.decision_engine, 'decision_factors', {}).get('generative_enabled', False)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting generative stats: {str(e)}")
            return {
                'error': str(e),
                'support_available': False,
                'feature_enabled': False
            }

    # Delegate methods to semantic chatbot
    def get_conversation_memory(self, session_id):
        return self.semantic_chatbot.get_conversation_memory(session_id)

    def clear_conversation_memory(self, session_id=None):
        return self.semantic_chatbot.clear_conversation_memory(session_id)

    def reload_after_qa_update(self):
        return self.semantic_chatbot.reload_after_qa_update()

    @property
    def model(self):
        return self.semantic_chatbot.model

    @property
    def index(self):
        return self.semantic_chatbot.index

    @property
    def knowledge_data(self):
        return self.semantic_chatbot.knowledge_data

    def get_cache_stats(self):
        return self.query_cache.get_cache_stats()

    def clear_cache(self):
        return self.query_cache.clear_cache()

    def update_cache_ttl(self, new_ttl: int):
        self.query_cache.update_ttl(new_ttl)
        logger.info(f"🔄 Cache TTL updated to {new_ttl} seconds")

# Global instance for the application
chatbot_ai = BDUChatbotService()