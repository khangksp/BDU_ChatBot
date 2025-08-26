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
from .interaction_logger_service import interaction_logger
from .query_response_cache import query_response_cache

logger = logging.getLogger(__name__)

class SemanticReRanker:
    def __init__(self, retriever_service):
        self.retriever_service = retriever_service
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
        boost = 0.0
        answer_length = len(candidate.get('answer', ''))
        if 100 <= answer_length <= 500:  # Optimal length range
            boost += 0.05
        elif answer_length > 1000:  # Penalty for very long answers
            boost -= 0.05
        
        question = candidate.get('question', '')
        answer = candidate.get('answer', '')
        
        query_words = set(query.lower().split())
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        question_overlap = len(query_words.intersection(question_words)) / max(len(query_words), 1)
        if question_overlap > 0.3:
            boost += 0.1
        
        return min(0.2, boost)

    def _detect_mismatch_severity(self, candidate, query):
        query_lower = query.lower()
        question_lower = candidate.get('question', '').lower()
        answer_lower = candidate.get('answer', '').lower()
        
        mismatch_analysis = {
            'concept_severity': 0.0,
            'topic_severity': 0.0, 
            'context_severity': 0.0,
            'issues': []
        }
        
        concept_conflicts = [
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
            
        mismatch_analysis = self._detect_mismatch_severity(candidate, query)
        
        if not mismatch_analysis['issues']:
            return 0.0, []  # No mismatch detected
        
        if base_semantic_score >= 0.8:
            confidence_tier = 'very_high'
        elif base_semantic_score >= 0.65:
            confidence_tier = 'high'
        elif base_semantic_score >= 0.45:
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'
        
        max_penalty_rate = self.config['adaptive_penalty_rates'][confidence_tier]
        
        concept_penalty = mismatch_analysis['concept_severity'] * max_penalty_rate * 0.6  # 60% weight
        topic_penalty = mismatch_analysis['topic_severity'] * max_penalty_rate * 0.3     # 30% weight  
        context_penalty = mismatch_analysis['context_severity'] * max_penalty_rate * 0.1 # 10% weight
        
        total_penalty = concept_penalty + topic_penalty + context_penalty
        
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
            semantic_score = candidate.get('similarity', candidate.get('semantic_score', 0.0))            
            semantic_boost = self.calculate_semantic_boost(candidate, query)            
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
        
        enhanced_candidates.sort(key=lambda x: x['stage1_score'], reverse=True)
        
        stage1_candidates = enhanced_candidates[:self.config['stage1_top_k']]
        
        logger.info(f"🎯 FIXED Stage 1: {len(stage1_candidates)} candidates selected for cross-encoder re-ranking")
        
        return stage1_candidates

    def stage2_cross_encoder_simulation(self, candidates, query):
        if not candidates:
            logger.info("🔄 Stage 2 skipped: No candidates available")
            return []
        
        logger.info(f"🔄 Stage 2: Cross-encoder re-ranking {len(candidates)} candidates")
        
        try:
            cross_encoder_scores = self._simulate_cross_encoder_semantic(query, candidates)
            final_candidates = []
            for i, candidate in enumerate(candidates):
                stage1_score = candidate.get('stage1_score', 0.0)
                stage2_score = cross_encoder_scores[i] if i < len(cross_encoder_scores) else 0.0
                
                final_score = (
                    self.config['semantic_weight'] * stage1_score + 
                    self.config['cross_encoder_weight'] * stage2_score
                )
                
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
        self.semantic_confidence_thresholds = {
            'very_high': 0.8,    # Keep original - for truly excellent matches
            'high': 0.65,        # Slightly raised - for good matches  
            'medium': 0.45,      # Slightly raised - for decent matches
            'low': 0.25,         # Kept original - for poor matches
            'very_low': 0.1      # Kept original - for very poor matches
        }
        
        self.decision_factors = {
            'preserve_high_confidence': True,     # Don't over-penalize good answers
            'mismatch_tolerance': {               # Tolerance levels by confidence
                'very_high': 0.8,  # High tolerance for high confidence
                'high': 0.6,       # Medium tolerance for good confidence
                'medium': 0.4,     # Low tolerance for medium confidence
                'low': 0.2         # Very low tolerance for poor confidence
            },
            'smart_clarification_threshold': 0.3  # When to use smart vs generic clarification
        }
        
        self.personal_info_keywords = [
            'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
            'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
            'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
            'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi',
            'hôm nay', 'hom nay', 'ngày mai', 'ngay mai', 'tuần này', 'tuan nay'
        ]
        
        self.education_keywords = [
            'học', 'trường', 'sinh viên', 'giảng viên', 'dạy', 'bdu', 'đại học',
            'ngân hàng đề thi', 'báo cáo', 'kê khai', 'tạp chí', 'nghiên cứu'
        ]
        
        logger.info("✅ FIXED PureSemanticDecisionEngine initialized")
        logger.info("   🎯 FIXED decision making với smart confidence preservation")
        logger.info("   🛡️ High confidence answer protection")
        logger.info("   🧠 Adaptive mismatch tolerance")

    def categorize_semantic_confidence(self, final_score):
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
        if not query:
            return False
        
        query_lower = query.lower()        
        education_found = any(kw in query_lower for kw in self.education_keywords)
        
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
        if not query:
            return False
        
        query_lower = query.lower()
        
        needs_api = any(keyword in query_lower for keyword in self.personal_info_keywords)
        
        logger.debug(f"🌐 API check: '{query}' -> {needs_api}")
        return needs_api

    def _assess_mismatch_impact(self, best_candidate, original_score):
        if not best_candidate:
            return False, []
        
        mismatch_issues = best_candidate.get('mismatch_issues', [])
        smart_penalty = best_candidate.get('smart_penalty', 0.0)
        
        if not mismatch_issues:
            return False, []  # No mismatch issues
        
        confidence_tier = self.categorize_semantic_confidence(original_score)
        
        tolerance = self.decision_factors['mismatch_tolerance'].get(confidence_tier, 0.5)
        
        severity_score = smart_penalty / 0.3  # Normalize to 0-1 scale (max penalty is ~0.3)
        
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
        try:
            personal_address = "giảng viên"  # Default fallback
        except:
            personal_address = "giảng viên"
        
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

Em sẽ tìm thông tin chính xác hơn! 🔍"""
        
        else:
            return f"""Dạ {personal_address}, để em có thể hỗ trợ chính xác nhất, {personal_address} có thể làm rõ hơn về vấn đề cần hỗ trợ không ạ?

Em sẽ tìm thông tin phù hợp nhất cho {personal_address}! 🎯"""

    def make_decision(self, query, candidates_list, session_memory=None, jwt_token=None, document_text=None):
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
        
        is_education = self.is_education_related(query)
        if not is_education and session_memory and len(session_memory) == 0:  # First message
            logger.info("📚 SCOPE: Rejecting non-education query")
            return 'reject_non_education', None, False
        
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
        
        if not candidates_list:
            logger.warning("⚠️ No candidates provided for decision making")
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
            logger.info(f"🔬 SMART SELECTION: Analyzing {len(candidates_list)} candidates")
            
            for i, candidate in enumerate(candidates_list[:5]):
                score = candidate.get('final_score', 0)
                mismatch_count = len(candidate.get('mismatch_issues', []))
                semantic_score = candidate.get('semantic_score', 0)
                
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
                    
                logger.debug(f"🔬 Candidate #{i+1}: score={score:.3f}, semantic={semantic_score:.3f}, mismatches={mismatch_count}, suitability={suitability:.3f}")
            
            if best_candidate:
                original_pos = None
                for info in selection_info:
                    if info['suitability'] == best_suitability:
                        original_pos = info['position']
                        break
                logger.info(f"🔬 SMART SELECTION: Chose candidate #{original_pos} (suitability: {best_suitability:.3f})")
        else:
            best_candidate = candidates_list[0]
            logger.info("🔬 SINGLE CANDIDATE: Using the only available candidate")
        
        final_score = best_candidate.get('final_score', 0.0)
        original_score = best_candidate.get('semantic_score', final_score)        
        should_impact, mismatch_issues = self._assess_mismatch_impact(best_candidate, original_score)        
        confidence_level = self.categorize_semantic_confidence(final_score)
        
        logger.info(f"🎯 ENHANCED Semantic Decision Analysis:")
        logger.info(f"   📊 Selected candidate position: {original_pos if 'original_pos' in locals() else 1}")
        logger.info(f"   📊 Original semantic score: {original_score:.3f}")
        logger.info(f"   📊 Final score: {final_score:.3f}")
        logger.info(f"   🎯 Confidence level: {confidence_level}")
        logger.info(f"   🧠 Mismatch should impact: {should_impact}")
        logger.info(f"   🔍 Mismatch issues: {len(mismatch_issues)}")
                
        if confidence_level == 'very_high':
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
        from .phobert_service import retriever_service
        
        self.sbert_retriever = ChatbotAI(shared_response_generator=shared_response_generator)
        self.retriever_service = retriever_service
        self.response_generator = shared_response_generator
        self.decision_engine = PureSemanticDecisionEngine()
        
        self.semantic_reranker = SemanticReRanker(retriever_service=self.retriever_service)
        
        self.conversation_memory = {}
        
        logger.info("🎯 ENHANCED PureSemanticChatbotAI initialized")
        logger.info("   🛡️ Smart penalty system enabled")
        logger.info("   🧠 Confidence-aware decision making")
        logger.info("   🎯 High-quality answer preservation")
        logger.info("   🔬 Top-5 smart candidate selection")
    
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
            'mode': 'fixed_semantic_rag',  # 🎯 UPDATED
            'architecture': 'fixed_semantic_rag',
            'semantic_reranking': {
                'enabled': True,
                'smart_penalty_system': True,
                'confidence_preservation': True,
                'adaptive_penalty_rates': True,
                'stage1_candidates': self.semantic_reranker.config['stage1_top_k'],
                'stage2_final': self.semantic_reranker.config['stage2_top_n']
            },
            'decision_engine': {
                'type': 'fixed_semantic',
                'confidence_thresholds': self.decision_engine.semantic_confidence_thresholds,
                'smart_mismatch_handling': True,
                'high_confidence_preservation': True,
                'adaptive_tolerance': True
            },
            'fixed_semantic_features': [
                'smart_penalty_system',
                'confidence_preservation', 
                'adaptive_mismatch_tolerance',
                'tiered_decision_logic',
                'targeted_clarification',
                'high_quality_answer_protection'
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
                response_data = self._get_no_match_response()
                
                # ✅ BƯỚC 2: GHI LOG KHI KHÔNG TÌM THẤY ỨNG VIÊN
                interaction_logger.log_interaction(
                    query=query,
                    response=response_data.get('response', ''),
                    confidence=0.0,
                    method='semantic_search',
                    reason='no_candidates_found'
                )
                return response_data
            
            # STEP 5: FIXED SEMANTIC RE-RANKING
            reranked_candidates = self.semantic_reranker.rerank(candidates, query)
            
            if not reranked_candidates:
                logger.warning("⚠️ No candidates after FIXED semantic re-ranking")
                response_data = self._get_no_match_response()

                # ✅ BƯỚC 3: GHI LOG KHI RE-RANKING KHÔNG CÓ KẾT QUẢ
                interaction_logger.log_interaction(
                    query=query,
                    response=response_data.get('response', ''),
                    confidence=0.0,
                    method='reranking',
                    reason='no_candidates_after_rerank'
                )
                return response_data
            
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
            
            logger.info(f"🎯 FIXED Best candidate analysis:")
            logger.info(f"   📊 Original semantic: {original_semantic:.3f}")
            logger.info(f"   📉 Smart penalty: {smart_penalty:.3f}")
            logger.info(f"   📊 Final score: {final_score:.3f}")
            logger.info(f"   🔍 Mismatch issues: {len(mismatch_issues)}")
            for issue in mismatch_issues:
                logger.info(f"     ⚠️ {issue}")
            
            # STEP 7: ENHANCED SEMANTIC DECISION MAKING with top 5 candidates
            decision_type, context, should_respond = self.decision_engine.make_decision(
                query, reranked_candidates, session_memory, jwt_token, document_text
            )
            
            if decision_type in ['say_dont_know', 'ask_clarification']:
                 interaction_logger.log_interaction(
                    query=query,
                    response=f"Bot decided to '{decision_type}'", # Ghi lại quyết định
                    confidence=best_candidate.get('final_score', 0.0),
                    method=decision_type,
                    reason=f'decision_engine_{decision_type}'
                )
            
            # STEP 8: EXECUTE DECISION
            if not should_respond:
                response_text = self._get_out_of_scope_response(session_id)
                method = 'rejected_non_education'
            else:
                response_text = self._execute_fixed_semantic_decision(
                    decision_type, query, context, session_id
                )
                method = decision_type
            
            # STEP 9: UPDATE MEMORY
            if session_id and should_respond:
                self._update_semantic_memory(
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
                    'semantic_decision': True
                },
                'sources': self._format_sources(reranked_candidates[:2]),
                'processing_time': processing_time,
                'is_education': context is not None,
                'enhanced_semantic_rag': True,  # Updated from fixed_semantic_rag
                'reference_links': best_candidate.get('reference_links', []),
                'external_api_used': decision_type == 'use_external_api',
                'semantic_reranking_used': best_candidate.get('fixed_semantic_reranking', False),
                'session_memory_used': bool(session_memory),
                'document_context_used': bool(document_text),
                'document_context_priority': decision_type == 'use_document_context',
                'architecture': 'enhanced_semantic_rag_top5',  # Updated architecture name
                'enhanced_features': ['smart_penalty', 'confidence_preservation', 'adaptive_tolerance', 'top5_selection', 'smart_candidate_selection'],
                'reranking_stats': {
                    'original_semantic_score': original_semantic,
                    'semantic_boost': best_candidate.get('semantic_boost', 0),
                    'smart_penalty': smart_penalty,
                    'stage1_score': best_candidate.get('stage1_score', 0),
                    'stage2_score': best_candidate.get('stage2_score', 0),
                    'final_score': final_score,
                    'selected_position': context.get('selected_position', 1) if context else 1,
                    'top5_enhanced': True
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
                'graceful_degradation_used': True
            }

    def _execute_fixed_semantic_decision(self, decision_type, query, context, session_id):
        logger.info(f"🎯 Executing FIXED semantic decision: {decision_type}")
        
        gemini_available = self._check_gemini_availability()
        
        if not gemini_available:
            logger.warning("⚠️ Gemini API not available - using FIXED graceful degradation")
            return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
        
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
            
            elif decision_type in ['use_db_direct', 'enhance_db_answer']:
                response = self.response_generator.generate_response(
                    query=query, context=context, intent_info=None, entities={}, session_id=session_id
                )
                response_text = response.get('response', '') if response else ''
                
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning("⚠️ Empty/invalid response from Gemini - using FIXED semantic fallback")
                    return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
                
                return response_text
            
            elif decision_type == 'ask_clarification':
                if context and context.get('smart_clarification', False):
                    logger.info("🤔 Creating FIXED smart clarification response")
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
                return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)
                
        except Exception as e:
            logger.error(f"❌ Error executing FIXED semantic decision: {str(e)}")
            return self._create_fixed_semantic_fallback_response(decision_type, query, context, session_id)

    def _create_fixed_semantic_fallback_response(self, decision_type, query, context, session_id):
        personal_address = self._get_personal_address(session_id)
        
        raw_answer = context.get('db_answer', '') if context else ''
        mismatch_issues = context.get('mismatch_issues', []) if context else []
        confidence_preserved = context.get('confidence_preserved', False) if context else False
        
        if mismatch_issues and decision_type in ['use_db_direct', 'enhance_db_answer', 'ask_clarification']:
            logger.info("🤔 FIXED fallback: Using smart clarification due to detected mismatches")
            return self.decision_engine._create_smart_clarification_response(
                query, mismatch_issues, session_id
            )
        
        if decision_type in ['use_db_direct', 'enhance_db_answer']:
            if raw_answer and raw_answer.strip():
                logger.info(f"🔍 DEBUG - Raw database answer: '{raw_answer[:300]}...'")
                
                clean_answer = raw_answer.strip()
                
                clean_answer = re.sub(r'^(dạ\s+(thầy|cô|giảng viên)[^,]*,?\s*)', '', clean_answer, flags=re.IGNORECASE)
                clean_answer = re.sub(r'^(xin chào|chào)[^.!?]*[.!?]\s*', '', clean_answer, flags=re.IGNORECASE)
                
                if clean_answer and not clean_answer[0].isupper():
                    clean_answer = clean_answer[0].upper() + clean_answer[1:]
                
                personalized_response = f"Dạ {personal_address}, {clean_answer}"
                
                if not personalized_response.strip().endswith(('?', '!', '.')):
                    personalized_response += '.'
                
                if confidence_preserved:
                    personalized_response += f' {personal_address.title()} có cần em hỗ trợ thêm gì không ạ? 🎯'
                else:
                    personalized_response += f' {personal_address.title()} cần em làm rõ thêm gì không ạ? 🎯'
                
                logger.info(f"🛡️ FIXED SEMANTIC FALLBACK: Formatted raw answer for {personal_address}")
                return personalized_response
            else:
                return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯"
        
        return f"Dạ {personal_address}, em sẵn sàng hỗ trợ {personal_address} về các vấn đề liên quan đến BDU. {personal_address.title()} có thể chia sẻ cụ thể hơn về điều cần hỗ trợ không ạ? 🎯"

    def _check_gemini_availability(self):
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

    def _validate_answer_relevance(self, query, answer):
        try:
            query_lower = query.lower()
            answer_lower = answer.lower()
            
            concept_patterns = {
                'báo cáo khối lượng': ['báo cáo', 'khối lượng', 'công việc'],
                'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ'],
                'tốt nghiệp': ['tốt nghiệp', 'graduation'],
                'tạp chí': ['tạp chí', 'journal', 'bài viết'],
                'lịch giảng': ['lịch', 'giảng dạy', 'schedule'],
                'hạn nộp': ['hạn', 'deadline', 'chậm nhất']
            }
            
            main_concept = None
            for concept, keywords in concept_patterns.items():
                if any(kw in query_lower for kw in keywords):
                    main_concept = concept
                    break
            
            if not main_concept:
                return True  # Can't determine, assume relevant
            
            concept_keywords = concept_patterns[main_concept]
            answer_has_concept = any(kw in answer_lower for kw in concept_keywords)
            
            relevance_issues = []
            
            if 'báo cáo khối lượng' in query_lower and 'khối lượng học tập' in answer_lower:
                relevance_issues.append("Query về 'báo cáo khối lượng công việc' nhưng answer về 'khối lượng học tập sinh viên'")
            if 'kê khai nhiệm vụ' in query_lower and 'kê khai' not in answer_lower:
                relevance_issues.append("Query về 'kê khai nhiệm vụ' nhưng answer không chứa 'kê khai'")
            if relevance_issues:
                logger.warning(f"🔍 ANSWER RELEVANCE WARNING:")
                for issue in relevance_issues:
                    logger.warning(f"   ⚠️ {issue}")
                return False
            
            return answer_has_concept
            
        except Exception as e:
            logger.error(f"❌ Error in answer relevance validation: {str(e)}")
            return True  # Default to relevant to avoid breaking flow

    def _clean_query(self, query):
        if not query:
            return ""
        
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[?]{2,}', '?', query)
        query = re.sub(r'[!]{2,}', '!', query)
        
        return query

    def _update_semantic_memory(self, session_id, query, final_score, decision_type, was_education, context, document_text):
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
                'semantic_decision': True
            },
            'timestamp': time.time(),
            'user_type': 'lecturer',
            'decision_type': decision_type,
            'was_education_related': was_education,
            'fixed_semantic_processed': True,
            'document_context_used': bool(document_text),
            'document_context_priority': decision_type == 'use_document_context',
            'external_api_used': decision_type == 'use_external_api',
            'query_length': len(query.split()),
            'architecture': 'fixed_semantic_rag'
        }
        
        self.conversation_memory[session_id].append(interaction)        
        self.conversation_memory[session_id] = self.conversation_memory[session_id][-15:]
        
        logger.info(f"🧠 FIXED semantic memory updated for session {session_id}: {len(self.conversation_memory[session_id])} interactions")

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
            'fixed_semantic_rag': True
        }

    def _get_no_match_response(self):
        return {
            'response': "Dạ giảng viên, em chưa có thông tin về vấn đề này. Giảng viên có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎯",
            'confidence': 0.1,
            'method': 'no_match_semantic',
            'decision_type': 'say_dont_know',
            'processing_time': 0.01,
            'fixed_semantic_rag': True
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
                    'fixed_semantic_reranking': result.get('fixed_semantic_reranking', False)
                })
        return sources

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
        logger.info("🔄 Reloading FIXED semantic knowledge base...")
        
        if hasattr(self.sbert_retriever, 'cached_data'):
            self.sbert_retriever.cached_data = None
            self.sbert_retriever.cache_timestamp = 0
        
        self.sbert_retriever.load_knowledge_base()
        
        if self.sbert_retriever.model and self.sbert_retriever.knowledge_data:
            self.sbert_retriever.build_faiss_index()
        
        logger.info("✅ FIXED semantic knowledge base reloaded successfully")

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
        try:
            from sentence_transformers import SentenceTransformer
            fine_tuned_path = os.path.join(settings.BASE_DIR, 'fine_tuned_phobert')
            if os.path.exists(fine_tuned_path):
                self.model = SentenceTransformer(fine_tuned_path)
                logger.info("✅ Fine-tuned SBERT loaded from: fine_tuned_phobert")
            else:
                self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
                logger.info("✅ Base Vietnamese SBERT loaded")
            
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model = None

    def load_link_mapping(self):
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
        try:
            self.load_link_mapping()            
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
            
            csv_knowledge = []
            try:
                drive_data = drive_service.get_csv_data()
                if drive_data:
                    csv_knowledge = drive_data
                    logger.info(f"✅ Loaded {len(csv_knowledge)} records from Google Drive")
            except Exception as e:
                logger.error(f"❌ Failed to load from Google Drive: {str(e)}")
            
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
            
            db_knowledge = list(KnowledgeBase.objects.filter(is_active=True).values(
                'question', 'answer', 'category'
            ))
            
            self.knowledge_data = db_qa_entries + csv_knowledge + db_knowledge
            
            if self.model and self.knowledge_data:
                self.build_faiss_index()
            
            logger.info(f"✅ FIXED semantic knowledge base loaded: {len(self.knowledge_data)} entries")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            self.knowledge_data = []

    def build_faiss_index(self):
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
        try:
            if not self.model or not self.index:
                logger.warning("⚠️ Model or index not available")
                return []
            
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
        self.response_generator = GeminiResponseGenerator()
        self.query_cache = query_response_cache        
        self.semantic_chatbot = PureSemanticChatbotAI(shared_response_generator=self.response_generator)
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
        
        logger.info("🎯 ENHANCED BDUChatbotService initialized with Top-5 Smart Selection")

    def _needs_external_api(self, query: str) -> bool:
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
                    'cache_hit': False
                }
            
            cached_response = self.query_cache.get(query)
            if cached_response:
                cached_response['processing_time'] = time.time() - start_time
                logger.info(f"⚡ CACHE HIT: Response served in {cached_response['processing_time']:.3f}s")
                return cached_response
            
            logger.info("💨 CACHE MISS: Proceeding with ENHANCED semantic processing")
            
            if self._needs_external_api(query):
                logger.info("🚨 API PRIORITY: Personal info query detected")
                
                if jwt_token and jwt_token.strip():
                    api_result = self._handle_external_api_call(query, session_id, jwt_token)
                    api_result['cache_hit'] = False
                    api_result['cache_skipped'] = 'personal_query'
                    return api_result
                else:
                    auth_result = self._handle_authentication_required(session_id)
                    auth_result['cache_hit'] = False
                    auth_result['cache_skipped'] = 'authentication_required'
                    return auth_result
            
            logger.info("📚 Using ENHANCED Semantic RAG System with Top-5 Smart Selection")
            result = self.semantic_chatbot.process_query(query, session_id, jwt_token, document_text)
            
            result['api_priority_activated'] = False
            result['fallback_to_enhanced_semantic'] = True
            result['cache_hit'] = False
            
            cache_stored = self.query_cache.set(query, result)
            result['cache_stored'] = cache_stored
            
            if cache_stored:
                logger.info(f"💾 Response cached for future requests")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ ENHANCED BDU Service Error: {str(e)}")
            
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
                'graceful_degradation_used': True,
                'cache_hit': False,
                'cache_stored': False
            }

    def _handle_external_api_call(self, query: str, session_id: str, jwt_token: str) -> dict:
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
                    'fixed_semantic_rag': True
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
                    'graceful_degradation_used': True
                }
                
        except Exception as e:
            logger.error(f"❌ Error in external API call: {str(e)}")
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
                'graceful_degradation_used': True
            }

    def _handle_authentication_required(self, session_id: str) -> dict:
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
            'authentication_required': True
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
        semantic_status = self.semantic_chatbot.get_system_status()
        api_status = external_api_service.get_system_status()
        cache_stats = self.query_cache.get_cache_stats()        
        return {
            'service_name': 'BDUChatbotService',
            'architecture': 'enhanced_semantic_rag_top5',
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
                'top5_smart_candidate_selection',  # New feature
                'relevance_analysis_debugging',    # New feature
                'suitability_based_selection',     # New feature
                'document_context_processing',
                'external_api_integration',
                'conversation_memory',
                'query_response_cache',
                'graceful_degradation'
            ],
            'removed_features': [
                'intent_classification',
                'keyword_matching',
                'ensemble_methods',
                'mega_intent_system',
                'complex_context_analysis',
                'hard_coded_rules',
                'over_aggressive_penalties',
                'single_candidate_limitation'  # New removal
            ],
            'processing_flow': [
                '1. Cache Check',
                '2. Personal Info API Detection',
                '3. ENHANCED Semantic Retrieval (Fine-tuned Model)',
                '4. Smart Two-Stage Semantic Re-ranking (Top-5)',
                '5. Smart Candidate Selection from Top-5',
                '6. Confidence-Aware Decision Making',
                '7. Response Generation with Smart Fallback',
                '8. Cache Storage'
            ]
        }

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

chatbot_ai = BDUChatbotService()