import logging
import re
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

    def calculate_context_boost(self, candidate, context_keywords):
        """🚀 NEW: Tính điểm thưởng cho candidates chứa context entities"""
        if not context_keywords:
            return 0.0
            
        question = candidate.get('question', '').lower()
        answer = candidate.get('answer', '').lower()
        candidate_text = f"{question} {answer}"
        
        boost = 0.0
        matched_keywords = 0
        
        for keyword in context_keywords:
            keyword_lower = keyword.lower()
            
            # Exact match bonus
            if keyword_lower in candidate_text:
                matched_keywords += 1
                boost += 0.15  # Base boost per keyword
                
                # Extra boost for person names in answer
                if len(keyword.split()) >= 2:  # Multi-word (likely person name)
                    if keyword_lower in answer:
                        boost += 0.1  # Extra bonus for names in answer
                        
        # Diminishing returns for multiple keywords
        if matched_keywords > 0:
            keyword_ratio = matched_keywords / len(context_keywords)
            boost = boost * (0.5 + 0.5 * keyword_ratio)  # Scale by match ratio
            
        return min(0.3, boost)    
    
    def stage1_semantic_scoring(self, candidates, query, context_keywords=None):
        """🚀 UPDATED: Stage 1 với context boosting"""
        if not candidates:
            return []        
        enhanced_candidates = []        
        for candidate in candidates:
            if not candidate:
                continue
            semantic_score = candidate.get('similarity', candidate.get('semantic_score', 0.0))            
            semantic_boost = self.calculate_semantic_boost(candidate, query)            
            concept_penalty, mismatch_issues = self._calculate_smart_penalty(candidate, query, semantic_score)
            
            # 🚀 NEW: Context boost
            context_boost = self.calculate_context_boost(candidate, context_keywords) if context_keywords else 0.0
            
            # Final stage 1 score: semantic + boost - penalty + context_boost
            stage1_score = semantic_score + semantic_boost - concept_penalty + context_boost
            stage1_score = max(0.0, min(1.0, stage1_score))  # Clamp to [0,1]
            
            # Create enhanced candidate
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'semantic_score': semantic_score,
                'semantic_boost': semantic_boost,
                'smart_penalty': concept_penalty,
                'context_boost': context_boost,  # 🚀 NEW
                'mismatch_issues': mismatch_issues,
                'stage1_score': stage1_score,
                'ranking_method': 'stage1_with_context_boost'  # 🚀 UPDATED
            })
            
            enhanced_candidates.append(enhanced_candidate)
            
            if context_boost > 0:
                logger.debug(f"🎯 Context boost: semantic={semantic_score:.3f}, context_boost={context_boost:.3f}, final={stage1_score:.3f}")
        
        enhanced_candidates.sort(key=lambda x: x['stage1_score'], reverse=True)
        stage1_candidates = enhanced_candidates[:self.config['stage1_top_k']]        
        logger.info(f"🎯 Context-boosted Stage 1: {len(stage1_candidates)} candidates selected")        
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
            cross_encoder_score = (
                0.4 * question_overlap +      # Primary: query-question match
                0.3 * answer_coverage +       # Secondary: answer relevance  
                0.2 * qa_coherence +          # Coherence between Q&A
                0.1 * length_score            # Length optimization
            )

            cross_encoder_score = min(1.0, cross_encoder_score)
            scores.append(cross_encoder_score)

        return scores

    def apply_exact_name_priority(self, candidates, context_keywords):
        """🎯 CRITICAL: Ưu tiên tuyệt đối candidates có exact name match"""
        if not context_keywords:
            return candidates
            
        person_names = []
        for keyword in context_keywords:
            # Check if keyword is likely a person name (2+ words, proper case)
            if len(keyword.split()) >= 2 and keyword[0].isupper():
                person_names.append(keyword.lower())
        
        if not person_names:
            return candidates
            
        exact_matches = []
        partial_matches = []
        no_matches = []
        
        logger.info(f"🎯 Applying exact name priority for: {person_names}")
        
        for candidate in candidates:
            question = candidate.get('question', '').lower()
            answer = candidate.get('answer', '').lower()
            candidate_text = f"{question} {answer}"
            
            has_exact_match = False
            has_partial_match = False
            
            for person_name in person_names:
                # Exact name match
                if person_name in candidate_text:
                    has_exact_match = True
                    logger.debug(f"🎯 Exact match found: '{person_name}' in candidate")
                    break
                    
                # Partial match (last name only)
                name_parts = person_name.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    if len(last_name) > 2 and last_name in candidate_text:
                        has_partial_match = True
                        logger.debug(f"🎯 Partial match found: '{last_name}' in candidate")
            
            # Categorize candidates
            if has_exact_match:
                # Boost exact matches significantly
                candidate = candidate.copy()
                original_score = candidate.get('final_score', candidate.get('stage1_score', candidate.get('semantic_score', 0)))
                candidate['name_match_boost'] = 0.4  # Huge boost
                candidate['boosted_score'] = min(1.0, original_score + 0.4)
                exact_matches.append(candidate)
            elif has_partial_match:
                candidate = candidate.copy()
                original_score = candidate.get('final_score', candidate.get('stage1_score', candidate.get('semantic_score', 0)))
                candidate['name_match_boost'] = 0.2  # Medium boost
                candidate['boosted_score'] = min(1.0, original_score + 0.2)
                partial_matches.append(candidate)
            else:
                candidate = candidate.copy()
                candidate['name_match_boost'] = 0.0
                candidate['boosted_score'] = candidate.get('final_score', candidate.get('stage1_score', candidate.get('semantic_score', 0)))
                no_matches.append(candidate)
        
        # Sort each group by their boosted scores
        exact_matches.sort(key=lambda x: x['boosted_score'], reverse=True)
        partial_matches.sort(key=lambda x: x['boosted_score'], reverse=True)
        no_matches.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        # Combine: exact matches first, then partial, then others
        prioritized_candidates = exact_matches + partial_matches + no_matches
        
        logger.info(f"🎯 Name priority applied: {len(exact_matches)} exact, {len(partial_matches)} partial, {len(no_matches)} no match")
        
        return prioritized_candidates

    def rerank(self, candidates, query="", context_keywords=None):
        """🚀 UPDATED: Re-ranking với exact name priority"""
        if not candidates:
            return []        
        logger.info(f"🎯 Starting context-aware semantic re-ranking for {len(candidates)} candidates")
        if context_keywords:
            logger.info(f"🔍 Using context keywords: {context_keywords}")
        
        # STAGE 1: Context-aware semantic scoring  
        stage1_candidates = self.stage1_semantic_scoring(candidates, query, context_keywords)        
        if not stage1_candidates:
            logger.warning("⚠️ No candidates after context-aware Stage 1")
            return []
        
        # STAGE 2: Cross-encoder re-ranking
        stage2_candidates = self.stage2_cross_encoder_simulation(stage1_candidates, query)
        
        # 🎯 STAGE 3: Apply exact name priority (CRITICAL for person names)
        if context_keywords:
            stage2_candidates = self.apply_exact_name_priority(stage2_candidates, context_keywords)
            
            # Update final_score with boosted_score
            for candidate in stage2_candidates:
                candidate['final_score'] = candidate.get('boosted_score', candidate.get('final_score', 0))
        
        final_candidates = stage2_candidates[:self.config['stage2_top_n']]
        
        logger.info(f"✅ Context-aware + name-priority re-ranking complete: {len(final_candidates)} final candidates")        
        return final_candidates
    
