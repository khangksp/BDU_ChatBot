import logging
import re
logger = logging.getLogger(__name__)

class PureSemanticDecisionEngine:
    def __init__(self):
        self.semantic_confidence_thresholds = {
            'very_high': 0.75,   # Lowered from 0.8
            'high': 0.55,        # Lowered from 0.65 
            'medium': 0.35,      # Lowered from 0.45
            'low': 0.20,         # Lowered from 0.25
            'very_low': 0.1      # Kept original
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
        if not is_education and session_memory and len(session_memory) == 0:
            logger.info("📚 SCOPE: Rejecting non-education query")
            return 'reject_non_education', None, False
        
        if self.needs_external_api(query, 0.0):
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
                
                position_bonus = (5 - i) * 0.01
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
