import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import pickle
import os
import re
from django.conf import settings
from knowledge.models import KnowledgeBase
import logging
from .phobert_service import PhoBERTIntentClassifier
from .gemini_service import GeminiResponseGenerator, SimpleVietnameseRestorer
import pandas as pd
import random

from .external_api_service import external_api_service
from qa_management.services import drive_service

logger = logging.getLogger(__name__)

class HybridReRanker:
    def __init__(self):
        # Trọng số cho công thức final_score = α × semantic_score + β × keyword_score
        self.alpha = 0.6  # Trọng số cho semantic score
        self.beta = 0.4   # Trọng số cho keyword score
        
        # Enhanced keywords for different intent categories
        self.intent_keywords = {
            'bank_exam_questions': [
                'ngân hàng đề thi', 'đề thi', 'file mềm', 'báo cáo đề thi', 'ldkham@bdu.edu.vn'
            ],
            'annual_task_declaration': [
                'kê khai nhiệm vụ', 'nhiệm vụ năm học', 'giờ chuẩn', 'thỉnh giảng', 'daotao@bdu.edu.vn'
            ],
            'academic_journal': [
                'tạp chí', 'tạp chí khoa học', 'bài viết', 'nghiên cứu', 'jst@bdu.edu.vn'
            ],
            'competition_awards': [
                'thi đua', 'khen thưởng', 'danh hiệu', 'bằng khen', 'lễ khen thưởng'
            ],
            'personal_schedule': [
                'lịch của tôi', 'thời khóa biểu của tôi', 'tkb của tôi', 'tôi giảng', 'tôi dạy'
            ],
            'personal_info': [
                'tôi là ai', 'thông tin của tôi', 'email của tôi', 'chức danh của tôi'
            ],
            'document_reference': [
                'thông báo số', 'TB_', 'quyết định số', 'QĐ_', 'theo thông báo'
            ],
            'deadline_temporal': [
                'hạn cuối', 'deadline', 'trước ngày', 'hết ngày', 'chậm nhất'
            ],
            'contact_responsibility': [
                'gửi cho ai', 'phụ trách', 'địa chỉ email', '@bdu.edu.vn', 'ai ký'
            ],
            'technical_specification': [
                'định dạng', 'file mềm', 'cách gửi', 'thể lệ', 'bản điện tử'
            ],
            'university_general_info': [
                'trường đại học bình dương', 'bdu', 'phòng', 'khoa', 'cơ cấu tổ chức',
                'ngành', 'chuyên ngành', 'giảng viên', 'sinh viên', 'cộng học'
            ],
            'quality_assessment_accreditation': [
                'chất lượng', 'đánh giá', 'kiểm định', 'chuẩn đầu ra', 'feedback',
                'akc', 'kiểm định chất lượng', 'mục tiêu học tập', 'ctđt', 'clo', 'plo',
                'đánh giá môn học', 'feedback sinh viên', 'khảo sát chất lượng',
                'rà soát chương trình', 'cải tiến chương trình', 'nâng cao chất lượng',
                'tiêu chuẩn chất lượng', 'đảm bảo chất lượng', 'giám sát chất lượng',
                'đánh giá nội bộ', 'đánh giá ngoài', 'tự đánh giá', 'báo cáo tự đánh giá'
            ],
            'financial_tuition_fees': [
                # Học phí sinh viên (không phải thù lao giảng viên)
                'học phí sinh viên', 'chi phí học tập', 'học phí đại học',
                'thanh toán học phí', 'nộp học phí', 'miễn giảm học phí sinh viên',
                'phí dịch vụ sinh viên', 'phí đào tạo', 'học phí theo tín chỉ',
                'hạn nộp học phí', 'cách thức thanh toán học phí',
                'chính sách học phí', 'mức học phí', 'điều chỉnh học phí',
                'hoàn trả học phí', 'chính sách miễn giảm học phí',
                'hỗ trợ tài chính sinh viên', 'vay vốn sinh viên',
                
                # Phí dịch vụ khác (không phải lương)
                'phí lễ tốt nghiệp', 'phí thuê lễ phục', 'phí chứng chỉ',
                'phí bằng cấp', 'phí sao y', 'phí làm thẻ sinh viên'
            ],
            'registration_admission': [
                'đăng ký', 'tuyển sinh', 'xét tuyển', 'nộp hồ sơ', 'thủ tục',
                'nhập học', 'tuyển sinh đại học', 'xét tuyển đại học',
                'hồ sơ tuyển sinh', 'thủ tục nhập học', 'điều kiện tuyển sinh',
                'phương thức tuyển sinh', 'chỉ tiêu tuyển sinh', 'ngành tuyển sinh',
                'đăng ký xét tuyển', 'đăng ký môn học', 'đăng ký học phần',
                'rút môn học', 'thêm môn học', 'thay đổi đăng ký'
            ],
            'scholarships_financial_support': [
                'học bổng', 'hỗ trợ', 'miễn giảm', 'khó khăn', 'ưu đãi',
                'học bổng khuyến khích học tập', 'học bổng xã hội',
                'hỗ trợ tài chính', 'chính sách hỗ trợ', 'điều kiện hỗ trợ',
                'đăng ký hỗ trợ', 'xét duyệt học bổng', 'quy định học bổng',
                'sinh viên khó khăn', 'hỗ trợ đặc biệt', 'ưu đãi học tập'
            ],
            'facilities_infrastructure': [
                'cơ sở vật chất', 'phòng học', 'thiết bị', 'thư viện', 'ký túc xá'
            ],
            'facilities_infrastructure': [
                'cơ sở vật chất', 'phòng học', 'thiết bị', 'thư viện', 'ký túc xá',
                'wifi', 'phòng thí nghiệm', 'phòng máy tính', 'phòng CAD/CAM',
                'sân thể thao', 'bãi xe', 'căn tin', 'khu vực nghỉ ngơi',
                'hệ thống âm thanh', 'máy chiếu', 'điều hòa', 'hệ thống điện',
                'camera an ninh', 'hệ thống báo cháy', 'thang máy',
                'cải tạo cơ sở', 'nâng cấp thiết bị', 'bảo trì cơ sở'
            ],
            'graduation_ceremony': [
                'tốt nghiệp', 'lễ tốt nghiệp', 'tham dự', 'cử nhân', 'bằng cấp',
                'ai tham dự', 'được phép', 'thành phần', 'danh sách',
                'văn bằng', 'cấp bằng', 'nhận bằng', 'lễ phục',
                'thời gian tốt nghiệp', 'địa điểm tốt nghiệp', 'thủ tục tốt nghiệp',
                'điều kiện tốt nghiệp', 'xét tốt nghiệp', 'bằng tạm thời',
                'chụp ảnh tốt nghiệp', 'phí tham dự', 'lệ phí tốt nghiệp'
            ],
            'academic_regulations': [
                'điểm', 'học lại', 'nâng điểm', 'tín chỉ', 'chuyển đổi', 'quy định',
                'điểm trung bình', 'dtb', 'tính điểm', 'chuyển đổi điểm',
                'công nhận tín chỉ', 'khối lượng kiến thức', 'tối thiểu',
                'phần trăm', 'tối đa', 'giới hạn', 'quy định học tập',
                'xử lý học vụ', 'cảnh báo học tập', 'đình chỉ học tập',
                'buộc thôi học', 'kỷ luật học tập', 'vi phạm quy định'
            ],
            'lecturer_compensation': [
                # Thù lao cốt lõi
                'thù lao', 'thù lao giảng dạy', 'tiền dạy', 'tiền giảng dạy', 'thu lao',
                
                # Hệ số và tính toán
                'hệ số giảng dạy', 'hệ số thực hành', 'hệ số lý thuyết', 'hệ số tiếng nước ngoài',
                'công thức tính thù lao', 'cách tính thù lao', 'tính thù lao',
                
                # Giờ dạy và định mức
                'định mức giờ dạy', 'định mức giờ chuẩn', 'giờ chuẩn giảng dạy',
                'dạy 1 giờ', 'giờ dạy', 'tiết dạy', 'buổi dạy', 'số giờ dạy',
                
                # Lương
                'lương giảng viên', 'mức lương', 'bậc lương', 'lương cơ bản', 'hệ số lương',
                'nâng lương', 'tăng lương', 'bảng lương', 'ngạch lương',
                
                # Phụ cấp
                'phụ cấp', 'phụ cấp trách nhiệm', 'phụ cấp giảng dạy xa', 'phụ cấp đi xa',
                'trợ cấp', 'trợ cấp ăn trưa', 'phụ cấp khu vực',
                
                # Thưởng
                'thưởng giảng viên', 'thưởng cuối năm', 'thưởng thành tích', 'tiền thưởng',
                'khen thưởng tài chính', 'bonus',
                
                # Các loại thù lao
                'thù lao hợp đồng', 'thù lao thỉnh giảng', 'mức thù lao tuyển dụng',
                'thù lao cộng tác viên', 'lương thử việc',
                
                # Thanh toán
                'chi trả lương', 'thanh toán thù lao', 'ngày trả lương', 'nhận lương',
                'chuyển khoản lương', 'bảng thanh toán', 'phiếu lương',
                
                # Quyết định liên quan (từ phân tích QA.csv)
                'QĐ 442', 'QĐ 895', 'QĐ 1733', 'QĐ 101', 'QĐ 1153', 'QĐ 2004',
                'quyết định 442', 'quyết định thù lao', 'quyết định lương',
                
                # Mức thù lao cụ thể (từ phân tích)
                '15 triệu', '13 triệu', '10 triệu', '5 triệu', '2 triệu',
                'giáo sư 15 triệu', 'phó giáo sư 13 triệu', 'tiến sĩ 10 triệu', 'thạc sĩ 5 triệu',
                
                # Câu hỏi thường gặp
                'dạy 1 giờ bao nhiêu', 'giờ dạy được bao nhiêu', 'mức giá giờ dạy',
                'giảng viên nhận bao nhiêu', 'lương giảng viên bao nhiêu'
            ],
            'salary_benefits': [
                # Chế độ làm việc
                'chế độ làm việc', 'chế độ công tác', 'chế độ nghỉ phép', 'nghỉ phép năm',
                'thời gian làm việc', 'giờ làm việc', 'lịch làm việc',
                'nghỉ lễ', 'nghỉ tết', 'nghỉ hè', 'nghỉ thai sản', 'nghỉ ốm',
                
                # Phúc lợi
                'phúc lợi', 'đãi ngộ', 'quyền lợi', 'chế độ ưu đãi',
                'bảo hiểm xã hội', 'bảo hiểm y tế', 'bảo hiểm thất nghiệp',
                'chăm sóc sức khỏe', 'khám sức khỏe định kỳ',
                
                # Nghỉ dưỡng
                'nghỉ dưỡng', 'du lịch công tác', 'tham quan học tập',
                'nghỉ mát', 'team building', 'hoạt động tập thể',
                
                # Đào tạo
                'đào tạo nâng cao', 'học tập nâng cao', 'học thêm', 'học cao học',
                'hỗ trợ học phí', 'học bổng nâng cao trình độ',
                
                # Hỗ trợ
                'laptop công tác', 'máy tính làm việc', 'điện thoại công việc',
                'hỗ trợ internet', 'wifi miễn phí', 'văn phòng phẩm',
                'hỗ trợ nhà ở', 'hỗ trợ đi lại', 'hỗ trợ con em',
                'quà tết', 'quà sinh nhật', 'tiệc cuối năm'
            ],
            'university_programs': [
                'chương trình đào tạo', 'chương trình học', 'khóa học', 'học chế',
                'tín chỉ tích lũy', 'chuẩn đầu ra', 'mục tiêu đào tạo', 'phương pháp giảng dạy',
                'giáo trình', 'tài liệu học tập', 'đào tạo đại học', 'liên kết đào tạo',
                'chất lượng đào tạo', 'cải tiến chương trình', 'phát triển chương trình'
            ],

            'academic_assessment': [
                'đánh giá học tập', 'đánh giá kết quả', 'kiểm tra đánh giá', 'phương pháp đánh giá',
                'thang điểm', 'tiêu chí đánh giá', 'feedback sinh viên', 'phản hồi học tập',
                'đánh giá giảng viên', 'chất lượng giảng dạy', 'khảo sát sinh viên',
                'cải thiện chất lượng', 'giám sát chất lượng', 'đảm bảo chất lượng',
                'rà soát chương trình', 'điều chỉnh chương trình', 'nâng cao chất lượng'
            ],

            'student_services': [
                'dịch vụ sinh viên', 'hỗ trợ sinh viên', 'cố vấn học tập', 'tư vấn học tập',
                'ký túc xá', 'phòng ở', 'căn tin', 'y tế sinh viên', 'bảo hiểm y tế',
                'hoạt động ngoại khóa', 'câu lạc bộ sinh viên', 'sinh hoạt cộng đồng',
                'đoàn thanh niên', 'hội sinh viên', 'định hướng nghề nghiệp',
                'thực tập sinh viên', 'kết nối doanh nghiệp'
            ],

            'administrative_procedures': [
                'thủ tục hành chính', 'quy trình hành chính', 'giấy tờ hành chính',
                'đơn từ', 'giấy xác nhận', 'bản sao bằng cấp', 'chứng minh học tập',
                'giấy chuyển trường', 'bảng điểm tích lũy', 'khai học', 'thôi học',
                'chuyển ngành', 'chuyển lớp', 'nghỉ học tạm thời', 'bảo lưu kết quả',
                'phúc khảo bài thi', 'khiếu nại điểm số', 'hồ sơ sinh viên'
            ],

            'academic_calendar': [
                'lịch học tập', 'thời khóa biểu', 'lịch thi', 'lịch giảng dạy',
                'học kỳ', 'niên khóa', 'năm học', 'khai giảng', 'bế giảng',
                'nghỉ lễ', 'nghỉ tết', 'nghỉ hè', 'học phần', 'tín chỉ',
                'đăng ký môn học', 'rút môn', 'thêm môn', 'thay đổi lịch học',
                'lịch tập trung', 'lịch học bù', 'lịch thi lại', 'bảo vệ khóa luận'
            ],
            'lecturer_affairs': [
                'công việc giảng viên', 'nhiệm vụ giảng viên', 'trách nhiệm giảng viên',
                'định mức giờ dạy', 'khối lượng công việc', 'đánh giá giảng viên',
                'thăng hạng', 'bổ nhiệm', 'thi đua giảng viên', 'khen thưởng giảng viên',
                'nghiên cứu khoa học', 'công trình nghiên cứu', 'đề tài nghiên cứu',
                'hội nghị khoa học', 'seminar', 'workshop', 'tập huấn giảng viên',
                'phát triển năng lực', 'nâng cao trình độ', 'trao đổi học thuật'
            ],

            'international_cooperation': [
                'hợp tác quốc tế', 'trao đổi sinh viên', 'chương trình liên kết',
                'du học', 'học bổng quốc tế', 'đối tác nước ngoài',
                'trao đổi giảng viên', 'nghiên cứu hợp tác', 'dự án quốc tế',
                'hội nghị quốc tế', 'chứng chỉ quốc tế', 'chuẩn quốc tế',
                'tiếng anh', 'ngoại ngữ', 'IELTS', 'TOEFL', 'TOEIC',
                'văn hóa quốc tế', 'giao lưu văn hóa', 'sự kiện quốc tế'
            ],

            'library_resources': [
                'thư viện', 'tài liệu tham khảo', 'sách giáo khoa', 'giáo trình',
                'cơ sở dữ liệu', 'tài nguyên điện tử', 'sách điện tử', 'tạp chí điện tử',
                'mượn sách', 'gia hạn sách', 'đặt chỗ sách', 'tìm kiếm tài liệu',
                'phòng đọc', 'khu vực học tập', 'máy tính tra cứu',
                'wifi thư viện', 'dịch vụ thư viện', 'hướng dẫn sử dụng',
                'đào tạo kỹ năng', 'tra cứu thông tin', 'nghiên cứu tài liệu'
            ],

        }
        
        logger.info("🎯 HybridReRanker initialized with α={}, β={}".format(self.alpha, self.beta))
    
    def extract_keywords_from_intent(self, intent_result):
        """Extract relevant keywords from intent classification result"""
        intent_name = intent_result.get('intent', 'general')
        keywords = self.intent_keywords.get(intent_name, [])
        
        # Also extract keywords from the query itself
        normalized_query = intent_result.get('normalized_query', '').lower()
        query_keywords = normalized_query.split()
        
        # Combine intent-specific keywords with query keywords
        all_keywords = keywords + query_keywords
        
        logger.debug(f"🔍 Extracted keywords for intent '{intent_name}': {keywords[:3]}...")
        return all_keywords
    
    def calculate_keyword_score(self, candidate, keywords):
        """Calculate keyword matching score for a candidate"""
        if not keywords:
            return 0.0
        
        candidate_text = (
            candidate.get('question', '') + ' ' + 
            candidate.get('answer', '') + ' ' + 
            candidate.get('category', '')
        ).lower()
        
        matched_keywords = 0
        total_weight = 0
        
        for keyword in keywords:
            if not keyword:
                continue
                
            keyword_lower = keyword.lower()
            
            # Different weights for different match types
            if keyword_lower in candidate_text:
                if keyword_lower in candidate.get('question', '').lower():
                    # Higher weight for matches in question
                    matched_keywords += 2.0
                elif keyword_lower in candidate.get('category', '').lower():
                    # Medium weight for category matches
                    matched_keywords += 1.5
                else:
                    # Lower weight for answer matches
                    matched_keywords += 1.0
            
            total_weight += 2.0  # Maximum possible weight per keyword
        
        # Normalize score
        keyword_score = matched_keywords / max(total_weight, 1.0)
        
        return min(1.0, keyword_score)  # Cap at 1.0
    
    def calculate_context_boost(self, candidate, intent_result):
        """Calculate context-specific boost for lecturer queries"""
        boost = 0.0
        
        # Boost for exact category matches
        intent_name = intent_result.get('intent', '')
        candidate_category = candidate.get('category', '').lower()
        
        if 'giảng viên' in candidate_category and intent_name in self.intent_keywords:
            boost += 0.1
        
        # Boost for high confidence intent classification
        intent_confidence = intent_result.get('confidence', 0)
        if intent_confidence > 0.8:
            boost += 0.05
        
        # Boost for entities matching
        entities = intent_result.get('entities', {})
        if entities.get('has_personal_context', False):
            if any(word in candidate.get('answer', '').lower() for word in ['tôi', 'của tôi', 'bạn']):
                boost += 0.15
        
        return min(0.3, boost)  # Cap boost at 0.3
    
    def rerank(self, candidates, intent_result):
        """
        Main re-ranking method combining semantic and keyword scores
        
        Args:
            candidates: List of candidate results from semantic search
            intent_result: Intent classification result from PhoBERT
            
        Returns:
            List of candidates sorted by final_score (highest first)
        """
        if not candidates:
            return []
        
        # Extract keywords from intent
        keywords = self.extract_keywords_from_intent(intent_result)
        
        enhanced_candidates = []
        
        for candidate in candidates:
            if not candidate:
                continue
            
            # Get original semantic score
            semantic_score = candidate.get('similarity', candidate.get('semantic_score', 0.0))
            
            # Calculate keyword score
            keyword_score = self.calculate_keyword_score(candidate, keywords)
            
            # Calculate context boost
            context_boost = self.calculate_context_boost(candidate, intent_result)
            
            # Calculate final score with weighted combination
            final_score = (
                self.alpha * semantic_score + 
                self.beta * keyword_score + 
                context_boost
            )
            
            # Create enhanced candidate with all scores
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'context_boost': context_boost,
                'final_score': final_score,
                'ranking_method': 'hybrid_reranking'
            })
            
            enhanced_candidates.append(enhanced_candidate)
            
            logger.debug(f"🔄 Candidate: sem={semantic_score:.3f}, kw={keyword_score:.3f}, "
                        f"boost={context_boost:.3f}, final={final_score:.3f}")
        
        # Sort by final_score in descending order
        enhanced_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"🎯 Re-ranked {len(enhanced_candidates)} candidates. "
                   f"Top score: {enhanced_candidates[0]['final_score']:.3f}")
        
        return enhanced_candidates


class LecturerDecisionEngine:
    """🚀 NÂNG CẤP: Enhanced Decision Engine với Session Memory Awareness"""
    
    def __init__(self):
        # ✅ BƯỚC 3: Tăng ngưỡng medium_trust lên 0.5
        self.confidence_thresholds = {
            'high_trust': 0.75,    # Slightly lower due to re-ranking boost
            'medium_trust': 0.5,   # ✅ tăng 0.5
            'low_trust': 0.25,
            'no_trust': 0.1         
        }
        
        # ✅ NEW: Generation boost factors
        self.generation_boost_settings = {
            'enable_boost': True,
            'boost_probability': 0.15,
            'boost_keywords': [
                'ngân hàng đề thi', 'kê khai nhiệm vụ', 'tạp chí', 'nghiên cứu',
                'thi đua', 'khen thưởng', 'báo cáo', 'lịch giảng dạy',
                'chất lượng', 'đánh giá', 'tiêu chuẩn', 'quy trình'
            ]
        }
        
        # 🚀 NÂNG CẤP: Enhanced external API config với session memory support
        self.external_api_config = {
            'low_confidence_threshold': 0.3,
            'personal_info_keywords': [
                'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
                'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
                'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
                'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi'
            ],
            'time_context_keywords': [
                'hôm nay', 'hom nay', 'today', 'ngày mai', 'ngay mai', 'tomorrow',
                'tuần này', 'tuan nay', 'this week', 'tuần tới', 'tuan toi', 'next week',
                'tuần sau', 'tuan sau', 'cuối tuần', 'cuoi tuan', 'đầu tuần', 'dau tuan'
            ],
            # 🚀 NEW: Schedule continuation keywords để nhận diện câu hỏi tiếp theo
            'schedule_continuation_keywords': [
                'còn', 'con', 'thêm', 'them', 'nữa', 'nua', 'khác', 'khac', 
                'và', 'va', 'tiếp theo', 'tiep theo', 'sau đó', 'sau do',
                'thế còn', 'the con', 'vậy còn', 'vay con', 'còn gì', 'con gi'
            ],
            # 🚀 NEW: Context memory thresholds
            'context_memory_threshold': 0.7,  # Ngưỡng confidence intent từ lịch sử
            'context_recency_limit': 2  # Chỉ xem 2 interaction gần nhất
        }
        
        # Enhanced education keywords for lecturers
        self.education_keywords = [
            'học', 'trường', 'sinh viên', 'tuyển sinh', 'học phí', 'ngành', 
            'đại học', 'bdu', 'gv', 'giảng viên', 'dạy', 'quy định', 'khoa',
            'chương trình', 'đào tạo', 'lịch', 'thời khóa biểu', 'phòng', 'lớp',
            'ngân hàng đề thi', 'file mềm', 'báo cáo', 'nộp', 'hạn cuối',
            'kê khai', 'nhiệm vụ năm học', 'giờ chuẩn', 'thỉnh giảng', 
            'tạp chí', 'khoa học công nghệ', 'bài viết', 'nghiên cứu',
            'thi đua', 'khen thưởng', 'danh hiệu', 'bằng khen',
            'điểm', 'rèn luyện', 'hạnh kiểm', 'xếp loại', 'kỷ luật', 'quyền lợi',
            'thủ tục', 'hành chính', 'mẫu đơn', 'bảng điểm', 'thẻ',
            'mật khẩu', 'tài khoản', 'email', 'hệ thống',
            'thù lao', 'lương', 'hệ số', 'chế độ', 'phúc lợi', 'bảo hiểm',
            'phúc khảo', 'chấm thi', 'điểm thi', 'bài thi', 'bài tập', 'đánh giá',
            'học bổng', 'miễn giảm', 'hỗ trợ', 'khó khăn', 'ưu đãi',
            'cơ sở vật chất', 'phòng học', 'thiết bị', 'thư viện', 'ký túc xá',
            'wifi', 'phòng thí nghiệm', 'phòng máy tính', 'sân thể thao',
            'bãi xe', 'căn tin', 'khu vực nghỉ ngơi',
        ]
        
        self.lecturer_keywords = [
            'giảng viên', 'gv', 'thầy', 'cô', 'phụ trách', 'giảng dạy',
            'nghiên cứu', 'hội đồng', 'khoa', 'bộ môn', 'chuyên ngành'
        ]
        
        self.vague_keywords = [
            'làm sao', 'như thế nào', 'cách nào', 'thủ tục', 'quy trình',
            'thông tin', 'chi tiết', 'hướng dẫn', 'giúp đỡ', 'hỗ trợ',
            'gì', 'nào', 'khi nào', 'ở đâu', 'ai', 'sao', 'có phải'
        ]
        
        logger.info("✅ Enhanced LecturerDecisionEngine initialized with Session Memory Support")
    
    def is_education_related(self, query):
        """Enhanced education detection for lecturers"""
        if not query:
            return False
        
        query_lower = query.lower()
        
        found_keywords = []
        for kw in self.education_keywords:
            if kw in query_lower:
                found_keywords.append(kw)
        
        education_count = len(found_keywords)
        lecturer_count = sum(1 for kw in self.lecturer_keywords if kw in query_lower)
        
        is_education = education_count >= 1 or lecturer_count >= 1
        
        if not is_education:
            education_patterns = [
                r'phí.*(?:học|tốt nghiệp|nhận|cấp)',
                r'(?:học|phí|tiền).*(?:phí|học|cấp|nhận)',
                r'(?:bằng|văn bằng|tốt nghiệp)',
                r'(?:thủ tục|quy trình|cách thức)',
                r'(?:bdu|đại học|trường)',
                r'(?:sinh viên|học sinh)',
                r'(?:giảng viên|thầy|cô|gv)'
            ]
            
            for pattern in education_patterns:
                if re.search(pattern, query_lower):
                    is_education = True
                    found_keywords.append(f"pattern:{pattern}")
                    break
        
        logger.info(f"🎓 Education check: '{query}' -> keywords:{found_keywords} -> {is_education}")
        return is_education
    
    def needs_clarification(self, query, confidence):
        """Check if query needs clarification"""
        if not query:
            return False
            
        query_lower = query.lower()
        vague_count = sum(1 for kw in self.vague_keywords if kw in query_lower)
        word_count = len(query.split())
        
        needs_clarification = (
            (vague_count >= 2 and word_count <= 5) or 
            (confidence < self.confidence_thresholds['low_trust'] and vague_count >= 1)
        )
        
        logger.info(f"❓ Clarification check: vague:{vague_count}, words:{word_count}, conf:{confidence:.3f} -> {needs_clarification}")
        return needs_clarification
    
    def categorize_confidence(self, final_score):
        """✅ UPDATED: Categorize confidence level using hybrid final_score"""
        if final_score >= self.confidence_thresholds['high_trust']:
            return 'high_trust'
        elif final_score >= self.confidence_thresholds['medium_trust']:
            return 'medium_trust'  
        elif final_score >= self.confidence_thresholds['low_trust']:
            return 'low_trust'
        else:
            return 'no_trust'
    
    def _should_boost_generation(self, query, confidence_level):
        """Determine if we should boost generation for this query"""
        if not self.generation_boost_settings['enable_boost']:
            return False
        
        query_lower = query.lower()
        has_boost_keywords = any(keyword in query_lower for keyword in self.generation_boost_settings['boost_keywords'])
        random_boost = random.random() < self.generation_boost_settings['boost_probability']
        
        should_boost = (has_boost_keywords and confidence_level == 'high_trust') or random_boost
        
        if should_boost:
            logger.info(f"🚀 GENERATION BOOST ACTIVATED: keywords={has_boost_keywords}, random={random_boost}")
        
        return should_boost

    def needs_external_api(self, query: str, confidence: float, recent_intent: str = None, session_memory: list = None) -> bool:
        """🚀 NÂNG CẤP: Determine if query should use external API với session memory awareness"""
        query_lower = query.lower()
        
        # ✅ CHECK 1: Direct personal keyword matches (unchanged)
        has_personal_keywords = any(
            keyword in query_lower 
            for keyword in self.external_api_config['personal_info_keywords']
        )
        
        # ✅ CHECK 2: Time context (usually indicates schedule query)
        has_time_context = any(
            keyword in query_lower 
            for keyword in self.external_api_config['time_context_keywords']
        )
        
        # ✅ CHECK 3: Intent-based detection (unchanged)
        intent_confidence = 0
        intent_is_personal = False
        if recent_intent:
            if isinstance(recent_intent, str):
                intent_is_personal = recent_intent in ['personal_schedule', 'personal_info']
                intent_confidence = 0.7
            elif isinstance(recent_intent, dict):
                intent_name = recent_intent.get('intent', '')
                intent_confidence = recent_intent.get('confidence', 0)
                intent_is_personal = intent_name in ['personal_schedule', 'personal_info']
        
        high_confidence_personal_intent = intent_is_personal and intent_confidence > 0.6
        
        # 🚀 NEW CHECK 4: Session Memory Context Analysis
        context_suggests_schedule = False
        has_continuation_words = False
        
        if session_memory and len(session_memory) > 0:
            # Lấy 2 interaction gần nhất
            recent_interactions = session_memory[-self.external_api_config['context_recency_limit']:]
            
            # Kiểm tra xem có interaction nào về schedule không
            for interaction in recent_interactions:
                past_intent = interaction.get('intent_info', {})
                if isinstance(past_intent, dict):
                    past_intent_name = past_intent.get('intent', '')
                    past_intent_confidence = past_intent.get('confidence', 0)
                    
                    if (past_intent_name in ['personal_schedule', 'teaching_schedule', 'schedule_general'] and 
                        past_intent_confidence > self.external_api_config['context_memory_threshold']):
                        context_suggests_schedule = True
                        logger.info(f"🧠 CONTEXT MEMORY: Found schedule intent '{past_intent_name}' with confidence {past_intent_confidence:.3f}")
                        break
            
            # Kiểm tra từ khóa continuation trong query hiện tại
            has_continuation_words = any(
                keyword in query_lower 
                for keyword in self.external_api_config['schedule_continuation_keywords']
            )
        
        # 🚀 NEW: Context-driven API decision
        # Nếu có ngữ cảnh lịch trình + từ khóa tiếp tục => rất có thể cần API
        context_driven_api_need = context_suggests_schedule and has_continuation_words
        
        # 🚀 NEW: Smart inference for ambiguous queries
        # Query ngắn + có time context + có context lịch trình => có thể cần API
        smart_inference = (
            len(query.split()) <= 5 and 
            has_time_context and 
            context_suggests_schedule
        )
        
        # ✅ CHECK 5: Other conditions (unchanged)
        schedule_related_intent = recent_intent in ['personal_schedule', 'teaching_schedule', 'schedule_general']
        contextual_schedule_query = has_time_context and schedule_related_intent
        
        # 🚀 FINAL DECISION với memory context
        needs_api = (
            has_personal_keywords or 
            contextual_schedule_query or 
            high_confidence_personal_intent or
            context_driven_api_need or  # ✅ NEW
            smart_inference  # ✅ NEW
        )

        # 🚀 ENHANCED LOGGING
        logger.info(f"🔍 ENHANCED External API check:")
        logger.info(f"   📝 Query: '{query}' (confidence={confidence:.3f})")
        logger.info(f"   🔑 Direct factors: personal_kw={has_personal_keywords}, time_ctx={has_time_context}")
        logger.info(f"   🧠 Context factors: suggests_schedule={context_suggests_schedule}, continuation_words={has_continuation_words}")
        logger.info(f"   🎯 Enhanced factors: context_driven={context_driven_api_need}, smart_inference={smart_inference}")
        logger.info(f"   ✅ Final decision: needs_api={needs_api}")
        
        return needs_api

    def make_decision(self, query, best_candidate, intent_result, session_memory=None, jwt_token=None):
        """🚀 NÂNG CẤP: Enhanced decision making với session memory integration"""
        
        # Xác định đây có phải tin nhắn đầu tiên không
        is_first_message = not session_memory or len(session_memory) == 0
        
        # ✅ ENHANCED: Kiểm tra ngữ cảnh từ các tin nhắn trước với deep analysis
        context_override = False
        recent_intent = None
        
        if not is_first_message:
            # Phân tích các interaction gần đây để hiểu ngữ cảnh
            recent_interactions = session_memory[-3:] if len(session_memory) >= 3 else session_memory
            
            schedule_related_intents = 0
            education_related_queries = 0
            
            for interaction in recent_interactions:
                intent_info_from_memory = interaction.get('intent_info', {}) 
                past_intent = intent_info_from_memory.get('intent', '')
                past_query = interaction.get('query', '').lower()
                
                # Đếm các intent liên quan đến lịch trình
                if past_intent in ['personal_schedule', 'teaching_schedule', 'schedule_general']:
                    schedule_related_intents += 1
                    recent_intent = past_intent  # Lưu intent gần nhất
                
                # Đếm các query liên quan giáo dục
                if self.is_education_related(past_query):
                    education_related_queries += 1
            
            # Context override logic
            if schedule_related_intents >= 1:
                context_override = True
                logger.info(f"🧠 ENHANCED MEMORY OVERRIDE: {schedule_related_intents} schedule-related intents detected")
            elif education_related_queries >= 2:
                context_override = True
                logger.info(f"🧠 ENHANCED MEMORY OVERRIDE: {education_related_queries} education-related queries detected")
            
        # Bỏ qua kiểm tra "có liên quan giáo dục không" cho tin nhắn đầu tiên hoặc có context override
        is_education = self.is_education_related(query) or context_override or is_first_message
        if not is_education:
            logger.info("DECISION: Rejecting non-education query on a non-first message without context.")
            return 'reject_non_education', None, False
        
        # Lấy điểm và phân loại độ tin cậy
        final_score = best_candidate.get('final_score', 0) if best_candidate else 0
        confidence_level = self.categorize_confidence(final_score)
        
        # 🚀 ENHANCED: Logic kiểm tra API với session memory
        needs_api = self.needs_external_api(
            query, final_score, intent_result, session_memory
        )
        has_jwt_token = bool(jwt_token and jwt_token.strip())
        
        logger.info(f"🤖 ENHANCED Hybrid Decision: final_score={final_score:.3f}, level={confidence_level}, needs_api={needs_api}, has_token={has_jwt_token}")
        
        # 🚀 ENHANCED: Ưu tiên logic API với memory context
        if needs_api and has_jwt_token:
            # ✅ Special handling: Nếu có context memory và query ngắn, ưu tiên API hơn nữa
            if session_memory and len(query.split()) <= 5:
                logger.info("🚀 CONTEXT PRIORITY: Short query with memory context -> prioritizing API")
            
            return 'use_external_api', {
                'instruction': 'external_api_lecturer',
                'query': query,
                'jwt_token': jwt_token,
                'fallback_qa_answer': best_candidate.get('answer', '') if best_candidate else '',
                'confidence': final_score,
                'message': 'Using external API for personal/schedule information',
                'enhanced_by_context': bool(session_memory)  # ✅ NEW flag
            }, True
        
        elif needs_api and not has_jwt_token:
            return 'require_authentication', {
                'instruction': 'authentication_required',
                'query': query,
                'confidence': final_score,
                'message': 'Personal information requires authentication',
                'context_suggested': bool(session_memory and len(session_memory) > 0)  # ✅ NEW flag
            }, True
        
        # 🚀 ENHANCED: Kiểm tra nhu cầu làm rõ với context awareness
        needs_clarification = self.needs_clarification(query, final_score)
        
        # ✅ SPECIAL CASE: Nếu có context memory mạnh, giảm nhu cầu clarification
        if needs_clarification and session_memory and len(session_memory) > 0:
            # Kiểm tra xem có context schedule không
            has_strong_schedule_context = any(
                interaction.get('intent_info', {}).get('intent', '') in ['personal_schedule', 'teaching_schedule']
                for interaction in session_memory[-2:]
            )
            
            if has_strong_schedule_context and confidence_level in ['low_trust', 'medium_trust']:
                logger.info("🧠 CONTEXT OVERRIDE: Strong schedule context -> reducing clarification need")
                needs_clarification = False
        
        if needs_clarification and confidence_level not in ['medium_trust', 'high_trust']:
            return 'ask_clarification', {
                'query': query,
                'confidence': final_score,
                'instruction': 'clarification_needed',
                'message': 'Question is too vague, need clarification',
                'context_available': bool(session_memory and len(session_memory) > 0)  # ✅ NEW flag
            }, True
        
        # Áp dụng generation boost
        should_boost = self._should_boost_generation(query, confidence_level)
        if should_boost and confidence_level == 'high_trust':
            confidence_level = 'medium_trust'
            logger.info("🚀 GENERATION BOOST: Downgraded high_trust to medium_trust")
        
        # Logic ra quyết định cuối cùng dựa trên độ tin cậy
        if confidence_level == 'high_trust':
            decision = 'use_db_direct'
            context = {
                'instruction': 'direct_answer_lecturer',
                'db_answer': best_candidate.get('answer', '') if best_candidate else '',
                'confidence': final_score,
                'message': 'High confidence - use database answer directly',
                'enhanced_by_context': bool(session_memory)  # ✅ NEW flag
            }
        elif confidence_level == 'medium_trust':
            decision = 'enhance_db_answer'
            context = {
                'instruction': 'enhance_answer_lecturer',
                'db_answer': best_candidate.get('answer', '') if best_candidate else '',
                'confidence': final_score,
                'message': 'Medium confidence - enhance database answer',
                'generation_boosted': should_boost,
                'enhanced_by_context': bool(session_memory)  # ✅ NEW flag
            }
        elif confidence_level == 'low_trust':
            decision = 'ask_clarification'
            context = {
                'instruction': 'clarification_needed',
                'db_answer': best_candidate.get('answer', '') if best_candidate else '',
                'confidence': final_score,
                'message': 'Low confidence - ask for clarification',
                'context_available': bool(session_memory and len(session_memory) > 0)  # ✅ NEW flag
            }
        else:  # no_trust
            decision = 'say_dont_know'
            context = {
                'instruction': 'dont_know_lecturer',
                'confidence': final_score,
                'message': 'No relevant information - say dont know'
            }
        
        logger.info(f"🎯 ENHANCED Hybrid Decision made: {decision} (final_score: {final_score:.3f}, context_enhanced: {bool(session_memory)})")
        return decision, context, True


class HybridChatbotAI:
    """🚀 NÂNG CẤP: Enhanced Hybrid Chatbot với Session Memory Integration"""
    
    def __init__(self, shared_response_generator):
        # Initialize components với shared response_generator
        self.sbert_retriever = ChatbotAI(shared_response_generator=shared_response_generator)
        self.intent_classifier = PhoBERTIntentClassifier()
        self.response_generator = shared_response_generator
        self.decision_engine = LecturerDecisionEngine()
        self.reranker = HybridReRanker()
        self.conversation_memory = {}
        
        logger.info("🚀 Enhanced HybridChatbotAI initialized with Session Memory Support")
    
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
        """Get system status including hybrid features"""
        gemini_status = self.response_generator.get_system_status()
        drive_status = drive_service.get_system_status()
        external_api_status = external_api_service.get_system_status()
        
        qa_management_status = {}
        try:
            from qa_management.models import QAEntry, QASyncLog
            qa_management_status = {
                'available': True,
                'total_entries': QAEntry.objects.count(),
                'active_entries': QAEntry.objects.filter(is_active=True).count(),
                'pending_sync': QAEntry.objects.filter(sync_status='pending').count(),
                'synced_entries': QAEntry.objects.filter(sync_status='synced').count(),
                'error_entries': QAEntry.objects.filter(sync_status='error').count(),
            }
        except Exception as e:
            qa_management_status = {
                'available': False,
                'error': str(e)
            }
        
        return {
            'sbert_model': bool(self.sbert_retriever.model),
            'faiss_index': bool(self.sbert_retriever.index),
            'phobert_available': not self.intent_classifier.fallback_mode,
            'gemini_available': gemini_status.get('gemini_api_available', False),
            'knowledge_entries': len(self.sbert_retriever.knowledge_data),
            'mode': 'hybrid_retrieval_reranking_lecturer_with_enhanced_memory',  # ✅ Updated
            'memory_sessions': gemini_status.get('memory_sessions', 0),
            'personalization_sessions': gemini_status.get('personalization_sessions', 0),
            'adaptive_token_range': self.response_generator.token_manager.adaptive_token_range,
            'confidence_thresholds': self.decision_engine.confidence_thresholds,
            'hybrid_reranking': {
                'enabled': True,
                'alpha': self.reranker.alpha,
                'beta': self.reranker.beta,
                'intent_categories': len(self.reranker.intent_keywords)
            },
            'lecturer_features': [
                'hybrid_retrieval_reranking', 'semantic_keyword_fusion', 'context_aware_boosting',
                'intent_based_reranking', 'lecturer_keyword_detection', 'clarification_requests', 
                'department_suggestions', 'formal_addressing', 'enhanced_generation_boost',
                'qa_management_integration', 'external_api_integration', 'jwt_token_authentication',
                'lecturer_schedule_access', 'personal_information_queries', 'user_memory_prompt_support',
                'flexible_personalization', 'dynamic_system_prompts', 'custom_user_instructions',
                'gender_based_addressing', 'no_fallback_addressing',
                'session_memory_integration',  # ✅ NEW feature
                'context_driven_api_decisions',  # ✅ NEW feature
                'enhanced_conversation_continuity',  # ✅ NEW feature
                'smart_clarification_reduction'  # ✅ NEW feature
            ],
            'gemini_status': gemini_status,
            'external_api_status': external_api_status,
            'qa_management_status': qa_management_status,
            'enhanced_features': {  # ✅ NEW section
                'session_memory_depth': 3,
                'context_recency_limit': self.decision_engine.external_api_config['context_recency_limit'],
                'context_memory_threshold': self.decision_engine.external_api_config['context_memory_threshold'],
                'schedule_continuation_keywords': len(self.decision_engine.external_api_config['schedule_continuation_keywords'])
            }
        }

    def _is_valid_short_query(self, query):
        """Kiểm tra query ngắn có hợp lệ không"""
        if not query:
            return False
            
        query_clean = query.strip().lower()
        
        valid_short_words = [
            'hi', 'hello', 'chào', 'xin chào', 'ok', 'okay', 'được', 'ừ', 'uh', 'uhm',
            'dạ', 'vâng', 'yes', 'no', 'không', 'à', 'ờ', 'ô', 'ơ', 'hả', 'hả?',
            'cảm ơn', 'thanks', 'thank you', 'cam on', 'tạm biệt', 'bye', 'goodbye', 'tam biet'
        ]
        
        for valid_word in valid_short_words:
            if query_clean == valid_word or query_clean.startswith(valid_word + ' '):
                return True
        
        greeting_patterns = [
            r'^(xin )?chào( .+)?',
            r'^hi( .+)?',
            r'^hello( .+)?'
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, query_clean):
                return True
        
        return False
    
    def process_query(self, query, session_id=None, jwt_token=None):
        """🚀 NÂNG CẤP: Main query processing với Enhanced Session Memory Integration"""
        start_time = time.time()
        
        logger.info(f"👨‍🏫 Processing ENHANCED hybrid query: '{query}' (session: {session_id}, has_token: {bool(jwt_token)})")
        
        try:
            # VALIDATE INPUT NGAY TỪ ĐẦU
            query = self._clean_query(query)
            if not query:
                return self._get_empty_query_response_lecturer()
            
            # 🚀 NEW: Get session memory EARLY để sử dụng trong decision making
            session_memory = self.get_conversation_context(session_id) if session_id else []
            logger.info(f"🧠 MEMORY STATUS: {len(session_memory)} interactions in history")
            
            # Kiểm tra query quá ngắn và không hợp lệ
            if len(query.strip()) < 3 and not self._is_valid_short_query(query):
                logger.info(f"🚫 EARLY VALIDATION: Query too short and invalid: '{query}'")
                return {
                    'response': self._get_personal_short_clarification_response(session_id),
                    'confidence': 0.1,
                    'method': 'early_validation_failed',
                    'decision_type': 'ask_clarification',
                    'processing_time': time.time() - start_time,
                    'is_education': True,
                    'lecturer_optimized': True,
                    'early_validation_triggered': True,
                    'session_memory_used': bool(session_memory)  # ✅ NEW
                }
            
            # Get intent and entities
            intent_result = self.intent_classifier.classify_intent(query)
            entities = self.intent_classifier.extract_entities(query)
            
            # HYBRID RETRIEVAL & RE-RANKING
            candidates = self.sbert_retriever.semantic_search_top_k(query, top_k=5)
            
            if not candidates:
                logger.warning("⚠️ No candidates found from semantic search")
                return self._get_no_match_response()
            
            # Re-rank candidates using hybrid approach
            reranked_candidates = self.reranker.rerank(candidates, intent_result)
            
            if not reranked_candidates:
                logger.warning("⚠️ No candidates after re-ranking")
                return self._get_no_match_response()
            
            # Get best candidate after re-ranking
            best_candidate = reranked_candidates[0]
            
            logger.info(f"🎯 Best candidate after re-ranking: final_score={best_candidate.get('final_score', 0):.3f}")
            
            # 🚀 ENHANCED DECISION MAKING với session memory
            decision_type, gemini_context, should_respond = self.decision_engine.make_decision(
                query, best_candidate, intent_result, session_memory, jwt_token
            )
            
            # Execute decision
            if not should_respond:
                response_text = self._get_personal_out_of_scope_response(session_id)
                method = 'rejected_non_education'
            else:
                response_text = self._execute_lecturer_decision(
                    decision_type, query, gemini_context, intent_result, entities, session_id
                )
                method = decision_type
            
            # 🚀 ENHANCED: Update memory với richer context information
            if session_id and should_respond:
                self._update_enhanced_memory(
                    session_id, query, intent_result, 
                    best_candidate.get('final_score', 0), 
                    decision_type, should_respond, 
                    gemini_context  # ✅ NEW: Pass full context
                )
            
            processing_time = time.time() - start_time
            
            return {
                'response': response_text,
                'confidence': best_candidate.get('final_score', 0),
                'method': method,
                'decision_type': decision_type,
                'intent': intent_result,
                'sources': self._format_sources(reranked_candidates[:2]),
                'entities': entities,
                'processing_time': processing_time,
                'is_education': gemini_context is not None,
                'generation_boosted': gemini_context.get('generation_boosted', False) if gemini_context else False,
                'lecturer_optimized': True,
                'reference_links': best_candidate.get('reference_links', []),
                'external_api_used': decision_type == 'use_external_api',
                'hybrid_reranking_used': True,
                'session_memory_used': bool(session_memory),  # ✅ NEW
                'enhanced_by_context': gemini_context.get('enhanced_by_context', False) if gemini_context else False,  # ✅ NEW
                'reranking_stats': {
                    'semantic_score': best_candidate.get('semantic_score', 0),
                    'keyword_score': best_candidate.get('keyword_score', 0),
                    'context_boost': best_candidate.get('context_boost', 0),
                    'final_score': best_candidate.get('final_score', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced hybrid processing error: {str(e)}")
            return {
                'response': self._get_personal_error_response(session_id),
                'confidence': 0.0,
                'method': 'error_fallback',
                'processing_time': time.time() - start_time,
                'error': str(e),
                'session_memory_used': bool(session_memory) if 'session_memory' in locals() else False  # ✅ NEW
            }
    
    def _get_personal_address(self, session_id):
        """Helper method để lấy personal address từ response generator"""
        if hasattr(self.response_generator, '_get_personal_address'):
            return self.response_generator._get_personal_address(session_id)
        return "giảng viên"  # ✅ FIXED: Default to neutral instead of "thầy/cô"
    
    def _get_personal_short_clarification_response(self, session_id):
        """Response for short invalid queries với personalization"""
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
    
    def _get_personal_out_of_scope_response(self, session_id):
        """Out of scope response với personalization"""
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em chỉ hỗ trợ các vấn đề liên quan đến công việc giảng viên tại BDU thôi ạ! 🎓 {personal_address.title()} có câu hỏi nào khác về trường không ạ?"
    
    def _get_personal_error_response(self, session_id):
        """Error response với personalization"""
        personal_address = self._get_personal_address(session_id)
        return f"Dạ {personal_address}, em gặp khó khăn kỹ thuật. {personal_address.title()} có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎓"
    
    def _get_no_match_response(self):
        """Response when no matches found"""
        return {
            'response': "Dạ giảng viên, em chưa có thông tin về vấn đề này. Giảng viên có thể liên hệ phòng ban liên quan để được hỗ trợ chi tiết ạ. 🎓",
            'confidence': 0.1,
            'method': 'no_match_hybrid',
            'decision_type': 'say_dont_know',
            'processing_time': 0.01,
            'hybrid_reranking_used': True
        }
    
    def _execute_lecturer_decision(self, decision_type, query, gemini_context, intent_result, entities, session_id):
        """Execute lecturer-specific decisions với gender-based addressing"""
        
        logger.info(f"🎯 Executing enhanced hybrid decision: {decision_type}")
        
        response_text = ""
        
        # Lấy response từ Gemini như bình thường
        if decision_type == 'use_external_api':
            response_text = self._handle_external_api_decision(query, gemini_context, intent_result, entities, session_id)
        
        elif decision_type == 'require_authentication':
            response_text = self._handle_authentication_required(session_id)
        
        elif decision_type == 'use_db_direct':
            response = self.response_generator.generate_response(
                query=query, context=gemini_context, intent_info=intent_result, entities=entities, session_id=session_id
            )
            personal_address = self._get_personal_address(session_id)
            response_text = response.get('response', f"Dạ {personal_address}, {gemini_context.get('db_answer', '')} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?")
        
        elif decision_type == 'enhance_db_answer':
            is_boosted = gemini_context.get('generation_boosted', False)
            enhanced_context = gemini_context.copy()
            if is_boosted:
                logger.info(f"🚀 HYBRID GENERATION BOOST: Using enhanced generation")
                enhanced_context['instruction'] = 'enhance_answer_lecturer_boosted'
            
            response = self.response_generator.generate_response(
                query=query, context=enhanced_context, intent_info=intent_result, entities=entities, session_id=session_id
            )
            personal_address = self._get_personal_address(session_id)
            response_text = response.get('response', f"Dạ {personal_address}, {gemini_context.get('db_answer', '')} 🎓 {personal_address.title()} có cần hỗ trợ thêm gì không ạ?")
        
        elif decision_type == 'ask_clarification':
            response = self.response_generator.generate_response(
                query=query, context=gemini_context, intent_info=intent_result, entities=entities, session_id=session_id
            )
            response_text = response.get('response', self._get_default_clarification_request(query, session_id))
        
        elif decision_type == 'say_dont_know':
            response = self.response_generator.generate_response(
                query=query, context=gemini_context, intent_info=intent_result, entities=entities, session_id=session_id
            )
            response_text = response.get('response', self._get_default_dont_know_response(query, session_id))
        
        else:
            logger.warning(f"⚠️ Unknown decision type: {decision_type}")
            personal_address = self._get_personal_address(session_id)
            response_text = f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"

        # Debugging personalization filter (keep existing logic)
        print("\n--- DEBUGGING PERSONALIZATION FILTER ---")
        print(f"1. Raw response from Gemini: '{response_text}'")

        try:
            user_context = self.response_generator.get_user_context(session_id)
            if user_context and response_text:
                memory_prompt = user_context.get('preferences', {}).get('user_memory_prompt', '').lower()
                print(f"2. Memory Prompt found for this session: '{memory_prompt}'")
                
                if 'desuwa' in memory_prompt:
                    print("3. 'desuwa' keyword FOUND! Applying hard-coded override...")
                    
                    # Cắt bỏ các đuôi câu mặc định
                    default_endings = [
                        "có cần em hỗ trợ thêm gì không ạ? 🎓",
                        "có cần em hỗ trợ thêm gì không ạ?",
                        "ạ. 🎓", "ạ?", "ạ."
                    ]
                    processed_text = response_text
                    for ending in default_endings:
                        if processed_text.rstrip().endswith(ending.rstrip()):
                            processed_text = processed_text.rstrip()[:-len(ending.rstrip())].strip()
                            break
                    
                    final_response = processed_text + " desuwa"
                    print(f"4. Final response after override: '{final_response}'")
                    print("--- END DEBUGGING (Override applied) ---\n")
                    return final_response
                else:
                    print("3. 'desuwa' keyword NOT FOUND in memory prompt. No override applied.")
            else:
                print("2. No user_context or empty response. Skipping filter.")

        except Exception as e:
            logger.error(f"Error during final personalization override: {e}")
            print(f"!!! ERROR during final filter: {e}")
            print("--- END DEBUGGING (Error occurred) ---\n")
            return response_text

        print("--- END DEBUGGING (No override applied) ---\n")
        return response_text
    
    def _handle_external_api_decision(self, query, gemini_context, intent_result, entities, session_id):
        """Handle decision to use external API với gender support"""
        try:
            jwt_token = gemini_context.get('jwt_token')
            
            logger.info("🌐 Calling external API service for lecturer schedule/info")
            
            api_result = external_api_service.get_lecturer_schedule(jwt_token, query)
            
            if api_result.get('success'):
                enhanced_context = {
                    'instruction': 'process_external_api_data',
                    'api_data': api_result,
                    'original_query': query,
                    'fallback_qa_answer': gemini_context.get('fallback_qa_answer', ''),
                    'confidence': gemini_context.get('confidence', 0)
                }
                
                response = self.response_generator.generate_response(
                    query=query,
                    context=enhanced_context,
                    intent_info=intent_result,
                    entities=entities,
                    session_id=session_id
                )
                
                return response.get('response', self._get_external_api_fallback(api_result, session_id))
            
            else:
                error_type = api_result.get('error_type', 'unknown')
                return self._get_external_api_error_response(error_type, session_id, gemini_context.get('fallback_qa_answer', ''))
                
        except Exception as e:
            logger.error(f"❌ Error handling external API decision: {str(e)}")
            personal_address = self._get_personal_address(session_id)
            return f"Dạ {personal_address}, em gặp khó khăn khi truy xuất thông tin cá nhân. {personal_address.title()} có thể thử lại sau hoặc liên hệ bộ phận IT để được hỗ trợ ạ. 🎓"
    
    def _handle_authentication_required(self, session_id):
        """Handle case where external API is needed but no token provided"""
        personal_address = self._get_personal_address(session_id)
        return f"""Dạ {personal_address}, để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng trước ạ. 🔐

{personal_address.title()} có thể:
• Đăng nhập lại vào ứng dụng BDU
• Kiểm tra kết nối mạng
• Liên hệ bộ phận IT nếu gặp khó khăn: it@bdu.edu.vn

Sau khi đăng nhập, {personal_address} có thể hỏi lại em về lịch giảng dạy nhé! 🎓"""
    
    def _get_external_api_fallback(self, api_result, session_id):
        """Get fallback response when external API data is available but Gemini fails"""
        lecturer_info = api_result.get('lecturer_info', {})
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        
        # Determine personal address from API data or session
        if ten_giang_vien:
            gender = lecturer_info.get('gender', 'other')
            if gender == 'male':
                salutation = 'thầy'
            elif gender == 'female':
                salutation = 'cô'
            else:
                salutation = 'giảng viên'
                
            if salutation in ['thầy', 'cô']:
                name_suffix = ten_giang_vien.split()[-1] if ten_giang_vien else ''
                personal_address = f"{salutation} {name_suffix}" if name_suffix else salutation
            else:
                personal_address = f"{salutation} {ten_giang_vien}" if ten_giang_vien else salutation
        else:
            personal_address = self._get_personal_address(session_id)
        
        schedule_summary = api_result.get('schedule_summary', {})
        total_classes = schedule_summary.get('total_classes', 0)
        
        return f"""Dạ {personal_address}, em đã tìm thấy thông tin lịch giảng dạy của {personal_address} với {total_classes} buổi học. 

Tuy nhiên em gặp khó khăn trong việc trình bày chi tiết. {personal_address.title()} có thể:
• Truy cập hệ thống quản lý đào tạo của trường
• Liên hệ phòng Đào tạo để được hỗ trợ
• Thử hỏi lại với câu hỏi cụ thể hơn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
    
    def _get_external_api_error_response(self, error_type, session_id, fallback_qa=''):
        """Get appropriate error response based on error type với gender support"""
        personal_address = self._get_personal_address(session_id)
        
        if error_type == 'token_decode_failed':
            return f"""Dạ {personal_address}, phiên đăng nhập đã hết hạn. {personal_address.title()} vui lòng đăng nhập lại vào ứng dụng BDU để em có thể hỗ trợ thông tin cá nhân ạ. 🔐

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        elif error_type == 'authentication_failed':
            return f"""Dạ {personal_address}, thông tin đăng nhập không hợp lệ hoặc đã hết hạn. {personal_address.title()} vui lòng:
• Đăng xuất và đăng nhập lại
• Kiểm tra kết nối mạng
• Liên hệ bộ phận IT nếu vẫn gặp khó khăn: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        elif error_type == 'network_error':
            return f"""Dạ {personal_address}, hiện tại có vấn đề kết nối đến hệ thống của trường. {personal_address.title()} vui lòng:
• Kiểm tra kết nối mạng
• Thử lại sau vài phút
• Liên hệ bộ phận IT nếu vấn đề kéo dài: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        else:
            if fallback_qa:
                return f"""Dạ {personal_address}, em gặp khó khăn khi truy xuất thông tin cá nhân, nhưng em có thể chia sẻ thông tin chung: {fallback_qa}

Để biết thông tin cá nhân chi tiết, {personal_address} có thể truy cập hệ thống quản lý đào tạo của trường ạ. 🎓

{personal_address.title()} có cần hỗ trợ thêm gì không ạ?"""
            else:
                return f"""Dạ {personal_address}, em gặp khó khăn kỹ thuật khi truy xuất thông tin. {personal_address.title()} có thể:
• Thử lại sau vài phút
• Truy cập trực tiếp hệ thống quản lý đào tạo
• Liên hệ bộ phận IT: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
    
    def _get_default_dont_know_response(self, query, session_id):
        """Default don't know response với department suggestion và gender support"""
        personal_address = self._get_personal_address(session_id)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ngân hàng đề', 'đề thi', 'khảo thí']):
            return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ Phòng Đảm bảo chất lượng và Khảo thí qua email ldkham@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        elif any(word in query_lower for word in ['kê khai', 'nhiệm vụ', 'giờ chuẩn']):
            return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ Phòng Tổ chức - Cán bộ qua email tcccb@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        elif any(word in query_lower for word in ['tạp chí', 'nghiên cứu', 'khoa học']):
            return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ Phòng Nghiên cứu - Hợp tác qua email nghiencuu@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
        else:
            return f"Dạ {personal_address}, em chưa có thông tin về vấn đề này. {personal_address.title()} có thể liên hệ phòng ban liên quan qua email info@bdu.edu.vn để được hỗ trợ chi tiết ạ. 🎓"
    
    def _clean_query(self, query):
        """Clean and prepare query for lecturers"""
        if not query:
            return ""
        
        query = re.sub(r'\s+', ' ', query.strip())
        query = re.sub(r'[?]{2,}', '?', query)
        query = re.sub(r'[!]{2,}', '!', query)
        
        return query
    
    def _update_enhanced_memory(self, session_id, query, intent_result, confidence, decision_type=None, was_education=True, gemini_context=None):
        """🚀 NÂNG CẤP: Enhanced memory update với richer context information"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        # ✅ ENHANCED: Store more context information
        interaction = {
            'query': query,
            'intent_info': intent_result,
            'confidence': confidence,
            'timestamp': time.time(),
            'user_type': 'lecturer',
            'decision_type': decision_type,
            'was_education_related': was_education,
            'is_education_query': self.decision_engine.is_education_related(query),
            'hybrid_processed': True,
            # ✅ NEW: Additional context fields
            'enhanced_by_context': gemini_context.get('enhanced_by_context', False) if gemini_context else False,
            'external_api_used': decision_type == 'use_external_api',
            'generation_boosted': gemini_context.get('generation_boosted', False) if gemini_context else False,
            'query_length': len(query.split()),
            'intent_confidence': intent_result.get('confidence', 0) if intent_result else 0
        }
        
        self.conversation_memory[session_id].append(interaction)
        
        # Keep only recent history (increased to 15 for better context)
        self.conversation_memory[session_id] = self.conversation_memory[session_id][-15:]
        
        logger.info(f"🧠 ENHANCED Memory updated for session {session_id}: {len(self.conversation_memory[session_id])} total interactions")
    
    def _get_empty_query_response_lecturer(self):
        """Response for empty queries from lecturers"""
        return {
            'response': "Dạ chào giảng viên! Em có thể hỗ trợ gì cho giảng viên về công việc tại BDU ạ? 🎓",
            'confidence': 0.9,
            'method': 'empty_query_lecturer',
            'processing_time': 0.01,
            'hybrid_reranking_used': False
        }
    
    def get_conversation_context(self, session_id):
        """Get conversation context for a lecturer session"""
        return self.conversation_memory.get(session_id, [])
    
    def get_conversation_memory(self, session_id):
        """Get conversation memory from Gemini service"""
        return self.response_generator.get_conversation_memory(session_id)
    
    def clear_conversation_memory(self, session_id=None):
        """Clear conversation memory"""
        if session_id:
            self.response_generator.clear_conversation_memory(session_id)
            if session_id in self.conversation_memory:
                del self.conversation_memory[session_id]
        else:
            self.response_generator.clear_conversation_memory()
            self.conversation_memory.clear()

    def _get_default_clarification_request(self, query, session_id):
        """Default clarification request if Gemini fails với gender support"""
        personal_address = self._get_personal_address(session_id)
        query_words = query.lower().split()
        
        topic_keywords = {
            'ngân hàng đề thi': ['ngân hàng', 'đề thi', 'đề'],
            'kê khai nhiệm vụ': ['kê khai', 'nhiệm vụ'],
            'tạp chí': ['tạp chí', 'bài viết'],
            'thi đua khen thưởng': ['thi đua', 'khen thưởng'],
            'báo cáo': ['báo cáo', 'nộp'],
            'lịch giảng dạy': ['lịch', 'giảng dạy']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_words for kw in keywords):
                return f"Dạ {personal_address}, để em hỗ trợ chính xác về {topic}, {personal_address} có thể nói rõ hơn về nội dung cụ thể cần hỗ trợ không ạ? 🎓"
        
        return f"Dạ {personal_address}, để em hỗ trợ chính xác nhất, {personal_address} có thể nói rõ hơn về vấn đề cần hỗ trợ không ạ? 🎓"
    
    def reload_after_qa_update(self):
        """Reload knowledge base after QA Management updates"""
        logger.info("🔄 Reloading hybrid knowledge base after QA Management update...")
        
        if hasattr(self.sbert_retriever, 'cached_data'):
            self.sbert_retriever.cached_data = None
            self.sbert_retriever.cache_timestamp = 0
        
        self.sbert_retriever.load_knowledge_base()
        
        if self.sbert_retriever.model and self.sbert_retriever.knowledge_data:
            self.sbert_retriever.build_faiss_index()
        
        logger.info("✅ Hybrid knowledge base reloaded successfully")
    
    def _format_sources(self, results):
        """Format sources for display with hybrid scores"""
        sources = []
        for result in results:
            if result and result.get('final_score', 0) > 0.2:
                sources.append({
                    'question': result['question'],
                    'category': result.get('category', 'Giảng viên'),
                    'final_score': result.get('final_score', 0),
                    'semantic_score': result.get('semantic_score', 0),
                    'keyword_score': result.get('keyword_score', 0)
                })
        return sources


# ChatbotAI class with top-k semantic search (unchanged, keeping original)
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
            self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
            logger.info("✅ Vietnamese SBERT loaded for lecturers")
            
            if self.vietnamese_restorer:
                 logger.info("✅ Vietnamese Restorer linked successfully for search.")
            else:
                 logger.warning("⚠️ Vietnamese Restorer not available.")
            
            self.load_knowledge_base()
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.model = None
    
    def load_link_mapping(self):
        """Load link mapping với reduced logging"""
        try:
            link_csv_path = os.path.join(settings.BASE_DIR, 'data', 'link.csv')
            logger.info(f"🔗 Loading reference links from: {link_csv_path}")
            
            if os.path.exists(link_csv_path):
                df_links = pd.read_csv(link_csv_path, encoding='utf-8')
                logger.info(f"🔗 CSV loaded successfully. Shape: {df_links.shape}")
                
                for index, row in df_links.iterrows():
                    stt = str(row['STT']).strip()
                    link = str(row['Link']).strip()
                    
                    if stt and link and stt != 'nan' and link != 'nan':
                        self.link_mapping[stt] = link
                
                logger.info(f"✅ Total loaded: {len(self.link_mapping)} reference links")
                
            else:
                logger.error(f"❌ link.csv not found at {link_csv_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading link mapping: {str(e)}")
            self.link_mapping = {}
    
    def get_reference_links(self, qa_item):
        """Get reference links với reduced logging"""
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
                logger.debug(f"✅ FOUND reference link: STT '{stt}' -> '{link_url}'")
        
        return reference_links

    def load_knowledge_base(self):
        """Enhanced knowledge base loading with QA Management integration"""
        try:
            self.load_link_mapping()
            
            # Load from QA Management database (highest priority)
            db_qa_entries = []
            try:
                from qa_management.models import QAEntry
                qa_entries = QAEntry.objects.filter(is_active=True).order_by('stt')
                
                for entry in qa_entries:
                    db_qa_entries.append({
                        'question': entry.question,
                        'answer': entry.answer,
                        'category': entry.category or 'Giảng viên',
                        'STT': entry.stt,
                    })
                logger.info(f"✅ Loaded {len(db_qa_entries)} entries from QA Management database")
            except Exception as e:
                logger.warning(f"⚠️ QA Management not available or no data: {str(e)}")
            
            # Load from legacy database
            db_knowledge = list(KnowledgeBase.objects.filter(is_active=True).values(
                'question', 'answer', 'category'
            ))
            
            # Load from Google Drive (fallback/backup)
            csv_knowledge = []
            try:
                csv_knowledge = drive_service.get_csv_data()
                if csv_knowledge:
                    logger.info(f"✅ Loaded {len(csv_knowledge)} records from Google Drive (backup/fallback)")
                else:
                    logger.warning("⚠️ No data from Google Drive, using empty list")
                    csv_knowledge = []
            except Exception as e:
                logger.error(f"❌ Failed to load from Google Drive: {str(e)}")
                csv_knowledge = []
            
            # Fallback to local CSV
            if not csv_knowledge and not db_qa_entries:
                logger.info("🔄 Attempting fallback to local CSV")
                csv_path = os.path.join(settings.BASE_DIR, 'data', 'QA.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8')
                        if 'question' in df.columns and 'answer' in df.columns:
                            csv_knowledge = df.fillna('').to_dict('records')
                            logger.info(f"✅ Fallback: Loaded {len(csv_knowledge)} records from local CSV")
                    except Exception as e:
                        logger.error(f"❌ Fallback CSV also failed: {str(e)}")
                        csv_knowledge = []
            
            # Priority: QA Management DB > Drive CSV > Legacy DB
            self.knowledge_data = db_qa_entries + csv_knowledge + db_knowledge
            
            # Build FAISS index
            if self.model and self.knowledge_data:
                self.build_faiss_index()
            
            logger.info(f"✅ Total loaded: {len(self.knowledge_data)} knowledge entries for lecturers")
            logger.info(f"   📊 QA Management: {len(db_qa_entries)} entries")
            logger.info(f"   📊 Google Drive: {len(csv_knowledge)} entries") 
            logger.info(f"   📊 Legacy DB: {len(db_knowledge)} entries")
            logger.info(f"✅ Reference links available: {len(self.link_mapping)} links")
            
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
            self.knowledge_data = self.get_fallback_knowledge_lecturer()

    def get_fallback_knowledge_lecturer(self):
        """Fallback knowledge data specifically for lecturers"""
        return [
            {
                'question': 'ngân hàng đề thi',
                'answer': 'Giảng viên cần báo cáo kết quả xây dựng ngân hàng đề thi kết thúc học phần và lập kế hoạch cho học kỳ tiếp theo. Nộp về Phòng Đảm bảo chất lượng và Khảo thí qua email ldkham@bdu.edu.vn trước hạn quy định.',
                'category': 'Giảng viên',
                'STT': '1'
            },
            {
                'question': 'kê khai nhiệm vụ năm học',
                'answer': 'Giảng viên cơ hữu và thỉnh giảng cần kê khai nhiệm vụ năm học bao gồm giảng dạy, nghiên cứu khoa học và các hoạt động khác. Khoa tổng hợp và báo cáo lên nhà trường.',
                'category': 'Giảng viên',
                'STT': '2'
            },
            {
                'question': 'tạp chí khoa học',
                'answer': 'Tạp chí Khoa học và Công nghệ Trường Đại học Bình Dương nhận bài viết từ giảng viên, nghiên cứu sinh và các nhà khoa học. Gửi bài qua email chỉ định của tòa soạn.',
                'category': 'Giảng viên',
                'STT': '3'
            },
            {
                'question': 'thi đua khen thưởng',
                'answer': 'Nhà trường tổ chức đánh giá thi đua, khen thưởng cá nhân và tập thể xuất sắc trong năm học. Có các danh hiệu như Chiến sĩ thi đua, Lao động tiên tiến...',
                'category': 'Giảng viên',
                'STT': '4'
            }
        ]
    
    def build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        try:
            questions = [item['question'] for item in self.knowledge_data]
            embeddings = self.model.encode(questions)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index built with {len(questions)} entries for lecturers")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.index = None
    
    def semantic_search_top_k(self, query, top_k=5):
        """Enhanced semantic search returning top-k candidates"""
        try:
            if not self.model or not self.index:
                logger.warning("⚠️ Model or index not available, falling back to keyword search")
                return self.keyword_search_top_k(query, top_k)
            
            # Restore Vietnamese if needed
            original_query = query
            if self.vietnamese_restorer and not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Using restored query for hybrid search: '{query}' -> '{restored_query}'")
                    query = restored_query
            
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Get top_k results instead of just 1
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.knowledge_data)))
            
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_data) and score > 0.1:  # Minimum threshold
                    candidate = self.knowledge_data[idx].copy()
                    candidate['semantic_score'] = float(score)
                    candidate['similarity'] = float(score)  # Backward compatibility
                    candidate['reference_links'] = self.get_reference_links(candidate)
                    candidates.append(candidate)
            
            logger.info(f"🔍 Semantic search found {len(candidates)} candidates for hybrid re-ranking")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return self.keyword_search_top_k(query, top_k)
    
    def keyword_search_top_k(self, query, top_k=5):
        """Enhanced keyword search returning top-k candidates"""
        query_words = set(query.lower().split())
        candidates = []
        
        for item in self.knowledge_data:
            question_words = set(item['question'].lower().split())
            answer_words = set(item['answer'].lower().split())
            
            question_common = query_words & question_words
            answer_common = query_words & answer_words
            
            if question_common or answer_common:
                question_score = len(question_common) / len(query_words | question_words) * 2
                answer_score = len(answer_common) / len(query_words | answer_words)
                
                total_score = question_score + answer_score
                
                candidate = item.copy()
                candidate['semantic_score'] = total_score
                candidate['similarity'] = total_score  # Backward compatibility
                candidate['reference_links'] = self.get_reference_links(candidate)
                candidates.append(candidate)
        
        # Sort by score and return top_k
        candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
        return candidates[:top_k]
    
    def semantic_search(self, query, top_k=3):
        """Maintains backward compatibility while using the new hybrid approach"""
        candidates = self.semantic_search_top_k(query, top_k)
        
        if candidates:
            best_match = candidates[0]
            return best_match, candidates
        else:
            return None, []
    
    def generate_response(self, query):
        """Generate response optimized for lecturer hybrid system"""
        try:
            if not query.strip():
                return {
                    'response': 'Dạ giảng viên, vui lòng nhập câu hỏi cụ thể ạ. 🎓',
                    'confidence': 0.1,
                    'method': 'empty_query',
                    'sources': [],
                    'reference_links': []
                }
            
            # Use hybrid approach
            candidates = self.semantic_search_top_k(query, top_k=3)
            
            if candidates:
                best_candidate = candidates[0]
                similarity = best_candidate.get('semantic_score', 0)
                
                # Collect reference links from top results
                all_reference_links = []
                for candidate in candidates:
                    if candidate and 'reference_links' in candidate:
                        all_reference_links.extend(candidate['reference_links'])
                
                # Remove duplicates
                unique_links = {}
                for link in all_reference_links:
                    stt = link['stt']
                    if stt not in unique_links:
                        unique_links[stt] = link
                
                final_links = list(unique_links.values())
                
                return {
                    'response': best_candidate['answer'],
                    'confidence': similarity,
                    'method': 'hybrid_retrieval',
                    'sources': self._format_sources(candidates[:2]),
                    'category': best_candidate.get('category', 'Giảng viên'),
                    'reference_links': final_links
                }
            else:
                return {
                    'response': 'Em chưa có thông tin về vấn đề này.',
                    'confidence': 0.1,
                    'method': 'no_match',
                    'sources': [],
                    'reference_links': []
                }
            
        except Exception as e:
            logger.error(f"Generate response error: {str(e)}")
            return {
                'response': 'Đã có lỗi xảy ra trong hệ thống.',
                'confidence': 0.1,
                'method': 'error',
                'sources': [],
                'reference_links': []
            }
    
    def _format_sources(self, results):
        """Format sources for display"""
        sources = []
        for result in results:
            if result and result.get('semantic_score', 0) > 0.2:
                sources.append({
                    'question': result['question'],
                    'category': result.get('category', 'Giảng viên'),
                    'similarity': result.get('semantic_score', 0)
                })
        return sources


class BDUChatbotService:
    """🚀 NÂNG CẤP: Enhanced Primary Service Layer với Context Memory Integration"""
    
    def __init__(self):
        # Tạo shared response_generator trước tiên
        self.response_generator = GeminiResponseGenerator()
        
        # Truyền shared response_generator vào hybrid_chatbot
        self.hybrid_chatbot = HybridChatbotAI(shared_response_generator=self.response_generator)
        
        self.intent_classifier = PhoBERTIntentClassifier()
        
        # 🚀 ENHANCED: API priority configuration with memory awareness
        self.api_priority_config = {
            'personal_info_keywords': [
                'lịch của tôi', 'lich cua toi', 'thời khóa biểu của tôi', 'tkb của tôi',
                'lịch giảng của tôi', 'lich giang cua toi', 'lịch dạy của tôi', 'lich day cua toi',
                'tôi giảng', 'toi giang', 'tôi dạy', 'toi day', 'môn của tôi', 'mon cua toi',
                'lớp của tôi', 'lop cua toi', 'phòng của tôi', 'phong cua toi',
                'hôm nay tôi', 'hom nay toi', 'ngày mai tôi', 'ngay mai toi',
                'tuần này tôi', 'tuan nay toi', 'tuần tới tôi', 'tuan toi toi',
                'tôi là ai', 'toi la ai', 'thông tin của tôi', 'thong tin cua toi',
                'tôi làm gì', 'toi lam gi', 'công việc của tôi', 'cong viec cua toi',
                'chức danh của tôi', 'chuc danh cua toi', 'vị trí của tôi', 'vi tri cua toi',
                'email của tôi', 'gmail của tôi', 'số điện thoại của tôi',
                'lịch giảng dạy', 'lich giang day', 'thời khóa biểu', 'thoi khoa bieu',
                'lịch học', 'lich hoc', 'lịch dạy', 'lich day', 'tkb', 'schedule',
                'lịch tuần', 'lich tuan', 'lịch ngày', 'lich ngay'
            ],
            'time_context_keywords': [
                'hôm nay', 'hom nay', 'today', 'ngày mai', 'ngay mai', 'tomorrow',
                'tuần này', 'tuan nay', 'this week', 'tuần tới', 'tuan toi', 'next week',
                'thứ 2', 'thu 2', 'thứ 3', 'thu 3', 'thứ 4', 'thu 4', 'thứ 5', 'thu 5',
                'thứ 6', 'thu 6', 'thứ 7', 'thu 7', 'chủ nhật', 'chu nhat',
                'tuần sau', 'tuan sau', 'cuối tuần', 'cuoi tuan', 'đầu tuần', 'dau tuan'
            ],
            'schedule_intent_names': [
                'personal_schedule', 'teaching_schedule', 'schedule_general', 'personal_info'
            ],
            # 🚀 NEW: Context-aware settings
            'context_continuation_keywords': [
                'còn', 'con', 'thêm', 'them', 'nữa', 'nua', 'khác', 'khac', 
                'và', 'va', 'tiếp theo', 'tiep theo', 'sau đó', 'sau do',
                'thế còn', 'the con', 'vậy còn', 'vay con', 'còn gì', 'con gi'
            ],
            'memory_lookback_limit': 3,  # Look back 3 interactions
            'schedule_intent_confidence_threshold': 0.6
        }
        
        logger.info("🚀 Enhanced BDUChatbotService initialized with Context Memory Integration")
    
    def _needs_external_api(self, query: str, intent_result: dict, session_memory: list = None) -> bool:
        """🚀 NÂNG CẤP: Enhanced API need detection với session memory context"""
        if not query:
            return False
        
        query_lower = query.lower()
        
        # ✅ CHECK 1: Direct personal keyword matches (unchanged)
        has_personal_keywords = any(
            keyword in query_lower 
            for keyword in self.api_priority_config['personal_info_keywords']
        )
        
        # ✅ CHECK 2: Time context (usually indicates schedule query)
        has_time_context = any(
            keyword in query_lower 
            for keyword in self.api_priority_config['time_context_keywords']
        )
        
        # ✅ CHECK 3: Intent-based detection (unchanged)
        intent_name = intent_result.get('intent', '')
        is_schedule_intent = intent_name in self.api_priority_config['schedule_intent_names']
        
        # ✅ CHECK 4: High confidence personal intent
        intent_confidence = intent_result.get('confidence', 0)
        high_confidence_personal = (
            is_schedule_intent and intent_confidence > 0.7
        )
        
        # 🚀 NEW CHECK 5: Session Memory Context Analysis
        context_suggests_api = False
        has_continuation_words = False
        
        if session_memory and len(session_memory) > 0:
            # Analyze recent interactions for schedule-related context
            recent_interactions = session_memory[-self.api_priority_config['memory_lookback_limit']:]
            
            schedule_intent_count = 0
            for interaction in recent_interactions:
                past_intent = interaction.get('intent_info', {})
                if isinstance(past_intent, dict):
                    past_intent_name = past_intent.get('intent', '')
                    past_intent_confidence = past_intent.get('confidence', 0)
                    
                    if (past_intent_name in self.api_priority_config['schedule_intent_names'] and 
                        past_intent_confidence > self.api_priority_config['schedule_intent_confidence_threshold']):
                        schedule_intent_count += 1
            
            # If we found schedule context in recent history
            if schedule_intent_count > 0:
                context_suggests_api = True
                logger.info(f"🧠 CONTEXT API: Found {schedule_intent_count} schedule-related intents in history")
            
            # Check for continuation keywords in current query
            has_continuation_words = any(
                keyword in query_lower 
                for keyword in self.api_priority_config['context_continuation_keywords']
            )
        
        # 🚀 NEW: Context-driven API decisions
        context_driven_api = context_suggests_api and has_continuation_words
        smart_short_query_api = (
            len(query.split()) <= 5 and 
            has_time_context and 
            context_suggests_api
        )
        
        # ✅ FINAL DECISION with memory integration
        needs_api = (
            has_personal_keywords or 
            has_time_context or 
            high_confidence_personal or
            context_driven_api or  # 🚀 NEW
            smart_short_query_api  # 🚀 NEW
        )
        
        # 🚀 ENHANCED LOGGING
        logger.info(f"🔍 ENHANCED API Priority Check:")
        logger.info(f"   📝 Query: '{query[:50]}...' ({len(query.split())} words)")
        logger.info(f"   🔑 Direct: personal_kw={has_personal_keywords}, time_ctx={has_time_context}")
        logger.info(f"   🎯 Intent: is_schedule={is_schedule_intent}, high_conf={high_confidence_personal}")
        logger.info(f"   🧠 Context: suggests_api={context_suggests_api}, continuation={has_continuation_words}")
        logger.info(f"   🚀 Enhanced: context_driven={context_driven_api}, smart_short={smart_short_query_api}")
        logger.info(f"   ✅ Final: needs_api={needs_api}")
        
        return needs_api
    
    def _handle_external_api_call(self, query: str, intent_result: dict, entities: dict, session_id: str, jwt_token: str) -> dict:
        """Handle external API call and response processing"""
        try:
            logger.info("🌐 ENHANCED PRIORITY: Calling external API for personal/schedule information")
            
            # Call external API service
            api_result = external_api_service.get_lecturer_schedule(jwt_token, query)
            
            if api_result.get('success'):
                # Use Gemini to process external API data
                enhanced_context = {
                    'instruction': 'process_external_api_data',
                    'api_data': api_result,
                    'original_query': query,
                    'confidence': 0.95  # High confidence for API data
                }
                
                response = self.response_generator.generate_response(
                    query=query,
                    context=enhanced_context,
                    intent_info=intent_result,
                    entities=entities,
                    session_id=session_id
                )
                
                return {
                    'response': response.get('response', self._get_api_fallback_response(api_result, session_id)),
                    'confidence': 0.95,
                    'method': 'external_api_success',
                    'decision_type': 'use_external_api',
                    'intent': intent_result,
                    'sources': [{'question': 'External API', 'category': 'Personal Info', 'similarity': 0.95}],
                    'entities': entities,
                    'processing_time': 0.5,
                    'is_education': True,
                    'lecturer_optimized': True,
                    'external_api_used': True,
                    'hybrid_reranking_used': False,  # API call bypassed hybrid system
                    'api_priority_activated': True,   # Flag showing API priority worked
                    'enhanced_by_context': True  # 🚀 NEW flag
                }
            
            else:
                # API failed, return error response
                error_type = api_result.get('error_type', 'unknown')
                return {
                    'response': self._get_api_error_response(error_type, api_result.get('error', ''), session_id),
                    'confidence': 0.1,
                    'method': 'external_api_failed',
                    'decision_type': 'api_error',
                    'processing_time': 0.3,
                    'external_api_used': True,
                    'api_error': api_result.get('error', ''),
                    'api_priority_activated': True
                }
                
        except Exception as e:
            logger.error(f"❌ Error in external API call: {str(e)}")
            personal_address = self._get_personal_address(session_id)
            return {
                'response': f"Dạ {personal_address}, em gặp khó khăn khi truy xuất thông tin cá nhân. {personal_address.title()} có thể thử lại sau hoặc liên hệ bộ phận IT để được hỗ trợ ạ. 🎓",
                'confidence': 0.1,
                'method': 'external_api_error',
                'processing_time': 0.2,
                'error': str(e),
                'api_priority_activated': True
            }
    
    def _handle_authentication_required(self, session_id: str, has_context: bool = False) -> dict:
        """🚀 NÂNG CẤP: Handle authentication với context awareness"""
        personal_address = self._get_personal_address(session_id)
        
        if has_context:
            # More specific message when we know there's schedule context
            message = f"""Dạ {personal_address}, em hiểu {personal_address} đang hỏi tiếp về lịch giảng dạy, nhưng để cung cấp thông tin cá nhân chính xác, {personal_address} cần đăng nhập vào ứng dụng trước ạ. 🔐

{personal_address.title()} có thể:
• Đăng nhập lại vào ứng dụng BDU
• Kiểm tra kết nối mạng
• Liên hệ bộ phận IT nếu gặp khó khăn: it@bdu.edu.vn

Sau khi đăng nhập, {personal_address} có thể hỏi lại em về lịch giảng dạy nhé! 🎓"""
        else:
            # Standard message
            message = f"""Dạ {personal_address}, để em có thể cung cấp thông tin cá nhân như lịch giảng dạy, {personal_address} cần đăng nhập vào ứng dụng trước ạ. 🔐

{personal_address.title()} có thể:
• Đăng nhập lại vào ứng dụng BDU
• Kiểm tra kết nối mạng
• Liên hệ bộ phận IT nếu gặp khó khăn: it@bdu.edu.vn

Sau khi đăng nhập, {personal_address} có thể hỏi lại em về lịch giảng dạy nhé! 🎓"""
        
        return {
            'response': message,
            'confidence': 0.9,
            'method': 'authentication_required',
            'decision_type': 'require_authentication',
            'processing_time': 0.01,
            'external_api_used': False,
            'api_priority_activated': True,
            'authentication_required': True,
            'context_aware': has_context  # 🚀 NEW flag
        }
    
    def _get_personal_address(self, session_id):
        """Helper method để lấy personal address từ response generator"""
        if hasattr(self.response_generator, '_get_personal_address'):
            return self.response_generator._get_personal_address(session_id)
        return "giảng viên"  # ✅ FIXED: Default to neutral instead of "thầy/cô"
    
    def process_query(self, query: str, session_id: str = None, jwt_token: str = None) -> dict:
        """🚀 NÂNG CẤP: Main method với Enhanced Context Memory Integration"""
        start_time = time.time()
        
        logger.info(f"🎯 Enhanced BDU Service Processing: '{query}' (session: {session_id}, has_token: {bool(jwt_token)})")
        
        try:
            if not query or len(query.strip()) < 2:
                return {
                    'response': "Dạ chào giảng viên! Em có thể hỗ trợ gì cho giảng viên về công việc tại BDU ạ? 🎓",
                    'confidence': 0.9,
                    'method': 'empty_query',
                    'processing_time': time.time() - start_time
                }
            
            # 🚀 NEW: Get session memory EARLY for context-aware decisions
            session_memory = self.hybrid_chatbot.get_conversation_context(session_id) if session_id else []
            has_context = len(session_memory) > 0
            
            # Intent Classification
            intent_result = self.intent_classifier.classify_intent(query)
            entities = self.intent_classifier.extract_entities(query)
            
            # 🚀 ENHANCED API PRIORITY CHECK với session memory integration
            if self._needs_external_api(query, intent_result, session_memory):
                logger.info("🚨 ENHANCED API PRIORITY ACTIVATED: Personal/Schedule query detected with context awareness")
                
                if jwt_token and jwt_token.strip():
                    # Has token -> Call external API
                    return self._handle_external_api_call(
                        query, intent_result, entities, session_id, jwt_token
                    )
                else:
                    # No token -> Require authentication (with context awareness)
                    return self._handle_authentication_required(session_id, has_context)
            
            # FALLBACK TO ENHANCED HYBRID SYSTEM
            logger.info("📚 Using Enhanced Hybrid Retrieval with Context Memory")
            result = self.hybrid_chatbot.process_query(query, session_id, jwt_token)
            
            # Add enhanced flags to show this went through enhanced hybrid flow
            result['api_priority_activated'] = False
            result['fallback_to_enhanced_hybrid'] = True
            result['context_memory_available'] = has_context
            result['enhanced_processing'] = True  # 🚀 NEW flag
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced BDU Service Error: {str(e)}")
            personal_address = self._get_personal_address(session_id)
            return {
                'response': f"Dạ {personal_address}, em gặp khó khăn kỹ thuật. {personal_address.title()} có thể liên hệ bộ phận IT qua email it@bdu.edu.vn để được hỗ trợ ạ. 🎓",
                'confidence': 0.0,
                'method': 'service_error',
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _get_api_fallback_response(self, api_result: dict, session_id: str) -> str:
        """Fallback response when API data is available but Gemini fails với gender support"""
        lecturer_info = api_result.get('lecturer_info', {})
        ten_giang_vien = lecturer_info.get('ten_giang_vien', '')
        
        # Determine personal address từ API data hoặc session
        if ten_giang_vien:
            gender = lecturer_info.get('gender', 'other')
            if gender == 'male':
                salutation = 'thầy'
            elif gender == 'female':
                salutation = 'cô'
            else:
                salutation = 'giảng viên'
                
            if salutation in ['thầy', 'cô']:
                name_suffix = ten_giang_vien.split()[-1] if ten_giang_vien else ''
                personal_address = f"{salutation} {name_suffix}" if name_suffix else salutation
            else:
                personal_address = f"{salutation} {ten_giang_vien}" if ten_giang_vien else salutation
        else:
            personal_address = self._get_personal_address(session_id)
        
        schedule_summary = api_result.get('schedule_summary', {})
        total_classes = schedule_summary.get('total_classes', 0)
        
        return f"""Dạ {personal_address}, em đã tìm thấy thông tin lịch giảng dạy của {personal_address} với {total_classes} buổi học. 

Tuy nhiên em gặp khó khăn trong việc trình bày chi tiết. {personal_address.title()} có thể:
• Truy cập hệ thống quản lý đào tạo của trường
• Liên hệ phòng Đào tạo để được hỗ trợ
• Thử hỏi lại với câu hỏi cụ thể hơn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
    
    def _get_api_error_response(self, error_type: str, error_message: str, session_id: str) -> str:
        """Get appropriate error response based on error type với gender support"""
        personal_address = self._get_personal_address(session_id)
        
        if error_type == 'token_decode_failed':
            return f"""Dạ {personal_address}, phiên đăng nhập đã hết hạn. {personal_address.title()} vui lòng đăng nhập lại vào ứng dụng BDU để em có thể hỗ trợ thông tin cá nhân ạ. 🔐

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        elif error_type == 'authentication_failed':
            return f"""Dạ {personal_address}, thông tin đăng nhập không hợp lệ hoặc đã hết hạn. {personal_address.title()} vui lòng:
• Đăng xuất và đăng nhập lại
• Kiểm tra kết nối mạng
• Liên hệ bộ phận IT nếu vẫn gặp khó khăn: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        elif error_type == 'network_error':
            return f"""Dạ {personal_address}, hiện tại có vấn đề kết nối đến hệ thống của trường. {personal_address.title()} vui lòng:
• Kiểm tra kết nối mạng
• Thử lại sau vài phút
• Liên hệ bộ phận IT nếu vấn đề kéo dài: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
        
        else:
            return f"""Dạ {personal_address}, em gặp khó khăn kỹ thuật khi truy xuất thông tin. {personal_address.title()} có thể:
• Thử lại sau vài phút
• Truy cập trực tiếp hệ thống quản lý đào tạo
• Liên hệ bộ phận IT: it@bdu.edu.vn

{personal_address.title()} có cần hỗ trợ thêm gì không ạ? 🎓"""
    
    # Delegate methods to hybrid chatbot with enhanced status
    def get_system_status(self):
        """🚀 NÂNG CẤP: Get comprehensive system status including enhanced context features"""
        hybrid_status = self.hybrid_chatbot.get_system_status()
        api_status = external_api_service.get_system_status()
        
        # Enhance with enhanced API priority info
        hybrid_status.update({
            'service_layer': 'Enhanced_BDUChatbotService',
            'enhanced_api_priority': {  # 🚀 NEW section
                'context_memory_integration': True,
                'personal_keywords_count': len(self.api_priority_config['personal_info_keywords']),
                'time_keywords_count': len(self.api_priority_config['time_context_keywords']),
                'continuation_keywords_count': len(self.api_priority_config['context_continuation_keywords']),
                'schedule_intents': self.api_priority_config['schedule_intent_names'],
                'memory_lookback_limit': self.api_priority_config['memory_lookback_limit'],
                'confidence_threshold': self.api_priority_config['schedule_intent_confidence_threshold']
            },
            'external_api_service_status': api_status,
            'enhanced_processing_flow': [  # 🚀 UPDATED
                '1. Enhanced Intent Classification',
                '2. Session Memory Context Analysis', 
                '3. Context-Aware API Priority Check',
                '4. Enhanced External API Call (if needed)',
                '5. Enhanced Hybrid Retrieval & Re-ranking (fallback)',
                '6. User Memory Prompt Integration',
                '7. Gender-based Addressing with Context',
                '8. Conversation Context Summary Integration'
            ]
        })
        
        return hybrid_status
    
    def get_conversation_memory(self, session_id):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.get_conversation_memory(session_id)
    
    def clear_conversation_memory(self, session_id=None):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.clear_conversation_memory(session_id)
    
    def reload_after_qa_update(self):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.reload_after_qa_update()
    
    @property
    def model(self):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.model
    
    @property
    def index(self):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.index
    
    @property
    def knowledge_data(self):
        """Delegate to hybrid chatbot"""
        return self.hybrid_chatbot.knowledge_data

# 🚀 ENHANCED: Create enhanced chatbot instance
chatbot_ai = BDUChatbotService()