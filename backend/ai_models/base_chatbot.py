import os
import time
import logging
import pandas as pd
import io
import re
import faiss
import numpy as np
from django.conf import settings
from knowledge.models import KnowledgeBase
from qa_management.services import drive_service

logger = logging.getLogger(__name__)

class ChatbotAI:
    def __init__(self, shared_response_generator):
        self.model = None
        self.index = None
        self.knowledge_data = []
        self.vietnamese_restorer = None  # Không cần nữa - LocalQwenGenerator tự xử lý tiếng Việt
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
            # Gọi service để lấy nội dung file link.csv từ Drive
            link_csv_content = drive_service.get_specific_csv_content('link.csv')
            
            if not link_csv_content:
                logger.error("❌ Could not load link.csv from Google Drive. Link mapping will be empty.")
                self.link_mapping = {}
                return

            # Dùng pandas để đọc nội dung CSV từ string
            df_links = pd.read_csv(io.StringIO(link_csv_content), encoding='utf-8')
            
            for index, row in df_links.iterrows():
                # Dùng .get() để tránh lỗi nếu cột không tồn tại
                stt = str(row.get('STT', '')).strip()
                link = str(row.get('Link', '')).strip()
                if stt and link and stt != 'nan' and link != 'nan':
                    self.link_mapping[stt] = link
            
            logger.info(f"✅ Loaded {len(self.link_mapping)} reference links FROM GOOGLE DRIVE")

        except Exception as e:
            logger.error(f"❌ Error loading link mapping FROM GOOGLE DRIVE: {str(e)}")
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
    
    def semantic_search_with_context(self, query, context_keywords=None, top_k=20):
        """🆕 THÊM MỚI: Semantic search với context enhancement"""
        try:
            if not self.model or not self.index:
                logger.warning("⚠️ Model or index not available for context search")
                return []
            
            # Restore Vietnamese tone nếu cần
            original_query = query
            if self.vietnamese_restorer and not self.vietnamese_restorer.has_vietnamese_accents(query):
                restored_query = self.vietnamese_restorer.restore_vietnamese_tone(query)
                if restored_query != query:
                    logger.info(f"🎯 Using restored query for context search: '{query}' -> '{restored_query}'")
                    query = restored_query
            
            # Build enhanced query với context
            enhanced_query = query
            if context_keywords and len(context_keywords) > 0:
                # Thêm context keywords vào query một cách tự nhiên
                context_str = " ".join(context_keywords[:3])  # Chỉ dùng 3 keywords đầu
                enhanced_query = f"{query} {context_str}"
                logger.info(f"🔍 Enhanced query với context: '{query}' -> '{enhanced_query}'")
            
            # Perform semantic search với enhanced query
            query_embedding = self.model.encode([enhanced_query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.knowledge_data))
            )
            
            # Build candidates với thông tin context
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.knowledge_data) and score > 0.1:
                    candidate = self.knowledge_data[idx].copy()
                    candidate['semantic_score'] = float(score)
                    candidate['similarity'] = float(score)
                    candidate['reference_links'] = self.get_reference_links(candidate)
                    # 🆕 THÊM: Đánh dấu đây là kết quả có context
                    candidate['context_enhanced'] = bool(context_keywords)
                    candidate['context_keywords_used'] = context_keywords or []
                    candidates.append(candidate)
            
            logger.info(f"🔍 Context-enhanced search found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Context-enhanced search error: {str(e)}")
            # Fallback to normal search
            return self.semantic_search_top_k(query, top_k)

    def dual_semantic_search(self, query, context_keywords=None, top_k=20):
        """
        🔧 STABILITY IMPROVED: Dual search với logic ổn định hơn
        - Ưu tiên consistency over optimization
        - Thêm fallback mechanisms
        """
        try:
            logger.info(f"🔄 STABLE Dual semantic search for: '{query}' with context: {context_keywords}")
            
            # ALWAYS perform normal search first (baseline)
            normal_candidates = self.semantic_search_top_k(query, top_k)
            logger.info(f"🔍 Normal search: {len(normal_candidates)} candidates, top_score={normal_candidates[0].get('semantic_score', 0):.3f if normal_candidates else 0}")
            
            # Context search only if meaningful context exists
            context_candidates = []
            if context_keywords and len(context_keywords) > 0:
                context_candidates = self.semantic_search_with_context(query, context_keywords, top_k)
                logger.info(f"🔍 Context search: {len(context_candidates)} candidates, top_score={context_candidates[0].get('semantic_score', 0):.3f if context_candidates else 0}")
            
            # STABLE DECISION LOGIC: Prefer consistency
            if not context_candidates or len(context_candidates) == 0:
                logger.info("🔍 Using normal search (no context results)")
                return normal_candidates, 'normal'
            
            if not normal_candidates or len(normal_candidates) == 0:
                logger.info("🔍 Using context search (no normal results)")  
                return context_candidates, 'context'
            
            # Compare with stability bias
            normal_top_score = normal_candidates[0].get('semantic_score', 0)
            context_top_score = context_candidates[0].get('semantic_score', 0)
            score_diff = context_top_score - normal_top_score
            
            # 🔧 STABILITY: More conservative switching với hysteresis
            # Context cần tốt hơn đáng kể MỚI được chọn
            if score_diff > 0.2:  # Tăng từ 0.15 lên 0.2
                logger.info(f"🔍 Context significantly better (+{score_diff:.3f}) - using context")
                return context_candidates, 'context'
            elif score_diff < -0.05:  # Normal rõ ràng tốt hơn
                logger.info(f"🔍 Normal clearly better ({score_diff:.3f}) - using normal")
                return normal_candidates, 'normal'
            else:
                # Trong vùng uncertain, ưu tiên theo query characteristics
                query_lower = query.lower()
                
                # Nếu query có đại từ hoặc tham chiếu, ưu tiên context
                if any(pronoun in query_lower for pronoun in ['ông ấy', 'bà ấy', 'người đó', 'thầy ấy', 'cô ấy']):
                    logger.info(f"🔍 Query has pronoun - preferring context (score_diff: {score_diff:.3f})")
                    return context_candidates, 'context'
                
                # Nếu query có tên riêng, ưu tiên normal để tránh confusion
                has_proper_name = any(word[0].isupper() for word in query.split() if len(word) > 1)
                if has_proper_name:
                    logger.info(f"🔍 Query has proper names - preferring normal for stability (score_diff: {score_diff:.3f})")
                    return normal_candidates, 'normal'
                
                # Default: ưu tiên normal cho stability
                logger.info(f"🔍 Ambiguous case - preferring normal for stability (score_diff: {score_diff:.3f})")
                return normal_candidates, 'normal'
            
        except Exception as e:
            logger.error(f"Dual search error: {str(e)}")
            # ALWAYS fallback to normal search
            return self.semantic_search_top_k(query, top_k), 'fallback'

    
