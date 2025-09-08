import re
import logging
import time

logger = logging.getLogger(__name__)

class SimpleEntityExtractor:
    """Trích xuất thực thể đơn giản từ Q&A để build context memory, hoạt động độc lập."""
    
    def __init__(self):
        # 🔧 CẢI TIẾN: Patterns chặt chẽ hơn cho các loại entity
        self.entity_patterns = {
            'person_name': [
                # 🔧 IMPROVED: Tên riêng người Việt (2-4 từ, viết hoa đầu từ) - CHẶT CHẼ HỚN
                r'\b([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+)\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+)(?:\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+))?(?:\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+))?\b',
                # Pattern với tiến sĩ, giáo sư
                r'(?:GS\.TS\.|TS\.|GS\.|tiến sĩ|giáo sư)\s+([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+(?:\s+[A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊẾỀỂỄỆÌÍỊỈĨÒÓỌỎÕÔỐỒỔỖỘƠỜỚỞỠỢÙÚỤỦŨƯỪỨỬỮỰỲÝỴỶỸĐ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹđ]+){1,2})'
            ],
            'position': [
                r'\b(hiệu trưởng|phó hiệu trưởng|trưởng phòng|phó trưởng phòng|trưởng khoa|phó trưởng khoa|giáo sư|phó giáo sư|tiến sĩ|thạc sĩ)\b',
                r'\b(chủ tịch|phó chủ tịch|ủy viên|thành viên|trưởng ban|phó ban)\b'
            ],
            'department': [
                r'(khoa [^.!?]*|phòng [^.!?]*|ban [^.!?]*|bộ môn [^.!?]*)',
                r'(đại học bình dương|bdu|trường đại học)'
            ],
            'numbers': [
                r'(\d+(?:[.,]\d+)*(?:\s*(?:triệu|nghìn|tỷ|đồng|vnđ|usd|phần trăm|%))?)',
                r'(\d+(?:\.\d+)?(?:\s*tín chỉ)?)'
            ],
            'dates': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(ngày \d{1,2}|tháng \d{1,2}|năm \d{4})',
                r'(học kỳ \d+|năm học \d{4}-\d{4})'
            ]
        }
        
        # 🆕 THÊM MỚI: Blacklist để loại bỏ false positives
        self.person_name_blacklist = {
            # Các cụm từ thường bị nhận nhầm là tên người
            'hoc phi chinh', 'quy khanh', 'duc tin', 'duc hanh', 'duc duc', 'nam duc',
            'hoc phi', 'chi phi', 'muc phi', 'le phi', 'phi le', 'thu phi',
            'quy dinh', 'quy che', 'quy trinh', 'quy tac', 'quy luat',
            'duc tinh', 'duc tich', 'duc hanh vi', 'duc han che',
            'nam hoc', 'nam tu', 'nam toi', 'nam sau', 'nam truoc',
            'tin chi', 'tin tin', 'chi tiet', 'chi tieu', 'chi phi',
            'bao cao', 'cao cap', 'cao dang', 'cap hoc', 'cap do',
            'sinh vien', 'giang vien', 'can bo', 'hoc sinh', 'nghien cuu sinh',
            'dai hoc', 'cao hoc', 'tien si', 'thac si', 'cu nhan',
            'mon hoc', 'bai hoc', 'gio hoc', 'lop hoc', 'hoc tap',
            # Thêm các từ khóa BDU thường gặp
            'binh duong', 'bdu', 'truong dai hoc', 'phong ban', 'khoa hoc',
            'nghien cuu', 'dao tao', 'quan ly', 'hanh chinh', 'ky thuat',
            'cong nghe', 'kinh te', 'ngoai ngu', 'su pham', 'y khoa'
        }
        
        # 🆕 THÊM MỚI: Common Vietnamese words that are not names
        self.common_words_blacklist = {
            'co the', 'co ban', 'co so', 'co hoi', 'co quan', 'co mat',
            'la mot', 'la cach', 'la gi', 'la ai', 'la khi', 'la lieu',
            'duoc su', 'duoc cap', 'duoc phep', 'duoc biet', 'duoc tang',
            'hay la', 'hay khong', 'hay nhat', 'hay gi', 'hay co',
            'neu co', 'neu khong', 'neu la', 'neu can', 'neu muon'
        }
        
        logger.info("✅ IMPROVED SimpleEntityExtractor initialized with enhanced patterns and blacklists")

    def extract_entities(self, text, query_context=""):
        """🔧 IMPROVED: Trích xuất entities với filtering tốt hơn"""
        if not text:
            return {}
            
        entities = {}
        text_cleaned = text.strip()
        
        # 🔧 IMPROVED: Clean text - remove common phrases but preserve names
        text_cleaned = re.sub(r'\b(dạ|ạ|thưa|xin chào|chào|em|anh|chị|cảm ơn)\b', ' ', text_cleaned, flags=re.IGNORECASE)
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
        
        # Extract theo từng loại pattern
        for entity_type, patterns in self.entity_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_cleaned, re.IGNORECASE)
                for match in matches:
                    if entity_type == 'person_name':
                        # 🔧 SPECIAL HANDLING: Person names need more careful extraction
                        entity_value = self._extract_person_name_from_match(match)
                    else:
                        entity_value = match.group(1) if match.groups() else match.group(0)
                    
                    entity_value = entity_value.strip()
                    
                    # 🔧 IMPROVED: Strict filtering với blacklist
                    if self._is_valid_entity(entity_value, entity_type):
                        # Normalize entity
                        if entity_type == 'person_name':
                            entity_value = self._normalize_person_name(entity_value)
                        else:
                            entity_value = entity_value.lower()
                            
                        if entity_value not in entities[entity_type]:
                            entities[entity_type].append(entity_value)
        
        # Chỉ giữ lại entities có ít nhất 1 item
        entities = {k: v for k, v in entities.items() if v}
        
        logger.debug(f"🔍 Entity extraction result: {entities}")
        return entities

    def _extract_person_name_from_match(self, match):
        """🆕 THÊM MỚI: Trích xuất tên người từ regex match"""
        if match.groups():
            # Combine all non-empty groups
            name_parts = [group for group in match.groups() if group and group.strip()]
            return ' '.join(name_parts)
        else:
            return match.group(0)

    def _is_valid_entity(self, entity_value, entity_type):
        """🔧 IMPROVED: Validate entity quality với blacklist mở rộng"""
        if not entity_value or len(entity_value.strip()) < 3:
            return False
            
        entity_lower = entity_value.lower().strip()
        
        # 🆕 CHECK BLACKLIST FIRST
        if entity_type == 'person_name':
            # Check person name blacklist
            if entity_lower in self.person_name_blacklist:
                logger.debug(f"🚫 Rejected by person blacklist: '{entity_value}'")
                return False
            
            # Check common words blacklist
            if entity_lower in self.common_words_blacklist:
                logger.debug(f"🚫 Rejected by common words blacklist: '{entity_value}'")
                return False
            
            # Check for parts in blacklist
            entity_words = entity_lower.split()
            for word in entity_words:
                if word in self.person_name_blacklist or word in self.common_words_blacklist:
                    logger.debug(f"🚫 Rejected by word-level blacklist: '{entity_value}' (word: '{word}')")
                    return False
        
        # Remove noise patterns
        noise_patterns = [
            r'\b(có|cần|thể|thêm|gì|không|hỗ|trợ|để|em|là|ai|nói|rõ|hơn|về|vấn|đề|chính|xác|nhất)\b',
            r'^(và|hoặc|với|để|khi|nếu|tại|về|cho|trong|của|từ)',
            r'(ạ|à|ơi|nhé)$'
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, entity_lower):
                logger.debug(f"🚫 Rejected by noise pattern: '{entity_value}' (pattern: {pattern})")
                return False
        
        # Specific validation by type
        if entity_type == 'person_name':
            # 🔧 STRICTER: Must have 2-4 words, each word >= 2 chars
            words = entity_lower.split()
            if len(words) < 2 or len(words) > 4:
                logger.debug(f"🚫 Rejected by word count: '{entity_value}' ({len(words)} words)")
                return False
            
            if any(len(word) < 2 for word in words):
                logger.debug(f"🚫 Rejected by word length: '{entity_value}'")
                return False
            
            # 🆕 NEW: Check for Vietnamese name patterns
            if not self._looks_like_vietnamese_name(entity_value):
                logger.debug(f"🚫 Rejected by Vietnamese name pattern: '{entity_value}'")
                return False
            
            # Must not contain common Vietnamese particles
            if any(word in ['cô', 'thầy', 'anh', 'chị', 'em', 'dạ', 'được', 'phải', 'theo', 'như', 'từ'] for word in words):
                logger.debug(f"🚫 Rejected by particle words: '{entity_value}'")
                return False
                
        elif entity_type == 'position':
            # Must be exactly one of the position words
            valid_positions = ['hiệu trưởng', 'phó hiệu trưởng', 'trưởng phòng', 'phó trưởng phòng', 'trưởng khoa', 'phó trưởng khoa', 'giáo sư', 'phó giáo sư', 'tiến sĩ', 'thạc sĩ', 'chủ tịch', 'phó chủ tịch']
            if entity_lower not in valid_positions:
                logger.debug(f"🚫 Rejected invalid position: '{entity_value}'")
                return False
        
        logger.debug(f"✅ Valid entity: '{entity_value}' (type: {entity_type})")
        return True
    
    def _looks_like_vietnamese_name(self, name):
        """🆕 THÊM MỚI: Kiểm tra xem có giống tên người Việt không"""
        name_lower = name.lower()
        words = name_lower.split()
        
        # Vietnamese surname patterns (họ phổ biến)
        common_surnames = {
            'nguyễn', 'trần', 'lê', 'phạm', 'hoàng', 'huỳnh', 'phan', 'vũ', 'võ', 'đặng', 
            'bùi', 'đỗ', 'hồ', 'ngô', 'dương', 'lý', 'cao', 'đậu', 'lưu', 'tô',
            'nguyen', 'tran', 'le', 'pham', 'hoang', 'huynh', 'phan', 'vu', 'vo', 'dang',
            'bui', 'do', 'ho', 'ngo', 'duong', 'ly', 'cao', 'dau', 'luu', 'to'
        }
        
        # Check if first word (surname) is common Vietnamese surname
        if words[0] in common_surnames:
            return True
        
        # Vietnamese name characteristics
        # - Usually has balanced syllable structure
        # - Contains Vietnamese-specific characters or patterns
        vietnamese_chars = 'ăâêôơưàáạảãằắặẳẵầấậẩẫềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹđ'
        
        # Count Vietnamese-specific characters
        vietnamese_char_count = sum(1 for char in name_lower if char in vietnamese_chars)
        
        # If has Vietnamese chars and reasonable length, likely a Vietnamese name
        if vietnamese_char_count >= 1 and len(words) >= 2:
            return True
        
        # Pattern check: avoid common false positives
        false_positive_patterns = [
            r'phi|phí|fee',  # Avoid fee-related terms
            r'quy|qúy',      # Avoid regulation-related terms
            r'học|hoc',      # Avoid study-related terms
            r'chí|chi',      # Avoid will/credit-related terms
        ]
        
        for pattern in false_positive_patterns:
            if re.search(pattern, name_lower):
                # Double check: if it really contains Vietnamese name elements, still accept
                if vietnamese_char_count >= 2:  # Higher threshold for suspicious cases
                    continue
                else:
                    return False
        
        # Default: accept if passes basic structure checks
        return True
    
    def build_entity_relationships(self, query, answer, entities):
        """Xây dựng mối quan hệ giữa entities dựa vào ngữ cảnh Q&A"""
        relationships = []
        
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Tìm mối quan hệ person - position
        if 'person_name' in entities and 'position' in entities:
            for person in entities['person_name']:
                for position in entities['position']:
                    # Kiểm tra trong query
                    if any(keyword in query_lower for keyword in ['là ai', 'ai là', 'chức vụ']):
                        relationships.append({
                            'type': 'person_position',
                            'entity1': person,
                            'entity2': position,
                            'relation': 'has_position',
                            'confidence': 0.8,
                            'source': 'query_answer_pair'
                        })
        
        # Tìm mối quan hệ person - department
        if 'person_name' in entities and 'department' in entities:
            for person in entities['person_name']:
                for dept in entities['department']:
                    relationships.append({
                        'type': 'person_department', 
                        'entity1': person,
                        'entity2': dept,
                        'relation': 'works_at',
                        'confidence': 0.7,
                        'source': 'context'
                    })
        
        return relationships

    def _normalize_person_name(self, name):
        """🔧 IMPROVED: Normalize person name to consistent format"""
        # Title case each word
        words = name.lower().split()
        normalized_words = []
        for word in words:
            if len(word) > 0:
                normalized_words.append(word[0].upper() + word[1:])
        return ' '.join(normalized_words)

    def get_context_keywords(self, entities, relationships):
        """🚀 FIX: Generate better context keywords"""
        context_keywords = []
        
        # From entities - prioritize person names and positions
        if 'person_name' in entities:
            for entity in entities['person_name'][:2]:  # Max 2 person names
                context_keywords.append(entity)
                
        if 'position' in entities:
            for entity in entities['position'][:1]:  # Max 1 position
                context_keywords.append(entity)
        
        # From relationships
        for rel in relationships[:3]:  # Max 3 relationships
            if rel['confidence'] > 0.7:
                context_keywords.extend([rel['entity1'], rel['entity2']])
        
        # 🚀 FIX: Deduplicate and limit
        context_keywords = list(set(context_keywords))
        context_keywords = [kw for kw in context_keywords if len(kw.strip()) > 2]
        
        return context_keywords[:3]