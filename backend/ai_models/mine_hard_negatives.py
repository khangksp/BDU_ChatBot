# FILE: mine_hard_negatives.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_training_triplets(qa_csv_path=None, output_csv_path=None, model_name='keepitreal/vietnamese-sbert'):
    """
    Đọc file QA.csv, sử dụng mô hình SBERT để tìm hard negatives cho mỗi câu hỏi,
    và tạo ra file training_triplets.csv để huấn luyện với TripletLoss.
    
    Args:
        qa_csv_path: Đường dẫn file QA.csv (default: data/QA.csv)
        output_csv_path: Đường dẫn file output (default: data/training_triplets.csv)
        model_name: Tên mô hình SBERT để mining hard negatives
    """
    try:
        # Thiết lập đường dẫn mặc định
        if not qa_csv_path:
            # Tự động tìm đường dẫn đến QA.csv
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, '..', 'data', 'QA.csv'),  # ../data/QA.csv
                os.path.join(current_dir, 'data', 'QA.csv'),       # ./data/QA.csv  
                'data/QA.csv',                                      # relative path
                '../data/QA.csv'                                    # parent dir
            ]
            
            qa_csv_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    qa_csv_path = abs_path
                    break
            
            if not qa_csv_path:
                logger.error("❌ Không tìm thấy file QA.csv trong các vị trí:")
                for path in possible_paths:
                    logger.error(f"   - {os.path.abspath(path)}")
                return False
                
        if not output_csv_path:
            output_csv_path = os.path.join(os.path.dirname(qa_csv_path), 'training_triplets.csv')

        logger.info(f"🚀 Bắt đầu quá trình Hard Negative Mining từ {qa_csv_path}...")

        # 1. Kiểm tra file đầu vào
        if not os.path.exists(qa_csv_path):
            logger.error(f"❌ Không tìm thấy file {qa_csv_path}")
            return False

        # 2. Tải mô hình SBERT cho Hard Negative Mining
        logger.info(f"📥 Tải mô hình sentence-transformer: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            logger.info("✅ Mô hình SBERT đã được tải thành công")
        except Exception as e:
            logger.warning(f"⚠️ Không thể tải {model_name}, thử vinai/phobert-base...")
            try:
                model = SentenceTransformer('vinai/phobert-base')
            except Exception as e2:
                logger.error(f"❌ Không thể tải bất kỳ mô hình nào: {e2}")
                return False

        # 3. Đọc và làm sạch dữ liệu QA
        logger.info(f"📊 Đọc dữ liệu từ {qa_csv_path}...")
        df = pd.read_csv(qa_csv_path, encoding='utf-8')
        logger.info(f"📊 Đã tải {len(df)} dòng dữ liệu thô")

        # Làm sạch dữ liệu như trong hệ thống hiện tại
        df = df.dropna(subset=['question', 'answer'])
        df['question'] = df['question'].astype(str)
        df['answer'] = df['answer'].astype(str)
        
        # Filter theo tiêu chí chất lượng
        df = df[(df['question'].str.len() > 10) & (df['answer'].str.len() > 20)]
        df = df.reset_index(drop=True)
        
        logger.info(f"✅ Đã làm sạch, còn lại {len(df)} cặp Q&A hợp lệ")

        questions = df['question'].tolist()
        answers = df['answer'].tolist()

        # 4. Mã hóa tất cả các câu trả lời thành vectors
        logger.info("🔢 Mã hóa tất cả các câu trả lời...")
        answer_embeddings = model.encode(answers, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
        logger.info(f"✅ Đã mã hóa {len(answer_embeddings)} câu trả lời")

        # 5. Hard Negative Mining cho mỗi câu hỏi
        logger.info("⚒️ Bắt đầu Hard Negative Mining...")
        triplets = []
        
        for i in tqdm(range(len(questions)), desc="Mining Hard Negatives"):
            anchor_question = questions[i]
            positive_answer = answers[i]
            
            # Mã hóa câu hỏi hiện tại
            question_embedding = model.encode(anchor_question, convert_to_numpy=True)
            
            # Tính độ tương đồng với TẤT CẢ câu trả lời
            similarities = cosine_similarity([question_embedding], answer_embeddings)[0]
            
            # Sắp xếp theo độ tương đồng giảm dần
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Tìm hard negative: câu trả lời có độ tương đồng cao nhất nhưng KHÔNG phải câu trả lời đúng
            hard_negative_found = False
            for idx in sorted_indices:
                if idx != i:  # Bỏ qua câu trả lời đúng
                    hard_negative_answer = answers[idx]
                    similarity_score = similarities[idx]
                    
                    # Chỉ chọn những hard negative có độ tương đồng đủ cao (>0.3) để đảm bảo "khó"
                    if similarity_score > 0.3:
                        triplets.append({
                            'anchor': anchor_question,
                            'positive': positive_answer,
                            'negative': hard_negative_answer,
                            'negative_similarity': similarity_score,
                            'source_stt': df.loc[i, 'STT'] if 'STT' in df.columns else f"q_{i}",
                            'negative_stt': df.loc[idx, 'STT'] if 'STT' in df.columns else f"q_{idx}"
                        })
                        hard_negative_found = True
                        break
            
            # Nếu không tìm được hard negative đủ khó, chọn random negative
            if not hard_negative_found and len(answers) > 1:
                # Tìm random negative (tránh câu trả lời đúng)
                negative_idx = (i + np.random.randint(1, len(answers))) % len(answers)
                triplets.append({
                    'anchor': anchor_question,
                    'positive': positive_answer,
                    'negative': answers[negative_idx],
                    'negative_similarity': similarities[negative_idx],
                    'source_stt': df.loc[i, 'STT'] if 'STT' in df.columns else f"q_{i}",
                    'negative_stt': df.loc[negative_idx, 'STT'] if 'STT' in df.columns else f"q_{negative_idx}"
                })

        # 6. Lưu kết quả ra file CSV
        if not triplets:
            logger.warning("⚠️ Không tạo được triplet nào. Kiểm tra lại dữ liệu đầu vào.")
            return False

        triplet_df = pd.DataFrame(triplets)
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Lưu file
        triplet_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        # Thống kê chất lượng
        avg_negative_similarity = triplet_df['negative_similarity'].mean()
        high_quality_triplets = len(triplet_df[triplet_df['negative_similarity'] > 0.5])
        
        logger.info(f"✅ HOÀN THÀNH Hard Negative Mining!")
        logger.info(f"   📊 Tạo được {len(triplet_df)} triplets")
        logger.info(f"   📈 Độ tương đồng trung bình của hard negatives: {avg_negative_similarity:.4f}")
        logger.info(f"   🎯 Triplets chất lượng cao (similarity > 0.5): {high_quality_triplets}")
        logger.info(f"   💾 Đã lưu tại: {output_csv_path}")
        
        return True

    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình Hard Negative Mining: {e}", exc_info=True)
        return False

def analyze_triplet_quality(triplets_csv_path):
    """
    Phân tích chất lượng của triplets đã tạo
    """
    try:
        df = pd.read_csv(triplets_csv_path, encoding='utf-8')
        
        logger.info("📊 PHÂN TÍCH CHẤT LƯỢNG TRIPLETS:")
        logger.info(f"   📊 Tổng số triplets: {len(df)}")
        logger.info(f"   📈 Độ tương đồng negative trung bình: {df['negative_similarity'].mean():.4f}")
        logger.info(f"   📈 Độ tương đồng negative cao nhất: {df['negative_similarity'].max():.4f}")
        logger.info(f"   📈 Độ tương đồng negative thấp nhất: {df['negative_similarity'].min():.4f}")
        
        # Phân loại theo mức độ khó
        easy_negatives = len(df[df['negative_similarity'] < 0.3])
        medium_negatives = len(df[(df['negative_similarity'] >= 0.3) & (df['negative_similarity'] < 0.6)])
        hard_negatives = len(df[df['negative_similarity'] >= 0.6])
        
        logger.info(f"   🟢 Easy negatives (<0.3): {easy_negatives} ({easy_negatives/len(df)*100:.1f}%)")
        logger.info(f"   🟡 Medium negatives (0.3-0.6): {medium_negatives} ({medium_negatives/len(df)*100:.1f}%)")
        logger.info(f"   🔴 Hard negatives (>0.6): {hard_negatives} ({hard_negatives/len(df)*100:.1f}%)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích chất lượng: {e}")
        return False

if __name__ == '__main__':
    # Chạy Hard Negative Mining
    success = create_training_triplets()
    
    if success:
        # Phân tích chất lượng triplets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_paths = [
            os.path.join(current_dir, '..', 'data', 'training_triplets.csv'),
            os.path.join(current_dir, 'data', 'training_triplets.csv'),
            'data/training_triplets.csv',
            '../data/training_triplets.csv'
        ]
        
        for output_path in output_paths:
            abs_path = os.path.abspath(output_path)
            if os.path.exists(abs_path):
                analyze_triplet_quality(abs_path)
                break
        
        print("✅ Hard Negative Mining hoàn thành! Có thể chạy train_retriever.py để huấn luyện.")
    else:
        print("❌ Hard Negative Mining thất bại. Kiểm tra log để biết chi tiết.")