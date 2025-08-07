import os
import pandas as pd
import logging
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
import json
from django.conf import settings

logger = logging.getLogger(__name__)

class PhoBERTRetrieverTrainer:
    """
    🚀 Enhanced PhoBERT Fine-tuning for Document Retrieval
    
    Supports TWO training methods:
    1. MultipleNegativesRankingLoss (original method) 
    2. TripletLoss with Hard Negative Mining (new advanced method)
    """
    
    def __init__(self, base_model_name='vinai/phobert-base', output_dir='./fine_tuned_phobert', training_method='triplet'):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.training_method = training_method.lower()  # 'triplet' or 'ranking'
        self.model = None
        self.train_examples = []
        self.eval_examples = []
        
        # Training configuration - điều chỉnh theo method
        if self.training_method == 'triplet':
            # TripletLoss configuration
            self.config = {
                'batch_size': 16,
                'epochs': 3,
                'warmup_steps': 100,
                'evaluation_steps': 500,
                'save_steps': 1000,
                'max_seq_length': 256,
                'learning_rate': 2e-5,
                'triplet_margin': 0.5,
                'distance_metric': 'cosine'
            }
        else:
            # MultipleNegativesRankingLoss configuration (original)
            self.config = {
                'batch_size': 8,
                'epochs': 2,
                'warmup_steps': 100,
                'evaluation_steps': 500,
                'save_steps': 1000,
                'max_seq_length': 256,
                'learning_rate': 2e-5,
                'temperature': 0.05
            }
        
        logger.info(f"🚀 PhoBERT Retriever Trainer initialized")
        logger.info(f"   📁 Output directory: {output_dir}")
        logger.info(f"   🎯 Base model: {base_model_name}")
        logger.info(f"   ⚒️ Training method: {self.training_method.upper()}")
    
    def load_data_from_csv(self, csv_path=None):
        """
        Load và prepare training data từ QA.csv (cho MultipleNegativesRankingLoss)
        """
        try:
            # Tìm CSV file
            if not csv_path:
                csv_path = os.path.join(settings.BASE_DIR, 'data', 'QA.csv')
            
            if not os.path.exists(csv_path):
                # Fallback: load from QA Management database
                logger.info("📊 CSV not found, loading from QA Management database...")
                return self._load_from_database()
            
            logger.info(f"📊 Loading training data from: {csv_path}")
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Clean data
            df = df.dropna(subset=['question', 'answer'])
            df['question'] = df['question'].astype(str)
            df['answer'] = df['answer'].astype(str)
            
            # Filter valid entries
            df = df[(df['question'].str.len() > 10) & (df['answer'].str.len() > 20)]
            
            logger.info(f"✅ Loaded {len(df)} valid QA pairs from CSV")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV data: {str(e)}")
            return self._load_from_database()
    
    def load_triplets_from_csv(self, triplets_csv_path=None):
        """
        🚀 NEW: Load dữ liệu triplets cho TripletLoss training
        """
        try:
            # Thiết lập đường dẫn mặc định
            if not triplets_csv_path:
                if hasattr(settings, 'BASE_DIR'):
                    triplets_csv_path = os.path.join(settings.BASE_DIR, 'data', 'training_triplets.csv')
                else:
                    triplets_csv_path = 'data/training_triplets.csv'
            
            if not os.path.exists(triplets_csv_path):
                logger.error(f"❌ File '{triplets_csv_path}' không tồn tại!")
                logger.info("💡 Chạy 'python mine_hard_negatives.py' trước để tạo dữ liệu triplets.")
                return None

            logger.info(f"📊 Loading triplets từ: {triplets_csv_path}")
            df_triplets = pd.read_csv(triplets_csv_path, encoding='utf-8').dropna()
            
            # Kiểm tra các cột cần thiết
            required_columns = ['anchor', 'positive', 'negative']
            if not all(col in df_triplets.columns for col in required_columns):
                logger.error(f"❌ File triplets thiếu các cột: {required_columns}")
                return None
            
            logger.info(f"✅ Đã load {len(df_triplets)} triplets cho TripletLoss training")
            
            # Thống kê chất lượng
            if 'negative_similarity' in df_triplets.columns:
                avg_sim = df_triplets['negative_similarity'].mean()
                high_quality = len(df_triplets[df_triplets['negative_similarity'] > 0.5])
                logger.info(f"📈 Độ tương đồng trung bình hard negatives: {avg_sim:.4f}")
                logger.info(f"🎯 Triplets chất lượng cao: {high_quality}/{len(df_triplets)}")
            
            return df_triplets
            
        except Exception as e:
            logger.error(f"❌ Lỗi load triplets: {str(e)}")
            return None
    
    def _load_from_database(self):
        """
        Fallback: Load data from QA Management database
        """
        try:
            from qa_management.models import QAEntry
            
            qa_entries = QAEntry.objects.filter(is_active=True)
            
            data = []
            for entry in qa_entries:
                data.append({
                    'question': entry.question,
                    'answer': entry.answer,
                    'category': entry.category or 'Giảng viên'
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df)} QA pairs from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading from database: {str(e)}")
            return pd.DataFrame()
    
    def prepare_training_examples(self, df):
        """
        Prepare training examples cho MultipleNegativesRankingLoss (original method)
        """
        logger.info("🔧 Preparing training examples for MultipleNegativesRankingLoss...")
        
        # Convert to list of texts
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        
        # Create positive pairs (question, answer)
        positive_pairs = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            question = str(question).strip()
            answer = str(answer).strip()
            
            if len(question) > 10 and len(answer) > 20:
                positive_pairs.append((question, answer))
        
        logger.info(f"✅ Created {len(positive_pairs)} positive pairs")
        
        # Create InputExamples for training
        train_examples = []
        
        for i, (question, answer) in enumerate(positive_pairs):
            # InputExample(texts=[anchor, positive], label=1.0)
            example = InputExample(
                texts=[question, answer], 
                label=1.0  # Positive pair
            )
            train_examples.append(example)
            
            # Also create reverse pair for robustness
            reverse_example = InputExample(
                texts=[answer, question],
                label=1.0
            )
            train_examples.append(reverse_example)
        
        logger.info(f"✅ Created {len(train_examples)} training examples (including reverse pairs)")
        
        # Split train/eval
        train_examples, eval_examples = train_test_split(
            train_examples, 
            test_size=0.2, 
            random_state=42
        )
        
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        
        logger.info(f"📊 Data split: {len(train_examples)} train, {len(eval_examples)} eval")
        
        return len(train_examples)
    
    def prepare_triplet_examples(self, df_triplets):
        """
        🚀 NEW: Prepare training examples cho TripletLoss
        """
        logger.info("🔧 Preparing training examples for TripletLoss...")
        
        train_examples = []
        
        for _, row in df_triplets.iterrows():
            # Làm sạch văn bản
            anchor = str(row['anchor']).strip()
            positive = str(row['positive']).strip()
            negative = str(row['negative']).strip()
            
            # Kiểm tra độ dài
            if len(anchor) > 10 and len(positive) > 20 and len(negative) > 20:
                # InputExample cho TripletLoss cần 3 texts
                example = InputExample(texts=[anchor, positive, negative])
                train_examples.append(example)
        
        self.train_examples = train_examples
        self.eval_examples = []  # TripletLoss không cần eval examples riêng
        
        logger.info(f"✅ Created {len(train_examples)} triplet examples for TripletLoss")
        
        return len(train_examples)
    
    def create_evaluation_data(self):
        """
        Tạo evaluation data (chỉ cho MultipleNegativesRankingLoss)
        """
        if not self.eval_examples:
            logger.warning("⚠️ No evaluation examples available")
            return None
        
        # Take subset for faster evaluation
        eval_subset = self.eval_examples[:min(100, len(self.eval_examples))]
        
        sentences1 = []
        sentences2 = []
        scores = []
        
        for example in eval_subset:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            name='qa_retrieval_eval'
        )
        
        logger.info(f"✅ Created evaluator with {len(sentences1)} pairs")
        return evaluator
    
    def train_model(self):
        """
        🚀 ENHANCED: Main training function - supports both methods
        """
        logger.info(f"🚀 Starting PhoBERT fine-tuning with {self.training_method.upper()}...")
        
        # Initialize model
        logger.info(f"📥 Loading base model: {self.base_model_name}")
        self.model = SentenceTransformer(self.base_model_name)
        
        # Set max sequence length
        self.model.max_seq_length = self.config['max_seq_length']
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_examples, 
            shuffle=True, 
            batch_size=self.config['batch_size']
        )
        
        # 🚀 Create loss function based on training method
        if self.training_method == 'triplet':
            # TripletLoss
            distance_metric = losses.TripletDistanceMetric.COSINE if self.config['distance_metric'] == 'cosine' else losses.TripletDistanceMetric.EUCLIDEAN
            train_loss = losses.TripletLoss(
                model=self.model,
                distance_metric=distance_metric,
                triplet_margin=self.config['triplet_margin']
            )
            logger.info(f"⚒️ Using TripletLoss with margin={self.config['triplet_margin']}")
        else:
            # MultipleNegativesRankingLoss (original)
            train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
            logger.info(f"⚒️ Using MultipleNegativesRankingLoss")
        
        # Create evaluator (only for ranking method)
        evaluator = None
        if self.training_method == 'ranking' and self.eval_examples:
            evaluator = self.create_evaluation_data()
        
        # Training arguments
        num_epochs = self.config['epochs']
        warmup_steps = min(
            self.config['warmup_steps'], 
            len(train_dataloader) * num_epochs // 10
        )
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   📊 Training examples: {len(self.train_examples)}")
        if self.eval_examples:
            logger.info(f"   📊 Evaluation examples: {len(self.eval_examples)}")
        logger.info(f"   🔄 Epochs: {num_epochs}")
        logger.info(f"   📦 Batch size: {self.config['batch_size']}")
        logger.info(f"   🔥 Warmup steps: {warmup_steps}")
        logger.info(f"   📏 Max sequence length: {self.config['max_seq_length']}")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 Training on GPU: {gpu_name}")
        else:
            logger.info("💻 Training on CPU")
        
        # Start training
        start_time = time.time()
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=self.config.get('evaluation_steps', 500) if evaluator else None,
            save_best_model=True,
            output_path=self.output_dir,
            optimizer_params={'lr': self.config['learning_rate']},
            scheduler='WarmupLinear',
            show_progress_bar=True,
            checkpoint_path=self.output_dir,
            checkpoint_save_steps=self.config.get('save_steps', 1000)
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"💾 Model saved to: {self.output_dir}")
        
        return True
    
    def save_training_metadata(self):
        """
        🚀 ENHANCED: Save training metadata với thông tin method
        """
        try:
            metadata = {
                'training_date': datetime.now().isoformat(),
                'base_model': self.base_model_name,
                'output_directory': self.output_dir,
                'training_method': self.training_method,
                'training_config': self.config,
                'training_examples_count': len(self.train_examples),
                'eval_examples_count': len(self.eval_examples),
                'model_type': 'sentence_transformer_retrieval',
                'version': '2.0_enhanced'
            }
            
            # Thêm thông tin specific cho từng method
            if self.training_method == 'triplet':
                metadata.update({
                    'fine_tuning_method': 'TripletLoss with Hard Negative Mining',
                    'triplet_margin': self.config['triplet_margin'],
                    'distance_metric': self.config['distance_metric'],
                    'improvement_note': 'Enhanced retrieval through hard negative mining'
                })
            else:
                metadata.update({
                    'fine_tuning_method': 'MultipleNegativesRankingLoss',
                    'temperature': self.config.get('temperature', 0.05)
                })
            
            metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Training metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving metadata: {str(e)}")
    
    def evaluate_model(self):
        """
        🚀 ENHANCED: Evaluate model cho cả hai methods
        """
        if not self.model:
            logger.error("❌ No model loaded for evaluation")
            return None
        
        try:
            logger.info("📊 Evaluating fine-tuned model...")
            
            if self.training_method == 'triplet':
                # Đánh giá cho TripletLoss
                test_examples = self.train_examples[-50:] if len(self.train_examples) > 50 else self.train_examples[:10]
                correct_predictions = 0
                
                for example in test_examples:
                    anchor, positive, negative = example.texts
                    
                    # Encode
                    anchor_emb = self.model.encode(anchor)
                    positive_emb = self.model.encode(positive)  
                    negative_emb = self.model.encode(negative)
                    
                    # Cosine similarity
                    anchor_pos_sim = np.dot(anchor_emb, positive_emb) / (np.linalg.norm(anchor_emb) * np.linalg.norm(positive_emb))
                    anchor_neg_sim = np.dot(anchor_emb, negative_emb) / (np.linalg.norm(anchor_emb) * np.linalg.norm(negative_emb))
                    
                    if anchor_pos_sim > anchor_neg_sim:
                        correct_predictions += 1
                
                accuracy = correct_predictions / len(test_examples)
                
                logger.info(f"📊 TripletLoss Evaluation:")
                logger.info(f"   🎯 Ranking Accuracy: {accuracy:.4f}")
                
                return {
                    'accuracy': accuracy,
                    'correct_predictions': correct_predictions,
                    'total_predictions': len(test_examples),
                    'method': 'triplet'
                }
            
            else:
                # Đánh giá cho MultipleNegativesRankingLoss (original)
                eval_subset = self.eval_examples[:50] if len(self.eval_examples) > 50 else self.eval_examples[:10]
                
                questions = [example.texts[0] for example in eval_subset]
                answers = [example.texts[1] for example in eval_subset]
                
                question_embeddings = self.model.encode(questions)
                answer_embeddings = self.model.encode(answers)
                
                similarities = cosine_similarity(question_embeddings, answer_embeddings)
                
                correct_predictions = 0
                for i in range(len(similarities)):
                    if np.argmax(similarities[i]) == i:
                        correct_predictions += 1
                
                accuracy = correct_predictions / len(similarities)
                avg_similarity = np.mean(np.diag(similarities))
                
                logger.info(f"📊 MultipleNegativesRankingLoss Evaluation:")
                logger.info(f"   🎯 Accuracy: {accuracy:.4f}")
                logger.info(f"   📈 Average similarity: {avg_similarity:.4f}")
                
                return {
                    'accuracy': accuracy,
                    'average_similarity': avg_similarity,
                    'eval_samples': len(eval_subset),
                    'method': 'ranking'
                }
                
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
            return None

def run_training(csv_path=None, output_dir='./fine_tuned_phobert', method='triplet', **kwargs):
    """
    🚀 ENHANCED: Main function hỗ trợ cả hai training methods
    
    Args:
        method: 'triplet' (TripletLoss + Hard Mining) hoặc 'ranking' (MultipleNegativesRankingLoss)
    """
    try:
        logger.info(f"🚀 Starting PhoBERT Retriever Fine-tuning Process with {method.upper()}...")
        
        # Initialize trainer với method được chọn
        trainer = PhoBERTRetrieverTrainer(output_dir=output_dir, training_method=method)
        
        # Update config từ kwargs
        trainer.config.update(kwargs)
        
        # Load data theo method
        if method == 'triplet':
            # Load triplets data cho TripletLoss
            triplets_path = kwargs.get('triplets_path', None)
            df = trainer.load_triplets_from_csv(triplets_path)
            if df is None or df.empty:
                logger.error("❌ Không load được triplets data cho TripletLoss")
                logger.info("💡 Chạy 'python mine_hard_negatives.py' trước!")
                return {'success': False, 'error': 'Cannot load triplets data'}
            
            # Prepare triplet examples
            num_examples = trainer.prepare_triplet_examples(df)
        else:
            # Load QA data cho MultipleNegativesRankingLoss (original)
            df = trainer.load_data_from_csv(csv_path)
            if df.empty:
                logger.error("❌ No training data available")
                return {'success': False, 'error': 'No training data available'}
            
            # Prepare training examples
            num_examples = trainer.prepare_training_examples(df)
        
        if num_examples == 0:
            logger.error("❌ No valid training examples created")
            return {'success': False, 'error': 'No valid training examples'}
        
        # Train model
        success = trainer.train_model()
        if not success:
            logger.error("❌ Training failed")
            return {'success': False, 'error': 'Training failed'}
        
        # Save metadata
        trainer.save_training_metadata()
        
        # Evaluate model
        eval_results = trainer.evaluate_model()
        
        logger.info(f"✅ PhoBERT {method.upper()} fine-tuning completed successfully!")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'training_examples': num_examples,
            'training_method': method,
            'evaluation_results': eval_results
        }
        
    except Exception as e:
        logger.error(f"❌ Training process failed: {str(e)}", exc_info=True)
        return {'success': False, 'error': str(e)}

def check_gpu_availability():
    """
    Check GPU availability cho training
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU Available: {gpu_name} (Count: {gpu_count})")
        return True
    else:
        logger.info("💻 Using CPU for training (GPU not available)")
        return False

if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check GPU
    check_gpu_availability()
    
    # Parse arguments
    method = 'triplet'  # Default to new TripletLoss method
    csv_path = None
    output_dir = './fine_tuned_phobert'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['triplet', 'ranking']:
            method = sys.argv[1]
            csv_path = sys.argv[2] if len(sys.argv) > 2 else None
            output_dir = sys.argv[3] if len(sys.argv) > 3 else './fine_tuned_phobert'
        else:
            csv_path = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else './fine_tuned_phobert'
    
    logger.info(f"🚀 Training method: {method.upper()}")
    
    # Run training
    if method == 'triplet':
        # Kiểm tra xem có file triplets chưa
        triplets_path = os.path.join('data', 'training_triplets.csv')
        if hasattr(settings, 'BASE_DIR'):
            triplets_path = os.path.join(settings.BASE_DIR, 'data', 'training_triplets.csv')
        
        if not os.path.exists(triplets_path):
            print("❌ File training_triplets.csv chưa tồn tại!")
            print("💡 Chạy lệnh sau trước:")
            print("   python mine_hard_negatives.py")
            sys.exit(1)
    
    result = run_training(
        csv_path=csv_path,
        output_dir=output_dir,
        method=method,
        batch_size=16 if method == 'triplet' else 8,
        epochs=3 if method == 'triplet' else 2,
        learning_rate=2e-5,
        triplet_margin=0.5 if method == 'triplet' else None
    )
    
    if result['success']:
        print(f"✅ Training completed successfully with {result['training_method'].upper()}!")
        print(f"📁 Model saved to: {result['output_dir']}")
        if result['evaluation_results']:
            print(f"🎯 Accuracy: {result['evaluation_results']['accuracy']:.4f}")
        print("\n🔄 Next steps:")
        print("1. Restart Django service")
        print("2. Model will auto-load from fine_tuned_phobert/")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)