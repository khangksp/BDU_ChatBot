import os
import pandas as pd
import logging
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import json
from django.conf import settings

logger = logging.getLogger(__name__)

class PhoBERTRetrieverTrainer:
    """
    Advanced PhoBERT Fine-tuning for Document Retrieval
    Sử dụng Multiple Negatives Ranking Loss cho tác vụ retrieval
    """
    
    def __init__(self, base_model_name='vinai/phobert-base', output_dir='./fine_tuned_phobert'):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.model = None
        self.train_examples = []
        self.eval_examples = []
        
        # Training configuration
        self.config = {
            'batch_size': 8,
            'epochs': 2,
            'warmup_steps': 100,
            'evaluation_steps': 500,
            'save_steps': 1000,
            'max_seq_length': 128,
            'learning_rate': 2e-5,
            'temperature': 0.05  # For Multiple Negatives Ranking Loss
        }
        
        logger.info(f"🚀 PhoBERT Retriever Trainer initialized")
        logger.info(f"   📁 Output directory: {output_dir}")
        logger.info(f"   🎯 Base model: {base_model_name}")
    
    def load_data_from_csv(self, csv_path=None):
        """
        Load và prepare training data từ QA.csv
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
        Prepare training examples cho sentence-transformers
        
        Tạo positive pairs (question, answer) và negative sampling
        """
        logger.info("🔧 Preparing training examples...")
        
        # Convert to list of texts
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        
        # Create positive pairs (question, answer)
        positive_pairs = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Clean text
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
            
            # Also create reverse pair (answer, question) for robustness
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
    
    def create_evaluation_data(self):
        """
        Tạo evaluation data từ eval_examples
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
        Main training function
        """
        logger.info("🚀 Starting PhoBERT fine-tuning for retrieval...")
        
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
        
        # Create loss function - Multiple Negatives Ranking Loss
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        # Create evaluator
        evaluator = self.create_evaluation_data()
        
        # Training arguments
        num_epochs = self.config['epochs']
        warmup_steps = min(
            self.config['warmup_steps'], 
            len(train_dataloader) * num_epochs // 10
        )
        
        logger.info(f"🎯 Training configuration:")
        logger.info(f"   📊 Training examples: {len(self.train_examples)}")
        logger.info(f"   📊 Evaluation examples: {len(self.eval_examples)}")
        logger.info(f"   🔄 Epochs: {num_epochs}")
        logger.info(f"   📦 Batch size: {self.config['batch_size']}")
        logger.info(f"   🔥 Warmup steps: {warmup_steps}")
        logger.info(f"   📏 Max sequence length: {self.config['max_seq_length']}")
        
        # Start training
        start_time = time.time()
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=self.config['evaluation_steps'],
            save_best_model=True,
            output_path=self.output_dir,
            optimizer_params={'lr': self.config['learning_rate']},
            scheduler='WarmupLinear'
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"💾 Model saved to: {self.output_dir}")
        
        return True
    
    def save_training_metadata(self):
        """
        Save training metadata và configuration
        """
        try:
            metadata = {
                'training_date': datetime.now().isoformat(),
                'base_model': self.base_model_name,
                'output_directory': self.output_dir,
                'training_config': self.config,
                'training_examples_count': len(self.train_examples),
                'eval_examples_count': len(self.eval_examples),
                'model_type': 'sentence_transformer_retrieval',
                'fine_tuning_method': 'MultipleNegativesRankingLoss',
                'version': '1.0'
            }
            
            metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Training metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving metadata: {str(e)}")
    
    def evaluate_model(self):
        """
        Evaluate fine-tuned model performance
        """
        if not self.model:
            logger.error("❌ No model loaded for evaluation")
            return None
        
        try:
            logger.info("📊 Evaluating fine-tuned model...")
            
            # Create evaluation subset
            eval_subset = self.eval_examples[:50]  # Smaller subset for faster eval
            
            # Calculate embeddings
            questions = [example.texts[0] for example in eval_subset]
            answers = [example.texts[1] for example in eval_subset]
            
            question_embeddings = self.model.encode(questions)
            answer_embeddings = self.model.encode(answers)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(question_embeddings, answer_embeddings)
            
            # Calculate accuracy (diagonal should have highest similarity)
            correct_predictions = 0
            for i in range(len(similarities)):
                if np.argmax(similarities[i]) == i:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(similarities)
            avg_similarity = np.mean(np.diag(similarities))
            
            logger.info(f"📊 Evaluation Results:")
            logger.info(f"   🎯 Accuracy: {accuracy:.4f}")
            logger.info(f"   📈 Average similarity: {avg_similarity:.4f}")
            
            return {
                'accuracy': accuracy,
                'average_similarity': avg_similarity,
                'eval_samples': len(eval_subset)
            }
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
            return None

def run_training(csv_path=None, output_dir='./fine_tuned_phobert'):
    """
    Main function để chạy training process
    """
    try:
        logger.info("🚀 Starting PhoBERT Retriever Fine-tuning Process...")
        
        # Initialize trainer
        trainer = PhoBERTRetrieverTrainer(output_dir=output_dir)
        
        # Load data
        df = trainer.load_data_from_csv(csv_path)
        if df.empty:
            logger.error("❌ No training data available")
            return False
        
        # Prepare training examples
        num_examples = trainer.prepare_training_examples(df)
        if num_examples == 0:
            logger.error("❌ No valid training examples created")
            return False
        
        # Train model
        success = trainer.train_model()
        if not success:
            logger.error("❌ Training failed")
            return False
        
        # Save metadata
        trainer.save_training_metadata()
        
        # Evaluate model
        eval_results = trainer.evaluate_model()
        
        logger.info("✅ PhoBERT Retriever fine-tuning completed successfully!")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'training_examples': num_examples,
            'evaluation_results': eval_results
        }
        
    except Exception as e:
        logger.error(f"❌ Training process failed: {str(e)}")
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
    # Direct script execution
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check GPU
    check_gpu_availability()
    
    # Run training
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './fine_tuned_phobert'
    
    result = run_training(csv_path, output_dir)
    
    if result.get('success'):
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {result['output_dir']}")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)