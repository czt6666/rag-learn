"""模块 8：评估"""
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from config import get_settings

logger = logging.getLogger(__name__)


class Evaluator:
    """评估器"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def load_qa_pairs(self) -> List[Dict]:
        """加载 QA 对"""
        qa_path = Path(self.settings.qa_jsonl)
        
        if not qa_path.exists():
            logger.error(f"QA 文件不存在: {qa_path}")
            return []
        
        qa_pairs = []
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        
        logger.info(f"加载了 {len(qa_pairs)} 个 QA 对")
        return qa_pairs
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """精确匹配"""
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        return 1.0 if pred == gt else 0.0
    
    def contains_match(self, prediction: str, ground_truth: str) -> float:
        """包含匹配"""
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()
        return 1.0 if gt in pred else 0.0
    
    def token_overlap(self, prediction: str, ground_truth: str) -> float:
        """Token F1"""
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not gt_tokens:
            return 0.0
        
        common = pred_tokens.intersection(gt_tokens)
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_single(
        self, 
        question: str,
        prediction: str,
        ground_truth: Dict,
        retrieval_results: Optional[Dict] = None
    ) -> Dict:
        """评估单个问答对"""
        gt_answer = ground_truth.get('answer', '')
        
        scores = {
            'exact_match': self.exact_match(prediction, gt_answer),
            'contains_match': self.contains_match(prediction, gt_answer),
            'token_f1': self.token_overlap(prediction, gt_answer)
        }
        
        return scores
    
    def evaluate_all(
        self, 
        qa_pairs: List[Dict],
        rag_system,
        save_path: Optional[str] = None
    ) -> Dict:
        """评估所有问答对"""
        all_scores = []
        results = []
        
        logger.info(f"开始评估 {len(qa_pairs)} 个问题")
        
        for qa in tqdm(qa_pairs, desc="评估中"):
            question = qa.get('question', '')
            ground_truth = {
                'answer': qa.get('answer', ''),
                'relevant_docs': qa.get('relevant_docs', []),
                'relevant_images': qa.get('relevant_images', [])
            }
            
            try:
                rag_result = rag_system.query(question)
                prediction = rag_result.get('answer', '')
                retrieval_results = rag_result.get('retrieval_results')
            except Exception as e:
                logger.error(f"查询失败: {e}")
                prediction = ""
                retrieval_results = None
            
            scores = self.evaluate_single(
                question, prediction, ground_truth, retrieval_results
            )
            
            all_scores.append(scores)
            results.append({
                'question': question,
                'prediction': prediction,
                'ground_truth': ground_truth['answer'],
                'scores': scores
            })
        
        avg_scores = self._compute_average_scores(all_scores)
        
        summary = {
            'total_questions': len(qa_pairs),
            'average_scores': avg_scores,
            'detailed_results': results
        }
        
        if save_path:
            self._save_results(summary, save_path)
        
        self._print_summary(avg_scores)
        
        return summary
    
    def _compute_average_scores(self, all_scores: List[Dict]) -> Dict:
        """计算平均分数"""
        if not all_scores:
            return {}
        
        all_metrics = set()
        for scores in all_scores:
            all_metrics.update(scores.keys())
        
        avg_scores = {}
        for metric in all_metrics:
            values = [scores.get(metric, 0.0) for scores in all_scores if metric in scores]
            avg_scores[metric] = sum(values) / len(values) if values else 0.0
        
        return avg_scores
    
    def _save_results(self, summary: Dict, save_path: str):
        """保存评估结果"""
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存: {output_file}")
    
    def _print_summary(self, avg_scores: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        for metric, score in sorted(avg_scores.items()):
            print(f"{metric:.<40} {score:.4f}")
        
        print("="*60)
