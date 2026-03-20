"""生成示例数据"""
import json
from pathlib import Path


def generate_sample_docs():
    """生成示例文档"""
    return [
        {
            "doc_id": "doc_001",
            "text": "人工智能（AI）是计算机科学的一个分支。<PIC>img_001</PIC>\n\n它致力于创建能够模拟人类智能的系统。",
            "images": [{"image_id": "img_001", "caption": "AI 系统架构图"}]
        },
        {
            "doc_id": "doc_002",
            "text": "机器学习是人工智能的核心技术。<PIC>img_002</PIC>\n\n深度学习是机器学习的重要分支。",
            "images": [{"image_id": "img_002", "caption": "神经网络结构图"}]
        },
        {
            "doc_id": "doc_003",
            "text": "自然语言处理（NLP）使计算机能理解人类语言。<PIC>img_003</PIC>",
            "images": [{"image_id": "img_003", "caption": "NLP 任务分类"}]
        }
    ]


def generate_sample_qa():
    """生成示例QA"""
    return [
        {
            "question": "什么是人工智能？",
            "answer": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统",
            "relevant_docs": ["doc_001"],
            "relevant_images": ["img_001"]
        },
        {
            "question": "什么是机器学习？",
            "answer": "机器学习是人工智能的核心技术",
            "relevant_docs": ["doc_002"],
            "relevant_images": ["img_002"]
        }
    ]


def save_sample_data(output_dir: str = "./data/mramg_bench"):
    """保存示例数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    docs = generate_sample_docs()
    with open(output_path / "docs.jsonl", 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    qa_pairs = generate_sample_qa()
    with open(output_path / "qa.jsonl", 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    (output_path / "images").mkdir(exist_ok=True)
    
    print(f"✓ 生成示例数据: {len(docs)} 文档, {len(qa_pairs)} QA")
    print(f"✓ 保存到: {output_path}")


if __name__ == "__main__":
    save_sample_data()
