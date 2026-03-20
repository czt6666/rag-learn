"""模块 0：数据拉取"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from config import get_settings

logger = logging.getLogger(__name__)


class DataDownloader:
    """数据下载和组织器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.dataset_path = Path(self.settings.dataset_path)
        
    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.dataset_path,
            self.dataset_path / "images",
            Path(self.settings.output_path),
            Path(self.settings.chromadb_path).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {directory}")
    
    def verify_dataset(self) -> Dict[str, bool]:
        """验证数据集是否存在"""
        checks = {
            "docs_jsonl": Path(self.settings.docs_jsonl).exists(),
            "qa_jsonl": Path(self.settings.qa_jsonl).exists(),
            "images_dir": Path(self.settings.images_path).exists(),
        }
        
        for name, exists in checks.items():
            status = "✓" if exists else "✗"
            logger.info(f"{status} {name}: {exists}")
        
        return checks
    
    def load_docs(self) -> List[Dict]:
        """加载文档 JSONL"""
        docs = []
        docs_path = Path(self.settings.docs_jsonl)
        
        if not docs_path.exists():
            logger.error(f"文档文件不存在: {docs_path}")
            return docs
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        
        logger.info(f"加载了 {len(docs)} 个文档")
        return docs
    
    def load_qa(self) -> List[Dict]:
        """加载问答 JSONL"""
        qa_pairs = []
        qa_path = Path(self.settings.qa_jsonl)
        
        if not qa_path.exists():
            logger.error(f"QA 文件不存在: {qa_path}")
            return qa_pairs
        
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        
        logger.info(f"加载了 {len(qa_pairs)} 个 QA 对")
        return qa_pairs
    
    def get_image_path(self, image_id: str) -> Path:
        """获取图像文件路径"""
        images_dir = Path(self.settings.images_path)
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = images_dir / f"{image_id}{ext}"
            if image_path.exists():
                return image_path
        
        logger.warning(f"图像不存在: {image_id}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = DataDownloader()
    downloader.setup_directories()
    downloader.verify_dataset()
