"""模块 1：数据解析"""
import re
import json
import logging
from typing import Dict, List
from dataclasses import dataclass, asdict
from pathlib import Path
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MultimodalChunk:
    """多模态文本块"""
    doc_id: str
    chunk_id: str
    text: str
    image_ids: List[str]
    image_captions: List[str]
    joined_text: str
    metadata: Dict = None
    
    def to_dict(self):
        return asdict(self)


class DocumentParser:
    """文档解析器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.pic_pattern = re.compile(r'<PIC>(\w+)</PIC>')
        
    def parse_document(self, doc: Dict) -> List[MultimodalChunk]:
        """解析单个文档"""
        doc_id = doc.get('doc_id', doc.get('id', 'unknown'))
        text = doc.get('text', '')
        images_metadata = doc.get('images', [])
        
        image_caption_map = {
            img['image_id']: img.get('caption', '')
            for img in images_metadata
        }
        
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            chunk_id = f"{doc_id}_chunk_{i}"
            image_ids = self._extract_image_ids(paragraph)
            image_captions = [image_caption_map.get(img_id, '') for img_id in image_ids]
            clean_text = self._remove_pic_tags(paragraph)
            joined_text = self._join_text_and_captions(clean_text, image_captions)
            
            chunk = MultimodalChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=clean_text,
                image_ids=image_ids,
                image_captions=image_captions,
                joined_text=joined_text,
                metadata={'chunk_index': i, 'has_images': len(image_ids) > 0}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """分割为段落"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        return paragraphs
    
    def _extract_image_ids(self, text: str) -> List[str]:
        """提取图像 ID"""
        return self.pic_pattern.findall(text)
    
    def _remove_pic_tags(self, text: str) -> str:
        """移除 <PIC> 标签"""
        return self.pic_pattern.sub('', text).strip()
    
    def _join_text_and_captions(self, text: str, captions: List[str]) -> str:
        """组合文本和图像描述"""
        parts = [text]
        if captions:
            caption_text = " [图像描述: " + "; ".join(
                f"{i+1}. {cap}" for i, cap in enumerate(captions) if cap
            ) + "]"
            parts.append(caption_text)
        return " ".join(parts)
    
    def parse_all_documents(self, docs: List[Dict]) -> List[MultimodalChunk]:
        """解析所有文档"""
        all_chunks = []
        for doc in docs:
            chunks = self.parse_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"共解析 {len(all_chunks)} 个 chunks")
        return all_chunks
    
    def save_chunks(self, chunks: List[MultimodalChunk], output_path: str):
        """保存 chunks"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Chunks 已保存到: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DocumentParser()
    print("DocumentParser 模块加载成功")
