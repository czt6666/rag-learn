"""
文本切分模块
负责将文本按照不同策略进行切分
"""

from typing import List, Optional
from abc import ABC, abstractmethod


class TextSplitter(ABC):
    """文本切分器基类"""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """切分文本"""
        pass


class SentenceSplitter(TextSplitter):
    """按句子切分"""

    def __init__(self, delimiters: List[str] = None):
        if delimiters is None:
            self.delimiters = ['。', '！', '？', '.', '!', '?', '\n']
        else:
            self.delimiters = delimiters

    def split(self, text: str) -> List[str]:
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in self.delimiters:
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ""

        if current.strip():
            sentences.append(current.strip())

        return sentences


class FixedLengthSplitter(TextSplitter):
    """按固定长度切分（带回溯）"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, backtrack: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.backtrack = backtrack
        self.delimiters = {
            '。', '！', '？', '；', '…',
            '.', '!', '?', ';',
            '\n'
        }

    def split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        n = len(text)

        while start < n:
            target = min(start + self.chunk_size, n)

            # 回溯查找最近的分隔符
            split_pos = None
            search_start = max(start, target - self.backtrack)

            for i in range(target - 1, search_start - 1, -1):
                if text[i] in self.delimiters:
                    split_pos = i + 1
                    break

            end = split_pos if split_pos else target

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 处理重叠
            if self.overlap > 0 and end < n:
                start = max(start + 1, end - self.overlap)
            else:
                start = end

        return chunks


class ParagraphSplitter(TextSplitter):
    """按段落切分"""

    def __init__(self, min_length: int = 10):
        self.min_length = min_length

    def split(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if len(p.strip()) >= self.min_length]


class SemanticSplitter(TextSplitter):
    """按语义单元切分"""

    def __init__(self, max_length: int = 500, sentence_delimiters: List[str] = None):
        self.max_length = max_length
        if sentence_delimiters is None:
            self.sentence_delimiters = ['。', '！', '？', '.', '!', '?']
        else:
            self.sentence_delimiters = sentence_delimiters

    def split(self, text: str) -> List[str]:
        sentence_splitter = SentenceSplitter(self.sentence_delimiters)
        sentences = sentence_splitter.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class CustomSplitter(TextSplitter):
    """自定义分隔符切分"""

    def __init__(self, separator: str = '\n', keep_separator: bool = False):
        self.separator = separator
        self.keep_separator = keep_separator

    def split(self, text: str) -> List[str]:
        if self.keep_separator:
            parts = text.split(self.separator)
            return [p + self.separator if i < len(parts) - 1 else p
                    for i, p in enumerate(parts) if p.strip()]
        else:
            return [p.strip() for p in text.split(self.separator) if p.strip()]


class NSentenceSplitter(TextSplitter):
    """每 N 句切分"""

    def __init__(self, n: int = 1, delimiters: List[str] = None):
        assert n > 0, "n 必须大于 0"
        self.n = n
        self.delimiters = delimiters or ['。', '！', '？', '.', '!', '?']

    def split(self, text: str) -> List[str]:
        sentence_splitter = SentenceSplitter(self.delimiters)
        sentences = sentence_splitter.split(text)

        chunks = []
        buffer = []

        for sentence in sentences:
            buffer.append(sentence)
            if len(buffer) == self.n:
                chunks.append("".join(buffer))
                buffer = []

        if buffer:
            chunks.append("".join(buffer))

        return chunks


def get_splitter(strategy: str, **kwargs) -> TextSplitter:
    """
    获取文本切分器

    Args:
        strategy: 切分策略，可选值：
            - 'sentence': 按句子切分
            - 'fixed': 按固定长度切分（带回溯）
            - 'paragraph': 按段落切分
            - 'semantic': 按语义单元切分
            - 'custom': 自定义分隔符切分
            - 'n_sentence': 每 N 句切分
        **kwargs: 切分器参数

    Returns:
        TextSplitter 实例
    """
    splitters = {
        'sentence': SentenceSplitter,
        'fixed': FixedLengthSplitter,
        'paragraph': ParagraphSplitter,
        'semantic': SemanticSplitter,
        'custom': CustomSplitter,
        'n_sentence': NSentenceSplitter
    }

    if strategy not in splitters:
        raise ValueError(f"未知的切分策略: {strategy}. 可选策略: {list(splitters.keys())}")

    return splitters[strategy](**kwargs)
