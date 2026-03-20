from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_tagged_corpus(file_path):
    """读取标注语料，返回词序列和词性序列。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'未找到语料文件: {file_path.resolve()}')

    sentences_words = []
    sentences_pos = []

    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) <= 1:
                continue

            parts = parts[1:]  # 跳过句子编号

            words, pos_tags = [], []
            for item in parts:
                if '/' not in item:
                    continue
                word, pos = item.rsplit('/', 1)
                word, pos = word.strip(), pos.strip()
                if word:
                    words.append(word)
                    pos_tags.append(pos)

            if words:
                sentences_words.append(words)
                sentences_pos.append(pos_tags)

    return sentences_words, sentences_pos


corpus_path = './199801UTF8.txt'
sentences_words, sentences_pos = read_tagged_corpus(corpus_path)

print(f'句子数: {len(sentences_words)}')

all_words = [w for sent in sentences_words for w in sent]
all_pos = [p for sent in sentences_pos for p in sent]

word_counter = Counter(all_words)
pos_counter = Counter(all_pos)

top20_words = word_counter.most_common(20)
top10_pos = pos_counter.most_common(10)

print('出现次数最多的20个词：')
for word, cnt in top20_words:
    print(f'{word}: {cnt}')

print('\n出现次数最多的10个词性：')
for pos, cnt in top10_pos:
    print(f'{pos}: {cnt}')

words, word_counts = zip(*top20_words)
plt.figure(figsize=(14, 6))
plt.bar(words, word_counts)
plt.title('Top 20 高频词')
plt.xlabel('词')
plt.ylabel('出现次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

pos_tags, pos_counts = zip(*top10_pos)
plt.figure(figsize=(10, 5))
plt.bar(pos_tags, pos_counts)
plt.title('Top 10 高频词性')
plt.xlabel('词性')
plt.ylabel('出现次数')
plt.tight_layout()
plt.show()

START = '<s>'
END = '</s>'


def build_bigram_model(sentences_words):
    unigram_counter = Counter()
    bigram_counter = Counter()
    vocabulary = set()

    for sent in sentences_words:
        tokens = [START] + sent + [END]
        vocabulary.update(tokens)

        for token in tokens:
            unigram_counter[token] += 1

        for i in range(len(tokens) - 1):
            bigram_counter[(tokens[i], tokens[i + 1])] += 1

    return unigram_counter, bigram_counter, vocabulary


unigram_counter, bigram_counter, vocabulary = build_bigram_model(sentences_words)

print(f'词表大小（含 <s>, </s>）: {len(vocabulary)}')
print(f'Unigram 总数: {sum(unigram_counter.values())}')
print(f'Bigram 种类数: {len(bigram_counter)}')
