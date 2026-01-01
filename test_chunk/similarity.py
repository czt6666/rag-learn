"""
相似度计算模块
提供多种向量相似度计算方法
"""

import numpy as np
from typing import List, Union, Tuple
from scipy.spatial.distance import cosine as scipy_cosine


class SimilarityCalculator:
    """相似度计算器"""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数，范围[0, 1]，越大越相似
        """
        # 归一化向量
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # 计算余弦相似度
        similarity = np.dot(vec1_norm, vec2_norm)

        # 确保在[0, 1]范围内
        return float(np.clip(similarity, 0, 1))

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的欧氏距离

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            欧氏距离，越小越相似
        """
        return float(np.linalg.norm(vec1 - vec2))

    @staticmethod
    def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的曼哈顿距离（L1距离）

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            曼哈顿距离，越小越相似
        """
        return float(np.sum(np.abs(vec1 - vec2)))

    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的点积

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            点积值
        """
        return float(np.dot(vec1, vec2))

    @staticmethod
    def pearson_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的皮尔逊相关系数

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相关系数，范围[-1, 1]
        """
        return float(np.corrcoef(vec1, vec2)[0, 1])

    @staticmethod
    def compare_two_vectors(
            vec1: np.ndarray,
            vec2: np.ndarray,
            method: str = 'cosine'
    ) -> float:
        """
        使用指定方法比较两个向量

        Args:
            vec1: 向量1
            vec2: 向量2
            method: 计算方法
                - 'cosine': 余弦相似度
                - 'euclidean': 欧氏距离
                - 'manhattan': 曼哈顿距离
                - 'dot': 点积
                - 'pearson': 皮尔逊相关系数

        Returns:
            相似度或距离值
        """
        methods = {
            'cosine': SimilarityCalculator.cosine_similarity,
            'euclidean': SimilarityCalculator.euclidean_distance,
            'manhattan': SimilarityCalculator.manhattan_distance,
            'dot': SimilarityCalculator.dot_product,
            'pearson': SimilarityCalculator.pearson_correlation
        }

        if method not in methods:
            raise ValueError(f"未知的计算方法: {method}. 可选: {list(methods.keys())}")

        return methods[method](vec1, vec2)


class BatchSimilarityCalculator:
    """批量相似度计算器"""

    @staticmethod
    def compare_one_to_many(
            query_vector: np.ndarray,
            vectors: Union[List[np.ndarray], np.ndarray],
            method: str = 'cosine',
            return_sorted: bool = True
    ) -> Union[List[float], List[Tuple[int, float]]]:
        """
        将一个向量与多个向量进行比较

        Args:
            query_vector: 查询向量
            vectors: 向量列表或numpy数组 (n, dim)
            method: 计算方法
            return_sorted: 是否返回排序后的结果（按相似度从高到低）

        Returns:
            如果return_sorted=False: 相似度/距离列表
            如果return_sorted=True: [(索引, 相似度/距离), ...] 排序后的列表
        """
        # 转换为numpy数组
        if isinstance(vectors, list):
            vectors = np.array(vectors)

        n_vectors = vectors.shape[0]
        scores = []

        # 计算每个向量的相似度
        for i in range(n_vectors):
            score = SimilarityCalculator.compare_two_vectors(
                query_vector,
                vectors[i],
                method=method
            )
            scores.append(score)

        if not return_sorted:
            return scores

        # 排序（相似度方法降序，距离方法升序）
        if method in ['cosine', 'dot', 'pearson']:
            # 相似度：从大到小
            indexed_scores = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            # 距离：从小到大
            indexed_scores = sorted(
                enumerate(scores),
                key=lambda x: x[1]
            )

        return indexed_scores

    @staticmethod
    def compare_many_to_many(
            vectors1: Union[List[np.ndarray], np.ndarray],
            vectors2: Union[List[np.ndarray], np.ndarray],
            method: str = 'cosine'
    ) -> np.ndarray:
        """
        计算两组向量之间的相似度矩阵

        Args:
            vectors1: 第一组向量 (m, dim)
            vectors2: 第二组向量 (n, dim)
            method: 计算方法

        Returns:
            相似度矩阵 (m, n)
        """
        # 转换为numpy数组
        if isinstance(vectors1, list):
            vectors1 = np.array(vectors1)
        if isinstance(vectors2, list):
            vectors2 = np.array(vectors2)

        m = vectors1.shape[0]
        n = vectors2.shape[0]

        similarity_matrix = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                similarity_matrix[i, j] = SimilarityCalculator.compare_two_vectors(
                    vectors1[i],
                    vectors2[j],
                    method=method
                )

        return similarity_matrix

    @staticmethod
    def find_top_k_similar(
            query_vector: np.ndarray,
            vectors: Union[List[np.ndarray], np.ndarray],
            k: int = 5,
            method: str = 'cosine'
    ) -> List[Tuple[int, float]]:
        """
        找出与查询向量最相似的前k个向量

        Args:
            query_vector: 查询向量
            vectors: 向量列表
            k: 返回前k个结果
            method: 计算方法

        Returns:
            [(索引, 相似度/距离), ...] 列表，长度为min(k, len(vectors))
        """
        sorted_results = BatchSimilarityCalculator.compare_one_to_many(
            query_vector,
            vectors,
            method=method,
            return_sorted=True
        )

        return sorted_results[:k]


# ============= 使用示例 =============

if __name__ == "__main__":
    # 创建一些测试向量
    np.random.seed(42)
    vec1 = np.random.rand(384)
    vec2 = np.random.rand(384)
    vec3 = vec1 + np.random.rand(384) * 0.1  # 与vec1相似

    print("=" * 50)
    print("测试1: 两个向量的相似度比较")
    print("=" * 50)

    calc = SimilarityCalculator()

    print(f"vec1 vs vec2 (余弦相似度): {calc.cosine_similarity(vec1, vec2):.4f}")
    print(f"vec1 vs vec3 (余弦相似度): {calc.cosine_similarity(vec1, vec3):.4f}")
    print(f"vec1 vs vec2 (欧氏距离): {calc.euclidean_distance(vec1, vec2):.4f}")
    print(f"vec1 vs vec3 (欧氏距离): {calc.euclidean_distance(vec1, vec3):.4f}")

    print("\n" + "=" * 50)
    print("测试2: 一个向量与多个向量比较")
    print("=" * 50)

    # 创建多个向量
    vectors = np.random.rand(10, 384)
    vectors[0] = vec1  # 第一个向量与查询向量相同
    vectors[1] = vec3  # 第二个向量与查询向量相似

    query = vec1
    batch_calc = BatchSimilarityCalculator()

    # 计算相似度（不排序）
    similarities = batch_calc.compare_one_to_many(
        query,
        vectors,
        method='cosine',
        return_sorted=False
    )
    print(f"前5个向量的相似度: {[f'{s:.4f}' for s in similarities[:5]]}")

    # 找出最相似的3个
    top_3 = batch_calc.find_top_k_similar(query, vectors, k=3, method='cosine')
    print(f"\n最相似的前3个向量:")
    for idx, score in top_3:
        print(f"  索引 {idx}: 相似度 {score:.4f}")

    print("\n" + "=" * 50)
    print("测试3: 两组向量之间的相似度矩阵")
    print("=" * 50)

    vectors1 = np.random.rand(3, 384)
    vectors2 = np.random.rand(4, 384)

    matrix = batch_calc.compare_many_to_many(
        vectors1,
        vectors2,
        method='cosine'
    )

    print(f"相似度矩阵形状: {matrix.shape}")
    print(f"矩阵内容:\n{matrix.round(4)}")

    print("\n" + "=" * 50)
    print("测试4: 不同相似度方法对比")
    print("=" * 50)

    methods = ['cosine', 'euclidean', 'manhattan', 'dot']
    print(f"比较 vec1 和 vec2:")
    for method in methods:
        score = calc.compare_two_vectors(vec1, vec2, method=method)
        print(f"  {method:12s}: {score:.4f}")