from collections import defaultdict
from Data_Processing.utilizing import get_file_path
import pandas as pd
import os
import glob

from pathlib import Path

class Filter(object):
    def process_similarity(self, match_list):
        grouped_data = defaultdict(list)
        for item in match_list:
            key = item[0]
            grouped_data[key].append(item)
        result = [item for sublist in grouped_data.values() for item in sublist]
        return result

    def get_filter_result(self, *lists, weights=None):
        # 只保留非空 list
        valid_lists = [lst for lst in lists if lst and isinstance(lst, list) and len(lst) > 0]
        if not valid_lists:
            return []
        filtered_lists = [
            [item for item in lst if item is not None and isinstance(item, tuple) and len(item) == 3]
            for lst in valid_lists
        ]
        if not filtered_lists:
            return []
        # 求所有 list 的实体对集合
        sets = [set((item[0], item[1]) for item in lst) for lst in filtered_lists]
        matched_pairs = set.intersection(*sets) if len(sets) > 1 else sets[0] if sets else set()
        unique_pairs = [set_ - matched_pairs for set_ in sets]
        matched_tuples = [
            (pair[0], pair[1], self._get_similarity(pair, *filtered_lists, weights=weights))
            for pair in matched_pairs
        ]
        unique_tuples = []
        all_unique_pairs = set().union(*unique_pairs)
        for pair in all_unique_pairs:
            sim = self._get_similarity(pair, *filtered_lists, weights=weights)
            unique_tuples.append((pair[0], pair[1], sim))

        combined_pairs = matched_tuples + unique_tuples
        unique_dict = {}
        for (e1, e2, sim) in combined_pairs:
            if (e1, e2) not in unique_dict:
                unique_dict[(e1, e2)] = (e1, e2, sim)
            # 如果已存在，直接跳过，不比较分数
        combined_pairs = list(unique_dict.values())
        # 返回所有匹配对的平铺列表
        result = self.process_similarity(combined_pairs)
        return result

    def save_filter_results(self, filter_list, onto_s_name, onto_t_name, result_folder):
        # 允许运行到这里是空结果
        if filter_list is None:
            filter_list = []
        # 扁平化
        if any(isinstance(item, list) for item in filter_list):
            data_list = [it for sub in filter_list for it in sub]
        else:
            data_list = list(filter_list)

        # 校验格式（仅对“非空”数据校验）
        if len(data_list) > 0:
            ok = all(isinstance(item, tuple) and len(item) == 3 for item in data_list)
            if not ok:
                raise ValueError("Invalid data format: expected iterable of (source, target, score)")

        # 构建 DataFrame（空也给标准列）
        if len(data_list) == 0:
            df = pd.DataFrame(columns=['Source', 'Target', 'Cosine Similarity'])
        else:
            df = pd.DataFrame(data_list, columns=['Source', 'Target', 'Cosine Similarity'])

        # 确保目录存在
        out_dir = Path(result_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = '_filter_results.xlsx'
        file_name = onto_s_name + '_' + onto_t_name
        # 如果你有 get_file_path 用它；否则用 pathlib 拼
        try:
            output_path = get_file_path(str(out_dir), file_name, suffix)
        except Exception:
            output_path = str(out_dir / f"{file_name}{suffix}")

        # 写出（空也写）
        df.to_excel(output_path, index=False, engine="openpyxl")
        return output_path

    def _reconstruct_tuple(self, pair, original_list):
        for item in original_list:
            if item[0] == pair[0] and item[1] == pair[1]:
                return item
        return (pair[0], pair[1], 0.0)

    def _get_similarity(self, pair, *lists, weights=None):
        similarities = []
        used_weights = []
        for i, lst in enumerate(lists):
            if lst and hasattr(lst, '__iter__') and len(lst) > 0:
                sim = [item[2] for item in lst if item[0] == pair[0] and item[1] == pair[1]]
                if sim:
                    similarities.append(sum(sim) / len(sim))
                    if weights is not None:
                        used_weights.append(weights[i])
                    else:
                        used_weights.append(1.0)
        if not similarities:
            return 0.0
        # 只要有1直接返回1
        if any(s == 1.0 for s in similarities):
            return 1.0
        # 否则加权平均
        total_weight = sum(used_weights)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in zip(similarities, used_weights)) / total_weight


if __name__ == '__main__':
    filter = Filter()
    list1 = [
        ['test sample', 'visit', 0.75376976],
        ['test sample', 'role', 0.7],
        ['storage', 'building', 0.7028562]
    ]

    list2 = [
        ['test sample', 'visit', 1],
        ['test', 'space', 0.8],
        ['test sample', 'role', 0.8],
        ['storage', 'building', 0.7]
    ]
    list3 = []

    list1 = [tuple(item) for item in list1]
    list2 = [tuple(item) for item in list2]
    list3 = [tuple(item) for item in list3]

    results = filter.get_filter_result(list1, list2, list3)
    print('get filter result: ', results)
    #folder = r'../Data/agent/results/conference/bert/filter'
   #filter.save_filter_results(results, "cmt", "conference", folder)
