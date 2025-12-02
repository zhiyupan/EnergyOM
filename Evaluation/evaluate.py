import pandas as pd
import math
from collections import Counter
from Data_Processing.utilizing import get_file_path


class F1Score(object):
    """
    A class for evaluating the F1 score, precision, recall, and reciprocal rank based on provided answers and results.
    """

    def evaluate_f1_score(self, TP, TP_FP, TP_FN):
        rank = 0
        # Calculate precision, recall, and F1 Score
        precision = (TP / TP_FP) if TP_FP != 0 else 0  # Precision = TP/(TP+FP)
        recall = (TP / TP_FN) if TP_FN != 0 else 0  # Recall = TP/(TP+FN)
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        if rank == 0:
            reciprocal_rank = 0
        else:
            reciprocal_rank = 1 / rank

        return precision, recall, f1_score, reciprocal_rank

    def get_f1_results(self, result_path, answer_path):
        total_TP = 0
        sum_reciprocal_rank = 0
        metrics = []
        table_names = []

        # 读入结果和答案
        result = pd.read_excel(result_path)
        answer = pd.read_excel(answer_path)

        # 如果预测结果本身是空表(0行)，直接返回全0指标，避免除零
        if len(result) == 0:
            # 我们还是想知道 gold 有多少条，方便你看 recall=0
            first_two_column_answer = list(zip(answer.iloc[:, 0], answer.iloc[:, 1]))
            # 只统计那些答案中第二列不是 NaN 的
            gold_valid = [
                item for item in first_two_column_answer
                if item[1] and not (isinstance(item[1], float) and math.isnan(item[1]))
            ]
            total_TP_FP = 0  # 我们没有任何预测
            total_TP_FN = len(gold_valid)  # gold 有多少条
            total_precision = 0.0
            total_recall = 0.0
            total_f1_score = 0.0
            mean_reciprocal_rank = 0.0

            # 直接构建一行 "Total"
            metrics.append([
                0,  # TP
                total_TP_FP,  # TP+FP
                total_TP_FN,  # TP+FN
                total_precision * 100,
                total_recall * 100,
                total_f1_score * 100,
                mean_reciprocal_rank
            ])
            table_names.append("Total")

            df_metrics = pd.DataFrame(
                metrics,
                index=table_names,
                columns=['TP', 'TP+FP', 'TP+FN', 'Precision', 'Recall', 'F1 Score', 'MRR']
            )
            return df_metrics

        # ============ 正常情况 (result 非空) ============

        first_two_column_result = list(zip(result.iloc[:, 0], result.iloc[:, 1]))
        first_two_column_answer = list(zip(answer.iloc[:, 0], answer.iloc[:, 1]))

        # 从预测里：拿到所有非空 target 的 source 名称
        result_name = [item[0] for item in first_two_column_result if item[1]]
        tp_fp_counts = Counter(result_name)
        total_TP_FP = sum(tp_fp_counts.values())

        # 从答案里：拿到所有有效 gold (过滤掉 NaN)
        answer_name_list = [
            item[0] for item in first_two_column_answer
            if item[1] and not (isinstance(item[1], float) and math.isnan(item[1]))
        ]
        tp_fn_counts = Counter(answer_name_list)
        total_TP_FN = sum(tp_fn_counts.values())

        for index, item in enumerate(first_two_column_answer):
            TP = 0
            match_name = item[0]

            # 这个拼接形式是你原来写的: answer_name = item[0] + str(item[1])
            # 下面保持原逻辑，不去推测你的真实意图
            answer_name_combo = item[0] + str(item[1])

            if item in first_two_column_result:
                # 命中
                TP += 1
                TP_FP = result_name.count(match_name)
                TP_FN = answer_name_list.count(match_name)  # 这里用 answer_name_list 替代原本的自引用错误
                precision, recall, f1_score, reciprocal_rank = self.evaluate_f1_score(TP, TP_FP, TP_FN)

                metrics.append([
                    TP,
                    TP_FP,
                    TP_FN,
                    precision * 100,
                    recall * 100,
                    f1_score * 100,
                    reciprocal_rank
                ])
                table_names.append(match_name)

                total_TP += TP
                sum_reciprocal_rank += reciprocal_rank
            else:
                # 没命中
                TP_FP = result_name.count(match_name)
                TP_FN = answer_name_list.count(match_name)
                precision, recall, f1_score, reciprocal_rank = self.evaluate_f1_score(TP, TP_FP, TP_FN)

                metrics.append([
                    TP,
                    TP_FP,
                    TP_FN,
                    precision * 100,
                    recall * 100,
                    f1_score * 100,
                    reciprocal_rank
                ])
                table_names.append(match_name)

        # 汇总行
        total_precision = total_TP / total_TP_FP if total_TP_FP != 0 else 0
        total_recall = total_TP / total_TP_FN if total_TP_FN != 0 else 0
        total_f1_score = (
            2 * total_precision * total_recall / (total_precision + total_recall)
            if (total_precision + total_recall) != 0 else 0
        )

        # 关键：len(result) 现在一定 >0（我们在上面已经 return 过空表情况）
        mean_reciprocal_rank = sum_reciprocal_rank / len(result)

        metrics.append([
            total_TP,
            total_TP_FP,
            total_TP_FN,
            total_precision * 100,
            total_recall * 100,
            total_f1_score * 100,
            mean_reciprocal_rank
        ])
        table_names.append("Total")

        df_metrics = pd.DataFrame(
            metrics,
            index=table_names,
            columns=['TP', 'TP+FP', 'TP+FN', 'Precision', 'Recall', 'F1 Score', 'MRR']
        )

        return df_metrics

    def save_evaluate_result(self,result_path, answer_path, top_k, output_folder):
        if top_k < 10:
            file_name = "top_0" + str(top_k)
        else:
            file_name = "top_" + str(top_k)
        df_metrics = self.get_f1_results(result_path, answer_path)
        # Save results
        output_path = get_file_path(output_folder, file_name, suffix="_evaluation.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            df_metrics.to_excel(writer, index=True)
        print(f"Evaluated results has been saved:{output_path}")


if __name__ == '__main__':
    result_path = r"../energy_domain_experiment/agentOM/sargon_sbeo.xlsx"
    answer_path = r"..\energy_domain_experiment\answer\sargon_sbeo_answer.xlsx"
    output = r"..\energy_domain_experiment\agentOM/sargon_sbeo"
    eva = F1Score()
    print(eva.get_f1_results(result_path, answer_path))
    eva.save_evaluate_result(result_path,answer_path,1,output)