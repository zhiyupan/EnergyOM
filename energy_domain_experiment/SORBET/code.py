import pandas as pd
import re
import os
from Evaluation.evaluate import F1Score
from itertools import combinations
from Data_Processing.utilizing import merge_final_results

def clean_value(x):
    if not isinstance(x, str):
        return x
    # 去掉 URL 前缀或命名空间前缀
    x = re.sub(r'.*[#/]', '', x)  # 去掉最后一个 '/' 或 '#' 前面的内容
    x = re.sub(r'.*:', '', x)  # 去掉最后一个 ':' 前面的内容
    return x.strip()


# 去掉 URL 前缀，只保留最后的部分
eva = F1Score()
sare4bldg_Sargon_answer_path = r"../answer/sare4bldg_Sargon_answer.xlsx"
sare4bldg_sbeo_answer_path = r"../answer/sare4bldg_sbeo_answer.xlsx"
sargon_sbeo_answer_path = r"../answer/Sargon_sbeo_answer.xlsx"


sare4bldg_Sargon_path = r"./saref4bldg2Sargon_-1.xlsx"
sare4bldg_sbeo_path = r"./saref4bldg2sbeo_-1.xlsx"
sargon_sbeo_path = r"./Sargon2sbeo_-1.xlsx"
answer_paths = {
    "sare4bldg_Sargon": sare4bldg_Sargon_answer_path,
    "sare4bldg_sbeo": sare4bldg_sbeo_answer_path,
    "Sargon_sbeo": sargon_sbeo_answer_path
}
results_path = {
    "sare4bldg_Sargon": sare4bldg_Sargon_path,
    "sare4bldg_sbeo": sare4bldg_sbeo_path,
    "Sargon_sbeo": sargon_sbeo_path
}
all_names = ["sare4bldg", "Sargon", "sbeo"]
for src, tgt in combinations(all_names, 2):
    result_path = {}
    eval_folder = os.path.join("SORBET", f"{src}_{tgt}")
    pair_name = f"{src}_{tgt}"
    os.makedirs(eval_folder, exist_ok=True)
    answer_path = answer_paths[pair_name]
    result_path = results_path[pair_name]
    eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                             output_folder=eval_folder)
final_result_path = "SORBET" + f"/experiment_total_results.xlsx"
merge_final_results(all_names, "SORBET", final_result_path)