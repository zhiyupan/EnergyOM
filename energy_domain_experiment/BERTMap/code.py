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
emkpi_respond_answer_path = r"../answer/emkpi_respond_answer.xlsx"
emkpi_sare4bldg_answer_path = r"../answer/emkpi_sare4bldg_answer.xlsx"
emkpi_Sargon_answer_path = r"../answer/emkpi_Sargon_answer.xlsx"
emkpi_sbeo_answer_path = r"../answer/emkpi_sbeo_answer.xlsx"
respond_sare4bldg_answer_path = r"../answer/respond_sare4bldg_answer.xlsx"
respond_Sargon_answer_path = r"../answer/respond_Sargon_answer.xlsx"
respond_sbeo_answer_path = r"../answer/respond_sbeo_answer.xlsx"
sare4bldg_Sargon_answer_path = r"../answer/sare4bldg_Sargon_answer.xlsx"
sare4bldg_sbeo_answer_path = r"../answer/sare4bldg_sbeo_answer.xlsx"
sargon_sbeo_answer_path = r"../answer/Sargon_sbeo_answer.xlsx"

emkpi_respond_path = r"./emkpi_to_respond.xlsx"
emkpi_sare4bldg_path = r"./emkpi_to_saref4bldg.xlsx"
emkpi_Sargon_path = r"./emkpi_to_Sargon.xlsx"
emkpi_sbeo_path = r"./emkpi_to_sbeo.xlsx"
respond_sare4bldg_path = r"./respond_to_saref4bldg.xlsx"
respond_Sargon_path = r"./respond_to_Sargon.xlsx"
respond_sbeo_path = r"./respond_to_sbeo.xlsx"
sare4bldg_Sargon_path = r"./saref4bldg_to_Sargon.xlsx"
sare4bldg_sbeo_path = r"./saref4bldg_to_sbeo.xlsx"
sargon_sbeo_path = r"./Sargon_to_sbeo.xlsx"
answer_paths = {
    "emkpi_respond": emkpi_respond_answer_path,
    "emkpi_sare4bldg": emkpi_sare4bldg_answer_path,
    "emkpi_Sargon": emkpi_Sargon_answer_path,
    "emkpi_sbeo": emkpi_sbeo_answer_path,
    "respond_sare4bldg": respond_sare4bldg_answer_path,
    "respond_Sargon": respond_Sargon_answer_path,
    "respond_sbeo": respond_sbeo_answer_path,
    "sare4bldg_Sargon": sare4bldg_Sargon_answer_path,
    "sare4bldg_sbeo": sare4bldg_sbeo_answer_path,
    "Sargon_sbeo": sargon_sbeo_answer_path
}
results_path = {
    "emkpi_respond": emkpi_respond_path,
    "emkpi_sare4bldg": emkpi_sare4bldg_path,
    "emkpi_Sargon": emkpi_Sargon_path,
    "emkpi_sbeo": emkpi_sbeo_path,
    "respond_sare4bldg": respond_sare4bldg_path,
    "respond_Sargon": respond_Sargon_path,
    "respond_sbeo": respond_sbeo_path,
    "sare4bldg_Sargon": sare4bldg_Sargon_path,
    "sare4bldg_sbeo": sare4bldg_sbeo_path,
    "Sargon_sbeo": sargon_sbeo_path
}
all_names = ["emkpi", "respond", "sare4bldg", "Sargon", "sbeo"]
for src, tgt in combinations(all_names, 2):
    result_path = {}
    eval_folder = os.path.join("BERTMap", f"{src}_{tgt}")
    pair_name = f"{src}_{tgt}"
    os.makedirs(eval_folder, exist_ok=True)
    answer_path = answer_paths[pair_name]
    result_path = results_path[pair_name]
    eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                             output_folder=eval_folder)
final_result_path = "BERTMap" + f"/experiment_total_results.xlsx"
merge_final_results(all_names, "BERTMap", final_result_path)