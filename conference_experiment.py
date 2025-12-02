from Experiment.Onto2Onto.mapping_methods import ClassMatching, PropertyMatching
from Evaluation.evaluate import F1Score
from itertools import combinations
from Model.sBERT import SentenceTransformers
from Model.bert_Energy_tsdae import BertEnergy
from Model.deepseek import DeepSeekMatcher

from Data_Processing.utilizing import merge_class_property, excel_to_dict, dict_to_excel, merge_final_results, \
    exchange_two_columns, combine_two_files

from pathlib import Path
import os

class_threshold = 0.8
property_threshold = 0.7

eva = F1Score()
# set path of ontology
cmt_path = r".\Data\agent\ontology\cmt.xml"
conference_path = r".\Data\agent\ontology\conference.xml"
confof_path = r".\Data\agent\ontology\confof.xml"
edas_path = r".\Data\agent\ontology\edas.xml"
ekaw_path = r".\Data\agent\ontology\ekaw.xml"
iasted_path = r".\Data\agent\ontology\iasted.xml"
sigkdd_path = r".\Data\agent\ontology\sigkdd.xml"

# set path of answer
cmt_conference_answer_path = r".\Data\agent\answer\cmt_conference_answer.xlsx"
cmt_confof_answer_path = r".\Data\agent\answer\cmt_confof_answer.xlsx"
cmt_edas_answer_path = r".\Data\agent\answer\cmt_edas_answer.xlsx"
cmt_ekaw_answer_path = r".\Data\agent\answer\cmt_ekaw_answer.xlsx"
cmt_iasted_answer_path = r".\Data\agent\answer\cmt_iasted_answer.xlsx"
cmt_sigkdd_answer_path = r".\Data\agent\answer\cmt_sigkdd_answer.xlsx"
conference_confof_answer_path = r".\Data\agent\answer\conference_confof_answer.xlsx"
conference_edas_answer_path = r".\Data\agent\answer\conference_edas_answer.xlsx"
conference_ekaw_answer_path = r".\Data\agent\answer\conference_ekaw_answer.xlsx"
conference_iasted_answer_path = r".\Data\agent\answer\conference_iasted_answer.xlsx"
conference_sigkdd_answer_path = r".\Data\agent\answer\conference_sigkdd_answer.xlsx"
confof_edas_answer_path = r".\Data\agent\answer\confof_edas_answer.xlsx"
confof_ekaw_answer_path = r".\Data\agent\answer\confof_ekaw_answer.xlsx"
confof_iasted_answer_path = r".\Data\agent\answer\confof_iasted_answer.xlsx"
confof_sigkdd_answer_path = r".\Data\agent\answer\confof_sigkdd_answer.xlsx"
edas_ekaw_answer_path = r".\Data\agent\answer\edas_ekaw_answer.xlsx"
edas_iasted_answer_path = r".\Data\agent\answer\edas_iasted_answer.xlsx"
edas_sigkdd_answer_path = r".\Data\agent\answer\edas_sigkdd_answer.xlsx"
ekaw_iasted_answer_path = r".\Data\agent\answer\ekaw_iasted_answer.xlsx"
ekaw_sigkdd_answer_path = r".\Data\agent\answer\ekaw_sigkdd_answer.xlsx"
iasted_sigkdd_answer_path = r".\Data\agent\answer\iasted_sigkdd_answer.xlsx"

ontology_paths = {
    "cmt": cmt_path,
    "conference": conference_path,
    "confof": confof_path,
    "edas": edas_path,
    "ekaw": ekaw_path,
    "iasted": iasted_path,
    "sigkdd": sigkdd_path,
}
answer_paths = {
    "cmt_conference": cmt_conference_answer_path,
    "cmt_confof": cmt_confof_answer_path,
    "cmt_edas": cmt_edas_answer_path,
    "cmt_ekaw": cmt_ekaw_answer_path,
    "cmt_iasted": cmt_iasted_answer_path,
    "cmt_sigkdd": cmt_sigkdd_answer_path,
    "conference_confof": conference_confof_answer_path,
    "conference_edas": conference_edas_answer_path,
    "conference_ekaw": conference_ekaw_answer_path,
    "conference_iasted": conference_iasted_answer_path,
    "conference_sigkdd": conference_sigkdd_answer_path,
    "confof_edas": confof_edas_answer_path,
    "confof_ekaw": confof_ekaw_answer_path,
    "confof_iasted": confof_iasted_answer_path,
    "confof_sigkdd": confof_sigkdd_answer_path,
    "edas_ekaw": edas_ekaw_answer_path,
    "edas_iasted": edas_iasted_answer_path,
    "edas_sigkdd": edas_sigkdd_answer_path,
    "ekaw_iasted": ekaw_iasted_answer_path,
    "ekaw_sigkdd": ekaw_sigkdd_answer_path,
    "iasted_sigkdd": iasted_sigkdd_answer_path,
}
all_names = ["cmt", "conference", "confof", "edas", "ekaw", "iasted", "sigkdd"]

sbert_model = SentenceTransformers()
energy_bert_model = BertEnergy()
deepseek_model = DeepSeekMatcher()

""""""


# 1.Experiment: use SentenceBERT, name-,hierarchy-,comment-method for class matching,
#               name-, structure-method for property matching and just forward
#
def experiment_1():
    # Set Path of 1.Experiment
    experiment_1_save_folder = f"./Data/agent/experiment_1_results"
    os.makedirs(experiment_1_save_folder, exist_ok=True)

    results_path_1 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 1====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_1[pair_name] = {}

        save_folder = os.path.join(experiment_1_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, sbert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=True,
                                                                 threshold=class_threshold, top_k=1)
        results_path_1[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_1[pair_name]["data_property"] = data_property_result_path

        results_path_1[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_1[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_1[pair_name]["class"],
                                                  results_path_1[pair_name]["object_property"],
                                                  results_path_1[pair_name]["data_property"],
                                                  results_path_1[pair_name]["merged"])
        print(print(f"----------Starting Evaluating----------"))
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_1_save_folder + "/experiment_1_total_results.xlsx"
    merge_final_results(all_names, experiment_1_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 1=========================")
    print("=====================================================================")


# 2.Experiment: use SentenceBERT, name-,hierarchy- for class matching,
#               name-, structure-method for property matching and just forward
def experiment_2():
    # Set Path of 2.Experiment
    experiment_2_save_folder = f"./Data/agent/experiment_2_2_results"
    os.makedirs(experiment_2_save_folder, exist_ok=True)

    results_path_2 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 2====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_2[pair_name] = {}

        save_folder = os.path.join(experiment_2_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, sbert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=False,
                                                                 threshold=class_threshold, top_k=1)
        results_path_2[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_2[pair_name]["data_property"] = data_property_result_path

        results_path_2[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_2[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_2[pair_name]["class"],
                                                  results_path_2[pair_name]["object_property"],
                                                  results_path_2[pair_name]["data_property"],
                                                  results_path_2[pair_name]["merged"])
        print(print(f"----------Starting Evaluating----------"))
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_2_save_folder + "/experiment_2_2_total_results.xlsx"
    merge_final_results(all_names, experiment_2_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 2=========================")
    print("=====================================================================")


def experiment_2_1():
    # Top 3
    # Set Path of 2.Experiment
    experiment_2_1_save_folder = f"./Data/agent/experiment_2_1_results"
    os.makedirs(experiment_2_1_save_folder, exist_ok=True)
    top_k = 3
    results_path_2_1 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 2_1====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_2_1[pair_name] = {}

        save_folder = os.path.join(experiment_2_1_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, sbert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=False,
                                                                 threshold=class_threshold, top_k=top_k)
        results_path_2_1[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=top_k)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=top_k)
        results_path_2_1[pair_name]["data_property"] = data_property_result_path

        results_path_2_1[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_2_1[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_2_1[pair_name]["class"],
                                                  results_path_2_1[pair_name]["object_property"],
                                                  results_path_2_1[pair_name]["data_property"],
                                                  results_path_2_1[pair_name]["merged"])
        print(print(f"----------Starting Evaluating----------"))
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=top_k,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_2_1_save_folder + "/experiment_2_1_total_results.xlsx"
    merge_final_results(all_names, experiment_2_1_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 2_1=========================")
    print("=====================================================================")


# 3.Experiment: use BERTEnergy, name-,hierarchy- for class matching,
#               name-, structure-method for property matching and just forward
def experiment_3():
    # Set Path of 3.Experiment
    experiment_3_save_folder = f"./Data/agent/experiment_3_results"
    os.makedirs(experiment_3_save_folder, exist_ok=True)

    results_path_3 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 3====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_3[pair_name] = {}

        save_folder = os.path.join(experiment_3_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, energy_bert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=False,
                                                                 threshold=class_threshold, top_k=1)
        results_path_3[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, energy_bert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, energy_bert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_3[pair_name]["data_property"] = data_property_result_path

        results_path_3[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_3[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_3[pair_name]["class"],
                                                  results_path_3[pair_name]["object_property"],
                                                  results_path_3[pair_name]["data_property"],
                                                  results_path_3[pair_name]["merged"])

        print(print(f"----------Starting Evaluating----------"))

        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_3_save_folder + "/experiment_3_total_results.xlsx"
    merge_final_results(all_names, experiment_3_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 3=========================")
    print("=====================================================================")


# 4.Experiment: use DeepSeek, Input:Dict from 3. experiment
def experiment_4():
    # Set Path of 4.Experiment
    experiment_4_save_folder = f"./Data/agent/experiment_4_results"
    os.makedirs(experiment_4_save_folder, exist_ok=True)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        merge_folder = Path(f"./Data/agent/experiment_3_results/{pair_name}/merged_result")
        for p in merge_folder.iterdir():
            d = excel_to_dict(p)
            matches = deepseek_model.test_for_equivalence(d)
            matching_result_path = experiment_4_save_folder + f"/{pair_name}/result/result.xlsx"
            dict_to_excel(matches, matching_result_path)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        save_folder = os.path.join(experiment_4_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        result_path = save_folder + f"/result/result.xlsx"
        eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)

    final_result_path = experiment_4_save_folder + "/experiment_4_total_results.xlsx"
    merge_final_results(all_names, experiment_4_save_folder, final_result_path)


# 5.Experiment: use DeepSeek, Input:Dict from 2. experiment
def experiment_5():
    # Set Path of 5.Experiment
    experiment_5_save_folder = f"./Data/agent/experiment_5_results"
    os.makedirs(experiment_5_save_folder, exist_ok=True)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        merge_folder = Path(f"./Data/agent/experiment_2_results/{pair_name}/merged_result")
        for p in merge_folder.iterdir():
            d = excel_to_dict(p)
            matches = deepseek_model.test_for_equivalence(d)
            matching_result_path = experiment_5_save_folder + f"/{pair_name}/result/result.xlsx"
            dict_to_excel(matches, matching_result_path)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        save_folder = os.path.join(experiment_5_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        result_path = save_folder + f"/result/result.xlsx"
        eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)

    final_result_path = experiment_5_save_folder + "/experiment_5_total_results.xlsx"
    merge_final_results(all_names, experiment_5_save_folder, final_result_path)


def experiment_6():
    # Set Path of 6.Experiment
    experiment_6_save_folder = f"./Data/agent/experiment_6_results"
    os.makedirs(experiment_6_save_folder, exist_ok=True)
    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        merge_folder = Path(f"./Data/agent/experiment_2_1_results/{pair_name}/merged_result")
        for p in merge_folder.iterdir():
            d = excel_to_dict(p)
            matches = deepseek_model.test_for_equivalence(d)
            matching_result_path = experiment_6_save_folder + f"/{pair_name}/result/result.xlsx"
            dict_to_excel(matches, matching_result_path)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        save_folder = os.path.join(experiment_6_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        result_path = save_folder + f"/result/result.xlsx"
        eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)

    final_result_path = experiment_6_save_folder + "/experiment_6_total_results.xlsx"
    merge_final_results(all_names, experiment_6_save_folder, final_result_path)


# name and hierarchy based on deepseek
def experiment_7():
    # Set Path of 7.Experiment
    experiment_7_save_folder = f"./Data/agent/experiment_7_results"
    os.makedirs(experiment_7_save_folder, exist_ok=True)
    results_path_7 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 7====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_7[pair_name] = {}

        save_folder = os.path.join(experiment_7_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, deepseek_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=False,
                                                                 threshold=class_threshold, top_k=1)
        results_path_7[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, deepseek_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=False,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, deepseek_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=False,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_7[pair_name]["data_property"] = data_property_result_path

        results_path_7[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_7[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_7[pair_name]["class"],
                                                  results_path_7[pair_name]["object_property"],
                                                  results_path_7[pair_name]["data_property"],
                                                  results_path_7[pair_name]["merged"])

        print(print(f"----------Starting Evaluating----------"))

        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_7_save_folder + "/experiment_7_total_results.xlsx"
    merge_final_results(all_names, experiment_7_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 7=========================")
    print("=====================================================================")


# Bidirectional Matching based on 2.Experiment
def experiment_8():
    # Set Path of 8.Experiment
    experiment_8_save_folder = f"./Data/agent/experiment_8_results"
    os.makedirs(experiment_8_save_folder, exist_ok=True)

    results_path_8 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 8====================")
    print("=====================================================================")
    """
    Swapping src and tgt
    """

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_8[pair_name] = {}

        save_folder = os.path.join(experiment_8_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[tgt]
        tgt_path = ontology_paths[src]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, sbert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=False,
                                                                 threshold=class_threshold, top_k=1)
        results_path_8[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_8[pair_name]["data_property"] = data_property_result_path

        results_path_8[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_8[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_8[pair_name]["class"],
                                                  results_path_8[pair_name]["object_property"],
                                                  results_path_8[pair_name]["data_property"],
                                                  results_path_8[pair_name]["merged"])

        print(f"----------Exchange the Columns and Combine bidirectional Results----------")
        reversed_result_path = merged_result_folder + "/reversed.xlsx"
        exchange_two_columns(merged_result_path, reversed_result_path)
        original_merged_result_path = f"./Data/agent/experiment_2_results/{src}_{tgt}/merged_result/merged.xlsx"
        combined_result_path = merged_result_folder + "/combined.xlsx"
        combine_two_files(original_merged_result_path, reversed_result_path, combined_result_path)

        print(f"----------Starting Evaluating----------")
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=combined_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_8_save_folder + "/experiment_8_total_results.xlsx"
    merge_final_results(all_names, experiment_8_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 8=========================")
    print("=====================================================================")


"""
def experiment_9():
    experiment_9_save_folder = f"./Data/agent/experiment_9_results"
    os.makedirs(experiment_9_save_folder, exist_ok=True)

    results_path_9 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 9====================")
    print("=====================================================================")
    

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_9[pair_name] = {}

        save_folder = os.path.join(experiment_9_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        print(f"----------Exchange the Columns and Combine bidirectional Results----------")

        merged_result_path = f"./Data/agent/experiment_8_results/{src}_{tgt}/merged_result/merged.xlsx"
        reversed_result_path = f"./Data/agent/experiment_9_results/{src}_{tgt}/merged_result/reversed.xlsx"
        Path(reversed_result_path).parent.mkdir(parents=True, exist_ok=True)
        exchange_two_columns(merged_result_path, reversed_result_path)
        original_merged_result_path = f"./Data/agent/experiment_1_results/{src}_{tgt}/merged_result/merged.xlsx"
        combined_result_path = f"./Data/agent/experiment_9_results/{src}_{tgt}/merged_result/combined.xlsx"
        combine_two_files(original_merged_result_path, reversed_result_path, combined_result_path)

        print(f"----------Starting Evaluating----------")
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=combined_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_9_save_folder + "/experiment_9_total_results.xlsx"
    merge_final_results(all_names, experiment_9_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 9=========================")
    print("=====================================================================")

"""


# Bidirectional Matching based on 1.Experiment (best without llm)
def experiment_9():
    # Set Path of 1.Experiment
    experiment_9_save_folder = f"./Data/agent/experiment_9_results"
    os.makedirs(experiment_9_save_folder, exist_ok=True)

    results_path_9 = {}
    print("=====================================================================")
    print("====================The Beginning of Experiment 9====================")
    print("=====================================================================")

    for src, tgt in combinations(all_names, 2):
        print(f"***********Starting matching between {src} and {tgt}***********")
        pair_name = f"{src}_{tgt}"
        results_path_9[pair_name] = {}

        save_folder = os.path.join(experiment_9_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)

        src_path = ontology_paths[src]
        tgt_path = ontology_paths[tgt]

        # Class Matching
        print(f"----------Starting Class Matching----------")
        class_matcher = ClassMatching(src_path, tgt_path, sbert_model, save_folder=save_folder)
        class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                 comment_based=True,
                                                                 threshold=class_threshold, top_k=1)
        results_path_9[pair_name]["class"] = class_result_path

        # Object Property Matching
        print(f"----------Starting Object Property Matching----------")
        object_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                   datatype="object_property")
        object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                        structure_based=True,
                                                                                        threshold=property_threshold,
                                                                                        top_k=1)

        # Data Property Matching
        print(f"----------Starting Data Property Matching----------")
        data_property_matcher = PropertyMatching(src_path, tgt_path, sbert_model, save_folder=save_folder,
                                                 datatype="data_property")
        data_property_result_path = data_property_matcher.property_matching_results(name_based=True,
                                                                                    structure_based=True,
                                                                                    threshold=property_threshold,
                                                                                    top_k=1)
        results_path_9[pair_name]["data_property"] = data_property_result_path

        results_path_9[pair_name]["object_property"] = object_property_result_path
        merged_result_folder = save_folder + "/merged_result"
        results_path_9[pair_name]["merged"] = merged_result_folder
        merged_result_path = merge_class_property(results_path_9[pair_name]["class"],
                                                  results_path_9[pair_name]["object_property"],
                                                  results_path_9[pair_name]["data_property"],
                                                  results_path_9[pair_name]["merged"])

        print(f"----------Exchange the Columns and Combine bidirectional Results----------")
        reversed_result_path = merged_result_folder + "/reversed.xlsx"
        exchange_two_columns(merged_result_path, reversed_result_path)
        original_merged_result_path = f"./Data/agent/experiment_1_results/{src}_{tgt}/merged_result/merged.xlsx"
        combined_result_path = merged_result_folder + "/combined.xlsx"
        combine_two_files(original_merged_result_path, reversed_result_path, combined_result_path)

        print(print(f"----------Starting Evaluating----------"))
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)
        print(f"***********Finish matching between {src} and {tgt}***********")
        print(f"-------------------------------------------------------------")

    final_result_path = experiment_9_save_folder + "/experiment_9_total_results.xlsx"
    merge_final_results(all_names, experiment_9_save_folder, final_result_path)

    print("=====================================================================")
    print("=========================Finish Experiment 9=========================")
    print("=====================================================================")


# 10.Experiment: use DeepSeek, Input:Dict from 9. experiment
def experiment_10():
    # Set Path of 10.Experiment
    experiment_10_save_folder = f"./Data/agent/experiment_10_results"
    os.makedirs(experiment_10_save_folder, exist_ok=True)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        merge_folder = Path(f"./Data/agent/experiment_9_results/{pair_name}/merged_result")
        target_file = merge_folder / "combined.xlsx"
        if target_file.exists():
            d = excel_to_dict(target_file)
            matches = deepseek_model.test_for_equivalence(d)
            matching_result_path = experiment_10_save_folder + f"/{pair_name}/result/result.xlsx"
            dict_to_excel(matches, matching_result_path)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        save_folder = os.path.join(experiment_10_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        result_path = save_folder + f"/result/result.xlsx"
        eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)

    final_result_path = experiment_10_save_folder + "/experiment_10_total_results.xlsx"
    merge_final_results(all_names, experiment_10_save_folder, final_result_path)


def experiment_12():
    # Set Path of 12.Experiment
    experiment_12_save_folder = f"./Data/agent/experiment_12_results"
    os.makedirs(experiment_12_save_folder, exist_ok=True)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        merge_folder = Path(f"./Data/agent/experiment_8_results/{pair_name}/merged_result/combined.xlsx")
        d = excel_to_dict(merge_folder)
        matches = deepseek_model.test_for_equivalence(d)
        matching_result_path = experiment_12_save_folder + f"/{pair_name}/result/result.xlsx"
        dict_to_excel(matches, matching_result_path)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        save_folder = os.path.join(experiment_12_save_folder, pair_name)
        os.makedirs(save_folder, exist_ok=True)
        eval_folder = os.path.join(save_folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        answer_path = answer_paths[pair_name]
        result_path = save_folder + f"/result/result.xlsx"
        eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                 output_folder=eval_folder)

    final_result_path = experiment_12_save_folder + "/experiment_12_total_results.xlsx"
    merge_final_results(all_names, experiment_12_save_folder, final_result_path)

if __name__ == '__main__':

    experiment_1()
