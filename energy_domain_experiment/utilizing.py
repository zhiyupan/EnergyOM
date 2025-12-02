from itertools import combinations
from Evaluation.evaluate import F1Score
from Experiment.Onto2Onto.mapping_methods import ClassMatching, PropertyMatching
from Data_Processing.utilizing import merge_class_property, excel_to_dict, dict_to_excel, merge_final_results, \
    exchange_two_columns, combine_two_files
import os
from pathlib import Path

class_threshold = 0.8
property_threshold = 0.75

# set path of ontology
emkpi_path = r"../energy_domain_experiment/ontology/emkpi.owl"
respond_path = r"../energy_domain_experiment/ontology/respond.owl"
saref4bldg_path = r"../energy_domain_experiment/ontology/saref4bldg.rdf"
sargon_path = r"../energy_domain_experiment/ontology/Sargon.owl"
sbeo_path = r"../energy_domain_experiment/ontology/sbeo.rdf"

# set path of answer
emkpi_respond_answer_path = r"../energy_domain_experiment/answer/emkpi_respond_answer.xlsx"
emkpi_sare4bldg_answer_path = r"../energy_domain_experiment/answer/emkpi_sare4bldg_answer.xlsx"
emkpi_Sargon_answer_path = r"../energy_domain_experiment/answer/emkpi_Sargon_answer.xlsx"
emkpi_sbeo_answer_path = r"../energy_domain_experiment/answer/emkpi_sbeo_answer.xlsx"
respond_sare4bldg_answer_path = r"../energy_domain_experiment/answer/respond_sare4bldg_answer.xlsx"
respond_Sargon_answer_path = r"../energy_domain_experiment/answer/respond_Sargon_answer.xlsx"
respond_sbeo_answer_path = r"../energy_domain_experiment/answer/respond_sbeo_answer.xlsx"
sare4bldg_Sargon_answer_path = r"../energy_domain_experiment/answer/sare4bldg_Sargon_answer.xlsx"
sare4bldg_sbeo_answer_path = r"../energy_domain_experiment/answer/sare4bldg_sbeo_answer.xlsx"
sargon_sbeo_answer_path = r"../energy_domain_experiment/answer/Sargon_sbeo_answer.xlsx"

ontology_paths = {
    "sare4bldg": saref4bldg_path,
    "Sargon": sargon_path,
    "sbeo": sbeo_path
}

answer_paths = {
    "sare4bldg_Sargon": sare4bldg_Sargon_answer_path,
    "sare4bldg_sbeo": sare4bldg_sbeo_answer_path,
    "Sargon_sbeo": sargon_sbeo_answer_path
}
all_names = ["sare4bldg", "Sargon", "sbeo"]
eva = F1Score()


class Experiment(object):
    def __init__(self, model, number):
        self.save_folder = f"../energy_domain_experiment/experiment_{number}_results"
        self.model = model
        self.number = number


    def experiment_without_llm_single_directional(self):
        experiment_save_folder = self.save_folder
        os.makedirs(experiment_save_folder, exist_ok=True)

        results_path_1 = {}
        print("=====================================================================")
        print(f"====================The Beginning of Experiment {self.number}====================")
        print("=====================================================================")

        for src, tgt in combinations(all_names, 2):
            print(f"***********Starting matching between {src} and {tgt}***********")
            pair_name = f"{src}_{tgt}"
            results_path_1[pair_name] = {}

            save_folder = os.path.join(experiment_save_folder, pair_name)
            os.makedirs(save_folder, exist_ok=True)

            src_path = ontology_paths[src]
            tgt_path = ontology_paths[tgt]

            # Class Matching
            print(f"----------Starting Class Matching----------")
            class_matcher = ClassMatching(src_path, tgt_path, self.model, save_folder=save_folder)
            class_result_path = class_matcher.class_matching_results(name_based=True, hierarchy_based=True,
                                                                     comment_based=True,
                                                                     threshold=class_threshold, top_k=1)
            results_path_1[pair_name]["class"] = class_result_path

            # Object Property Matching
            print(f"----------Starting Object Property Matching----------")
            object_property_matcher = PropertyMatching(src_path, tgt_path, self.model, save_folder=save_folder,
                                                       datatype="object_property")
            object_property_result_path = object_property_matcher.property_matching_results(name_based=True,
                                                                                            structure_based=True,
                                                                                            threshold=property_threshold,
                                                                                            top_k=1)

            # Data Property Matching
            print(f"----------Starting Data Property Matching----------")
            data_property_matcher = PropertyMatching(src_path, tgt_path, self.model, save_folder=save_folder,
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

        final_result_path = experiment_save_folder + f"/experiment_{self.number}_total_results.xlsx"
        merge_final_results(all_names, experiment_save_folder, final_result_path)

        print("=====================================================================")
        print(f"=========================Finish Experiment {self.number}=========================")
        print("=====================================================================")

    def experiment_without_llm_bidirectional(self):
        import pandas as pd
        import os
        from pathlib import Path
        from itertools import combinations

        experiment_save_folder = self.save_folder
        os.makedirs(experiment_save_folder, exist_ok=True)

        print("=====================================================================")
        print(f"====================The Beginning of Experiment {self.number} (Bi-dir)====================")
        print("=====================================================================")

        # ---------- helpers ----------
        SAFE_EMPTY_COLUMNS = ["Source", "Target", "Cosine Similarity"]  # 与过滤输出列名一致

        def _ensure_min_cols(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
            cols = list(df.columns)
            while len(cols) < n:
                cols.append(f"col{len(cols)}")
            if len(df.columns) < len(cols):
                df = pd.DataFrame(df, columns=list(df.columns) + cols[len(df.columns):])
            return df

        def _safe_empty_df(columns=None):
            cols = columns if columns else SAFE_EMPTY_COLUMNS
            if len(cols) < 2:
                cols = cols + [f"col{i}" for i in range(2 - len(cols))]
            return pd.DataFrame(columns=cols)

        def _safe_read_excel(path: str, expected_cols=None) -> pd.DataFrame:
            try:
                if path and os.path.isfile(path):
                    df = pd.read_excel(path, engine="openpyxl")
                    if df is None or df.shape[0] == 0:
                        return _safe_empty_df(expected_cols)
                    return _ensure_min_cols(df, n=2)
                else:
                    return _safe_empty_df(expected_cols)
            except Exception as e:
                print(f"[WARN] read_excel failed for '{path}': {e}")
                return _safe_empty_df(expected_cols)

        def _swap_first_two_cols(df: pd.DataFrame) -> pd.DataFrame:
            if df.shape[1] < 2:
                return _ensure_min_cols(df, 2)
            cols = list(df.columns)
            cols[0], cols[1] = cols[1], cols[0]
            return df[cols]

        def _union_dedup_on_first_two(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
            a = _ensure_min_cols(df_a, 2)
            b = _ensure_min_cols(df_b, 2)
            merged = pd.concat([a, b], ignore_index=True)
            merged = merged.drop_duplicates(subset=[merged.columns[0], merged.columns[1]])
            return merged

        def _ensure_file_or_placeholder(path_str: str, folder: str, default_filename: str) -> str:
            """
            若 path 存在则返回；否则在 folder 下创建一个合法空 xlsx（default_filename）并返回。
            """
            folder_p = Path(folder)
            folder_p.mkdir(parents=True, exist_ok=True)
            if path_str and os.path.isfile(path_str):
                return os.path.normpath(path_str)
            out = folder_p / default_filename
            if not out.exists():
                _safe_empty_df(SAFE_EMPTY_COLUMNS).to_excel(out, index=False)
            return str(out)

        def _resolve_result_xlsx(path_str: str, fallback_dir: str, create_empty_if_missing: bool = False) -> str:
            """
            返回一个可读的 xlsx 文件路径：
            1) 若 path 是已存在的文件，直接返回；
            2) 若 path 是目录或不存在，则在 fallback_dir 中查找第一份 *.xlsx；
            3) 若仍找不到且 create_empty_if_missing=True，则创建空的 merged.xlsx 并返回。
            """
            p = Path(os.path.normpath(path_str)) if path_str else None
            if p and p.is_file():
                return str(p)

            fb = Path(os.path.normpath(fallback_dir))
            if fb.is_dir():
                candidates = sorted(fb.glob("*.xlsx"))
                if candidates:
                    return str(candidates[0])

            if create_empty_if_missing:
                fb.mkdir(parents=True, exist_ok=True)
                out = fb / "merged.xlsx"
                if not out.exists():
                    _safe_empty_df(SAFE_EMPTY_COLUMNS).to_excel(out, index=False)
                return str(out)

            if p and p.is_dir():
                candidates = sorted(p.glob("*.xlsx"))
                if candidates:
                    return str(candidates[0])

            raise FileNotFoundError(f"No xlsx file found at '{path_str}' or '{fallback_dir}'")

        # ---------- main loop ----------
        for src, tgt in combinations(all_names, 2):
            print(f"***********Starting bidirectional matching between {src} and {tgt}***********")
            pair_name = f"{src}_{tgt}"

            save_folder = os.path.join(experiment_save_folder, pair_name)
            os.makedirs(save_folder, exist_ok=True)

            tmp_forward = os.path.join(save_folder, "_tmp_forward")
            tmp_backward = os.path.join(save_folder, "_tmp_backward")
            os.makedirs(tmp_forward, exist_ok=True)
            os.makedirs(tmp_backward, exist_ok=True)

            src_path = ontology_paths[src]
            tgt_path = ontology_paths[tgt]

            # ---------- forward ----------
            print(f"----------[FWD] {src} -> {tgt}----------")
            f_cls = ClassMatching(src_path, tgt_path, self.model, save_folder=tmp_forward)
            f_cls_path = f_cls.class_matching_results(
                name_based=True, hierarchy_based=True, comment_based=True,
                threshold=class_threshold, top_k=1
            )
            f_obj = PropertyMatching(src_path, tgt_path, self.model, save_folder=tmp_forward,
                                     datatype="object_property")
            f_obj_path = f_obj.property_matching_results(
                name_based=True, structure_based=True, threshold=property_threshold, top_k=1
            )
            f_dat = PropertyMatching(src_path, tgt_path, self.model, save_folder=tmp_forward, datatype="data_property")
            f_dat_path = f_dat.property_matching_results(
                name_based=True, structure_based=True, threshold=property_threshold, top_k=1
            )

            # 若子结果缺失，写占位空表，避免 merge 崩溃
            f_cls_path = _ensure_file_or_placeholder(f_cls_path, tmp_forward, "class_empty.xlsx")
            f_obj_path = _ensure_file_or_placeholder(f_obj_path, tmp_forward, "object_property_empty.xlsx")
            f_dat_path = _ensure_file_or_placeholder(f_dat_path, tmp_forward, "data_property_empty.xlsx")

            f_merged_folder = os.path.join(tmp_forward, "merged_result")
            os.makedirs(f_merged_folder, exist_ok=True)
            f_merged_path0 = merge_class_property(f_cls_path, f_obj_path, f_dat_path, f_merged_folder)
            f_merged_path = _resolve_result_xlsx(f_merged_path0, f_merged_folder, create_empty_if_missing=True)

            # ---------- backward ----------
            print(f"----------[BWD] {tgt} -> {src}----------")
            b_cls = ClassMatching(tgt_path, src_path, self.model, save_folder=tmp_backward)
            b_cls_path = b_cls.class_matching_results(
                name_based=True, hierarchy_based=True, comment_based=True,
                threshold=class_threshold, top_k=1
            )
            b_obj = PropertyMatching(tgt_path, src_path, self.model, save_folder=tmp_backward,
                                     datatype="object_property")
            b_obj_path = b_obj.property_matching_results(
                name_based=True, structure_based=True, threshold=property_threshold, top_k=1
            )
            b_dat = PropertyMatching(tgt_path, src_path, self.model, save_folder=tmp_backward, datatype="data_property")
            b_dat_path = b_dat.property_matching_results(
                name_based=True, structure_based=True, threshold=property_threshold, top_k=1
            )

            b_cls_path = _ensure_file_or_placeholder(b_cls_path, tmp_backward, "class_empty.xlsx")
            b_obj_path = _ensure_file_or_placeholder(b_obj_path, tmp_backward, "object_property_empty.xlsx")
            b_dat_path = _ensure_file_or_placeholder(b_dat_path, tmp_backward, "data_property_empty.xlsx")

            b_merged_folder = os.path.join(tmp_backward, "merged_result")
            os.makedirs(b_merged_folder, exist_ok=True)
            b_merged_path0 = merge_class_property(b_cls_path, b_obj_path, b_dat_path, b_merged_folder)
            b_merged_path = _resolve_result_xlsx(b_merged_path0, b_merged_folder, create_empty_if_missing=True)

            # ---------- normalize + union ----------
            print(f"----------[Bi-dir] Normalizing backward & Union-dedup to {src}->{tgt}----------")
            f_df = _safe_read_excel(f_merged_path, expected_cols=SAFE_EMPTY_COLUMNS)
            b_df_raw = _safe_read_excel(b_merged_path, expected_cols=SAFE_EMPTY_COLUMNS)
            b_df_norm = _swap_first_two_cols(b_df_raw)  # 反向的 Source/Target 互换到同一方向

            bi_df = _union_dedup_on_first_two(f_df, b_df_norm)

            merged_result_folder = os.path.join(save_folder, "merged_result")
            os.makedirs(merged_result_folder, exist_ok=True)
            merged_result_path = os.path.join(merged_result_folder, "merged.xlsx")
            bi_df.to_excel(merged_result_path, index=False)

            # ---------- evaluate ----------
            print(f"----------Starting Evaluating----------")
            eval_folder = os.path.join(save_folder, "evaluation")
            os.makedirs(eval_folder, exist_ok=True)
            if bi_df.shape[0] == 0:
                print("[INFO] merged is empty -> skip evaluate for this pair.")
                with open(os.path.join(eval_folder, "SKIPPED.txt"), "w", encoding="utf-8") as f:
                    f.write("No predictions; evaluation skipped.\n")
            else:
                answer_path = answer_paths[pair_name]
                eva.save_evaluate_result(result_path=merged_result_path, answer_path=answer_path, top_k=1,
                                         output_folder=eval_folder)

            print(f"***********Finish bidirectional matching between {src} and {tgt}***********")
            print(f"-------------------------------------------------------------")

        # ---------- final merge across pairs ----------
        def _safe_merge_across_pairs(root_dir: str, all_pair_names):
            collected = []
            for pair in all_pair_names:
                p = os.path.join(root_dir, pair, "merged_result", "merged.xlsx")
                if not os.path.isfile(p):
                    print(f"[WARN] missing merged.xlsx for {pair}, skip.")
                    continue
                df = _safe_read_excel(p, expected_cols=SAFE_EMPTY_COLUMNS)
                if df.shape[0] == 0:
                    continue
                df.insert(0, "pair", pair)  # 可选：保留来源
                collected.append(df)
            if not collected:
                return _safe_empty_df(["pair"] + SAFE_EMPTY_COLUMNS)
            return pd.concat(collected, ignore_index=True)

        final_df = _safe_merge_across_pairs(experiment_save_folder, all_names)
        final_result_path = os.path.join(experiment_save_folder, f"experiment_{self.number}_total_results.xlsx")
        final_df.to_excel(final_result_path, index=False)

        print("=====================================================================")
        print(f"=========================Finish Experiment {self.number} (Bi-dir)=========================")
        print("=====================================================================")

    def experiment_with_llm_single_directional(self, file_folder, llm_model):
        experiment_save_folder = self.save_folder
        os.makedirs(experiment_save_folder, exist_ok=True)
        os.makedirs(experiment_save_folder, exist_ok=True)

        for src, tgt in combinations(all_names, 2):
            pair_name = f"{src}_{tgt}"
            merge_folder = Path(file_folder + f"/{pair_name}/merged_result")
            for p in merge_folder.iterdir():
                d = excel_to_dict(p)
                matches = llm_model.test_for_equivalence(d)
                matching_result_path = experiment_save_folder + f"/{pair_name}/result/result.xlsx"
                dict_to_excel(matches, matching_result_path)

        for src, tgt in combinations(all_names, 2):
            pair_name = f"{src}_{tgt}"
            save_folder = os.path.join(experiment_save_folder, pair_name)
            os.makedirs(save_folder, exist_ok=True)
            eval_folder = os.path.join(save_folder, "evaluation")
            os.makedirs(eval_folder, exist_ok=True)
            answer_path = answer_paths[pair_name]
            result_path = save_folder + f"/result/result.xlsx"
            eva.save_evaluate_result(result_path=result_path, answer_path=answer_path, top_k=1,
                                     output_folder=eval_folder)

        final_result_path = experiment_save_folder + f"/experiment_{self.number}_total_results.xlsx"
        merge_final_results(all_names, experiment_save_folder, final_result_path)
    # def experiment_with_llm_bidirectional(self):
