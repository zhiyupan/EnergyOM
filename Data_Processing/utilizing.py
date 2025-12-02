import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from pathlib import Path
from itertools import combinations


def get_file_path(output_dir, file_name, suffix):
    # check if output path exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = file_name.replace(" ", "_")
    output_file_name = file_name + suffix
    output_path = os.path.join(output_dir, output_file_name)
    return output_path


def get_embedding(transformer_model, data_list):
    """
    Generates embedding for a list of data using the transformer model,
    :param transformer_model: The model used for generating embeddings.
    :param data_list: A list of data to embed
    :return: embeddings: Numpy array of embeddings for the inputs data.
    """
    # Ensure data_list is not empty
    if not data_list:
        raise ValueError(f"{data_list} is empty. Provide at least one sentence.")

    # Call the embedding method with appropriate parameter
    embedding = transformer_model.embedding(data_list).numpy()
    return embedding


def get_cosine_similarity(transformer_model, list1, list2):
    """
    Calculate cosine similarity between embeddings of two list of data.
    :param list1: List of names(strings) from data 1.
    :param list2: List of names(strings) from data 2.
    :return: cos_sim: Cosine similarity matrix between list1 and list2.
    """
    list1_embedding = get_embedding(transformer_model, list1)
    list2_embedding = get_embedding(transformer_model, list2)
    cos_sim = cosine_similarity(list1_embedding, list2_embedding)
    return cos_sim


def get_top_k_result(data_list, k, result_folder):
    classified_data = defaultdict(list)
    for item in data_list:
        category, _, value = item
        classified_data[category].append(item)

    top_k_data_dict = {}
    for category, items in classified_data.items():
        sorted_items = sorted(items, key=lambda x: x[2], reverse=True)
        top_k_data_dict[category] = sorted_items[:k]

    top_k_data_list = []
    for category, items in top_k_data_dict.items():
        for item in items:
            top_k_data_list.append(item)

    if k < 10:
        result_file_name_prefix = r"top_0" + str(k)
    else:
        result_file_name_prefix = r"top_" + str(k)

    # save results
    results_df = pd.DataFrame(top_k_data_list, columns=['Source', 'Target', 'Cosine Similarity'])
    output_path = get_file_path(result_folder, result_file_name_prefix, suffix='_results.xlsx')
    results_df.to_excel(output_path, index=False)
    return top_k_data_list, output_path


def csv_processing(data_path, save_path):
    df = pd.read_csv(data_path)
    print(df)
    # df['Entity1'] = df['Entity1'].str.split('#').str[-1]
    # df['Entity2'] = df['Entity2'].str.split('#').str[-1]
    df.to_excel(save_path, index=False)
    print(df)


def merge_forward_backward(forward_path, backward_path, output_path):
    df_f = pd.read_excel(forward_path)
    df_b = pd.read_excel(backward_path)
    df_b_conv = df_b.rename(columns={'Source': 'Target', 'Target': 'Source'})
    df_b_conv = df_b_conv[['Source', 'Target', 'Cosine Similarity']]
    forward_pairs = set(zip(df_f['Source'], df_f['Target']))
    new_rows = df_b_conv[~df_b_conv.apply(lambda row: (row['Source'], row['Target']) in forward_pairs, axis=1)]
    df_merged = pd.concat([df_f, new_rows], ignore_index=True)
    df_merged.to_excel(output_path, index=False)


def merge_class_property(class_result, object_property_result, data_property_result, output_folder):
    def read_if_exists(path):
        if path and isinstance(path, str) and os.path.isfile(path):
            df = pd.read_excel(path)
            cols = ['Source', 'Target', 'Cosine Similarity']
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"{os.path.basename(path)} Missing columns: {missing}")
            df = df[cols].copy()
            df = df.dropna(how='all')
            return df
        else:
            if path:
                print(f"[SKIP] Invalid or non-existent path: {path}")
            return None

    frames = []
    for p in (class_result, object_property_result, data_property_result):
        df = read_if_exists(p)
        if df is not None:
            frames.append(df)

    def non_empty_df(df):
        if df is None or df.empty:
            return False
        return not df.isna().all().all()

    frames = [df for df in frames if non_empty_df(df)]

    if not frames:
        print("[SKIP] All three input result files are missing; skipping merge.")
        return None

    for i, df in enumerate(frames):
        df = df.copy()
        df['Source'] = df['Source'].astype(str)
        df['Target'] = df['Target'].astype(str)
        df['Cosine Similarity'] = pd.to_numeric(df['Cosine Similarity'], errors='coerce')
        frames[i] = df

    df_merged = pd.concat(frames, ignore_index=True)

    os.makedirs(output_folder, exist_ok=True)
    output_path = get_file_path(output_folder, "merged", ".xlsx")
    df_merged.to_excel(output_path, index=False, engine="openpyxl")
    return output_path


def excel_to_dict(path, sheet_name=0, keep="last"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df2 = df.iloc[:, :2].dropna(how="all")
    df2 = df2.dropna(subset=[df2.columns[0], df2.columns[1]])
    df2 = df2.drop_duplicates(subset=df2.columns[0], keep=keep)
    mapping = dict(zip(df2.iloc[:, 0].astype(str), df2.iloc[:, 1]))
    return mapping


def dict_to_excel(mapping: dict, out_path: str, sheet_name: str = "Sheet1"):
    df = pd.DataFrame(list(mapping.items()), columns=["Source", "Target"])
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df.to_excel(out_path, index=False, sheet_name=sheet_name, engine="openpyxl")
    return out_path


def merge_final_results(all_names, results_root: str, output_path: str) -> str | None:
    rows = []
    results_root = Path(results_root)

    for src, tgt in combinations(all_names, 2):
        pair_name = f"{src}_{tgt}"
        eva_folder = results_root / pair_name / "evaluation"
        if not eva_folder.is_dir():
            print(f"[SKIP] Not a directory: {eva_folder}")
            continue

        for p in list(eva_folder.glob("*.xlsx")) + list(eva_folder.glob("*.xls")):
            try:
                df = pd.read_excel(p)
            except Exception as e:
                print(f"[SKIP] Failed to read {p}: {e}")
                continue
            if df.empty:
                continue

            if str(df.columns[0]).lower().startswith("unnamed"):
                df = df.drop(columns=df.columns[0])

            last_row = df.iloc[-1]
            row_dict = {"Matched_Onto": pair_name}
            row_dict.update(last_row.to_dict())
            rows.append(row_dict)

    if not rows:
        print("[SKIP] No evaluation files found.")
        return None

    merged = pd.DataFrame(rows)

    cols = ["Matched_Onto"] + [c for c in merged.columns if c != "Matched_Onto"]
    merged = merged[cols]

    num_cols = [c for c in merged.columns if c != "Matched_Onto"]
    merged[num_cols] = merged[num_cols].apply(pd.to_numeric, errors="coerce")
    avg_row = {"Matched_Onto": "Average"}
    avg_row.update(merged[num_cols].mean(skipna=True).to_dict())
    merged = pd.concat([merged, pd.DataFrame([avg_row])], ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_path, index=False, engine="openpyxl")
    print(f"[OK] Final Results have been Saved: {output_path}")
    return output_path


def exchange_two_columns(input_filepath, output_filepath):
    df = pd.read_excel(input_filepath)
    df.iloc[:, [0, 1]] = df.iloc[:, [1, 0]].values
    df.to_excel(output_filepath, index=False)

def combine_two_files(file_1, file_2, output_path):
    df1 = pd.read_excel(file_1)
    df2 = pd.read_excel(file_2)
    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.drop_duplicates(subset=merged.columns[:2])
    merged.to_excel(output_path, index=False)


if __name__ == '__main__':
    # forward = r"../Data/agent/threshold_0.8_results_bidirectional/n_h/bert/cmt_conference/cmt_conference_top1_matching_results.xlsx/top_01_results.xlsx"
    # backward = r"../Data/agent/threshold_0.8_results_bidirectional/n_h/bert/conference_cmt/conference_cmt_top1_matching_results.xlsx/top_01_results.xlsx"
    # output = r"../Data/agent/threshold_0.8_results_bidirectional/n_h/bert/cmt_conference/cmt_conference_top1_matching_results.xlsx/merge.xlsx"
    # merge_forward_backward(forward, backward, output)

    # class_result = r"../Data/agent/threshold_0.8_results/n_h/bert/cmt_conference/cmt_conference_class_matching_results/cmt_conference_filter_results.xlsx"
    # data_pr = r"../Data/agent/threshold_0.8_results/n_h/bert/cmt_conference/cmt_conference_data_property_matching_results/cmt_conference_filter_results.xlsx"
    # object_pr = r"../Data/agent/threshold_0.8_results/n_h/bert/cmt_conference/cmt_conference_object_property_matching_results/cmt_conference_filter_results.xlsx"
    # output = r"../Data/agent/threshold_0.8_results/n_h/bert/cmt_conference"
    # print(merge_class_property(class_result, object_pr, data_pr, output))
    """
    mapping = {'Review': 'Review', 'Co-author': 'Contribution_co-author', 'Paper': 'Paper', 'Meta-Reviewer': 'Reviewer',
               'Person': 'Person', 'Reviewer': 'Reviewer', 'Chairman': 'Reviewer', 'Author': 'Regular_author',
               'Document': 'Conference_document', 'AssociatedChair': 'Conference_participant',
               'ConferenceMember': 'Conference_participant', 'ExternalReviewer': 'Reviewer', 'Conference': 'Conference',
               'AuthorNotReviewer': 'Contribution_co-author', 'PaperAbstract': 'Paper', 'email': 'has_an_email',
               'name': 'has_a_name'}
    dict_to_excel(mapping, "../Data/test/dict_to_excel.xlsx")
    """
    merge = r"../Data/agent/experiment_2_results/cmt_conference/merged_result/merged.xlsx"
    reverse = r"../Data/agent/experiment_2_results/cmt_conference/merged_result/reverse.xlsx"
    combine = r"../Data/agent/experiment_8_results/cmt_conference/merged_result/combine.xlsx"
    original_merged_result_path = f"../Data/agent/experiment_2_results/cmt_conference/merged_result/merged.xlsx"
    reversed_result = r"../Data/agent/experiment_8_results/edas_sigkdd/merged_result/reversed.xlsx"
    exchange_two_columns(merge, reverse)
    combine_two_files(original_merged_result_path, reversed_result, combine)

