from Data_Reading.ontology_reading import OntologyReading
from Data_Processing.utilizing import get_file_path, get_cosine_similarity
from Model.bert_Energy_tsdae import BertEnergy
from Model.sBERT import SentenceTransformers
from Model.deepseek import DeepSeekMatcher
from Model.gemini import GeminiMatcher

import pandas as pd


class NameMatching(object):
    """
    A class used for Table2Table, Table2Onto, Onto2Onto name matching
    """

    def __init__(self, transformer_model=None):
        """
        Initialize the NameMatching class with a transformer model
        :param transformer_model: The model used for generating embeddings.
        """
        if transformer_model is None:
            transformer_model = SentenceTransformers()
        self.model = transformer_model

    def get_name_matching_pairs(self, names1_list, names2_list, threshold):
        """
        Matches names from two lists based on cosine similarity above a given threshold.
        :param threshold: The similarity thresholds to consider a match.
        :param names1_list: List of names(strings).
        :param names2_list: List of names(strings).

        :return: None. Prints matched pairs and saves them to an Excel file.
        """
        cos_sim = get_cosine_similarity(self.model, names1_list, names2_list)
        matches = []
        for idx1, row in enumerate(cos_sim):
            for idx2, sim in enumerate(row):
                if sim > threshold:
                    match = (names1_list[idx1], names2_list[idx2], sim)
                    matches.append(match)
        print("Name based matching is finished!")
        return matches

    def get_name_matching_results(self, names1_list, names2_list, threshold: float = None):
        if isinstance(self.model, BertEnergy):
            if threshold is None:
                threshold = 0.8  # Default threshold for cosine similarity
            return self.get_name_matching_pairs(names1_list, names2_list, threshold)
        elif isinstance(self.model, SentenceTransformers):
            if threshold is None:
                threshold = 0.8
            return self.get_name_matching_pairs(names1_list, names2_list, threshold)
        elif isinstance(self.model, DeepSeekMatcher):
            matches = self.model.get_name_matched_pair(names1_list, names2_list)
            return matches
        elif isinstance(self.model, GeminiMatcher):
            matches = self.model.get_name_matched_pair(names1_list,names2_list)
            return matches
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def save_result(self, matches, column1, column2, results_folder):
        """
        :param column1: Name of the first columns for results
        :param column2: Name of the second columns for results
        :param results_folder: Directory path to save the output file
        """
        # save results
        match_df = pd.DataFrame(matches, columns=[column1, column2, 'Cosine Similarity'])
        suffix = '_name_based_results.xlsx'
        output_path = get_file_path(results_folder, column1 + '_' + column2, suffix)
        match_df.to_excel(output_path, index=False)
        return output_path


if __name__ == '__main__':
    onto1_path = r'../Data/agent/ontology/cmt.xml'
    onto2_path = r'../Data/agent/ontology/conference.xml'
    onto1 = OntologyReading(onto1_path)
    onto2 = OntologyReading(onto2_path)
    onto1_classes = onto1.get_classes()
    onto2_classes = onto2.get_classes()
    nm = NameMatching(DeepSeekMatcher())
    nm_pairs = nm.get_name_matching_results(onto1_classes, onto2_classes)
    print(nm_pairs)
