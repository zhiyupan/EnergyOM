import pandas as pd
import numpy as np
from Model.bert_Energy_tsdae import BertEnergy
from Model.sBERT import SentenceTransformers
from Model.deepseek import DeepSeekMatcher
from Data_Processing.utilizing import get_cosine_similarity, get_file_path
from Data_Reading.ontology_reading import OntologyReading


class CommentsMatching(object):
    def __init__(self, transformer_model=None):
        """
        Initialize the CommentsMatching class with an ontology_path and a transformer_model for comment matching.
        :param transformer_model: The transformer model used for embedding generation, used to process text comments
                                  for matching. Defaults to an instance of the SentenceTransformer model.
        """
        if transformer_model is None:
            transformer_model = SentenceTransformers()
        self.model = transformer_model

    def triple_to_sentence(self, triple):
        """
        Convert a triple into a list of sentences.
        :param triple: rdf triple
        :return: A list of sentence
        """
        sentence_list = [f"{item[0]}: {item[2]}" for item in triple]
        return sentence_list

    def comment_matching_triple_pairs(self, comment1_triple, comment2_triple, threshold=None,match_type="comment"):
        comment1_list = self.triple_to_sentence(comment1_triple)
        comment2_list = self.triple_to_sentence(comment2_triple)

        if isinstance(self.model, BertEnergy) or isinstance(self.model, SentenceTransformers):
            cos_sim = get_cosine_similarity(self.model, comment1_list, comment2_list)
            matches = []
            for idx1, row in enumerate(cos_sim):
                for idx2, sim in enumerate(row):
                    try:
                        sim_value = float(sim)
                    except ValueError:
                        print(f"Skipping invalid similarity value: {sim}")
                        continue
                    if sim_value > threshold:
                        match = (comment1_list[idx1], comment2_list[idx2], sim)
                        matches.append(match)
            return matches
        elif isinstance(self.model, DeepSeekMatcher):
            matches = self.model.get_matched_pairs(comment1_list, comment2_list, match_type)
            return matches
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def get_comment_matching_pairs_from_triples(self, comment1_list, comment2_list, threshold=None):
        """
        use this function
        """
        matches = self.comment_matching_triple_pairs(comment1_list, comment2_list, threshold)
        pairs = set()
        for item in matches:
            if ': ' not in item[0] or ': ' not in item[1]:
                continue
            property1 = item[0].split(': ')[0]
            property2 = item[1].split(': ')[0]

            similarity = item[2]
            pairs.add((property1, property2, similarity))
        pairs_list = list(pairs)
        pairs_dict = {}
        for item in pairs_list:
            key = (item[0], item[1])
            value = item[2]
            if key not in pairs_dict or value > pairs_dict[key][2]:
                pairs_dict[key] = item

        pairs = list(pairs_dict.values())
        return pairs

    def save_result(self, matches, name1, name2, result_folder):
        # save the results
        match_df = pd.DataFrame(matches, columns=[name1, name2, 'Cosine Similarity'])
        suffix = '_comment_based_results.xlsx'
        output_path = get_file_path(result_folder, name1 + '_' + name2, suffix)
        match_df.to_excel(output_path, index=False)
        return output_path


if __name__ == '__main__':
    cm = CommentsMatching()

    ta5 = OntologyReading(r'..\Data\agent\data\conference\cmt-conference\component\target.xml')
    oeo = OntologyReading(r'..\Data\agent\data\conference\cmt-conference\component\source.xml')
    ta5_comments = ta5.get_class_comment_triples("cmt")
    oeo_comments = oeo.get_class_comment_triples("conference")

    save_folder = r'../Data/agent/results/conference/bert/comment'
    # pairs = cm.get_comment_matching(ta5_comments, oeo_comments, 0.2)
    pairs = cm.get_comment_matching_pairs_from_triples(ta5_comments, oeo_comments, 0.2)
    cm.save_result(pairs, "cmt", "conference", save_folder)
