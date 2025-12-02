from Model.bert_Energy_tsdae import BertEnergy
from Model.sBERT import SentenceTransformers
from Data_Reading.ontology_reading import OntologyReading
from Data_Processing.utilizing import get_file_path, get_cosine_similarity
from collections import defaultdict
import pandas as pd
from Model.deepseek import DeepSeekMatcher
from Model.gemini import GeminiMatcher


class HierarchyMatching(object):
    def __init__(self, transformer_model=None):
        if transformer_model is None:
            transformer_model = SentenceTransformers()
        if transformer_model is DeepSeekMatcher:
            transformer_model = DeepSeekMatcher()
        if transformer_model is GeminiMatcher:
            transformer_model = GeminiMatcher()
        self.model = transformer_model

    def triple_to_sentence(self, triples):
        """
        Convert a triple into a list of sentences.
        :param triple: rdf triple
        :return: A list of sentence
        """
        return [f"Class {child} is a subclass of {parent}" for (child, rel, parent) in triples]

    def hierarchy_matching_pairs(self, onto1, onto2, threshold):
        onto1_name = onto1.name
        onto2_name = onto2.name
        hierarchy1 = onto1.get_onto_hierarchy_triples(onto1_name)
        hierarchy2 = onto2.get_onto_hierarchy_triples(onto2_name)

        hierarchy1_list = self.triple_to_sentence(hierarchy1)
        hierarchy2_list = self.triple_to_sentence(hierarchy2)

        if isinstance(self.model, BertEnergy) or isinstance(self.model, SentenceTransformers):
            cos_sim = get_cosine_similarity(self.model, hierarchy1_list, hierarchy2_list)
            matches = []
            for idx1, row in enumerate(cos_sim):
                for idx2, sim in enumerate(row):
                    try:
                        sim_value = float(sim)
                    except ValueError:
                        print(f"Skipping invalid similarity value: {sim}")
                        continue
                    if sim_value > threshold:
                        match = (hierarchy1_list[idx1], hierarchy2_list[idx2], sim)
                        matches.append(match)
            return matches
        elif isinstance(self.model, DeepSeekMatcher):
            matches = self.model.get_hierarchy_matched_pair(hierarchy1_list, hierarchy2_list)
            return matches
        elif isinstance(self.model, GeminiMatcher):
            matches = self.model.get_hierarchy_matched_pair(hierarchy1_list, hierarchy2_list)
            return matches
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def get_hierarchy_matching_classes_pairs(self, onto1, onto2, threshold: float = None):
        if threshold is None:
            threshold = 0.7
        matches = self.hierarchy_matching_pairs(onto1, onto2, threshold)
        class_pairs = set()
        if isinstance(self.model, BertEnergy) or isinstance(self.model, SentenceTransformers):
            for item in matches:
                if 'is a subclass of' not in item[0] or 'is a subclass of' not in item[1]:
                    continue

                # 分别提取subclass和parent
                subclass_1, parent_class_1 = item[0].split(' is a subclass of ')
                subclass_1 = subclass_1.replace('Class ', '', 1).strip()
                parent_class_1 = parent_class_1.replace('Class ', '', 1).strip()

                subclass_2, parent_class_2 = item[1].split(' is a subclass of ')
                subclass_2 = subclass_2.replace('Class ', '', 1).strip()
                parent_class_2 = parent_class_2.replace('Class ', '', 1).strip()
                similarity = item[2]

                # 组合四种匹配
                class_pairs.add((parent_class_1, parent_class_2, similarity))  # 父-父
                class_pairs.add((subclass_1, subclass_2, similarity))  # 子-子
                # class_pairs.add((subclass_1, parent_class_2, similarity))  # 子-父
            # class_pairs.add((parent_class_1, subclass_2, similarity))  # 父-子

            # 保留最大similarity的pair
            class_pair_list = list(class_pairs)
            class_pairs_dict = {}
            for item in class_pair_list:
                key = (item[0], item[1])
                value = item[2]
                if key not in class_pairs_dict or value > class_pairs_dict[key][2]:
                    class_pairs_dict[key] = item

            class_pairs = list(class_pairs_dict.values())
            print("Hierarchy based matching is finished!")
            return class_pairs
        elif isinstance(self.model, DeepSeekMatcher):
            for item in matches:
                if 'is a subclass of' not in item[0] or 'is a subclass of' not in item[1]:
                    continue

                # 分别提取subclass和parent
                subclass_1, parent_class_1 = item[0].split(' is a subclass of ')
                subclass_1 = subclass_1.replace('Class ', '', 1).strip()
                parent_class_1 = parent_class_1.replace('Class ', '', 1).strip()

                subclass_2, parent_class_2 = item[1].split(' is a subclass of ')
                subclass_2 = subclass_2.replace('Class ', '', 1).strip()
                parent_class_2 = parent_class_2.replace('Class ', '', 1).strip()

                class_pairs.add((parent_class_1, parent_class_2))  # 父-父
                class_pairs.add((subclass_1, subclass_2))  # 子-子

            print("Hierarchy based matching is finished!")
            return class_pairs
        elif isinstance(self.model, GeminiMatcher):
            for item in matches:
                if 'is a subclass of' not in item[0] or 'is a subclass of' not in item[1]:
                    continue

                # 分别提取subclass和parent
                subclass_1, parent_class_1 = item[0].split(' is a subclass of ')
                subclass_1 = subclass_1.replace('Class ', '', 1).strip()
                parent_class_1 = parent_class_1.replace('Class ', '', 1).strip()

                subclass_2, parent_class_2 = item[1].split(' is a subclass of ')
                subclass_2 = subclass_2.replace('Class ', '', 1).strip()
                parent_class_2 = parent_class_2.replace('Class ', '', 1).strip()

                class_pairs.add((parent_class_1, parent_class_2))  # 父-父
                class_pairs.add((subclass_1, subclass_2))  # 子-子

            print("Hierarchy based matching is finished!")
            return class_pairs

    def save_results(self, class_pairs, onto1_name, onto2_name, result_folder):
        # Save results
        match_df = pd.DataFrame(class_pairs, columns=[onto1_name, onto2_name, 'Cosine Similarity'])
        suffix = '_hierarchy_based_results.xlsx'
        output_path = get_file_path(result_folder, onto1_name + '_' + onto2_name, suffix)
        match_df.to_excel(output_path, index=False)
        return output_path


if __name__ == '__main__':
    onto1_path = r'../Data/agent/ontology/cmt.xml'
    onto2_path = r'../Data/agent/ontology/conference.xml'
    onto1 = OntologyReading(onto1_path)
    onto2 = OntologyReading(onto2_path)
    result_folder = r'../Data/agent/results/test/hierarchy'
    hc = HierarchyMatching(GeminiMatcher)
    hierarchy_pairs = hc.get_hierarchy_matching_classes_pairs(onto1, onto2, 0.7)
    print(hierarchy_pairs)
    # hc.save_results(hierarchy_pairs, "cmt", "conference", result_folder)
