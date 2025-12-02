from Data_Reading.ontology_reading import OntologyReading
from Data_Matching.name_based import NameMatching
from Data_Matching.hierarchy_based import HierarchyMatching
from Data_Matching.comment_based import CommentsMatching
from Data_Matching.structure_based import StructureMatching
from Data_Processing.filter import Filter
from Data_Processing.utilizing import get_top_k_result
from Model.deepseek import DeepSeekMatcher
from Model.sBERT import SentenceTransformers
from Model.bert_Energy_tsdae import BertEnergy
import pandas as pd
import os


class MatchingMethods(object):
    def __init__(self, os, ot, model, save_folder, datatype="object_property"):
        self.os = os
        self.ot = ot
        self.save_folder = save_folder
        self.model = model
        if datatype not in ("object_property", "data_property"):
            raise ValueError("datatype must be 'object_property' or 'data_property'")
        self.datatype = datatype
        self.matching_methods = {
            'name_based': {
                'enabled': False,
                'folder': f"{self.save_folder}/name_based",
                'instance': NameMatching(self.model),
                'match_func': self.match_name_based,
                'result_path': None
            },
            'hierarchy_based': {
                'enabled': False,
                'folder': f"{self.save_folder}/hierarchy_based",
                'instance': HierarchyMatching(self.model),
                'match_func': self.match_hierarchy_based,
                'result_path': None
            },
            'comment_based': {
                'enabled': False,
                'folder': f"{self.save_folder}/comment_based",
                'instance': CommentsMatching(self.model),
                'match_func': self.match_comment_based,
                'result_path': None
            },
            'structure_based': {
                'enabled': False,
                'folder': f"{self.save_folder}/structure_based",
                'instance': StructureMatching(self.model),
                'match_func': self.match_structure_based,
                'result_path': None
            },
            'property_name_based': {
                'enabled': False,
                'folder': f"{self.save_folder}/property_name_based",
                'instance': NameMatching(self.model),
                'match_func': self.match_property_name_based,
                'result_path': None
            }
        }

    def match_name_based(self, threshold: float = None, datatype: int = None):

        list1 = self.os.get_classes()
        list2 = self.ot.get_classes()
        nm = self.matching_methods["name_based"]["instance"]
        matched_name_list = nm.get_name_matching_results(list1, list2, threshold)
        return [tuple(item) for item in matched_name_list]

    def match_property_name_based(self, threshold: float = None):
        list1 = []
        list2 = []
        if self.datatype == "data_property":
            list1 = self.os.get_data_properties()
            list2 = self.ot.get_data_properties()
        elif self.datatype == "object_property":
            list1 = self.os.get_object_properties()
            list2 = self.ot.get_object_properties()
        nm = self.matching_methods["property_name_based"]["instance"]
        matched_name_list = nm.get_name_matching_results(list1, list2, threshold)
        return [tuple(item) for item in matched_name_list]

    def match_hierarchy_based(self, threshold: float = None):
        hm = self.matching_methods['hierarchy_based']['instance']
        matched_hierarchy_list = hm.get_hierarchy_matching_classes_pairs(self.os, self.ot, threshold)
        return matched_hierarchy_list

    def match_comment_based(self, threshold: float = None):
        os_comments = self.os.get_class_comment_triples('OS')
        ot_comments = self.ot.get_class_comment_triples('OT')
        cm = self.matching_methods['comment_based']['instance']
        matched_comment_list = cm.get_comment_matching_pairs_from_triples(os_comments, ot_comments, threshold)
        return matched_comment_list

    def match_structure_based(self, threshold: float = None):
        os_prop_structure = self.os.get_property_structure_triple(self.datatype)
        ot_prop_structure = self.ot.get_property_structure_triple(self.datatype)
        sm = self.matching_methods['structure_based']['instance']
        matched_structure_list = sm.structure_matching_pairs(self.datatype, os_prop_structure, ot_prop_structure,
                                                             threshold,
                                                             weight=(0.3, 0.7, 0))
        return matched_structure_list

    def execute_enabled_methods(self, threshold: float = None):
        results = {}
        for method, params in self.matching_methods.items():
            if params['enabled']:
                print(f"Running {method} matching...")
                matched_list = params['match_func'](threshold)
                folder = params['folder']
                os.makedirs(folder, exist_ok=True)
                result_path = f"{folder}/{method}_matched_results.xlsx"

                if isinstance(self.model, BertEnergy) or isinstance(self.model, SentenceTransformers):
                    df = pd.DataFrame(matched_list, columns=['Source', 'Target', 'Score'])
                    df.to_excel(result_path, index=False)
                elif isinstance(self.model, DeepSeekMatcher):
                    df = pd.DataFrame(matched_list, columns=['Source', 'Target'])
                    df.to_excel(result_path, index=False)
                params['result_path'] = result_path
                results[method] = result_path
        return results


class ClassMatching(object):
    def __init__(self, os_path, ot_path, model, save_folder):
        self.os = OntologyReading(os_path)
        self.ot = OntologyReading(ot_path)
        self.save_folder = save_folder
        self.model = model

    def perform_class_matching(self, name_based=False, hierarchy_based=False, comment_based=False,
                               threshold: float = None):
        """
        Perform class matching based on enabled methods
        """
        methods = MatchingMethods(self.os, self.ot, self.model, self.save_folder)
        methods.matching_methods['name_based']['enabled'] = name_based
        methods.matching_methods['hierarchy_based']['enabled'] = hierarchy_based
        methods.matching_methods['comment_based']['enabled'] = comment_based

        results = methods.execute_enabled_methods(threshold)

        if not results:
            raise ValueError("At least one matching method (name, hierarchy, or comment) must be enabled.")

        print("Matching completed. Results saved at:")
        for method, path in results.items():
            print(f"{method}: {path}")

        return results

    def class_matching_results(self, name_based=False, hierarchy_based=False, comment_based=False,
                               threshold: float = None, top_k: int = 1):
        data = {}
        paths = self.perform_class_matching(name_based, hierarchy_based, comment_based, threshold)
        for key, path in paths.items():
            df = pd.read_excel(path)
            data[key] = df.values.tolist()

        name_list = None
        hierarchy_list = None
        comment_list = None

        if 'name_based' in data:
            name_based_list = data['name_based']
            name_list = [tuple(item) for item in name_based_list]
        if 'hierarchy_based' in data:
            hierarchy_based_list = data['hierarchy_based']
            hierarchy_list = [tuple(item) for item in hierarchy_based_list]
        if 'comment_based' in data:
            comment_based_list = data['comment_based']
            comment_list = [tuple(item) for item in comment_based_list]
        output_folder = f"{self.save_folder}/{self.os.name}_{self.ot.name}_class_matching_results"
        data_filter = Filter()
        matched_list = data_filter.get_filter_result(name_list, hierarchy_list, comment_list)
        data_filter.save_filter_results(matched_list, self.os.name, self.ot.name, output_folder)

        top_k_output_folder = f"{self.save_folder}/{self.os.name}_{self.ot.name}_top{top_k}_class_matching_results"
        top_k_class_matching_results, output_path = get_top_k_result(matched_list, top_k, top_k_output_folder)

        return output_path


class PropertyMatching(object):
    def __init__(self, os_path, ot_path, model, save_folder, datatype):
        self.os = OntologyReading(os_path)
        self.ot = OntologyReading(ot_path)
        self.save_folder = save_folder
        self.model = model
        self.datatype = datatype

    def perform_property_matching(self, name_based=False, structure_based=False, comment_based=False,
                                  threshold: float = None):
        methods = MatchingMethods(self.os, self.ot, self.model, self.save_folder, self.datatype)
        methods.matching_methods["property_name_based"].update(enabled=name_based, datatype=self.datatype)
        methods.matching_methods['structure_based']['enabled'] = structure_based
        methods.matching_methods['comment_based']['enabled'] = comment_based

        results = methods.execute_enabled_methods(threshold)

        if not results:
            raise ValueError("At least one matching method (name, structure, or comment) must be enabled.")

        print("Matching completed. Results saved at:")
        for method, path in results.items():
            print(f"{method}: {path}")

        return results

    def property_matching_results(self, name_based=False, structure_based=False, comment_based=False,
                                  threshold: float = None, top_k: int = 1):
        if self.datatype == "data_property":
            os_list = self.os.get_data_properties()
            ot_list = self.ot.get_data_properties()
        elif self.datatype == "object_property":
            os_list = self.os.get_object_properties()
            ot_list = self.ot.get_object_properties()
        else:
            raise ValueError(f"Unsupported datatype: {self.datatype}")
        if not os_list or not ot_list:
            print(f"[SKIP] {self.os.name} ↔ {self.ot.name} of {self.datatype} is empty, skip property matching")
            return None
        property_data = {}
        paths = self.perform_property_matching(name_based, structure_based, comment_based, threshold)
        for key, path in paths.items():
            df = pd.read_excel(path)
            property_data[key] = df.values.tolist()

        name_list = None
        structure_list = None
        comment_list = None

        if 'name_based' in property_data:
            name_based_list = property_data['property_name_based']
            name_list = [tuple(item) for item in name_based_list]
        if 'structure_based' in property_data:
            structure_based_list = property_data['structure_based']
            structure_list = [tuple(item) for item in structure_based_list]
        if 'comment_based' in property_data:
            comment_based_list = property_data['comment_based']
            comment_list = [tuple(item) for item in comment_based_list]
        output_folder = f"{self.save_folder}/{self.os.name}_{self.ot.name}_{self.datatype}_matching_results"
        data_filter = Filter()
        matched_list = data_filter.get_filter_result(name_list, structure_list, comment_list)
        data_filter.save_filter_results(matched_list, self.os.name, self.ot.name, output_folder)

        top_k_output_folder = f"{self.save_folder}/{self.os.name}_{self.ot.name}_top{top_k}_{self.datatype}_matching_results"
        top_k_class_matching_results, output_path = get_top_k_result(matched_list, top_k, top_k_output_folder)

        return output_path

