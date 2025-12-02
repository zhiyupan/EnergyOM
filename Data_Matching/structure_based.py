from Model.bert_Energy_tsdae import BertEnergy
from sklearn.metrics.pairwise import cosine_similarity
from Data_Processing.utilizing import get_file_path
import pandas as pd


class StructureMatching(object):
    def __init__(self, transformer_model=BertEnergy()):
        self.model = transformer_model

    def encode_structure_separate(self, data_type, structure_list):
        encode_structures = []
        if data_type == "data_property":
            for domain, property_, range_ in structure_list:
                domain_emb = self.model.embedding([f"Domain: {domain}"])[0]
                property_emb = self.model.embedding([f"Property: {property_}"])[0]
                if range_ == "int" or range_ is None:
                    range_ = "float"
                encode_structures.append((domain_emb, property_emb, range_))

        elif data_type == "object_property":
            for domain, property_, range_ in structure_list:
                domain_emb = self.model.embedding([f"Domain: {domain}"])[0]
                property_emb = self.model.embedding([f"Property: {property_}"])[0]
                range_emb = self.model.embedding([f"Range: {range_}"])[0]
                encode_structures.append((domain_emb, property_emb, range_emb))
        return encode_structures

    def encode_structure_overall(self, data_type, structure_list):
        encode_structures = []
        if data_type == "data_property":
            for structure in structure_list:
                if structure[-1] == "int":
                    structure = list(structure)
                    structure[-1] = "float" if structure[-1] == "int" else structure[-1]
                    structure = tuple(structure)
                structure_emb = self.model.embedding(structure)[0]
                encode_structures.append(structure_emb)
        elif data_type == "object_property":
            structure_emb = self.model.embedding(structure_list)[0]
            encode_structures.append(structure_emb)
        return encode_structures

    @staticmethod
    def calculate_weighted_similarity(data_type, emb1, emb2, weight=None):
        overall_sim = 0
        if weight is None:
            overall_sim = cosine_similarity([emb1], [emb2])[0][0]
            return overall_sim
        if data_type == "data_property":
            weights = weight
            domain_sim = cosine_similarity([emb1[0]], [emb2[0]])[0][0]
            property_sim = cosine_similarity([emb1[1]], [emb2[1]])[0][0]
            overall_sim = weights[0] * domain_sim + weights[1] * property_sim
        elif data_type == "object_property":
            weights = weight
            domain_sim = cosine_similarity([emb1[0]], [emb2[0]])[0][0]
            property_sim = cosine_similarity([emb1[1]], [emb2[1]])[0][0]
            range_sim = cosine_similarity([emb1[2]], [emb2[2]])[0][0]
            overall_sim = weights[0] * domain_sim + weights[1] * property_sim + weights[2] * range_sim
        return overall_sim

    def structure_matching_pairs(self, data_type, structure1_list, structure2_list, threshold, weight=None):
        if weight is None:
            structure1_emb = self.encode_structure_overall(data_type, structure1_list)
            structure2_emb = self.encode_structure_overall(data_type, structure2_list)
        else:
            structure1_emb = self.encode_structure_separate(data_type, structure1_list)
            structure2_emb = self.encode_structure_separate(data_type, structure2_list)
        matches = []
        if data_type == "data_property":
            for idx1, emb1 in enumerate(structure1_emb):
                for idx2, emb2 in enumerate(structure2_emb):
                    # Check if ranges match before calculating similarity
                    if emb1[2] != emb2[2]:  # Compare range types
                        continue
                    sim = self.calculate_weighted_similarity(data_type, emb1, emb2, weight)
                    if sim > threshold:
                            matches.append((structure1_list[idx1], structure2_list[idx2], sim))
        elif data_type == "object_property":
            for idx1, emb1 in enumerate(structure1_emb):
                for idx2, emb2 in enumerate(structure2_emb):
                    sim = self.calculate_weighted_similarity(data_type, emb1, emb2, weight)
                    if sim > threshold:
                        matches.append((structure1_list[idx1], structure2_list[idx2], sim))
        results = [
            (item[0][1], item[1][1], item[2])
            for item in matches
        ]
        return results

    def save_structure_matching_result(self, data, data_type, name1, name2, result_folder):
        extracted_data = [(pair[0][1], pair[1][1], pair[2]) for pair in data]
        df = pd.DataFrame(extracted_data, columns=[name1 + '_' + data_type, name2 + '_' + data_type, "Similarity"])
        suffix = "_structure_based_result.xlsx"
        output_path = get_file_path(result_folder, name1 + '_' + name2 + '_' + data_type, suffix)
        df.to_excel(output_path, index=False)


if __name__ == '__main__':
    data_list1 = [('AuxiliaryCost', 'AuxiliaryCost.intervalStartTime', 'datetime.datetime'), (
        'ExcIEEEDC2A', 'ExcIEEEDC2A.seefd1', 'float'), (
                      'RegisteredGenerator', 'RegisteredGenerator.maxWeeklyStarts', 'int'), (
                      'ResponseMethod', 'ResponseMethod.activePowerUOM', 'str'), (
                      'ConsumptionTariffInterval', 'ConsumptionTariffInterval.startValue', 'float')]
    data_list2 = [('TransformerEndInfo', 'TransformerEndInfo.phaseAngleClock', 'int'),
                  ('OperationalLimitType', 'OperationalLimitType.isInfiniteDuration', 'bool'),
                  ('EndDeviceEventType', 'EndDeviceEventType.subDomain', 'str'),
                  ('DotInstruction', 'DotInstruction.nonSpinReserve', 'float'), ('SetPoint', 'SetPoint.value', 'float')]

    object_list1 = [('GovCT1', 'GovCT1.tb', 'Seconds'), ('UnderexcLimIEEE2', 'UnderexcLimIEEE2.tuv', 'Seconds'),
                    ('WindTurbineType3IEC', 'WindTurbineType3IEC.WindMechIEC', 'WindMechIEC'),
                    ('TopologicalNode', 'TopologicalNode.Terminal', 'Terminal'), ('Pss1', 'Pss1.vsmn', 'PU'),
                    ('RegisteredResource', 'RegisteredResource.OrgResOwnership', 'OrgResOwnership')]
    object_list2 = [('OverexcLimIEEE', 'OverexcLimIEEE.ifdmax', 'PU'), ('PssIEEE2B', 'PssIEEE2B.t10', 'Seconds'),
                    ('RUCAwardInstruction', 'RUCAwardInstruction.ClearingResourceAward', 'ResourceAwardClearing'),
                    ('GovCT1', 'GovCT1.tc', 'Seconds'),
                    ('TapChangerControl', 'TapChangerControl.reverseLineDropR', 'Resistance'),
                    ('PssSH', 'PssSH.t2', 'Seconds'), ('AuxiliaryAccount', 'AuxiliaryAccount.principleAmount', 'Money'),
                    ('ChargeType', 'ChargeType.MajorChargeGroup', 'MajorChargeGroup')]

    sm = StructureMatching()

    matches = sm.structure_matching_pairs("object_property", object_list1, object_list2, 0.5,  weight=(0.3, 0.6,0.1))
    print(matches)
    #print(sm.structure_matching_pairs("data_property", data_list1, data_list2, 0.2))
    print(sm.structure_matching_pairs("data_property", data_list1, data_list2, 0.5, weight=(0.3, 0.6,0.1)))
