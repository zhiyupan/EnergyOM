import pandas as pd
from Data_Processing.utilizing import get_file_path
import xml.etree.ElementTree as ET


class RDFReading(object):
    """
    A class to read and process RDF files, particularly for extracting alignment pairs from RDF documents
    structured with specific namespace.
    """

    def __init__(self, rdf_file):
        """
        Initializes the RDFReading object by parsing the RDF file.
        :param rdf_file: Path to the RDF file.
        """
        tree = ET.parse(rdf_file)
        self.root = tree.getroot()

    def process_entity(self, entity):

        entity = entity.strip()

        # Extract namespace and class from the URI
        if "rdf:resource" not in entity:
            if '#' in entity:
                prefix, class_name = entity.split('#')
            elif '/' in entity:
                prefix, class_name = entity.rsplit('/', 1)
            else:
                prefix, class_name = None, entity
            return prefix, class_name

        # If it's XML, try to parse it
        try:
            element = ET.fromstring(entity)
        except ET.ParseError as e:
            print(f"Invalid XML format for entity: {entity}, error: {e}")
            return None, None

        # Extract rdf:resource attribute
        uri = element.attrib.get('rdf:resource', None)
        if uri is None:
            raise ValueError(f"Missing rdf:resource in entity: {entity}")

        # Extract namespace and class name
        if '#' in uri:
            prefix, class_name = uri.split('#')
        elif '/' in uri:
            prefix, class_name = uri.rsplit('/', 1)
        else:
            prefix, class_name = None, uri
        return prefix, class_name

    def get_alignment_pairs(self, name1, name2, output_folder):
        """
        Extracts alignment pairs from RDF file and saves them as an Excel file.
        :param name1: Name of ontology 1 and is used as column name for the first entity in the alignment pair.
        :param name2: Name of ontology 2 and is used as column name for the second entity in the alignment pair.
        :param output_folder: Path to  folder where the output file will be saved.
        """
        # Define namespace for the RDF syntax and alignment elements
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            '': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}
        # Find all Cell elements within the alignment namespace in the XML structure.
        cells = self.root.findall('.//{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}Cell')

        alignment_pairs = []  # Initialize an empty list to store alignment pairs.
        for cell in cells:
            # Extract the URI resource for entity1 and entity2 using the defined namespaces.
            entity1 = cell.find('entity1', namespaces).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']
            entity2 = cell.find('entity2', namespaces).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']

            prefix1, class1 = self.process_entity(entity1)
            prefix2, class2 = self.process_entity(entity2)

            prefix1 = prefix1.split('/')[-1]
            prefix2 = prefix2.split('/')[-1]

            print(f"Entity1 Prefix: {prefix1}, Class: {class1}")
            print(f"Entity2 Prefix: {prefix2}, Class: {class2}")

            # Create an alignment tuple with the classes and add to the list
            alignment_pair = (class1, class2)
            alignment_pairs.append(alignment_pair)

        # save results
        answer_df = pd.DataFrame(alignment_pairs, columns=[name1, name2])
        suffix = '_alignment_answer.xlsx'
        output_path = get_file_path(output_folder, name1 + '_' + name2, suffix)
        answer_df.to_excel(output_path, index=False)


if __name__ == '__main__':
    rdf = RDFReading('../energy_domain_experiment/SORBET/emkpi2Sargon_-1.rdf')
    output = '../energy_domain_experiment/SORBET'
    rdf.get_alignment_pairs('emkpi', 'Sargon', output)
