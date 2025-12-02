from collections import defaultdict
import pandas as pd


class TableReading(object):
    """
    A class read data from tables, include: table name, column name, attribute
    """

    def __init__(self, table_path):
        self.path = table_path
    '''
    def get_table_name(self):


    def get_column_name(self):

    def get_attribute(self):
    '''
    def get_properties(self):
        df = pd.read_excel(self.path, header=None)
        properties = df.iloc[:, 1]
        properties_list = properties.tolist()
        return properties_list

    def get_comments(self):
        df = pd.read_excel(self.path)


if __name__ == '__main__':
    path = r"C:\HiWi_Hanyang\ontology_matching\ontology_matching\Data\matpower_reference_table.xlsx"
    tr = TableReading(path)
    print(tr.get_properties())

