'''
This file preprocesses EcoCyc and MetaCyc to an appropriate
formats to be available as inputs to designated machine
learning models.
'''

import os
import os.path
import sys
from collections import OrderedDict

import numpy as np
from scipy.sparse import lil_matrix

from feature_builder.feature import Feature
from preprocess.compound import Compound
from preprocess.enzyme import Enzyme
from preprocess.gene import Gene
from preprocess.pathway import Pathway
from preprocess.protein import Protein
from preprocess.reaction import Reaction


class BioCyc(object):
    # INITIALIZATION ------------------------------------------------------------------------

    def __init__(self):
        self.list_kb_paths = list()
        self.processed_kb = OrderedDict()

        # List of ids
        self.protein_id = OrderedDict()
        self.gene_id = OrderedDict()
        self.enzyme_id = OrderedDict()
        self.compound_id = OrderedDict()
        self.reaction_id = OrderedDict()
        self.pathway_id = OrderedDict()
        self.ec_id = OrderedDict()
        self.gene_name_id = OrderedDict()
        self.go_id = OrderedDict()
        self.product_id = OrderedDict()

    def biocyc_data(self):
        items = self.__dict__.copy()
        return items

    # BUILD DATA FROM DATABASES -------------------------------------------------------------

    def extract_info_from_database(self, db_path, data_folder, constraint_on_kb='metacyc'):
        """ Build data from a given knowledge-base path
        :type db_path: str
        :param db_path: The RunPathoLogic knowledge base path, where all the data folders
                            are located
        :type data_folder: str
        :param data_folder: The data folder under a particular knowledge base
                           that includes the files for data preprocessing
        """

        self.list_kb_paths = sorted([os.path.join(db_path, folder, data_folder) for folder in os.listdir(db_path)
                                     if not folder.startswith('.')])

        print('\t>> Building from {0} PGDBs...'.format(len(self.list_kb_paths)))
        for (index, db_path) in enumerate(self.list_kb_paths):
            if os.path.isdir(db_path) or os.path.exists(db_path):
                core_kbname = str(db_path.split(os.sep)[-2]).upper()
                self.list_kb_paths[index] = core_kbname.lower()
                print('\t\t{0:d})- {1:s} (progress: {3:.2f}%, {0:d} out of {2:d}):'
                      .format(index + 1, core_kbname, len(self.list_kb_paths),
                              (index + 1) * 100.00 / len(self.list_kb_paths)))

                # Objects
                gene = Gene()
                protein = Protein()
                enzyme = Enzyme()
                reaction = Reaction()
                pathway = Pathway()
                compound = Compound()

                # List of objects to be passed in processing
                list_ids = [self.protein_id, self.go_id, self.gene_id, self.gene_name_id,
                            self.product_id, self.enzyme_id, self.reaction_id, self.ec_id,
                            self.pathway_id, self.compound_id]

                # Process proteins
                protein.process_proteins(protein_idx_list_ids=0, go_idx_list_ids=1, list_ids=list_ids,
                                         data_path=db_path)

                # Process compounds
                compound.process_compounds(compound_idx_list_ids=9, list_ids=list_ids, data_path=db_path)

                # Process genes
                gene.process_genes(gene_idx_list_ids=2, gene_name_idx_list_ids=3, product_idx_list_ids=4,
                                   list_ids=list_ids, data_path=db_path)

                # Process enzymes
                enzyme.process_enzymes(enzyme_position_idx=5, list_ids=list_ids, data_path=db_path, header=False)

                # Process reactions
                reaction.process_reactions(reaction_idx_list_ids=6, list_ids=list_ids, data_path=db_path)

                # Process pathways
                pathway.process_pathways(pathway_idx_list_ids=8, list_ids=list_ids, data_path=db_path, header=False)

                # Add various information
                gene.add_protein_info(protein_info=protein.protein_info, go_position_idx=4, catalyzes_position_idx=2,
                                      product_name_position_idx=1, product_idx_list_ids=4, list_ids=list_ids)
                gene.add_pathway_info(pathway_info=pathway.pathway_info, gene_name_id_position_idx=12,
                                      gene_id_position_idx=13,
                                      gene_idx_list_ids=2, gene_name_idx_list_ids=3, list_ids=list_ids)
                reaction.add_ec_info(ec_idx_list_ids=7, list_ids=list_ids, data_path=db_path)
                pathway.add_reaction_info(reactions_info=reaction.reaction_info, ec_position_idx=3,
                                          in_pathway_position_idx=4,
                                          orphan_position_idx=5, spontaneous_position_idx=12)
                reaction.add_gene_info(genes_info=gene.gene_info, gene_name_id_position_idx=1, reaction_position_idx=5)
                gene.add_reaction_info(reaction_info=reaction.reaction_info, gene_name_id_position_idx=13,
                                       gene_name_id=3, list_ids=list_ids)
                datum = {core_kbname.lower(): (protein.protein_info, compound.compound_info, gene.gene_info,
                                               enzyme.enzyme_info, reaction.reaction_info, pathway.pathway_info)}
                self.processed_kb.update(datum)

            else:
                print('\t\t## Failed preprocessing {0} database...'.format(db_path.split('/')[-2]),
                      file=sys.stderr)

        if constraint_on_kb:
            print('\t>> Distilling information based on {0}...'.format(constraint_on_kb.upper()))
            data = self.processed_kb[constraint_on_kb]

            data_info = data[0]
            item_list = [item for (item, idx) in self.protein_id.items() if item in data_info]
            self.protein_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            data_info = data[1]
            item_list = [item for (item, idx) in self.compound_id.items() if item in data_info]
            self.compound_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            data_info = data[2]
            item_list = [item for (item, idx) in self.gene_id.items() if item in data_info]
            self.gene_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            item_list = [items[1][1][1] for items in data_info.items() if items[1][1][1]]
            item_list = np.unique(item_list)
            self.gene_name_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            item_list = [items[1][3][1] for items in data_info.items() if items[1][3][1] != '']
            item_list = np.unique(item_list)
            self.product_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            item_list = [i for items in data_info.items() if items[1][6][1] for i in items[1][6][1]]
            item_list = np.unique(item_list)
            self.go_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            data_info = data[3]
            item_list = [item for (item, idx) in self.enzyme_id.items() if item in data_info]
            self.enzyme_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            data_info = data[4]
            item_list = [item for (item, idx) in self.reaction_id.items() if item in data_info]
            self.reaction_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            item_list = [i for items in data_info.items() if items[1][3][1] for i in items[1][3][1]]
            item_list = np.unique(item_list)
            self.ec_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

            data_info = data[5]
            item_list = [item for (item, idx) in self.pathway_id.items() if item in data_info]
            self.pathway_id = OrderedDict(zip(item_list, list(range(len(item_list)))))

    # -----------------------------------------------------


# EXTRACT FEATURES FROM DATA ------------------------------------------------------------

def extract_pathway_properties(biocyc_data, ptwy_ec_matrix, ec_idx, num_features, rxn_position=4, ptwy_position=5,
                               kb='metacyc', display_interval=-1):
    '''

    :param display_interval:
    :param ptwy_ec_matrix:
    :param ptwy_position:
    :param kb:
    :return:
    '''
    print('\t  >> Extracting a set of properties of each pathway from: {0}'.format(kb.upper()))
    feat_obj = Feature(protein_id=biocyc_data['protein_id'], product_id=biocyc_data['product_id'],
                       gene_id=biocyc_data['gene_id'], gene_name_id=biocyc_data['gene_name_id'],
                       go_id=biocyc_data['go_id'], enzyme_id=biocyc_data['enzyme_id'],
                       reaction_id=biocyc_data['reaction_id'], ec_id=biocyc_data['ec_id'],
                       pathway_id=biocyc_data['pathway_id'], compound_id=biocyc_data['compound_id'])
    ptwy_info = biocyc_data['processed_kb'][kb][ptwy_position]
    rxn_info = biocyc_data['processed_kb'][kb][rxn_position]
    infoList = [ptwy_info] + [rxn_info]
    ptw_matrix = feat_obj.pathway_features(info_list=infoList, ptwy_ec_matrix=ptwy_ec_matrix, ec_idx=ec_idx,
                                           num_features=num_features, display_interval=display_interval)
    return ptw_matrix


def extract_reaction_properties(biocyc_data, ptwy_ec_matrix, rxn_ec_matrix, ptwy_ec_idx, rxn_ec_idx, num_features,
                                rxn_position=4, ptwy_position=5, kb='metacyc', display_interval=-1):
    '''

    :param display_interval:
    :param ptwy_ec_matrix:
    :param ptwy_position:
    :param kb:
    :return:
    '''
    print('\t  >> Extracting a set of properties of each EC from: {0}'.format(kb.upper()))
    feat_obj = Feature(protein_id=biocyc_data['protein_id'], product_id=biocyc_data['product_id'],
                       gene_id=biocyc_data['gene_id'], gene_name_id=biocyc_data['gene_name_id'],
                       go_id=biocyc_data['go_id'], enzyme_id=biocyc_data['enzyme_id'],
                       reaction_id=biocyc_data['reaction_id'], ec_id=biocyc_data['ec_id'],
                       pathway_id=biocyc_data['pathway_id'], compound_id=biocyc_data['compound_id'])
    ptwy_info = biocyc_data['processed_kb'][kb][ptwy_position]
    rxn_info = biocyc_data['processed_kb'][kb][rxn_position]
    info_list = [ptwy_info] + [rxn_info]
    matrix_list = [ptwy_ec_matrix] + [rxn_ec_matrix]
    ec_matrix = feat_obj.ec_features(info_list=info_list, matrix_list=matrix_list, ptwy_ec_idx=ptwy_ec_idx,
                                     rxn_ec_idx=rxn_ec_idx, num_features=num_features,
                                     display_interval=display_interval)
    return ec_matrix


# ---------------------------------------------------------------------------------------


# MAPPING AND CREATING MATRICES ---------------------------------------------------------

def __build_indicator_matrix(row_data, num_col_data, remove_zero_entries=False):
    '''
    This function creates a binary indicator (or adjacency) matrix given a list of
    row-wise data and a list of column-wise data. The matrix is saved (or pickled)
    in a binary format. The function has following arguments:
    :param remove_zero_entries:
    :type row_data: dict
    :param row_data: a dictionary of data from which to construct the matrix
    :type num_col_data: int
    :param num_col_data: integer number indicating the length of the column data for
                     the matrix
    '''

    nrowdata = len(row_data)
    matrix = np.zeros((nrowdata, num_col_data), dtype=np.int32)

    # Fill the sparse matrices
    for ridx, ritem in row_data.items():
        for idx in ritem:
            matrix[ridx, idx] += 1

    col_idx = list(range(num_col_data))

    if remove_zero_entries:
        total = np.sum(matrix, axis=0)
        col_idx = np.nonzero(total)[0]
        zeroIdx = np.where(total == 0)[0]
        matrix = np.delete(matrix, zeroIdx, axis=1)
    matrix = lil_matrix(matrix)
    return matrix, col_idx


def build_mapping_matrices(row_data, col_data, list_kb_paths, processed_kb, map_row_based_data_id, map_col_id,
                           constrain_on_kb, remove_zero_entries: bool = True, display_interval=-1, tag=''):
    '''
    This function is used for mapping any list of enzymes, reactions to reactions
    or pathways (not including superpathways). The function has following
    arguments:
    :param display_interval:
    :param constrain_on_kb:
    :type row_data: dict
    :param row_data: dictionary of data that a certain id is required to be mapped where
                    id is a dictionary
    :type col_data: dict
    :param col_data: dictionary of data that a certain id is required for mapping onto
                    rowdata where id is a dictionary
    :type map_col_id: int
    :param map_col_id: to be mapped using this key
    :type tag: str
    :param tag: to demonstrate this nugget of text onto the printing screen
    '''
    print('\t>> Constructing the sparse indicator (or adjacency) binary matrices for {0:s}'.format(tag))
    data_dict = OrderedDict()
    if constrain_on_kb:
        lst_kb = [constrain_on_kb]
    else:
        lst_kb = list_kb_paths

    for kb in lst_kb:
        data_info = processed_kb[kb][map_row_based_data_id]
        for n, item in enumerate(row_data.items()):
            (ritem, ridx) = item
            __desc(display_interval=display_interval, count=n + 1, total=len(row_data))
            if ridx not in data_dict:
                if ritem in data_info:
                    ls = data_info[ritem][map_col_id][1]
                    ls_idx = [col_data[citem] for citem in ls if citem in col_data]
                    data_dict.update({ridx: ls_idx})

    return __build_indicator_matrix(row_data=data_dict, num_col_data=len(col_data),
                                    remove_zero_entries=remove_zero_entries)

    # ---------------------------------------------------------------------------------------


def __desc(display_interval, count: int = 1, total: int = 1):
    desc = '\t   --> Processed: {0:.2f}%'.format(count / total * 100)
    if count == 1 or count % display_interval == 0:
        print(desc, end="\r")
    if count == total:
        print(desc)
