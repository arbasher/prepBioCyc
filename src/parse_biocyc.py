import os
import os.path
import pickle as pkl
import sys
import time
import traceback

import numpy as np
from scipy.sparse import lil_matrix

import biocyc as bco
import dataset_builder.build_dataset as dsb
from feature_builder.features_helper import build_features_matrix
from feature_builder.parse_features_list import extract_features_names
from graph_builder.graph import BioGraph
from kernels.pathway_similarity import build_similarity_matrix
from utility.access_file import save_data, load_data


def __parse_files(arg):
    steps = 1

    ##########################################################################################################
    ######################                   PREPROCESSING DATABASES                    ######################
    ##########################################################################################################

    data_object = bco.BioCyc()

    if arg.build_biocyc_object:
        print('\n{0})- Preprocessing BioCyc databases...'.format(steps))
        steps = steps + 1
        data_object.extract_info_from_database(db_path=arg.kbpath, data_folder='', constraint_on_kb=arg.constraint_kb)
        save_data(data=data_object.biocyc_data(), file_name=arg.object_name, save_path=arg.ospath,
                  tag='the biocyc object')

    ##########################################################################################################
    ######################          CREATING INDICATOR AND ADJACENCY MATRICES           ######################
    ##########################################################################################################

    if arg.build_indicator:
        print('\n{0})- Constructing the following indicator (or adjacency) binary matrices...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        reaction2ec, reaction2ec_idx = bco.build_mapping_matrices(row_data=data_object['reaction_id'],
                                                                  col_data=data_object['ec_id'],
                                                                  list_kb_paths=data_object['list_kb_paths'],
                                                                  processed_kb=data_object['processed_kb'],
                                                                  map_row_based_data_id=4, map_col_id=3,
                                                                  constrain_on_kb=arg.constraint_kb,
                                                                  display_interval=arg.display_interval,
                                                                  tag='Reactions vs ECs')
        pathway2ec, pathway2ec_idx = bco.build_mapping_matrices(row_data=data_object['pathway_id'],
                                                                col_data=data_object['ec_id'],
                                                                list_kb_paths=data_object['list_kb_paths'],
                                                                processed_kb=data_object['processed_kb'],
                                                                map_row_based_data_id=5, map_col_id=16,
                                                                constrain_on_kb=arg.constraint_kb,
                                                                display_interval=arg.display_interval,
                                                                tag='Pathways vs ECs')
        save_data(data=reaction2ec, file_name=arg.reaction2ec, save_path=arg.ospath, tag='reaction2ec matrix')
        save_data(data=reaction2ec_idx, file_name=arg.reaction2ec_idx, save_path=arg.ospath, tag='reaction2ec indices')
        save_data(data=pathway2ec, file_name=arg.pathway2ec, save_path=arg.ospath, tag='pathway2ec matrix')
        save_data(data=pathway2ec_idx, file_name=arg.pathway2ec_idx, save_path=arg.ospath, tag='pathway2ec indices')

        print('\n*** Mapping labels with functions...')
        row_tag = 'pathway'
        row_id = data_object['pathway_id']
        col_tag = 'ec'
        col_id = data_object['ec_id']
        if arg.construct_reaction:
            row_tag = 'reaction'
            row_id = data_object['reaction_id']
        if not arg.use_ec:
            col_tag = 'gene'
            col_id = data_object['gene_name_id']
        file_name = row_tag + '2' + col_tag
        dsb.__map_labels2functions(row_data_matrix=pathway2ec, col_postion_idx=pathway2ec_idx, y=None,
                                   num_samples=arg.num_sample, col_id=col_id, row_id=row_id, map_all=arg.map_all,
                                   col_tag=col_tag, row_tag=row_tag, display_interval=arg.display_interval,
                                   file_name=file_name, save_path=arg.ospath)

    ##########################################################################################################
    ################################          EXTRACTING PROPERTIES           ################################
    ##########################################################################################################

    if arg.build_pathway_properties:
        print('\n{0})- Extracting pathway properties...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        ptwy_features_matrix = bco.extract_pathway_properties(biocyc_data=data_object,
                                                              ptwy_ec_matrix=pathway2ec, ec_idx=pathway2ec_idx,
                                                              num_features=arg.num_pathway_features,
                                                              display_interval=arg.display_interval)
        save_data(data=ptwy_features_matrix, file_name=arg.pathway_feature, save_path=arg.ospath,
                  tag='the pathway properties')
        del ptwy_features_matrix

    if arg.build_ec_properties:
        print('\n{0})- Extracting EC properties...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        reaction2ec = load_data(file_name=arg.reaction2ec, load_path=arg.ospath)
        reaction2ec_idx = load_data(file_name=arg.reaction2ec_idx, load_path=arg.ospath)
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        ec_features_matrix = bco.extract_reaction_properties(biocyc_data=data_object, ptwy_ec_matrix=pathway2ec,
                                                             rxn_ec_matrix=reaction2ec, ptwy_ec_idx=pathway2ec_idx,
                                                             rxn_ec_idx=reaction2ec_idx,
                                                             num_features=arg.num_ec_features,
                                                             display_interval=arg.display_interval)
        save_data(data=ec_features_matrix, file_name=arg.ec_feature, save_path=arg.ospath,
                  tag='the reaction (ec) properties')
        del ec_features_matrix

    ##########################################################################################################
    ######################                  BUILD PATHWAY SIMILARITIES                  ######################
    ##########################################################################################################

    if arg.build_pathway_similarities:
        print('\n{0})- Building pathway similarity matrix...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        build_similarity_matrix(ptwy_ec_matrix=pathway2ec, pathway_id=data_object['pathway_id'],
                                processed_kb=data_object['processed_kb'], ptwy_position_idx=5, kb=arg.constraint_kb,
                                file_name=arg.pathway_similarity, save_path=arg.ospath)

    ##########################################################################################################
    ######################                          BUILD GRAPH                         ######################
    ##########################################################################################################

    if arg.build_graph:
        print('\n{0})- Building graph...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        reaction2ec = load_data(file_name=arg.reaction2ec, load_path=arg.ospath)
        reaction2ec_idx = load_data(file_name=arg.reaction2ec_idx, load_path=arg.ospath)
        ptwy_features_matrix = load_data(file_name=arg.pathway_feature, load_path=arg.ospath)
        ec_features_matrix = load_data(file_name=arg.ec_feature, load_path=arg.ospath)
        dict_features = extract_features_names(path=arg.featpath, file_name='features_list.txt',
                                               print_feats=False, tag='a list of feature_builder name')
        G = BioGraph(ec_id=data_object['ec_id'], compound_id=data_object['compound_id'],
                     pathway_id=data_object['pathway_id'], ec_features=ec_features_matrix,
                     ec_features_names=dict_features['Reaction-Print Features'],
                     pathway_features=ptwy_features_matrix,
                     pathway_features_names=dict_features['Pathway-Print Features'])
        G.build_graph(compound_id=data_object['compound_id'], pathway_id=data_object['pathway_id'],
                      reaction_id=data_object['reaction_id'], processed_kb=data_object['processed_kb'],
                      rxn_ec_spmatrix=reaction2ec, rxn_ec_idx=reaction2ec_idx, rxn_position_idx=4,
                      ptwy_position_idx=5, kb=arg.constraint_kb, filter_compound=arg.filter_compound_graph,
                      display_interval=arg.display_interval, save_path=arg.ospath)

    ##########################################################################################################
    ######################                 CONSTRUCT SYNTHETIC CORPORA                  ######################
    ##########################################################################################################

    if arg.build_synset:
        print('\n{0})- Constructing synthetic dataset...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        row_tag = 'pathway'
        col_tag = 'ec'
        row_id = data_object['pathway_id']
        col_id = data_object['ec_id']
        file_name = arg.synset_file_name
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        dsb.build_synthetic_dataset(num_samples=arg.num_sample, row_data_matrix=pathway2ec, col_idx=pathway2ec_idx,
                                    col_id=col_id, row_id=row_id, processed_kb=data_object['processed_kb'],
                                    average_num_items_per_sample=arg.average_item_per_sample,
                                    num_components_to_corrupt=arg.num_components_to_corrupt,
                                    lower_bound_num_item_ptwy=arg.lower_bound_num_item_ptwy,
                                    num_components_to_corrupt_outside=arg.num_components_to_corrupt_outside,
                                    add_noise=arg.add_noise, display_interval=arg.display_interval,
                                    provided_lst=None, constraint_kb=arg.constraint_kb, row_tag=row_tag,
                                    col_tag=col_tag, file_name=file_name, save_path=arg.dspath)

    if arg.ex_features_from_synset:
        print('\n{0})- Extracting feature_builder from synthetic dataset...'.format(steps))
        steps = steps + 1
        f_name = os.path.join(arg.dspath, arg.synset_file_name + '_X.pkl')
        print('\t\t## Loading dataset from: {0:s}'.format(f_name))
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is lil_matrix:
                    X = data
                    break
        f_name = os.path.join(arg.ospath, arg.ec_feature)
        print('\t\t## Loading the EC properties from: {0:s}'.format(f_name))
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_features_matrix = data
                    break
        feature_lst = [arg.num_reaction_evidence_features] + [arg.num_ec_evidence_features] + [
            arg.num_ptwy_evidence_features]
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        matrix_list = [pathway2ec] + [ec_features_matrix]
        f_name = arg.synset_file_name

        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        build_features_matrix(biocyc_object=data_object, X=X, matrix_list=matrix_list, col_idx=pathway2ec_idx,
                              features_list=feature_lst, display_interval=arg.display_interval,
                              constraint_kb=arg.constraint_kb, file_name=f_name, save_path=arg.dspath)

    ##########################################################################################################
    ######################                   CONSTRUCT GOLDEN CORPORA                   ######################
    ##########################################################################################################

    if arg.build_golden_dataset:
        print('\n{0})- Constructing golden dataset...'.format(steps))
        steps = steps + 1
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        row_tag = 'pathway'
        col_tag = 'ec'
        row_id = data_object['pathway_id']
        col_id = data_object['ec_id']
        file_name = arg.golden_file_name
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        dsb.build_golden_dataset(row_data_matrix=pathway2ec, col_idx=pathway2ec_idx, col_id=col_id, row_id=row_id,
                                 kb_list=data_object['list_kb_paths'], processed_kb=data_object['processed_kb'],
                                 display_interval=arg.display_interval, constraint_kb=arg.constraint_kb,
                                 build_pathologic_input=arg.build_pathologic_input,
                                 build_minpath_dataset=arg.build_minpath_dataset, minpath_map_file=arg.minpath_map,
                                 row_tag=row_tag, col_tag=col_tag, file_name=file_name, save_path=arg.dspath)

    if arg.ex_features_from_golden_dataset:
        print('\n{0})- Extracting feature_builder from golden dataset...'.format(steps))
        f_name = arg.golden_file_name + '_X.pkl'
        print('\t\t## Loading the golden dataset (X) from: {0:s}'.format(f_name))
        f_name = os.path.join(arg.dspath, f_name)
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is lil_matrix:
                    X = data
                    break
        print('\t\t## Loading the EC properties from: {0:s}'.format(arg.ec_feature))
        f_name = os.path.join(arg.ospath, arg.ec_feature)
        with open(f_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_features_matrix = data
                    break
        feature_lst = [arg.num_reaction_evidence_features] + [arg.num_ec_evidence_features] + [
            arg.num_ptwy_evidence_features]
        pathway2ec = load_data(file_name=arg.pathway2ec, load_path=arg.ospath)
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx, load_path=arg.ospath)
        matrix_list = [pathway2ec] + [ec_features_matrix]
        f_name = arg.golden_file_name
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        build_features_matrix(biocyc_object=data_object, X=X, matrix_list=matrix_list, col_idx=pathway2ec_idx,
                              features_list=feature_lst, display_interval=arg.display_interval,
                              constraint_kb=arg.constraint_kb, file_name=f_name, save_path=arg.dspath)


def biocyc_main(arg):
    try:
        if os.path.isdir(arg.kbpath):
            timeref = time.time()
            print(
                '*** PREPROCSSING BIOCYC DATABASES, BUILDING GRAPHS, EXTRACTING FEATURES, AND CREATING SYNTHETIC/GOLDEN DATASETS...')
            __parse_files(arg)
            print('\n*** THE PREPROCESSING CONSUMED {0:f} SECONDS\n'
                  .format(round(time.time() - timeref, 3)), file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE DATABASES', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
