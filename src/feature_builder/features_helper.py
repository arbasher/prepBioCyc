import numpy as np
from feature_builder.feature import Feature
from scipy.sparse import lil_matrix
from utility.access_file import save_data


def build_features_matrix(biocyc_object, X, matrix_list, col_idx, provided_list=None, features_list=[42, 68, 32],
                          rxn_position=4, ptwy_position=5, display_interval=100, construct_reaction=False,
                          constraint_kb='metacyc', file_name='', save_path='.'):
    print('\t>> Building feature_builder from input data: {0}'.format(file_name + '_Xm.pkl'))
    file_name = file_name + '_Xm.pkl'

    if construct_reaction:
        if provided_list is None:
            idx_lst = biocyc_object['reaction_id']
        else:
            idx_lst = [(id, biocyc_object['reaction_id'][id]) for id in biocyc_object['reaction_id'].items() if
                       id in provided_list]
    else:
        if provided_list is None:
            idx_lst = biocyc_object['pathway_id']
        else:
            idx_lst = [(id, biocyc_object['pathway_id'][id]) for id in biocyc_object['pathway_id'].items() if
                       id in provided_list]

    feat_obj = Feature(protein_id=biocyc_object['protein_id'], product_id=biocyc_object['product_id'],
                       gene_id=biocyc_object['gene_id'], gene_name_id=biocyc_object['gene_name_id'],
                       go_id=biocyc_object['go_id'], enzyme_id=biocyc_object['enzyme_id'],
                       reaction_id=biocyc_object['reaction_id'], ec_id=biocyc_object['ec_id'],
                       pathway_id=biocyc_object['pathway_id'], compound_id=biocyc_object['compound_id'])

    ## For now on we are concentrating on pathways only
    ptw_info = biocyc_object['processed_kb'][constraint_kb][ptwy_position]
    rxn_info = biocyc_object['processed_kb'][constraint_kb][rxn_position]
    info_list = [ptw_info] + [rxn_info]

    for idx in np.arange(X.shape[0]):
        matrix_features = feat_obj.ec_evidence_features(instance=X[idx, :], info_list=info_list,
                                                        matrix_list=matrix_list, col_idx=col_idx,
                                                        num_features=features_list[1],
                                                        num_pathway_features=features_list[2])

        desc = '\t   --> Progress ({0:.2f}%): extracted feature_builder from {1:d} samples (out of {2:d})...'.format(
            (idx + 1) * 100.00 / X.shape[0], idx + 1, X.shape[0])
        if idx == 0 or idx % display_interval == 0:
            print(desc, end="\r")
        if idx + 1 == X.shape[0]:
            print(desc)

        tmp = X[idx, :].toarray()[0]
        tmp = np.hstack((tmp.reshape(1, X.shape[1]), matrix_features))
        tmp = lil_matrix(tmp)
        save_data(data=tmp, file_name=file_name, save_path=save_path, mode='a+b', print_tag=False)
