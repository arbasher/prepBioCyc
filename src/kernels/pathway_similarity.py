import re

import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from kernels.shortest_path import ShortestPath
from kernels.smith_waterman import SmithWaterman
from scipy.sparse import lil_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel
from utility.access_file import save_data, reverse_idx


def build_similarity_matrix(ptwy_ec_matrix, pathway_id, processed_kb, ptwy_position_idx=4, kb='metacyc',
                            file_name='pathway_similarity', save_path='.'):
    ptwy_ec_matrix = lil_matrix(ptwy_ec_matrix).toarray()

    print(
        '\t>> Building pathway similarities based on Smith-Waterman and Shortest-Path algorithms from {0}...'.format(
            kb))
    ptw_info = processed_kb[kb][ptwy_position_idx]
    regex = re.compile(r'\(| |\)')
    ptw_idx = reverse_idx(pathway_id)
    sw = SmithWaterman()
    sp = ShortestPath()
    sw_matrix = lil_matrix((len(ptw_info), len(ptw_info)), dtype=np.float32)
    sp_matrix = lil_matrix((len(ptw_info), len(ptw_info)), dtype=np.float32)

    for i in np.arange(sw_matrix.shape[0]):
        query_ptwy_id = ptw_idx[i]

        # Extract the textual information of the query pathway
        query_ptwy = ptw_info[query_ptwy_id]
        query_text = str(query_ptwy[0][1]) + ' ' + ' '.join(query_ptwy[1][1]) + ' ' + ' '.join(query_ptwy[7][1])
        query_text = query_text.lower()

        query_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptw_info[query_ptwy_id][3][1]]
        for idx, itm in enumerate(query_end2end_rxn_series):
            itm = ' '.join(itm).replace('\"', '')
            query_end2end_rxn_series[idx] = itm.split()
        query_graph = nx.DiGraph()
        query_graph.add_nodes_from(ptw_info[query_ptwy_id][4][1])
        for itm in query_end2end_rxn_series:
            if len(itm) == 1:
                continue
            else:
                query_graph.add_edge(itm[1], itm[0])
        for j in np.arange(start=i + 1, stop=sw_matrix.shape[1]):
            target_ptwy_id = ptw_idx[j]

            # Find mutual pathways and change the alignment score accordingly
            target_ptwy = ptw_info[target_ptwy_id]
            target_text = str(target_ptwy[0][1]) + ' ' + ' '.join(target_ptwy[1][1]) + ' ' + ' '.join(
                target_ptwy[7][1])
            target_text = target_text.lower()
            if fuzz.token_sort_ratio(query_text, target_text) > 75:
                alignment_score = 2
            else:
                alignment_score = 1

            target_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptw_info[target_ptwy_id][3][1]]
            for idx, itm in enumerate(target_end2end_rxn_series):
                itm = ' '.join(itm).replace('\"', '')
                target_end2end_rxn_series[idx] = itm.split()
            target_graph = nx.DiGraph()
            target_graph.add_nodes_from(ptw_info[target_ptwy_id][4][1])
            for itm in target_end2end_rxn_series:
                if len(itm) == 1:
                    continue
                else:
                    target_graph.add_edge(itm[1], itm[0])

            sw_matrix[i, j] = sw.compare(G_1=query_graph, G_2=target_graph, alignment_score=alignment_score)
            sp_matrix[i, j] = sp.compare(G_1=query_graph, G_2=target_graph)
            sw_matrix[i, j] = sw_matrix[j, i]
            sp_matrix[i, j] = sp_matrix[j, i]

    file = file_name + '_sw.pkl'
    file_desc = '#File Description: number of pathways x number of pathways\n'
    save_data(data=file_desc, file_name=file, save_path=save_path,
              tag='the Smith-Waterman based pathway similarity matrix', mode='w+b')
    save_data(data=('nPathways:', str(sw_matrix.shape[0])), file_name=file, save_path=save_path, mode='a+b',
              print_tag=False)
    save_data(data=sw_matrix, file_name=file, save_path=save_path, mode='a+b', print_tag=False)

    file = file_name + '_sp.pkl'
    file_desc = '#File Description: number of pathways x number of pathways\n'
    save_data(data=file_desc, file_name=file, save_path=save_path,
              tag='the shortest-path based pathway similarity matrix', mode='w+b')
    save_data(data=('nPathways:', str(sp_matrix.shape[0])), file_name=file, save_path=save_path, mode='a+b',
              print_tag=False)
    save_data(data=sp_matrix, file_name=file, save_path=save_path, mode='a+b', print_tag=False)

    print('\t>> Building pathway similarities based on cosine similarity from {0}...'.format(kb))
    matrix = cosine_similarity(X=ptwy_ec_matrix)
    matrix = (matrix * 100).astype(int)
    matrix = lil_matrix(matrix, dtype=np.float32)
    file = file_name + '_cos.pkl'
    file_desc = '#File Description: number of pathways x number of pathways\n'
    save_data(data=file_desc, file_name=file, save_path=save_path,
              tag='the cosine based pathway similarity matrix', mode='w+b')
    save_data(data=('nPathways:', str(matrix.shape[0])), file_name=file, save_path=save_path, mode='a+b',
              print_tag=False)
    save_data(data=matrix, file_name=file, save_path=save_path, mode='a+b', print_tag=False)

    print('\t>> Building pathway similarities based on chi-squared kernel from {0}...'.format(kb))
    matrix = chi2_kernel(X=ptwy_ec_matrix)
    matrix = (matrix * 100).astype(int)
    matrix = lil_matrix(matrix, dtype=np.float32)
    file = file_name + '_chi2.pkl'
    file_desc = '#File Description: number of pathways x number of pathways\n'
    save_data(data=file_desc, file_name=file, save_path=save_path,
              tag='the chi-squared kernel based pathway similarity matrix', mode='w+b')
    save_data(data=('nPathways:', str(matrix.shape[0])), file_name=file, save_path=save_path, mode='a+b',
              print_tag=False)
    save_data(data=matrix, file_name=file, save_path=save_path, mode='a+b', print_tag=False)

    print('\t>> Building pathway similarities based on gaussian kernel from {0}...'.format(kb))
    rbf = RBF(length_scale=1.)
    matrix = rbf(X=ptwy_ec_matrix)
    matrix = (matrix * 100).astype(int)
    matrix = lil_matrix(matrix, dtype=np.float32)
    file = file_name + '_rbf.pkl'
    file_desc = '#File Description: number of pathways x number of pathways\n'
    save_data(data=file_desc, file_name=file, save_path=save_path,
              tag='the gaussian kernel based pathway similarity matrix', mode='w+b')
    save_data(data=('nPathways:', str(matrix.shape[0])), file_name=file, save_path=save_path, mode='a+b',
              print_tag=False)
    save_data(data=matrix, file_name=file, save_path=save_path, mode='a+b', print_tag=False)
