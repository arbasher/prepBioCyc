import os
import pickle as pkl
import re
from itertools import combinations
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from utility.access_file import save_data, reverse_idx


def __desc(display_interval, count: int = 1, total: int = 1):
    desc = '\t   --> Processed: {0:.2f}%'.format(count / total * 100)
    if count == 1 or count % display_interval == 0:
        print(desc, end="\r")
    if count == total:
        print(desc)


def __build_pathologic_input(X, col_idx, dict_id, dataset_list_ids, num_samples, save_path):
    print('\t>> Building the PathoLogic input file for: {0} samples'.format(num_samples))
    col_id = reverse_idx(dict_id)
    for idx in np.arange(X.shape[0]):
        file_name = ''
        dsname = ''
        for ds in dataset_list_ids[idx]:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            file_name = 'golden_' + str(len(dataset_list_ids[idx])) + '_' + str(idx)
            if len(dataset_list_ids[idx]) > 1:
                dsname += str(ds) + ' , '
            else:
                dsname = ds

        file = '0.pf'
        spath = os.path.join(save_path, file_name)

        id = 'ID\t' + file_name + '\n'
        name = 'NAME\t' + str(file_name) + '\n'
        type = 'TYPE\t:READ/CONTIG\n'
        annot = 'ANNOT-FILE\t' + file + '\n'
        comment = ';; DATASET\t' + dsname + '\n'
        datum = id + name + type + annot + comment
        save_data(data=datum, file_name='genetic-elements.dat', save_path=spath, tag='genetic elements info',
                  mode='w', w_string=True)

        id = 'ID\t' + file_name + '\n'
        storage = 'STORAGE\tFILE\n'
        name = 'NAME\t' + str(file_name) + '\n'
        abbr_name = 'ABBREV-NAME\t' + str(file_name) + '\n'
        strain = 'STRAIN\t1\n'
        rank = 'RANK\t|species|\n'
        ncbi_taxon = 'NCBI-TAXON-ID\t12908\n'

        datum = id + storage + name + abbr_name + strain + rank + ncbi_taxon
        save_data(data=datum, file_name='organism-params.dat', save_path=spath, tag='organism params info',
                  mode='w', w_string=True)

        save_data(data='', file_name=file, save_path=spath, tag='data description for ' + dsname, mode='w',
                  w_string=True)
        tmp = X[idx].rows[0]
        replicate = X[idx, tmp].data[0]
        stbase = 0
        edbase = 1
        func = 'ORF'
        total = 0
        for i, r in enumerate(replicate):
            for rep in range(r):
                id = 'ID\t' + str(idx) + '_' + str(total) + '\n'
                name = 'NAME\t' + str(idx) + '_' + str(total) + '\n'
                start_base = 'STARTBASE\t' + str(stbase) + '\n'
                end_base = 'ENDBASE\t' + str(edbase) + '\n'
                function = 'PRODUCT\t' + func + '\n'
                product_type = 'PRODUCT-TYPE\tP\n'
                tmp_ec = re.split("[.\-]+", col_id[col_idx[tmp[i]]])[1:]
                ec = list()
                not_found = False
                for e in tmp_ec:
                    if e.isdigit():
                        ec.append(e)
                    else:
                        not_found = True
                        ec.append('0')
                len_ec = len(ec)
                if len_ec < 4:
                    not_found = True
                    for e in range((4 - len_ec)):
                        ec.append('0')
                if not_found:
                    break
                ec = 'EC\t' + '.'.join(ec) + '\n'
                datum = id + name + start_base + end_base + function + product_type + ec + '//\n'
                total += 1
                save_data(data=datum, file_name=file, save_path=spath, tag='data description', mode='a',
                          w_string=True, print_tag=False)


def __build_minpath_dataset(X, col_idx, dict_id, num_samples, display_interval, file_name, save_path):
    print('\t>> Building the MinPath data file in (sample_(idx), EC or gene) format for: {0} samples'.format(
        num_samples))
    file = file_name + '_minpath_data.txt'
    dict_id = reverse_idx(dict_id)
    save_data(data='', file_name=file, save_path=save_path, tag='data description', mode='w', w_string=True)
    for idx in range(X.shape[0]):
        tmp = X[idx].rows[0]
        replicate = X[idx, tmp].data[0]
        for i, r in enumerate(replicate):
            sample_name = str(idx) + '\t' + dict_id[col_idx[tmp[i]]] + '\n'
            lst_samples = sample_name * r
            save_data(data=lst_samples, file_name=file, save_path=save_path, tag='data description', mode='a',
                      w_string=True, print_tag=False)
        __desc(display_interval=display_interval, count=idx + 1, total=X.shape[0])


def __corrput_item_by_removing_components(X_lst, id_item, dict_id, info_list, col_idx, num_components_to_corrupt,
                                          lower_bound_num_item_ptwy, construct_reaction):
    # Corrupting pathways that exceed some threshold by removing true ec or genes
    num_components = len(X_lst)
    ptw_info = info_list[0]
    rxn_info = info_list[1]

    if num_components > lower_bound_num_item_ptwy:
        if not construct_reaction:
            regex = re.compile(r'\(| |\)')
            if id_item not in ptw_info:
                pass
            text = str(ptw_info[id_item][0][1]) + ' ' + ' '.join(
                ptw_info[id_item][1][1]) + ' ' + ' '.join(
                ptw_info[id_item][6][1])
            text = text.lower()
            lst_rxn = [list(filter(None, regex.split(itm))) for itm in ptw_info[id_item][3][1]]
            for idx, itm in enumerate(lst_rxn):
                itm = ' '.join(itm).replace('\"', '')
                lst_rxn[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(ptw_info[id_item][4][1])
            for itm in lst_rxn:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])
            if 'detoxification' in text or 'degradation' in text:
                lst_rxn = dg.in_degree()
                lst_rxn = [n for n, d in sorted(lst_rxn, key=itemgetter(1))]
                lst_ec = [j for itm in lst_rxn for j in rxn_info[itm][3][1] if j in ptw_info[id_item][16][1]]
                # retain the first reaction
                if num_components_to_corrupt < len(lst_ec[1:]):
                    lst_rd = np.random.choice(a=lst_ec[1:], size=num_components_to_corrupt, replace=False)
                    for r in lst_rd:
                        X_lst.remove(np.where(col_idx == dict_id[r])[0])
            elif 'biosynthesis' in text:
                lst_rxn = dg.out_degree()
                lst_rxn = [n for n, d in sorted(lst_rxn, key=itemgetter(1))]
                lst_ec = [j for itm in lst_rxn for j in rxn_info[itm][3][1] if j in ptw_info[id_item][16][1]]

                ## retain the last two reactions
                if num_components_to_corrupt < len(lst_ec[2:]):
                    lst_rd = np.random.choice(a=lst_ec[2:], size=num_components_to_corrupt, replace=False)
                    for r in lst_rd:
                        X_lst.remove(np.where(col_idx == dict_id[r])[0])
            else:
                if num_components_to_corrupt < len(X_lst):
                    lst_rd = np.random.choice(a=X_lst, size=num_components_to_corrupt, replace=False)
                    for r in lst_rd:
                        X_lst.remove(r)
        else:
            if num_components_to_corrupt < len(X_lst):
                lst_rd = np.random.choice(a=X_lst, size=num_components_to_corrupt, replace=False)
                for r in lst_rd:
                    X_lst.remove(r)
    return X_lst


def __format_curated_dataset(num_samples, ptwy_ec_matrix, col_idx, col_id, row_id, col_tag, row_tag,
                             build_pathologic_input,
                             build_minpath_dataset, minpath_map_file, map_all, display_interval, file_name='',
                             load_path=''):
    file = file_name + '.pkl'
    X = lil_matrix((num_samples, len(col_idx)), dtype=np.int32)
    y = np.empty((num_samples,), dtype=np.object)

    count = 0
    sample_ids = list()
    print('\t>> Constructing sparse matrix for: {0} samples...'.format(num_samples))
    print('\t\t## Loading curated dataset: {0}'.format(file))
    file = os.path.join(load_path, file)

    with open(file, 'rb') as f_in:
        while count < num_samples:
            try:
                tmp = pkl.load(f_in)
                if len(tmp) == 3:
                    count += 1
                    lst_id, lst_x, lst_y = tmp
                    sidx = len(sample_ids)
                    sample_ids.append(lst_id)
                    for idx in lst_x:
                        X[sidx, idx] = X[sidx, idx] + 1
                    y[sidx] = np.unique(lst_y)
            except IOError:
                break

    if os.path.exists(file):
        os.remove(file)

    if build_pathologic_input:
        __build_pathologic_input(X=X, col_idx=col_idx, dict_id=col_id, dataset_list_ids=sample_ids,
                                 num_samples=num_samples, save_path=os.path.join(load_path, 'ptools'))

    if build_minpath_dataset:
        __build_minpath_dataset(X=X, col_idx=col_idx, dict_id=col_id, num_samples=num_samples,
                                display_interval=display_interval, file_name=file_name,
                                save_path=load_path)

    if minpath_map_file:
        __map_labels2functions(row_data_matrix=ptwy_ec_matrix, col_postion_idx=col_idx, col_id=col_id, row_id=row_id,
                               col_tag=col_tag, row_tag=row_tag, display_interval=display_interval, file_name=file_name,
                               save_path=load_path)

    file = file_name + '_X.pkl'
    save_data(data=X, file_name=file, save_path=load_path, mode='w+b', print_tag=False)

    file = file_name + '_y.pkl'
    # map pathways in y according to pathway dict
    y_csr = lil_matrix(np.zeros((y.shape[0], len(row_id))))
    for idx, item in enumerate(y):
        for ptwy in item:
            if ptwy in row_id:
                y_csr[idx, row_id[ptwy]] = 1
    save_data(data=y_csr, file_name=file, save_path=load_path, mode='w+b', print_tag=False)

    file = file_name + '_ids.pkl'
    save_data(data=sample_ids, file_name=file, save_path=load_path, mode='w+b', print_tag=False)


def build_synthetic_dataset(num_samples, row_data_matrix, col_idx, col_id, row_id, processed_kb,
                            average_num_items_per_sample=500, num_components_to_corrupt=2, lower_bound_num_item_ptwy=5,
                            num_components_to_corrupt_outside=3, add_noise=True, display_interval=-1,
                            construct_reaction=False, build_pathologic_input=False, build_minpath_dataset=False,
                            minpath_map_file='', map_all=True, provided_lst=None, constraint_kb='metacyc', row_tag='',
                            col_tag='', file_name=None, save_path=None):
    '''
    :type num_components_to_corrupt: int
    :param num_components_to_corrupt: Number of corrupted components for each true representation of pathway
    :type lower_bound_num_item_ptwy: int
    :param lower_bound_num_item_ptwy: the corruption process is constrained to only those pathways that have more than this number of ECs
    :type num_components_to_corrupt_outside: int
    :param num_components_to_corrupt_outside: Number of ECs to be corrupted by inserting false ECs to each true representation of a pathway
    '''

    file = file_name + '_' + str(num_samples) + '.pkl'

    if add_noise:
        print('\t>> The following settings are used for constructing the corpora:'
              '\n\t\t1. Number of true ECs to be corrupted for each pathway '
              '\n\t\t   by removing ECs except those do not exceed {0} ECs: {1}'
              '\n\t\t2. Number of samples to be generated: {2}'
              '\n\t\t3. Number of false ECs to be inserted for each pathway: {3}'
              '\n\t\t4. Constraints using the knowledge-base: {4}'
              '\n\t\t5. Saving the constructed dataset (as "{5}") to: {6}'
              .format(lower_bound_num_item_ptwy, num_components_to_corrupt, num_samples,
                      num_components_to_corrupt_outside,
                      constraint_kb.upper(), file, save_path))
    else:
        print('\t>> The following settings are used for constructing the corpora:'
              '\n\t\t1. Add Noise: False'
              '\n\t\t2. Number of samples to be generated: {0}'
              '\n\t\t3. Constraints using the knowledge-base: {1}'
              '\n\t\t4. Saving the constructed dataset (as "{2}") to: {3}'
              .format(num_samples, constraint_kb.upper(), file, save_path))

    use_idx = range(len(col_idx))

    item_idx_lst = reverse_idx(row_id)
    if provided_lst is None:
        idx_list = [idx for (item, idx) in row_id.items()]
    else:
        idx_list = [idx for (item, idx) in row_id.items() if item in provided_lst]

    ptw_info = processed_kb[constraint_kb][5]
    rxn_info = processed_kb[constraint_kb][4]
    info_list = [ptw_info] + [rxn_info]

    file_desc = '# Synthetic dataset is stored in this format: sample index, list of data components, list of labels'
    save_data(data=file_desc, file_name=file, save_path=save_path, tag='synthetic dataset', mode='w+b')

    for sidx in range(num_samples):
        num_pathways = np.random.poisson(lam=average_num_items_per_sample)
        item_list = np.random.choice(a=idx_list, size=num_pathways, replace=True)
        X = list()
        y = list()
        for idx in item_list:
            lst_item_x = list()
            id_item = item_idx_lst[idx]
            y.append(id_item)
            tmp = row_data_matrix[idx].rows[0]
            replicate = row_data_matrix[idx, tmp].data[0]
            for i, r in enumerate(replicate):
                t = [tmp[i]] * r
                lst_item_x.extend(t)

            # Choosing whether to corrupt by removing or inserting
            corrupt_type = np.random.choice(a=[0, 1, 2], size=1, replace=False)

            if corrupt_type == 1 and add_noise:
                lst_item_x = __corrput_item_by_removing_components(X_lst=lst_item_x, id_item=id_item, dict_id=col_id,
                                                                   info_list=info_list, col_idx=col_idx,
                                                                   num_components_to_corrupt=num_components_to_corrupt,
                                                                   lower_bound_num_item_ptwy=lower_bound_num_item_ptwy,
                                                                   construct_reaction=construct_reaction)
            elif corrupt_type == 2 and add_noise:
                # Corrupting pathways by adding false ec or genes
                lst_rd = np.random.choice(a=use_idx, size=num_components_to_corrupt_outside, replace=True)
                for r in lst_rd:
                    lst_item_x.append(r)
            else:
                pass
            X.extend(lst_item_x)
        __desc(display_interval=display_interval, count=sidx + 1, total=num_samples)
        save_data(data=(sidx, X, y), file_name=file, save_path=save_path, mode='a+b', print_tag=False)

    __format_curated_dataset(num_samples=num_samples, ptwy_ec_matrix=row_data_matrix, col_idx=col_idx, col_id=col_id,
                             row_id=row_id, col_tag=col_tag, row_tag=row_tag,
                             build_pathologic_input=build_pathologic_input, build_minpath_dataset=build_minpath_dataset,
                             minpath_map_file=minpath_map_file, map_all=map_all, display_interval=display_interval,
                             file_name=file_name, load_path=save_path)


def build_golden_dataset(row_data_matrix, col_idx, col_id, row_id, kb_list, processed_kb, display_interval=10,
                         construct_reaction=False, constraint_kb='metacyc', build_pathologic_input=True,
                         build_minpath_dataset=True,
                         minpath_map_file=True, map_all=True, row_tag='', col_tag='', file_name=None, save_path=None):
    kb_list = [kb for kb in kb_list if kb != constraint_kb]
    ds_lst = [list(combinations(kb_list, r + 1)) for r in range(len(kb_list))]
    ds_lst = [ds_tuple for item_lst in ds_lst for ds_tuple in item_lst]
    num_samples = len(ds_lst)
    item_info_by_kb = list()

    if construct_reaction:
        for ds in kb_list:
            item_info_by_kb.append([rxn for rxn in processed_kb[ds][4] if rxn in row_id])
    else:
        for ds in kb_list:
            item_info_by_kb.append([ptw for ptw in processed_kb[ds][5] if ptw in row_id])

    print('\t>> Constructing golden dataset...')
    file = file_name + '.pkl'
    if os.path.exists(os.path.join(save_path, file)):
        os.remove(os.path.join(save_path, file))
    for sidx, item_lst in enumerate(ds_lst):
        X = list()
        y = list()
        item_lst = list(item_lst)
        for ds in item_lst:
            ptw_lst = item_info_by_kb[kb_list.index(ds)]
            y.extend(ptw_lst)
            lst_item_x = list()
            for ptw in ptw_lst:
                tmp = row_data_matrix[row_id[ptw], :].rows[0]
                replicate = row_data_matrix[row_id[ptw], tmp].data[0]
                for i, r in enumerate(replicate):
                    t = [tmp[i]] * r
                    lst_item_x.extend(t)
            X.extend(lst_item_x)
        __desc(display_interval=display_interval, count=sidx + 1, total=num_samples)
        save_data(data=(item_lst, X, y), file_name=file, save_path=save_path, mode='a+b', print_tag=False)
    __format_curated_dataset(num_samples=num_samples, ptwy_ec_matrix=row_data_matrix, col_idx=col_idx, col_id=col_id,
                             row_id=row_id, col_tag=col_tag, row_tag=row_tag,
                             build_pathologic_input=build_pathologic_input, build_minpath_dataset=build_minpath_dataset,
                             minpath_map_file=minpath_map_file, map_all=map_all, display_interval=display_interval,
                             file_name=file_name, load_path=save_path)


def __map_labels2functions(row_data_matrix, col_postion_idx, col_id, row_id, col_tag='ec', row_tag='pathway',
                           display_interval=-1, file_name='ptw2ec.txt', save_path='.'):
    r_idx = reverse_idx(col_id)
    col_id = {}
    for idx in col_postion_idx:
        col_id.update({idx: r_idx[idx]})
    print('\t>> Referencing labels with functions...')
    file = file_name + '.txt'
    lst_y = [id for id, idx in row_id.items()]
    lst_y = [row_name for row_name in lst_y if row_data_matrix[row_id[row_name]].rows[0]]
    file_desc = '# File Description: ' + str(row_tag) + ' to ' + str(col_tag) + ' mapping file for ' + str(
        len(lst_y)) + ' functions\n#Pathway\tEC\n'
    save_data(data=file_desc, file_name=file, save_path=save_path, tag='mapping information', mode='w', w_string=True)
    for n, row_name in enumerate(lst_y):
        idx = row_data_matrix[row_id[row_name]].rows[0]
        for i in idx:
            replicate = row_data_matrix[row_id[row_name], i]
            mapping = row_name + '\t' + str(col_id[col_postion_idx[i]]) + '\n'
            list_rows = mapping * replicate
            if list_rows:
                save_data(data=list_rows, file_name=file, save_path=save_path, tag='mapping file', mode='a',
                          w_string=True, print_tag=False)
        __desc(display_interval=display_interval, count=n + 1, total=len(lst_y))
