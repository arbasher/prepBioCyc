import re

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from utility.access_file import save_data, save_json, save_gexf


class BioGraph(object):
    def __init__(self, ec_id, compound_id, pathway_id, ec_features=None, ec_features_names=None, compound_features=None,
                 compound_features_names=None, pathway_features=None, pathway_features_names=None):
        ec_graph = self.__add_nodes(ec_id, type='E', color='red', features=ec_features,
                                    features_names=ec_features_names, graph_name='ec_graph')
        compound_graph = self.__add_nodes(compound_id, type='C', color='green', features=compound_features,
                                          features_names=compound_features_names, graph_name='compound_graph')
        pathway_graph = self.__add_nodes(pathway_id, type='T', color='blue', features=pathway_features,
                                         features_names=pathway_features_names, graph_name='pathway_graph')
        self.multi_graphs = dict({'ec_graph': ec_graph, 'compound_graph': compound_graph,
                                  'pathway_graph': pathway_graph})
        self.ec2compound = lil_matrix((len(ec_id), len(compound_id)), dtype=np.int32)
        self.compound2pathway = lil_matrix((len(compound_id), len(pathway_id)), dtype=np.int32)

    def __add_nodes(self, dict_id, type, color, features, features_names, graph_name):
        G = nx.Graph()
        item_lst = [(item, {'idx': dict_id[item], 'type': type, 'color': color}) for item in dict_id]
        G.add_nodes_from(item_lst)
        G.features = features
        G.features_names = features_names
        G.name = graph_name
        return G

    def __save_graphs(self, save_path):
        for G_id, G_item in self.multi_graphs.items():
            save_data(data=G_item, file_name=G_id + ".pkl", save_path=save_path,
                      tag='the ' + G_id, mode='w+b')
            save_gexf(data=G_item, file_name=G_id + ".gexf", save_path=save_path,
                      tag='the ' + G_id)
            save_json(data=G_item, file_name=G_id + ".json", save_path=save_path,
                      tag='the ' + G_id)
        save_data(data=self.ec2compound, file_name="ec2compound.pkl",
                  save_path=save_path, tag='the ec-compound incidence matrix', mode='w+b')
        save_data(data=self.compound2pathway, file_name="compound2pathway.pkl",
                  save_path=save_path, tag='the compound-pathway incidence matrix', mode='w+b')

    def build_graph(self, compound_id, pathway_id, reaction_id, processed_kb, rxn_ec_spmatrix, rxn_ec_idx,
                    rxn_position_idx, ptwy_position_idx=5, kb='metacyc', filter_compound=True, display_interval=100,
                    save_path='.'):
        regex = re.compile(r'\(| |\)')
        print('\t>> Generate individual graphs for EC, compound, and pathway...')
        count = 1.
        for ptwy_id, ptwy_idx in pathway_id.items():
            desc = '\t   --> Generated: {0:.2f}%'.format(count / len(pathway_id) * 100)
            if count == 1 or count % display_interval == 0:
                print(desc, end="\r")
            if count == len(pathway_id):
                print(desc)
            count = count + 1

            ptwy_info = processed_kb[kb][ptwy_position_idx][ptwy_id]
            initial_incoming = False
            for ptwy_link in ptwy_info[2][1]:
                tmp_list = regex.split(ptwy_link)
                tmp_list = [tmp for tmp in tmp_list if tmp != '']
                ptwy_id_list = [ptwy for ptwy in tmp_list[1:] if ptwy in pathway_id]
                ptwy_idx_list = [pathway_id[ptwy] for ptwy in ptwy_id_list]

                if ptwy_idx_list:
                    if tmp_list[0] in compound_id:
                        self.compound2pathway[compound_id[tmp_list[0]], ptwy_idx_list] = 1
                if len(ptwy_info[2][1]) >= 2 and initial_incoming == False:
                    for ptwy in ptwy_id_list:
                        if ptwy == ptwy_id:
                            continue
                        if self.multi_graphs['pathway_graph'].has_edge(ptwy, ptwy_id):
                            self.multi_graphs['pathway_graph'].add_edge(ptwy, ptwy_id)
                    initial_incoming = True
                elif ":INCOMING" in tmp_list:
                    for ptwy in ptwy_id_list:
                        if ptwy == ptwy_id:
                            continue
                        if not self.multi_graphs['pathway_graph'].has_edge(ptwy, ptwy_id):
                            self.multi_graphs['pathway_graph'].add_edge(ptwy, ptwy_id)
                else:
                    for ptwy in ptwy_id_list:
                        if ptwy == ptwy_id:
                            continue
                        if not self.multi_graphs['pathway_graph'].has_edge(ptwy_id, ptwy):
                            self.multi_graphs['pathway_graph'].add_edge(ptwy_id, ptwy)

            reaction_predecessors = [list(filter(None, regex.split(itm))) for itm in ptwy_info[3][1]]
            for idx, itm in enumerate(reaction_predecessors):
                itm = ' '.join(itm).replace('\"', '')
                reaction_predecessors[idx] = itm.split()
            for itm in reaction_predecessors:
                if len(itm) == 1:
                    continue
                else:
                    left_ecs = processed_kb[kb][rxn_position_idx][itm[1]][3][1]
                    right_ecs = processed_kb[kb][rxn_position_idx][itm[0]][3][1]
                    for l_ec in left_ecs:
                        if l_ec in self.multi_graphs['ec_graph'].nodes():
                            for r_ec in right_ecs:
                                if r_ec in self.multi_graphs['ec_graph'].nodes():
                                    if l_ec == r_ec:
                                        continue
                                    if not self.multi_graphs['ec_graph'].has_edge(l_ec, r_ec):
                                        self.multi_graphs['ec_graph'].add_edge(l_ec, r_ec)

            rxn_layouts = [list(filter(None, regex.split(itm))) for itm in ptwy_info[5][1]]
            for rxn in rxn_layouts:
                first_item = False
                left_primaries = False
                right_primaries = False
                r2l = False
                ec_lst = list()
                left_primaries_list = list()
                right_primaries_list = list()
                for item in rxn:
                    if not first_item:
                        first_item = True
                        left_primaries = True
                        if item not in reaction_id:
                            continue
                        ec_lst = rxn_ec_spmatrix[reaction_id[item]].rows[0]
                        ec_lst = np.where(rxn_ec_idx == ec_lst)[0]
                    else:
                        if left_primaries:
                            if item == ':LEFT-PRIMARIES':
                                continue
                            else:
                                if item in compound_id:
                                    left_primaries_list.append(item)
                                    self.ec2compound[ec_lst, compound_id[item]] = 1
                                    self.compound2pathway[compound_id[item], ptwy_idx] = 1
                        if item == ':DIRECTION':
                            left_primaries = False
                            right_primaries = True
                        if item == ':R2L':
                            r2l = True
                        if right_primaries:
                            if item == ':RIGHT-PRIMARIES':
                                continue
                            else:
                                if item in compound_id:
                                    right_primaries_list.append(item)
                                    self.ec2compound[ec_lst, compound_id[item]] = 1
                                    self.compound2pathway[compound_id[item], ptwy_idx] = 1
                if filter_compound:
                    if len(rxn) > 8 or len(rxn) < 7:
                        continue
                for left in left_primaries_list:
                    for right in right_primaries_list:
                        if right == left:
                            continue
                        if r2l:
                            if not self.multi_graphs['compound_graph'].has_edge(right, left):
                                self.multi_graphs['compound_graph'].add_edge(right, left)
                        else:
                            if not self.multi_graphs['compound_graph'].has_edge(left, right):
                                self.multi_graphs['compound_graph'].add_edge(left, right)
        self.__save_graphs(save_path=save_path)
