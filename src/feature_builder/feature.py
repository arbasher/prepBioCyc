import re
from operator import itemgetter

import networkx as nx
import numpy as np
from fuzzywuzzy import process
from scipy.sparse import lil_matrix

ROMAN_CONSTANTS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XX", "XXX",
                   "XL", "L", "LX", "LXX", "LXXX", "XC", "C", "CC", "CCC", "CD", "D", "DC",
                   "DCC", "DCCC", "CM", "M", "MM", "MMM"]


class Feature(object):
    def __init__(self, protein_id, product_id, gene_id, gene_name_id, go_id, enzyme_id, reaction_id,
                 ec_id, pathway_id, compound_id):
        self.protein_id = protein_id
        self.product_id = product_id
        self.gene_id = gene_id
        self.gene_name_id = gene_name_id
        self.go_id = go_id
        self.enzyme_id = enzyme_id
        self.reaction_id = reaction_id
        self.ec_id = ec_id
        self.pathway_id = pathway_id
        self.compound_id = compound_id

    def __reverse_idx(self, value2idx):
        idx2value = {}
        for key, value in value2idx.items():
            idx2value.update({value: key})
        return idx2value

    def __desc(self, display_interval, count: int = 1, total: int = 1):
        desc = '\t\t--> Processed: {0:.2f}%'.format(count / total * 100)
        if count == 1 or count % display_interval == 0:
            print(desc, end="\r")
        if count == total:
            print(desc)

    ##########################################################################################################
    ############################           FEATURES FROM KNOWLEDGE-BASE            ###########################
    ##########################################################################################################

    def pathway_features(self, info_list, ptwy_ec_matrix, ec_idx, num_features=33, text_match_threshold=95,
                         display_interval=-1):
        ## Add the EC to each pathway and define kernel distance metric

        regex = re.compile(r'\(| |\)')
        matrix_features = lil_matrix((len(self.pathway_id), num_features), dtype=np.float32)

        for n, data_item in enumerate(info_list[0].items()):
            idx, item = data_item
            self.__desc(display_interval=display_interval, count=n + 1, total=len(info_list[0].items()))
            # 0. has-orphan-reaction (boolean)
            if item[14][1] > 0:
                matrix_features[self.pathway_id[idx], 0] = 1

            # 1. has-spontaneous-reaction (boolean)
            if item[15][1] > 0:
                matrix_features[self.pathway_id[idx], 1] = 1

            # 2. has-single-reaction (boolean)
            # 3. num-reactions (numeric)
            # 4. multiple-reaction-pathway (boolean)
            if len(item[4][1]) == 1:
                matrix_features[self.pathway_id[idx], 2] = 1
                matrix_features[self.pathway_id[idx], 3] = 1
            else:
                matrix_features[self.pathway_id[idx], 3] = len(item[4])
                matrix_features[self.pathway_id[idx], 4] = 1

            # 5. is-subpathway (boolean)
            if len(item[10][1]) != 0:
                matrix_features[self.pathway_id[idx], 5] = 1

            text = str(item[0][1]) + ' ' + ' '.join(item[1][1]) + ' ' + ' '.join(item[6][1])
            text = text.lower()

            # 6. is-energy-pathway (boolean)
            if 'energy' in text:
                matrix_features[self.pathway_id[idx], 6] = 1

            # 7. is-deg-or-detox-pathway (boolean)
            # 8. is-detoxification-pathway (boolean)
            # 9. is-degradation-pathway (boolean)
            if 'detoxification' in text or 'degradation' in text:
                matrix_features[self.pathway_id[idx], 7] = 1
            else:
                if 'detoxification' in text:
                    matrix_features[self.pathway_id[idx], 8] = 1
                if 'degradation' in text:
                    matrix_features[self.pathway_id[idx], 9] = 1

            # 10. is-biosynthesis-pathway (boolean)
            if 'biosynthesis' in text:
                matrix_features[self.pathway_id[idx], 10] = 1

            # 11. is-variant (boolean)
            for t in text.split():
                if t.upper() in ROMAN_CONSTANTS:
                    matrix_features[self.pathway_id[idx], 11] = 1

            # 12. num-initial-reactions (numeric)
            # 13. num-final-reactions (numeric)
            # 14. first-reaction-is-enzymatic (boolean)
            # 15. last-reaction-is-enzymatic (boolean)
            if item[3][1]:
                lst_rxn = [list(filter(None, regex.split(itm))) for itm in item[3][1]]
                dg = nx.DiGraph()
                dg.add_nodes_from(item[4][1])
                for itm in lst_rxn:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])

                count = 0
                for itm in dg.pred.items():
                    if len(itm[1]) == 0:
                        count += 1
                        itm = ''.join(itm[0]).replace('\"', '')
                        if info_list[1][itm][3][1]:
                            matrix_features[self.pathway_id[idx], 14] = 1
                matrix_features[self.pathway_id[idx], 12] = count

                count = 0
                for itm in dg.succ.items():
                    if len(itm[1]) == 0:
                        count += 1
                        itm = ''.join(itm[0]).replace('\"', '')
                        if info_list[1][itm][3][1]:
                            matrix_features[self.pathway_id[idx], 15] = 1
                matrix_features[self.pathway_id[idx], 13] = count

            # 16. has-unique-reactions (boolean)
            # 17. num-unique-reactions (numeric)
            if item[17][1]:
                matrix_features[self.pathway_id[idx], 16] = 1
                matrix_features[self.pathway_id[idx], 17] = len(item[17][1])

            # 18. has-enzymatic-reactions (boolean)
            # 19. num-enzymatic-reactions (numeric)
            # 20. fraction-unique-enzymes (numeric)
            if item[16][1]:
                matrix_features[self.pathway_id[idx], 18] = 1
                matrix_features[self.pathway_id[idx], 19] = len(item[16][1])
                matrix_features[self.pathway_id[idx], 20] = len(item[16][1]) / len(item[4][1])

            # 21. has-unique-enzymatic-reactions (boolean)
            # 22. num-unique-enzymatic-reactions (numeric)
            # 23. fraction-unique-enzymes (numeric)
            # 24. every-unique-reaction-has-enzyme (boolean)
            unique_ec_lst = list()
            num_ecs = 0
            for rxn in item[17][1]:
                rxn = ''.join(rxn).replace('\"', '')
                if info_list[1][rxn][3][1]:
                    e = [e for e in info_list[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
                    num_ecs += 1
            if unique_ec_lst:
                matrix_features[self.pathway_id[idx], 21] = 1
                matrix_features[self.pathway_id[idx], 22] = len(unique_ec_lst)
                matrix_features[self.pathway_id[idx], 23] = len(unique_ec_lst) / len(item[4][1])
            if num_ecs == len(item[17][1]):
                matrix_features[self.pathway_id[idx], 24] = 1

            # 25. has-key-reactions (boolean)
            # 26. num-key-reactions (numeric)
            if item[6][1]:
                matrix_features[self.pathway_id[idx], 25] = 1
                matrix_features[self.pathway_id[idx], 26] = len(item[6][1])

            # 27. subset-has-same-evidence (boolean)
            # 28. other-pathway-has-more-evidence (boolean)
            # 29. variant-has-more-evidence (boolean)
            lst_ptw_var = list()
            for pidx, pitem in info_list[0].items():
                if pidx != idx:
                    if set.intersection(set(item[4][1]), set(pitem[4][1])):
                        matrix_features[self.pathway_id[idx], 27] = 1
                    if set(item[4][1]) <= set(pitem[4][1]):
                        matrix_features[self.pathway_id[idx], 28] = 1
                    for t in text.split():
                        if t.upper() in ROMAN_CONSTANTS:
                            match = process.extract(pitem[0][1], [item[0][1]])
                            if match[0][1] > text_match_threshold:
                                lst_ptw_var.append(pidx)
                            break
            if lst_ptw_var:
                for id in lst_ptw_var:
                    if set(item[4][1]) <= set(info_list[0][id][4][1]):
                        matrix_features[self.pathway_id[idx], 29] = 1

            # 30. species-range-includes-target (boolean)
            if item[8][1]:
                matrix_features[self.pathway_id[idx], 30] = 1

            # 31. taxonomic-range-includes-target (boolean)
            if item[9][1]:
                matrix_features[self.pathway_id[idx], 31] = 1

            # 32. evidence-info-content-norm-all (numeric)
            m = np.sum(ptwy_ec_matrix[self.pathway_id[idx], :])
            total = 0
            for ec in set(item[16][1]):
                total += 1 / ptwy_ec_matrix[:, np.where(ec_idx == self.ec_id[ec])[0]].nnz
            if m != 0:
                matrix_features[self.pathway_id[idx], 32] = total / m

            # 33. evidence-info-content-unnorm (numeric)
            matrix_features[self.pathway_id[idx], 33] = total

            # TODO: 1)- DEAD-END-COMPOUND
            #       2)-
            # 34. has-genes-in-directon (boolean)
            # 35. has-proximal-genes (boolean)
            # 36. fraction-genes-in-directon (numeric)
            # 37. num-genes-in-directon (numeric)
            # 38. fraction-proximal-genes (numeric)
            # 39. num-proximal-genes (numeric)
        return matrix_features

    def ec_features(self, info_list, matrix_list, ptwy_ec_idx, rxn_ec_idx, num_features=25, initial_rxn=2, last_rxn=2,
                    display_interval=-1):
        regex = re.compile(r'\(| |\)')
        matrix_features = np.zeros(shape=(len(self.ec_id), num_features), dtype=np.object)
        ptw_idx = self.__reverse_idx(self.pathway_id)
        rxn_idx = self.__reverse_idx(self.reaction_id)

        for n, ec in enumerate(self.ec_id):
            self.__desc(display_interval=display_interval, count=n + 1, total=len(self.ec_id))
            ptw_lst = matrix_list[0][:, np.where(ptwy_ec_idx == self.ec_id[ec])[0]].nonzero()[0]
            count_lst = matrix_list[0][ptw_lst, np.where(ptwy_ec_idx == self.ec_id[ec])[0]].toarray()[0]

            # 0. num-pathways (numeric)
            matrix_features[self.ec_id[ec], 0] = len(ptw_lst)

            # 1. list-of-pathways (list)
            matrix_features[self.ec_id[ec], 1] = ptw_lst

            # 2. is-mapped-to-single-pathway (boolean)
            if len(ptw_lst) == 1:
                matrix_features[self.ec_id[ec], 2] = 1

            for idx, pidx in enumerate(ptw_lst):
                ptw = info_list[0][ptw_idx[pidx]]

                # 3. num-contributions-in-mapped-pathways (numeric)
                matrix_features[self.ec_id[ec], 3] += count_lst[idx]

                # 4. contributes-in-subpathway-as-inside-superpathways (boolean)
                # 5. num-contributions-in-subpathway-as-inside-superpathways (numeric)
                if len(ptw[10][1]):
                    matrix_features[self.ec_id[ec], 4] = 1
                    matrix_features[self.ec_id[ec], 5] += len(ptw[9][1])

                text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[7][1])
                text = text.lower()
                true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in ptw[3][1]]
                for idx, itm in enumerate(true_rxn_predecessors):
                    itm = ' '.join(itm).replace('\"', '')
                    true_rxn_predecessors[idx] = itm.split()
                dg = nx.DiGraph()
                dg.add_nodes_from(ptw[4][1])
                for itm in true_rxn_predecessors:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])

                initial_ec_lst = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
                initial_ec_lst = [j for itm in initial_ec_lst for j in info_list[1][itm][3][1] if
                                  j in ptw[16][1]]
                final_ec_lst = [n for n, d in sorted(dg.out_degree(), key=itemgetter(1))]
                final_ec_lst = [j for itm in final_ec_lst for j in info_list[1][itm][3][1] if
                                j in ptw[16][1]]

                # 6. act-as-initial-reactions (boolean)
                # 7. num-act-as-initial-reactions (numeric)
                if ec in initial_ec_lst[:initial_rxn]:
                    matrix_features[self.ec_id[ec], 6] = 1
                    matrix_features[self.ec_id[ec], 7] += 1

                # 8. act-as-final-reactions (boolean)
                # 9. num-act-as-final-reactions (numeric)
                if ec in final_ec_lst[:last_rxn]:
                    matrix_features[self.ec_id[ec], 8] = 1
                    matrix_features[self.ec_id[ec], 9] += 1

                # 10. act-as-initial-and-final-reactions (boolean)
                # 11. num-act-as-initial-and-final-reactions  (numeric)
                if ec in initial_ec_lst[:initial_rxn] and ec in final_ec_lst[:last_rxn]:
                    matrix_features[self.ec_id[ec], 10] = 1
                    matrix_features[self.ec_id[ec], 11] += 1

                # 12. act-in-deg-or-detox-pathway (boolean)
                # 13. num-act-in-deg-or-detox-pathway (numeric)
                if 'detoxification' in text or 'degradation' in text:
                    matrix_features[self.ec_id[ec], 12] = 1
                    matrix_features[self.ec_id[ec], 13] += 1

                # 14. act-in-biosynthesis-pathway (boolean)
                # 15. num-act-in-biosynthesis-pathway (numeric)
                if 'biosynthesis' in text:
                    matrix_features[self.ec_id[ec], 14] = 1
                    matrix_features[self.ec_id[ec], 15] += 1

                # 16. act-in-energy-pathway (boolean)
                # 17. num-act-in-energy-pathway (numeric)
                if 'energy' in text:
                    matrix_features[self.ec_id[ec], 16] = 1
                    matrix_features[self.ec_id[ec], 17] += 1

            rxn_lst = matrix_list[1][:, np.where(rxn_ec_idx == self.ec_id[ec])[0]].nonzero()[0]

            # 18. num-reactions (numeric)
            matrix_features[self.ec_id[ec], 18] = len(rxn_lst)

            # 19. list-of-reactions (list)
            matrix_features[self.ec_id[ec], 19] = rxn_lst

            # 20. act-in-unique-reaction (boolean)
            if len(rxn_lst) == 1:
                matrix_features[self.ec_id[ec], 20] = 1

            # 21. reactions-orphaned (boolean)
            # 22. num-reactions-orphaned (numeric)
            # 23. reactions-has-species (boolean)
            # 24. reactions-has-taxonomic-range (boolean)
            for idx, ridx in enumerate(rxn_lst):
                rxn = info_list[1][rxn_idx[ridx]]
                if ec in rxn[3][1]:
                    if rxn[5][1] != False:
                        matrix_features[self.ec_id[ec], 21] = 1
                        matrix_features[self.ec_id[ec], 22] += 1
                    if len(rxn[8][1]):
                        matrix_features[self.ec_id[ec], 23] = 1
                    if len(rxn[9][1]):
                        matrix_features[self.ec_id[ec], 24] = 1

        return matrix_features

    ##########################################################################################################
    ##########################           FEATURES FROM EXPERIMENTAL DATA            ##########################
    ##########################################################################################################

    def ec_evidence_features(self, instance, info_list, matrix_list, col_idx, num_features=82, num_pathway_features=28,
                             initial_reaction=2, last_reaction=2, threshold=0.5, beta=0.45):
        '''

        :param instance:
        :param info_list:
        :param matrix_list:
        :param num_features:
        :param initial_reaction:
        :param last_reaction:
        :param threshold:
        :return:
        '''
        one_hot = np.copy(instance.toarray()[0])
        one_hot[one_hot > 0] = 1
        ec_idx = col_idx[np.nonzero(one_hot)[0]]
        selected_list = matrix_list[1][ec_idx]
        m_features = np.zeros(shape=(1, num_features), dtype=np.float32)
        possible_pathways = np.zeros(shape=(1, len(info_list[0])), dtype=np.float32)
        ratio_possible_pathways = np.zeros(shape=(1, len(info_list[0])), dtype=np.float32)
        pathway_features = np.zeros(shape=(len(info_list[0]), num_pathway_features), dtype=np.float32)

        '''
        Extracting Various EC Features from Experimental Data
        '''
        # 0. fraction-total-ecs-to-distinct-ecs (numeric)
        if len(selected_list) != 0:
            m_features[0, 0] = np.sum(instance) / len(selected_list)

        # 1. fraction-total-possible-pathways-to-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 1] = np.sum(selected_list[:, 0]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 2. fraction-total-ecs-to-ecs-mapped-to-single-pathways (numeric)
        if np.sum(instance) != 0:
            m_features[0, 2] = np.sum(selected_list[:, 2]) / np.sum(instance)

        # 3. fraction-total-ecs-mapped-to-pathways (numeric)
        if np.sum(selected_list[:, 3]) != 0:
            m_features[0, 3] = np.sum(instance) / np.sum(selected_list[:, 3])

        # 4. fraction-total-distinct-ecs-contribute-in-subpathway-as-inside-superpathways (numeric)
        if np.sum(instance) != 0:
            m_features[0, 4] = np.sum(selected_list[:, 4]) / np.sum(instance)

        # 5. fraction-total-ecs-contribute-in-subpathway-as-inside-superpathways (numeric)
        if np.sum(instance) != 0:
            m_features[0, 5] = np.sum(selected_list[:, 5]) / np.sum(instance)

        # 6. fraction-total-distinct-ecs-act-as-initial-reactions (numeric)
        if np.sum(instance) != 0:
            m_features[0, 6] = np.sum(selected_list[:, 6]) / np.sum(instance)

        # 7. fraction-total-ecs-act-as-initial-reactions (numeric)
        if np.sum(instance) != 0:
            m_features[0, 7] = np.sum(selected_list[:, 7]) / np.sum(instance)

        # 8. fraction-total-distinct-ecs-act-as-final-reactions (numeric)
        if np.sum(instance) != 0:
            m_features[0, 8] = np.sum(selected_list[:, 8]) / np.sum(instance)

        # 9. fraction-total-ecs-act-as-final-reactions (numeric)
        if np.sum(instance) != 0:
            m_features[0, 9] = np.sum(selected_list[:, 9]) / np.sum(instance)

        # 10. fraction-total-distinct-ecs-act-as-initial-and-final-reactions (numeric)
        if np.sum(instance) != 0:
            m_features[0, 10] = np.sum(selected_list[:, 10]) / np.sum(instance)

        # 11. fraction-total-ecs-act-as-initial-and-final-reactions  (numeric)
        if np.sum(instance) != 0:
            m_features[0, 11] = np.sum(selected_list[:, 11]) / np.sum(instance)

        # 12. fraction-total-distinct-ecs-act-in-deg-or-detox-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 12] = np.sum(selected_list[:, 12]) / np.sum(instance)

        # 13. fraction-total-ecs-act-in-deg-or-detox-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 13] = np.sum(selected_list[:, 13]) / np.sum(instance)

        # 14. fraction-total-distinct-ec-act-in-biosynthesis-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 14] = np.sum(selected_list[:, 14]) / np.sum(instance)

        # 15. fraction-total-ec-act-in-biosynthesis-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 15] = np.sum(selected_list[:, 15]) / np.sum(instance)

        # 16. fraction-total-distinct-ec-act-in-energy-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 16] = np.sum(selected_list[:, 16]) / np.sum(instance)

        # 17. fraction-total-ec-act-in-energy-pathway (numeric)
        if np.sum(instance) != 0:
            m_features[0, 17] = np.sum(selected_list[:, 17]) / np.sum(instance)

        # 18. fraction-total-ecs-to-total-reactions (numeric)
        if np.sum(selected_list[:, 18]) != 0:
            m_features[0, 18] = np.sum(instance) / np.sum(selected_list[:, 18])

        # 19. fraction-total-distinct-ecs-to-total-distinct-reactions (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 19]))) != 0:
            m_features[0, 19] = len(selected_list) / len(np.unique(np.concatenate(selected_list[:, 19])))

        # 20. fraction-total-ec-contribute-in-unique-reaction (numeric)
        if np.sum(instance) != 0:
            m_features[0, 20] = np.sum(selected_list[:, 20]) / np.sum(instance)

        # 21. fraction-total-distinct-ec-contribute-to-reactions-has-taxonomic-range (numeric)
        if np.sum(instance) != 0:
            m_features[0, 21] = np.sum(selected_list[:, 24]) / np.sum(instance)

        # 22. fraction-total-pathways-over-total-ecs (numeric)
        if np.sum(instance) != 0:
            m_features[0, 22] = np.sum(selected_list[:, 0]) / np.sum(instance)

        # 23. fraction-total-pathways-over-distinct-ec (numeric)
        if len(selected_list) != 0:
            m_features[0, 23] = np.sum(selected_list[:, 0]) / len(selected_list)

        # 24. fraction-total-distinct-pathways-over-distinct-ec (numeric)
        if len(selected_list) != 0:
            m_features[0, 24] = len(np.unique(np.concatenate(selected_list[:, 1]))) / len(selected_list)

        # 25. fraction-distinct-ec-contributes-in-subpathway-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 25] = np.sum(selected_list[:, 4]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 26. fraction-ec-contributes-in-subpathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 26] = np.sum(selected_list[:, 5]) / np.sum(selected_list[:, 0])

        # 27. fraction-distinct-ec-act-in-deg-or-detox-pathway-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 27] = np.sum(selected_list[:, 12]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 28. fraction-distinct-ec-act-in-deg-or-detox-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 28] = np.sum(selected_list[:, 12]) / np.sum(selected_list[:, 0])

        # 29. fraction-ec-act-in-deg-or-detox-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 29] = np.sum(selected_list[:, 13]) / np.sum(selected_list[:, 0])

        # 30. fraction-distinct-ec-act-in-biosynthesis-pathway-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 30] = np.sum(selected_list[:, 14]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 31. fraction-distinct-ec-act-in-biosynthesis-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 31] = np.sum(selected_list[:, 14]) / np.sum(selected_list[:, 0])

        # 32. fraction-ec-act-in-biosynthesis-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 32] = np.sum(selected_list[:, 15]) / np.sum(selected_list[:, 0])

        # 33. fraction-distinct-ec-act-in-energy-pathway-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 33] = np.sum(selected_list[:, 16]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 34. fraction-distinct-ec-act-in-energy-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 34] = np.sum(selected_list[:, 16]) / np.sum(selected_list[:, 0])

        # 35. fraction-ec-act-in-energy-pathway-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 35] = np.sum(selected_list[:, 17]) / np.sum(selected_list[:, 0])

        # 36. fraction-total-reactions-over-total-pathways (numeric)
        if np.sum(selected_list[:, 0]) != 0:
            m_features[0, 36] = np.sum(selected_list[:, 18]) / np.sum(selected_list[:, 0])

        # 37. fraction-total-reactions-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 37] = np.sum(selected_list[:, 18]) / len(np.unique(np.concatenate(selected_list[:, 1])))

        # 38. fraction-distinct-reaction-over-distinct-pathways (numeric)
        if len(np.unique(np.concatenate(selected_list[:, 1]))) != 0:
            m_features[0, 38] = len(np.unique(np.concatenate(selected_list[:, 19]))) / len(
                np.unique(np.concatenate(selected_list[:, 1])))

        '''
        Extracting Pathway Features from Experimental Data
        '''

        regex = re.compile(r'\(| |\)')
        ptw_idx = self.__reverse_idx(self.pathway_id)

        for pidx in np.unique(np.concatenate(selected_list[:, 1])):
            bin_ptw = matrix_list[0][pidx]
            bin_ptw[bin_ptw > 0] = 1
            num_common_items = (one_hot & bin_ptw.toarray()[0]).sum()
            if int(num_common_items) != 0:
                ratio_possible_pathways[0, pidx] = np.divide(num_common_items, bin_ptw.sum())
            pid = ptw_idx[pidx]
            ptw = info_list[0][pid]
            text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[7][1])
            text = text.lower()
            true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in info_list[0][pid][3][1]]
            for idx, itm in enumerate(true_rxn_predecessors):
                itm = ' '.join(itm).replace('\"', '')
                true_rxn_predecessors[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(info_list[0][pid][4][1])
            for itm in true_rxn_predecessors:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])

            true_prev_ec_lst = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
            true_prev_ec_lst = [self.ec_id[j] for itm in true_prev_ec_lst for j in info_list[1][itm][3][1] if
                                j in info_list[0][pid][16][1]]
            true_succ_ec_lst = [n for n, d in sorted(dg.out_degree(), key=itemgetter(1))]
            true_succ_ec_lst = [self.ec_id[j] for itm in true_succ_ec_lst for j in info_list[1][itm][3][1] if
                                j in info_list[0][pid][16][1]]

            unique_ec_lst = list()
            orphan_ec_lst = list()
            for rxn in ptw[17][1]:
                if info_list[1][rxn][3][1]:
                    e = [self.ec_id[e] for e in info_list[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
                if info_list[1][rxn][5][1] != False:
                    orphan_ec_lst.extend(e)

            '''
            Extracting Info from Knowledge Data
            '''

            sample_ec_lst = [ec for ec in true_prev_ec_lst if ec in ec_idx]
            sample_unique_ec_lst = [ec for ec in sample_ec_lst if ec in unique_ec_lst]
            if true_prev_ec_lst:
                sample_initial_ec_lst = [1 for ec in true_prev_ec_lst[:initial_reaction] if ec not in sample_ec_lst]
            if true_succ_ec_lst:
                sample_final_ec_lst = [1 for ec in true_succ_ec_lst[:last_reaction] if ec not in sample_ec_lst]

            '''
            Continue Pathway Features Extracting
            '''
            # 39. ecs-in-energy-pathways-mostly-missing (numeric)
            if 'energy' in text:
                if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                    m_features[0, 39] += 1

            # 40. ecs-in-pathways-mostly-present (numeric)
            # 0. ecs-mostly-present-in-pathway (boolean)
            # 1. prob-ecs-mostly-present-in-pathway (numeric)
            if sample_ec_lst:
                if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                    m1 = len(sample_ec_lst) + 1 == len(ptw[16][1])
                    m2 = (len(sample_ec_lst) / len(ptw[16][1])) >= threshold
                    if m1 and m2:
                        m_features[0, 40] += 1
                        pathway_features[pidx, 0] = 1
                        if len(ptw[16][1]) != 0:
                            pathway_features[pidx, 1] = len(sample_ec_lst) / len(ptw[16][1])

            # 41. all-initial-ecs-present-in-pathways (numeric)
            # 2. all-initial-ecs-present-in-pathway (boolean)
            # 3. prob-initial-ecs-present-in-pathway (numeric)
            if sample_initial_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[0, 41] += 1
                    pathway_features[pidx, 2] = 1
                else:
                    if len(true_prev_ec_lst[:initial_reaction]) != 0:
                        pathway_features[pidx, 3] = len(sample_initial_ec_lst) / len(
                            true_prev_ec_lst[:initial_reaction])

            # 42. all-final-ecs-present-in-pathways (numeric)
            # 4.  all-final-ecs-present-in-pathway (boolean)
            # 5.  prob-final-ecs-present-in-pathway (numeric)
            if sample_final_ec_lst:
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[0, 42] += 1
                    pathway_features[pidx, 4] = 1
                else:
                    if len(true_succ_ec_lst[:last_reaction]) != 0:
                        pathway_features[pidx, 5] = len(sample_final_ec_lst) / len(true_succ_ec_lst[:last_reaction])

            # 43. all-initial-and-final-ecs-present-in-pathways (numeric)
            # 6.  all-initial-and-final-ecs-present-in-pathway (boolean)
            # 7.  prob-all-initial-and-final-ecs-present-in-pathway (numeric)
            if sample_initial_ec_lst and sample_final_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]) and len(
                        sample_final_ec_lst) == len(
                    true_succ_ec_lst[:last_reaction]):
                    m_features[0, 43] += 1
                    pathway_features[pidx, 6] = 1
                    totalPECs = len(sample_initial_ec_lst) + len(sample_final_ec_lst)
                    totalTECs = len(true_prev_ec_lst[:initial_reaction]) + len(true_succ_ec_lst[:last_reaction])
                    pathway_features[pidx, 7] = totalPECs / totalTECs

            # 44. all-initial-ecs-present-in-deg-or-detox-pathways (numeric)
            # 45. all-final-ecs-present-in-deg-or-detox-pathways (numeric)
            # 8. all-initial-ecs-present-in-deg-or-detox-pathway (boolean)
            # 9. prob-all-initial-ecs-present-in-deg-or-detox-pathway (numeric)
            if 'detoxification' in text or 'degradation' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[0, 44] += 1
                    pathway_features[pidx, 8] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[0, 45] += 1
                    pathway_features[pidx, 9] = 1

            # 46. all-initial-ecs-present-in-biosynthesis-pathways (numeric)
            # 47. all-final-ecs-present-in-biosynthesis-pathways (numeric)
            # 10. all-initial-ecs-present-in-biosynthesis-pathway (boolean)
            # 11. prob-all-initial-ecs-present-in-biosynthesis-pathway (numeric)
            if 'biosynthesis' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[0, 46] += 1
                    pathway_features[pidx, 10] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[0, 47] += 1
                    pathway_features[pidx, 11] = 1

            # 48. most-ecs-absent-in-pathways (numeric)
            # 12. most-ecs-absent-in-pathway (boolean)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    m_features[0, 48] += 1
                    pathway_features[0, 12] = 1

            # 49. most-ecs-absent-not-distinct-in-pathways (numeric)
            # 13. most-ecs-absent-not-distinct-in-pathway (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                if unique_ec_lst:
                    m_features[0, 49] += 1
                    pathway_features[0, 13] = 1
                    for e in unique_ec_lst:
                        if e in sample_ec_lst:
                            m_features[0, 49] -= 1
                            pathway_features[0, 13] = 0
                            break

            # 50. one-ec-present-but-in-minority-in-pathways (numeric)
            # 14. one-ec-present-but-in-minority-in-pathway (boolen)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    m_features[0, 50] += 1
                    pathway_features[0, 14] = 1

            # 51. all-distinct-ec-present-in-pathways (numeric)
            # 52. all-ecs-present-in-pathways (numeric)
            # 53. all-distinct-ec-present-or-orphaned-in-pathways (numeric)
            # 54. all-ec-present-or-orphaned-in-pathways (numeric)

            # 15. all-distinct-ec-present-in-pathway (boolean)
            # 16. all-ecs-present-in-pathway (boolean)
            # 17. all-distinct-ec-present-or-orphaned-in-pathway (boolean)
            # 18. all-ec-present-or-orphaned-in-pathway (boolean)

            m_features[0, 51] += 1
            m_features[0, 52] += 1
            m_features[0, 53] += 1
            m_features[0, 54] += 1

            pathway_features[pidx, 15] = 1
            pathway_features[pidx, 16] = 1
            pathway_features[pidx, 17] = 1
            pathway_features[pidx, 18] = 1

            if sample_unique_ec_lst != unique_ec_lst:
                m_features[0, 51] -= 1
                pathway_features[pidx, 15] = 0

            if sample_ec_lst != true_prev_ec_lst:
                m_features[0, 52] -= 1
                pathway_features[pidx, 16] = 0
            u = 0
            for ec in unique_ec_lst:
                if ec in sample_unique_ec_lst or ec in orphan_ec_lst:
                    u += 1
            if u != len(unique_ec_lst):
                m_features[0, 53] -= 1
                pathway_features[pidx, 17] = 0
            a = 0
            for ec in true_prev_ec_lst:
                if ec in sample_ec_lst or ec in orphan_ec_lst:
                    a += 1
            if a != len(true_prev_ec_lst):
                m_features[0, 54] -= 1
                pathway_features[pidx, 18] = 0

            # 55. majority-of-ecs-absent-in-pathways (numeric)
            # 56. majority-of-ecs-present-in-pathways (numeric)
            # 19. majority-of-ecs-absent-in-pathway (boolean)
            # 20. majority-of-ecs-present-in-pathway (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                m_features[0, 55] += 1
                pathway_features[pidx, 19] = 1
            else:
                m_features[0, 56] += 1
                pathway_features[pidx, 20] = 1

            # 57. majority-of-distinct-ecs-present-in-pathways (numeric)
            # 21. majority-of-distinct-ecs-present-in-pathway (boolean)
            if len(sample_unique_ec_lst) > int(threshold * len(unique_ec_lst)):
                m_features[0, 57] += 1
                pathway_features[pidx, 21] = 1

            # 58. majority-of-reactions-present-distinct-in-pathways (numeric)
            # 22. majority-of-reactions-present-distinct-in-pathway (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                m_features[0, 58] += 1
                pathway_features[pidx, 22] = 1
                for ec in sample_ec_lst:
                    if ec not in unique_ec_lst:
                        m_features[0, 58] -= 1
                        pathway_features[pidx, 22] = 0
                        break

            # 59.  missing-at-most-one-ec-in-pathways (numeric)
            # 23.  missing-at-most-one-ec-in-pathway (boolean)
            if len(sample_ec_lst) + 1 == len(true_prev_ec_lst):
                ec = set.difference(set(sample_ec_lst), set(true_prev_ec_lst))
                if ec not in orphan_ec_lst:
                    m_features[0, 59] += 1
                    pathway_features[pidx, 23] = 1

            # 60.  has-distinct-ecs-present-in-pathways (numeric)
            # 24.  has-distinct-ecs-present-in-pathway (boolean)
            if sample_unique_ec_lst:
                m_features[0, 60] += 1
                pathway_features[pidx, 24] = 1

            # 62.  fraction-reactions-present-or-orphaned-distinct-in-pathways (numeric)
            # 26.  fraction-reactions-present-or-orphaned-distinct-in-pathway (numeric)
            sample_rxn_unique_orphand = set.union(set(sample_unique_ec_lst), set(orphan_ec_lst))
            true_ec_orphand = set.union(set(true_prev_ec_lst), set(orphan_ec_lst))
            if len(true_ec_orphand) != 0:
                m_features[0, 61] += len(sample_rxn_unique_orphand) / len(true_ec_orphand)
                pathway_features[pidx, 25] = len(sample_rxn_unique_orphand) / len(true_ec_orphand)

            # 61.  fraction-distinct-ecs-present-or-orphaned-in-pathways (numeric)
            # 25.  fraction-distinct-ecs-present-or-orphaned-in-pathway (numeric)
            if len(unique_ec_lst):
                true_ec_unique_orphand = set.union(set(unique_ec_lst), set(orphan_ec_lst))
                if len(true_ec_unique_orphand) != 0:
                    m_features[0, 62] += len(sample_rxn_unique_orphand) / len(true_ec_unique_orphand)
                    pathway_features[pidx, 26] = len(sample_rxn_unique_orphand) / len(true_ec_unique_orphand)

            # 63.  fraction-reactions-present-or-orphaned-in-pathways (numeric)
            # 27.  fraction-reactions-present-or-orphaned-in-pathway (numeric)
            sample_rxn_orphand = set.union(set(sample_ec_lst), set(orphan_ec_lst))
            if len(true_ec_orphand) != 0:
                m_features[0, 63] += len(sample_rxn_orphand) / len(true_ec_orphand)
                pathway_features[pidx, 27] = len(sample_rxn_orphand) / len(true_ec_orphand)

            # 64.  num-distinct-reactions-present-or-orphaned-in-pathways (numeric)
            # 28.  num-distinct-reactions-present-or-orphaned-in-pathway (numeric)
            m_features[0, 64] += len(sample_rxn_unique_orphand)
            pathway_features[pidx, 28] = len(sample_rxn_unique_orphand)

            # 65.  num-reactions-present-or-orphaned-in-pathways (numeric)
            # 29.  num-reactions-present-or-orphaned-in-pathway (numeric)
            m_features[0, 65] += len(sample_rxn_orphand)
            pathway_features[pidx, 29] = len(sample_rxn_orphand)

            # 66.  evidence-info-content-norm-present-in-pathways (numeric)
            # 67.  evidence-info-content-present-in-pathways (numeric)
            # 30.  evidence-info-content-norm-present-in-pathway (numeric)
            # 31.  evidence-info-content-present-in-pathway (numeric)
            total = 0
            for ec in set(sample_ec_lst):
                total += 1 / matrix_list[0][:, np.where(col_idx == ec)[0]].nnz
            if sample_ec_lst:
                if len(true_prev_ec_lst) != 0:
                    m_features[0, 66] += total / len(true_prev_ec_lst)
                    pathway_features[pidx, 30] = total / len(true_prev_ec_lst)
                m_features[0, 67] += total
                pathway_features[pidx, 31] = total

        m_features[0, 39:] = m_features[0, 39:] / pathway_features.shape[0]
        pathway_features = pathway_features.reshape(1, pathway_features.shape[0] * pathway_features.shape[1])
        # 68.  possible-pathways-present (boolean)
        max_val = np.max(ratio_possible_pathways) * beta
        possible_pathways[ratio_possible_pathways >= max_val] = 1
        possible_pathways[ratio_possible_pathways >= threshold] = 1
        # 69.  prob-possible-pathways-present (numeric)
        ratio_possible_pathways = ratio_possible_pathways
        total_features = np.hstack((m_features, possible_pathways, ratio_possible_pathways, pathway_features))
        return total_features

    def reaction_evidence_features(self, instance, info_list, ptwy_ec_matrix, num_features=42, initial_reaction=2,
                                   last_reaction=2, threshold=0.5):
        m_features = np.zeros(shape=(len(self.pathway_id), num_features), dtype=np.float32)
        regex = re.compile(r'\(| |\)')

        for id in self.pathway_id:
            ptw = info_list[0][id]
            text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[6][1])
            text = text.lower()
            true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in info_list[0][id][3][1]]
            for idx, itm in enumerate(true_rxn_predecessors):
                itm = ' '.join(itm).replace('\"', '')
                true_rxn_predecessors[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(info_list[0][id][4][1])
            for itm in true_rxn_predecessors:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])
            true_prev_ec_lst = sorted(dg.in_degree(), key=dg.in_degree().get)
            true_prev_ec_lst = [j for itm in true_prev_ec_lst for j in info_list[1][itm][3][1] if
                                j in info_list[0][id][15][1]]
            true_succ_ec_lst = sorted(dg.out_degree(), key=dg.out_degree().get)
            true_succ_ec_lst = [j for itm in true_succ_ec_lst for j in info_list[1][itm][3][1] if
                                j in info_list[0][id][15][1]]

            unique_ec_lst = list()
            orphan_ec_lst = list()
            for rxn in ptw[16][1]:
                if info_list[1][rxn][3][1]:
                    e = [e for e in info_list[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
                if info_list[1][rxn][5][1] != False:
                    orphan_ec_lst.extend(e)

            #######################################################################################################
            ################################ Extracting Info from Experimental Data ###############################
            #######################################################################################################

            sample_ec_lst = [ec for ec in ptw[15][1] if instance[:, self.ec_id[ec]] != 0]
            sample_unique_ec_lst = [ec for ec in sample_ec_lst if ec in unique_ec_lst]
            sample_orphan_ec_lst = [ec for ec in sample_ec_lst if ec in orphan_ec_lst]
            if true_prev_ec_lst:
                sample_initial_ec_lst = [1 for ec in true_prev_ec_lst[:initial_reaction] if ec not in sample_ec_lst]
            if true_succ_ec_lst:
                sample_final_ec_lst = [1 for ec in true_succ_ec_lst[:last_reaction] if ec not in sample_ec_lst]

            #######################################################################################################
            ############################## Extracting Features from Experimental Data #############################
            #######################################################################################################

            # 0. energy-pathway-mostly-missing (boolean)
            if 'energy' in text:
                if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                    m_features[self.pathway_id[id], 0] = 1

            # 1. mostly-present (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                m1 = len(sample_ec_lst) + 1 == len(ptw[15][1])
                m2 = (len(sample_ec_lst) / len(ptw[15][1])) >= threshold
                if m1 and m2:
                    m_features[self.pathway_id[id], 1] = 1

            # 2.   some-initial-reactions-present (boolean)
            if sample_initial_ec_lst:
                m_features[self.pathway_id[id], 2] = 1

            # 3.  all-initial-reactions-present (boolean)
            if sample_initial_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[self.pathway_id[id], 3] = 1

            # 4.   some-final-reactions-present (boolean)
            if sample_final_ec_lst:
                m_features[self.pathway_id[id], 4] = 1

            # 5.   all-final-reactions-present (boolean)
            if sample_final_ec_lst:
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[self.pathway_id[id], 5] = 1

            # 6.   some-initial-and-final-reactions-present (boolean)
            if sample_initial_ec_lst and sample_final_ec_lst:
                m_features[self.pathway_id[id], 6] = 1

            # 7.   all-initial-and-final-reactions-present  (boolean)
            if sample_initial_ec_lst and sample_final_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]) and len(
                        sample_final_ec_lst) == len(
                    true_succ_ec_lst[:last_reaction]):
                    m_features[self.pathway_id[id], 7] = 1

            # 8.   deg-or-detox-pathway-all-initial-reactions-present (boolean)
            # 9.   deg-or-detox-pathway-all-final-reactions-present (boolean)
            if 'detoxification' in text or 'degradation' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[self.pathway_id[id], 8] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[self.pathway_id[id], 9] = 1

            # 10.   biosynthesis-pathway-all-initial-reactions-present (boolean)
            # 11.   biosynthesis-pathway-all-final-reactions-present (boolean)
            if 'biosynthesis' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initial_reaction]):
                    m_features[self.pathway_id[id], 10] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:last_reaction]):
                    m_features[self.pathway_id[id], 11] = 1

            # 12.  mostly-absent (boolean)
            if len(sample_ec_lst) == 0:
                m_features[self.pathway_id[id], 12] = 1
            elif len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    m_features[self.pathway_id[id], 12] = 1

            # 13.  mostly-absent-not-unique (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                if unique_ec_lst:
                    m_features[self.pathway_id[id], 13] = 1
                    for e in unique_ec_lst:
                        if e in sample_ec_lst:
                            m_features[self.pathway_id[id], 13] = 0
                            break

            # 14.  one-reaction-present-but-in-minority (boolean)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    m_features[self.pathway_id[id], 14] = 1

            # 15.  every-unique-reaction-present (boolean)
            # 16.  every-reaction-present (boolean)
            # 17.  every-unique-reaction-present-or-orphaned (boolean)
            # 18.  every-reaction-present-or-orphaned (boolean)
            if sample_ec_lst:
                m_features[self.pathway_id[id], 15] = 1
                m_features[self.pathway_id[id], 16] = 1
                m_features[self.pathway_id[id], 17] = 1
                m_features[self.pathway_id[id], 18] = 1
                if sample_ec_lst != unique_ec_lst:
                    m_features[self.pathway_id[id], 15] = 0
                if sample_ec_lst != true_prev_ec_lst:
                    m_features[self.pathway_id[id], 16] = 0
                u = 0
                for ec in unique_ec_lst:
                    if ec in sample_unique_ec_lst or ec in orphan_ec_lst:
                        u += 1
                if u != len(unique_ec_lst):
                    m_features[self.pathway_id[id], 17] = 0
                a = 0
                for ec in true_prev_ec_lst:
                    if ec in sample_ec_lst or ec in orphan_ec_lst:
                        a += 1
                if a != len(true_prev_ec_lst):
                    m_features[self.pathway_id[id], 18] = 0

            # 19.  majority-of-reactions-absent (boolean)
            # 20.  majority-of-reactions-present (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                m_features[self.pathway_id[id], 19] = 1
            else:
                m_features[self.pathway_id[id], 20] = 1

            # 21.  majority-of-unique-reactions-present (boolean)
            if len(sample_unique_ec_lst) > int(threshold * len(unique_ec_lst)):
                m_features[self.pathway_id[id], 21] = 1

            # 22.  majority-of-reactions-present-unique (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                m_features[self.pathway_id[id], 22] = 1
                for ec in sample_ec_lst:
                    if ec not in unique_ec_lst:
                        m_features[self.pathway_id[id], 22] = 0
                        break

            # 23.  missing-at-most-one-reaction (boolean)
            if len(sample_ec_lst) + 1 == len(true_prev_ec_lst):
                ec = set.difference(set(sample_ec_lst), set(true_prev_ec_lst))
                if ec not in orphan_ec_lst:
                    m_features[self.pathway_id[id], 23] = 1

            # 24.  has-unique-reactions-present (boolean)
            if sample_ec_lst:
                if sample_unique_ec_lst:
                    m_features[self.pathway_id[id], 24] = 1

            # 25.  has-reactions-present (boolean)
            if sample_ec_lst:
                m_features[self.pathway_id[id], 25] = 1

            # 26.  fraction-final-reactions-present (numeric)
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 26] = len(sample_final_ec_lst) / len(true_prev_ec_lst)

            # 27.  fraction-initial-reactions-present (numeric)
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 27] = len(sample_initial_ec_lst) / len(true_prev_ec_lst)

            # 28.  num-final-reactions-present (numeric)
            m_features[self.pathway_id[id], 28] = len(sample_final_ec_lst)

            # 29.  num-initial-reactions-present (numeric)
            m_features[self.pathway_id[id], 29] = len(sample_initial_ec_lst)

            # 30.  fraction-unique-reactions-present-or-orphaned (numeric)
            sample_rxn_unique_orphand = set.union(set(sample_unique_ec_lst), set(orphan_ec_lst))
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 30] = len(sample_rxn_unique_orphand) / len(true_prev_ec_lst)

            # 31.  fraction-reactions-present-or-orphaned-unique (numeric)
            if len(unique_ec_lst):
                m_features[self.pathway_id[id], 31] = len(sample_rxn_unique_orphand) / len(unique_ec_lst)

            # 32.  fraction-reactions-present-or-orphaned (numeric)
            sample_rxn_orphand = set.union(set(sample_ec_lst), set(orphan_ec_lst))
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 32] = len(sample_rxn_orphand) / len(true_prev_ec_lst)

            # 33.  fraction-unique-reactions-present (numeric)
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 33] = len(sample_unique_ec_lst) / len(true_prev_ec_lst)

            # 34.  fraction-reactions-present-unique (numeric)
            if len(unique_ec_lst):
                m_features[self.pathway_id[id], 34] = len(sample_unique_ec_lst) / len(unique_ec_lst)

            # 35.  fraction-reactions-present (numeric)
            if len(true_prev_ec_lst):
                m_features[self.pathway_id[id], 35] = len(sample_ec_lst) / len(true_prev_ec_lst)

            # 36.  num-unique-reactions-present-or-orphaned (numeric)
            m_features[self.pathway_id[id], 36] = len(sample_rxn_unique_orphand)

            # 37.  num-reactions-present-or-orphaned (numeric)
            m_features[self.pathway_id[id], 37] = len(sample_rxn_orphand)

            # 38.  num-unique-reactions-present (numeric)
            m_features[self.pathway_id[id], 38] = len(sample_unique_ec_lst)

            # 39.  num-reactions-present (numeric)
            m_features[self.pathway_id[id], 39] = len(sample_ec_lst)

            # 40.  evidence-info-content-norm-present (numeric)
            total = 0
            for ec in set(sample_ec_lst):
                total += 1 / np.count_nonzero(ptwy_ec_matrix[:, self.ec_id[ec]])
            if sample_ec_lst:
                m_features[self.pathway_id[id], 40] = total / len(sample_ec_lst)

            # 41.  evidence-info-content-unnorm-present (numeric)
            m_features[self.pathway_id[id], 41] = total

        return m_features
