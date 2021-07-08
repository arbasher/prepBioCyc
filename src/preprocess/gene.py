'''
This file preprocesses two gene files: genes.dat and genes.col
from BioCyc PGDBs to an appropriate format that is could be used
as inputs to designated machine learning models.
'''

import os
import os.path
from collections import OrderedDict


class Gene(object):
    def __init__(self, gene_col_file_name='genes.col', gene_dat_file_name='genes.dat'):
        """ Initialization
        :type gene_col_file_name: str
        :param gene_col_file_name: file name for the genes.col, containing id for enzymatic-reactions
        :type gene_dat_file_name: str
        :param gene_dat_file_name: file name for the genes.dat, containing id for enzymatic-reactions
        """
        self.gene_col_file_name = gene_col_file_name
        self.gene_dat_file_name = gene_dat_file_name
        self.gene_info = OrderedDict()

    def process_genes(self, gene_idx_list_ids, gene_name_idx_list_ids, product_idx_list_ids, list_ids, data_path,
                      header=False):
        self.__process_genes_dat_file(gene_idx_list_ids=gene_idx_list_ids, g_name_idx_list_ids=gene_name_idx_list_ids,
                                      product_idx_list_ids=product_idx_list_ids, list_ids=list_ids, data_path=data_path)
        self.__process_genes_col_file(g_idx_list_ids=gene_idx_list_ids, g_name_idx_list_ids=gene_name_idx_list_ids,
                                      product_idx_list_ids=product_idx_list_ids, list_ids=list_ids, data_path=data_path,
                                      header=header)

    def __process_genes_dat_file(self, gene_idx_list_ids, g_name_idx_list_ids, product_idx_list_ids, list_ids,
                                 data_path):
        gene_file = os.path.join(data_path, self.gene_dat_file_name)
        if os.path.isfile(gene_file):
            print('\t\t\t--> Prepossessing genes database from: '
                  '{0}'.format(gene_file.split('/')[-1]))
            with open(gene_file, errors='ignore') as f:
                for text in f:
                    if not str(text).strip().startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                gene_id = ' '.join(ls[2:])
                                lst_gene_types = list()
                                gene_name_id = ''
                                gene_product = ''
                            elif ls[0] == 'TYPES':
                                lst_gene_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMMON-NAME':
                                gene_name_id = ''.join(ls[2:])
                                if gene_name_id:
                                    if gene_name_id not in list_ids[g_name_idx_list_ids]:
                                        list_ids[g_name_idx_list_ids].update(
                                            {gene_name_id: len(list_ids[g_name_idx_list_ids])})
                            elif ls[0] == 'PRODUCT':
                                gene_product = ''.join(ls[2].split('|'))
                                if gene_product not in list_ids[product_idx_list_ids]:
                                    list_ids[product_idx_list_ids].update(
                                        {gene_product: len(list_ids[product_idx_list_ids])})
                            elif ls[0] == '//':
                                if gene_id not in list_ids[gene_idx_list_ids]:
                                    list_ids[gene_idx_list_ids].update({gene_id: len(list_ids[gene_idx_list_ids])})

                                if gene_id not in self.gene_info:
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {gene_id: (['TYPES', lst_gene_types],
                                                       ['NAME', gene_name_id],
                                                       ['PRODUCT', gene_product],
                                                       ['PRODUCT-NAME', ''],
                                                       ['SWISS-PROT-ID', ''],
                                                       ['REACTION-LIST', []],
                                                       ['GO-TERMS', []])}
                                    self.gene_info.update(datum)

    def __process_genes_col_file(self, g_idx_list_ids, g_name_idx_list_ids, product_idx_list_ids, list_ids, data_path,
                                 header=False):
        gene_file = os.path.join(data_path, self.gene_col_file_name)
        if os.path.isfile(gene_file):
            print('\t\t\t--> Prepossessing genes database from: {0}'
                  .format(gene_file.split('/')[-1]))
            with open(gene_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'UNIQUE-ID':
                                    header = True
                                    gene_id = ls.index('UNIQUE-ID')
                                    gene_name_id = ls.index('NAME')
                                    gene_product_name = ls.index('PRODUCT-NAME')
                                    gene_swiss_prot_id = ls.index('SWISS-PROT-ID')
                            else:
                                if ls[gene_id] not in list_ids[g_idx_list_ids]:
                                    list_ids[g_idx_list_ids].update({ls[gene_id]: len(list_ids[g_idx_list_ids])})

                                if ls[gene_name_id]:
                                    if ls[gene_name_id] not in list_ids[g_name_idx_list_ids]:
                                        list_ids[g_name_idx_list_ids].update(
                                            {ls[gene_name_id]: len(list_ids[g_name_idx_list_ids])})

                                if ls[gene_id] not in self.gene_info:
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {ls[gene_id]: (['TYPES', []],
                                                           ['NAME', ls[gene_name_id]],
                                                           ['PRODUCT', ''],
                                                           ['PRODUCT-NAME', ls[gene_product_name]],
                                                           ['SWISS-PROT-ID', ls[gene_swiss_prot_id]],
                                                           ['REACTION-LIST', []],
                                                           ['GO-TERMS', []])}
                                    self.gene_info.update(datum)
                                else:
                                    (t_gene_types, t_gene_name_id, t_gene_product, t_gene_product_name,
                                     t_gene_swiss_prot_id, lst_catalyzes, lst_go) = self.gene_info[ls[gene_id]]
                                    if not t_gene_name_id[1]:
                                        t_gene_name_id[1] = ls[gene_name_id]
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {ls[gene_id]: (t_gene_types,
                                                           t_gene_name_id,
                                                           t_gene_product,
                                                           ['PRODUCT-NAME', ls[gene_product_name]],
                                                           ['SWISS-PROT-ID', ls[gene_swiss_prot_id]],
                                                           lst_catalyzes,
                                                           lst_go)}
                                    self.gene_info.update(datum)

    def add_protein_info(self, protein_info, go_position_idx, catalyzes_position_idx,
                         product_name_position_idx, product_idx_list_ids, list_ids):
        print('\t\t\t--> Adding proteins information to genes')
        for (gid, g_item) in self.gene_info.items():
            (t_gene_types, t_gene_name_id, t_gene_product, t_gene_product_name,
             t_gene_swiss_prot_id, lst_catalyzes, lst_go) = g_item
            if t_gene_product[1] in protein_info:
                if protein_info[t_gene_product[1]][catalyzes_position_idx][1]:
                    lst_catalyzes[1].extend(protein_info[t_gene_product[1]][catalyzes_position_idx][1])
                if protein_info[t_gene_product[1]][go_position_idx][1]:
                    lst_go[1].extend(protein_info[t_gene_product[1]][go_position_idx][1])
                if not t_gene_product_name[1]:
                    product_name = protein_info[t_gene_product[1]][product_name_position_idx][1]
                    if product_name not in list_ids[product_idx_list_ids] and product_name != '':
                        list_ids[product_idx_list_ids].update({product_name: len(list_ids[product_idx_list_ids])})
                else:
                    product_name = t_gene_product_name[1]
                datum = {gid: (t_gene_types,
                               t_gene_name_id,
                               t_gene_product,
                               ['PRODUCT-NAME', product_name],
                               t_gene_swiss_prot_id,
                               lst_catalyzes,
                               lst_go)}
                self.gene_info.update(datum)

    def add_pathway_info(self, pathway_info, gene_name_id_position_idx, gene_id_position_idx,
                         gene_idx_list_ids, gene_name_idx_list_ids, list_ids):
        print('\t\t\t--> Adding additional genes to gene id and gene name id from pathways genes')
        for (p_id, p_item) in pathway_info.items():
            if p_item[gene_id_position_idx][1]:
                for g in p_item[gene_id_position_idx][1]:
                    if g not in list_ids[gene_idx_list_ids]:
                        list_ids[gene_idx_list_ids].update({g: len(list_ids[gene_idx_list_ids])})
                        self.gene_info.update({g: (['TYPES', []],
                                                   ['NAME', ''],
                                                   ['PRODUCT', ''],
                                                   ['PRODUCT-NAME', ''],
                                                   ['SWISS-PROT-ID', ''],
                                                   ['REACTION-LIST', []],
                                                   ['GO-TERMS', []])})
                for g in p_item[gene_name_id_position_idx][1]:
                    if g not in list_ids[gene_name_idx_list_ids]:
                        list_ids[gene_name_idx_list_ids].update({g: len(list_ids[gene_name_idx_list_ids])})

    def add_reaction_info(self, reaction_info, gene_name_id_position_idx, gene_name_id, list_ids):
        print('\t\t\t--> Adding additional genes to gene name id from reactions genes')
        for (r_id, r_item) in reaction_info.items():
            if r_item[gene_name_id_position_idx][1]:
                for g in r_item[gene_name_id_position_idx][1]:
                    if g not in list_ids[gene_name_id]:
                        list_ids[gene_name_id].update({g: len(list_ids[gene_name_id])})
