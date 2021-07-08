'''
This file preprocesses two enzymes files: enzymes.dat and
enzrxns.dat from BioCyc PGDBs to an appropriate format that
is could be used as inputs to designated machine learning
models.
'''

import os
import os.path
from collections import OrderedDict


class Enzyme(object):
    def __init__(self, enzyme_col_file_name='enzymes.col', enzrxn_dat_file_name='enzrxns.dat'):
        """ Initialization
        :type fname_enzymes: str
        :param fname_enzymes: file name for the enzyme.col, containing id for enzymatic-reactions
        :type fname_enzrxns: str
        :param fname_enzrxns: file name for the enzrxns.dat, containing id for enzymatic-reactions
        """
        self.enzyme_col_file_name = enzyme_col_file_name
        self.enzrxn_dat_file_name = enzrxn_dat_file_name
        self.enzyme_info = OrderedDict()

    def process_enzymes(self, enzyme_position_idx, list_ids, data_path, header=False):
        self.__process_enzymes_col(e_id=enzyme_position_idx, list_ids=list_ids, data_path=data_path, header=header)
        self.__process_enzymatic_reactions_dat(e_id=enzyme_position_idx, list_ids=list_ids, data_path=data_path)

    def __process_enzymes_col(self, e_id, list_ids, data_path, header=False):
        enzyme_file = os.path.join(data_path, self.enzyme_col_file_name)
        if os.path.isfile(enzyme_file):
            print('\t\t\t--> Prepossessing enzymes database from: {0}'.format(enzyme_file.split('/')[-1]))
            with open(enzyme_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'UNIQUE-ID':
                                    header = True
                                    enzyme_id = ls.index('UNIQUE-ID')
                                    enzyme_name_id = ls.index('NAME')
                            else:
                                if not ls[enzyme_id] in list_ids[e_id]:
                                    list_ids[e_id].update({ls[enzyme_id]: len(list_ids[e_id])})

                                if not ls[enzyme_id] in self.enzyme_info:
                                    # datum is comprised of {UNIQUE-ID: NAME}
                                    datum = {ls[enzyme_id]:
                                                 ['NAME', ls[enzyme_name_id]]}
                                    self.enzyme_info.update(datum)

    def __process_enzymatic_reactions_dat(self, e_id, list_ids, data_path):
        enzyme_file = os.path.join(data_path, self.enzrxn_dat_file_name)
        if os.path.isfile(enzyme_file):
            print('\t\t\t--> Prepossessing additional enzymatic reactions database from: '
                  '{0}'.format(enzyme_file.split('/')[-1]))
            with open(enzyme_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                enzrxn_id = ' '.join(ls[2:])
                                enzrxn_name = ''
                            elif ls[0] == 'COMMON-NAME':
                                enzrxn_name = ' '.join(ls[2:])
                            elif ls[0] == '//':
                                if not enzrxn_id in list_ids[e_id]:
                                    list_ids[e_id].update({enzrxn_id: len(list_ids[e_id])})
                                if not enzrxn_id in self.enzyme_info:
                                    # datum is comprised of {UNIQUE-ID: COMMON-NAME}
                                    datum = {enzrxn_id:
                                                 ['NAME', enzrxn_name]}
                                    self.enzyme_info.update(datum)
