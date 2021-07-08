'''
This file preprocesses the compounds.dat from BioCyc PGDBs to
an appropriate format that is could be used as inputs to
designated machine learning models.
'''

import os
import os.path
from collections import OrderedDict


class Compound(object):
    def __init__(self, file_name_compound='compounds.dat'):
        """ Initialization
        :type fname_reaction: str
        :param fname_reaction: file name for the proteins
        """
        self.compound_dat_file_name = file_name_compound
        self.compound_info = OrderedDict()

    def process_compounds(self, compound_idx_list_ids, list_ids, data_path):
        compound_file = os.path.join(data_path, self.compound_dat_file_name)
        if os.path.isfile(compound_file):
            print('\t\t\t--> Prepossessing compounds database from: {0}'.format(compound_file.split('/')[-1]))
            with open(compound_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                compound_id = ' '.join(ls[2:])
                                compound_name = ''
                                compound_types = list()
                                atom_charges = list()
                                chemical_formula = list()
                                cofactors_of = list()
                                component_of = list()
                                components = list()
                                group_coords_2d = list()
                                group_internals = list()
                                has_no_structure = False
                                in_mixture = list()
                                inchi = list()
                                inchi_key = list()
                                internals_of_group = list()
                                molecular_weight = 0.0
                                monoisotopic_mw = 0.0
                                n_p1_name = list()
                                n_m1_name = list()
                                n_name = list()
                                non_standard_inchi = list()
                                pka1 = list()
                                pka2 = list()
                                pka3 = list()
                                radical_atoms = list()
                                regulates = list()
                                smiles = ''
                                species = list()
                                structure_groups = list()
                                structure_links = list()
                                superatoms = list()
                                synonyms = list()
                                systematic_name = ''
                                tautomers = list()
                            elif ls[0] == 'TYPES':
                                compound_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMMON-NAME':
                                compound_name = ' '.join(ls[2:])
                            elif ls[0] == 'ATOM-CHARGES':
                                atom_charges.append(' '.join(ls[2:]))
                            elif ls[0] == 'CHEMICAL-FORMULA':
                                chemical_formula.append(' '.join(ls[2:]))
                            elif ls[0] == 'COFACTORS-OF':
                                cofactors_of.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMPONENT-OF':
                                component_of.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMPONENTS':
                                components.append(' '.join(ls[2:]))
                            elif ls[0] == 'GROUP-COORDS-2D':
                                group_coords_2d.append(' '.join(ls[2:]))
                            elif ls[0] == 'GROUP-INTERNALS':
                                group_internals.append(' '.join(ls[2:]))
                            elif ls[0] == 'HAS-NO-STRUCTURE?':
                                has_no_structure = True
                            elif ls[0] == 'IN-MIXTURE':
                                in_mixture.append(' '.join(ls[2:]))
                            elif ls[0] == 'INCHI':
                                inchi.append(' '.join(ls[2:]))
                            elif ls[0] == 'INCHI-KEY':
                                inchi_key.append(' '.join(ls[2:]))
                            elif ls[0] == 'INTERNALS-OF-GROUP':
                                internals_of_group.append(' '.join(ls[2:]))
                            elif ls[0] == 'MOLECULAR-WEIGHT':
                                molecular_weight = float(' '.join(ls[2:]))
                            elif ls[0] == 'MONOISOTOPIC-MW':
                                monoisotopic_mw = float(' '.join(ls[2:]))
                            elif ls[0] == 'N+1-NAME':
                                n_p1_name.append(' '.join(ls[2:]))
                            elif ls[0] == 'N-1-NAME':
                                n_m1_name.append(' '.join(ls[2:]))
                            elif ls[0] == 'N-NAME':
                                n_name.append(' '.join(ls[2:]))
                            elif ls[0] == 'NON-STANDARD-INCHI':
                                non_standard_inchi.append(' '.join(ls[2:]))
                            elif ls[0] == 'PKA1':
                                pka1.append(' '.join(ls[2:]))
                            elif ls[0] == 'PKA2':
                                pka2.append(' '.join(ls[2:]))
                            elif ls[0] == 'PKA3':
                                pka3.append(' '.join(ls[2:]))
                            elif ls[0] == 'RADICAL-ATOMS':
                                radical_atoms.append(' '.join(ls[2:]))
                            elif ls[0] == 'REGULATES':
                                regulates.append(' '.join(ls[2:]))
                            elif ls[0] == 'SMILES':
                                smiles = ' '.join(ls[2:])
                            elif ls[0] == 'SPECIES':
                                species.append(' '.join(ls[2:]))
                            elif ls[0] == 'STRUCTURE-GROUPS':
                                structure_groups.append(' '.join(ls[2:]))
                            elif ls[0] == 'STRUCTURE-LINKS':
                                structure_links.append(' '.join(ls[2:]))
                            elif ls[0] == 'SUPERATOMS':
                                superatoms.append(' '.join(ls[2:]))
                            elif ls[0] == 'SYNONYMS':
                                synonyms.append(' '.join(ls[2:]))
                            elif ls[0] == 'SYSTEMATIC-NAME':
                                systematic_name = ' '.join(ls[2:])
                            elif ls[0] == 'TAUTOMERS':
                                tautomers.append(' '.join(ls[2:]))
                            elif ls[0] == '//':
                                if compound_id not in list_ids[compound_idx_list_ids]:
                                    list_ids[compound_idx_list_ids].update(
                                        {compound_id: len(list_ids[compound_idx_list_ids])})
                                if compound_id not in self.compound_info:
                                    # datum is comprised of {UNIQUE-ID: (List of attributes)}
                                    datum = {compound_id: (['COMMON-NAME', compound_name],
                                                           ['TYPES', compound_types],
                                                           ['ATOM-CHARGES', atom_charges],
                                                           ['CHEMICAL-FORMULA', chemical_formula],
                                                           ['COFACTORS-OF', cofactors_of],
                                                           ['COMPONENT-OF', component_of],
                                                           ['COMPONENTS', components],
                                                           ['GROUP-COORDS-2D', group_coords_2d],
                                                           ['GROUP-INTERNALS', group_internals],
                                                           ['HAS-NO-STRUCTURE?', has_no_structure],
                                                           ['IN-MIXTURE', in_mixture],
                                                           ['INCHI', inchi],
                                                           ['INCHI-KEY', inchi_key],
                                                           ['INTERNALS-OF-GROUP', internals_of_group],
                                                           ['MOLECULAR-WEIGHT', molecular_weight],
                                                           ['MONOISOTOPIC-MW', monoisotopic_mw],
                                                           ['N+1-NAME', n_p1_name],
                                                           ['N-1-NAME', n_m1_name],
                                                           ['N-NAME', n_name],
                                                           ['NON-STANDARD-INCHI', non_standard_inchi],
                                                           ['PKA1', pka1],
                                                           ['PKA2', pka2],
                                                           ['PKA3', pka3],
                                                           ['RADICAL-ATOMS', radical_atoms],
                                                           ['REGULATES', regulates],
                                                           ['SMILES', smiles],
                                                           ['SPECIES', species],
                                                           ['STRUCTURE-GROUPS', structure_groups],
                                                           ['STRUCTURE-LINKS', structure_links],
                                                           ['SUPERATOMS', superatoms],
                                                           ['SYNONYMS', synonyms],
                                                           ['SYSTEMATIC-NAME', systematic_name],
                                                           ['TAUTOMERS', tautomers]
                                                           )}
                                    self.compound_info.update(datum)
