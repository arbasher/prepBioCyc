__author__ = "Abdurrahman Abul-Basher"
__date__ = '03/01/2019'
__copyright__ = "Copyright 2019, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman Abul-Basher"
__email__ = "arbasher@student.ubc.ca"
__status__ = "Production"
__description__ = "This is the main entry to extract various information from BioCyc database."

import datetime
import json
import os, sys
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from parse_biocyc import biocyc_main
from utility.arguments import Arguments


def __print_header():
    if sys.platform.startswith('win'): 
        os.system("cls")
    else:
        os.system("clear")
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__, "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45, subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()
    arg.num_jobs = parse_args.num_jobs
    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1

    ###***************************          Path arguments          ***************************###

    arg.kbpath = parse_args.kbpath
    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.featpath = fph.FEATURE_PATH

    ###************************          File name arguments          *************************###

    arg.object_name = 'biocyc.pkl'
    arg.reaction2ec = 'reaction2ec.pkl'
    arg.reaction2ec_idx = 'reaction2ec_idx.pkl'
    arg.pathway2ec = 'pathway2ec.pkl'
    arg.pathway2ec_idx = 'pathway2ec_idx.pkl'
    arg.pathway_feature = 'pathwayfeature.pkl'
    arg.ec_feature = 'ecfeature.pkl'
    arg.pathway_similarity = 'pathway_similarity'
    arg.synset_file_name = 'synset_ptwy_ec'
    arg.golden_file_name = 'goldset_ptwy_ec'

    ###**************       Building database object and features arguments       *************###

    arg.build_biocyc_object = parse_args.build_biocyc_object
    arg.build_indicator = parse_args.build_indicator
    arg.build_pathway_properties = parse_args.build_pathway_properties
    arg.build_ec_properties = parse_args.build_ec_properties
    arg.build_pathway_similarities = parse_args.build_pathway_similarities

    ###*************************       Building graph arguments       *************************###

    arg.build_graph = parse_args.build_graph
    arg.constraint_kb = parse_args.constraint_kb
    arg.filter_compound_graph = parse_args.filter_compound_graph

    ###******************       Constructing synthetic data arguments       *******************###
    arg.build_synset = parse_args.build_synset
    arg.ex_features_from_synset = parse_args.ex_features_from_synset
    arg.build_golden_dataset = parse_args.build_golden_dataset
    arg.ex_features_from_golden_dataset = parse_args.ex_features_from_golden_dataset
    arg.num_sample = parse_args.num_sample
    arg.add_noise = True
    if parse_args.no_noise:
        arg.add_noise = False
    arg.num_components_to_corrupt = parse_args.num_components_to_corrupt
    arg.lower_bound_num_item_ptwy = parse_args.lower_bound_num_item_ptwy
    arg.num_components_to_corrupt_outside = parse_args.num_components_to_corrupt_outside
    arg.average_item_per_sample = parse_args.average_item_per_sample
    arg.build_minpath_dataset = parse_args.build_minpath_dataset
    arg.minpath_map = parse_args.minpath_map
    arg.build_pathologic_input = parse_args.build_pathologic_input

    ###***********************       Extracting features arguments       **********************###

    arg.num_pathway_features = 34
    arg.num_ec_features = 25
    arg.num_mol_features = 18
    arg.num_reaction_evidence_features = 42
    arg.num_ec_evidence_features = 68
    arg.num_ptwy_evidence_features = 32

    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run prepBioCyc.")

    parser.add_argument('--display-interval', type=int, default=1,
                        help='display intervals. -1 means display per each '
                             'iteration. (default value: 1).')
    parser.add_argument('--num-jobs', type=int, default=2,
                        help='Number of parallel workers. (default value: 2).')

    # Arguments for path
    parser.add_argument('--kbpath', type=str, default=fph.DATABASE_PATH,
                        help='The path to the BioCyc databases. '
                             'The default is set to the database folder outside '
                             'the source code.')
    parser.add_argument('--ospath', type=str, default=fph.OBJECT_PATH,
                        help='The path to the data object that contains extracted '
                             'information from the BioCyc databases. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--dspath', type=str, default=fph.DATASET_PATH,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')

    # Arguments for building data object and extracting features
    parser.add_argument('--build-biocyc-object', action='store_true', default=False,
                        help='Whether to build reaction in the mapping process. '
                             'process. (default value: False).')
    parser.add_argument('--build-indicator', action='store_true', default=False,
                        help='Whether to build reaction in the mapping process. '
                             'process. (default value: False).')
    parser.add_argument('--build-pathway-properties', action='store_true', default=False,
                        help='Whether to build reaction in the mapping process. '
                             'process. (default value: False).')
    parser.add_argument('--build-ec-properties', action='store_true', default=False,
                        help='Whether to build reaction in the mapping process. '
                             'process. (default value: False).')
    parser.add_argument('--build-pathway-similarities', action='store_true', default=False,
                        help='Whether to build reaction in the mapping process. '
                             'process. (default value: False).')

    # Arguments for graph preprocessing
    parser.add_argument('--build-graph', action='store_true', default=False,
                        help='Whether to build gene, EC, and pathway graphs. (default value: False).')
    parser.add_argument('--filter-compound-graph', action='store_true', default=False,
                        help='Whether to filter compound graph in the graph construction '
                             'process. (default value: False).')
    parser.add_argument('--constraint-kb', type=str, default='metacyc',
                        help='Building graphs, synthetic samples, and golden data are '
                             'constraint on a given knowledge base. (default value: metacyc).')

    # Arguments for constructing synthetic dataset
    parser.add_argument('--build-synset', action='store_true', default=False,
                        help='Whether to construct simulated samples. (default value: False).')
    parser.add_argument('--build-golden-dataset', action='store_true', default=False,
                        help='Whether to construct golden data. (default value: False).')
    parser.add_argument('--ex-features-from-synset', action='store_true', default=False,
                        help='Whether to extract features from generated simulated '
                             'samples. (default value: False).')
    parser.add_argument('--ex-features-from-golden-dataset', action='store_true', default=False,
                        help='Whether to extract features from generated golden '
                             'data. (default value: False).')
    parser.add_argument('--build-pathologic-input', action='store_true', default=False,
                        help='Whether to construct pathologic input file (*.pf) from golden data. '
                             '(default value: False).')
    parser.add_argument('--build-minpath-dataset', action='store_true', default=False,
                        help='Whether to construct MinPath input file (*.txt) from golden data. '
                             '(default value: False).')
    parser.add_argument('--minpath-map', action='store_true', default=False,
                        help='Whether to create reference mapping file for MinPath tool. '
                             '(default value: False).')
    parser.add_argument('--num-sample', type=int, default=50,
                        help='The size of simulated samples. (default value: 50).')
    parser.add_argument('--no-noise', action='store_true', default=False,
                        help='Whether to add noise in constructing simulated samples. '
                             '(default value: False).')
    parser.add_argument('--num-components-to-corrupt', type=int, default=2,
                        help='Number of corrupted components for each true representation '
                             'of pathway. (default value: 2).')
    parser.add_argument('--lower-bound-num-item-ptwy', type=int, default=1,
                        help='the corruption process is constrained to only those pathways '
                             'that have more than this number of ECs. (default value: 1).')
    parser.add_argument('--num-components-to-corrupt-outside', type=int, default=2,
                        help='Number of ECs to be corrupted by inserting false ECs to each '
                             'true representation of a pathway. (default value: 2).')
    parser.add_argument('--average-item-per-sample', type=int, default=500,
                        help='The number of items (e.g. pathways) for each '
                             'synthetic sample. (default value: 500).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    biocyc_main(arg=args)


if __name__ == '__main__':
    parse_command_line()
