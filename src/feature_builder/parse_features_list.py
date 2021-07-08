import os.path
import sys
import traceback


def extract_features_names(path=os.getcwd(), file_name='features_list.txt', print_feats=False,
                           tag='a list of features name'):
    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file = os.path.join(path, file_name)
        dict_features = dict()
        with open(file, 'r') as f_in:
            list_features = list()
            for data in f_in:
                if not data.startswith('#'):
                    if print_feats:
                        print(data.strip())
                    if len(list_features) != 0:
                        dict_features.update({feature_name.split('.')[1].strip(): list_features})
                    feature_name = data.strip()
                    list_features = list()
                elif data.startswith('#'):
                    feature = data.split('#')[1]
                    if print_feats:
                        print('\item \\textbf{' + feature.split('.')[1].strip() + '}')
                    list_features.append(feature.split('.')[1].split(' ')[1].strip())
        dict_features.update({feature_name.split('.')[1].strip(): list_features})
        return dict_features
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file), file=sys.stderr)
        print(traceback.print_exc())
        raise e
