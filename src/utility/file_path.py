import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-2])

FEATURE_PATH = os.path.join(DIRECTORY_PATH, 'feature_builder')

DATABASE_PATH = os.path.join(REPO_PATH, 'database', 'biocyc-flatfiles')
OBJECT_PATH = os.path.join(REPO_PATH, 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'dataset')
