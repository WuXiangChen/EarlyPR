
'''
   导包区
'''

import sys
import os

script_directory_path = "../"
sys.path.append(script_directory_path)
sys.path.append("DataPreproAndModelExecutor")
sys.path.append("datasets")
sys.path.append("CONSTANT")


root_path = 'datasets/data//'

saved_model_path = os.path.join(root_path, "prRelated_data_Model")


repo_owner_name_Filepath = "CONSTANT/repo_data/repo_owner_name.csv"

Test_ownerSha = "datasets/data/Test_auxiliary_information/OwnerShas_for_generate_SequentialtestData/*.csv"


train_batch_size = 16

CB_max_Output_length = 256
