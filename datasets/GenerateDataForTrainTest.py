
import pathlib
import torch
from datasets.dataset_utils import *
from CONSTANT.CONSTANT import root_path
import os
import glob
from sklearn.preprocessing import MinMaxScaler

class GenerateDataForTrainTest:
    def __init__(self, data_file, repo_owner_name_Filepath, Test_ownerSha, max_input_length, tokenizer):
        self.data_file = data_file + ".csv"
        self.data_file = os.path.join(root_path, self.data_file)


        self.max_input_length = max_input_length

        self.tokenizer = tokenizer
        self.repo_owner_name_Filepath = repo_owner_name_Filepath
        self.save_path = Test_ownerSha
        self.csv_values = pd.read_csv(repo_owner_name_Filepath)
        self.Test_ownerShas_ls = glob.glob(self.save_path)
        self.generate_data()

    def generate_data(self):

        print(pathlib.Path.cwd())
        if not os.path.exists(self.data_file):
            raise FileNotFoundError("Data path not exists")

        data = pd.read_csv(self.data_file)
        data['mergedPR'].fillna(-1, inplace=True)
        data.insert(1, 'mergedPR', data.pop("mergedPR"))
        data.rename(columns={"sha": "commits", "mergedPR": "merged_PR"}, inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

        repo_owner = {}
        repo_owner["repo_name"] = self.data_file.split("/")[-1].split(".csv")[0]
        repo_owner["owner_name"] = \
            (self.csv_values[self.csv_values["repo_names"] == repo_owner["repo_name"]]["owner_names"].iloc)[0]
        repo_owner["PR_start_time"] = \
        self.csv_values[self.csv_values["repo_names"] == repo_owner["repo_name"]]["PR_start_time"].iloc[0]

        data.dropna(axis=0, inplace=True)
        columns_to_normalize = ["#Churn", "#Commits", "#FilesChanged", "NumOverlapfiles",
                                "RatePrCommits","TwoCommit","last_10_merged","last_10_rejected","prNum"]
        for column in columns_to_normalize:
            data[column] = normalize_column(data, column)
        self.data = data


        self.MP_X = data[data['label'] == 1].iloc[:, 1:]
        self.MP_y = data[data['label'] == 1].iloc[:, 1]
        self.P_X = data.iloc[:, 1:]
        self.P_y = data.iloc[:, 0]

        P_group_id = self.P_X["group_id"].drop_duplicates().reset_index(drop=True)
        group_id = P_group_id.sample(frac=1, random_state=42).reset_index(drop=True)
        len_ = int(len(group_id) / 10)
        self.Pgroup_test_IDindex = [group_id[i * len_:i * len_ + len_] for i in range(10)]

        MP_group_id = self.MP_X["group_id"].drop_duplicates().reset_index(drop=True)
        group_id = MP_group_id.sample(frac=1, random_state=42).reset_index(drop=True)
        len_ = int(len(group_id) / 10)
        self.MPgroup_test_IDindex = [group_id[i * len_:i * len_ + len_] for i in range(10)]


    def generate_PRtraintest_data(self,trainFlag=True):

        test_groupId = self.Pgroup_test_IDindex[-1]
        if trainFlag:
            Tmp_data = self.data[~self.data["group_id"].isin(test_groupId)]
            Tmp_data = self.balance_traingData(Tmp_data)
            Px_ = Tmp_data.iloc[:, 5:-1]
            Py_ = Tmp_data.iloc[:, 0]
            return Px_, Py_
        else:

            Tmp_data = FilterCommits(self.P_X,'group_id',50)
            index = pd.Series(Tmp_data[self.P_X["group_id"].isin(test_groupId)].index)
            Px_ = self.P_X.loc[index].iloc[:, 0:4]
            Py_ = self.P_y.loc[index]
            return Px_, Py_

    def generate_MPRtraintest_data(self,trainFlag=True):
        test_groupId = self.MPgroup_test_IDindex[-1]
        if trainFlag:
            index = pd.Series(self.MP_X[~self.MP_X["group_id"].isin(test_groupId)].index)
            Temp_data = pd.concat([self.MP_X.loc[index], self.MP_y.loc[index]], axis=1)

            MPx_ = Temp_data.iloc[:, 4:-1]
            MPy_ = Temp_data.iloc[:,0]
            return MPx_, MPy_
        else:
            index = pd.Series(self.MP_X[self.MP_X["group_id"].isin(test_groupId)].index)
            MPx_ = self.MP_X.loc[index].iloc[:, 0:4]
            MPy_ = self.MP_y.loc[index]
            return MPx_, MPy_

    def balance_traingData(self, data,label_column='label', balance_ratio=1.1, positive_label=1, negative_label=0):
        positive_count = (data[label_column] == positive_label).sum()
        negative_count = (data[label_column] == negative_label).sum()

        if max(positive_count, negative_count) / min(positive_count, negative_count) <= balance_ratio:
            return data


        target_count = int(min(positive_count, negative_count) * balance_ratio)

        balanced_data = (pd.concat([
            data[data[label_column] == positive_label].sample(target_count, replace=True),
            data[data[label_column] == negative_label].sample(target_count, replace=True)
        ]))
        balanced_data.reset_index(inplace=True,drop=True)

        return balanced_data

    def TokenizeMsgAndCodeDiff(self, msg, codediff):
        nlnl_tokenizer = self.tokenize_nlnl(msg)
        nlpl_tokenizer = self.tokenize_nlpl(codediff)

        padding_length = self.max_input_length - nlnl_tokenizer['input_ids'].shape[1] - \
                         nlpl_tokenizer['input_ids'].shape[1] + 1  # '+1' for [CLS] of pr_tokens

        paddings_tokens: dict = {
            'input_ids': torch.full((padding_length,), self.tokenizer.pad_token_id),
            'attention_mask': torch.full((padding_length,), 0),
        }
        res_tokens = {
            'input_ids': torch.cat(
                (
                    nlnl_tokenizer['input_ids'][0, :],
                    nlpl_tokenizer['input_ids'][0, 1:], paddings_tokens['input_ids'])
            ),  # [1:] to remove [CLS] of pr_tokens
            'attention_mask': torch.cat(
                (
                    nlnl_tokenizer['attention_mask'][0, :],
                    nlpl_tokenizer['attention_mask'][0, 1:], paddings_tokens['attention_mask'])
            ),
        }
        return res_tokens

    def tokenize_nlpl(self, codediff):
        codediff = ",".join(codediff)
        codediff_tokens = self.tokenizer(
            codediff,
            max_length=self.max_input_length//2,
            padding=False,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )

        return codediff_tokens

    def tokenize_nlnl(self,  msg):
        msg = ",".join(msg)
        msg_tokens = self.tokenizer(
            msg,
            max_length=self.max_input_length//2 + 1,
            padding=False,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True
        )

        return msg_tokens

