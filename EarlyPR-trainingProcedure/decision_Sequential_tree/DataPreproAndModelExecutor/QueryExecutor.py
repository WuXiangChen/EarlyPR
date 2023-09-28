import json
from pprint import pprint

import numpy as np

from data.Test_auxiliary_information.querySets_GenerateTestCommits import _10_PRs_Files_JaccardSimilarity,_11_PRs_Files_JS_TopK,_2_statistics_pNCM,\
    _1_statistics_pNCM,_preWork_for_13_14_15_Query,_13_contributorFeatures,_14_refusedPR_FilesLS_For_calculate_file_rejected_proportion,\
    _15_last_10_mergedAndRejected_Andmost_recent, _10_negative_CM_JSFiles_within_1Mon
from generate_TestData.MongoDB_Related.MongoDB_utils import database
from decision_Sequential_tree.decision_Seq_utils import  find_element,recursive_replace,replace_value_recursive
from datapreprocessor import datapreprocessor
import pandas as pd
dp = datapreprocessor()


class queryexecutor:
    def __init__(self,repo_owner,extendFeatures, suffix="commits"):
        self.pre_query_preparation(repo_owner, suffix)
        self.redis_st = None # 该变量是为了配合query过程中的内存持久化而得，由modelex负责实例化
        self.mergedPR_predictor = None
        self.extendFeatures = extendFeatures

    def pre_query_preparation(self,repo_owner, suffix="commits"):
        coll_names = database.list_collection_names()
        repo_name = repo_owner["repo"]
        owner_name = repo_owner["owner"]
        PR_start_time = repo_owner["PR_start_time"]
        repo_name_INmongo = find_element(coll_names, repo_name, C=suffix)
        referenced_name = find_element(coll_names, repo_name, C="commits")
        replacement = {"owner": owner_name, "repo": repo_name, "referenced_name": referenced_name,
                       "PR_start_time":PR_start_time}
        coll = database[repo_name_INmongo]

        self.coll = coll
        self.replacement = replacement

        return coll, replacement

    def search_and_store(self, com):
        #com = ['890178e6557d7210387445d5830b5c8de0a81e66', '9727916cdf42b5b3c8013905ceaaf960ae888793', '1d06ecb7a94f14d037b73c5f7bb350f3ac4d65e3']
        str_com = ",".join(com)
        #redis_com_value = None
        redis_com_value = self.redis_st.get_value(str_com)
        coll, replacement = self.coll, self.replacement
        # 判断是否存在键，判断其存储的值是否为None
        if redis_com_value and json.loads(redis_com_value) != None and 'TwoCommit' in json.loads(redis_com_value):
            statistcsCom = json.loads(redis_com_value)
            if isinstance(statistcsCom['TwoCommit'],str):
                statistcsCom['TwoCommit'] = eval(statistcsCom['TwoCommit'])
            # 这里再做一下额外的判断JSfiles是否在其中，如果非的话 需要进行一次额外的查询过程
            if 'JS_files' in statistcsCom:

                if "changed_files_LS" in statistcsCom:
                    statistcsCom.pop('changed_files_LS')
            else:
                com_saved_values = self.SearchJsFilesForCommitCom(replacement, com)
                self.redis_st.append_value(key=str_com, value=com_saved_values)
                statistcsCom.update(com_saved_values)

            statistcsCom = pd.json_normalize(statistcsCom)
            statistcsCom = statistcsCom.reindex(columns=self.extendFeatures)
            return statistcsCom

        else:
            statistcsCom = self.SearchStaForCommitCom(com, coll, replacement)
            # 如果在重新查询以后还是为None 那么仍然把None存储其中
            if statistcsCom == None or len(statistcsCom) == 0:
                com_saved_values = json.dumps(statistcsCom)
                self.redis_st.set_key_value(key=str_com, value=com_saved_values)

            # 如果原值为None，但是当前查询出来的结果并不是None 那么重新存储
            elif redis_com_value == None and statistcsCom != None:
                tmp_staCom = statistcsCom[0]
                com_saved_values = json.dumps(tmp_staCom)
                self.redis_st.set_key_value(key=str_com, value=com_saved_values)

            # 如果原键存在，但是不为有效值，则将其重新存储
            elif redis_com_value != None and statistcsCom != None:
                com_saved_values = statistcsCom[0]
                self.redis_st.append_value(key=str_com, value=com_saved_values)

            if statistcsCom!=None and len(statistcsCom)!=0:
                statistcsCom = statistcsCom[0]
                if isinstance(statistcsCom['TwoCommit'], str):
                    statistcsCom['TwoCommit'] = eval(statistcsCom['TwoCommit'])
                statistcsCom = pd.json_normalize([statistcsCom])
                statistcsCom = statistcsCom.reindex(columns=self.extendFeatures)
                return statistcsCom

        return None

    def SearchJsFilesForCommitCom(self, replacement, test_com):
        query = _10_negative_CM_JSFiles_within_1Mon
        JS_files = self.query5_get_JS_files(query, replacement, test_com)
        return JS_files

    # 这边的查询过程应当调整
    def SearchStaForCommitCom(self, test_com, coll, replacement):
        coll_names = database.list_collection_names()
        repo_name = replacement["repo"]
        new_data = {}
        query = _1_statistics_pNCM
        query_result = self.execute_query(coll, query, test_com, replacement)
        if not query_result:
            return []
        obj = query_result[0]
         # 保存所有必要的信息
        commits_Time = obj.get('Commtis_Time')
        if commits_Time is not None:
            entropy = dp.calculate_entropy(commits_Time)
            del obj['Commtis_Time']
            obj["TimecommitsEntropy"] = entropy[0]
        else:
            print("query_result_时间查询为空" + str(test_com))
            return []

        # 执行 二次查询
        created_at = obj["last_commit"]
        files_changed = obj["Files_changed"]
        del obj['Files_changed'], obj["last_commit"]
        repo_name_INmongo = find_element(coll_names, repo_name)

        # query_2_result = self.execute_query_2(repo_name_INmongo, created_at, files_changed, replacement)
        # 这个值暂时不存储
        # if not query_2_result:
        #     return None
        # JS_files = query_2_result[0]['pr_1Mon_files_Jaccard_Similarity']
        # obj["JS_files"] = JS_files
        # 执行 三次查询

        query_3_result = self.execute_query_3(repo_name_INmongo, created_at, files_changed, replacement)
        if not query_3_result:
            return []
        obj["MeanJsPrFiles"] = np.mean(query_3_result[0]['pr_1Mon_files_Kth_JS'])
        obj["MaxJsPrFiles"] = np.max(query_3_result[0]['pr_1Mon_files_Kth_JS'])
        obj['pr_1Mon_files_Jaccard_Similarity_Self'] = query_3_result[0]['pr_1Mon_files_Jaccard_Similarity_Self']
        new_data.update(obj)

        # 执行extend_features的查询过程
        obj = self.extend_query_process(database, test_com, replacement)
        if obj == None:
            return None
        new_data.update(obj)
        new_data['last_pr'] = 1 if new_data['last_pr'] else 0
        obj = self.SearchJsFilesForCommitCom(replacement, test_com)
        if obj==None:
            return None
        new_data.update(obj)

        sorted_dict = {key: new_data[key] for key in self.extendFeatures}
        return [sorted_dict]

    def execute_query(self, coll, query,test_com, replacement):
        query = recursive_replace(query, replacement)
        query[1]["$match"]["sha"]["$in"] = test_com
        # pprint(query)
        try:
            query_result = list(coll.aggregate(query, allowDiskUse=True))
        except Exception as e:
            print("query_result")
            print(e)
            return []
        return query_result

    def execute_query_2(self, repo_name_INmongo, created_at, files_changed, replacement):
        coll_ = database[repo_name_INmongo]
        repla_query_2 = recursive_replace(_10_PRs_Files_JaccardSimilarity, replacement)
        repla_query_2[1] = replace_value_recursive(repla_query_2[1], '%this_input_date', created_at)
        repla_query_2[7] = replace_value_recursive(repla_query_2[7], '%this_input_filesLS', files_changed)
        try:
            query_2_result = list(coll_.aggregate(repla_query_2, allowDiskUse=True))
        except Exception as e:
            print("query_2_result")
            print(e)
            return []
        return query_2_result

    def execute_query_3(self, repo_name_INmongo, created_at, files_changed, replacement):
        coll_ = database[repo_name_INmongo]
        repla_query_3 = recursive_replace(_11_PRs_Files_JS_TopK, replacement)
        repla_query_3[1] = replace_value_recursive(repla_query_3[1], '%this_input_date', created_at)
        repla_query_3[5] = replace_value_recursive(repla_query_3[5], '%this_input_filesLS', files_changed)
        repla_query_3[11] = replace_value_recursive(repla_query_3[11], '%this_input_filesLS', files_changed)
        try:
            query_3_result = list(coll_.aggregate(repla_query_3, allowDiskUse=True))
        except Exception as e:
            print("query_3_result")
            print(e)
            return []
        if query_3_result == []:
            query_3_result.append({
                'pr_1Mon_files_Kth_JS':[0.0,0.0,0.0,0.0],
                'pr_1Mon_files_Jaccard_Similarity_Self':0
            })
        return query_3_result

    def extend_query_process(self, database, test_com, replacement):
        repo_name = replacement["repo"]
        coll_names = database.list_collection_names()
        repo_name_INmongo = find_element(coll_names, repo_name, C="commits")
        if repo_name_INmongo == None:
            return None
        coll = database[repo_name_INmongo]
        pre_query_result = self.extend_query_pre_info_prepare(coll, test_com, replacement)
        # 当这里的query_result不为空的时候进行遍历
        if len(pre_query_result) == 0:
            return None
        else:
            # query_result是一个字典，其中包含 coco_after3mon_time，coco_firstCommit_time，coco_owner，files_changed
            batch = pre_query_result[0]
            replacement['coco_after3mon_time'] = batch["coco_after3mon_time"]
            replacement['coco_firstCommit_time'] = batch["coco_firstCommit_time"]
            replacement['files_changed'] = batch["files_changed"]
            replacement['coco_owner'] = batch["coco_owner"]

            coll_names = database.list_collection_names()
            repo_name_INmongo = find_element(coll_names, replacement["repo"])
            coll_ = database[repo_name_INmongo]

            repla_query_2 = recursive_replace(_13_contributorFeatures, replacement)
            q2_re = self.query2_With_contributorFeatures(coll_, repla_query_2, replacement)
            if q2_re == None:
                q2_re={
                    'RateMergedPr':0,
                    'prNum':0,
                    'RatePrCommits':0
                }

            repla_query_3 = recursive_replace(_14_refusedPR_FilesLS_For_calculate_file_rejected_proportion, replacement)
            q3_re = self.query3_With_contributorFeatures(coll_, repla_query_3, replacement)
            if q3_re == None:
                q3_re = {
                    'file_rejected_proportion':0
                }

            repla_query_4 = recursive_replace(_15_last_10_mergedAndRejected_Andmost_recent, replacement)
            q4_re = self.query4_With_contributorFeatures(coll_, repla_query_4, replacement)
            if q4_re == None:
                q4_re={
                    'last_10_merged':0,
                    'last_10_rejected':10,
                    'last_pr':False
                }
            re = {}
            re.update(q2_re)
            re.update(q3_re)
            re.update(q4_re)
        return re

    def extend_query_pre_info_prepare(self, coll, test_com, replacement):
        query = _preWork_for_13_14_15_Query
        repla_query_1 = recursive_replace(query, replacement)
        repla_query_1[0] = replace_value_recursive(repla_query_1[0], '%coco_shaLS', test_com)
        # pprint(repla_query_1)
        try:
            query_result = list(coll.aggregate(repla_query_1, allowDiskUse=True))
        except Exception as e:
            print(e)
            return
        return query_result

    def query2_With_contributorFeatures(self,coll_, query, replacement):
        query = recursive_replace(query, replacement)
        query[1] = replace_value_recursive(query[1], '%coco_firstCommit_time', replacement['coco_firstCommit_time'])
        query[1] = replace_value_recursive(query[1], '%coco_after3mon_time', replacement['coco_after3mon_time'])
        query[1] = replace_value_recursive(query[1], '%coco_owner', replacement['coco_owner'])
        # pprint(query)
        try:
            q2_result = list(coll_.aggregate(query, allowDiskUse=True))
        except Exception as e:
            print(e)
            return None
        # RateMergedPr,prNum,RatePrCommits
        if len(q2_result) == 0:
            return {"RateMergedPr": 0, "prNum": 0,
                    "RatePrCommits": 0}
        else:
            q2_result = q2_result[0]
            return {"RateMergedPr": q2_result['RateMergedPr'], "prNum": q2_result["prNum"],
                    "RatePrCommits": q2_result["RatePrCommits"]}

    def query3_With_contributorFeatures(self,coll_, query, replacement):
        query = recursive_replace(query, replacement)
        query[1] = replace_value_recursive(query[1], '%coco_firstCommit_time', replacement['coco_firstCommit_time'])
        query[1] = replace_value_recursive(query[1], '%coco_after3mon_time', replacement['coco_after3mon_time'])
        query[-3] = replace_value_recursive(query[-3], '%files_changed', replacement['files_changed'])
        try:
            q3_result = list(coll_.aggregate(query, allowDiskUse=True))
        except Exception as e:
            print(e)
            return None
        # file_rejected_proportion
        if len(q3_result) == 0:
            return {"file_rejected_proportion": 0}
        else:
            q3_result = q3_result[0]
            return {"file_rejected_proportion": q3_result['file_rejected_proportion']}

    def query4_With_contributorFeatures(self,coll_, query, replacement):
        query = recursive_replace(query, replacement)
        query[1] = replace_value_recursive(query[1], '%coco_firstCommit_time', replacement['coco_firstCommit_time'])
        try:
            q4_result = list(coll_.aggregate(query, allowDiskUse=True))
        except Exception as e:
            print(e)
            return None
        # last_10_merged,last_10_rejected,last_pr
        if len(q4_result) == 0:
            return {"last_10_merged": 0, "last_10_rejected": 10,
                    "last_pr": False}
        else:
            q4_result = q4_result[0]
            return {"last_10_merged": q4_result['last_10_merged'], "last_10_rejected": q4_result['last_10_rejected'],
                    "last_pr": q4_result['last_pr']}

    def query5_get_JS_files(self, query, replacement, coco):
        coll_names = database.list_collection_names()
        repo_name = replacement["repo"]
        PR_start_time = replacement["PR_start_time"]
        repo_name_INmongo = find_element(coll_names, repo_name, C="commits")
        if repo_name_INmongo == None:
            return None
        coll = database[repo_name_INmongo]
        repla_query_1 = recursive_replace(query, replacement)
        repla_query_1[0] = replace_value_recursive(repla_query_1[0], '%Neg_Com', coco)
        repla_query_1[1] = replace_value_recursive(repla_query_1[1], '%NCM_start_time', PR_start_time)
        # pprint(repla_query_1)
        try:
            query_result = list(coll.aggregate(repla_query_1, allowDiskUse=True))
        except Exception as e:
            print(e)
            return None
        # 当这里的query_result不为空的时候进行遍历
        if len(query_result) == 0:
            return {"JS_files": 0}

        else:
            batch = query_result[0]
            repo_name_INmongo = find_element(coll_names, repo_name)
            coll = database[repo_name_INmongo]
            repla_query_2 = recursive_replace(_10_PRs_Files_JaccardSimilarity, replacement)
            repla_query_2[1] = replace_value_recursive(repla_query_2[1], '%this_input_date', batch['created_at'])
            repla_query_2[-2] = replace_value_recursive(repla_query_2[-2], '%this_input_filesLS', batch['file_names'])
            try:
                query_2_result = list(coll.aggregate(repla_query_2, allowDiskUse=True))
            except Exception as e:
                print(e)
                return None
            if len(query_2_result) != 0:
                JS_files = query_2_result[0]["pr_1Mon_files_Jaccard_Similarity"]
                if JS_files == None:
                    JS_files = 0
                new_data = {"JS_files": JS_files}
                return new_data
        return {"JS_files": 0}
