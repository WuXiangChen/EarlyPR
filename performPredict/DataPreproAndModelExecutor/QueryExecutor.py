import json

import numpy as np

from datasets.Test_auxiliary_information.querySets_GenerateTestCommits import _10_PRs_Files_JaccardSimilarity,_11_PRs_Files_JS_TopK, \
    _1_statistics_pNCM,_preWork_for_13_14_15_Query,_13_contributorFeatures,_14_refusedPR_FilesLS_For_calculate_file_rejected_proportion,\
    _15_last_10_mergedAndRejected_Andmost_recent, _10_negative_CM_JSFiles_within_1Mon,_16_getMsgAndCodeDiff_FromSHA
from utils.MongoDBUtils import database
from utils.cleanMethods import filterCodeDiff
from utils.predictingUtils import find_element,recursive_replace,replace_value_recursive

from .datapreprocessor import datapreprocessor
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

    # 实现增补查询
    def addForMsgContent(self,str_com,  redis_com_value, replacement, com):
        # 解析 JSON 数据
        statistcsCom = json.loads(redis_com_value)
        query = _16_getMsgAndCodeDiff_FromSHA
        com_saved_values = self.getMsgAndCodeDiff(query, replacement, com)
        self.redis_st.append_value(key=str_com, value=com_saved_values)

        statistcsCom.update(com_saved_values)
        statistcsCom = pd.json_normalize(statistcsCom)
        # 重新排列列
        statistcsCom = statistcsCom.reindex(columns=self.extendFeatures)
        return statistcsCom

    '''
        #redis_com_value = None
        #com = ['890178e6557d7210387445d5830b5c8de0a81e66', '9727916cdf42b5b3c8013905ceaaf960ae888793', '1d06ecb7a94f14d037b73c5f7bb350f3ac4d65e3']
        if redis_com_value and json.loads(redis_com_value) is not None and 'TwoCommit' in json.loads(redis_com_value):
            redis_com_value = self.addForStatisticsCom('TwoCommit', str_com, redis_com_value, replacement, com)

        if redis_com_value and json.loads(redis_com_value) is not None and 'JS_files' in json.loads(redis_com_value):
            redis_com_value = self.addForStatisticsCom('JS_files', str_com, redis_com_value, replacement, com)
    '''
    def search_and_store(self, com):
        # 先对它进行排序，然后再查找
        com.sort()
        str_com = ",".join(com)
        # 这里是看redis中是否存储了对应的内容
        redis_com_value = self.redis_st.get_value(str_com)
        coll, replacement = self.coll, self.replacement

        if redis_com_value is not None and json.loads(redis_com_value) is None:
            redis_com_value = None

        # 如果redis中存储了对应的内容，那么进行内容检查，满足则返回
        if redis_com_value is not None:
            if 'Msg' not in json.loads(redis_com_value):
                redis_com_value = self.addForMsgContent(str_com, redis_com_value, replacement, com)
                return redis_com_value
            return redis_com_value

        else:
            # redis中没有存储任何东西 那么就从MongoDb中重新查询
            statistcsCom = self.SearchStaForCommitCom(com, coll, replacement)
            if statistcsCom is not None and len(statistcsCom)==0:
                statistcsCom = None

            # 如果在重新查询以后还是为None 那么仍然把None存储其中
            if statistcsCom is None:
                com_saved_values = json.dumps(statistcsCom)
                self.redis_st.set_key_value(key=str_com, value=com_saved_values)
                return statistcsCom

            # 如果查询出来的结果不是None 那么重新存储
            elif  statistcsCom is not None:
                tmp_staCom = statistcsCom[0]
                statistcsCom = pd.json_normalize([tmp_staCom])
                statistcsCom = statistcsCom.reindex(columns=self.extendFeatures)

                if "TwoCommit" in tmp_staCom.keys() and isinstance(tmp_staCom['TwoCommit'], str):
                    tmp_staCom['TwoCommit'] = eval(tmp_staCom['TwoCommit'])

                com_saved_values = json.dumps(tmp_staCom)
                self.redis_st.set_key_value(key=str_com, value=com_saved_values)
                return statistcsCom

        return None

    def SearchJsFilesForCommitCom(self, replacement, test_com):
        query = _10_negative_CM_JSFiles_within_1Mon
        JS_files = self.getJSfiles(query, replacement, test_com)
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

        query_3_result = self.getPR1MonFilesKthJS(repo_name_INmongo, created_at, files_changed, replacement)
        if not query_3_result:
            return []
        obj["MeanJsPrFiles"] = np.mean(query_3_result[0]['pr_1Mon_files_Kth_JS'])
        obj["MaxJsPrFiles"] = np.max(query_3_result[0]['pr_1Mon_files_Kth_JS'])
        obj['pr_1Mon_files_Jaccard_Similarity_Self'] = query_3_result[0]['pr_1Mon_files_Jaccard_Similarity_Self']
        new_data.update(obj)

        # 执行extend_features的查询过程
        obj = self.extend_query_process(database, test_com, replacement)
        if obj is None:
            return None
        new_data.update(obj)
        new_data['last_pr'] = 1 if new_data['last_pr'] else 0
        obj = self.SearchJsFilesForCommitCom(replacement, test_com)
        if obj is None:
            return None
        new_data.update(obj)

        # 这里需要补充进行MsgAndCodeDiff的查询
        query = recursive_replace(_16_getMsgAndCodeDiff_FromSHA, replacement)
        obj = self.getMsgAndCodeDiff(query, replacement, test_com)
        # 这里还不知道它的结构
        if obj is not None:
            new_data.update(obj)
        else:
            new_data.update({"msg": "", "OriginalCodeDiff": ""})
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

    def getPR1MonFilesKthJS(self, repo_name_INmongo, created_at, files_changed, replacement):
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
        if repo_name_INmongo is None:
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
            q2_re = self.getRateAboutPR(coll_, repla_query_2, replacement)
            if q2_re is None:
                q2_re={
                    'RateMergedPr':0,
                    'prNum':0,
                    'RatePrCommits':0
                }

            repla_query_3 = recursive_replace(_14_refusedPR_FilesLS_For_calculate_file_rejected_proportion, replacement)
            q3_re = self.getFileRejectedProportion(coll_, repla_query_3, replacement)
            if q3_re is None:
                q3_re = {
                    'file_rejected_proportion':0
                }

            repla_query_4 = recursive_replace(_15_last_10_mergedAndRejected_Andmost_recent, replacement)
            q4_re = self.getLast10MergedAndReject(coll_, repla_query_4, replacement)
            if q4_re is None:
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

    def getRateAboutPR(self, coll_, query, replacement):
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

    def getFileRejectedProportion(self, coll_, query, replacement):
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

    def getLast10MergedAndReject(self, coll_, query, replacement):
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

    def getJSfiles(self, query, replacement, coco):
        coll_names = database.list_collection_names()
        repo_name = replacement["repo"]
        PR_start_time = replacement["PR_start_time"]
        repo_name_INmongo = find_element(coll_names, repo_name, C="commits")
        if repo_name_INmongo is None:
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
                if JS_files is None:
                    JS_files = 0
                new_data = {"JS_files": JS_files}
                return new_data
        return {"JS_files": 0}

    def getMsgAndCodeDiff(self,query, replacement, coco):
        coll_names = database.list_collection_names()
        repo_name = replacement["repo"]
        repo_name_INmongo = find_element(coll_names, repo_name, C="commits")
        coll = database[repo_name_INmongo]

        query[0] = replace_value_recursive(query[0], '%sha', coco)

        repla_query = recursive_replace(query, replacement)
        MsgAndCodeDiff_result = list(coll.aggregate(repla_query, allowDiskUse=True))[0]
        cleaned_results = filterCodeDiff(MsgAndCodeDiff_result)
        return cleaned_results