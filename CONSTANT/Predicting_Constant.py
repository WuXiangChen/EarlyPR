
from pymongo import MongoClient

db_id_Torepo = {
    "ansible":0,
    "corefx":1,
    "terraform":2,
    "elasticsearch":3,
    "julia":4,
    "node":5,
    "rails":6,
    "rust":7,
    "salt":8,
    "symfony":9,
    "joomla-cms":10,
    "cocos2d-x":11,
    "django":12,
    "servo":13,
    "angular.js":14,
    "scikit-learn":15,
    "kubernetes":16,
    "ceph":17
}

extendFeaturesIndex = ['#Churn', '#Commits', '#FilesChanged','JS_files', 'MaxJsPrFiles', 'MeanJsPrFiles',
       'NumOverlapfiles', 'RateMergedPr', 'RateOverlapFiles',
       'RateOverlapFilesSelf', 'RatePrCommits', 'TimecommitsEntropy',
       'TwoCommit', 'file_rejected_proportion', 'last_10_merged',
       'last_10_rejected', 'last_pr', 'prNum',
       'pr_1Mon_files_Jaccard_Similarity_Self', "msg", "OriginalCodeDiff"]