_1_CommitsPos_pSpecificFork = [
    {
        '$match': {
            'head.owner': '%owner_specific'
        }
    }, {
        '$project': {
            '_id': 1,
            'owner': '$head.owner',
            'commits': '$commits.sha',
            'events_pr': '$events.action'
        }
    }, {
        '$match': {
            'commits': {
                '$ne': []
            }
        }
    }, {
        '$lookup': {
            'as': 'resultTmp',
            'from': '%referenced_name',
            'let': {
                'pr_owner': '$owner',
                'commits': '$commits'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$eq': [
                                        '$$pr_owner', '$owner'
                                    ]
                                }, {
                                    '$in': [
                                        '$sha', '$$commits'
                                    ]
                                }
                            ]
                        },
                        'payload.files.filename': {
                            '$exists': True
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'cm_sha': '$sha'
                    }
                }
            ]
        }
    }, {
        '$project': {
            '_id': 0,
            'owner': 1,
            'commits': '$resultTmp.cm_sha',
            'merged_PR': {
                '$cond': {
                    'if': {
                        '$in': [
                            'merged', '$events_pr'
                        ]
                    },
                    'then': 1,
                    'else': 0
                }
            }
        }
    }, {
        '$match': {
            'commits': {
                '$ne': []
            }
        }
    }
]
_1_CommitSqe_pSpecificFork = [
    {
        '$match': {
            'owner': '%owner_specific',
            'payload.files.filename': {
                '$exists': True
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'sha': 1,
            'owner': 1
        }
    }, {
        '$group': {
            '_id': None,
            'commits': {
                '$push': '$sha'
            },
            'owner': {
                '$first': '$owner'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'commits': 1,
            'owner': 1
        }
    }
]

_1_CommitNeg_pSpecificFork = [
    {
        '$match': {
            'owner': '%owner_specific',
            'payload.files.filename': {
                '$exists': True
            },
            'pull_requests.pull_number':{
                '$exists':False
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'sha': 1,
            'owner': 1
        }
    }, {
        '$group': {
            '_id': None,
            'commits': {
                '$push': '$sha'
            },
            'owner': {
                '$first': '$owner'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'commits': 1,
            'owner': 1
        }
    }
]

_2_statistics_pNCM = [
    {
        '$match': {
            'owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$match': {
            'payload.sha': {
                '$exists': True
            },
            'sha': {
                '$in': "%sha_ls"
            }
        }
    }, {
        '$sort': {
            'created_at': 1
        }
    }, {
        '$group': {
            'Churn_LS': {
                '$push': '$payload.stats.total'
            },
            'CommitsNum': {
                '$sum': 1
            },
            'Commtis_Time': {
                '$push': '$created_at'
            },
            'Files_changed_LS': {
                '$push': '$payload.files.filename'
            },
            '_id': None
        }
    }, {
        '$project': {
            'Churn_Num': {
                '$sum': '$Churn_LS'
            },
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$Files_changed_LS'
                }
            },
            'TS_INTwoCommits': {
                '$toString': {
                    '$divide': [
                        {
                            '$subtract': [
                                {
                                    '$toDate': {
                                        '$last': '$Commtis_Time'
                                    }
                                }, {
                                    '$toDate': {
                                        '$first': '$Commtis_Time'
                                    }
                                }
                            ]
                        }, 3600000
                    ]
                }
            },
            '_id': 0,
            'first_commit': {
                '$toString': {
                    '$subtract': [
                        {
                            '$toDate': {
                                '$last': '$Commtis_Time'
                            }
                        }, {
                            '$multiply': [
                                1, 2629800000
                            ]
                        }
                    ]
                }
            },
            'last_commit': {
                '$last': '$Commtis_Time'
            },
            'master_owner': '%owner'
        }
    },  {
        '$lookup': {
            'from': '%referenced_name',
            'as': 'result',
            'let': {
                'first_commit': '$first_commit',
                'last_commit': '$last_commit',
                'master_owner': '$master_owner'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$gte': [
                                        '$created_at', '$$first_commit'
                                    ]
                                }, {
                                    '$lte': [
                                        '$created_at', '$$last_commit'
                                    ]
                                }, {
                                    '$eq': [
                                        '$owner', '$$master_owner'
                                    ]
                                }
                            ]
                        },
                        'payload.files': {
                            '$exists': True
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'master_files_changed': '$payload.files.filename'
                    }
                }
            ]
        }
    }, {
        '$project': {
            'Churn_Num': 1,
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': 1,
            'TS_INTwoCommits': 1,
            '_id': 1,
            'master_FilesChanged': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.master_files_changed', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$result'
                }
            }
        }
    }, {
        '$project': {
            'Num_master_FilesChanged': {
                '$size': {
                    '$ifNull': [
                        '$master_FilesChanged', []
                    ]
                }
            },
            'Churn_Num': 1,
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': 1,
            'Inter_FilesChangedNum': {
                '$setIntersection': [
                    '$Files_changed', '$master_FilesChanged'
                ]
            },
            'NCM_FilesChangedNum': {
                '$size': {
                    '$ifNull': [
                        '$Files_changed', []
                    ]
                }
            },
            'TS_INTwoCommits': 1,
            'master_FilesChanged': 1
        }
    }, {
        '$project': {
            '#Churn': '$Churn_Num',
            '#Commits': '$CommitsNum',
            '#FilesChanged': '$NCM_FilesChangedNum',
            'Commtis_Time': 1,
            'NumOverlapfiles': {
                '$size': {
                    '$ifNull': [
                        '$Inter_FilesChangedNum', []
                    ]
                }
            },
            'RateOverlapFiles': {
                '$cond': {
                    'else': {
                        '$divide': [
                            {
                                '$size': {
                                    '$ifNull': [
                                        '$Inter_FilesChangedNum', []
                                    ]
                                }
                            }, '$Num_master_FilesChanged'
                        ]
                    },
                    'if': {
                        '$eq': [
                            '$Num_master_FilesChanged', 0
                        ]
                    },
                    'then': 0
                }
            },
            'TwoCommit': '$TS_INTwoCommits',
            'RateOverlapFilesSelf': {
                '$cond': {
                    'else': {
                        '$divide': [
                            {
                                '$size': {
                                    '$ifNull': [
                                        '$Inter_FilesChangedNum', []
                                    ]
                                }
                            }, '$NCM_FilesChangedNum'
                        ]
                    },
                    'if': {
                        '$eq': [
                            '$NCM_FilesChangedNum', 0
                        ]
                    },
                    'then': 0
                }
            }
        }
    }
]

_1_statistics_pNCM = [
    {
        '$match': {
            'owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$match': {
            'payload.sha': {
                '$exists': True
            },
            'sha': {
                '$in': "%sha_ls"
            }
        }
    }, {
        '$sort': {
            'created_at': 1
        }
    }, {
        '$group': {
            'Churn_LS': {
                '$push': '$payload.stats.total'
            },
            'CommitsNum': {
                '$sum': 1
            },
            'Commtis_Time': {
                '$push': '$created_at'
            },
            'Files_changed_LS': {
                '$push': '$payload.files.filename'
            },
            '_id': None
        }
    }, {
        '$project': {
            'Churn_Num': {
                '$sum': '$Churn_LS'
            },
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$Files_changed_LS'
                }
            },
            'TS_INTwoCommits': {
                    '$divide': [
                        {
                            '$subtract': [
                                {
                                    '$toDate': {
                                        '$last': '$Commtis_Time'
                                    }
                                }, {
                                    '$toDate': {
                                        '$first': '$Commtis_Time'
                                    }
                                }
                            ]
                        }, 3600000
                    ]
            },
            '_id': 0,
            'first_commit': {
                '$toString': {
                    '$subtract': [
                        {
                            '$toDate': {
                                '$last': '$Commtis_Time'
                            }
                        }, {
                            '$multiply': [
                                1, 2629800000
                            ]
                        }
                    ]
                }
            },
            'last_commit': {
                '$last': '$Commtis_Time'
            },
            'master_owner': '%owner'
        }
    },  {
        '$lookup': {
            'from': '%referenced_name',
            'as': 'result',
            'let': {
                'first_commit': '$first_commit',
                'last_commit': '$last_commit',
                'master_owner': '$master_owner'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$gte': [
                                        '$created_at', '$$first_commit'
                                    ]
                                }, {
                                    '$lte': [
                                        '$created_at', '$$last_commit'
                                    ]
                                }, {
                                    '$eq': [
                                        '$owner', '$$master_owner'
                                    ]
                                }
                            ]
                        },
                        'payload.files': {
                            '$exists': True
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'master_files_changed': '$payload.files.filename'
                    }
                }
            ]
        }
    }, {
        '$project': {
            'Churn_Num': 1,
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': 1,
            'TS_INTwoCommits': 1,
            '_id': 1,
            'last_commit':1,
            'master_FilesChanged': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.master_files_changed', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$result'
                }
            }
        }
    }, {
        '$project': {
            'Num_master_FilesChanged': {
                '$size': {
                    '$ifNull': [
                        '$master_FilesChanged', []
                    ]
                }
            },
            'Churn_Num': 1,
            'CommitsNum': 1,
            'Commtis_Time': 1,
            'Files_changed': 1,
            'last_commit':1,
            'Inter_FilesChangedNum': {
                '$setIntersection': [
                    '$Files_changed', '$master_FilesChanged'
                ]
            },
            'NCM_FilesChangedNum': {
                '$size': {
                    '$ifNull': [
                        '$Files_changed', []
                    ]
                }
            },
            'TS_INTwoCommits': 1,
            'master_FilesChanged': 1
        }
    }, {
        '$project': {
            '#Churn': '$Churn_Num',
            '#Commits': '$CommitsNum',
            'last_commit':1,
            '#FilesChanged': '$NCM_FilesChangedNum',
            'Commtis_Time': 1,
            'Files_changed': 1,
            'NumOverlapfiles': {
                '$size': {
                    '$ifNull': [
                        '$Inter_FilesChangedNum', []
                    ]
                }
            },
            'RateOverlapFiles': {
                '$cond': {
                    'else': {
                        '$divide': [
                            {
                                '$size': {
                                    '$ifNull': [
                                        '$Inter_FilesChangedNum', []
                                    ]
                                }
                            }, '$Num_master_FilesChanged'
                        ]
                    },
                    'if': {
                        '$eq': [
                            '$Num_master_FilesChanged', 0
                        ]
                    },
                    'then': 0
                }
            },
            'TwoCommit': '$TS_INTwoCommits',
            'RateOverlapFilesSelf': {
                '$cond': {
                    'else': {
                        '$divide': [
                            {
                                '$size': {
                                    '$ifNull': [
                                        '$Inter_FilesChangedNum', []
                                    ]
                                }
                            }, '$NCM_FilesChangedNum'
                        ]
                    },
                    'if': {
                        '$eq': [
                            '$NCM_FilesChangedNum', 0
                        ]
                    },
                    'then': 0
                }
            }
        }
    }
]

_10_PRs_Files_JaccardSimilarity = [
    {
        '$match': {
            'commits.created_at': {
                '$exists': True
            },
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            'PR_owner': '$head.owner',
            'commits_sha': '$commits.sha',
            'created_at': 1,
            'last_created_at': {
                '$toDate': '%this_input_date'
            }
        }
    }, {
        '$project': {
            'PR_owner': 1,
            'commits_sha': 1,
            'created_at': {
                '$toDate': '$created_at'
            },
            'last_created_at': 1,
            'first_commit': {
                '$subtract': [
                    { '$toDate': '$last_created_at' },
                    { '$multiply': [2592000000, 1] }  
            }
        }
    }, {
        '$match': {
            '$expr': {
                '$and': [
                    {
                        '$gt': [
                            '$created_at', '$first_commit'
                        ]
                    }, {
                        '$lt': [
                            '$created_at', '$last_created_at'
                        ]
                    }
                ]
            }
        }
    }, {
        '$lookup': {
            'from': '%referenced_name',
            'as': 'result',
            'let': {
                'result': '$result',
                "pr_owner": "$PR_owner",
                "commits_sha": "$commits_sha",
            },
            'pipeline': [
                {'$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$eq': [
                                        '$owner', '$$pr_owner'
                                    ]
                                }, {
                                    '$in': [
                                        '$sha', '$$commits_sha'
                                    ]
                                }
                            ]
                        }
                    }},
                    {
                    '$project': {
                        '_id': 0,
                        'files_changed': '$payload.files.filename'
                    }
                    }
            ]
        }
    }, {
        '$project': {
            '_id': 0,
            'pr_files': {
                '$reduce': {
                    'input': '$result',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.files_changed', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }, {
        '$group': {
            '_id': None,
            'pr_1Mon_filesLS': {
                '$push': '$pr_files'
            }
        }
    }, {
        '$project': {
            'this_files': "%this_input_filesLS",
            'pr_1Mon_filesLS': {
                '$reduce': {
                    'input': '$pr_1Mon_filesLS',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }, {
        '$project': {
            '_id':0,
            'pr_1Mon_files_Jaccard_Similarity': {
                '$divide': [
                    {
                        '$size': {
                            '$setIntersection': [
                                '$this_files', '$pr_1Mon_filesLS'
                            ]
                        }
                    }, {
                        '$size': {
                            '$setUnion': [
                                '$this_files', '$pr_1Mon_filesLS'
                            ]
                        }
                    }
                ]
            }
        }
    }
]

_10_negative_CM_JSFiles_within_1Mon = [
    {
        '$match': {
            'sha': {
                '$in': '%Neg_Com'
            },
            "payload.files.filename":{
                '$exists':True
            }
        }
    }, {
        '$project': {
            'pr_owner': '$owner',
            'sha': 1,
            'created_at': 1,
            'file_name': '$payload.files.filename',
            'dateInMonths': {
                '$divide': [
                    {
                        '$subtract': [
                            {
                                '$toDate': '$created_at'
                            }, {
                                '$toDate': '%NCM_start_time'
                            }
                        ]
                    }, 1000 * 60 * 60 * 24 * 30
                ]
            }
        }
    }, {
        '$match': {
            '$expr': {
                '$gte': [
                    '$dateInMonths', 0
                ]
            }
        }
    }, {
        '$group': {
            '_id': None,
            'created_at': {
                '$min': '$created_at'
            },
            'sha': {
                '$push': '$sha'
            },
            'file_name': {
                '$push': '$file_name'
            },
            'pr_owner': {
                '$first': '$pr_owner'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'pr_owner': 1,
            'sha': 1,
            'created_at':1,
            'file_names': {
                '$reduce': {
                    'input': '$file_name',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }
]

_11_PRs_Files_JS_TopK = [
    {
        '$match': {
            'commits.created_at': {
                '$exists': True
            },
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            'PR_owner': '$head.owner',
            'commits_sha': '$commits.sha',
            'created_at': 1,
            'last_created_at': {
                '$toDate': '%this_input_date'
            }
        }
    }, {
        '$project': {
            'PR_owner': 1,
            'commits_sha': 1,
            'created_at': {
                '$toDate': '$created_at'
            },
            'first_commit': {
                '$subtract': [
                    {
                        '$toDate': '$last_created_at'
                    }, {
                        '$multiply': [
                            2592000000, 1
                        ]
                    }
                ]
            },
            'last_created_at': 1
        }
    }, {
        '$match': {
            '$expr': {
                '$and': [
                    {
                        '$gt': [
                            '$created_at', '$first_commit'
                        ]
                    }, {
                        '$lt': [
                            '$created_at', '$last_created_at'
                        ]
                    }
                ]
            }
        }
    }, {
        '$lookup': {
            'as': 'result',
            'from': '%referenced_name',
            'let': {
                'commits_sha': '$commits_sha',
                'pr_owner': '$PR_owner',
                'result': '$result'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$eq': [
                                        '$owner', '$$pr_owner'
                                    ]
                                }, {
                                    '$in': [
                                        '$sha', '$$commits_sha'
                                    ]
                                }
                            ]
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'files_changed': '$payload.files.filename'
                    }
                }
            ]
        }
    }, {
        '$project': {
            '_id': 0,
            'this_files': "%this_input_filesLS",
            'pr_files': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.files_changed', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$result'
                }
            }
        }
    }, {
        '$project': {
            'pr_files': 1,
            'pr_1Mon_files_Kth_JS': {
                '$cond': {
                            'if': { '$eq': [ {'$size': {'$setUnion': ['$this_files', '$pr_files']} }, 0 ] },
                            'then': 0,  
                            'else': {
                                  '$divide': [
                                {
                                    '$size': {
                                        '$setIntersection': [
                                            '$this_files', '$pr_files'
                                        ]
                                    }
                                }, {
                                    '$size': {
                                        '$setUnion': [
                                            '$this_files', '$pr_files'
                                        ]
                                    }
                                }
                ]
                }
            }
            }
        }
    }, {
        '$group': {
            '_id': None,
            'pr_1Mon_filesLS': {
                '$push': '$pr_files'
            },
            'pr_1Mon_files_Kth_JS': {
                '$push': '$pr_1Mon_files_Kth_JS'
            }
        }
    }, {
        '$unwind': {
            'path': '$pr_1Mon_files_Kth_JS',
            'preserveNullAndEmptyArrays': True
        }
    }, {
        '$sort': {
            'pr_1Mon_files_Kth_JS': -1
        }
    }, {
        '$group': {
            '_id': None,
            'pr_1Mon_filesLS': {
                '$first': '$pr_1Mon_filesLS'
            },
            'pr_1Mon_files_Kth_JS': {
                '$push': '$pr_1Mon_files_Kth_JS'
            }
        }
    }, {
        '$project': {
            'pr_1Mon_filesLS': {
                '$reduce': {
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    },
                    'initialValue': [],
                    'input': '$pr_1Mon_filesLS'
                }
            },
            'pr_1Mon_files_Kth_JS': {
                '$slice': [
                    '$pr_1Mon_files_Kth_JS', 5
                ]
            },
            'this_files': "%this_input_filesLS"
        }
    }, {
        '$project': {
            '_id': 0,
            'pr_1Mon_files_Kth_JS': 1,
            'pr_1Mon_files_Jaccard_Similarity_Self': {
                '$cond': {
                    'if': { '$eq': [ { '$size': '$this_files' }, 0 ] },
                    'then': 0,  
                    'else': {
                        '$divide': [
                            { '$size': { '$setIntersection': [ '$this_files', '$pr_1Mon_filesLS' ] } },
                            { '$size': '$this_files' }
                        ]
                    }
                }
            }
        }
    }
]

_12_getFilesLs_By_CMSHA = [
    {
        '$match': {
            'payload.sha': {
                '$exists': True
            },
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            '_id': 1,
            'sha': 1,
            'payload.files.filename': 1,
            'sha_in_pos': "%sha_in_pos",
            'sha_in_neg': "%sha_in_neg"
        }
    }, {
        '$project': {
            '_id': 1,
            'sha': 1,
            'payload.files.filename': 1,
            'sha_in_pos': 1,
            'sha_in_neg': 1,
            'all_SHA': {
                '$setUnion': [
                    '$sha_in_pos', '$sha_in_neg'
                ]
            }
        }
    }, {
        '$match': {
            '$expr': {
                '$in': [
                    '$sha', '$all_SHA'
                ]
            }
        }
    }, {
        '$project': {
            'files_in_pos': {
                '$cond': {
                    'if': {
                        '$in': [
                            '$sha', '$sha_in_pos'
                        ]
                    },
                    'then': '$payload.files.filename',
                    'else': None
                }
            },
            'files_in_neg': {
                '$cond': {
                    'if': {
                        '$in': [
                            '$sha', '$sha_in_neg'
                        ]
                    },
                    'then': '$payload.files.filename',
                    'else': None
                }
            }
        }
    }, {
        '$group': {
            '_id': None,
            'files_in_pos': {
                '$addToSet': '$files_in_pos'
            },
            'files_in_neg': {
                '$addToSet': '$files_in_neg'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'files_in_pos': {
                '$reduce': {
                    'input': '$files_in_pos',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            },
            'files_in_neg': {
                '$reduce': {
                    'input': '$files_in_neg',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }
]

_12_getFilesLS_By_cmsha = [
    {
        '$match': {
            'payload.sha': {
                '$exists': True
            },
            'head.owner': {
                '$ne': '%owner'
            },
            '$expr': {
                '$in': "%sha_LS"
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'changed_files': '$payload.files.filename'
        }
    }, {
        '$group': {
            '_id': None,
            'changed_files': {
                '$addToSet': '$changed_files'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'changed_files_LS': {
                '$reduce': {
                    'input': '$changed_files',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }
]
_preWork_for_13_14_15_Query = [
    {
        '$match': {
            'sha': {
                '$in': '%coco_shaLS'
            }
        }
    }, {
        '$sort': {
            'created_at': 1
        }
    }, {
        '$project': {
            '_id': 0,
            'created_at': 1,
            'head.owner': 1,
            'coco': 1,
            'coco_firstCommit_time': '$created_at',
            'coco_owner': '$owner',
            'files_changed': '$payload.files.filename'
        }
    }, {
        '$addFields': {
            'coco_after3mon_time': {
                '$toString': {
                    '$subtract': [
                        {
                            '$toDate': '$coco_firstCommit_time'
                        }, {
                            '$multiply': [
                                2592000000, 3
                            ]
                        }
                    ]
                }
            }
        }
    }, {
        '$unwind': {
            'path': '$files_changed'
        }
    }, {
        '$group': {
            '_id': '$coco_owner',
            'coco_firstCommit_time': {
                '$first': '$coco_firstCommit_time'
            },
            'coco_after3mon_time': {
                '$first': '$coco_after3mon_time'
            },
            'files_changed': {
                '$addToSet': '$files_changed'
            }
        }
    }, {
        '$project': {
            'coco_owner': '$_id',
            '_id': 0,
            'coco_firstCommit_time': 1,
            'coco_after3mon_time': 1,
            'files_changed': 1
        }
    }
]

_13_contributorFeatures = [
    {
        '$match': {
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'created_at': 1,
            'head.owner': 1,
            'coco': 1,
            'coco_firstCommit_time': '%coco_firstCommit_time',
            'coco_owner': '%coco_owner',
            'coco_after3mon_time': '%coco_after3mon_time',
            'events.action': 1
        }
    }, {
        '$match': {
            '$expr': {
                '$and': [
                    {
                        '$gt': [
                            '$created_at', '$coco_after3mon_time'
                        ]
                    }, {
                        '$lt': [
                            '$created_at', '$coco_firstCommit_time'
                        ]
                    }, {
                        '$eq': [
                            '$coco_owner', '$head.owner'
                        ]
                    }
                ]
            }
        }
    }, {
        '$project': {
            'coco_firstCommit_time': 1,
            'coco_owner': 1,
            'mergedPR': {
                '$cond': {
                    'if': {
                        '$in': [
                            'merged', '$events.action'
                        ]
                    },
                    'then': True,
                    'else': False
                }
            }
        }
    }, {
        '$lookup': {
            'from': '%referenced_name',
            'as': 'result',
            'let': {
                'coco_firstCommit_time': '$coco_firstCommit_time',
                'coco_owner': '$coco_owner'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$lt': [
                                        '$created_at', '$$coco_firstCommit_time'
                                    ]
                                }, {
                                    '$eq': [
                                        '$owner', '$$coco_owner'
                                    ]
                                }
                            ]
                        }
                    }
                }, {
                    '$group': {
                        '_id': None,
                        'commitsNum_before_cocoFirst': {
                            '$sum': 1
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'commitsNum_before_cocoFirst': 1
                    }
                }
            ]
        }
    }, {
        '$addFields': {
            'commitsNum_before_cocoFirst': {
                '$first': '$result.commitsNum_before_cocoFirst'
            }
        }
    }, {
        '$group': {
            '_id': None,
            'prNum': {
                '$sum': 1
            },
            'commitsNum_before_cocoFirst': {
                '$first': '$commitsNum_before_cocoFirst'
            },
            'mergedPrNum': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$mergedPR', True
                            ]
                        }, 1, 0
                    ]
                }
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'RatePrCommits': {
                '$divide': [
                    {
                        '$ifNull': [
                            '$commitsNum_before_cocoFirst', 0
                        ]
                    }, '$prNum'
                ]
            },
            'prNum': 1,
            'RateMergedPr': {
                '$divide': [
                    '$mergedPrNum', '$prNum'
                ]
            }
        }
    }
]
_14_refusedPR_FilesLS_For_calculate_file_rejected_proportion = [
    {
        '$match': {
            'commits.created_at': {
                '$exists': True
            },
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            'PR_owner': '$head.owner',
            'commits': 1,
            'created_at': 1,
            'pre_created_at': '%coco_firstCommit_time',
            'post_created_at': '%coco_after3mon_time',
            'events.action': 1
        }
    }, {
        '$match': {
            '$expr': {
                '$and': [
                    {
                        '$lt': [
                            '$created_at', '$pre_created_at'
                        ]
                    }, {
                        '$gt': [
                            '$created_at', '$post_created_at'
                        ]
                    }
                ]
            }
        }
    }, {
        '$project': {
            'PR_owner': 1,
            'created_at': 1,
            'commits_sha': '$commits.sha',
            'events_action': '$events.action'
        }
    }, {
        '$lookup': {
            'from': '%referenced_name',
            'as': 'result',
            'let': {
                'result': '$result',
                'pr_owner': '$PR_owner',
                'commits_sha': '$commits_sha'
            },
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$and': [
                                {
                                    '$eq': [
                                        '$owner', '$$pr_owner'
                                    ]
                                }, {
                                    '$in': [
                                        '$sha', '$$commits_sha'
                                    ]
                                }
                            ]
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'files_changed': '$payload.files.filename'
                    }
                }
            ]
        }
    }, {
        '$project': {
            'created_at': 1,
            'pr_files': {
                '$reduce': {
                    'input': '$result',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.files_changed', []
                                ]
                            }
                        ]
                    }
                }
            },
            'events_action': 1
        }
    }, {
        '$group': {
            '_id': None,
            'pr_filesAll': {
                '$push': {
                    '$cond': {
                        'if': {
                            '$not': {
                                '$in': [
                                    'merged', '$events_action'
                                ]
                            }
                        },
                        'then': '$pr_files',
                        'else': None
                    }
                }
            }
        }
    }, {
        '$project': {
            '_id': 1,
            'pr_files': {
                '$reduce': {
                    'input': '$pr_filesAll',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }, {
        '$unwind': {
            'path': '$pr_files'
        }
    }, {
        '$group': {
            '_id': None,
            'pr_files': {
                '$addToSet': '$pr_files'
            }
        }
    }, {
        '$addFields': {
            'files_changed':'%files_changed'
        }
    }, {
        '$project': {
            'overlapping_filesLS': {
                '$setIntersection': [
                    '$pr_files', '$files_changed'
                ]
            },
            'files_changed': 1
        }
    }, {
        '$project': {
            '_id': 0,
            'file_rejected_proportion': {
                '$cond': {
                    'if': {
                        '$eq': [
                            {
                                '$size': '$files_changed'
                            }, 0
                        ]
                    },
                    'then': None,
                    'else': {
                        '$divide': [
                            {
                                '$size': '$overlapping_filesLS'
                            }, {
                                '$size': '$files_changed'
                            }
                        ]
                    }
                }
            }
        }
    }
]

_15_last_10_mergedAndRejected_Andmost_recent = [
    {
        '$match': {
            'head.owner': {
                '$ne': '%owner'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'created_at': 1,
            'head.owner': 1,
            'coco': 1,
            'coco_firstCommit_time': '%coco_firstCommit_time',
            'events.action': 1
        }
    }, {
        '$match': {
            '$expr': {
                '$lt': [
                    '$created_at', '$coco_firstCommit_time'
                ]
            }
        }
    }, {
        '$sort': {
            'created_at': -1
        }
    }, {
        '$limit': 10
    }, {
        '$project': {
            'commitsNum_before_cocoFirst': 1,
            'mergedPR': {
                '$cond': {
                    'if': {
                        '$in': [
                            'merged', '$events.action'
                        ]
                    },
                    'then': True,
                    'else': False
                }
            }
        }
    }, {
        '$group': {
            '_id': None,
            'prNum': {
                '$sum': 1
            },
            'last_10_merged': {
                '$sum': {
                    '$cond': [
                        {
                            '$eq': [
                                '$mergedPR', True
                            ]
                        }, 1, 0
                    ]
                }
            },
            'last_pr': {
                '$first': '$mergedPR'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'last_10_merged': 1,
            'last_10_rejected': {
                '$subtract': [
                    '$prNum', '$last_10_merged'
                ]
            },
            'prNum': 1,
            'last_pr': 1
        }
    }
]
