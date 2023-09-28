# 本节存储查询代码段

_1_Commits_pSpecificFork = [
    {
        '$match': {
            'head.owner': '%owner_specific'
        }
    }, {
        '$project': {
            '_id': 1, 
            'reporter': '$head.owner', 
            'commits': 1
        }
    }, {
        '$group': {
            '_id': None, 
            'reporter': {
                '$last': '$reporter'
            }, 
            'commitSHA': {
                '$push': '$commits.sha'
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'reporter': 1, 
            'pr_commitSHA': {
                '$reduce': {
                    'input': '$commitSHA', 
                    'initialValue': [], 
                    'in': {
                        '$setUnion': [
                            '$$value', '$$this'
                        ]
                    }
                }
            }
        }
    }, {
        '$lookup': {
            'from': '%referenced_name', 
            'localField': 'pr_commitSHA', 
            'foreignField': 'sha', 
            'as': 'resultTmp', 
            'let': {
                'result': '$resultTmp', 
                'pr_owner': '$reporter'
            }, 
            'pipeline': [
                {
                    '$match': {
                        '$expr': {
                            '$eq': [
                                '$owner', '$$pr_owner'
                            ]
                        }, 
                        'payload.sha': {
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
            'reporter': 1, 
            'pr_commitSHA': {
                '$reduce': {
                    'input': '$resultTmp', 
                    'initialValue': [], 
                    'in': {
                        '$concatArrays': [
                            '$$value', [
                                '$$this.cm_sha'
                            ]
                        ]
                    }
                }
            }
        }
    }, {
        '$lookup': {
            'from': '%referenced_name',
            'localField': 'reporter',
            'foreignField': 'owner',
            'as': 'result',
            'let': {
                'result': '$result'
            },
            'pipeline': [
                {
                    '$match': {
                        'payload.sha': {
                            '$exists': True
                        }
                    }
                }, {
                    '$project': {
                        '_id': 1,
                        'sha': 1
                    }
                }
            ]
        }
    }, {
        '$project': {
            'pr_commitSHA': 1,
            'commitsAll': {
                '$reduce': {
                    'input': '$result',
                    'initialValue': [],
                    'in': {
                        '$concatArrays': [
                            '$$value', [
                                '$$this.sha'
                            ]
                        ]
                    }
                }
            }
        }
    }
]

_2_statistics_pNCM = [
    {
        '$match': {
            'owner': {"$ne":'%owner'},
            'payload.sha': {'$exists': True}
        }
    }, {
        '$match': {
            'sha': {
                '$in': "%sha_ls"
            },
            'payload.sha': {
                '$exists': True
            }
        }
    }, {
        '$group': {
            '_id': None,
            'CommitsNum': {
                '$sum': 1
            },
            'Churn_LS': {
                '$push': '$payload.stats.total'
            },
            'Files_changed_LS': {
                '$push': '$payload.files.filename'
            },
            'Commtis_Time': {
                '$push': '$created_at'
            }
        }
    }, {
        '$project': {
            '_id': 0,
            'CommitsNum': 1,
            'Churn_Num': {
                '$sum': '$Churn_LS'
            },
            'Files_changed': {
                '$reduce': {
                    'input': '$Files_changed_LS',
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
            'Commtis_Time': 1,
            'master_owner': '%owner',
            'first_commit': {
                '$toString': {
                    '$dateSubtract': {
                        'startDate': {
                            '$toDate': {
                                '$first': '$Commtis_Time'
                            }
                        },
                        'unit': 'month',
                        'amount': 1
                    }
                }
            },
            'last_commit': {
                '$last': '$Commtis_Time'
            }
        }
    }, {
        '$lookup': {
            'from': '%referenced_name',
            'localField': 'master_owner',
            'foreignField': 'owner',
            'let': {
                'result': '$result',
                'first_commit': '$first_commit',
                'last_commit': '$last_commit'
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
                                }
                            ]
                        }
                    }
                }, {
                    '$project': {
                        '_id': 0,
                        'master_files_changed': '$payload.files.filename'
                    }
                }
            ],
            'as': 'result'
        }
    }, {
        '$project': {
            '_id': 1,
            'Files_changed': 1,
            'CommitsNum': 1,
            'Churn_Num': 1,
            'pr_FilesChanged': '$pr_files',
            'TS_INTwoCommits': 1,
            'Commtis_Time': 1,
            'master_FilesChanged': {
                '$reduce': {
                    'input': '$result',
                    'initialValue': [],
                    'in': {
                        '$setUnion': [
                            '$$value', {
                                '$ifNull': [
                                    '$$this.master_files_changed', []
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }, {
        '$project': {
            'Files_changed': 1,
            'CommitsNum': 1,
            'Churn_Num': 1,
            'pr_FilesChanged': '$pr_files',
            'TS_INTwoCommits': 1,
            'Commtis_Time': 1,
            'NCM_FilesChangedNum': {
                '$size': {
                    '$ifNull': [
                        '$Files_changed', []
                    ]
                }
            },
            'Inter_FilesChangedNum': {
                '$setIntersection': [
                    '$Files_changed', '$master_FilesChanged'
                ]
            }
        }
    }, {
        '$project': {
            'CommitsNum': 1,
            'Churn_Num': 1,
            'pr_FilesChanged': '$pr_files',
            'TS_INTwoCommits': 1,
            'Commtis_Time': 1,
            'PR_FilesChangedNum': "$NCM_FilesChangedNum",
            'diff_FilesChangedRate': {
                '$cond': {
                    'if': {
                        '$eq': [
                            '$NCM_FilesChangedNum', 0
                        ]
                    },
                    'then': 0,
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
                    }
                }
            }
        }
    }
]


query_pPR = {"_1_Commits_pSpecificFork":_1_Commits_pSpecificFork,
             "_2_statistics_pNCM":_2_statistics_pNCM}

query_pPR_ = {"_1_Commits_pSpecificFork":_1_Commits_pSpecificFork}