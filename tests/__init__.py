LOGS = [
    {'train_log':
        {
            'xgb_classification_learner': {
                'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                'prediction_column': 'prediction', 'package': 'xgboost',
                'package_version': '0.6',
                'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                               'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                'training_samples': 8, 'running_time': '0.019 s'
            }
        },
        'validator_log': [
            {'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
             'split_log': {'train_size': 8, 'test_size': 8}},
            {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
             'split_log': {'train_size': 8, 'test_size': 8}}],
        'used_subsets': ['first', 'third']},
    {'train_log':
        {
            'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                           'prediction_column': 'prediction', 'package': 'xgboost',
                                           'package_version': '0.6',
                                           'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                          'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                           'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                           'training_samples': 8, 'running_time': '0.019 s'}
        },
        'validator_log': [{'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 1.0}],
                           'split_log': {'train_size': 8, 'test_size': 8}},
                          {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.8}],
                           'split_log': {'train_size': 8, 'test_size': 8}}],
        'used_subsets': ['first', 'second']
     }
]


PARALLEL_LOGS = [
    [{'train_log':
        {
            'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                           'prediction_column': 'prediction', 'package': 'xgboost',
                                           'package_version': '0.6',
                                           'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                          'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                           'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                           'training_samples': 8, 'running_time': '0.019 s'}
        },
        'validator_log': [
            {'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.5}],
             'split_log': {'train_size': 8, 'test_size': 8}},
            {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.5}],
             'split_log': {'train_size': 8, 'test_size': 8}}],
        'used_subsets': ['first', 'second', 'third']},
        {'train_log':
            {
                'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                               'prediction_column': 'prediction', 'package': 'xgboost',
                                               'package_version': '0.6',
                                               'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                              'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                               'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                               'training_samples': 8, 'running_time': '0.019 s'}
            },
            'validator_log': [{'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.7}],
                               'split_log': {'train_size': 8, 'test_size': 8}},
                              {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.7}],
                               'split_log': {'train_size': 8, 'test_size': 8}}],
            'used_subsets': ['first', 'second', 'third']}
     ],
    [{'train_log':
        {
            'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                           'prediction_column': 'prediction', 'package': 'xgboost',
                                           'package_version': '0.6',
                                           'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                          'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                           'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                           'training_samples': 8, 'running_time': '0.019 s'}
        },
        'validator_log': [
            {'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
             'split_log': {'train_size': 8, 'test_size': 8}},
            {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
             'split_log': {'train_size': 8, 'test_size': 8}}],
        'used_subsets': ['first', 'second', 'third']},
        {'train_log': {
            'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                           'prediction_column': 'prediction', 'package': 'xgboost',
                                           'package_version': '0.6',
                                           'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                          'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                           'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                           'training_samples': 8, 'running_time': '0.019 s'}},
            'validator_log': [{'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 1.0}],
                               'split_log': {'train_size': 8, 'test_size': 8}},
                              {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.8}],
                               'split_log': {'train_size': 8, 'test_size': 8}}],
            'used_subsets': ['first', 'second', 'third']}
     ]
]
