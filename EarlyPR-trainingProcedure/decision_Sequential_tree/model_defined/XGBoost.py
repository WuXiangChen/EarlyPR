import xgboost as xgb

xgb_trainer = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)
#xgb_trainer = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_estimators=100, learning_rate=0.1, max_depth=5)