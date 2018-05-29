import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

application_train=pd.read_csv('data/application_train.csv')
application_test=pd.read_csv('data/application_test.csv')

application_train['CODE_GENDER'].replace('XNA','F',inplace=True)

categorical_columns = ['NAME_CONTRACT_TYPE',
                       'CODE_GENDER',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE',
                       'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE',
                       'WALLSMATERIAL_MODE',
                       'EMERGENCYSTATE_MODE']
for column in categorical_columns:
    application_train[column]=application_train[column].astype('category')
    application_test[column]=application_test[column].astype('category')

input_columns = application_train.columns#zz.columns
input_columns = input_columns[input_columns != 'TARGET']
target_column = 'TARGET'

X = application_train[input_columns]
y = application_train[target_column]
X_train, X_dev, y_train, y_dev = train_test_split(X, y)

lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_dev, label=y_dev)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 200,
        'verbose': 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)

y_values=gbm.predict(application_test)
submission=pd.DataFrame(application_test['SK_ID_CURR'])
submission['TARGET']=y_values.tolist()
submission.to_csv('first_submission.csv',index=None)