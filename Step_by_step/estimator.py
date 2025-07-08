import statsmodels.formula.api as smf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

class Estimator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = None
        self.num_predictors = None
        self.cat_predictors = None
        self.predictors = None
        self.target = None
        self.group_col = None
        self.fit_algo = None
    def fit(self, training_data, fit_algo, seed=0):
        self.fit_algo = fit_algo
        self.models = pd.DataFrame() # use dataframe to store the models as a way to assign them a unique index
        if self.fit_algo == 'ensemble':
            '''
            For each distinct stmt_date, spinkle evenly the minority class (1) to the majority class (0) to avoid 
            the issues where some splits returned by TimeSeriesSplit only see the majority class. This relies on the
            assumption that during a prediction period (1 year by default), the ordering of the data does not matter.
            
            As an example, if the ratio of majority class vs. minority class is 100:1, then the ordered dataset will be
            of the form (100 * rate) 0's followed by (1 * rate) 1's, so on and so forth, until we have exhauseed all the 
            original data.
            '''
            n_splits = 3
            cv = TimeSeriesSplit(n_splits=n_splits)
            ordered_training_data = pd.DataFrame()
            for stmt_date in training_data['stmt_date'].drop_duplicates().sort_values():
                # print(stmt_date)
                subset = training_data[training_data['stmt_date'] == stmt_date]
                majority = subset[subset[self.target] == 0]
                minority = subset[subset[self.target] == 1]
                rate = n_splits
                chunk = len(majority) // len(minority)
                chunk *= rate
                i = j = 0
                t = pd.DataFrame()
                while i < len(majority) and j < len(minority):
                    t = pd.concat((t, majority.iloc[i:i+chunk], minority.iloc[j:j+rate]))
                    i += chunk
                    j += rate
                if i < len(majority): t = pd.concat((t, majority.iloc[i:]))
                if j < len(minority): t = pd.concat((t, minority.iloc[j:]))
                ordered_training_data = pd.concat((ordered_training_data, t))
            
            assert len(ordered_training_data) == len(training_data), print(f"{len(training_data)}, {len(ordered_training_data)}")

            # Parameters for training lightgbm, in particular 'is_unbalance' handles the imbalanced dataset
            params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "auc",
                "max_depth": 10,  
                "learning_rate": 0.05,
                "n_estimators": 2000,  
                "colsample_bytree": 0.8,
                "colsample_bynode": 0.8,
                "verbose": -1,
                "random_state": seed,
                "reg_alpha": 0.1,
                "reg_lambda": 10,
                "extra_trees":True,
                'num_leaves':64,
                "device": 'cpu', 
                "verbose": -1,
                "is_unbalance": True
            }
            # In total, the ensemble consists of (# of distinct groups) x (# of splits) lightgbm's
            for group in ordered_training_data[self.group_col].drop_duplicates().sort_values():
                mask = ordered_training_data[self.group_col] == group
                training_data = ordered_training_data[mask]
                for i, (idx_train, idx_valid) in enumerate(cv.split(training_data[self.predictors], training_data[self.target])):
                    X_train, y_train = training_data[self.predictors].iloc[idx_train], training_data[self.target].iloc[idx_train]
                    X_valid, y_valid = training_data[self.predictors].iloc[idx_valid], training_data[self.target].iloc[idx_valid]
                    print(f"{self.group_col}: {group}, split: {i}, shape {X_train.shape}")
                    # print(f"Traing label distribution:\n{y_train.value_counts()}")
                    # print(f"Evaluation label distribution:\n{y_valid.value_counts()}")
                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set = [(X_valid, y_valid)],
                        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)]
                    )
                    self.models = pd.concat((self.models, pd.DataFrame({f"{self.group_col}": group,
                                                                        "index": i, 
                                                                        "weight": len(X_train), 
                                                                        "model": [model]})), ignore_index=True)
            print(f"Number of trained models: {len(self.models)}")
        elif self.fit_algo == 'linear':
            X_train = training_data[self.predictors]
            y_train = training_data[self.target]
            # print(X_train.head())
            for group in training_data[self.group_col].drop_duplicates().sort_values():
                mask = training_data[self.group_col] == group
                print(f"{self.group_col}: {group}, shape {X_train[mask].shape}")
                formula = f"{self.target} ~ {'+'.join(self.predictors)}"
                model = smf.logit(
                    formula=formula, data=pd.concat((X_train[mask],y_train[mask]),axis=1)
                ).fit_regularized(disp=0)
                self.models = pd.concat((self.models, pd.DataFrame({f"{self.group_col}": [group], 
                                                                    "model": [model]})), ignore_index=True)
            print(f"Number of trained models: {len(self.models)}")
           
            # formula = f"{self.target} ~ {'+'.join(self.predictors)}"
            # model = smf.logit(
            #         formula=formula, data=pd.concat((X_train,y_train),axis=1)
            # ).fit_regularized(disp=0)
            # self.models = pd.concat((self.models, pd.DataFrame(
            #         {   f"{self.group_col}": [-1],
            #             "model": [model]})), 
            #         ignore_index=True
            #     )
        else:
            raise Exception("Fit algorithm must be either 'linear' or 'ensemble'!")
    def predict(self, testing_data):
        X_test = testing_data[self.predictors]
        if self.fit_algo == 'ensemble':
            predictions = []
            conditions = []
            for group in self.models[self.group_col].drop_duplicates().sort_values():
                predictions.append(self.__predict(X_test, group))
                conditions.append(testing_data[self.group_col] == group)
            return pd.Series(np.select(conditions, predictions))
        elif self.fit_algo == 'linear': # for smf logistic
            predictions = []
            conditions = []
            for _, row in self.models.iterrows():
                predictions.append(row['model'].predict(X_test))
                conditions.append((testing_data[self.group_col] == row[self.group_col]))
            return pd.Series(np.select(conditions, predictions))
            # return pd.Series(self.models.iloc[0]['model'].predict(X_test))
        else:
            raise Exception("Fit algorithm must be either 'linear' or 'ensemble'!")
    
    # (For ensemble only): given a group, helper function to aggregate (weighted) results from the ensembles models
    # Models are weighted by the number of samples seen during the respective TimeSeriesSplit
    def __predict(self, X_test, group):
        group_models = self.models[self.models[self.group_col] == group]
        predictions = []
        for _, row in group_models.iterrows():
            predictions.append(row['model'].predict_proba(X_test)[:, 1])
        predictions = np.array(predictions) # shape: (# of models, # of testing data)
        weights = group_models['weight'].to_numpy()[:, np.newaxis] # shape: (# of models, 1)
        weighted_predictions = predictions * weights # shape: (# of models, # of testing data)
        sum_of_weights = np.sum(weights) # shape: (1,)
        return weighted_predictions.sum(axis=0) / sum_of_weights # shape: (1, # of testing data)