from bisect import bisect_left
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import string
import numpy as np
import pandas as pd

class Preprocessor(): 
    def __init__(self, imputer):
        self.imputer = imputer
        self.target = None
        self.group = None
        self.group_bins = None
        self.num_cols = [
            'AR', 'COGS', 'asst_current', 'asst_fixed_fin', 
            'asst_intang_fixed', 'asst_tang_fixed', 'asst_tot', 'cash_and_equiv', 
            'cf_operations', 'debt_lt', 'debt_st', 'ebitda', 'eqty_tot', 'exp_financing', 
            'goodwill', 'inc_extraord', 'inc_financing', 'liab_lt', 'liab_lt_emp', 
            'margin_fin', 'prof_financing', 'prof_operations', 'profit', 
            'rev_operating', 'roa', 'roe', 'taxes', 'wc_net'
        ]
        self.ratio_cols = [
            'debt_to_equity',
            'current_ratio',
            'operating_margin',
            'ebitda_assets',
            'ebitda_margin',
            'nwc_ratio',
            'cashflow_to_debt',
            'profit_margin', 
            'days_receivable_turnover', 
            'longtermliabilities_assets', 
            'interest_coverage_ratio', 
            'asset_turnover',
            'equity_ratio',
            'quick_ratio',

            'current_asset_to_sales',
            'cash_and_securities_to_assets',
            'cashflow_to_interest_payment',
            'asset_minus_liab_asst_tot',

            'altman'
        ]
        # self.ratio_cols = [
        #     'debt_to_equity', 'operating_margin', 'ebitda_assets',
        #     'cashflow_to_debt', 'profit_margin', 'days_receivable_turnover',
        #     'longtermliabilities_assets', 'asset_turnover'
        # ]
        # 'debt_to_equity' 'ebitda_assets' 'cashflow_to_debt' 'profit_margin'
        # 'days_receivable_turnover' 'longtermliabilities_assets' 'asset_turnover'
        self.cat_cols = [
            'legal_struct', 'ateco_sector'
        ]
        self.num_transformers = None
        self.cat_transformers = None
        self.ratio_transformers = None
        self.transformed_cat_cols = []
        self.transformed_num_cols = []

    def convert_dtypes(self, df_):
        # https://www.istat.it/classificazione/classificazione-delle-attivita-economiche-ateco/
        '''
        A: 'Agriculture, Forestry And Fisheries', 
        B: 'Extraction Of Minerals From Quarries And Mines',
        C: 'Manufacturing Activities',
        D: 'Supply Of Electricity, Gas, Steam And Air Conditioning',
        E: 'And Water Supply; Sewerage Networks, Waste Management And Remediation Activities',
        F: 'Constructions',
        G: 'Wholesale And Retail Trade; Repair Of Motor Vehicles And Motorcycles',
        H: 'Transport And Storage',
        I: 'Accommodation And Catering Service Activities',
        J: 'Information And Communication Services',
        K: 'Financial And Insurance Activities',
        L: 'Real Estate Activities',
        M: 'Professional, Scientific And Technical Activities',
        N: 'Rental, Travel Agencies, Business Support Services',
        O: 'Or Public Administration And Defense; Compulsory Social Insurance',
        P: 'Education',
        Q: 'Health and Social Care',
        R: 'Artistic, Sports, Entertainment And Fun Activities',
        S: 'Other Service Activities',
        T: 'Activities Of Families And Cohabitants As Employers Of Domestic Staff; Production Of Undifferentiated Goods And Services For Own Use By Families And Cohabitants',
        U: 'Extraterritorial Organizations And Bodies'
        '''
        conditions = [
            (df_['ateco_sector'] >= 1) & (df_['ateco_sector'] <= 3),
            (df_['ateco_sector'] >= 5) & (df_['ateco_sector'] <= 9),
            (df_['ateco_sector'] >= 10) & (df_['ateco_sector'] <= 33),
            (df_['ateco_sector'] == 35),
            (df_['ateco_sector'] >= 36) & (df_['ateco_sector'] <= 39),
            (df_['ateco_sector'] >= 41) & (df_['ateco_sector'] <= 43),
            (df_['ateco_sector'] >= 45) & (df_['ateco_sector'] <= 47),
            (df_['ateco_sector'] >= 49) & (df_['ateco_sector'] <= 53),
            (df_['ateco_sector'] >= 55) & (df_['ateco_sector'] <= 56),
            (df_['ateco_sector'] >= 58) & (df_['ateco_sector'] <= 63),
            (df_['ateco_sector'] >= 64) & (df_['ateco_sector'] <= 66),
            (df_['ateco_sector'] == 68),
            (df_['ateco_sector'] >= 69) & (df_['ateco_sector'] <= 75),
            # (df_['ateco_sector'] >= 77) & (df_['ateco_sector'] <= 84),
            (df_['ateco_sector'] >= 77) & (df_['ateco_sector'] <= 82),
            (df_['ateco_sector'] == 84),
            (df_['ateco_sector'] == 85),
            (df_['ateco_sector'] >= 86) & (df_['ateco_sector'] <= 88),
            (df_['ateco_sector'] >= 90) & (df_['ateco_sector'] <= 93),
            (df_['ateco_sector'] >= 94) & (df_['ateco_sector'] <= 96),
            (df_['ateco_sector'] >= 97) & (df_['ateco_sector'] <= 98),
            (df_['ateco_sector'] == 99)
        ]
        choices = list(string.ascii_uppercase[:21])
        # choices = list(string.ascii_uppercase[:14] + string.ascii_uppercase[15:21])
        df_['ateco_sector'] = np.select(conditions, choices)
        df_['ateco_sector'] = df_['ateco_sector'].astype('category')
        # 1: obliged to publish balance sheet, 0: not obliged to publish balance sheet
        df_['legal_struct'] = df_['legal_struct'].replace({"SRL": "1", "SRS": "1", "SRU": "1", "SPA": "1", "SAA": "0", "SAU": "0"})
        df_['stmt_date'] = pd.to_datetime(df_['stmt_date'], format="%Y-%m-%d")
        df_['def_date'] = pd.to_datetime(df_['def_date'], format="%d/%m/%Y")
        return df_

    # Generate label taking into account the time lag
    def generate_label(self, df_, offset_month=6, label_name='label'):
        self.target = label_name
        t = df_['stmt_date'].to_numpy().astype('datetime64[M]')
        df_[label_name] = ((t + np.timedelta64(offset_month, 'M') <= df_['def_date']) & \
                        (df_['def_date'] < t + np.timedelta64(offset_month + 12, 'M'))).astype(int)
        assert all(df_[pd.isnull(df_['def_date'])][label_name] == 0) # check all NaT rows are mapped to label 0
        return df_
    
    # Generate group labels from 0 to num_group - 1
    def generate_group(self, df_, group='asst_tot', num_group=3, new=True):
        self.group = 'size_group'
        if new:
            df_[self.group], self.group_bins = pd.qcut(df_[group], q=num_group, labels=False, retbins=True)
        else:
            # search the insertion point to compute the bins for the test data
            # capped at 0 and num_group - 1
            df_[self.group] = df_.apply(lambda x: max(min(bisect_left(self.group_bins, x[group]), num_group-1), 0), axis=1)
        assert df_[self.group].notna().all(), df_[self.group].isnull().sum()
        return df_
        
    # Generate and optionally transform the features (quantization by default)
    def generate_features(self, df_, use_knn, new=True):
        if new:
            # Transformation (numerical)
            if use_knn: 
                pipe = Pipeline([('scaler', QuantileTransformer()), ('knn_imputer', KNNImputer())])
            else:
                pipe = Pipeline([('scaler', QuantileTransformer())])
            pipe.fit(df_[self.num_cols])
            self.num_transformers = pipe
            self.transformed_num_cols = list(pipe.get_feature_names_out())
            # Transformation (categorical)
            enc = OneHotEncoder(drop='first', sparse=False).fit(df_[self.cat_cols])
            self.cat_transformers = [enc]
            self.transformed_cat_cols = list(enc.get_feature_names_out())
        
        # Transform num_cols
        df_[self.transformed_num_cols] = pd.DataFrame(self.num_transformers.transform(df_[self.num_cols]), 
                                                        index = df_.index,
                                                        columns=self.transformed_num_cols)
        if use_knn:
            for col in self.num_cols: assert df_[col].notnull().all(), f"{col}: {df_[col].isnull().sum()}"
        
        # Transform cat_cols
        for cat_transformer in self.cat_transformers:
            t = pd.DataFrame(cat_transformer.transform(df_[self.cat_cols]), 
                             index = df_.index,
                             columns=self.transformed_cat_cols)
        df_ = pd.concat((df_, t), axis=1)

        # Generate ratios
        DENOM_ROUND = 0.01 # avoid division by zero
        df_['debt_to_equity'] = (df_['debt_lt'] + df_['debt_st']) / (df_['eqty_tot'] + DENOM_ROUND)
        df_['current_ratio'] = df_['asst_current'] / (df_['debt_st'] + DENOM_ROUND)
        df_['operating_margin'] = df_['prof_operations'] / (df_['rev_operating'] + DENOM_ROUND)
        df_['ebitda_assets'] = df_['ebitda'] / (df_['asst_tot'] + DENOM_ROUND)
        df_['ebitda_margin'] = df_['ebitda'] / (df_['rev_operating'] + DENOM_ROUND)
        df_['nwc_ratio'] = df_['wc_net'] / (df_['asst_tot'] + DENOM_ROUND)
        df_['cashflow_to_debt'] = df_['cf_operations'] / (df_['debt_st'] + DENOM_ROUND)
        df_['profit_margin'] = df_['profit'] / (df_['rev_operating'] + DENOM_ROUND)
        df_['days_receivable_turnover'] = (df_['AR'] / (df_['rev_operating'] + DENOM_ROUND)) * 365
        df_['longtermliabilities_assets'] = df_['liab_lt'] / (df_['asst_tot'] + DENOM_ROUND)
        df_['interest_coverage_ratio'] = df_['prof_operations'] / (df_['exp_financing'] + DENOM_ROUND)
        df_['asset_turnover'] = df_['rev_operating'] / (df_['asst_tot'] + DENOM_ROUND)
        df_['equity_ratio'] = df_['eqty_tot'] / (df_['asst_tot'] + DENOM_ROUND)
        df_['quick_ratio'] = (df_['cash_and_equiv'] + df_['AR']) / (df_['debt_st'] + DENOM_ROUND)

        df_['current_asset_to_sales'] = df_['asst_current'] / (df_['rev_operating'] + DENOM_ROUND) # activity ratio
        df_['cash_and_securities_to_assets'] = df_['cash_and_equiv'] / (df_['asst_tot'] + DENOM_ROUND) # liquidity ratio
        df_['cashflow_to_interest_payment'] = df_['cf_operations'] / (df_['exp_financing'] + DENOM_ROUND) # debt coverage ratio
        
        df_['asset_minus_liab_asst_tot'] = (df_['asst_tot'] - df_['liab_lt']) / (df_['asst_tot'] + DENOM_ROUND)

        df_['altman'] = 0.717 * df_['asset_minus_liab_asst_tot'] + 0.847 * df_['roa'] + \
                        3.107 * df_['ebitda_assets'] + 0.420 * 1 / (df_['debt_to_equity'] + DENOM_ROUND) + 0.998 * df_['asset_turnover']

        # Transform ratio columns
        if new: 
            pipe = Pipeline([('scaler', QuantileTransformer())])
            pipe.fit(df_[self.ratio_cols])
            self.ratio_transformers = pipe
        df_[self.ratio_cols] = pd.DataFrame(self.ratio_transformers.transform(df_[self.ratio_cols]), 
                                            index = df_.index,
                                            columns=self.ratio_cols)
        
        if use_knn:
            for col in self.ratio_cols: assert df_[col].notnull().all(), f"{col}: {df_[col].isnull().sum()}"
        
        return df_
    
    def __call__(self, df, new, use_knn):
        df_ = df.copy()
        df_ = self.convert_dtypes(df_)
        df_ = self.imputer(df_)
        df_ = self.generate_features(df_, new=new, use_knn=use_knn)
        df_ = self.generate_label(df_)
        df_ = self.generate_group(df_, new=new)
        if new: self.df_ = df_ # save for debugging
        if new: 
            if not use_knn:
                # If transformation is not used, the model is able to handle the collinear/NA features, so we retain all columns
                return df_, self.num_cols + self.ratio_cols, \
                        [col for col in self.transformed_num_cols if col not in self.num_cols] + self.transformed_cat_cols, \
                        self.target, self.group
            else:
                # If transformation is used, we want to drop the original num_cols and only retain the ratios
                return df_, self.ratio_cols, \
                        [col for col in self.transformed_num_cols if col not in self.num_cols] + self.transformed_cat_cols, \
                        self.target, self.group
        else: 
            return df_