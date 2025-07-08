import sys
import inspect
import numpy as np
import pandas as pd

class Imputer():
    def __init__(self, verbose):
        self.verbose = verbose
        self.DENOM_ROUND = 1e-9
        self.knn_imputer = None
    """
        Helper function to impute the missing values, using the known formula. If one row contains
        more than one missing value among `cols`, the function amounts to do nothing
        Input: 
            - df_: dataframe (copied)
            - cols: all columns involved in the imputation
            - signs: support only either '+' or '-'
        Output:
            - df_: dataframe (imputed)
    """
    def __calculate (self, df_, cols, signs):
        if self.verbose:
            print(f"{sys._getframe(1).f_code.co_name}: total mismatch {self.__check(df_, cols, signs)} rows, \
    imputing {self.__check(df_, cols, signs, na_count=1)} rows with 1 NA!")
        # Impute column by column
        for target, target_col in enumerate(cols):
            mask = df_[target_col].isna() # Extract potential imputable entries in the target column 
            
            # Example: a = b - c - d - e
            # To impute column d, we keep a's and d's sign, and flip all the rest, equivalently a - b + c - e = -d
            if target == 0: 
                flipped_signs = signs
            else: 
                flipped_signs = [sign if i == target or i == 0 else '-' if sign == '+' else '+' for i, sign in enumerate(signs)]
            
            res = pd.Series(0, index=df_.loc[mask].index) # placeholder
            for col, sign in zip(cols, flipped_signs):
                if col == target_col: 
                    continue
                if sign == '+': 
                    res += df_.loc[mask, col]
                else: 
                    res -= df_.loc[mask, col]
            # if there is an initial negative sign before the target column
            df_.loc[mask, target_col] = res if signs[target] == '+' else -res
        # there should be no more rows with only 1 missing values
        assert df_[df_[cols].isna().sum(axis=1) == 1].empty
        # Check NA conditions after the imputation
        if self.verbose:
            print(f"After imputation, there are still {self.__check(df_, cols, signs)} mismatched rows!")
            print("----------------------------------------------------------------")
        return df_
    
    '''
        Helper function to check the number of mismatched rows (with `na_count` NAs) according to the given formula
        Input: 
            - Same as __calculate
            - na_count: number of NAs in the row, 0 <= na_count <= len(cols)
        Output:
            - num_mismatch: number of mismatched rows
    '''
    def __check(self, df_, cols, signs, na_count=None):
        if na_count is None: # full dataset
            df_subset = df_
        else:
            df_subset = df_[df_[cols].isna().sum(axis=1) == na_count]
        left = df_subset[cols[0]]
        right = pd.Series(0, index=left.index)
        for col_right, sign_right in zip(cols[1:], signs[1:]):
            if sign_right == '+': right += df_subset[col_right]
            else: right -= df_subset[col_right]
        num_mismatch = sum(~np.isclose(left - right, 0))
        return num_mismatch

    def impute_margin_fin(self, df_, cols=['margin_fin', 'eqty_tot', 'asst_fixed_fin', 'asst_intang_fixed', 'asst_tang_fixed']):
        if self.verbose: print("Rule: {} = {} - {} - {} - {}".format(*cols))
        return self.__calculate(df_, cols, ['+'] * 2 + ['-'] * 3)
    def impute_cf_operations(self, df_, cols=['cf_operations', 'ebitda', 'profit', 'inc_financing', 'inc_extraord', 'taxes']):
        if self.verbose: print("Rule: {} = {} + {} + {} + {} - {}".format(*cols))
        return self.__calculate(df_, cols, ['+'] * 5 + ['-'])
    def impute_wc_net(self, df_, cols=['wc_net', 'asst_current', 'debt_st']):
        if self.verbose: print("Rule: {} = {} - {}".format(*cols))
        return self.__calculate(df_, cols, ['+'] * 2 + ['-'])
    def impute_prof_operating(self, df_, cols=['prof_operations', 'rev_operating', 'COGS']):
        if self.verbose: print("Rule: {} = {} - {}".format(*cols))
        return self.__calculate(df_, cols, ['+'] * 2 + ['-'])
    def impute_roe(self, df_, cols=['roe', 'profit', 'eqty_tot'], atol=1e-2):
        if self.verbose: print("Rule: {} = {} / {} * 100".format(*cols))
       
        num_mismatch = np.sum(~np.isclose(df_['roe'], df_['profit'] / df_['eqty_tot'] * 100, atol=atol))
        if self.verbose: print(f"{sys._getframe(0).f_code.co_name}: total mismatch {num_mismatch} rows")
       
        df_.loc[df_['roe'].isna(), 'roe'] = (df_['profit'] / (df_['eqty_tot'] + self.DENOM_ROUND) * 100).round(2)
        df_.loc[df_['profit'].isna(), 'profit'] = df_['roe'] * (df_['eqty_tot'] + self.DENOM_ROUND) / 100
        df_.loc[df_['eqty_tot'].isna(), 'eqty_tot'] = df_['profit'] / (df_['roe'] + self.DENOM_ROUND) / 100

        num_mismatch = np.sum(~np.isclose(df_['roe'], df_['profit'] / df_['eqty_tot'] * 100, atol=atol))
        if self.verbose:
            print(f"After imputation, there are still {num_mismatch} mismatched rows!")
            print("----------------------------------------------------------------")
        return df_

    def __call__(self, df_):
        # Impute NaNs using the rules
        function_list = [member for name, member in inspect.getmembers(self.__class__, inspect.isfunction) if name.startswith("impute")]
        for function in function_list:
            df_ = function(self, df_)
        return df_