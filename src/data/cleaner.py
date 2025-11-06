from utils.logger import default_logger as logger


class DataCleaner:    
    def __init__(self, df, 
                 irrelevant_features, categorical_features, numerical_features, target):
        self.df = df.copy()
        self.irrelevant_features = irrelevant_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target = target
        logger.info("Initialized DataCleaner")


    def _drop_irrelevant_features(self):
        self.df = self.df.drop(columns=self.irrelevant_features)
        return self.df


    def _missing_values_handling(self, nan_cols):
        # Missing means doesnt have.
        cols_fill_none = [
            'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
        ]
        for col in cols_fill_none:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("N/A")

        # Fill nan with median.
        if 'LotFrontage' in nan_cols:
            self.df["LotFrontage"] = self.df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median())
            )

        # Fill nan with zero
        cols_fill_zero = [
            'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
        ]
        for col in cols_fill_zero:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Drop rows.
        cols_to_drop = ["Electrical"]
        for col in nan_cols:
            if col in cols_to_drop:
                self.df = self.df[self.df[col].notna()]

        return self.df


    def _outlier_handling(self, numerical_cols, lower_pct=0.01, upper_pct=0.99):
        for col in numerical_cols:
            lower_bound = self.df[col].quantile(lower_pct)
            upper_bound = self.df[col].quantile(upper_pct)
            mask_outlier = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_indices = self.df[mask_outlier].index

            if len(outlier_indices) <= 2 and len(outlier_indices) > 0:
                self.df = self.df.drop(outlier_indices)

        return self.df


    def clean(self):
        try:
            df_clean = self._drop_irrelevant_features()

            nan_cols = df_clean.columns[df_clean.isna().any()].tolist()
            df_clean = self._missing_values_handling(nan_cols)

            df_clean = self._outlier_handling(self.numerical_features)

            logger.info("Data cleaner completed")

            X = df_clean[self.numerical_features + self.categorical_features]
            y = df_clean[self.target]

            return X, y

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
