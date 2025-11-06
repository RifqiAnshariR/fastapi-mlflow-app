from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logger import default_logger as logger


class DataPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):  # From df_clean.
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.preprocessor = self._preprocessing_pipeline()
        logger.info("Initialized DataPreprocessor")


    def _preprocessing_pipeline(self):
        numerical_transformer = Pipeline(steps=[
            ('distribution_normalizer', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
      
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical_transformer', numerical_transformer, self.numerical_cols),
                ('categorical_transformer', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        logger.info(f"Preprocessor prepared")

        return preprocessor
