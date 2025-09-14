from zenml import pipeline, step, Model
from steps.data_loading_step import data_loader
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.transformer_step import transformer_step

@pipeline(model = Model(name='cost_predictor'),)
def training_pipeline():
    raw_data =  data_loader()
    X_train, X_test, y_train, y_test = data_splitter_step(raw_data, target_column="charges")
    
    # Log transformation for 'children' column
    X_train_log_transformed, log_transformer = feature_engineering_step(df=X_train, strategy="log", features=["children"])
    X_test_log_transformed = transformer_step(df=X_test, engineer=log_transformer)

    # Binary encoding for 'sex' and 'smoker' columns
    X_train_binary_transformed, binary_encoder = feature_engineering_step(df=X_train_log_transformed, strategy="binary_encoding", features=["sex", "smoker"])
    X_test_binary_transformed = transformer_step(df=X_test_log_transformed, engineer=binary_encoder)

    # One-hot encoding for 'region' column
    X_train_transformed, ohe_encoder = feature_engineering_step(df=X_train_binary_transformed, strategy="onehot_encoding", features=["region"])
    X_test_transformed = transformer_step(df=X_test_binary_transformed, engineer=ohe_encoder)
    return X_train_transformed, X_test_transformed, y_train, y_test
    
if __name__ == "__main__":
    run = training_pipeline()