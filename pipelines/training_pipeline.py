from zenml import pipeline, step, Model
from steps.data_loading_step import data_loader
@pipeline(model = Model(name='cost_predictor'),)
def training_pipeline():
    raw_data =  data_loader()
    return raw_data
    
if __name__ == "__main__":
    run = training_pipeline()



