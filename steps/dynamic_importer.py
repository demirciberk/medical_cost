import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
        "age": [25],
        "sex": ["male"],
        "bmi": [22.0],
        "children": [0],
        "smoker": ["no"],
        "region": ["southeast"],
        
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
