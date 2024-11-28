import pandas as pd

def update_to_csv(model_name, mae, mape, mse, rmse, r2):
    file = "D:/SAMLFRFM/notebooks/model_metrics.csv"
    
    # Read the existing metrics
    existing_metrics = pd.read_csv(file)
    
    # Check if the model exists
    model_exists = existing_metrics["Model"].str.contains(model_name).any()
    
    if model_exists:
        # Get the row corresponding to the model
        previous_metrics = existing_metrics[existing_metrics["Model"] == model_name].iloc[0]
        
        # Check if the RMSE is better
        if rmse < previous_metrics["RMSE"]:
            # Remove the old row
            existing_metrics = existing_metrics[existing_metrics["Model"] != model_name]
            
            # Prepare new metrics row
            new_metrics = pd.DataFrame({
                "Model": [model_name],
                "MAE": [mae],
                "MAPE": [mape],
                "MSE": [mse],
                "RMSE": [rmse],
                "R2": [r2]
            })
            
            # Append the new metrics
            updated_metrics = pd.concat([existing_metrics, new_metrics], ignore_index=True)
            
            # Save the updated metrics to the file
            updated_metrics.to_csv(file, index=False)
    else:
        # If the model does not exist, add a new row
        new_metrics = pd.DataFrame({
            "Model": [model_name],
            "MAE": [mae],
            "MAPE": [mape],
            "MSE": [mse],
            "RMSE": [rmse],
            "R2": [r2]
        })
        updated_metrics = pd.concat([existing_metrics, new_metrics], ignore_index=True)
        updated_metrics.to_csv(file, index=False)