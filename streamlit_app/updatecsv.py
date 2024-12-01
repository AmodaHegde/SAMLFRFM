#imports
import pandas as pd

#function to update csv conditionally
def update_to_csv(file, ticker, mae, mape, mse, rmse, r2):

    # Read the existing metrics of the particular model
    existing_metrics = pd.read_csv(file)
    
    # Check if the stock exists
    stock_exists = existing_metrics["Stock"].str.contains(ticker).any()
    
    if stock_exists:
        # Get the row corresponding to the stock
        previous_metrics = existing_metrics[existing_metrics["Stock"] == ticker].iloc[0]
        
        # Check if the RMSE is better
        if rmse < previous_metrics["RMSE"]:
            # Remove the old row
            existing_metrics = existing_metrics[existing_metrics["Stock"] != ticker]
            
            # Prepare new metrics row
            new_metrics = pd.DataFrame({
                "Stock": [ticker],
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
        # If the stock does not exist, add a new row
        new_metrics = pd.DataFrame({
            "Stock": [ticker],
            "MAE": [mae],
            "MAPE": [mape],
            "MSE": [mse],
            "RMSE": [rmse],
            "R2": [r2]
        })
        updated_metrics = pd.concat([existing_metrics, new_metrics], ignore_index=True)
        updated_metrics.to_csv(file, index=False)