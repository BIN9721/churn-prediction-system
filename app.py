import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import uvicorn
import imblearn
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware


#1. Initialize the App
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="The API predicts the likelihood of bank customers leaving based on their transaction behavior.",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. Load Model Pipeline
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'models', 'churn_prediction_pipeline.pkl')
    model_pipeline = joblib.load(file_path)  
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# 3. Define input data structures (Input Schema)
# Use Pydantic to validate data right from the input
class CustomerProfile(BaseModel):
    Customer_Age: int
    Gender: str
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: Optional[float] = None
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


#4. Feature Engineering Function (Recreate logic from step 1)
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    if 'Months_on_book' in df_new.columns:
        df_new['Age_at_Onboarding'] = df_new['Customer_Age'] - (df_new['Months_on_book'] / 12)
    # Add 0.001 to avoid dividing by 0.
    df_new['Avg_Trans_Value'] = df_new['Total_Trans_Amt'] / (df_new['Total_Trans_Ct'] + 0.001)
    df_new['Trans_to_Limit_Ratio'] = df_new['Total_Trans_Amt'] / (df_new['Credit_Limit'] + 0.001)
    df_new['Activity_Per_Relationship'] = df_new['Total_Trans_Ct'] / (df_new['Total_Relationship_Count'] + 0.001)

    cols_to_drop = ['Months_on_book']
    df_new = df_new.drop(columns=cols_to_drop, errors='ignore')

    expected_cols = [
        'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category', 'Total_Relationship_Count', 
        'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 
        'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio', 'Age_at_Onboarding', 'Avg_Trans_Value', 
        'Trans_to_Limit_Ratio', 'Activity_Per_Relationship'
    ]
    df_final = df_new[expected_cols]

    return df_final

# 5. API Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction API. Go to /docs to test."}

@app.post("/predict")
def predict_churn(customer: CustomerProfile):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # B1: Convert Input (JSON) to DataFrame
    input_data = pd.DataFrame([customer.dict()])
    
    # B2: Create a new Feature (IMPORTANT: The Model needs these columns)
    processed_data = add_engineered_features(input_data)
    
    # B3: Forecast
    try:
        prediction = model_pipeline.predict(processed_data)[0]
        probability = model_pipeline.predict_proba(processed_data)[0][1] # Get the Churn probability (class 1)
        
        # B4: Return the results
        result = "Attrited Customer" if prediction == 1 else "Existing Customer"
        
        # Probability-Based Action Suggestions (Based on Business Analysis)
        action = "No Action"
        if probability >= 0.7:
            action = "CRITICAL RISK: Call Immediately & Offer VIP Retention Package ($20 cost)"
            
        elif probability >= 0.3:
            action = "HIGH RISK: Send $10 Discount Voucher"
            
        elif probability >= 0.02:
            action = "LOW RISK (WATCHLIST): Send Automated 'We Miss You' Email"
            
        return {
            "prediction": result,
            "churn_probability": round(float(probability), 4),
            "recommended_action": action
        }
        
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# Run the server if the file is to be executed directly.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)