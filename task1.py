import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# 1. DEFINE FOLDERS & FILES
# ==========================================
# CHANGE THESE to match your actual folder names
main_folder = "./Data Set For Task"      
churn_subfolder = "./Churn Prediction Data"  

# 4 files located directly inside the main folder
root_files = [
    os.path.join(main_folder, "1) iris.csv"),
    os.path.join(main_folder, "2) Stock Prices Data Set.csv"),
    os.path.join(main_folder, "3) Sentiment dataset.csv"),
    os.path.join(main_folder, "4) house Prediction Data Set.csv")
]

# 2 files located inside the subfolder, which is inside the main folder
subfolder_files = [
    os.path.join(main_folder, churn_subfolder, "churn-bigml-20.csv"),
    os.path.join(main_folder, churn_subfolder, "churn-bigml-80.csv")
]

output_folder = "preprocessed_data"
os.makedirs(output_folder, exist_ok=True)

# Combine all target files into one list
all_files = root_files + subfolder_files

def universal_clean(df):
    """Fills missing numbers with median and text with mode."""
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    
    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if len(cat_cols) > 0:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df

# ==========================================
# 2. PROCESS EACH FILE
# ==========================================
for file_path in all_files:
    if not os.path.exists(file_path):
        print(f"[SKIPPING] File not found: {file_path}")
        continue
        
    print(f"Processing: {file_path}...")
    file_name = os.path.basename(file_path)
    
    # Handle the House Prediction dataset (it uses spaces instead of commas)
    if "house" in file_name.lower():
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
    else:
        df = pd.read_csv(file_path)
    
    # Drop junk "Unnamed" index columns if they exist
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    
    # Clean missing data automatically
    df = universal_clean(df)
    
    # --- Custom Logic Based on Dataset Type ---
    le = LabelEncoder()
    scaler = StandardScaler()
    
    if "iris" in file_name.lower():
        df['species'] = le.fit_transform(df['species'])
        
    elif "sentiment" in file_name.lower():
        if 'Sentiment' in df.columns:
            df['Sentiment'] = le.fit_transform(df['Sentiment'])
            
    elif "churn" in file_name.lower():
        binary_cols = ['International plan', 'Voice mail plan', 'Churn']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
        if 'State' in df.columns:
            df['State'] = le.fit_transform(df['State'])
            
    # Save the processed DataFrame to the new folder
    output_path = os.path.join(output_folder, f"processed_{file_name}")
    df.to_csv(output_path, index=False)
    print(f" -> Saved clean file to: {output_path}")

print("\nAll files successfully preprocessed and saved!")