import os
import zipfile
import kaggle

def download_kaggle_competition_data(competition_name, folder="data"):
    if not os.path.exists(folder) or not os.listdir(folder):
        kaggle.api.authenticate() 
        try:
            # Download data
            kaggle.api.competition_download_files(competition_name, path=folder)
            
            # Extract
            for file in os.listdir(folder):
                if file.endswith(".zip"):
                    zip_path = os.path.join(folder, file)
                    print(f"Unzipping '{file}'...")
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(folder)
                    
                    os.remove(zip_path)
                    
            print("Data Downloaded.")
            
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print(f"Data already exists. Skipping download.")

# Test
if __name__ == "__main__":
    COMPETITION = "elec-378-sp-26-final-project"
    DATA_DIR = "Data"
    
    download_kaggle_competition_data(COMPETITION, DATA_DIR)