Set up the virtual envroment for python 2.12.8
- python -m venv .venv
- source .venv/bin/activate   # On macOS/Linux
- .venv\Scripts\activate      # On Windows

Install necessary lib packages

Scripts in sequence:
1. Run the DataVisualize to have the df.csv file  
2. Running DataVisualize to visualize the plots.
3. Run the DataPreprocessing to archive df_updated_padded.csv
4. Running the DL.py for predictions
