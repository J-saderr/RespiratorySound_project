Set up the virtual envroment for python 2.13.8
- python -m venv .venv
- source .venv/bin/activate   # On macOS/Linux
- .venv\Scripts\activate      # On Windows

Install necessary lib packages
Scripts in sequence:
1. Running the A.py first to get df.csv file
2. Running the feature_selection.py to generate df_updated_file.csv
3. Running the DL.py for predictions
