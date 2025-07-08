import sys
import pickle
from estimator import *
from preprocessor import *
from imputer import *

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print("""Error: please use the following complaint arguments:
python3 harness.py --input_csv  <input file in csv> --output_csv <output csv file path to which the predictions are written>
        """)
    print("Kindly allow up to a minute before completion. Predicting...")
    input_file, output_file = args[2], args[4]
    
    with open("estimator.pkl", "rb") as f:
        estimator = pickle.load(f)
    
    preprocessor = estimator.preprocessor
    
    data = pd.read_csv(input_file)
    predictions = estimator.predict(preprocessor(data, use_knn=False, new=False))
    predictions.to_csv(output_file, header=False, index=False)  
    



    