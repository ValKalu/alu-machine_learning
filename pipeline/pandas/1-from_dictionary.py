#!/user/bin/enc python3
import pandas as pd

#Data dictionary
data= {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

#Row labels Definition
index_labels = ['A', 'B', 'C', 'D']

#DataFrame
df = pd.DataFrame(data, index=index_labels)

#This is for testing script
if __name__ == "__main__":
    print(df)
