import pandas as pd

if __name__ == '__main__':
    df = pd.read_feather('train_ver2.fth')
    print(df)
