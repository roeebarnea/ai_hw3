import pandas as pd



if __name__ == "__main__":
    data = pd.read_csv("test.csv")
    df = pd.DataFrame(data)
    print(df['area_mean'][0])
