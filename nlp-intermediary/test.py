import pandas as pd

if __name__ == "__main__":
    # lendo o arquivo csv que contém as sentenças e criando um dataframe
    df = pd.read_csv('./outputs/final.csv')

    print(df['sentiment'].value_counts())
    print(df['dictionary_classification'].value_counts())
    exit(0)

    df = df.loc[df['dictionary_classification'] == '-1']



    print(df['sentiment'].value_counts())