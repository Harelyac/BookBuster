import pandas as pd

def main():
    file1 = "books_no_movies.csv"
    df1 = pd.read_csv(file1)
    df1.insert(6,"into_movie",0)
    #df.to_csv(file)
    #df1.drop(0)

    file2 = "booksIntoMovies.csv"
    df2 = pd.read_csv(file2)
    df2.insert(6,"into_movie",1)
    #df.to_csv(file)
    #df2.drop(0)

    file3 = "merged.csv"
    df3 = pd.concat([df1, df2]).reset_index(drop=True)
    #df3.sort_values('2')
    df3.to_csv(file3, index=True)

if __name__ == '__main__':
    main()