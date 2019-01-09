#coding:utf-8

file_name = '../dataset2/word_df.txt'

vocab = []

max_df = -1
min_df = 1.2

with open(file_name) as file:
    for line in file:
        word, df = line.strip().split()
        df = float(df)
        max_df = max(df, max_df)
        min_df = min(df, min_df)
        if 0.004<df<0.401:
            vocab.append(word)

print(min_df, max_df)

vocab.sort()

with open('../dataset2/vocab_add.txt', 'w') as file:
    for word in vocab:
        file.write('{}\n'.format(word))