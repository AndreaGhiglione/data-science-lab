import csv
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

print('\nEX1.1')
with open('online_retail.csv') as file:
    next(csv.reader(file))
    dataset = []
    for line in csv.reader(file):
        if line[0][0] != 'C':
            dataset.append(line)
print('The dataset has been cleaned from lines we wanted to delete')


print('\nEX1.2')
invoices_dictionary = {}  # A dictionary where, to each invoice number, are associated its items (a list)
items = []  # A list which will contain all the possible items of the dataset
for line in dataset:
    if line[0] in invoices_dictionary:
        invoices_dictionary[line[0]].append(line[2])
    else:
        invoices_dictionary[line[0]] = []
        invoices_dictionary[line[0]].append(line[2])
    if line[2] not in items:
        items.append(line[2])

invoices_itemsets = [itemset for itemset in invoices_dictionary.values()]  # List of lists of items
N = len(invoices_itemsets)
M = len(items)
print(f'There are {N} invoices itemsets and {M} different items')

print('\nEX1.3')
items.sort()
matrix = []  # It will be a list of lists
for i in range(N):
    tmp_line = [0]*M
    for j in range(M):
        if items[j] in invoices_itemsets[i]:
            tmp_line[j] = 1
    matrix.append(tmp_line)

df = pd.DataFrame(data=matrix, columns=items)  # df will be a boolean matrix with 22064 rows (invoices different numbers) and 4208 columns (items)
print(df)

print('\nEX1.4')
min_sup = [0.5, 0.1, 0.05, 0.02, 0.01]
for sup in min_sup:
    print(f'Support: {min_sup}')
    fi = fpgrowth(df, sup)
    print(len(fi))
    print(fi.to_string())

print('The less the minsup is and the more itemsets we find, as expected')


print('\nEX1.5')
fi = fpgrowth(df, 0.02)
print(len(fi))
print(fi.to_string())


print('\nEX1.6')
# We can calculate support with this function, or in a compact form (below the function)
def calculate_support(items, mat):
    frequency_occurence = 0
    for row in range(len(mat)):
        flag = True
        for item_index in items:
            if mat[row][item_index] != 1:
                flag = False
                break
        if flag:
            frequency_occurence += 1
    return frequency_occurence/len(mat)

print('For example we will consider the itemset (2566,1599)')
M = df.values  # matrix from the df dataframe
support_2656 = len(M[M[:, 2656] == 1])/len(M)  # M[M[:,2656] == 1] selects all the rows where the item 2656 (column 2656) is present
support_1599 = len(M[M[:, 1599] == 1])/len(M)
support_both = len(M[(M[:, 2656] == 1) & (M[:, 1599] == 1)])/len(M)
print(f"Confidence 2656 => 1599: {calculate_support([2656,1599], M) / calculate_support([2656], M)}")
print(f"Confidence 1599 => 2656: {support_both / support_1599}")


print('\nEX1.7')
fi = fpgrowth(df, 0.01)
print(association_rules(fi,'confidence',0.85))
