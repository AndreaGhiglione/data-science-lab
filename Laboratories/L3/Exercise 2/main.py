import json
import itertools
import copy
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
import timeit

print('Ex2.1')
def count_support(itemset,itemsets):  # (itemset for which we want to count the support, all itemsets of the dataset)
    sup_count = 0
    for curr_itemset in itemsets:
        if all(item in curr_itemset for item in itemset):  # If itemset is a subset of curr_itemset (must be a check between list-tuple or both list and tuple
            sup_count += 1
    return sup_count/len(itemsets)

def pruning_check(c_itemset,l_itemsets):
    prune = False
    subsets = set(itertools.combinations(c_itemset,len(c_itemset)-1))  # Set of tuples (containing the subsets)
    for subset in subsets:
        present = False
        for itemset in l_itemsets.keys():
            if isinstance(itemset,str):
                if subset == tuple({itemset}):
                    present = True
                    break
            else:
                if all(item in itemset for item in subset):
                    present = True
                    break
        if not present:
            prune = True
            break
    return prune

def my_apriori(itemsets,minsup):
    c = {}  # First we calculate C1
    for itemset in itemsets:
        for item in itemset:
            if item not in c.keys():
                cnt_supp = count_support(list({item}), itemsets)
                if cnt_supp > minsup:
                    c[item] = cnt_supp
    c = dict(sorted(c.items()))  # Sorting the dictionary C (of candidates) by key
    result = copy.deepcopy(c)
    items = c.keys()
    # Once we have C1, we can code the next step of the algorithm
    flag = True
    while flag:
        new_c = {}  # It will contain the new candidates (example: if we are at first iteration it will contain C2)
        for itemset in c.keys():  # c.keys are the current candidates
            for itemset2 in items:  # items are the "base" items, in this case a,b,c,d,e
                if isinstance(itemset, tuple):
                    if all(item in itemset for item in {itemset2}):  # I append the item to the current candidate itemset only if item is not included in itemset
                        continue
                    possible_itemset = set(item for item in itemset) | set({item for item in {itemset2}})
                else:
                    if all(item in {itemset} for item in {itemset2}):
                        continue
                    possible_itemset = set(item for item in {itemset}) | set({item for item in {itemset2}})
                if not pruning_check(possible_itemset, c):  # I prune itemsets which subsets are not frequent in Li (current candidates)
                    pi = tuple(possible_itemset)
                    if pi not in new_c.keys():
                        cnt_supp = count_support(pi, itemsets)
                        if cnt_supp > minsup:  # If the support is greater than minimum support requested I add the new candidate to the new set of candidates
                            new_c[tuple(sorted(possible_itemset))] = cnt_supp
                            result[tuple(sorted(possible_itemset))] = cnt_supp
        c = new_c
        if not c.keys():
            flag = False
    return result


dataset = [['a','b'],
           ['b','c','d'],
           ['a','c','d','e'],
           ['a','d','e'],
           ['a','b','c'],
           ['a','b','c','d'],
           ['b','c'],
           ['a','b','c'],
           ['a','b','d'],
           ['b', 'c','e']]

itemsets = my_apriori(dataset,0.1)
for itemset in itemsets.keys():
    print(f'{itemset} -> {itemsets[itemset]}')


print('\nEx2.2')
with open('modified_coco.json') as file:
    dataset = json.load(file)
    annotations = [data['annotations'] for data in dataset]
    # I save the annotations in c1 :
    c1 = set([item for annotation in annotations for item in annotation])
    c1 = sorted(c1)

print('\nEx2.3')
itemsets = my_apriori(annotations,0.02)
for itemset in itemsets.keys():
    print(f'{itemset} -> {itemsets[itemset]}')

print('\nEx2.4')
matrix = []  # It will be a list of lists
N = len(annotations)
M = len(c1)
for i in range(N):
    tmp_line = [0]*M
    for j in range(M):
        if c1[j] in annotations[i]:
            tmp_line[j] = 1
    matrix.append(tmp_line)

df = pd.DataFrame(data=matrix, columns=c1)
fi = fpgrowth(df,0.02)

print(fi.to_string())

print('\nEx2.5')
print(f'fpgrowth time: {timeit.timeit(lambda: fpgrowth(df, 0.02), number=1)}')
print(f'apriori time: {timeit.timeit(lambda: apriori(df, 0.02), number=1)}')
print(f'my_apriori time: {timeit.timeit(lambda: my_apriori(annotations, 0.02), number=1)}')