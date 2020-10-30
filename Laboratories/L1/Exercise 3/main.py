import csv

print('\nEX 3.1 - 3.2')
input_row = 130
dataset = []
labels = []
counter = 0
with open("mnist_test.csv") as file:
    for row in csv.reader(file):
        labels.append(int(row.pop(0)))  # The first digit shows the draw of the number in the MNIST
        dataset.append(list(map(int,row)))  # Dataset is a list of lists of integers
    signs = list(map(lambda x: ' ' if 0 <= x < 64
                          else '.' if 64 <= x < 128
                          else '*' if 128 <= x < 192
                          else '#' , dataset[input_row-1]))  # Assumption of no errors in the database
    for number in signs:
        if counter == 28:
            counter = 0
            print('')
        print(number, end='')  # No new-line
        counter += 1

print('\nEX 3.3')

# Euclidean distance between pairs of vectors
def euclidean_distance(x, y):
    return sum([(x_i - y_i) ** 2 for x_i,y_i in zip(x,y)]) ** 0.5

positions = [25, 29, 31, 34]
for i in range(len(positions)):
    for j in range(i+1,len(positions)):
        print(positions[i], positions[j], euclidean_distance(dataset[positions[i]],dataset[positions[j]]))


print('\nEX 3.4')
print('Looking at the euclidean distance I can guess: ')
print('25: 0 , 29: 1 , 31: 1 , 34: 7')

print('\nEX 3.5')
ZERO = [0] * 784  # I initialize the two lists of counter at 0
ONE = [0] * 784
for i in range(len(labels)):
    if labels[i] == 0:
        for j in range(len(dataset[i])):  # Length of dataset[i] is 784
            if dataset[i][j] >= 128:
                ZERO[j] += 1
    else:
        if labels[i] == 1:
            for j in range(len(dataset[i])):
                if dataset[i][j] >= 128:
                    ONE[j] += 1

index_max_diff = None
differences = []
for i in range(len(ZERO)):
    diff = abs(ZERO[i]-ONE[i])
    differences.append(diff)
    if diff >= max(differences):
        index_max_diff = i


print(f'The biggest difference between 0 and 1 is {max(differences)}, which is the {index_max_diff} pixel')

for i in range(28):
    print('')
    for j in range(28):
        if i*28 + j != index_max_diff:
            print(' ', end='')
        else:
            print('*')

print('\nAs we can see, the index is around the centre, because of the shape of 0 and 1')
