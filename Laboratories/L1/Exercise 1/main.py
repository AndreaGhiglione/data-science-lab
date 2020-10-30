import csv
import math

print('EX 1.1 - 1.2')
mean_list = [None, None, None, None]
sum_list = [0.0, 0.0, 0.0, 0.0]
std_deviation = [None, None, None, None]
length = 0
with open("iris.csv") as file:
    for row in csv.reader(file):  # csv.reader reads row by row
        length += 1
        sum_list[0] += float(row[0])
        sum_list[1] += float(row[1])
        sum_list[2] += float(row[2])
        sum_list[3] += float(row[3])
    n = length - 1
    mean_list[0] = 1/n * sum_list[0]
    mean_list[1] = 1/n * sum_list[1]
    mean_list[2] = 1/n * sum_list[2]
    mean_list[3] = 1/n * sum_list[3]
    sum_list = [0.0, 0.0, 0.0, 0.0]
with open("iris.csv") as file:
    for cols in csv.reader(file):
        sum_list[0] += (float(cols[0]) - mean_list[0]) ** 2
        sum_list[1] += (float(cols[1]) - mean_list[1]) ** 2
        sum_list[2] += (float(cols[2]) - mean_list[2]) ** 2
        sum_list[3] += (float(cols[3]) - mean_list[3]) ** 2
    std_deviation[0] = math.sqrt(1/n * sum_list[0])
    std_deviation[1] = math.sqrt(1/n * sum_list[1])
    std_deviation[2] = math.sqrt(1/n * sum_list[2])
    std_deviation[3] = math.sqrt(1/n * sum_list[3])
print(f"Sepal length, mean: {round(mean_list[0],2)} std_dev: {round(std_deviation[0],2)}")
print(f"Sepal width, mean: {round(mean_list[1],2)} std_dev: {round(std_deviation[1],2)}")
print(f"Petal length, mean: {round(mean_list[2],2)} std_dev: {round(std_deviation[2],2)}")
print(f"Petal width, mean: {round(mean_list[3],2)} std_dev: {round(std_deviation[3],2)}")


#1.3
print('\nEX 1.3')
iris_versicolor = []  # list of lists
iris_viriginica = []
iris_setosa = []
with open("iris.csv") as file:
    for row in csv.reader(file):
        tmp_list = []
        for cell in row:
            tmp_list.append(cell)
        tmp_list.pop(-1)
        if row[-1] == 'Iris-versicolor': iris_versicolor.append(tmp_list)
        if row[-1] == 'Iris-virginica': iris_viriginica.append(tmp_list)
        if row[-1] == 'Iris-setosa': iris_setosa.append(tmp_list)

def mean_std_dev(x):  # x is a list of lists
    n = len(x)
    sum_list = [0.0, 0.0, 0.0, 0.0]
    mean_list = [None, None, None, None]
    std_dev_list = [None, None, None, None]
    for row in x:
        for i in range(len(row)):
            sum_list[i] += float(row[i])
    for i in range(len(row)):
        mean_list[i] = 1/n * sum_list[i]
    sum_list = [0.0, 0.0, 0.0, 0.0]
    for row in x:
        for i in range(len(row)):
            sum_list[i] += (float(row[i]) - mean_list[i]) ** 2
    for i in range(len(row)):
        std_dev_list[i] = math.sqrt(1/n * sum_list[i])
    return [mean_list,std_dev_list]

print('[Sepal length, Sepal width, Petal length, Petal width]')
print(f'Iris-Versicolor, mean: {[round(el,2) for el in mean_std_dev(iris_versicolor)[0]]}' +
      f' std dev: {[round(el,2) for el in mean_std_dev(iris_versicolor)[1]]}')
print(f'Iris-Viriginica, mean: {[round(el,2) for el in mean_std_dev(iris_viriginica)[0]]}' +
      f' std dev: {[round(el,2) for el in mean_std_dev(iris_viriginica)[1]]}')
print(f'Iris-Setosa, mean: {[round(el,2) for el in mean_std_dev(iris_setosa)[0]]}' +
      f' std dev: {[round(el,2) for el in mean_std_dev(iris_setosa)[1]]}')
