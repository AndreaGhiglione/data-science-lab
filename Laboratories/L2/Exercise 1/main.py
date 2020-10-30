import csv

print('EX1.1 - 1.2')
dataset = []
with open('GLT_filtered.csv') as file:
    next(csv.reader(file))  # I skip the first line of the csv
    for row in csv.reader(file):
        dataset.append(row)

dataset_length = len(dataset)
closest_antecedent = None
closest_successive = None
city = None
for i in range(0,dataset_length):
    if dataset[i][1] == '':  # AverageTemperature is the field 1 of the list dataset[i]
        closest_antecedent = 0
        closest_successive = 0
        city = dataset[i][3]
        for j in range(i,0,-1):
            if dataset[j][1] != '' and dataset[j][3] == city:
                closest_antecedent = float(dataset[j][1])
                break
        for j in range(i,dataset_length):
            if dataset[j][1] != '' and dataset[j][3] == city:
                closest_successive = float(dataset[j][1])
                break
        dataset[i][1] = (closest_antecedent + closest_successive) / 2


print('\nEX 1.3')
def print_n_measurements(city, N):
    measurements = []
    for row_measure in dataset:
        if row_measure[3] == city:
            measurements.append(round(float(row_measure[1]),3))
    measurements.sort()
    print(f'Top {N} coldest measurements in {city} : {measurements[0:N]}')
    print(f'Top {N} hottest measurements in {city} : {list(reversed(measurements))[0:N]}')

print_n_measurements('Abidjan',3)
