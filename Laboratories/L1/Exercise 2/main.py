import json

print('EX 2.1')
with open('to-bike.json') as file:
    dataset = json.load(file)

print(f'keys: {list(dataset.keys())}')
# We understood that the dataset is a dictionary with one key called 'network' , we have to check 'network' type
print(type(dataset['network']))
# It's another dictionary, so let's check into it
print(f'Network keys: {dataset["network"].keys()}')
# We are interested in stations, let's check its type
print(type(dataset['network']['stations']))  # List
# A station looks like this:
print(dataset['network']['stations'][0])

print('\nEX 2.2')
# How many stations are actives ?
number_of_actives_stations = 0
for station in dataset['network']['stations']:  # Every station is a dictionary
    if station['extra']['status'] == 'online':
        number_of_actives_stations += 1
print(f'There are {number_of_actives_stations} active stations')

print("\nEX 2.3")
# How many bikes are available ?
number_of_available_bikes = sum([station['free_bikes'] for station in dataset['network']['stations']])
print(f'Availables bikes through the stations: {number_of_available_bikes}')
# How many free docks are there through the stations ?
number_of_free_docks = sum([station['empty_slots'] for station in dataset['network']['stations']])
print(f'Through the stations there are {number_of_free_docks} free docks')


print('\nEX 2.4')
# Given 2 coordinates of a point, identify the closest bike station to it that has available bikes
from math import cos, acos, sin, pi

def distance_coords(lat1, lng1, lat2, lng2):
    """Compute the distance among two points"""
    deg2rad = lambda x: x * 3.141592 / 180
    lat1, lng1, lat2, lng2 = map(deg2rad, [lat1, lng1, lat2, lng2])
    R = 6378100  # Radius of the Earth, in meters
    return R * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))

# latitude = float(input('Insert the latitude: '))  # Input return a string, we need a cast
# longitude = float(input('Insert the longitude: '))
latitude = 45.074512
longitude = 7.694419
best_distance = 999999
best_bikes = 0
best_name = None
for station in dataset['network']['stations']:
    if station['free_bikes'] != 0:
        latitude_2 = station['latitude']
        longitude_2 = station['longitude']
        current_distance = distance_coords(latitude, longitude, latitude_2, longitude_2)
        if current_distance < best_distance:
            best_distance = current_distance
            best_bikes = station['free_bikes']
            best_name = station['name']

# If I found a station with bikes available:
if best_bikes != 0:
    print(f'The closest station with available bikes is {best_name} , which is {round(best_distance,2)} meters from here,'
          f' with {best_bikes} bikes availables')
else:
    print('At the moment there are any stations with availables bikes')