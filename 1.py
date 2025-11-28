import csv
import random

names = ['Anuj', 'Sanchit', 'Aditya', 'Sagar', 'Prateek']
branches = ['COE', 'IT', 'SE', 'MCE']
years = [1, 2, 3, 4]
cgpas = [7.8, 8.5, 9.0, 9.1, 9.3, 9.5]


with open('university_records.csv', 'w', newline='') as csvfile:
    fieldnames = ['name', 'branch', 'year', 'cgpa']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(20):  
        row = {
            'name': random.choice(names),
            'branch': random.choice(branches),
            'year': random.choice(years),
            'cgpa': random.choice(cgpas)
        }
        writer.writerow(row)

with open('university_records.csv', mode='r') as file:
    csvFile = csv.DictReader(file)
    for line in csvFile:
        print(line)