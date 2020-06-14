import csv

# ? Reading a csv file and printing.
# with open('data.csv', 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
    
#     # To allow us to skip the first line.
#     next(csv_reader)

#     for line in csv_reader:

# ? Write the contents into a new file, with - delimiter

# with open('data.csv', 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)

#     with open('new_data.csv', 'w') as new_csv_file:
#         csv_writer = csv.writer(new_csv_file, delimiter='-')

#         for line in csv_reader:
#             csv_writer.writerow(line)

# ? You can also use Dict methods, to Read or Write as a Dictionary.
# ? Here I'm going to read as a Dictionary, and then use the values
# ? to store into a list of Dictionary Tuples.

# list_values = []
# with open('data.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
    
#     for line in csv_reader:
#         dict_values = {}
#         dict_values['PassengerId'] = line['PassengerId']
#         dict_values['Survived'] = line['Survived']
#         list_values.append(dict_values)
#     print(list_values)

# ? Writing using WriteDict

# with open('data.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)

#     with open('new_data.csv', 'w') as new_csv_file:
#         field_names = csv_reader.fieldnames
        
#         csv_writer = csv.DictWriter(new_csv_file, fieldnames=field_names)
#         csv_writer.writeheader()

#         for line in csv_reader:
#             csv_writer.writerow(line)




    
        



        