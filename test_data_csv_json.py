import csv
import json

# Étape 1: Créer le fichier data.csv
data = [
    [25, "red", "S"],
    [30, "blue", "L"],
    [35, "green", "M"],
    [40, "blue", "XL"]
]

# name csv file
csv_filename = "data.csv"
# name json file
json_filename = "metadata.json"

# name of the columns
headers = ["age", "color", "size"]


with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data)   

print(f"Fichier {csv_filename} créé avec succès!")

metadata = {
    "columns": [
        {"name": "age", "type": "continuous"},
        {"name": "color", "type": "categorical"},
        {"name": "size", "type": "categorical"}
    ]
}



# write to json file
with open(json_filename, mode="w") as file:
    json.dump(metadata, file, indent=4)

print(f"Fichier {json_filename} créé avec succès!")
