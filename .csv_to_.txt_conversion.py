# converting the comma seperated CSV file into a text file with Comma separation

# for the path values we can either specify the ground truth CSV files or the tracker data CSV files
csv_folder = f'{HOME}/drive/MyDrive/CSVOutputs/SingleVideo'
output_folder = f'{HOME}/drive/MyDrive/MOT15/TextFormat'

for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
      csv_file = os.path.join(csv_folder, filename)
      txt_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0].replace('.mkv', '')}.txt")

      # processing the CSV file and save as a text file
      with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
           [my_output_file.write(",".join(row) + '\n') for row in csv.reader(my_input_file)]

      # removing the header from the generated text file
      with open(txt_file, 'r') as fin:
        data = fin.read().splitlines(True)
      with open(txt_file, 'w') as fout:
        fout.writelines(data[1:])