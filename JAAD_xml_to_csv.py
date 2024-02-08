import xml.etree.ElementTree as ET
import os
import csv

# input and output directories
input_dir = f'{HOME}/drive/MyDrive/JAAD/annotations'
output_dir = f'{HOME}/drive/MyDrive/JAAD/Conversion'

# creating the output directory if it doesnt exist
os.makedirs(output_dir, exist_ok=True)

# function to process each XML file
def process_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data_list = []

    # creating a variable to later map the aplhanumeric ids into integer ids
    id_mapping = {}
    id_counter = 1

    for track in root.findall('.//track'):
        if track.get('label') in ['pedestrian', 'ped']:
            for box in track.findall('.//box'):
                # shifting the frame number to +1 to avoid errors whilst running TrackEval
                frame = int(box.get('frame')) + 1
                id_element = box.find(".//attribute[@name='id']").text

                if id_element in id_mapping:
                    id_int = id_mapping[id_element]
                else:
                    id_int = id_counter
                    id_mapping[id_element] = id_counter
                    id_counter += 1

                xbr = box.get('xbr')
                xtl = box.get('xtl')
                ybr = box.get('ybr')
                ytl = box.get('ytl')

                new_width = float(xbr) - float(xtl)
                new_height = float(ybr) - float(ytl)

                occlusion = box.find(".//attribute[@name='occlusion']").text
                conf = 0 if occlusion == 'full' else 1

                data_list.append([frame, id_int, xtl, ytl, new_width, new_height, conf, -1, -1, -1])

    data_list.sort(key=lambda x: int(x[0]))

    # constructing the output CSV file path
    output_csv_file = os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.csv'))

    # writing the extracted data to a CSV file
    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
        csvwriter.writerows(data_list)

# iterating through all XML files in the input directory
for xml_file_name in os.listdir(input_dir):
    if xml_file_name.endswith('.xml'):
        xml_file_path = os.path.join(input_dir, xml_file_name)
        process_xml_file(xml_file_path)