
import json
my_dict = {}
with open('/media/test/run/count/countx/CounTX-main-arg/LearningToCountEverything-master/data/Train_Test_Val_FSC_147.json') as f:
    fsc172_d_annotations = json.load(f)
file_path = "/media/test/run/count/countx/CounTX-main-arg/LearningToCountEverything-master/data/ImageClasses_FSC147.txt"
with open(file_path, "r") as file:
    lines = file.readlines()
test = fsc172_d_annotations['test']
val = fsc172_d_annotations['val']
train = fsc172_d_annotations['train']
for line in lines:
    a=line.split("\t")

    if a[0] in test:
        my_dict[a[0]] = {}
        my_dict[a[0]]["data_split"]="test"
        my_dict[a[0]]["text_description"]=a[1].replace("\n", "")
    elif a[0] in train:
        my_dict[a[0]] = {}
        my_dict[a[0]]["data_split"]="train"
        my_dict[a[0]]["text_description"]=a[1].replace("\n", "")
    elif a[0] in val:
        my_dict[a[0]] = {}
        my_dict[a[0]]["data_split"]="val"
        my_dict[a[0]]["text_description"]=a[1].replace("\n", "")
    else:
        print(a[0])
        print("error")

with open("data.json", "w") as json_file:
    json.dump(my_dict, json_file, indent=4)
dict={}
