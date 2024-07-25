
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
my_dict["test"]=[]
my_dict["val"]=[]
my_dict["train"]=[]

for line in lines:
    a=line.split("\t")

    if a[0] in test:
        my_dict["test"].append(a[0])

    elif a[0] in train:
        my_dict["train"].append(a[0])
    elif a[0] in val:
        my_dict["val"].append(a[0])
    else:
        print(a[0])
        print("error")

with open("data.json", "w") as json_file:
    json.dump(my_dict, json_file, indent=4)
dict={}
