file_path = "out.txt"
a=[]
with open(file_path, "r") as file:
    lines = file.readlines()
with open("output.txt", "w") as file:
    for line in lines:
        line=line.replace("   ", "\t")
        file.write(line)