input_file_path = "out.txt"
output_file_path = "output1.txt"

with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

wor=[]
for i in range(len(lines)):
    words = lines[i].split()

    wor.append("\t" +"\t" +'"'+ words[0] + '"' +','+'  '+ words[1] + '\n')
quoted_lines = [line.strip() + '"\n' for line in wor]

with open(output_file_path, "w") as output_file:
    output_file.writelines(quoted_lines)
