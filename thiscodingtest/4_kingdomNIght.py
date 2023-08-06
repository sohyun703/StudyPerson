input_data= input()
row = int(input_data[1])
column = input_data[0]
column_table ={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8}
columnNum = int(column_table[column])

move = [(-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2)]

cnt =0

for i in range(len(move)):
    next_row = row+ move[i][0]
    next_column = columnNum+ move[i][1]
    if next_row<1 or next_column<1 or next_row>8 or next_column>8:
        continue
    else:
        cnt +=1
    # if next_row>=1 and next_row<=8 and next_column>=1 and next_column<=8:
    #     cnt+=1

print(cnt)