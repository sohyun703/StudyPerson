def solution(array, commands):
    answer = []
    
    for i in range(len(commands)):
        arr =[]
        
        start = commands[i][0]-1
        end = commands[i][1]
        target = commands[i][2]
        arr = array[start:end]
        arr.sort()
        answer.append(arr[target-1])
    
    return answer