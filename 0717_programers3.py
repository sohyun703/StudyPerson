def solution(numbers):
    answer = []
    #인덱스 제외 더하기
    
    for i in range(len(numbers)):
        for j in range(i+1,len(numbers)):
            answer.append(numbers[i]+numbers[j])
            
    result =[]
    
    for val in answer:
        if val not in result:
            result.append(val)
    result.sort()
            
    
            
    return result