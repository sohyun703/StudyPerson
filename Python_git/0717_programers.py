def solution(strings, n):
    #answer = strings.sort(key = lambda x : x[n])
    answer = sorted(strings,key=lambda x: (x[n],x))
    return answer