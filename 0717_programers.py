def solution(strings, n):
    #이건 새로운 리스트를 지정해주는 게 아니다!
    #answer = strings.sort(key = lambda x : x[n])
    #람다 조건 두개하려면 -> x :(x[n],x)로 묶어준다.
    answer = sorted(strings,key=lambda x: (x[n],x))
    return answer