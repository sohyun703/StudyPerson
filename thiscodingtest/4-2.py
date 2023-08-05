N = int(input())
number_list = [number for number in range(60) if '3' in str(number)]
count3 = len(number_list)
k = count3*60*2 - count3**2

if N >=3:
    result = N*k + 60*60

else:
    result = (N+1)*k

print(result)