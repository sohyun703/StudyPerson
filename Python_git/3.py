n, m = map(int,input().split())

result = []
for i in range(n):
    list_test = list(map(int, input().split()))
    result.append(list_test)

target = []
for i in range(n):
    target_min = min(result[i])
    target.append(target_min)

print(max(target))