n,m = map(int,input().split())
array=[]
for i in range(n):
    array.append(list(map(int,input())))

#dfs

def dfs(x,y):
    #만약 탐색 범위를 벗어난다면
    if x <=-1 or x>=n or y<=-1 or y>=m:
        return False
    
    if array[x][y]==0:
        array[x][y]=1
        dfs(x-1,y)
        dfs(x,y-1)
        dfs(x+1,y)
        dfs(x,y+1)
        return True
    return False

    
result =0
for i in range(n):
    for j in range(m):
        if dfs(i,j) == True:
            result +=1

print(result)