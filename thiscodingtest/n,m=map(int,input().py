n,m=map(int,input().split())
d = []
x,y,direction = map(int,input().split())
array = [[0]*m for i in range(n)]
for i in range(n):
    d.append(list(map(int,input().split())))

#왼쪽으로 돌기

def turnleft():
    global direction
    direction -=1
    
    if direction ==-1:
        direction =3

dx=[-1,0,1,0]
dy=[0,1,0,-1]

array[x][y]=1
count =1
turn = 0

while True:
    turnleft()
    
    nx = x+dx[direction]
    ny = y+ dy[direction]
    
    if array[nx][ny] ==0 and d[nx][ny]==0:
        x = nx
        y = ny
        count +=1
        turn +=1
        continue
    else:
        turn +=1
    
    #네번다 돈 경우
    
    if turn ==4:
        nx = x - dx[direction]
        ny = y - dy[direction]
        if array[nx][ny] ==0:
            x = nx
            y= ny
        else:
            break
        turn =0

print(count)
