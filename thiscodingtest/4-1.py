N = int(input())
plans = input().split()

move_type=["L","R","U","D"]
r,c =1,1
dr = [0,0,-1,1]
dc = [-1,1,0,0]

for plan in plans:
    for i in range(len(move_type)):
        if plan == move_type[i]:
            nr = r+dr[i]
            nc = c +dc[i]
        
    if nr <1 or nc<1 or nr>N or nc>N:
        continue
        
    r,c = nr,nc

print(r,c)