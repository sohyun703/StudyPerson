def dfs(graph,start_node):
    #dfs는 방문할 노드와, 방문한 노드를 따로 관리해주어야 함
    need_visited,visited=list(),list()
    
    #방문 노드 지정
    need_visited.append(start_node)
    
    while need_visited:
        node= need_visited.pop()#방문이 필요한 노드 추출
        #노드가 방문한 목록에 없다면
        if node not in visited:
            #방문한 목록에 추가하기
            visited.append(node)
            need_visited.extend(graph[node]) #append는 리스트 자체로 넣어주지만 extend는 요소로 넣어준다.
    
    return visited

def dfs2(graph,start_node):
    from collections import deque
    visited=[]
    need_visited = deque()
    need_visited.append(start_node)
    
    while need_visited:
        node = need_visited.pop()
        if node  not in visited:
            #방문 리스트
            visited.append(node)
            #인접 노드들을 방문 예정 리스트에 추가
            need_visited.extend(graph[node])
    
    return visited
    