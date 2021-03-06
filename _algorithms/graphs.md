---
layout: page
author: daniel
name: Graphs
lesson: 7
---

## Course Schedule
There are a total of numCourses courses you have to take, labeled from 0 to numCourses-1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

````python 
def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = defaultdict(lambda: 0)

    for a, b in prerequisites:
        graph[a].append(b)
        in_degree[b] += 1

    queue = []
    for i in range(numCourses):
        if in_degree[i] == 0:
            queue.append(i)

    cnt = 0

    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        cnt += 1

    if cnt > numCourses:
        return False
    return len(top_order) == numCourses
````