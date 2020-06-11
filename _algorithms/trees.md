---
layout: page
author: daniel
name: Trees
lesson: 4
---
# Maximum Level sum of a Binary Tree
Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.
Return the smallest level X such that the sum of all the values of nodes at level X is maximal.
## Iterative solution
````python
def maxLevelSum(root):
        level_sum = []
        queue = collections.deque()
        queue.append((0, root))
        while queue:
            level, node = queue.popleft()
            if node != None:
                if len(level_sum) <= level:
                    level_sum.append(node.val)
                else:
                    level_sum[level] += node.val
            
                queue.append((level + 1, node.left))
                queue.append((level + 1, node.right))

        return level_sum.index(max(level_sum)) + 1
````

### Recursive solution
````python
def maxLevelSum(root):
    def rec(root, lev, depth):
        if not root:
            return

        if len(lev) > depth:
            lev[depth] += root.val
        else:
            lev.append(root.val)

        rec(root.left, lev, depth+1)
        rec(root.right, lev, depth+1)

    lev = []
    rec(root, lev, 0)

    return lev.index(max(lev)) + 1
```` 
