---
layout: page
author: daniel
name: Binary Trees
---
## Complete Tree
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
 
Computational Complexity O(N*)
Space Complexity O(N*)
 
N is the number of nodes.
 
````python
def is_complete(root):
    nodes = [(root, 1)]
    i = 0
    while i < len(nodes):
        node, v = nodes[i]
        i += 1
        if node:
            nodes.append((node.left, 2*v))
            nodes.append((node.right, 2*v+1))

    return  nodes[-1][1] == len(nodes)
````

## Flatten Binary Tree to Linked List
Given a binary tree, flatten it to a linked list in-place.
### Stack Version
````python
def flatten(root):
    """
    Do not return anything, modify root in-place instead.
    """
    if not root:
        return

    stack = []

    if root.right:
        stack.append(root.right)
    if root.left:
        stack.append(root.left)

    while stack:
        leftmost = stack.pop()
        root.right = leftmost
        root.left = None
        root = root.right
        if leftmost.right:
            stack.append(leftmost.right)
        if leftmost.left:
            stack.append(leftmost.left)
````

### Recursive Version

````python
def appendRightToLeft(left, right):
    if left.right:
        appendRightToLeft(left.right, right)
        
    else:
        left.right = right
        
def flatten(root):
    if not root:
        return 
    
    if root.left:
        appendRightToLeft(root.left, root.right)
        root.right = root.left
        root.left = None
    
   flatten(root.right)
````

## Kth Smallest Element in a BST
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
### Stack Version

````python
def kthSmallest(root, k):
    stack = []

    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if not k:
            return root.val
        root = root.right
````

Time complexity: O(N) to build a traversal
Space complexity: O(N) to keep an inorder traversal
### Recursive Version
````python
def kthSmallest(self, root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: int
    """
    def inorder(r):
        return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    
    return inorder(root)[k - 1]
````

Time complexity: O(H+k), where H is a tree height. This complexity is defined by the stack, which contains at least H + k elements, since before starting to pop out one has to go down to a leaf. This results in O(logN + k) for the balanced tree and O(N + k) for completely unbalanced tree with all the nodes in the left subtree.
 
Space complexity: O(H+k), the same as for time complexity, O(N + k) in the worst case, and O(logN + k) in the average case.
