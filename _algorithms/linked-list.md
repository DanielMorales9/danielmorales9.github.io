---
layout: page
author: daniel
name: Linked List
---
## Flatten a Multilevel Doubly Linked List
You are given a doubly linked list which in addition to the next and previous pointers, it could have a child pointer, which may or may not point to a separate doubly linked list. These child lists may have one or more children of their own, and so on, to produce a multilevel data structure, as shown in the example below.
 
Flatten the list so that all the nodes appear in a single-level, doubly linked list. You are given the head of the first level of the list.
 
````python
def rec_flatten(head):
    curr = head
    while curr.next or curr.child:
        if curr.child:
            son, last = rec_flatten(curr.child)
            
            if curr.next: curr.next.prev = last
            last.next = curr.next
            curr.child = None
            son.prev = curr
            curr.next = son
            curr = last
        else:
            curr = curr.next

    return head, curr
````

````python
def flatten(head):
    if head is None:
        return head
    
    return rec_flatten(head)[0]
````

## Palindrome Linked List
Given a singly linked list, determine if it is a palindrome.

````python
def isPalindrome(head):
    fast = head
    slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    # reverse list
    node = None
    while slow:
        nxt = slow.next
        slow.next = node
        node = slow
        slow = nxt

    while node:
        if node.val != head.val:
            return False
        node = node.next
        head = head.next
    return True
````

## Linked List Cycle
Given a linked list, determine if it has a cycle in it.
 
To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

````python
def hasCycle(head):
    slow = head
    fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            return True

    return False
````
## Remove Duplicates from Sorted List
Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

````python
def deleteDuplicates(head):
    d = ListNode(val=None, next=head)
    prev = d
    curr = d.next
    while curr:

        atleast = False
        while curr and curr.next and curr.val == curr.next.val:
            atleast = True
            curr = curr.next

        if atleast:
            prev.next = curr.next
        else:
            prev = curr

        curr = curr.next

    return d.next
````
 
## Convert Sorted List to Binary Search Tree
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

````python
def findMiddleAndCut(head):

        # The pointer used to disconnect the left half from the mid node.
        prev = None # you could use a dumb done here as well
        slow = head
        fast = head

        # Iterate until fastPr doesn't reach the end of the linked list.
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        # Handling the case when slow was equal to head.
        if prev:
            prev.next = None

        return slow

def sortedListToBST(head):

    mid = findMiddleAndCut(head)

    t = TreeNode(val=mid.val)

    if mid == head:
        return t

    t.left = sortedListToBST(head)

    t.right = self.sortedListToBST(mid.next)
    return t
````

Time Complexity O(NlogN)
Space Complexity O(logN)
 
````python
def makeArray(head):
    arr = []
    while head:
        arr.append(head.val)
        head = head.next
    return arr

def makeBST(arr, l, r):
    if l > r:
        return None
    
    mid = (r - l) // 2 + l
    
    a = arr[mid]
    t = TreeNode(val=a)
    t.left = makeBST(arr, l=l, r=mid-1)
    t.right = makeBST(arr, l=mid+1, r=r)
    
    return t

def sortedListToBST(head):
    arr = makeArray(head)

    return makeBST(arr, 0, len(arr)-1)
````
Time Complexity: O(N)
Space Complexity: O(logN)
