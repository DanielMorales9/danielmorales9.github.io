---
layout: post
author: daniel
---

Computational complexity O(log n)
## Recursive version
````python
def bin_search(arr, low, high, key): 
  
    if high < low: 
        return -1
          
    mid = low + (high - low)//2
      
    if key == arr[mid]: 
        return mid 
    if key > arr[mid]: 
        return bin_search(arr, (mid + 1), high, key)


    return bin_search(arr, low, (mid - 1), key)
````

Space Complexity O(log n)

## Imperative version
````python
def bin_search(target, nums):
	low = 0
	high = len(nums) - 1
	while low <= high:
		mid = low + (high - low) // 2
		if nums[mid] == target:
			return mid
		elif nums[mid] < target:
			low = mid + 1
		elif nums[mid] > target:
			high = mid - 1
	return -1
```` 
Space Complexity O(1)
 
## Square Root
Implement int sqrt(int x).
Compute and return the square root of x, where x is guaranteed to be a non-negative integer.
Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

````python
def sqrt(x):
    if x == 0:
        return 0

    l = 0
    r = x
    mid = 0

    while l <= r:

        mid = (r - l) // 2 + l
        sqrd = mid * mid
        if sqrd == x:
            return mid
        elif sqrd < x:
            l = mid + 1
        else:
            r = mid - 1

    if sqrd > x:
        mid -= 1

    return mid
````

## Valid Perfect Square
Given a positive integer num, write a function which returns True if num is a perfect square else False.
 
````python
def isValidSquare(x):
    if x == 0:
        return False

    l = 0
    r = x
    mid = 0

    while l <= r:

        mid = (r - l) // 2 + l
        sqrd = mid * mid
        if sqrd == x:
            return True
        elif sqrd < x:
            l = mid + 1
        else:
            r = mid - 1

    return False
````

## Find Smallest Element Greater Than Target
Letters could be duplicate. Thus, whenever the current letter is equal we move the search to the right by pushing left.
 
````python
def nextGreatestLetter(letters, target):
    n = len(letters) - 1
    low = 0
    high = n
    while low <= high:
        mid = low + (high - low) // 2
        if letters[mid] > target:
            high = mid - 1
        else:
            low = mid + 1

    if letters[mid] > target:
        return letters[mid]

    return letters[(mid+1) % (n+1)]
````
 
Alternative solution would be to collect the smallest value greater than target into ans.

````python
def nextGreatestLetter(letters, target):
    n = len(letters) - 1
    low = 0
    high = n
    ans = letters[0]
    while low <= high:
        mid = low + (high - low) // 2
        if letters[mid] > target:
            ans = letters[mid]
            high = mid - 1
        else:
            low = mid + 1

    return ans
````

## Find Minimum in Rotated Sorted Array
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. (i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).
Find the minimum element. You may assume no duplicate exists in the array.

````python 
def findMin(nums):
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:
            l = mid + 1
        else:
            r = mid

    return nums[l]
````
## Find Peak Element
A peak element is an element that is greater than its neighbors. Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index. The array may contain multiple peaks, in that case return the index to any one of the peaks is fine. You may imagine that nums[-1] = nums[n] = -∞.
 
````python
def findPeakElement(nums):
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = (r + l) // 2
        if nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1

    return l
````

## Arranging Coins
You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins.
Given n, find the total number of full staircase rows that can be formed.
n is a non-negative integer and fits within the range of a 32-bit signed integer.

````python
def arrangeCoins(n):
    ans = 0
    l = 0
    r = n
    while l <= r:
        mid = (l + r) // 2
        if mid * (mid + 1) // 2 <= n:
            ans = mid
            l = mid + 1
        else:
            r = mid - 1

    return ans
````

````python
def singleNonDuplicate(nums):
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = (r + l) // 2
        if mid % 2 == 0 and nums[mid] == nums[mid+1]:
            l = mid + 2
        elif mid % 2 == 1 and nums[mid] == nums[mid-1]:
            l = mid + 1
        else:
            r = mid
    return nums[l]
````