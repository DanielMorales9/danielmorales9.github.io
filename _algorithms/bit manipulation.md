---
layout: page
author: daniel
name: Bit Manipulation
lesson: 10
---
## Integer Replacement
Given a positive integer n and you can do operations as follow:
 
If n is even, replace n with n/2.
If n is odd, you can replace n with either n + 1 or n - 1.
 
What is the minimum number of replacements needed for n to become 1?

````python
def integerReplacement(n):
    cnt = 0
    while n != 1:
        if n % 2 == 0:
            n = n >> 1
        else:
            if math.log(n - 1, 2).is_integer():
                n = n - 1
            else:
                n = n + 1
        cnt += 1

    return cnt
````

## Missing Number
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
````python
def missingNumber(nums):
    missing = len(nums)
    for i, num in enumerate(nums):
        missing ^= i ^ num
    return missing
````
