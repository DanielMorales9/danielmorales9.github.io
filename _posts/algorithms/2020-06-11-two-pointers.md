---
layout: post
author: daniel
---
## Two Sum II
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.
 
Note:
Your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution and you may not use the same element twice.
 
````python 
def twoSum(nums, target):
    l = 0
    r = len(nums) - 1
    while l < r:
        if nums[l] + nums[r] == target:
            return [l+1, r+1]
        elif nums[l] + nums[r] > target:
            r -= 1
        else:
            l += 1
````

## Minimum Size Subarray Sum
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

````python
def minSubArrayLen(s, num):
    n = len(nums)
    p = [0] * (n + 1)

    for i, ni in enumerate(nums):
        p[i+1] = ni + p[i] 

    l = 0
    r = 1
    ans = n
    exists = False
    while l <= n and r <= n:
        if p[r] - p[l] < s:
            r += 1
        else:
            ans = min(ans, r - l)
            exists = True
            l += 1

    return ans if exists else 0
````