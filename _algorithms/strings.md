---
layout: page
author: daniel
name: Strings
lesson: 1
---
## Remove K digits
Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.
 
Note:
The length of num is less than 10002 and will be ≥ k.
The given num does not contain any leading zero.
 
````python
def removeKdigits(num, k):

    res = []
    n = len(num)

    if n == k: return "0"

    for i in range(n):
        while k and res and res[-1] > num[i]:
            res.pop()
            k -= 1
        res.append(num[i])


    while k:
        res.pop()
        k -= 1

    return "".join(res).lstrip('0') or "0"
````

 
## Find All Anagrams in a String
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100. The order of output does not matter.

### HashTable
````python
 def findAnagrams(s, p):
    cnt_p = Counter(p); n = len(s); m = len(p); res = []
    for i in range(n-m+1):
        if i == 0: cnt_s = Counter(s[:m])
        else:
            prev = s[i-1]; curr = s[i+m-1]
            cnt_s[prev] -= 1; cnt_s[curr] += 1
            if cnt_s[prev] == 0: del cnt_s[prev]
        if cnt_s == cnt_p: res.append(i)
    return res        
````

### Sort
````python 
def findAnagrams(s, p):
    p = sorted(p); n = len(s); m = len(p); res = []
    for i in range(n-m+1):
        q = sorted(s[i:i+m])
        if q == p: res.append(i)
    return res
````