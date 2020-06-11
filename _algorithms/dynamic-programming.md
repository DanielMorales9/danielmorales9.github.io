---
layout: page
author: daniel
name: Dynamic Programming
---
Is about finding some work that the naive usually recursive solution would repeat multiple times unnecessarily. Instead, it’s better to save the result of that subproblem and reuse it multiple times to avoid extra computation. 
 
Most of dynamic programming problems belong to one of three types of or categories:
Count something, often the number of ways - Combinatorics
Minimize or maximize certain value - Optimisation
boolean problems - Does a solution exists?
 
For the last two it is usually better to think whether greedy should be used instead. 
Iteration e Recursion
Also know as bottom-up or top-down approaches - Tabulation or memoization.
 

## Fibonacci problem
Naive solution: repeated computation in states
````python 
def fibonacci(n):
    if n == 0:
        return 1
    elif n == 1:
	  return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
````

Memoized version: stores states in dictionary

````python 
dp = {}

def fibonacci(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    if n in dp:
        return dp[n]

    dp[n] = fibonacci(n - 1) + fibonacci(n - 2)
    return dp[n]
````

Tabulated version: stores result of subproblem in array

````python 
def fibonacci(n):
    dp = [ 0 for _ in range(n + 1)]
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]
````

## Climbing Stairs
You are climbing a staircase. It takes N steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
 
Choosing a state and transition is relatively important in this problems. 
Indeed, it does not matter how we got to a specific position, it only matters where we are, that’s why we consider the position as a state of our dp solution. 
The transition are computed considering the possible positions I could have been before, which are actually equal to the number of jumps I can do: 1 step of 2 steps. 
So the value for the current position is derived from the sum of the values for the previous two possible positions. 
 
Once you found out the state and the transitions of your problem you can start coding the solution.

````python  
def stairs(n):
    dp = [ 0 for _ in range(n+1)]
    dp[0] = 1
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]
````

As you can see you always need to initialise the some states to get going. So don’t forget to consider the initial states. 
 
Let’s suppose now, you are allowed to do at most k steps instead of just 1 or 2. Then it means, you need to count all the states between current and the previous k back. 

````
def stairs(n, k):
   dp = [0 for _ in range(n+1)]
   dp[0] = 1
   dp[1] = 1
   dp[2] = 2

   for i in range(3, n+1):
       for j in range(1, k+1):
           if i - j >= 0:
               dp[i] += dp[i - j]

   return dp[-1]
````
 
Now, let’s suppose, you are allowed to do only k jumps of 1 step or 2 steps. How does our problem-solution change?
In this case, we added a constraint to the problem, meaning we should add that constraint as part of the state. K is the current number of jumps, so we count for all the position the number of ways to get there having only k jumps available and by only doing 1 step or 2 steps. Thus, we could build the solution by starting with only one jump available and summing up the number of ways to get to the same positions by using one more jump previously. Thus, we sum the result for k-1, for one and two position behind of the current one.

````
def stairs(n, k):
   dp = [[0 for _ in range(n)] for _ in range(k)]

   dp[0][0] = 1
   dp[0][1] = 1

   for j in range(1, k):
       for i in range(1, n):
           dp[j][i] = dp[j-1][i-1] + dp[j-1][i-2]

   return dp[-1][-1]
````

As you can see the solution is relatively fast O(k*n) and the space complexity is O(k*n).
However, we could keep O(n), space as we only need the values of k - 1 of each position to compute the current k.
 
## Minimum Path Sum
Given a grid, find a path from the top-left to the bottom-right corner that minimizes the sum of numbers along the path. You can only move right or down.

````python 
def min_path_sum(grid):
   n = len(grid)
   m = len(grid[0])
   dp = [[0 for _ in range(m)] for _ in range(n)]
   dp[0][0] = grid[0][0]

   for i in range(1, m):
       dp[0][i] = grid[0][i] + dp[0][i-1]

   for i in range(1, n):
       dp[i][0] = grid[i][0] + dp[i-1][0]

   for i in range(1, n):
       for j in range(1, m):
           dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
  
   return dp[-1][-1]
````

In this case, the dp state is the row and the column and we incrementally build the solution by looking at the left and top cell from the current cell. In this case, we want to find the optimal solution, thus, we compute the minimum of the left and top cells. Moreover, the initialization here is tough as you do not only initialise a few cells but all the leftmost and topmost cells with the accumulated sum.
 
Let’s assume that you cannot make two consecutive steps down, then you can’t anymore say that the only thing that matters is the sum of the previous value, but also the previous step. 
By adding a new constraint we know that we need to incorporate the direction of the previous step into state of the dp.
 
## Combination Sum
Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

````
def combinationSum(nums, target):
    n = target+1
    dp = [0] * n
    dp[0] = 1

    for i in range(n):
        for x in nums:
            if i >= x:
                dp[i] += dp[i-x]

    return dp[-1]
````

## Coin Change (min)
You are given denominations of coins and the target amount N. What is the minimum possible number of coins?
 
This problem is similar to the second version of the stairs problems, although we no longer count here, but we want to find the optimum. 
The solution is easy as the target amount becomes the state of our problem and we can build the final solution by iteratively consider the minimum number of coins among the target amounts we can reach with the current denominations. 

```` 
def coin_change(den, n):
    dp = [float('inf') for _ in range(n+1)]

    dp[0] = 0

    for i in range(1, n + 1):
        _min = dp[i]
        for d in den:
            if i >= d:
                _min = min(_min, dp[i - d])
        dp[i] = _min + 1

    return -1 if dp[-1] == float('inf') else dp[-1]
````

## Coin Change (count without repeating sets)
The common way to solve this problem is dynamic programming using 2D dp table dp[i][j] means the number of combinations to make up amount j by using the first i types of coins
For a coin of a denomination that you see, consider adding it to the “Total Amount”. Because the number of coins can be selected infinitely, therefore:
 
For a new denomination coin coins[i-1] (note that there is a offset). We can consider selecting 0, 1 and 2 and so on, until the total amount of coins with this denomination exceeds the total amount j.
### Approach 1
Write the code according to the state transition equation
````python
# time: O(NM^2)  space: O(MN)

def change(amount, coins):
    dp = [[0] * (amount + 1) for _ in range(len(coins) + 1)]
    dp[0][0] = 1
    for i in range(len(coins)+1):
        for j in range(1,amount+1):
            for k in range(j//coins[i-1]+1):
                dp[i][j] += dp[i-1][j-k*coins[i-1]]
    return dp[-1][-1]
````

Time complexity: O(NM^2), Space complexity: O(MN)
The amount is M and the number of coins is N.
### Approach 2
````python 

# time: O(NM)  space: O(MN)

def change(amount, coins):
   dp = [[0] * (amount + 1) for _ in range(len(coins) + 1)]
   dp[0][0] = 1
   for i in range(1, len(coins) + 1):
       for j in range(amount + 1):
           d = coins[i - 1]
           if j >= d:
               dp[i][j] = dp[i - 1][j] + dp[i][j - d]
           else:
               dp[i][j] = dp[i - 1][j]

   return dp[-1][-1]
````

### Approach 3
On the right hand side of equation, the first index is always i-1 or i. We can reduce the space complexity by using a scrolling array

````python 
def change(amount, coins):
   dp = [1]+[0]*amount
   for i in range(len(coins)):
       dp2 = [1]+[0]*amount
       for j in range(amount+1):
           d = coins[i]
           if j >= d:
               dp2[j] = dp[j] + dp2[j-d]
           else:
               dp2[j] = dp[j]
       dp = dp2

   return dp[-1]
```` 
### Approach 4
dp[i][j] only rely on dp[i-1][j] and dp[i][j-coins[i]], then we can optimize the space by only using 1-D array

````python 
# time: O(NM)  space: O(M)
def change(amount, coins):
    dp = [1] + [0] * amount
    for i in range(len(coins)):
        for j in range(amount + 1):
            if j - coins[i] >= 0:
                dp[j] += dp[j - coins[i]]
    return dp[-1]
````
## Longest Common Subsequence
The naive solution to this problem is to recursively consider all the possible subsequences and take the longest common one. This can be solved recursively:

````python 
def lcs(A, B, i=0, j=0):
    if i == len(A) or j == len(B):
        return 0
    elif A[i] == B[j]:
        return 1 + lcs(A, B, i, j)
    else:
        return max(lcs(A, B, i+1, j), lcs(A, B, i, j+1))
````
 
In the above solution, we move forward and add a one to the solution whenever we find a match, or we either take the maximum solution starting either from the next character of A or B. As you may have already understood, there are many overlapping subproblems to solve.
A simple way to solve it would be to just store the result of a repeating subproblem via memoization. 

````python  
def lcs(A, B, i=0, j=0, memo={}):
    if i == len(A) or j == len(B):
        return 0
    if (i, j) in memo:
        return memo[(i, j)]
    
    if A[i] == B[j]:
         memo[(i, j)] = 1 + lcs(A, B, i, j)
    else:
         memo[(i, j)] = max(lcs(A, B, i+1, j), lcs(A, B, i, j+1))
    return memo[(i, j)]
````

The tabulation approach considers the two indices as part of the state, just like in the memoization approach, and the transition are computed considering the possible positions I could have been before, which can actually be inferred from the naive approach. Indeed, the naive approach if there is a match the lcs is 1 plus the lcs starting at the next two character of the current subsequence or computes the longest common subsequence starting either at the next character of A or B. Thus in our tabulation approach we will increase the previous longest common subsequence by one or take the maximum of the two previous lcs, which either start at i-1,j or i,j-1.

````python 
 
def lcs(A, B):
    # find the length of the strings 
    n = len(A)
    m = len(B)
    dp = [[0] * (m+1) for i in range(n+1)]

    for i in range(1, n+1): 
        for j in range(1, m+1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else: 
                dp[i][j] = max(dp[i - 1][j] , dp[i][j - 1]) 

    return dp[-1][-1]

```` 
## Longest Palindrome Subsequence
The solution is to take the Longest common sequence solution and consider the second argument of s as the current string reversed. 

````python

def longestPalindromeSubseq(s):
    # find the length of the strings 
    n = len(s)

    dp = [[0] * (n+1) for i in range(n+1)]

    for i in range(1, n + 1):
        for j in range(n, 0, -1):
            z = n - j + 1
            if s[i - 1] == s[j - 1]:
                dp[i][z] = dp[i - 1][z - 1] + 1
            else: 
                dp[i][z] = max(dp[i - 1][z] , dp[i][z - 1]) 

    return dp[-1][-1]
````

## Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.

````python  
def canJump(nums):
    if len(nums) <= 1:
        return True

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    for i in range(1, n):
        jumps = dp[i - 1] - 1
        if jumps >= 0:
            dp[i] = max(nums[i], jumps)

    return dp[-2] > 0
````

## Maximal Square
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

````python  
def maximalSquare(matrix):
    if not matrix:
        return 0

    n = len(matrix)
    m = len(matrix[0])
    grid = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        grid[i][0] = int(matrix[i][0])

    for i in range(m):
        grid[0][i] = int(matrix[0][i])

    for i in range(1, n):
        for j in range(1, m):
            if matrix[i][j] == '1':
                a = min(grid[i-1][j-1], grid[i-1][j], grid[i][j-1])
                b = int(matrix[i][j])
                grid[i][j] = int((sqrt(a) + b) ** 2)
            else:
                grid[i][j] = 0

    return max(max(row) for row in grid)

````

## Form Largest Integer With Digits That Add up to Target
Given an array of integers cost and an integer target. Return the maximum integer you can paint under the following rules:
 
The cost of painting a digit (i+1) is given by cost[i] (0 indexed).
The total cost used must be equal to target.
Integer does not have digits 0.
Since the answer may be too large, return it as string.
 
If there is no way to paint any integer given the condition, return "0".

````python  
def largestNumber(cost, target):
    """
    :type cost: List[int]
    :type target: int
    :rtype: str
    """

    dp = [0] + [-1] * target

    for t in range(1, target+1):
        for i, c in enumerate(cost):
            if t >= c:
                dp[t] = max(dp[t], dp[t - c] * 10 + i + 1)

    return str(max(dp[-1], 0))
````

## Perfect Squares
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

````python  
def numSquares(n):
    dp = [i for i in range(n + 1)]

    for i in range(n + 1):
        j = 1
        while i >= j*j:
            dp[i] = min(dp[i], dp[i-j*j]+1)
            j += 1
    return dp[-1]
````
 
## Count Square Submatrices with All Ones
Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.
 
````python 
def countSquares(matrix):
    n = len(matrix)
    m = len(matrix[0])

    tab = [[ 0 for i in range(m)] for _ in range(n) ]

    cnt = 0
    for i in range(0, n):
        for j in range(0, m):
            if matrix[i][j] == 0:
                tab[i][j] = 0
            else:
                tab[i][j] = matrix[i][j] + min(tab[i-1][j], tab[i-1][j-1], tab[i][j-1])
            cnt += tab[i][j]
    return cnt
````

## Levenshtein Distance
````python 
def minDistance(word1, word2):
    n = len(word1)
    m = len(word2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]

    for i in range(0, n+1):
        dp[i][0] = i

    for j in range(0, m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min([dp[i-1][j], dp[i][j-1], dp[i-1][j-1]]) + 1
    return dp[-1][-1]
````