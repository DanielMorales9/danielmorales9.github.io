---
layout: page
author: daniel
name: Math
lesson: 3
---
## Count Primes
Count the number of prime numbers less than a non-negative number, n.

````python
def is_prime(n):
    if n <= 1:
        return False
    
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
        
    return True

def countPrimes(n):
    cnt = 0
    for i in range(n):
        if is_prime(i):
            cnt += 1

    return cnt
````

Alternatively, you can use the Sieve of Eratosthenes
````python
def countPrimes(n):
    prime = [True] * n

    i = 2
    while i * i < n:
        if prime[i]:
            j = i * i
            while j < n:
                prime[j] = False
                j += i
        i += 1

    return sum(prime[2:])
````