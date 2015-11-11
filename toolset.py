# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 19:10:20 2015

@author: itithilien
"""

import numpy as np
from functools import wraps

def memoize(f):
    "memoize function with 1 argument"
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def memo(func):
    "memoize function with > 1 argument"
    cache = {}
    @ wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

def primesfrom2toPy2(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    for i in xrange(1,int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k/3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

def mobius_sieve(n):
    "generates mobius function values upto n, needs numpy and a primesieve(in the same file)"
    pist = primesfrom2toPy2(int(n**0.5))
    L = np.ones(n).astype(int)
    
    for p in pist:
        L[::p]    *= -1
        L[::p**2] *=  0
    return L

def primefactor_sieve(n):
  sieve = [[] for x in xrange(0, n)]
  for i in xrange(2, n):
    if not sieve[i]:
      q = i
      while q < n:
        for r in xrange(q, n, q):
          sieve[r].append(i)
        q *= i
  return sieve

def hamming2(multipliers  = (2, 3, 5)): #can also accept list of primes
    '''
    Usage:
    from itertools import islice
    hummus = list(islice(hamming2([2,3,5,7]), 200000))
    
    This version is based on a snippet from:
        http://dobbscodetalk.com/index.php?option=com_content&task=view&id=913&Itemid=85

        When expressed in some imaginary pseudo-C with automatic
        unlimited storage allocation and BIGNUM arithmetics, it can be
        expressed as:
            hamming = h where
              array h;
              n=0; h[0]=1; i=0; j=0; k=0;
              x2=2*h[ i ]; x3=3*h[j]; x5=5*h[k];
              repeat:
                h[++n] = min(x2,x3,x5);
                if (x2==h[n]) { x2=2*h[++i]; }
                if (x3==h[n]) { x3=3*h[++j]; }
                if (x5==h[n]) { x5=5*h[++k]; } 
    '''
    h = 1
    _h=[h]    # memoized
    
    multindeces  = [0 for i in multipliers] # index into _h for multipliers
    multvalues   = [x * _h[i] for x,i in zip(multipliers, multindeces)]
    yield h
    while True:
        h = min(multvalues)
        _h.append(h)
        for (n,(v,x,i)) in enumerate(zip(multvalues, multipliers, multindeces)):
            if v == h:
                i += 1
                multindeces[n] = i
                multvalues[n]  = x * _h[i]
        # cap the memoization
        mini = min(multindeces)
        if mini >= 1000:
            del _h[:mini]
            multindeces = [j - mini for j in multindeces]
        #
        yield h

#==============================================================================
# def primorial(n):
#     if n==1:
#         return 2
#     if n==2:
#         return 2*3
#     plist = primesfrom2toPy2(100)
#     
#     
#     reduce(int.__mul__, plist.astype(int)[:18])
#==============================================================================


#==============================================================================
# #very slow; don't use
# def primeSieve(cap):
#     N = np.ones(cap).astype(int)
#     N[:2] = 0
#     Plist = []
#     p = 1
#     while p < np.sqrt(cap):
#         p = np.argmax(N)
#         N[::p] = 0
#         Plist += [p]
#     
#     Plist = np.hstack((Plist, np.where(N > 0)[0]))
#     return Plist
#==============================================================================

def xgcd(a,b):
    """
    Extended euclidean algorithm: Iterative version
    http://anh.cs.luc.edu/331/notes/xgcd.pdf
    """
    prevx, x = 1, 0; prevy, y = 0, 1
    while b:
        q = a/b
        x, prevx = prevx - q*x, x
        y, prevy = prevy - q*y, y
        a, b = b, a % b
    return a, prevx, prevy

def egcd(a, b):
    """
    Extended euclidean algorithm: Iterative version
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    gcd = b
    return gcd, x, y

def egcd_rec(a, b):
    """
    Extended euclidean algorithm: Recursive version
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    """
    Modular inverse using Extended euclidean algorithm
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    gcd, x, y = egcd(a, m)
    if gcd != 1:
        return None  # modular inverse does not exist
    else:
        return x % m

def gcd_bin(u, v):
    "GCD calculation using bit wise operations"
    #u, v = abs(u), abs(v) # u >= 0, v >= 0
    if u < v:
        u, v = v, u # u >= v >= 0
    if v == 0:
        return u
 
    # u >= v > 0
    k = 1
    while u & 1 == 0 and v & 1 == 0: # u, v - even
        u >>= 1; v >>= 1
        k <<= 1
 
    t = -v if u & 1 else u
    while t:
        while t & 1 == 0:
            t >>= 1
        if t > 0:
            u = t
        else:
            v = -t
        t = u - v
    return u * k

def factors_lima(n):
    step = 2 if n%2 else 1
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5)+1, step) if n%i == 0)))

def factors_agf(n):
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5)+1) if n%i == 0)))

def factors_itool(n):
    import itertools
    flatten_iter = itertools.chain.from_iterable
    return set(flatten_iter((i, n//i) for i in range(1, int(n**0.5)+1) if n%i == 0))

def factors_genf(n):
    return set(x for tup in ([i, n//i] for i in range(1, int(n**0.5)+1) if n%i == 0) for x in tup)

def digsum(n):
    "sum of digits of a number"
    elbus = 0
    while n >=1:
        elbus += n%10
        n /= 10
    return elbus

def digits(n):
    "split a number into its digits"
    tuna = []
    while n > 0:
        tuna.append(n%10)
        n /= 10
    return tuna

def diophantine_count(a, n):
    """Computes the number of nonnegative solutions (x) of the linear
    Diophantine equation
        a[0] * x[0] + ... a[N-1] * x[N-1] = n
    
    Theory: For natural numbers a[0], a[2], ..., a[N - 1], n, and j,
    let p(a, n, j) be the number of nonnegative solutions.
    
    Then one has:
        p(a, m, j) = sum p(a[1:], m - k * a[0], j - 1), where the sum is taken
        over 0 <= k <= floor(m // a[0])
    
    Examples
    --------
    >>> diophantine_count([3, 2, 1, 1], 47)
    3572
    >>> diophantine_count([3, 2, 1, 1], 40)
    2282
    """
    
    def p(a, m, j):
        if j == 0:
            return int(m == 0)
        else:
            return sum([p(a[1:], m - k * a[0], j - 1)
                        for k in xrange(1 + m // a[0])])

    return p(a, n, len(a))
    

"""Compute solutions to the diophantine Pell equation x^2-D*y^2=1."""

def pell (D):
    """Return the smallest integer set solving Pell equation
    x^2-D*y^2=1 where x, D and y are positive integers. If there are no
    solution (D is a square), return None.

    >>> pell(3)
    (2, 1)"""
    a0 = int (D**0.5)
    if a0*a0 == D: return None
    gp = [0, a0]
    gq = [1, D-a0**2]
    a = [a0, int((a0+gp[1])/gq[1])]
    p = [a[0], a[0]*a[1]+1]
    q = [1, a[1]]
    maxdepth = None
    n = 1
    while maxdepth is None or n < maxdepth:
        if maxdepth is None and a[-1] == 2*a[0]:
            r = n-1
            if r % 2 == 1: return p[r], q[r]
            maxdepth = 2*r+1
        n += 1
        gp.append (a[n-1]*gq[n-1]-gp[n-1])
        gq.append ((D-gp[n]**2)//gq[n-1])
        a.append (int ((a[0]+gp[n])//gq[n]))
        p.append (a[n]*p[n-1]+p[n-2])
        q.append (a[n]*q[n-1]+q[n-2])
    return p[2*r+1], q[2*r+1]

def gen_pell (D):
    """Return the first solutions to Pell equation x^2-D*y^2=1 where x,
    D and y are positive integers. Computation of solution couples is
    done in floating point. As soon as the precision is not high enough,
    the generator stops. All the solutions the caller receives are correct.

    >>> for (x, y) in gen_pell(1053): print (x, y)
    ...
    (649, 20)
    (842401, 25960)
    (1093435849, 33696060)
    (1419278889601L, 43737459920L)"""
    
    import itertools
    r = pell (D)
    if not r: return
    p, q = r
    sd = D**0.5
    qd = q * sd
    sd2 = 2*sd
    lm = p + qd
    rm = p - qd
    for n in itertools.count (1):
        lmn = lm**n
        rmn = rm**n
        x, y = int ((lmn+rmn)/2+0.5), int ((lmn-rmn)/sd2+0.5)
        if x**2-D*y**2 != 1: return
        yield x, y

def isPalindrome(number):
    return int(str(number)[::-1])==number

def getPalindrome():
    """
        Generator for palindromes.
        Generates palindromes, starting with 0.
        A palindrome is a number which reads the same in both directions.
    """
    from itertools import count
    yield 0
    for digits in count(1):
        first = 10 ** ((digits - 1) // 2)
        for s in map(str, range(first, 10 * first)):
            yield int(s + s[-(digits % 2)-1::-1])

def allPalindromes(minP, maxP):
    """
        Get a sorted list of all palindromes in intervall [minP, maxP]
        Source: http://stackoverflow.com/a/19857182/2724299
    """
    palindromGenerator = getPalindrome()
    palindromeList = []
    for palindrome in palindromGenerator:
        if palindrome > maxP:
            break
        if palindrome < minP:
            continue
        palindromeList.append(palindrome)
    return palindromeList

def fib_it(n):
    a,b = 1,1
    for i in range(n-1):
        a,b = b,a+b
    return a

def fib_phi(n):
    from math import sqrt, floor
    phi = (1+ sqrt(5))/2
    return floor(phi**n/sqrt(5))

def fib_fast(n):
    """Source: http://www.nayuki.io/res/fast-fibonacci-algorithms/fastfibonacci.py
    Further reading: http://www.nayuki.io/page/fast-fibonacci-algorithms
    """
    if n == 0:
        return (0, 1)
    else:
        a, b = fib_fast(n / 2)
        c = a * (b * 2 - a)
        d = a * a + b * b
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

def fib_it_mod(n, q):
    a,b = 1,1
    i = 0
    while i  < (n-1):
        a,b = (b%q),((a+b)%q)
        i += 1
    return a

def fib_fast_mod(n, q):
    if n == 0:
        return (0, 1)
    else:
        a, b = fib_fast_mod(n / 2, q)
        c = (a * (b * 2 - a))%q
        d = (a * a + b * b)%q
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

#==============================================================================
# def sieve(n):
#     """Generates prime sieve of upto n**2: Slow version
#     http://en.literateprograms.org/Sieve_of_Eratosthenes_%28Python,_arrays%29"""
#     flags = np.zeros(n**2) + 1
#     for i in np.arange(2,n):
#         flags[i*i::i] = 0
#     return np.flatnonzero(flags)[2:]
#==============================================================================

def prime_nsquare(n):
    """Generates prime sieve of upto n**2: Much faster version
    http://en.literateprograms.org/Sieve_of_Eratosthenes_%28Python,_arrays%29"""
    import numpy as np
    flags = np.resize((0,1,0,0,0,1), (n**2,))
    np.put(flags, (0,2,3), 1)
    for i in np.arange(5,n,2):
        if flags[i]:
            flags[i*i::i] = 0
    return np.flatnonzero(flags)[2:]

def rootContFrac(n):
    '''
    Source: https://gist.github.com/rubik/1454917#file_a_cfrac.py
    Construct a continued fraction from a square root. The argument
    `n` should be an integer representing the radicand of the root:

        >>> rootContFrac(2)
        (1, [2])
        >>> rootContFrac(4)
        (2,)
        >>> rootContFrac(97)
        (9, [1, 5, 1, 1, 1, 1, 1, 1, 5, 1, 18])
    '''

    a0 = int(n**0.5)
    r = (a0, [])
    a, b, c = 1, 2 * a0, a0 ** 2 - n
    delta = (4*n)**0.5

    while True:
        try:
            d = int((b + delta) / (2 * c))
        except ZeroDivisionError: # a perfect square
            return (r[0],)
        a, b, c = c, -b + 2*c*d, a - b*d + c*d ** 2
        r[1].append(abs(d))
        if abs(a) == 1:
            break
    return r

def normfcontFrac2normFrac(cf):
    from fractions import Fraction
    return cf[0] + reduce(lambda d, n: 1 / (d + n), cf[:0:-1], Fraction(0))

def genIntegerPartition(n):
    "Source: http://jeromekelleher.net/partitions.php"
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

@memoize
def IntPartitionCount(n):
    if n<0:
        return 0
    if (n==0) or (n==1):
        return 1
    pain = 0
    for k in range(1, n+1):
        pain += (-1)**(k+1)*( IntPartitionCount(n-(k*(3*k-1))/2) + IntPartitionCount(n-(k*(3*k+1))/2) )
    return pain

def PartitionCount(n):
    p = [1]*(n + 1)
    for i in xrange(1, n + 1):
    	j, k, s = 1, 1, 0
    	while j > 0:
    		j = i - (3 * k * k + k) / 2
    		if j >= 0:
    			s -= (-1) ** k * p[j]
    		j = i - (3 * k * k - k) / 2
    		if j >= 0:
    			s -= (-1) ** k * p[j]
    		k += 1
    	p[i] = s
    return p[n]

def combntion(N,k): # from scipy.comb(), but MODIFIED!
    if (k > N) or (N < 0) or (k < 0):
        return 0L
    N,k = map(long,(N,k))
    top = N
    val = 1L
    while (top > (N-k)):
        val *= top
        top -= 1
    n = 1L
    while (n < k+1L):
        val /= n
        n += 1
    return val

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok / ktok
    else:
        return 0

def PartitionCountModVec(n, q=10**6):
    "Very fast calc of all partition counts <= n modulo q"
    p = [1]*(n + 1)
    for i in range(1, n + 1):
    	j, k, s = 1, 1, 0
    	while j > 0:
    		j = i - (3 * k * k + k) // 2
    		if j >= 0:
    			s -= (-1) ** k * p[j]
    		j = i - (3 * k * k - k) // 2
    		if j >= 0:
    			s -= (-1) ** k * p[j]
    		k += 1
    	p[i] = s%q
    return p[1:]

def modular_sqrt(a, p):
    """ Find a quadratic residue (mod p) of 'a'. p
        must be an odd prime.

        Solve the congruence of the form:
            x^2 = a (mod p)
        And returns x. Note that p - x is also a root.

        0 is returned is no square root exists for
        these a and p.

        The Tonelli-Shanks algorithm is used (except
        for some simple cases in which the solution
        is known from an identity). This algorithm
        runs in polynomial time (unless the
        generalized Riemann hypothesis is false).
    """
    # Source: http://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return p
    elif p % 4 == 3:
        return pow(a, (p + 1) / 4, p)

    # Partition p-1 to s * 2^e for an odd s (i.e.
    # reduce all the powers of 2 from p-1)
    #
    s = p - 1
    e = 0
    while s % 2 == 0:
        s /= 2
        e += 1

    # Find some 'n' with a legendre symbol n|p = -1.
    # Shouldn't take long.
    #
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1

    # Here be dragons!
    # Read the paper "Square roots from 1; 24, 51,
    # 10 to Dan Shanks" by Ezra Brown for more
    # information
    #

    # x is a guess of the square root that gets better
    # with each iteration.
    # b is the "fudge factor" - by how much we're off
    # with the guess. The invariant x^2 = ab (mod p)
    # is maintained throughout the loop.
    # g is used for successive powers of n to update
    # both a and b
    # r is the exponent - decreases with each update
    #
    x = pow(a, (s + 1) / 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e

    while True:
        t = b
        m = 0
        for m in xrange(r):
            if t == 1:
                break
            t = pow(t, 2, p)

        if m == 0:
            return x

        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m


def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
        Euler's criterion. p is a prime, a is
        relatively prime to p (if p divides
        a, then a|p = 0)

        Returns 1 if a has a square root modulo
        p, -1 otherwise.
    """
    ls = pow(a, (p - 1) / 2, p)
    return -1 if ls == p - 1 else ls

class Totient:
    def __init__(self, n):
        self.totients = [1 for i in range(n)]
        for i in range(2, n):
            if self.totients[i] == 1:
                for j in range(i, n, i):
                    self.totients[j] *= i - 1
                    k = j / i
                    while k % i == 0:
                        self.totients[j] *= i
                        k /= i
    def __call__(self, i):
        return self.totients[i]

def totientpair(factors, N) :
    """
    Generates all number-totient pairs below N, unordered, from the prime factors.
    Yields (n, phi(n))
    """
    ps = sorted(set(factors))
    omega = len(ps)

    def rec_gen(n = 0) :
        if n == omega :
            yield (1,1)
        else :
            pows = [(1,1)]
            val = ps[n]
            while val <= N :
                pows += [(val, val - pows[-1][0])]
                val *= ps[n]
            for q, phi_q in rec_gen(n + 1) :
                for p, phi_p in pows :
                    if p * q > N :
                        break
                    else :
                        yield p * q, phi_p * phi_q

    for p in rec_gen() :
        yield p

def divisorsum(n, k=1):
    u = factorint(n) # from sympy.ntheory
    tulip = 1
    for p in u:
        tulip *= (p**(k*(u[p]+1))-1)/(p**k-1)
    return tulip

def divisorGen(n):
    """
    Generates divisors from dict of prime factors
    Source: http://stackoverflow.com/a/171784/2724299
    """
    factors = list(factorint(n))  # from sympy.ntheory
    nfactors = len(factors)
    f = [0] * nfactors
    while True:
        yield reduce(lambda x, y: x*y, [factors[x][0]**f[x] for x in range(nfactors)], 1)
        i = 0
        while True:
            f[i] += 1
            if f[i] <= factors[i][1]:
                break
            f[i] = 0
            i += 1
            if i >= nfactors:
                return
