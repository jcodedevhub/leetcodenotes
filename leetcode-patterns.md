## 1. Fundamentals & Utilities

**Swapping**: a, b = b, a

**Enumerate**: for i, x in enumerate(arr):

**Range**: For i in range(1,6): print(i) -> 1,2,3,4,5

**List**: list[start:end]: Gets elements from start index to end-1 index.

**Zip:**

- **Purpose**: Iterate two (or more) sequences in parallel, pairing up their elements.

```python
A = [1, 2, 3]
B = ['a', 'b', 'c']

for x, y in zip(A, B):
    print(x, y)
# Output:
# 1 a
# 2 b
# 3 c
```

- **Common uses**  
  1. Summing pairwise elements  
     ```python
     C = [x + y for x, y in zip([1,2,3], [4,5,6])]
     # C == [5, 7, 9]
     ```
  2. Building a dict from two lists  
     ```python
     keys   = ['apple', 'banana', 'cherry']
     values = [3, 2, 5]
     d = dict(zip(keys, values))
     # d == {'apple': 3, 'banana': 2, 'cherry': 5}
     ```
  3. Stopping at the shortest list  
     ```python
     list(zip([1,2,3,4], ['x','y']))
     # [(1,'x'), (2,'y')]
     ```
- `zip(...)` → parallel iteration  


**Unpacking (`*args`, `**kwargs`, and sequence unpacking)**

```python
def foo(a, b, *args, **kwargs):
    print("a =", a)
    print("b =", b)
    print("additional positional:", args)
    print("keyword args:", kwargs)

foo(1, 2, 3, 4, x=10, y=20)
# a = 1
# b = 2
# additional positional: (3, 4)
# keyword args: {'x': 10, 'y': 20}
```

- `*args` collects extra positional arguments into a tuple.  
- `**kwargs` collects extra keyword arguments into a dict.

2. Function calls

```python
def bar(x, y, z):
    return x + y + z

nums = [10, 20, 30]
print(bar(*nums))          # unpack list → 60

params = {'x': 1, 'y': 2, 'z': 3}
print(bar(**params))       # unpack dict → 6
```
 
- `*` in definitions/calls → pack/unpack positional args  
- `**` in definitions/calls → pack/unpack keyword args  
- Sequence unpacking lets you split lists/tuples flexibly.

---

## 2. Sorting Algorithms (Templates)

### 2.1 Merge Sort (O(n log n), stable)  
Use when you need a stable, divide-and-conquer sort.  
**Example (912. Sort an Array)**  
Question:  
​ Given an integer array `nums`, return the array sorted in non-decreasing order.  
Task: implement merge sort on `nums` to produce a new sorted list.

```python
def merge_sort(a):
    if len(a) <= 1:
        return a
    m = len(a)//2
    L, R = merge_sort(a[:m]), merge_sort(a[m:])
    res, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            res.append(L[i]); i += 1
        else:
            res.append(R[j]); j += 1
    res.extend(L[i:]); res.extend(R[j:])
    return res
```

ASCII Split/Merge:
```
[5,2,4,6,1,3]
   /         \
[5,2,4]     [6,1,3]
 /\  /\      /\  /\
[5][2,4]   [6][1,3]
   … merge ↑      ↑
```

**Tweak**: change `<` → `<=` to preserve original order of equal elements.

---

### 2.2 Quick Sort (avg O(n log n), worst O(n²))  
Use when in-place sorting with average n log n is acceptable.  
**Example (215. Kth Largest Element in an Array)**  
Question:  
​ Given an integer array `nums` and integer `k`, return the kᵗʰ largest element in the array.  
Task: you can use quick-select (a variant of quick sort partition) to find the kᵗʰ largest in average O(n) time.

```python
import random

def quick_sort(a, l=0, r=None):
    if r is None: r = len(a)-1
    if l < r:
        pi = random.randint(l, r)
        a[pi], a[r] = a[r], a[pi]      # random pivot
        p = partition(a, l, r)
        quick_sort(a, l, p-1)
        quick_sort(a, p+1, r)
    return a

def partition(a, l, r):
    pivot = a[r]
    i = l
    for j in range(l, r):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]; i += 1
    a[i], a[r] = a[r], a[i]
    return i
```

Partition visualization:
```
[3,6,8,10,1,2,1], pivot=1
→ smaller: [ ], larger: [3,6,8...]
swap pivot into final place at index i
```

**Tweak**: use median-of-three or left/middle/right randomization.

---

### 2.3 Heap Sort (O(n log n), not stable)  
Use when you need in-place sort with guaranteed n log n.  
**Example (912. Sort an Array)**  
Question: same as Merge Sort example above.  
Task: heapify then pop repeatedly.

```python
import heapq

def heap_sort(a):
    heapq.heapify(a)           # O(n)
    return [heapq.heappop(a)   # n pops, each O(log n)
            for _ in range(len(a))]
```

Binary‐heap layout in array:  
```
      a[0]
     /    \
   a[1]  a[2]
  /\      /\
 ...
```

**Tweak**: for descending order, store `-x` and invert on pop.

---

## 3. Pattern Templates

Each pattern below shows  
1. a **concrete problem** (number & title),  
2. **Question** & **Task** in one or two lines,  
3. an **ASCII sketch**, and  
4. **tweak points** for adapting the template.

---

### A. Sliding Window  
When: contiguous subarray problems (fixed or variable size).  
**Example (209. Minimum Size Subarray Sum)**  
Question:  
​ Given array `nums` of positive ints and target `s`, find the minimal length of a contiguous subarray of which the sum ≥ *s*. Return 0 if none.  
Task: expand and shrink a window to maintain sum ≥ *s*.

```python
def min_subarray_len(nums, s):
    left = 0
    curr = 0
    res = float('inf')
    for right, x in enumerate(nums):
        curr += x
        while curr >= s:
            res = min(res, right - left + 1)
            curr -= nums[left]
            left += 1
    return res if res != float('inf') else 0
```

ASCII (variable size):
```
s = 7, nums = [2,3,1,2,4,3]
window [2,3,1,2] sum=8 → shrink → [3,1,2] sum=6 → expand
```

**Tweak**:  
- Fixed size *k*: subtract `nums[i-k]` once `i ≥ k`.  
- Variable size: `while` to shrink, `if` only to expand.

---

### B. Two Pointers  
When: sorted arrays/strings, pair sums, container water.  
**Example (11. Container With Most Water)**  
Question:  
​ Given `height` array, pick two indices *i* < *j* so that area = `(j−i)*min(height[i],height[j])` is maximized. Return the max area.  
Task: move the smaller pointer inward to possibly increase area.

```python
def container_max_area(height):
    l, r = 0, len(height)-1
    best = 0
    while l < r:
        best = max(best, min(height[l],height[r])*(r-l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return best
```

ASCII:
```
l→           ←r
|             |
|     /\      |
|    /  \     |
---------------
```

**Tweak**: for 3Sum or closest-sum, nest one pointer & skip duplicates.

---

### C. Fast & Slow Pointers  
When: cycle detection, middle of list, in-place reorder.  
**Example (142. Linked List Cycle II)**  
Question:  
​ Given head of singly linked list, return the node where the cycle begins. If none, return `None`.  
Task: detect cycle, then find entry by resetting one pointer to head and advancing both by one.

```python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # find entry
            ptr = head
            while ptr is not slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None
```

ASCII:
```
      _______
     ↓       |
1→2→3→4→5→6—
      ↑   ↑
    slow fast
```

---

### D. Prefix Sum / Difference Array  
When: range sum queries, count subarrays summing to k.  
**Example (560. Subarray Sum Equals K)**  
Question:  
​ Given `nums` and integer `k`, return the count of continuous subarrays whose sum equals *k*.  
Example 1:

Input: nums = [1,1,1], k = 2
Output: 2

Task: maintain running sum and a hash-map of frequencies of previous prefix sums.

By maintaining a running sum of elements and using a hash map to store the frequency of each prefix sum encountered, we can efficiently count the subarrays that sum to k. 

If the difference between the current prefix sum and k has been seen before, it indicates that there is a subarray that sums to k. 

i.e K = 7 

nums = [7, 1, 6] 

Cumsum = [7, 8, 14] 

Cumsum - K = [0, 1, 7]

0 and 7 appeared so count = 2

```python
from collections import defaultdict

def subarraySum(nums, k):
    count = 0
    curr = 0
    freq = defaultdict(int)
    freq[0] = 1
    for x in nums:
        curr += x
        count += freq[curr - k]
        freq[curr] += 1
    return count

#faster alternative
def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = 0
        prefix_sum = 0
        prefix_sum_count = {0: 1}  # Initialize with prefix sum 0 and count 1
        
        for num in nums:
            prefix_sum += num  # Update the running prefix sum
            if (prefix_sum - k) in prefix_sum_count:
                count += prefix_sum_count[prefix_sum - k]  # Increment count if (prefix_sum - k) is found
            if prefix_sum in prefix_sum_count:
                prefix_sum_count[prefix_sum] += 1  # Update the frequency of the current prefix sum
            else:
                prefix_sum_count[prefix_sum] = 1  # Initialize the frequency if the prefix sum is seen for the first time
        
        return count        

```

**Tweak**: for difference array on updates, apply +val at `l`, –val at `r+1`.

---

### E. Hash-Map (Dict)  
When: lookups, frequency counts, grouping, two-sum.  
**Example (1. Two Sum)**  
Question:  
​ Given an array `nums` and `target`, return indices `[i,j]` such that `nums[i] + nums[j] == target`.  
Task: store seen values in a dict for O(n) lookup.

```python
def two_sum(nums, target):
    d = {}
    for i, x in enumerate(nums):
        if target-x in d:
            return [d[target-x], i]
        d[x] = i
    return []
```

**Tweak**: use `Counter` for frequency problems, `defaultdict(list)` for grouping anagrams.

---

### F. Binary Search  
When: sorted arrays, monotonic functions.  
**Example (33. Search in Rotated Sorted Array)**  
Question:  
​ Given a rotated sorted array `nums` (no duplicates) and `target`, return its index or –1 if not found.  
Task: modify binary search by checking which half is sorted.

```python
def search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l + r)//2
        if nums[m] == target:
            return m
        # left half is sorted
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return -1
```

**Tweak**: implement `lower_bound`/`upper_bound` by tightening `l<r` and moving pointers.

---

### G. Heap (Priority Queue)  
When: top-k, merging sorted lists, streaming median.  
**Example (347. Top K Frequent Elements)**  
Question:  
​ Given `nums` and integer `k`, return the `k` most frequent elements.  
Task: build a min-heap of size `k` on frequencies.

```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    freq = Counter(nums)
    h = []
    for num, f in freq.items():
        heapq.heappush(h, (f, num))
        if len(h) > k:
            heapq.heappop(h)
    return [num for f, num in h]
```

**Tweak**: invert sign for max-heap, use heapify on initial k elements.

---

### H. BFS / DFS (Graph & Tree)  
When: shortest paths, connectivity, tree traversals.  

**Example (200. Number of Islands — BFS)**  
Question:  
​ Given 2D grid of `'1'` (land) and `'0'` (water), count the number of islands (4-dir adjacencies).  
```python
from collections import deque

def num_islands(grid):
    if not grid: return 0
    n, m = len(grid), len(grid[0])
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
    cnt = 0
    for i in range(n):
      for j in range(m):
        if grid[i][j] == '1':
          cnt += 1
          grid[i][j] = '0'
          q = deque([(i,j)])
          while q:
            x,y = q.popleft()
            for dx,dy in dirs:
              nx,ny = x+dx, y+dy
              if 0<=nx<n and 0<=ny<m and grid[nx][ny]=='1':
                grid[nx][ny]='0'
                q.append((nx,ny))
    return cnt
```

**Example (94. Binary Tree Inorder Traversal — DFS)**  
Question:  
​ Given `root` of a binary tree, return its inorder traversal (left, root, right).  
```python
def inorderTraversal(root):
    res = []
    def dfs(node):
        if not node: return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res
```

---

### I. Dynamic Programming  

#### 1D DP  
**Example (121. Best Time to Buy and Sell Stock)**  
Question:  
​ Given array `prices`, compute the maximum profit from one buy‐sell transaction.  
```python
def max_profit(prices):
    minp, ans = float('inf'), 0
    for p in prices:
        minp = min(minp, p)
        ans = max(ans, p - minp)
    return ans
```

#### 2D DP  
**Example (62. Unique Paths)**  
Question:  
​ Given `m×n` grid, count paths from top-left to bottom-right moving only down or right.  
```python
def unique_paths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
      for j in range(1, n):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

**Tweak**: roll 2D → 1D array to save space.

---

### J. Backtracking (Permutations/Subsets/Combinations)  
When: generate all configurations, n-queens, Sudoku.  
**Example (46. Permutations)**  
Question:  
​ Given array `nums` of distinct ints, return all possible permutations.  
```python
def permute(nums):
    res = []
    def dfs(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]: continue
            used[i] = True
            path.append(x)
            dfs(path, used)
            path.pop()
            used[i] = False
    dfs([], [False]*len(nums))
    return res
```

**Tweak**: to generate combinations, pass start index and avoid used array.

---

### K. Greedy  
When: local optimal choice implies global optimum.  
**Example (435. Non-overlapping Intervals)**  
Question:  
​ Given list of `intervals`, find the minimum number to remove so that the rest do not overlap.  
```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    prev_end = float('-inf')
    cnt = 0
    for s, e in intervals:
        if s >= prev_end:
            prev_end = e
        else:
            cnt += 1
    return cnt
```

**Tweak**: sort by end time, pick earliest finishing intervals.

---

### L. Union-Find (Disjoint Set)  
When: connectivity, Kruskal MST, accounts-merge.  
**Example (547. Number of Provinces)**  
Question:  
​ Given adjacency matrix `isConnected`, return number of connected components (provinces).  
```python
class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def findCircleNum(isConnected):
    n = len(isConnected)
    uf = UF(n)
    for i in range(n):
      for j in range(n):
        if isConnected[i][j]:
          uf.union(i, j)
    return len({uf.find(i) for i in range(n)})
```

**Tweak**: add rank or size heuristics for speed.

---

### M. Bit Manipulation  
When: parity, XOR tricks, subset masks.  
**Example (136. Single Number)**  
Question:  
​ Given non-empty array `nums` where every element appears twice except one, return the single one.  
```python
def singleNumber(nums):
    res = 0
    for x in nums:
        res ^= x
    return res
```

**Tweak**: use `x & (x−1)` to clear lowest set bit when counting bits.

---

### N. Monotonic Queue (Deque)  
When: sliding‐window min/max, stock span, “next greater” problems.  
**Example (239. Sliding Window Maximum)**  
- **Question**: Given array `nums` and window size `k`, return an array of the maximums of each sliding window of size `k`.  
- **Task**: maintain a deque of indices whose corresponding values are in decreasing order.

```python
from collections import deque

def max_sliding_window(nums, k):
    q = deque()   # store indices, nums[q] decreasing
    res = []
    for i, x in enumerate(nums):
        # pop out‐of‐window
        if q and q[0] == i - k:
            q.popleft()
        # pop smaller elements
        while q and nums[q[-1]] < x:
            q.pop()
        q.append(i)
        # window has formed
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```

ASCII (k=3):
```
[1,3,−1,−3,5,3,6,7]
  ↑
deque stores indices of [3,−1] in order
```

**Tweak**: reverse comparison for sliding‐window minimum.

---

### O. Segment Tree / Fenwick Tree (BIT)  
When: frequent range‐sum / range‐update / point‐update queries.  
**Example (307. Range Sum Query – Mutable)**  
- **Question**: Design a data structure that supports updating an element and querying the sum of a range in an array.  
- **Task**: implement a Fenwick Tree (BIT) for O(log n) updates & queries.

```python
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.fw = [0] * (n+1)
    def update(self, i, delta):
        # i: 0-based
        i += 1
        while i <= self.n:
            self.fw[i] += delta
            i += i & -i
    def query(self, i):
        # prefix sum [0..i]
        i += 1
        s = 0
        while i > 0:
            s += self.fw[i]
            i -= i & -i
        return s

class NumArray:
    def __init__(self, nums):
        self.nums = nums
        self.ft = Fenwick(len(nums))
        for i, x in enumerate(nums):
            self.ft.update(i, x)
    def update(self, i, val):
        delta = val - self.nums[i]
        self.nums[i] = val
        self.ft.update(i, delta)
    def sumRange(self, i, j):
        return self.ft.query(j) - self.ft.query(i-1)
```

**Tweak**: use segment tree for range‐min/max or range‐assign.  

---

### P. Topological Sort  
When: DAG ordering, cycle detection, scheduling.  
**Example (207. Course Schedule)**  
- **Question**: Given `numCourses` and `prerequisites` pairs `[a,b]` (must take b before a), return `True` if you can finish all courses.  
- **Task**: perform Kahn’s algorithm (BFS) to check for cycle.

```python
from collections import deque, defaultdict

def canFinish(numCourses, prereq):
    indeg = [0] * numCourses
    g = defaultdict(list)
    for a, b in prereq:
        g[b].append(a)
        indeg[a] += 1
    q = deque([i for i in range(numCourses) if indeg[i] == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == numCourses
```

**Tweak**: for actual order, collect nodes in `order` list; for DFS version, track visit‐states.

---

### Q. Trie (Prefix Tree)  
When: prefix search, autocomplete, wildcard matching.  
**Example (208. Implement Trie)**  
- **Question**: Implement `insert(word)`, `search(word)`, and `startsWith(prefix)` on a Trie.  
- **Task**: build a tree of dict‐children and an `isWord` flag.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.isWord = True

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.isWord

    def startsWith(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True
```

**Tweak**: store counts at each node for fast prefix‐count queries; add `weight` for weighted autocomplete.

---

### R. Meet-in-the-Middle  
When: split array into halves to reduce 2ⁿ → 2·2ⁿ⁄² for sum/combination counting.  
**Example (454. 4Sum II)**  
- **Question**: Given four lists `A,B,C,D`, count tuples `(i,j,k,l)` with `A[i]+B[j]+C[k]+D[l] == 0`.  
- **Task**: compute all `A+B` sums, store frequencies, then for each `C+D` sum look up negation.

```python
from collections import Counter

def fourSumCount(A, B, C, D):
    ab = Counter(a + b for a in A for b in B)
    cnt = 0
    for c in C:
        for d in D:
            cnt += ab[-(c+d)]
    return cnt
```

**Tweak**: apply same idea to 3Sum into O(n²) by hashing 2-sum, or subset‐sum for n up to 40.

---

### S. Bitmask DP  
When: DP over subsets, TSP, assignment problems.  
**Example (698. Partition to K Equal Sum Subsets)**  
- **Question**: Given `nums` and `k`, can you partition `nums` into `k` subsets with equal sums?  
- **Task**: DP over mask of used elements and current subset sum.

```python
from functools import lru_cache

def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k: return False
    target = total // k
    n = len(nums)

    @lru_cache(None)
    def dfs(mask, curr_sum, groups_done):
        if groups_done == k:
            return True
        if curr_sum == target:
            return dfs(mask, 0, groups_done + 1)
        for i in range(n):
            if not (mask & (1<<i)) and curr_sum + nums[i] <= target:
                if dfs(mask | (1<<i), curr_sum + nums[i], groups_done):
                    return True
        return False

    return dfs(0, 0, 0)
```

**Tweak**: memoize only on `mask` if you derive `groups_done` and `curr_sum` from bit‐count and prefix sums.

---

### T. Geometry / Computational Geometry  
When: convex hull, rectangle/triangle areas, closest‐pair.  
**Example (84. Largest Rectangle in Histogram)**  
- **Question**: Given `heights` of histogram bars, find the largest rectangular area.  
- **Task**: use a monotonic stack to find nearest smaller bar on left/right for each index.

```python
def largestRectangleArea(heights):
    stack = [-1]   # indices of increasing heights
    res = 0
    for i, h in enumerate(heights):
        while stack[-1] != -1 and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            res = max(res, height * width)
        stack.append(i)
    # clean up remaining bars
    n = len(heights)
    while stack[-1] != -1:
        height = heights[stack.pop()]
        width = n - stack[-1] - 1
        res = max(res, height * width)
    return res
```

ASCII (bars + stack):
```
|   ■
| ■ ■
| ■ ■ ■
------------
 idx→ 0 1 2
```

**Tweak**: adapt to “maximal rectangle” in matrix by treating each row as a histogram.

---

## 4. Plug-and-Play Snippets

```python
# 1. Max Profit (Single Transaction)
def maxProfit(prices):
    minp, ans = float('inf'), 0
    for p in prices:
        minp = min(minp, p)
        ans = max(ans, p-minp)
    return ans

# 2. Distribute Money (Greedy + Fix-Up)
def distMoney(money, children):
    if money < children: return -1
    x = min(money//8, children)
    while x > 0:
        left = money - 8*x
        k = children - x
        if left < k or (k==1 and left==4) or (k==0 and left>0):
            x -= 1
        else:
            break
    return x

# 3. Find Missing & Repeated in n×n Grid
def findMissingAndRepeated(grid):
    n2 = len(grid)**2
    seen, dup = set(), 0
    total = 0
    for x in sum(grid, []):
        if x in seen:
            dup = x
            total -= x
        else:
            seen.add(x)
        total += x
    miss = n2*(n2+1)//2 - total
    return [dup, miss]
```

### Self notes
```python
import numpy as np
import yfinance as yf
import scipy.optimize as sco
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader.data as pdr


def get_sp500_tickers(limit: int = 100) -> List[str]:
    
    df = pd.read_csv("nasdaqscreener.csv")

    # Clean the Market Cap column — remove '$' and ',' and convert to numeric
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

    # Drop rows with missing or invalid Market Cap
    df = df.dropna(subset=['Market Cap'])

    # Sort by Market Cap ascending
    df_sorted = df.sort_values(by='Market Cap', ascending=True)

    # Select the 100 smallest market caps
    smallest_100 = df_sorted.head(100)

    # Get list of stock symbols
    smallest_100_symbols = smallest_100['Symbol'].tolist()

    return smallest_100_symbols


def fetch_data(ticker_sym):
    t = yf.Ticker(ticker_sym)
    # 1-year daily price history
    hist = t.history(period="1y")["Close"].dropna()
    # equity volatility annualized
    ret = hist.pct_change().dropna()
    sigma_E = ret.std() * np.sqrt(252)
    # market cap (equity value)
    E = t.info["marketCap"]
    
    qbs = t.quarterly_balance_sheet
    # short- and long-term debt (default to 0 if missing)
    SD = t.info.get("shortTermDebt", 0.0)
    LD = t.info.get("longTermDebt",  0.0)
    
    latest_q = qbs.columns[0]
    SD = qbs.at["Current Debt And Capital Lease Obligation", latest_q]
    LD = qbs.at["Long Term Debt And Capital Lease Obligation",  latest_q]
    
    
    # Default point D := short-term debt + 0.5 * long-term debt
    D = SD + 0.5 * LD
    if D == 0:
        print("Default point = 0, check SD and LD")
    
    print(D)
    
    return E, sigma_E, D

def solve_assets(E, sigma_E, D, T=1.0, r=0.02):
    # Black-Scholes call price and equity vol equations
    def equations(x):
        
        #E market value (like the spot price)
        #D face value of zero coupon debt (similar to strike price Because at debt maturity T, equity holders get max(V_T – D, 0))
        #V = total value for firms asset (equity + debt)
        #σ_E observed volatility (calced as annaulized vol)
        #σ_V unknown volatility of firm's assets
        #r risk free rate
        #T is time to maturity
        
        V, sigma_V = x
        d1 = (np.log(V/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
        d2 = d1 - sigma_V*np.sqrt(T)
        # call value
        C = V * norm.cdf(d1) - D * np.exp(-r*T) * norm.cdf(d2)
        # implied equity vol
        sigma_E_calc = (V/E) * norm.cdf(d1) * sigma_V
        
        #C-E the call‐price on assets (with strike D) must equal today’s equity price
        #sigma_E_calc - sigma_E pins the volatility of that equity to its observed vol
        return [C - E, sigma_E_calc - sigma_E]

    # initial guesses: V ~ E + D, sigma_V ~ sigma_E * E/(E+D)
    V0 = E + D
    
    #pretending N(d₁)≃1 (or roughly of order unity) rearrage where sigma_V =
    sigma0 = sigma_E * E/V0
    
    sol = sco.root(equations, x0=[V0, sigma0], tol=1e-8)
    if not sol.success:
        raise RuntimeError("Root finding failed")
    return sol.x  # V0, sigma_V



def kmv_edf(ticker_sym, T=1.0, r=0.02, mu=None):
    # 1) Fetch data
    E, sigma_E, D = fetch_data(ticker_sym)
    # 2) Solve for asset value & vol
    V0, sigma_V = solve_assets(E, sigma_E, D, T=T, r=r)
    # 3) Choose drift mu = r if not specified
    mu = mu if mu is not None else r
    # 4) Distance to default
    DD = (np.log(V0/D) + (mu - 0.5*sigma_V**2)*T) / (sigma_V * np.sqrt(T))
    # 5) Approximate EDF
    EDF = norm.cdf(-DD)
    return {
        "EquityValue": E,
        "EquityVol": sigma_E,
        "AssetValue": V0,
        "AssetVol": sigma_V,
        "DefaultPoint": D,
        "DistanceToDefault": DD,
        "EDF": EDF
    }



def get_one_year_rf_rate(past_days: int = 30):

    end = datetime.utcnow().date()
    start = end - timedelta(days=past_days)

    try:
        # Fetch the series; this returns a DataFrame with column 'DGS1'
        df = pdr.get_data_fred('DGS1', start, end)
        # Drop any missing values, take the last valid observation
        latest = df['DGS1'].dropna().iloc[-1]
        # The raw data is in percent (e.g. 4.53), so convert to decimal
        return float(latest) / 100.0

    except Exception:
        # If anything goes wrong (network, no data, etc.) return None
        return None

    

if __name__ == "__main__":
    # Step 1: Get 100 smallest market-cap tickers
    tickers = get_sp500_tickers(limit=100)

    r = get_one_year_rf_rate()
    results = []

    print(f"Fetched {len(tickers)} tickers. Starting KMV EDF calculations...\n")

    for ticker in tickers:
        try:
            res = kmv_edf(ticker, r=r)
            res["Ticker"] = ticker
            results.append(res)
            print(f" {ticker}: EDF={res['EDF']:.6f}, DD={res['DistanceToDefault']:.3f}")
        except Exception as e:
            print(f" {ticker}: skipped due to {e}")
            continue

    # Step 2: Build DataFrame from all results
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("\nNo valid KMV results computed.")
    else:
        # Step 3: Sort by EDF (ascending = safer, descending = riskier)
        df_sorted = df_results.sort_values(by="EDF", ascending=False)

        # Step 4: Select bottom 20 (highest EDF)
        bottom_20 = df_sorted.head(20)

        # Step 5: Display summary
        print("\n--- Bottom 20 (Highest Default Risk) ---")
        print(bottom_20[["Ticker", "EDF", "DistanceToDefault"]])

        # Optional: save results to CSV
        bottom_20.to_csv("kmv_bottom20.csv", index=False)
        print("\nSaved bottom 20 results to 'kmv_bottom20.csv'.")

```

Feel free to refine:

Use an empirical DD→EDF lookup table (if you have one).
Estimate μ (asset drift) via CAPM or historical drift.
Tweak the “default point” formula to match your firm’s capital structure.
