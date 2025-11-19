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

### Self notes 2
```python
import pandas as pd
import numpy as np
import yfinance as yf

def get_stock_data(ticker_list):
    """
    Fetches 5y of history for each ticker, computes daily returns and volume‐scores.
    Returns:
      price_df       : absolute Close prices
      close_pct_df   : daily pct‐changes of Close
      volume_df      : EWM‐z‐scores of log1p(volume)
      successful, failed : lists of tickers
    """
    now = pd.Timestamp.now().normalize()
    five_years_ago = now - pd.DateOffset(years=5)

    price_dict, ret_dict, vol_dict = {}, {}, {}
    successful, failed = [], []

    for tic in ticker_list:
        try:
            df = yf.download(
                tic,
                start=(five_years_ago - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                end=now.strftime("%Y-%m-%d"),
                progress=False,
            )
            if df.empty:
                print(f"  → {tic}: no data")
                failed.append(tic)
                continue

            df = df.reset_index()[['Date','Close','Volume']]
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df = df.set_index('Date').sort_index()

            if df.index.min() > five_years_ago:
                print(f"  → {tic}: earliest date {df.index.min().date()} is after {five_years_ago.date()}")
                failed.append(tic)
                continue

            # compute returns & vol score
            ret = df['Close'].pct_change()
            lv  = np.log1p(df['Volume'])
            μ   = lv.ewm(span=15).mean()
            σ   = lv.ewm(span=15).std()
            vol_score = (lv - μ) / σ

            joint = pd.concat([df['Close'], ret, vol_score], axis=1).dropna()
            joint.columns = ['Price','Ret','VolScore']

            # stash into dicts
            price_dict[tic]     = joint['Price']
            ret_dict[tic]       = joint['Ret']
            vol_dict[tic]       = joint['VolScore']

            successful.append(tic)

        except Exception as e:
            print(f"  → {tic}: Exception: {e}")
            failed.append(tic)

    # one-shot concat (no fragmentation)
    if price_dict:
        price_df      = pd.concat(price_dict, axis=1)
        close_pct_df  = pd.concat(ret_dict,   axis=1)
        volume_df     = pd.concat(vol_dict,   axis=1)
    else:
        price_df = close_pct_df = volume_df = pd.DataFrame()

        
    price_df.to_excel("close_price.xlsx")
    close_pct_df.to_excel("price_prc.xlsx")
    volume_df.to_excel("volume.xlsx")

    print(f"Done. Successful: {len(successful)}, Failed: {len(failed)}")
    return price_df, close_pct_df, volume_df, successful, failed


def correlations_calc(close_pct_df, volume_df, weights):
    w_ret, w_vol = weights['returns'], weights['volume']
    C_ret = close_pct_df.corr()
    C_vol = volume_df.corr()
    return C_ret * w_ret + C_vol * w_vol


def eigenvector_centrality(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.0,
    use_pagerank: bool = True,
    seed_ticker: str = None,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100
) -> pd.Series:
    A = corr_matrix.abs().fillna(0).values  # ABS for centrality - measures strength of connection
    np.fill_diagonal(A, 0.0)
    A[A < threshold] = 0.0

    if use_pagerank:
        if seed_ticker is None:
            raise ValueError("seed_ticker must be provided when using PageRank")
        row_sums = A.sum(axis=1, keepdims=True)
        P = np.divide(A, row_sums, where=row_sums!=0)
        n = A.shape[0]
        e = np.zeros(n, dtype=float)
        seed_idx = corr_matrix.columns.get_loc(seed_ticker)
        e[seed_idx] = 1.0
        x = e.copy()
        for i in range(max_iter):
            x_new = alpha * (P.T @ x) + (1 - alpha) * e
            if np.linalg.norm(x_new - x, 1) < tol:
                break
            x = x_new
        v = x / x.max() if x.max()!=0 else x
    else:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argmax(eigvals.real)
        v = np.abs(eigvecs[:, idx].real)
        v = v / v.max() if v.max()!=0 else v

    return pd.Series(v, index=corr_matrix.index, name="centrality")


def get_centrality_bucket(centrality_value, centrality_series):
    """
    Returns the bucket (0, 1, or 2) for a given centrality value.
    Buckets are based on tertiles of the centrality distribution.
    """
    tertiles = np.percentile(centrality_series.dropna(), [33.33, 66.67])
    if centrality_value <= tertiles[0]:
        return 0
    elif centrality_value <= tertiles[1]:
        return 1
    else:
        return 2


def build_score_table(
    corr_composite: pd.DataFrame,
    centrality: pd.Series,
    target: str,
    corr_thresh: float = 0.7,
    corr_weight: float = 0.7,
    cent_weight: float = 0.3,
    centrality_filter: bool = False,
    use_abs_corr: bool = False
) -> pd.DataFrame:
    """
    Returns stocks with:
      - |corr to target| > corr_thresh (or corr > corr_thresh if not using abs)
      - optional centrality_filter using bucket-based filtering
      - score = corr_weight*corr + cent_weight*centrality
    """
    if use_abs_corr:
        df = pd.DataFrame({
            'corr' : corr_composite[target].abs()
        }).drop(index=target)
    else:
        # Only positive correlations (stocks moving in same direction)
        df = pd.DataFrame({
            'corr' : corr_composite[target]
        }).drop(index=target)
    
    df = df[df['corr'] > corr_thresh]

    df['cent'] = centrality

    if centrality_filter:
        target_cent = centrality[target]
        target_bucket = get_centrality_bucket(target_cent, centrality)
        
        # Filter stocks in the same bucket
        df['bucket'] = df['cent'].apply(lambda x: get_centrality_bucket(x, centrality))
        df = df[df['bucket'] == target_bucket]
        df = df.drop(columns=['bucket'])

    df['score'] = corr_weight * df['corr'] + cent_weight * df['cent']
    return df.sort_values('score', ascending=False)


def rolling_backtest(
    price_df: pd.DataFrame,
    close_pct_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    target: str,
    param_combinations: list,
    train_years: int = 3,
    test_months: int = 1,
    use_abs_corr: bool = False,
    corr_weights: dict = {'returns': 0.7, 'volume': 0.3},
    corr_thresh: float = 0.0,
    centrality_filter: bool = False,
    stock_list: pd.DataFrame = None,
    filter_sector: str = None,
    filter_industry: str = None
):
    """
    For each (corr_weight, cent_weight) & rolling month:
      - snap cur to last trading day ≤ cur
      - train on prior train_years up to that snap‐date
      - test on next test_months up to its snap‐date
      - compute short‐sell max profit
    
    New parameters:
      - corr_thresh: minimum correlation threshold for composite strategy
      - centrality_filter: whether to filter by centrality bucket
      - stock_list: DataFrame with Symbol, Sector, Industry columns
      - filter_sector: if provided, only include stocks from this sector
      - filter_industry: if provided, only include stocks from this industry
    """
    if price_df.empty:
        raise ValueError("price_df is empty—nothing to backtest on.")

    # Apply sector/industry filtering if requested
    if stock_list is not None and (filter_sector or filter_industry):
        allowed_stocks = set(stock_list['Symbol'].tolist())
        
        if filter_sector:
            allowed_stocks = allowed_stocks.intersection(
                set(stock_list[stock_list['Sector'] == filter_sector]['Symbol'].tolist())
            )
        
        if filter_industry:
            allowed_stocks = allowed_stocks.intersection(
                set(stock_list[stock_list['Industry'] == filter_industry]['Symbol'].tolist())
            )
        
        # Filter dataframes to only include allowed stocks
        available_stocks = [col for col in price_df.columns if col in allowed_stocks]
        price_df = price_df[available_stocks]
        close_pct_df = close_pct_df[available_stocks]
        volume_df = volume_df[available_stocks]
        
        print(f"Filtered to {len(available_stocks)} stocks based on sector/industry criteria")

    # sorted unique trading dates
    dates = price_df.index.unique().sort_values()
    start = dates.min() + pd.DateOffset(years=train_years)
    end   = dates.max() - pd.DateOffset(months=test_months)

    results = {}

    for params in param_combinations:
        cw, tw = params['correlation_weight'], params['centrality_weight']

        for strat in ('corr','composite'):
            rows = []
            cur = start

            while cur <= end:
                # snap to last trading day on or before cur / cur+test_months
                train_end = dates[dates.get_indexer([cur], method='pad')[0]]
                test_end_unclipped = cur + pd.DateOffset(months=test_months)
                test_end  = dates[dates.get_indexer([test_end_unclipped], method='pad')[0]]

                # slices for price data
                train_slice = price_df.loc[:train_end]
                train_slice = train_slice.loc[train_end - pd.DateOffset(years=train_years):train_end]
                test_slice  = price_df.loc[train_end:test_end]
                
                # slices for returns and volume data
                returns_train = close_pct_df.loc[:train_end]
                returns_train = returns_train.loc[train_end - pd.DateOffset(years=train_years):train_end]
                volume_train = volume_df.loc[:train_end]
                volume_train = volume_train.loc[train_end - pd.DateOffset(years=train_years):train_end]

                # Calculate composite correlation using both returns and volume
                C_composite = correlations_calc(returns_train, volume_train, corr_weights)

                # Calculate returns correlation separately for the correlation-only strategy
                R = train_slice.pct_change().dropna(how='all')
                C_returns = R.corr()
                
                if centrality_filter:
                    cent = eigenvector_centrality(C_composite,
                                                  threshold=0.0,
                                                  use_pagerank=False)  # Use eigenvector instead
                else:
                    cent = eigenvector_centrality(C_composite,
                                                  threshold=0.0,
                                                  use_pagerank=True,
                                                  seed_ticker=target)

                # Get target's centrality for this period
                target_centrality = cent[target] if target in cent.index else np.nan
                

                # select & weight
                if strat == 'corr':
                    if use_abs_corr:
                        tb = pd.Series(C_returns[target].abs(), name='corr').drop(index=target)
                    else:
                        tb = pd.Series(C_returns[target], name='corr').drop(index=target)
                        tb = tb[tb > 0]  # Only positive correlations
                    
                    if len(tb) < 1:
                        # Not enough stocks, skip this period
                        continue
                    
                    else:
                        if len(tb) < 3: 
                    
                            top3 = tb.nlargest(len(tb)).index
                            wts  = tb.loc[top3] / tb.loc[top3].sum()
                        
                        else:
                            top3 = tb.nlargest(3).index
                            wts  = tb.loc[top3] / tb.loc[top3].sum()
                    
                    # Store top 3 correlation values
                    top3_corr = tb.loc[top3].to_dict()
                    top3_cent = cent.loc[top3].to_dict()
                    
                    # Calculate correlation of portfolio with target in test period
                    test_R = test_slice.pct_change().dropna(how='all')
                    portfolio_returns = sum(test_R[s] * wts[s] for s in top3)
                    portfolio_corr = portfolio_returns.corr(test_R[target]) if len(test_R) > 1 else np.nan

                else:  # composite
                    comp = build_score_table(
                        corr_composite=C_composite,
                        centrality=cent,
                        target=target,
                        corr_thresh=corr_thresh,
                        corr_weight=cw,
                        cent_weight=tw,
                        centrality_filter=centrality_filter,
                        use_abs_corr=use_abs_corr
                    )
                    
                    if len(comp) < 1:
                        # Not enough stocks, skip this period
                        print(f"Warning: Only {len(comp)} stocks met criteria for {strat} strategy on {train_end.date()}. Skipping period.")
                        continue
                    
                    else:
                        if len(comp) < 3:
                            top3 = comp.head(len(comp)).index
                            wts  = comp.loc[top3,'score'] / comp.loc[top3,'score'].sum()
                            
                        else:
                            top3 = comp.head(3).index
                            wts  = comp.loc[top3,'score'] / comp.loc[top3,'score'].sum()
                    
                    
                    # Store top 3 correlation and centrality values
                    top3_corr = comp.loc[top3, 'corr'].to_dict()
                    top3_cent = comp.loc[top3, 'cent'].to_dict()
                    
                    # Calculate correlation of portfolio with target in test period
                    test_R = test_slice.pct_change().dropna(how='all')
                    portfolio_returns = sum(test_R[s] * wts[s] for s in top3)
                    portfolio_corr = portfolio_returns.corr(test_R[target]) if len(test_R) > 1 else np.nan

                # compute short‐sell profit
                profs = {}
                for stock in top3:
                    P_entry = price_df.at[train_end, stock]
                    P_min   = test_slice[stock].min()
                    profs[stock] = (P_entry - P_min) / P_entry

                port_profit = sum(profs[s]*wts[s] for s in top3)

                rows.append({
                    'train_end'        : train_end,
                    'test_start'       : train_end,
                    'test_end'         : test_end,
                    'strategy'         : strat,
                    'corr_weight'      : cw,
                    'cent_weight'      : tw,
                    'selected_stocks'  : list(top3),
                    'weights'          : wts.to_dict(),
                    'top3_correlations': top3_corr,
                    'top3_centralities': top3_cent,
                    'target_centrality': target_centrality,
                    'portfolio_corr_with_target': portfolio_corr,
                    'individual_profits': profs,
                    'portfolio_profit' : port_profit
                })

                # roll forward one month
                cur += pd.DateOffset(months=1)

            results[((cw,tw), strat)] = pd.DataFrame(rows)
    
    all_results = []
    for key, df in results.items():
        (cw, tw), strat = key
        df['param_corr_weight'] = cw
        df['param_cent_weight'] = tw
        df['strategy_type'] = strat
        all_results.append(df)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_excel("backtest_raw.xlsx", index=False)
        print("Raw backtest results saved to backtest_raw.xlsx")
    else:
        print("No results to save")
    
    return results


def create_comparison_report(results, param_combo, output_file='backtest_comparison.xlsx'):
    """
    Creates a comparison dataframe showing correlation and composite strategies side by side
    for a specific parameter combination.
    
    Args:
        results: Dictionary from rolling_backtest
        param_combo: Tuple like (0.7, 0.3) for (corr_weight, cent_weight)
        output_file: Excel file name to export
    
    Returns:
        comparison_df: DataFrame with monthly comparison
        summary_df: DataFrame with average statistics
    """
    cw, tw = param_combo
    
    # Get the two strategy dataframes
    corr_df = results[((cw, tw), 'corr')].copy()
    comp_df = results[((cw, tw), 'composite')].copy()
    
    # Merge on test dates
    comparison_rows = []
    
    for idx in range(len(corr_df)):
        corr_row = corr_df.iloc[idx]
        comp_row = comp_df.iloc[idx]
        
        # Format date range
        test_start = corr_row['test_start'].strftime('%Y-%m-%d')
        test_end = corr_row['test_end'].strftime('%Y-%m-%d')
        date_range = f"{test_start} to {test_end}"
        
        comparison_rows.append({
            'Test Period': date_range,
            'Correlation Strategy - Correlation': corr_row['portfolio_corr_with_target'],
            'Composite Strategy - Correlation': comp_row['portfolio_corr_with_target'],
            'Correlation Strategy - Profit (%)': corr_row['portfolio_profit'] * 100,
            'Composite Strategy - Profit (%)': comp_row['portfolio_profit'] * 100,
            'Correlation Strategy - Stocks': ', '.join(corr_row['selected_stocks']),
            'Composite Strategy - Stocks': ', '.join(comp_row['selected_stocks']),
            'Correlation Strategy - Top3 Corr': str(corr_row['top3_correlations']),
            'Composite Strategy - Top3 Corr': str(comp_row['top3_correlations']),
            'Composite Strategy - Top3 Cent': str(comp_row['top3_centralities']),
            'Target Centrality': corr_row['target_centrality']
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Create summary statistics
    summary_data = {
        'Metric': [
            'Average Correlation with Target',
            'Number of Months'
        ],
        'Correlation Strategy': [
            comparison_df['Correlation Strategy - Correlation'].mean(),
            len(comparison_df)
        ],
        'Composite Strategy': [
            comparison_df['Composite Strategy - Correlation'].mean(),
            len(comparison_df)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Monthly Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Format the sheets
        workbook = writer.book
        
        # Format Monthly Results sheet
        ws1 = writer.sheets['Monthly Results']
        for column in ws1.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws1.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # Format Summary Statistics sheet
        ws2 = writer.sheets['Summary Statistics']
        for column in ws2.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            ws2.column_dimensions[column[0].column_letter].width = adjusted_width
    
    print(f"\nComparison report saved to {output_file}")
    print("\n=== SUMMARY STATISTICS ===")
    print(summary_df.to_string(index=False))
    
    return comparison_df, summary_df


def calculate_stock_metrics(stock_list, target, close_pct_df, volume_df, corr_weights={'returns': 0.7, 'volume': 0.3}, lookback_years=3, centrality_filter: bool = False):
    """
    Calculate correlation and centrality metrics for all stocks relative to a target stock.
    Merges results back into stock_list and removes rows with NaN correlations.
    
    Args:
        stock_list: DataFrame with Symbol, Name, Sector, Industry columns
        target: Target ticker symbol
        close_pct_df: DataFrame of daily returns
        volume_df: DataFrame of volume scores
        corr_weights: Weights for composite correlation
        lookback_years: Number of years to look back for calculations
    
    Returns:
        DataFrame with original columns plus 'Correlation' and 'Centrality' columns
    """
    # Get last 3 years of data
    end_date = close_pct_df.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    returns_slice = close_pct_df.loc[start_date:end_date]
    volume_slice = volume_df.loc[start_date:end_date]
    
    # Calculate composite correlation
    C_composite = correlations_calc(returns_slice, volume_slice, corr_weights)
    
    
    if centrality_filter:
        cent = eigenvector_centrality(C_composite,
                                      threshold=0.0,
                                      use_pagerank=False)  # Use eigenvector instead
    else:
        cent = eigenvector_centrality(C_composite,
                                      threshold=0.0,
                                      use_pagerank=True,
                                      seed_ticker=target)
        
    # Extract correlation with target
    target_corr = C_composite[target]
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Symbol': target_corr.index,
        'Correlation': target_corr.values,
        'Centrality': cent.values
    })
    
    # Merge with stock_list
    result_df = stock_list.merge(metrics_df, on='Symbol', how='left')
    
    # Drop rows with NaN correlations
    result_df = result_df.dropna(subset=['Correlation'])
    
    # Save to Excel
    result_df.to_excel(f'stock_metrics_{target}.xlsx', index=False)
    print(f"Stock metrics saved to stock_metrics_{target}.xlsx")
    print(f"Total stocks with valid correlations: {len(result_df)}")
    
    return result_df



```

Feel free to refine:

Use an empirical DD→EDF lookup table (if you have one).
Estimate μ (asset drift) via CAPM or historical drift.
Tweak the “default point” formula to match your firm’s capital structure.
