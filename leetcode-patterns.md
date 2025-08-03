# LeetCode Patterns: Sorting, Templates & Snippets

Below you’ll find (1) **sorting templates**, (2) **core patterns** with example questions, ASCII “visualizations,” and “tweak points,” plus suggestions for **medium/hard** add-ons, and (3) **plug-and-play snippets** you can copy/paste.

---

## 2. Sorting Algorithms (Templates)

### 2.1 Merge Sort (O(n log n), stable)  
**Use when** you need a stable sort or external merge.  
**Example**: 148. Sort List

```python
def merge_sort(a):
    if len(a) <= 1:
        return a
    m = len(a)//2
    L, R = merge_sort(a[:m]), merge_sort(a[m:])
    res, i, j = [], 0, 0
    # merge L and R
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

**Tweak**: change `L[i]<R[j]` → `<=` if equal elements should stay in original order.

---

### 2.2 Quick Sort (avg O(n log n), worst O(n²))  
**Use when** in-place is critical.  
**Example**: 215. Kth Largest Element (via nth-element)

```python
import random

def quick_sort(a, l=0, r=None):
    if r is None: r = len(a)-1
    if l < r:
        # random pivot for average O(n log n)
        pi = random.randint(l, r)
        a[pi], a[r] = a[r], a[pi]
        p = partition(a, l, r)
        quick_sort(a, l, p-1)
        quick_sort(a, p+1, r)
    return a

def partition(a, l, r):
    pivot = a[r]
    i = l
    for j in range(l, r):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[r] = a[r], a[i]
    return i
```

Partition visualization:

```
[3,6,8,10,1,2,1], pivot=1
→ smaller: [ ], larger: [3,6,8...]
swap into place, pivot ends at idx i
```

**Tweak**: choose `pivot = median-of-three(a[l],a[m],a[r])` for fewer worst-case scenarios.

---

### 2.3 Heap Sort (O(n log n), not stable)  
**Use when** constant memory (in-place) is required (except recursion).  
**Example**: find k-th largest

```python
import heapq

def heap_sort(a):
    heapq.heapify(a)               # O(n)
    return [heapq.heappop(a)       # n pops, each O(log n)
            for _ in range(len(a))]
```

Binary‐heap array:

```
      a[0]
     /    \
   a[1]  a[2]
  /\      /\
 ...
```

**Tweak**: to get descending order, store `-x` or pop from max‐heap (invert sign).

---

## 3. Pattern Templates

Below each pattern includes: a **LeetCode example**, an **ASCII sketch** of how pointers/structures move, and **tweak points**.

### A. Sliding Window  
**When**: contiguous subarray problems (fixed or variable size).  
**Example**: 3. Longest Substring Without Repeating (var-size), 209. Minimum Size Subarray Sum

```python
def min_subarray_len(nums, s):
    left = 0
    curr = 0
    res = float('inf')
    for right, x in enumerate(nums):
        curr += x
        # shrink window while ≥ s
        while curr >= s:
            res = min(res, right - left + 1)
            curr -= nums[left]
            left += 1
    return res if res != float('inf') else 0
```

ASCII (fixed k):

```
[1,2,3,4,5], k=3
[1,2,3] sum=6 → slide right
  [2,3,4] sum=9 → slide
    ...
```

**Tweak**:  
- Fixed‐size: subtract `nums[i-k]` when `i ≥ k`.  
- Variable: use `while` to shrink, `if` (or no shrink) to expand.

---

### B. Two Pointers  
**When**: sorted arrays/strings, pair‐sum, container water.  
**Example**: 11. Container With Most Water, 3Sum (15)

```python
def container_max_area(height):
    l, r = 0, len(height)-1
    best = 0
    while l < r:
        best = max(best, min(height[l],height[r])*(r-l))
        # move the smaller end inward
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

**Tweak**: for “closest sum” change `if s<target: l+=1`.

---

### C. Fast & Slow Pointers  
**When**: linked‐list cycle, find middle, overwrite in-place.  
**Example**: 876. Middle of the Linked List, 142. Linked List Cycle II

```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

ASCII:

```
head→1→2→3→4→5→…  
slow:  →2→3→…  
fast:    →3→5→…
```

**Tweak**: to detect start of cycle, reset one pointer to head then advance both by one.

---

### D. Prefix Sum / Difference Array  
**When**: range‐sum queries, count subarrays with sum k.  
**Example**: 560. Subarray Sum Equals K, 303. Range Sum Query

```python
# build prefix
presum = [0]
for x in nums:
    presum.append(presum[-1] + x)
# sum of nums[i..j] = presum[j+1] - presum[i]
```

**Tweak**: use hash‐map of prefix sums to count `presum[j] - k` seen so far.

---

### E. Hash-Map (Dict)  
**When**: lookups, frequency counts, two-sum, grouping.  
**Example**: 1. Two Sum, 49. Group Anagrams, 128. Longest Consecutive Sequence

```python
def two_sum(nums, target):
    d = {}
    for i, x in enumerate(nums):
        if target-x in d:
            return [d[target-x], i]
        d[x] = i
```

**Tweak**: use `Counter` for freq, `defaultdict(list)` for grouping.

---

### F. Binary Search  
**When**: sorted arrays, search in monotonic function.  
**Example**: 33. Search in Rotated Sorted Array, 74. Search a 2D Matrix

```python
def bsearch(a, target):
    l, r = 0, len(a)-1
    while l <= r:
        m = (l+r)//2
        if a[m] == target:
            return m
        if a[m] < target:
            l = m+1
        else:
            r = m-1
    return -1
```

**Tweak**:  
- lower_bound: `while l<r: ... r=m`  
- upper_bound: `while l<r: ... l=m+1`

---

### G. Heap (Priority Queue)  
**When**: top-k, merging, median-finder.  
**Example**: 23. Merge k Sorted Lists, 347. Top K Frequent Elements

```python
import heapq

def kth_largest(nums, k):
    h = nums[:k]
    heapq.heapify(h)        # min‐heap of size k
    for x in nums[k:]:
        if x > h[0]:
            heapq.heapreplace(h, x)
    return h[0]
```

**Tweak**: invert sign for max‐heap, store tuples `(priority, item)`.

---

### H. BFS / DFS (Graph & Tree)  
**When**: shortest paths, connectivity, tree traversals.  
**Example**: 200. Number of Islands (BFS), 94. Binary Tree Inorder Traversal (DFS)

```python
# BFS on grid
from collections import deque
def num_islands(grid):
    n, m = len(grid), len(grid[0])
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
    cnt = 0
    for i in range(n):
      for j in range(m):
        if grid[i][j]=='1':
          cnt += 1
          grid[i][j]='0'
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

**Tweak**: use recursion for DFS, add `seen` set for graphs.

---

### I. Dynamic Programming  
**1D DP** (e.g. max profit, Fibonacci)  
- **Example**: 121. Best Time to Buy and Sell Stock, 70. Climbing Stairs

```python
def max_profit(prices):
    minp, ans = float('inf'), 0
    for p in prices:
        minp = min(minp, p)
        ans = max(ans, p-minp)
    return ans
```

**2D DP** (Matrix paths, edit distance)  
- **Example**: 62. Unique Paths, 72. Edit Distance  

```python
def unique_paths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1,m):
      for j in range(1,n):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

**Tweak**:  
- “State” = index or two indices.  
- Transition: iterate choices, take min/max.

---

### J. Backtracking (Permutations/Subsets/Combinations)  
**When**: generate all, n-queens, sudoku.  
**Example**: 46. Permutations, 17. Letter Combinations of a Phone Number  

```python
def permute(nums):
    res = []
    def dfs(path, used):
        if len(path)==len(nums):
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

**Tweak**: for combinations (i from start→n), for subsets (include/exclude at each index).

---

### K. Greedy  
**When**: local choice leads to global.  
**Example**: 435. Non-overlapping Intervals, 55. Jump Game  

```python
def can_jump(nums):
    reach = 0
    for i, x in enumerate(nums):
        if i > reach: return False
        reach = max(reach, i + x)
    return True
```

**Tweak**: prove by exchange, always pick interval with earliest end.

---

### L. Union-Find (Disjoint Set)  
**When**: connectivity, Kruskal’s MST, accounts merge.  
**Example**: 547. Friend Circles, 721. Accounts Merge  

```python
class UF:
    def __init__(self,n):
        self.p = list(range(n))
    def find(self,x):
        if self.p[x]!=x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        ra, rb = self.find(a), self.find(b)
        if ra!=rb:
            self.p[rb] = ra
```

**Tweak**: add union by rank/size, path-compression.

---

### M. Bit Manipulation  
**When**: subsets, parity, XOR tricks.  
**Example**: 136. Single Number, 190. Reverse Bits  

```python
def single_number(nums):
    res = 0
    for x in nums:
        res ^= x
    return res
```

**Tweak**: for “count bits,” `while x: x &= x-1; cnt+=1`.

---

### **Additional Patterns for Medium/Hard**  
- N. **Monotonic Queue** (Sliding‐window max, 239)  
- O. **Segment Tree / Fenwick Tree** (range updates/queries)  
- P. **Topological Sort** (courses, jobs scheduling)  
- Q. **Trie** (prefix search, word squares)  
- R. **Meet-in-the-Middle** (subset sum)  
- S. **Bitmask DP** (TSP, subset covering)  
- T. **Geometry / Computational Geometry**  

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
    while x>0:
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

> **Tip**: Keep one “cookbook” file with all patterns, copy the template, then tweak only the marked parts (`pivot`, `window condition`, `dp state`, **etc.**). Good luck!